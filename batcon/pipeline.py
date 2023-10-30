from collections import Counter

class Pipeline:
    def __init__(self, net, train_dataset, test_dataset, single_step=True, first_prompt=None, second_prompt=None):
        #first prompt should have placeholders named "examples" and "question"
        #second prompt should have placeholders named "question" and "reasoning" if single_step=False
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.single_step = single_step
        self.first_prompt = first_prompt
        self.second_prompt = second_prompt

    def _evaluation_step(self, examples, question, verbose=False, word_list=None):
        if isinstance(examples, list):
            examples = "".join(examples)
        prompt = self.first_prompt.format(examples=examples, question=question)
        if verbose:
            print(f"First prompt: {prompt}")
        reasoning = self.net.inference(prompt)
        reasoning = reasoning.split("##")[0]
        if self.single_step:
            return reasoning
        else:
            prompt = self.second_prompt.format(question=question, reasoning=reasoning, examples=examples)
            if verbose:
                print(f"Second prompt: {prompt}")
            result = self.net.inference(prompt, word_list)
            return result

    def evaluate(self, verbose=False, max_voters=0, label_map=None):
        all_results = []
        if label_map is None:
            word_list = None
        else:
            word_list = list(label_map.keys())
        for question in self.test_dataset:
            results = []
            for examples in self.train_dataset:
                result = self._evaluation_step(examples, question, verbose, word_list)
                #result = result.split("##")[0]
                if verbose:
                    print(f"Result: {result}\n")
                results.append(result)
            if max_voters > 0:
                results = results[-max_voters:]
            counter = Counter(results)
            all_results.append(counter.most_common(1)[0][0])
        return all_results

class EntangledPipeline(Pipeline):
    def _evaluation_step(self, examples, question, verbose=False, word_list=None):
        first_examples = [example[0] for example in examples]
        second_examples = [example[1] for example in examples]
        if isinstance(first_examples, list):
            first_examples = "".join(first_examples)
        prompt = self.first_prompt.format(examples=first_examples, question=question)
        if verbose:
            print(f"First prompt: {prompt}")
        reasoning = self.net.inference(prompt)
        #reasoning = reasoning.split("##")[0]
        #reasoning = reasoning.split("\n")[0]
        if isinstance(second_examples, list):
            second_examples = "".join(second_examples)        
        prompt = self.second_prompt.format(examples=second_examples, question=question, reasoning=reasoning)
        if verbose:
            print(f"Second prompt: {prompt}")
        result = self.net.inference(prompt, word_list)
        return result
                