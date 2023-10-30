from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import LongTensor
import re

class Net:
    def __init__(self, config):
        self.config = config
        with open(config.token_file, 'r') as file:
            self.token = file.read().rstrip()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir, token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir=config.cache_dir, token=self.token, device_map='auto')

    def inference(self, prompt, word_list=None):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=not self.config.chat)
        generate_ids = self.model.generate(inputs.input_ids.cuda(), **self.config.generation_args)[:, inputs.input_ids.shape[1]:]
        generated_sentence = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if word_list is None:
            return generated_sentence
        
        word_set = set(word_list)
        sentence_words = re.findall(r'\b\w+\b', generated_sentence)
        for word in sentence_words:
            if word in word_set:
                return word
        return word_list[0]