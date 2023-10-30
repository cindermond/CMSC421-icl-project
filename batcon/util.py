class DataProcessor:
    def __init__(self, name):
        self.name = name

    def process(self, dataset):
        match self.name:
            case "gsm8k":
                full_answer = [a.split("#### ") for a in dataset["answer"]]
                reasoning = [a[0] for a in full_answer]
                correct = [a[1] for a in full_answer]
                dataset = dataset.add_column(name="reasoning", column=reasoning)
                dataset = dataset.add_column(name="correct", column=correct)
                dataset = dataset.remove_columns('answer')
            case _:
                raise NotImplementedError
        return dataset