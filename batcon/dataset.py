import random
import warnings
from batcon.config import DatasetConfig
from typing import List

class Dataset:
    def __init__(self, source_dataset: List[dict], prompt_template: str, config: DatasetConfig):
        self.source_dataset = source_dataset
        self.prompt_template = prompt_template
        self.config = config
        self._preprocess()

    def _preprocess(self):
        pass

    def _format_template(self, entry):
        return self.prompt_template.format(**entry)
    
    def __iter__(self):
        self.place = -1
        return self
    
    def __next__(self):
        self.place += 1
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
        else:
            max_steps = len(self.source_dataset)
        if self.place < max_steps:
            return self._format_template(self.source_dataset[self.place])
        else:
            raise StopIteration

class EntangledDataset(Dataset):
    def __init__(self, source_dataset: List[dict], prompt_template_list: List[str], config: DatasetConfig):
        self.source_dataset = source_dataset
        self.prompt_template_list = prompt_template_list
        self.config = config
        self._preprocess()

    def _format_template(self, entry):
        return [prompt_template.format(**entry) for prompt_template in self.prompt_template_list]


class MiniBatchDataset(Dataset):
    def _preprocess(self):
        processed_dataset = [self._format_template(entry) for entry in self.source_dataset]
        repeated_dataset = processed_dataset * self.config.repeat_times
        if self.config.shuffle_item:
            random.shuffle(repeated_dataset)
        self.batches = [repeated_dataset[i:i+self.config.example_size] for i in range(0, len(repeated_dataset), self.config.example_size)]
        if self.config.shuffle_batch:
            random.shuffle(self.batches)

    def __next__(self):
        self.place += 1
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
        else:
            max_steps = len(self.batches)
        if self.place < max_steps:
            return self.batches[self.place]
        else:
            raise StopIteration


class MiniBatchEntangledDataset(EntangledDataset, MiniBatchDataset):
    pass



class RandomDataset(Dataset):
    def __next__(self):
        if self.config.max_steps > 0 or not self.config.override_limits:
            self.place += 1
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
        else:
            max_steps = 1000000
        if self.place < max_steps:
            selected = random.sample(self.source_dataset, self.config.example_size)
            return [self._format_template(entry) for entry in selected]
        else:
            if self.config.max_steps == 0 and self.config.override_limits:
                warnings.warn("RandomDataset called over 1000000 times, there's probably an infinite loop. If you are certain there's no error, set override_limits to True to ignore this constraint.")
            raise StopIteration    

class RandomEntangledDataset(EntangledDataset, RandomDataset):
    pass