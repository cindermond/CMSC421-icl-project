from batcon.config import DatasetConfig, NetConfig
from batcon.net import Net
from batcon.consts import *
from batcon.dataset import MiniBatchEntangledDataset, Dataset
from batcon.pipeline import EntangledPipeline
from batcon.util import DataProcessor

import argparse
from datasets import load_dataset
from evaluate import load
import random
import torch
import time
import re

def main(dataset_config_path, net_config_path, dataset_name, verbose, seed, label_key):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start_time = time.time()

    data_processor = DataProcessor("gsm8k")
    train_source = list(data_processor.process(load_dataset("gsm8k", "main", split="train", cache_dir="cache")))
    dev_source = data_processor.process(load_dataset("gsm8k", "main", split="test[:100]", cache_dir="cache"))
    
    labels = dev_source[label_key]
    dataset_config_train = DatasetConfig(dataset_config_path)
    dataset_config_dev = DatasetConfig()
    train_dataset = MiniBatchEntangledDataset(train_source, [prompt_dict[dataset_name]["single_example_prompt"], prompt_dict[dataset_name]["single_get_answer_prompt"]], dataset_config_train)
    dev_dataset = Dataset(dev_source, prompt_dict[dataset_name]["question_prompt"], dataset_config_dev)
    net_config = NetConfig(net_config_path)
    net = Net(net_config)
    first_prompt = prompt_dict[dataset_name]["multi_example_prompt"]
    second_prompt = prompt_dict[dataset_name]["multi_get_answer_prompt"]
    pipeline = EntangledPipeline(net, train_dataset, dev_dataset, single_step=False, first_prompt=first_prompt, second_prompt=second_prompt)
    results = pipeline.evaluate(verbose=verbose)
    evaluation_results = sum([label == get_number(result) for result, label in zip(results, labels)])/len(labels)
    print(f'Evaluation results on {dataset_name}: {evaluation_results}')
    end_time = time.time()
    print(f'Used time: {end_time-start_time}')

def get_number(result):
    try:
        return re.search(r'\d+', result).group()
    except AttributeError: 
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', default="configs/dataset_config_few_shot.json", type=str)
    parser.add_argument('--net_config_path', default="configs/net_config_basic.json", type=str)
    parser.add_argument('--dataset_name', default="gsm8k", type=str)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--label_key', default="correct", type=str)
    args = parser.parse_args()
    main(**vars(args))