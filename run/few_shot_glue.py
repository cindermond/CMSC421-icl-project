from batcon.config import DatasetConfig, NetConfig
from batcon.net import Net
from batcon.consts import *
from batcon.dataset import RandomDataset, Dataset
from batcon.pipeline import Pipeline

import argparse
from datasets import load_dataset
from evaluate import load
import random
import torch
import time

def main(dataset_config_path, net_config_path, dataset_name, verbose, seed, single_step, valid_limit):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start_time = time.time()

    train_source = list(load_dataset("glue", dataset_name, split="train", cache_dir="cache"))
    for entry in train_source:
        entry["label_text"] = prompt_dict[dataset_name]["label_map_reverse"][entry["label"]]
    if valid_limit > 0:
        dev_source = load_dataset("glue", dataset_name, split=f"validation[:{valid_limit}]", cache_dir="cache")
    else:
        dev_source = load_dataset("glue", dataset_name, split="validation", cache_dir="cache")
    labels = dev_source["label"]
    dataset_config_train = DatasetConfig(dataset_config_path)
    dataset_config_dev = DatasetConfig()
    train_dataset = RandomDataset(train_source, prompt_dict[dataset_name]["single_example_prompt"], dataset_config_train)
    dev_dataset = Dataset(dev_source, prompt_dict[dataset_name]["question_prompt"], dataset_config_dev)

    net_config = NetConfig(net_config_path)
    net = Net(net_config)
    if single_step:
        first_prompt = prompt_dict[dataset_name]["multi_example_prompt_no_reasoning"]
        second_prompt = None
    else:
        first_prompt = prompt_dict[dataset_name]["multi_example_prompt"]
        second_prompt = prompt_dict[dataset_name]["multi_get_answer_prompt"]
    pipeline = Pipeline(net, train_dataset, dev_dataset, single_step=single_step, first_prompt=first_prompt, second_prompt=second_prompt)
    results = pipeline.evaluate(verbose=verbose, label_map=prompt_dict[dataset_name]["label_map"])
    results = [prompt_dict[dataset_name]["label_map"][r] for r in results]
    metric = load('glue', dataset_name)
    evaluation_results = metric.compute(predictions=results, references=labels)
    print(f'Evaluation results on {dataset_name}: {evaluation_results}')
    end_time = time.time()
    print(f'Used time: {end_time-start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', default="configs/dataset_config_few_shot.json", type=str)
    parser.add_argument('--net_config_path', default="configs/net_config_basic.json", type=str)
    parser.add_argument('--dataset_name', default="sst2", type=str)
    parser.add_argument('--single_step', action=argparse.BooleanOptionalAction)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--valid_limit', default=100, type=int)
    args = parser.parse_args()
    main(**vars(args))