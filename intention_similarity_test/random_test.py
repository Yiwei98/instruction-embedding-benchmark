import os
import sys
import random

from torch.utils.data import DataLoader, SequentialSampler
import numpy
import fire
from tqdm import tqdm
import torch
import torch.nn as nn
import scipy.stats

import fire
import accelerate

from utils.ist_parser import *
from utils.load_checkpoints import *

def generate_random_list(n):
    return [random.choice([0, 1]) for _ in range(n)]

def eval_on_iis(data_loader: DataLoader):
    labels = []
    spearman_list = []

    for _, data in enumerate(tqdm(data_loader, desc="Iteration")):
        labels += data['labels']

    for _ in range(10):
        random_list = generate_random_list(len(labels))
        spearman_list.append(scipy.stats.spearmanr(random_list, labels)[0])
    print(spearman_list)
    print(sum(spearman_list) / 10)

def main(
    model_path: str='/backup/sjy/models/bert-base-uncased',
    eval_data_path: str='../data/intention_similarity_test.json',
    bert: bool=True,
):
    
    tokenizer = load_tokenizer(model_path, bert=bert)


    dataset = ITSDataset(eval_data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=8, sampler=SequentialSampler(dataset),
                             collate_fn=ITSDataCollator(tokenizer, template=None))
    result = eval_on_iis(data_loader)
    # with open('iis-test-result.txt', 'a') as f:
    #     f.write(f"{model_path}:\t{result}\n")
    print(result)


if __name__ == "__main__":
    fire.Fire(main)
                


        

