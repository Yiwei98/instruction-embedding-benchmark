import os
import sys

import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import fire
import accelerate

from utils.ieb_parser import *
from utils.load_checkpoints import *
from train import *

def main(
    model_path: str,
    train_data_path: str,
    eval_data_path,
    save_path: str,  
    pooling: str,                   # specify the pooling strategy, ['last-2', 'last-1', 'last-and-first', 'mid', 'cls]
    hard_negative_weight: float=0.5,
    hard_neg: bool=False,
    bert: bool=False,
    tokenizer_path: str=None,
    # train_num: int=5000,
    batch_size: int=8,
    epochs: int=3,
    lr: float=1e-5,
    lora_weight: str=None,
    lora: bool=False,               # whether to train with lora
    load_in_8bit: bool=False,
    use_prompt: bool=False,
    short_prompt: bool=False
):
    print(
            f"Training Embedding model with params:\n"
            f"base_model: {model_path}\n"
            f"tokenizer_path: {tokenizer_path}\n"
            f"train_data_path: {train_data_path}\n"
            f"eval_data_path: {eval_data_path}\n"
            f"output_dir: {save_path}\n"
            f"pooling: {pooling}\n"
            f"hard_negative_weight: {hard_negative_weight}\n"
            f"hard_neg: {hard_neg}\n"
            f"bert: {bert}\n"
            # f"train_num: {train_num}\n"
            f"batch_size: {batch_size}\n"
            f"epochs: {epochs}\n"
            f"lr: {lr}\n"
            f"lora_weight: {lora_weight}\n"
            f"lora: {lora}\n"
            f"load_in_8bit: {load_in_8bit}\n"
            f"use_prompt: {use_prompt}\n"
            f"short_prompt: {short_prompt}\n"
        )
    
    tokenizer = load_tokenizer(tokenizer_path if tokenizer_path else model_path, bert=bert)
    model = load_model(
        model_path, 
        lora_weight=lora_weight,
        lora=lora,
        load_in_8bit=load_in_8bit,
        device_map='auto',
        train=True,
        bert=bert
        )
    model.train()

    if use_prompt:
        # template = "Below is an instruction that describes a task\n\n" \
        #         "*sent*\n\n" \
        #         "The task of the given instruction is:"
        
        template =  "The essence of an instruction is its task intention. With this in mind, " \
                    "given the instruction below:\n\n" \
                    "*sent*\n\n" \
                    "after thinking step by step, the task of the given instruction is:"

        if short_prompt:
            template = "This sentence of \"*sent*\" means:"
    else:
        template = None

    train_dataset = IEBTrainDataset(train_data_path)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset),
                                collate_fn=IEBTrainDataCollator(tokenizer, template=template, path=train_data_path, hard_neg=hard_neg))
    
    test_dataset = IEBTestDataset(eval_data_path)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                collate_fn=IEBTestDataCollator(tokenizer, template=template))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EmbeddingTrainer(
        train_loader=train_data_loader, 
        eval_loader=test_data_loader, 
        model=model, 
        optimizer=optimizer, 
        save_path=save_path, 
        epochs=epochs,
        pooling=pooling,
        template=template,
        hard_negative_weight=hard_negative_weight,
    )
    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)