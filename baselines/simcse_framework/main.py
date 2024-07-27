import os
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from utils.simcse_parser import *
from utils.load_checkpoints import *
from train import *
from models import *

def main(
    model_path: str,
    data_path: str,
    save_path: str,
    bert: bool=False,
    tokenizer_path: str=None,
    attention_dropout: float=0.3,
    hidden_state_dropout: float=0.3,
    batch_size=8,
    epochs: int=3,
    lr: float=1e-5,
    lora: bool=False, 
    load_in_8bit: bool=False,
    use_prompt: bool=False,
    short_prompt: bool=False,
):
    print(
            f"Training baseline model with params:\n"
            f"base_model: {model_path}\n"
            f"tokenizer_path: {tokenizer_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {save_path}\n"
            f"bert: {bert}\n"
            f"attention_dropout: {attention_dropout}\n"
            f"hidden_state_dropout: {hidden_state_dropout}\n"
            f"batch_size: {batch_size}\n"
            f"epochs: {epochs}\n"
            f"lr: {lr}\n"
            f"lora: {lora}\n"
            f"load_in_8bit: {load_in_8bit}\n"
            f"use_prompt: {use_prompt}\n"
            f"short_prompt: {short_prompt}\n"
        )
    os.makedirs("../../checkpoints", exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_path if tokenizer_path else model_path, bert=bert)

    if use_prompt:
        template =  "The essence of an instruction is its task intention. With this in mind, " \
                    "given the instruction below:\n\n" \
                    "{}\n\n" \
                    "after thinking step by step, the task of the given instruction is:"

        if short_prompt:
            template = "This sentence of \"{}\" means:"
    else:
        template = None

    dataset = DatasetForCL(data_path, template, use_prompt)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=RandomSampler(dataset),
                                collate_fn=DataCollatorForCL(tokenizer))
    
    config = AutoConfig.from_pretrained(model_path)
    config.attention_probs_dropout_prob = attention_dropout  
    config.hidden_dropout_prob = hidden_state_dropout
    
    if bert:
        model = BertForCL(config=config, model_path=model_path)
    else:
        model = LlamaForCL(config=config, model_path=model_path, lora=lora, load_in_8bit=load_in_8bit)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EmbeddingTrainer(data_loader, model, optimizer, save_path, epochs)
    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)