import os
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler

import fire
import accelerate

from utils.ieb_parser import *
from utils.load_checkpoints import *
from train import *

def main(
    model_path: str,
    eval_data_path,
    save_path: str,  
    pooling: str,
    bert: bool=False,
    tokenizer_path: str=None,
    batch_size: int=8,
    lora_weight: str=None,
    lora: bool=False,
    load_in_8bit: bool=False,
    use_prompt: bool=False,
    list_data: bool=False,
    short_prompt: bool=False,
):
    print(
            f"Instruction embedding infer with params:\n"
            f"base_model: {model_path}\n"
            f"tokenizer_path: {tokenizer_path}\n"
            f"eval_data_path: {eval_data_path}\n"
            f"output_dir: {save_path}\n"
            f"pooling: {pooling}\n"
            f"bert: {bert}\n"
            f"batch_size: {batch_size}\n"
            f"lora_weight: {lora_weight}\n"
            f"lora: {lora}\n"
            f"load_in_8bit: {load_in_8bit}\n"
            f"use_prompt: {use_prompt}\n"
            f"list_data: {list_data}\n"
            f"short_prompt: {short_prompt}\n"
        )
    device = "cuda"

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
    model.eval()

    if use_prompt:
        # template = "Below is an instruction that describes a task\n\n" \
        #         "*sent*\n\n" \
        #         "The task of the given instruction is:"
        
        template =  "The essence of an instruction is its task intention. With this in mind, " \
                    "given the instruction below:\n\n" \
                    "*sent*\n\n" \
                    "after thinking step by step, the task of the given instruction is:"
        
        # template = "Below is an instruction that describes a task\n" \
        #             "*sent*\n" \
        #             "The task of the given instruction is:"

        # template = "The following instruction\n" \
        #             "*sent*\n" \
        #             "wants you to:"
        
        # template = "Given the following instruction\n" \
        #             "*sent*\n" \
        #             "  please identify its task type:"

        # template = "What type of task does the following instruction represent?\n" \
        #             "*sent*\n" 

        # template = "Indentify the task category associated with the following instruction:\n" \
        #             "*sent*\n" \

        if short_prompt:
            template = "This sentence of \"*sent*\" means:"
    else:
        template = None
    
    if not list_data:
        test_dataset = IEBTestDataset(eval_data_path)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                    collate_fn=IEBTestDataCollator(tokenizer, template=template))
    else:
        test_dataset = ListStyleDataset(eval_data_path)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                    collate_fn=ListStyleDataCollator(tokenizer, template=template))
    
    eval_loader = tqdm(test_data_loader, desc="Iteration")
    embedding_dict = {}
    for _, batch in enumerate(eval_loader):
        instructions = batch['instructions'].copy()
        index = batch['index'].copy()

        del batch['instructions']
        del batch['index']

        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device):
            # Get the embeddings
            with torch.no_grad():
                hidden_states = model.forward(**inputs, output_hidden_states=True, return_dict=True)['hidden_states']
                embeddings_last = hidden_states[-1]
                # last 2 layers
                if pooling == 'last-2':
                    embeddings_second_last = hidden_states[-2]
                    embeddings = (embeddings_last[:, -1, :] + embeddings_second_last[:, -1, :]) / 2.
                # last 1 layer
                elif pooling == 'last-1':
                    embeddings = embeddings_last[:, -1, :]
                # first and last layer
                elif pooling == 'last-and-first':
                    embedings_first = hidden_states[0]
                    embeddings = (embeddings_last[:, -1, :] + embedings_first[:, -1, :]) / 2.
                # mid layer
                elif pooling == 'mid':
                    embeddings_mid = hidden_states[len(hidden_states) // 2]
                    embeddings = embeddings_mid[:, -1, :]
                elif pooling == 'cls':
                    embeddings = embeddings_last[:, 0, :]
                else:
                    raise NotImplementedError

                for i in range(len(instructions)):
                    embedding_dict[embeddings[i].cpu()] = index[i]

    os.makedirs("../saved_instruction_embeddings", exist_ok=True)
    torch.save(embedding_dict, f"{save_path}")

if __name__ == "__main__":
    fire.Fire(main)