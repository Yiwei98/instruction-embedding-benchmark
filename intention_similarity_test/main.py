import os
import sys

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


def eval_on_iis(data_loader: DataLoader, model, pooling='last-2'):
    labels = []
    cos_similarities = []

    cos = nn.CosineSimilarity(dim=-1)

    for _, data in enumerate(tqdm(data_loader, desc="Iteration")):
        labels += data['labels']
        del data['labels']

        inputs = {key: value.to("cuda") for key, value in data.items()}

        with torch.autocast("cuda"):
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

                n_dim = embeddings.shape[-1]
                embeddings = embeddings.reshape(-1, 2, n_dim)
                similarity = cos(embeddings[:, 0, :], embeddings[:, 1, :]).detach().cpu().numpy().tolist()
                cos_similarities += similarity

    return scipy.stats.spearmanr(cos_similarities, labels)[0]


def main(
    model_path: str,
    eval_data_path,
    pooling: str,
    bert: bool=False,
    tokenizer_path: str=None,
    batch_size: int=4,
    lora_weight: str=None,
    lora: bool=False,
    load_in_8bit: bool=False,
    use_prompt: bool=False,
    short_prompt: bool=False
):
    print(
            f"Evaluate instruction embedding model with params:\n"
            f"base_model: {model_path}\n"
            f"tokenizer_path: {tokenizer_path}\n"
            f"eval_data_path: {eval_data_path}\n"
            f"pooling: {pooling}\n"
            f"bert: {bert}\n"
            f"batch_size: {batch_size}\n"
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

    dataset = ITSDataset(eval_data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler(dataset),
                             collate_fn=ITSDataCollator(tokenizer, template=template))
    result = eval_on_iis(data_loader, model, pooling)
    with open('iis-test-result.txt', 'a') as f:
        f.write(f"{model_path}:\t{result}\n")
    print(result)


if __name__ == "__main__":
    fire.Fire(main)
                


        

