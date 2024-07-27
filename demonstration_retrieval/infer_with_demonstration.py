import os
import sys

import time
import json

import fire
from tqdm import tqdm
import torch
from transformers import GenerationConfig

from utils.load_checkpoints import *

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ),
}


def jload(path, alpaca_eval=False):
    with open(path, 'r') as f:
        data = json.load(f)
        data_pool = []
        if not alpaca_eval:
            for key in data.keys():
                for sample in data[key]['samples']:
                    data_pool.append(sample)
        else:
            return data

    return data_pool


def formalize_input(input, indices, data_pool, template, alpaca_eval=False):
    demonstrations = ''
    if indices is not None:
        demonstrations = [data_pool[indice] for indice in indices]
        demonstrations = '\n\n'.join([template.format(instruction=data['input'].strip(), response=data['output'].strip()) for data in demonstrations])

    instruction = template.format(instruction=input['input''input' if not alpaca_eval else 'instruction'].strip(), response='')
    instruction = demonstrations + '\n\n' + instruction

    return instruction


def main(
    model_path: str,
    eval_data_path,
    save_path: str,  
    bert: bool=False,
    tokenizer_path: str=None,
    lora_weight: str=None,
    lora: bool=False,
    load_in_8bit: bool=False,
    data_pool_path: str = '',
    demonstration_map_path: str = None,
    # inference parameters  
    temperature: float=0.1,
    top_p: float=0.75,
    top_k: int=4,
    num_beams: int=1,
    max_new_tokens: int=512,
    alpaca_eval: bool=False,
):
    print(
            f"ICL inference with params:\n"
            f"base_model: {model_path}\n"
            f"tokenizer_path: {tokenizer_path}\n"
            f"data_pool_path: {data_pool_path}\n"
            f"eval_data_path: {eval_data_path}\n"
            f"output_dir: {save_path}\n"
            f"bert: {bert}\n"
            f"lora_weight: {lora_weight}\n"
            f"lora: {lora}\n"
            f"load_in_8bit: {load_in_8bit}\n"
            f"demonstration_map_path: {demonstration_map_path}\n"
            f"temperature: {temperature}\n"
            f"top_p: {top_p}\n"
            f"top_k: {top_k}\n"
            f"num_beams: {num_beams}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"alpaca_eval: {alpaca_eval}\n"
        )

    generator = model_path.split('/')[-1].strip()
    tokenizer = load_tokenizer(tokenizer_path if tokenizer_path else model_path, bert=bert)
    model = load_causallm_model(
        model_path, 
        lora_weight=lora_weight,
        lora=lora,
        load_in_8bit=load_in_8bit,
        device_map='auto',
        train=True,
        # bert=bert
        )
    model.eval()


    def evaluate(
        input,
        temperature=0.1,
        top_p=0.75,
        top_k=4,
        num_beams=4,
        max_new_tokens=512,
    ):
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)

        return output.split("### Response:")[-1].strip()


    template = PROMPT_DICT['prompt_no_input']

    with open(demonstration_map_path, 'r') as f:
        demonstration_map = json.load(f)
    data_pool = jload(data_pool_path)
    eval_data = jload(eval_data_path, alpaca_eval)

    result = []
    for index in tqdm(range(len(eval_data))):
        sample = eval_data[index]
        indices = demonstration_map[str(index)] if demonstration_map_path is not None else None

        input = formalize_input(sample, indices, data_pool, template, alpaca_eval)
        output = evaluate(
            input=input,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens
        )
        record = {
            'instruction': sample['input' if not alpaca_eval else 'instruction'],
            'formalized_input': input,
            'output': output,
            'generator': generator
        }
        result.append(record)
        if 'alpaca_eval' in eval_data_path:
            record['dataset'] = sample['dataset']
            record['datasplit'] = 'eval'

        print(f"[Input]\n{input}\n\n[Output]\n{output}\n")

    os.makedirs('../demonstration_infer_output', exist_ok=True)
    with open(f'{save_path}', 'w') as f:
        json.dump(result, f, indent=4)
        

if __name__ == "__main__":
    fire.Fire(main)
