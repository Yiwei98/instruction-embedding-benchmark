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


# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

def jload(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def formalize_input(input, template):
    instruction = template.format(instruction=input.strip())
    return instruction


def main(
    model_path: str,
    eval_data_path: str,
    save_path: str, 
    bert: bool=False, 
    tokenizer_path: str=None,
    lora_weight: str=None,
    lora: bool=False,
    load_in_8bit: bool=False,  
    # inference parameters  
    temperature: float=0.1,
    top_p: float=0.75,
    top_k: int=4,
    num_beams: int=1,
    max_new_tokens: int=512,
    # flag
    alpaca_eval: bool=False,
    use_prompt: bool=False,
    short_prompt: bool=False,
):
    print(
            f"LLM inference with params:\n"
            f"base_model: {model_path}\n"
            f"eval_data_path: {eval_data_path}\n"
            f"output_dir: {save_path}\n"
            f"bert: {bert}\n"
            f"tokenizer_path: {tokenizer_path}\n"
            f"lora_weight: {lora_weight}\n"
            f"lora: {lora}\n"
            f"load_in_8bit: {load_in_8bit}\n"
            f"temperature: {temperature}\n"
            f"top_p: {top_p}\n"
            f"top_k: {top_k}\n"
            f"num_beams: {num_beams}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"alpaca_eval: {alpaca_eval}\n"
        )
    
    template = "{instruction}"
    if use_prompt:
        # template = "Below is an instruction that describes a task\n\n" \
        #         "*sent*\n\n" \
        #         "The task of the given instruction is:"
        
        template =  "The essence of an instruction is its task intention. With this in mind, " \
                    "given the instruction below:\n\n" \
                    "{instruction}\n\n" \
                    "after thinking step by step, the task of the given instruction is:"

        if short_prompt:
            template = "This sentence of \"{instruction}\" means:"
    
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

    def generate(
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
    
    eval_data = jload(eval_data_path)
    result = []
    for sample in tqdm(eval_data):
        instruction = sample['instruction'].strip()
        input = formalize_input(instruction, template)
        output = generate(
            input=input,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens
        )
        sample['model_output'] = output
        result.append(sample)
        print(f"[Input]\n{input}\n\n[Output]\n{output}\n")

    os.makedirs('./inference_output', exist_ok=True)
    with open(f"./inference_output/{save_path}.json", 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)