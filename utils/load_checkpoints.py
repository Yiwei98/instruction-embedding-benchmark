import os
import torch

import accelerate
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaModel, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft import PeftModel
import peft


def load_tokenizer(tokenzier_path, max_length=256, bert=False):
    if bert:
        tokenizer = AutoTokenizer.from_pretrained(tokenzier_path, max_length=max_length, local_files_only=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(tokenzier_path, max_length=max_length, local_files_only=True)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference

    return tokenizer


def load_model(model_path, lora_weight=None, lora=False, load_in_8bit=False, device_map="auto", train=False, bert=False):
    if bert:
        model = AutoModel.from_pretrained(model_path, local_files_only=True).cuda()
    else:
        model = LlamaModel.from_pretrained(
                model_path,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
                local_files_only=True
            )
    
    if load_in_8bit and train:
        model = prepare_model_for_kbit_training(model)

    if lora:
        if lora_weight is None:
            config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=['q_proj','v_proj'],
                lora_dropout=0.05,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

            world_size = int(os.environ.get("WORLD_SIZE", 1))
            ddp = world_size != 1
            if not ddp and torch.cuda.device_count() > 1:
                model.is_parallelizable = True
                model.model_parallel = True

            model.config.use_cache = False
        else:
            model = PeftModel.from_pretrained(
                model,
                lora_weight,
                torch_dtype=torch.float16,
            )

        
    model.config.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    return model


def load_causallm_model(model_path, lora_weight=None, lora=False, load_in_8bit=False, device_map="auto", train=False):
    try:
        model = LlamaForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
            )
    except:
        model = AutoModelForCausalLM.from_pretrained(
                model_path
            ).cuda()

    model.config.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if load_in_8bit and train:
        model = prepare_model_for_kbit_training(model)

    if lora:
        if lora_weight is None:
            config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=['q_proj','v_proj'],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

            world_size = int(os.environ.get("WORLD_SIZE", 1))
            ddp = world_size != 1
            if not ddp and torch.cuda.device_count() > 1:
                model.is_parallelizable = True
                model.model_parallel = True

            model.config.use_cache = False
        else:
            model = PeftModel.from_pretrained(
                model,
                lora_weight,
                torch_dtype=torch.float16,
            )

    return model
