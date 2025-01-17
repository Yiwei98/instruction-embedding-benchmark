#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH=".."
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH='model_path'
EVAL_DATA_PATH='../data/eft_test.json'
MODEL_SAVE_PATH='model_save_path'
EMBEDDING_SAVE_PATH='embedding_save_path'
POOLING='last-2'

python encode.py \
       --model_path "$MODEL_PATH" \
       --eval_data_path "$EVAL_DATA_PATH" \
       --save_path "$EMBEDDING_SAVE_PATH" \
       --pooling "$POOLING" \
       --lora_weight "$MODEL_SAVE_PATH" \
       --lora \
       --load_in_8bit \
       --use_prompt