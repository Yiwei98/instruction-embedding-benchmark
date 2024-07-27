#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH=".."
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3

#--------------- Function Parameters ---------------#
EVAL_DATA_PATH='../data/ift_test.json'
MODEL_PATH='model_path'
SAVE_PATH='save_path'

python infer.py \
    --model_path $MODEL_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    -save_path $SAVE_PATH \
    --use_prompt \
    --short_prompt