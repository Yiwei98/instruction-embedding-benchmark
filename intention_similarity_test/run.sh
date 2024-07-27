#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH=".."
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_path='model_path'
iis_data_path='../data/intention_similarity_test.json'
pooling='last-2'

python main.py \
        --model_path $model_path \
        --eval_data_path $iis_data_path \
        --pooling $pooling \
        --load_in_8bit \
        --use_prompt

#python random_test.py