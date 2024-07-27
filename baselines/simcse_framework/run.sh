#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH="../../"
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH


export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH='model_path'
TRAIN_DATA_PATH='../../data/wiki1m_for_simcse.json'
MODEL_SAVE_PATH='modela_save_path'
batch_size=8
EPOCHS=1

python main.py \
    --model_path "$MODEL_PATH" \
    --data_path "$TRAIN_DATA_PATH" \
    --save_path "$MODEL_SAVE_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$batch_size" \
    --use_prompt \
    --bert