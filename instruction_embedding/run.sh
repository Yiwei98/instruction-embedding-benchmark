#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH=".."
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH

export CUDA_VISIBLE_DEVICES=0

## lora fine-tune
MODEL_PATH='model_path'
TRAIN_DATA_PATH='../data/eft_train.json'
EVAL_DATA_PATH='../data/eft_test.json'
MODEL_SAVE_PATH='model_save_path'
EMBEDDING_SAVE_PATH='embedding_save_path.pth'
POOLING='last-2'
EPOCHS=1

python main.py \
    --model_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --save_path "$MODEL_SAVE_PATH" \
    --pooling "$POOLING" \
    --epochs "$EPOCHS" \
    --lora \
    --load_in_8bit \
    --use_prompt

python encode.py \
       --model_path "$MODEL_PATH" \
       --eval_data_path "$EVAL_DATA_PATH" \
       --save_path "$EMBEDDING_SAVE_PATH" \
       --pooling "$POOLING" \
       --lora_weight "$MODEL_SAVE_PATH" \
       --lora \
       --load_in_8bit \
       --use_prompt


# ## full fine-tune
# MODEL_PATH='/backup/sjy/models/LLaMA/llama-7b-hf'
# TRAIN_DATA_PATH='/home/sjy/code/data/embedding_train.json'
# EVAL_DATA_PATH='/home/sjy/code/data/embedding_test.json'
# MODEL_SAVE_PATH='finetuned-llama'
# EMBEDDING_SAVE_PATH='finetuned-prompt-llama_final.pth'
# POOLING='last-2'
# EPOCHS=3

# python main.py \
#     --model_path "$MODEL_PATH" \
#     --train_data_path "$TRAIN_DATA_PATH" \
#     --eval_data_path "$EVAL_DATA_PATH" \
#     --save_path "$MODEL_SAVE_PATH" \
#     --pooling "$POOLING" \
#     --epochs "$EPOCHS" \
#     --use_prompt

# python encode.py \
#        --model_path "$MODEL_SAVE_PATH" \
#        --eval_data_path "$EVAL_DATA_PATH" \
#        --save_path "$EMBEDDING_SAVE_PATH" \
#        --pooling "$POOLING" \
#        --use_prompt
