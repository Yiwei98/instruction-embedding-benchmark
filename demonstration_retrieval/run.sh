#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH=".."
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3

#--------------- Function Parameters ---------------#
MODEL_PATH='model_path'
MODEL_SAVE_PATH='model_save_path'
TRAIN_DATA_PATH='../data/eft_train.json'
EVAL_DATA_PATH='../data/eft_test.json'
POOLING='last-2'

INSTRUCTION_TUNING_TRAIN_PATH='../data/ift_train_sup.json'
INSTRUCTION_TUNING_TRAIN_EMBEDDING_SAVE_PATH='train_data_embedding_save_path'
INSTRUCTION_TUNING_TEST_PATH='../data/ift_test.json'
INSTRUCTION_TUNING_TEST_EMBEDDING_SAVE_PATH='test_data_embedding_save_path'
DEMONSTRATION_SAVE_PATH='./demonstration_save_path'
NUM_DEMO=4

# --------------- Selection Demonstrations with Instruction Embedding ---------------#
cd ../instruction-embedding
python encode.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $INSTRUCTION_TUNING_TRAIN_PATH \
       --save_path $INSTRUCTION_TUNING_TRAIN_EMBEDDING_SAVE_PATH \
       --pooling $POOLING \
       --use_prompt

cd ../instruction-embedding
python encode.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $INSTRUCTION_TUNING_TEST_PATH \
       --save_path $INSTRUCTION_TUNING_TEST_EMBEDDING_SAVE_PATH \
       --pooling $POOLING \
       --use_prompt

cd ../demonstration_retrieval
python selection.py \
       --data_pool_embedding_path $INSTRUCTION_TUNING_TRAIN_EMBEDDING_SAVE_PATH \
       --data_test_embedding_path $INSTRUCTION_TUNING_TEST_EMBEDDING_SAVE_PATH \
       --demonstration_save_path $DEMONSTRATION_SAVE_PATH \
       --num_demo $NUM_DEMO

DEMONSTRATION_INFER_SAVE_PATH='output_save_path'
python infer_with_demonstration.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $INSTRUCTION_TUNING_TEST_PATH \
       --save_path $DEMONSTRATION_INFER_SAVE_PATH \
       --data_pool_path $INSTRUCTION_TUNING_TRAIN_PATH \
       --demonstration_map_path $DEMONSTRATION_SAVE_PATH
