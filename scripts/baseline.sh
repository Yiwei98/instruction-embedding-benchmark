#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH="../"
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH


export CUDA_VISIBLE_DEVICES=2

#--------------- Function Parameters ---------------#
MODEL_PATH='path'
MODEL_SAVE_PATH='path'
TRAIN_DATA_PATH='../../data/embedding_train_for_simcse.json'
TRAIN_DATA_PATH='../../data/wiki1m_for_simcse.json'
EVAL_DATA_PATH='../data/embedding_test.json'
MODEL_SAVE_PATH='path'

EMBEDDING_SAVE_PATH='path'
batch_size=32
# POOLING='last-2'
POOLING='cls'
EPOCHS=1

IIS_DATA_PATH='../data/intention_similarity_test.json'
CLUSTERING_EVAL_RESULT_SAVE_PATH='path'
NUM_CATEGORY=145
EPS=0.05
MIN_SAMPLES=1
N_COMPONENTS=2
RANDOM_STATE=0
DECOMPOSITION_ALGORITHM='t-sne'

INSTRUCTION_TUNING_TRAIN_PATH='../data/instruction_tuning_train.json'
INSTRUCTION_TUNING_TRAIN_EMBEDDING_SAVE_PATH='path'
INSTRUCTION_TUNING_TEST_PATH='../data/instruction_tuning_test.json'
INSTRUCTION_TUNING_TEST_EMBEDDING_SAVE_PATH='path'
DEMONSTRATION_SAVE_PATH='./demonstration_map_ieb-sup-bert.json'
NUM_DEMO=1

DEMONSTRATION_INFER_SAVE_PATH='path'

cd ../baselines/simcse_framework
python main.py \
    --model_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --save_path ../$MODEL_SAVE_PATH \
    --epochs $EPOCHS \
    --batch_size $batch_size \
    --bert \


# --------------- Get Instruction Embedding ---------------#
cd ../../instruction-embedding
python encode.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $EVAL_DATA_PATH \
       --save_path $EMBEDDING_SAVE_PATH \
       --pooling $POOLING \
       --bert \


# --------------- Run Instruction Intention Similarity Test ---------------#
cd ../intention-similarity-test
python main.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $IIS_DATA_PATH \
       --pooling $POOLING \
       --bert \


#--------------- Run Instruction Embedding Clustering Test ---------------#
cd ../clustering-eval
python main.py \
    --data_path $EVAL_DATA_PATH \
    --embedding_path $EMBEDDING_SAVE_PATH \
    --save_path $CLUSTERING_EVAL_RESULT_SAVE_PATH \
    --num_category $NUM_CATEGORY \
    --eps $EPS \
    --min_samples $MIN_SAMPLES \
#     --decomposition \
#     --n_components $N_COMPONENTS \
#     --random_state $RANDOM_STATE \
#     --algorithm $DECOMPOSITION_ALGORITHM \


#--------------- Selection Demonstrations with Instruction Embedding ---------------#
cd ../instruction-embedding
python encode.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $INSTRUCTION_TUNING_TRAIN_PATH \
       --save_path $INSTRUCTION_TUNING_TRAIN_EMBEDDING_SAVE_PATH \
       --pooling $POOLING \
       --use_prompt

python encode.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $INSTRUCTION_TUNING_TEST_PATH \
       --save_path $INSTRUCTION_TUNING_TEST_EMBEDDING_SAVE_PATH \
       --pooling $POOLING \
       --use_prompt

cd ../select-demonstration
python selection.py \
       --data_pool_embedding_path $INSTRUCTION_TUNING_TRAIN_EMBEDDING_SAVE_PATH \
       --data_test_embedding_path $INSTRUCTION_TUNING_TEST_EMBEDDING_SAVE_PATH \
       --demonstration_save_path $DEMONSTRATION_SAVE_PATH \
       --num_demo $NUM_DEMO

python infer_with_demonstration.py \
       --model_path $MODEL_SAVE_PATH \
       --tokenizer_path $MODEL_PATH \
       --eval_data_path $INSTRUCTION_TUNING_TEST_PATH \
       --save_path $DEMONSTRATION_INFER_SAVE_PATH \
       --data_pool_path $INSTRUCTION_TUNING_TRAIN_PATH \
       --demonstration_map_path $DEMONSTRATION_SAVE_PATH \


#--------------- Instruction Data Selection with Instruction Embedding ---------------#
cd ../instruction_data_selection
python compress.py \
    --data_path $DATA_PATH \
    --embedding_path $EMBEDDING_PATH \
    --save_path $SAVE_PATH \
    --algorithm $ALGORITHM \
    --top_k $TOP_K \
    --n_clusters $N_CLUSTERS \
    --random_state $RANDOM_STATE \
    --n_init $N_INIT