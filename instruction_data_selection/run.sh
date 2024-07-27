#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH=".."
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH

DATA_PATH='../data/ift_train.json'
EMBEDDING_PATH='embedding_path'
SAVE_PATH='data_save_path'
ALGORITHM='k-means'
TOP_K=1
# DECOMPOSITION_ALGORITHM='t-sne'
# N_COMPONENTS=2
# if DBSCAN
EPS=0.05
MIN_SAMPLES=2
# if k-means
N_CLUSTERS=600
RANDOM_STATE=0
N_INIT='auto'


python compress.py \
    --data_path $DATA_PATH \
    --embedding_path $EMBEDDING_PATH \
    --save_path $SAVE_PATH \
    --algorithm $ALGORITHM \
    --top_k $TOP_K \
    --n_clusters $N_CLUSTERS \
    --random_state $RANDOM_STATE \
    --n_init $N_INIT

