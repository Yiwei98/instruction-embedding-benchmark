#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RELATIVE_PATH=".."
ABSOLUTE_PATH=$(realpath "$SCRIPT_DIR/$RELATIVE_PATH")
export PYTHONPATH=$PYTHONPATH:$ABSOLUTE_PATH

data_path='../data/eft_test.json'
embedding_path='embedding_path'
save_path='save_path'
num_category=145

python random_test.py \
    --data_path $data_path \
    --num_category $num_category

 python main.py \
     --data_path $data_path \
     --embedding_path $embedding_path \
     --save_path $save_path \
     --num_category $num_category
