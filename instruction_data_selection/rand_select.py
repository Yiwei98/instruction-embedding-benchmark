import os
import sys
sys.path.append(os.path.abspath('/home/sjy/instruction-embedding-benchmark'))

import json
import random
import argparse

from utils.clustering import *

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, default='../data/instruction_tuning_train.json', required=False)
    parse.add_argument('--save_path', type=str, default='./random_600.json', required=False)
    parse.add_argument('--num', type=int, default=600, required=False)
    parse.add_argument('--seed', type=int, default=20231223, required=False)
    args = parse.parse_args()
    return args


def main(args):
    random.seed(args.seed)

    with open(args.data_dir, 'r') as f:
        data = json.load(f)

    pool = []
    for label in data.keys():
        samples = data[label]['samples']
        for sample in samples:
           pool.append(sample)

    data = random.sample(pool, args.num)
    with open(args.save_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)