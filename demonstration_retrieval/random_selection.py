
import numpy as np
import json
from tqdm import tqdm
import random
import fire
        
def jload(path):
    with open(path) as f:
        data = json.load(f)
    return data



def main( ):
    test_data = jload('/home/sjy/instruction-embedding-benchmark/data/alpaca-eval/alpaca_eval.json')
    data_pool = jload('/home/sjy/instruction-embedding-benchmark/data/instruction_tuning_train_list.json')

    demonstration_map = {}

    for index in tqdm(range(len(test_data))):
        indices = random.sample(range(len(data_pool)), 4)
        demonstration_map[index] = indices

    with open('./alpaca_eval-random_demonstrations.json', 'w') as f:
        json.dump(demonstration_map, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
