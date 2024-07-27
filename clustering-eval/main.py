import os
from utils.cluster_eval import *
import fire
import json


def main(
    data_path: str,
    embedding_path: str,
    save_path: str,
    num_category: int=63,
    # dbscan clustering parameters
    eps: float=0.05,
    min_samples: int=1,
    # decomposition parameters
    decomposition: bool=False,
    n_components: int=2,
    random_state: int=0,
    algorithm: str='t-sne'
):
    print(
        f"Evaluate embedding with params\n"
        f"data_path: {data_path}\n"
        f"embedding_path: {embedding_path}\n"
        f"save_path: {save_path}\n"
        f"num_category: {num_category}\n"
        f"eps: {eps}\n"
        f"min_samples: {min_samples}\n"
    )
    if decomposition:
        print(
            f"n_components: {n_components}\n"
            f"random_state: {random_state}\n"
            f"algorithm: {algorithm}\n"
        )

        decomposition = {
            "n_components": n_components,
            "random_state": random_state,
            "algorithm": algorithm,
        }

    result = cal_ari(data_path, embedding_path, num_category, eps, min_samples, decomposition)
    os.makedirs('../embedding_clustering_eval_results', exist_ok=True)
    with open(f'{save_path}', 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
    