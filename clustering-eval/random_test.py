import os
from utils.cluster_eval import *
import fire
import json

import random


def main(
    data_path: str,
    # embedding_path: str,
    # save_path: str,
    num_category: int=145,
):
    print(
        f"Evaluate embedding with params\n"
        f"data_path: {data_path}\n"
        # f"embedding_path: {embedding_path}\n"
        # f"save_path: {save_path}\n"
        f"num_category: {num_category}\n"
    )

    # embeddings = torch.load(embedding_path)
    # embeddings = np.array([key.detach().numpy() for key in embeddings.keys()])

    metric_results = {
        'random': {
            'ari': [],
            'acc': [],
            'homogeneity': [],
            # 'silhouette': []
        }
    }
    labels, verbs = preprocess(data_path)
    for _ in range(100):
        labels4ari = np.array([random.choice(range(num_category)) for _ in range(len(labels))])

        labels_ari = adjusted_rand_score(labels4ari, labels)
        # print("RANDOM ARI: ", labels_ari)
        metric_results['random']['ari'].append(float(labels_ari))

        acc = purity_score(labels4ari, labels)
        # print("RANDOM Accuracy: ", acc)
        metric_results['random']['acc'].append(float(acc))

        hs = homogeneity_score(labels4ari, labels)
        # print("RANDOM Homogeneity Score: ", hs)
        metric_results['random']['homogeneity'].append(float(hs))

        # silhouette_avg = silhouette_score(labels4ari, labels)
        # print("RANDOM Silhouette Score: ", silhouette_avg)
        # metric_results['random']['silhouette'].append(float(silhouette_avg))

    for key in metric_results['random'].keys():
        print(f"{key}: {sum(metric_results['random'][key]) / len(metric_results['random'][key])}")

    # os.makedirs('../embedding_clustering_eval_results', exist_ok=True)
    # with open(f'{save_path}', 'w') as f:
    #     json.dump(metric_results, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
    