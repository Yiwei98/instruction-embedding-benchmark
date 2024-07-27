import os
from utils.cluster_eval import *
import fire
import json


def main(
    data_path: str,
    embedding_path: str,
    save_path: str,
    # dbscan clustering parameters
    min_samples: int=1,
    # decomposition parameters
    decomposition_flag: bool=False,
    n_components: int=2,
    random_state: int=0,
    algorithm: str='t-sne'
):
    print(
        f"Search best eps with params\n"
        f"data_path: {data_path}\n"
        f"embedding_path: {embedding_path}\n"
        f"save_path: {save_path}\n"
        f"min_samples: {min_samples}\n"
    )
    if decomposition_flag:
        print(
            f"n_components: {n_components}\n"
            f"random_state: {random_state}\n"
            f"algorithm: {algorithm}\n"
        )

        decomposition_params = {
            "n_components": n_components,
            "random_state": random_state,
            "algorithm": algorithm,
        }

    embeddings = torch.load(embedding_path)
    embeddings = np.array([key.detach().numpy() for key in embeddings.keys()])
    if decomposition_params:
        embeddings = decomposition(embeddings, **decomposition_params)
    
    labels, verbs = preprocess(data_path)
    label_dict = {}
    cnt = 0
    for item in labels:
        if item in label_dict.keys():
            continue
        label_dict[item] = cnt
        cnt += 1
    labels4ari = np.array([label_dict[item] for item in labels])

    metric_results = {}
    output_labels = {
        'ground_truth': labels4ari.tolist(),
    }
    for step in range(1, 21):
        eps = 0.01 * step
        metric_results[eps] = {}
        dbscan_clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan_clustering.fit_predict(embeddings)
        output_labels[str(eps)] = labels.tolist()

        print(f'CURRENT DBSCAN EPS: {eps}')
        labels_ari = adjusted_rand_score(labels4ari, labels)
        print("DBSCAN ARI: ", labels_ari)
        metric_results[eps]['ari'] = float(labels_ari)

        acc = purity_score(labels4ari, labels)
        print("DBSCAN Accuracy: ", acc)
        metric_results[eps]['acc'] = float(acc)

        hs = homogeneity_score(labels4ari, labels)
        print("DBSCAN Homogeneity Score: ", hs)
        metric_results[eps]['homogeneity'] = float(hs)

        silhouette_avg = silhouette_score(embeddings, labels)
        print("DBSCAN Silhouette Score: ", silhouette_avg)
        metric_results[eps]['silhouette'] = float(silhouette_avg)

        weighted_f1 = f1_score(labels4ari, labels, average='weighted')
        print("DBSCAN Weighted-F1 Score: ", weighted_f1)
        metric_results[eps]['weighted-f1'] = float(weighted_f1)
    with open(save_path, 'w') as f:
        json.dump(metric_results, f, indent=4)
    with open('output_label_check.json', 'w') as f:
        json.dump(output_labels, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)