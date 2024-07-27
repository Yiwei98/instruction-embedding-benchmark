from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from tqdm import tqdm
import numpy as np
import torch
import json

from typing import List, Dict


def make_index(path: str, list_style: bool=False) -> Dict[int, Dict]:
    with open(path, 'r') as f:
        dataset = json.load(f)

    idx2data = {}
    if list_style:
        for sample in dataset:
            idx2data[len(idx2data.keys())] = sample

    else:
        for label in dataset.keys():
            samples = dataset[label]['samples']
            for sample in samples:
                sample['label'] = label
                idx2data[len(idx2data.keys())] = sample
    return idx2data


def index2data(index: List[int], idx2data: Dict[int, Dict]) -> List[Dict]:
    data = [idx2data[item] for item in index]
    return data


def decomposition(embeddings, n_components=2, random_state=0, algorithm='t-sne'):
    if algorithm == 't-sne':
        tsne = TSNE(n_components=n_components, random_state=random_state)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif algorithm == 'pca':
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        raise NotImplementedError
    
    return reduced_embeddings



def kmeans_clustering(path: str, n_clusters: int, random_state=0, n_init="auto", top_k: int=1, decompose_algorithm=None, n_components=None) -> List[int]:
    embeddingsDict = torch.load(path)       # embeddingsDict: {embedding, data_index}
    all_embeddings = list(embeddingsDict.keys())
    embeddings = np.array([key.numpy() for key in embeddingsDict.keys()]).astype(np.float32)
    reduced_embeddings = embeddings
    
    if decompose_algorithm is not None and n_components is not None:
        reduced_embeddings = decomposition(embeddings, n_components=n_components, algorithm=decompose_algorithm)

    print(f"BEGIN CLUSTERING")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(reduced_embeddings if decompose_algorithm else embeddings)
    kmeans_labels = kmeans.labels_
    kmeans_centers = kmeans.cluster_centers_
    print(f"FINISH CLUSTERING")

    data_index = []
    for label in tqdm(range(kmeans_centers.shape[0])):
        class_member_mask = (kmeans_labels == label)
        cluster_indices = np.where(class_member_mask)[0]
        cluster_points = reduced_embeddings[cluster_indices]
        centroid = kmeans_centers[label]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        top_k_indices = np.argsort(distances)[:top_k]
        original_indices = cluster_indices[top_k_indices].tolist()

        for indice in original_indices:
            data_index.append(embeddingsDict[all_embeddings[indice]])
    
    return data_index

    # n_dim = kmeans_centers.shape[1]
    # for i in tqdm(range(kmeans_centers.shape[0])):
    #     center = torch.tensor(kmeans_centers[i], requires_grad=False).cpu().unsqueeze(0)
    #     index = []
    #     for j in range(kmeans_labels.shape[0]):
    #         if kmeans_labels[j] == i:
    #             index.append(j)
    #     embeddings = [all_embeddings[j] for j in index]
    #     similarity = torch.cosine_similarity(torch.cat(embeddings, dim=0).reshape(-1, n_dim), center, dim=1).flatten()
    #     indices = torch.topk(-similarity, k=min(cluster_size, similarity.shape[0]), dim=0).indices.numpy().tolist()
    #     for indice in indices:
    #         data_index.append(embeddingsDict[embeddings[indice]])

    # return data_index


def dbscan_clustering(path: str, eps=0.5, min_samples=1, top_k=1, decompose_algorithm=None, n_components=None):
    embeddingsDict = torch.load(path)       # embeddingsDict: {embedding, data_index}
    all_embeddings = list(embeddingsDict.keys())
    embeddings = np.array([key.numpy() for key in embeddingsDict.keys()]).astype(np.float32)

    if decompose_algorithm is not None and n_components is not None:
        reduced_embeddings = decomposition(embeddings, n_components=n_components, algorithm=decompose_algorithm)
    
    print(f"BEGIN CLUSTERING")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(reduced_embeddings)
    print(f"FINISH CLUSTERING")

    labels2index = {}
    data_index = []
    for i in range(reduced_embeddings.shape[0]):
        if labels[i] == -1:
            continue
        if labels[i] not in labels2index.keys():
            labels2index[labels[i]] = []
        labels2index[labels[i]].append(i)

    for label in labels2index.keys():
        class_member_mask = (labels == label)
        cluster_indices = np.where(class_member_mask)[0]
        cluster_points = reduced_embeddings[class_member_mask]
        centroid = cluster_points.mean(axis=0)

        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        top_k_indices = np.argsort(distances)[:min(top_k, cluster_points.shape[0])]
        original_indices = cluster_indices[top_k_indices].tolist()

        for indice in original_indices:
            data_index.append(embeddingsDict[all_embeddings[indice]])
    
    return data_index


def dataset_compression(data_path: str, embedding_path: str, save_path: str, algorithm: str='dbscan', top_k: int=1,
                        n_clusters: int=63, random_state=0, n_init="auto",
                        eps=0.5, min_samples=1,
                        decompose_algorithm=None, n_components=None,
                        list_style=False
                        ):
    idx2data = make_index(data_path, list_style)
    if algorithm == 'k-means':
        data_index = kmeans_clustering(path=embedding_path, 
                                       n_clusters=n_clusters, random_state=random_state, n_init=n_init, top_k=top_k, 
                                       decompose_algorithm=decompose_algorithm, n_components=n_components)
    elif algorithm == 'dbscan':
        data_index = dbscan_clustering(path=embedding_path, eps=eps, min_samples=min_samples, top_k=top_k, decompose_algorithm=decompose_algorithm, n_components=n_components)
    else:
        raise NotImplementedError
    
    data = index2data(data_index, idx2data)

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)



