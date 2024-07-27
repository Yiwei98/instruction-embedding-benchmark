import os
import sys

from tqdm import tqdm
import numpy as np
import torch
import json

from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.clustering import decomposition


def visualization_with_label(embedidng_path, decomposition_params, img_save_path, labels=None, clustering_algorithm='dbscan', clustering_params=None, sample_num=None):
    embeddings = torch.load(embedidng_path)
    embeddings = np.array([key.detach().numpy() for key in embeddings.keys()]).astype(np.float32)
    reduced_embeddings = decomposition(embeddings, **decomposition_params)

    if labels is None:
        if clustering_algorithm == 'dbscan':
            dbscan = DBSCAN(**clustering_params)
            labels = dbscan.fit_predict(reduced_embeddings)
        elif clustering_algorithm == 'k-means':
            kmeans = KMeans(**clustering_params).fit(reduced_embeddings)
            labels = kmeans.labels_
        else:
            raise NotImplementedError

    idx2label = {}
    for item in labels:
        idx2label[len(idx2label.keys())] = item

    label_count = {}
    for i in range(len(labels)):
        if labels[i] not in label_count.keys():
            label_count[labels[i]] = 0
        label_count[labels[i]] += 1

    sampled_labels = []
    for label in label_count.keys():
        if label_count[label] > 30:
            sampled_labels.append(label)
    # print(len(sampled_labels))

    sampled_idx = []
    for i in range(len(labels)):
        if labels[i] in sampled_labels:
            sampled_idx.append(i)

    # print(len(sampled_idx))
    labels = np.array(labels)[sampled_idx]
    reduced_embeddings = reduced_embeddings[sampled_idx]
        
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    df['label'] = labels

    # if sample_num:
    #     df = df.sample(n=min(sample_num, reduced_embeddings.shape[0]))

    sns.scatterplot(data=df, x='x', y='y', hue='label', palette=sns.color_palette("tab20", n_colors=len(set(labels))), s=100)
    plt.legend().remove()
    ax = plt.gca()
    ax.axis('off')
    plt.xlim(df['x'].min() - 1, df['x'].max() + 1)
    plt.ylim(df['y'].min() - 1, df['y'].max() + 1)
    plt.tight_layout()
    plt.savefig(img_save_path, format='svg')


def inter_dataset_cosine_similarity(embeddings1, embeddings2):
    # embeddings1 = torch.load(embedding_path1)
    # embeddings1 = np.array([key.detach().numpy() for key in embeddings1.keys()]).astype(np.float32)
    # embeddings2 = torch.load(embedding_path2)
    # embeddings2 = np.array([key.detach().numpy() for key in embeddings2.keys()]).astype(np.float32)

    mean1, cov1 = np.mean(embeddings1, axis=0), np.cov(embeddings1, rowvar=False)
    mean2, cov2 = np.mean(embeddings2, axis=0), np.cov(embeddings2, rowvar=False)    
    # jsd = jensenshannon(mean1, mean2)

    # mean_diff = mean1 - mean2
    # cov_mean = (cov1 + cov2) / 2
    combined1 = np.concatenate((mean1, cov1.flatten()))
    combined2 = np.concatenate((mean2, cov2.flatten()))
    distribution_sim = jensenshannon(combined1, combined2)

    print(f"Distribution Similarity (Jensen-Shannon Divergence): {distribution_sim}")
    return distribution_sim


def nearest_neighbor_similarity(embeddings1, embeddings2):
    # embeddings1 = torch.load(embedding_path1)
    # embeddings1 = np.array([key.detach().numpy() for key in embeddings1.keys()]).astype(np.float32)
    # embeddings2 = torch.load(embedding_path2)
    # embeddings2 = np.array([key.detach().numpy() for key in embeddings2.keys()]).astype(np.float32)

    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    nearest_neighbor_similarities = similarity_matrix.max(axis=1)
    mean_nearest_neighbor_similarity = nearest_neighbor_similarities.mean()
    print(f"Mean Nearest Neighbor Cosine Similarity: {mean_nearest_neighbor_similarity}")
    return mean_nearest_neighbor_similarity


def calculate_wasserstein(embeddings1, embeddings2):
    # 确保两组嵌入的hidden size相同
    assert embeddings1.shape[1] == embeddings2.shape[1], "Hidden size must be the same"

    num_features = embeddings1.shape[1]
    distances = []

    for i in range(num_features):
        distances.append(wasserstein_distance(embeddings1[:, i], embeddings2[:, i]))

    avg_distance = np.mean(distances)
    
    print(f"Average Wasserstein Distance: {avg_distance}")
    return avg_distance


def calculate_jsd_histogram(embeddings1, embeddings2, bins=100):
    num_features = embeddings1.shape[1]
    jsd_distances = []

    for i in range(num_features):
        hist1, _ = np.histogram(embeddings1[:, i], bins=bins, density=True)
        hist2, _ = np.histogram(embeddings2[:, i], bins=bins, density=True)
        
        # 为避免计算log(0)的情况，确保直方图中没有零
        hist1 += 1e-10
        hist2 += 1e-10
        
        jsd_distances.append(jensenshannon(hist1, hist2))

    avg_jsd = np.mean(jsd_distances)
    
    print(f"Average Jensen-Shannon Divergence: {avg_jsd}")
    return avg_jsd


def calculate_overlap_and_unique_in_batches(embeddings1, embeddings2, batch_size=1000, threshold=0.3):
    """
    Calculate the overlap and unique counts between two datasets of embeddings using batch processing.
    
    Parameters:
    embeddings1 (numpy.ndarray): The first set of embeddings.
    embeddings2 (numpy.ndarray): The second set of embeddings.
    batch_size (int): Size of the batch for processing.
    threshold (float): Distance threshold to consider an embedding as overlapping.

    Returns:
    tuple: (overlap_count, unique_in_dataset1, unique_in_dataset2)
    """
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(embeddings2)
    
    overlap_count = 0
    num_batches = int(np.ceil(len(embeddings1) / batch_size))
    
    for i in range(num_batches):
        batch_embeddings1 = embeddings1[i * batch_size:(i + 1) * batch_size]
        distances, _ = neighbors.kneighbors(batch_embeddings1)
        overlap_count += np.sum(distances < threshold)
    
    unique_dataset1_count = len(embeddings1) - overlap_count
    unique_dataset2_count = len(embeddings2) - overlap_count

    return overlap_count, unique_dataset1_count, unique_dataset2_count

# # Example usage:
# # embeddings1 and embeddings2 are numpy arrays of shape (n_samples, n_features)
# overlap_count, unique_dataset1_count, unique_dataset2_count = calculate_overlap_and_unique_in_batches(embeddings1, embeddings2, batch_size=1000, threshold=0.5)

# print(f"Overlap count: {overlap_count}")
# print(f"Unique in Dataset 1: {unique_dataset1_count}")
# print(f"Unique in Dataset 2: {unique_dataset2_count}")


def max_cosine_similarity(A, B, block_size=100):
    m, d = A.shape
    n = B.shape[0]
    
    max_similarities = np.zeros(m)
    
    for i in range(0, m, block_size):
        end_i = min(i + block_size, m)
        A_block = A[i:end_i]

        cosine_sim_block = cosine_similarity(A_block, B)
        max_similarities[i:end_i] = np.max(cosine_sim_block, axis=1)
    
    return max_similarities

