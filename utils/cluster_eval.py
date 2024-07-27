from sklearn.decomposition import PCA

# clustering methods
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import skfuzzy as fuzz

# clustering metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import f1_score


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import json

from utils.clustering import decomposition


def preprocess(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    
    labels = []
    verbs = []
    for key in data.keys():
        labels += [key] * data[key]['number']
        verbs += [key.split('|')[0]] * data[key]['number']

    return labels, verbs


def visualize(embeddings, labels, preds):
    pca = PCA(n_components=2)
    pca.fit(embeddings)

    data = {
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'labels': labels,
        'preds': preds
    }

    sns.set_theme(style = 'whitegrid')
    ax = sns.scatterplot(x='x',y='y', hue='labels',data=data, sizes=(1,1), palette = 'ch:r=-.2,d=.3_r',)
    ax.legend(loc=2, bbox_to_anchor=(1.05,1.0), borderaxespad=0.)

    plt.show()


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def cal_ari(data_path, embedding_path, num_category, eps=0.05, min_samples=1, decomposition_params=None):
    embeddings = torch.load(embedding_path)
    embeddings = np.array([key.detach().numpy() for key in embeddings.keys()])
    if decomposition_params:
        embeddings = decomposition(embeddings, **decomposition_params)

    metric_results = {}

    labels, verbs = preprocess(data_path)

    label_dict = {}
    cnt = 0
    for item in labels:
        if item in label_dict.keys():
            continue
        label_dict[item] = cnt
        cnt += 1
    labels4ari = np.array([label_dict[item] for item in labels])

    # ------------------------------------------------------
    metric_results['kmeans'] = {}
    kmeans = KMeans(n_clusters=num_category, random_state=0, n_init="auto").fit(embeddings)
    labels = kmeans.labels_

    labels_ari = adjusted_rand_score(labels4ari, labels)
    print("K-means ARI: ", labels_ari)
    metric_results['kmeans']['ari'] = float(labels_ari)

    acc = purity_score(labels4ari, labels)
    print("K-means Accuracy: ", acc)
    metric_results['kmeans']['acc'] = float(acc)

    hs = homogeneity_score(labels4ari, labels)
    print("K-means Homogeneity Score: ", hs)
    metric_results['kmeans']['homogeneity'] = float(hs)

    silhouette_avg = silhouette_score(embeddings, labels)
    print("K-means Silhouette Score: ", silhouette_avg)
    metric_results['kmeans']['silhouette'] = float(silhouette_avg)

    # # chi_score = calinski_harabasz_score(embeddings, labels)
    # # print("K-means Calinski_harabasz Score: ", chi_score)

    # # weighted_f1 = f1_score(labels4ari, labels, average='weighted')
    # # print("K-means Weighted-F1 Score: ", weighted_f1)
    # # metric_results['kmeans']['weighted-f1'] = float(weighted_f1)
    # # ------------------------------------------------------

    # # ------------------------------------------------------
    # metric_results['ahc'] = {}
    # clustering = AgglomerativeClustering(n_clusters=num_category).fit(embeddings)
    # labels = clustering.labels_

    # labels_ari = adjusted_rand_score(labels4ari, labels)
    # print("AHC ARI: ", labels_ari)
    # metric_results['ahc']['ari'] = float(labels_ari)

    # acc = purity_score(labels4ari, labels)
    # print("AHC Accuracy: ", acc)
    # metric_results['ahc']['acc'] = float(acc)

    # hs = homogeneity_score(labels4ari, labels)
    # print("AHC Homogeneity Score: ", hs)
    # metric_results['ahc']['homogeneity'] = float(hs)

    # silhouette_avg = silhouette_score(embeddings, labels)
    # print("AHC Silhouette Score: ", silhouette_avg)
    # metric_results['ahc']['silhouette'] = float(silhouette_avg)

    # # chi_score = calinski_harabasz_score(embeddings, labels)
    # # print("AHC Calinski_harabasz Score: ", chi_score)

    # # weighted_f1 = f1_score(labels4ari, labels, average='weighted')
    # # print("AHC Weighted-F1 Score: ", weighted_f1)
    # # metric_results['ahc']['weighted-f1'] = float(weighted_f1)
    # # ------------------------------------------------------

    # # ------------------------------------------------------
    # metric_results['fuzz'] = {}
    # centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(embeddings.T, num_category, 2, error=0.005, maxiter=1000)
    # labels = np.argmax(u, axis=0)

    # labels_ari = adjusted_rand_score(labels4ari, labels)
    # print("Fuzzy ARI: ", labels_ari)
    # metric_results['fuzz']['ari'] = float(labels_ari)

    # acc = purity_score(labels4ari, labels)
    # print("Fuzzy Accuracy: ", acc)
    # metric_results['fuzz']['acc'] = float(acc)

    # hs = homogeneity_score(labels4ari, labels)
    # print("Fuzzy Homogeneity Score: ", hs)
    # metric_results['fuzz']['homogeneity'] = float(hs)

    # silhouette_avg = silhouette_score(embeddings, labels)
    # print("Fuzzy Silhouette Score: ", silhouette_avg)
    # metric_results['fuzz']['silhouette'] = float(silhouette_avg)

    # # chi_score = calinski_harabasz_score(embeddings, labels)
    # # print("Fuzzy Calinski_harabasz Score: ", chi_score)

    # # weighted_f1 = f1_score(labels4ari, labels, average='weighted')
    # # print("Fuzzy Weighted-F1 Score: ", weighted_f1)
    # # metric_results['fuzz']['weighted-f1'] = float(weighted_f1)
    # # ------------------------------------------------------

    # # ------------------------------------------------------
    # metric_results['spectral'] = {}
    # spectral_clustering = SpectralClustering(n_clusters=num_category, affinity='nearest_neighbors', random_state=42)
    # labels = spectral_clustering.fit_predict(embeddings)

    # labels_ari = adjusted_rand_score(labels4ari, labels)
    # print("Spectral ARI: ", labels_ari)
    # metric_results['spectral']['ari'] = float(labels_ari)

    # acc = purity_score(labels4ari, labels)
    # print("Spectral Accuracy: ", acc)
    # metric_results['spectral']['acc'] = float(acc)

    # hs = homogeneity_score(labels4ari, labels)
    # print("Spectral Homogeneity Score: ", hs)
    # metric_results['spectral']['homogeneity'] = float(hs)

    # silhouette_avg = silhouette_score(embeddings, labels)
    # print("Spectral Silhouette Score: ", silhouette_avg)
    # metric_results['spectral']['silhouette'] = float(silhouette_avg)

    # chi_score = calinski_harabasz_score(embeddings, labels)
    # print("Spectral Calinski_harabasz Score: ", chi_score)

    # weighted_f1 = f1_score(labels4ari, labels, average='weighted')
    # print("Spectral Weighted-F1 Score: ", weighted_f1)
    # metric_results['spectral']['weighted-f1'] = float(weighted_f1)
    # ------------------------------------------------------


    # # ------------------------------------------------------
    # metric_results['dbscan'] = {}
    # dbscan_clustering = DBSCAN(eps=eps, min_samples=min_samples)
    # labels = dbscan_clustering.fit_predict(embeddings)

    # labels_ari = adjusted_rand_score(labels4ari, labels)
    # print("DBSCAN ARI: ", labels_ari)
    # metric_results['dbscan']['ari'] = float(labels_ari)

    # acc = purity_score(labels4ari, labels)
    # print("DBSCAN Accuracy: ", acc)
    # metric_results['dbscan']['acc'] = float(acc)

    # hs = homogeneity_score(labels4ari, labels)
    # print("DBSCAN Homogeneity Score: ", hs)
    # metric_results['dbscan']['homogeneity'] = float(hs)

    # silhouette_avg = silhouette_score(embeddings, labels)
    # print("DBSCAN Silhouette Score: ", silhouette_avg)
    # metric_results['dbscan']['silhouette'] = float(silhouette_avg)

    # # chi_score = calinski_harabasz_score(embeddings, labels)
    # # print("Spectral Calinski_harabasz Score: ", chi_score)

    # weighted_f1 = f1_score(labels4ari, labels, average='weighted')
    # print("DBSCAN Weighted-F1 Score: ", weighted_f1)
    # metric_results['dbscan']['weighted-f1'] = float(weighted_f1)
    # # ------------------------------------------------------

    return metric_results

    