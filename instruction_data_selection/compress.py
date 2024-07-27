import os
import sys

from utils.clustering import *
import fire


def main(
        data_path: str, 
        embedding_path: str, 
        save_path: str,  
        algorithm: str='dbscan',        # ['dbscan', 'k-means']
        top_k: int=1, 
        decompose_algorithm=None,       # ['pca', 't-sne']
        n_components=None,
        # k-means parameters
        n_clusters: int=None, 
        random_state: int=None, 
        n_init: str=None,
        # dbscan parameters
        eps: float=None,
        min_samples: int=None,
        list_style: bool=False,
):
    print(
        f"Dataset Compression with Parameters\n"
        f"data_path: {data_path}\n"
        f"embedding_path: {embedding_path}\n"
        f"save_path: {save_path}\n"
        f"algorithm: {algorithm}\n"
        f"top_k: {top_k}\n"
        f"decompose_algorithm: {decompose_algorithm}\n"
        f"n_components: {n_components}\n"
        f"n_clusters: {n_clusters}\n"
        f"random_state: {random_state}\n"
        f"n_init: {n_init}\n"
        f"eps: {eps}\n"
        f"min_samples: {min_samples}\n"
        f"list_style: {list_style}\n"
    )
    dataset_compression(data_path=data_path, embedding_path=embedding_path, save_path=save_path, 
                        algorithm=algorithm, top_k=top_k, 
                        n_clusters=n_clusters, random_state=random_state, n_init=n_init, 
                        eps=eps, min_samples=min_samples, 
                        decompose_algorithm=decompose_algorithm, n_components=n_components,
                        list_style=list_style)


if __name__ == '__main__':
    fire.Fire(main)