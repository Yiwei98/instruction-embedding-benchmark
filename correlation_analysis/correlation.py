import os
import sys

import json
import fire
from tqdm import tqdm
from utils.embedding_analysis import *

label_map = { 
    'gsm8k_train.json.pth': 'gsm8k',
    'all_math.json.pth': 'MATH',
    'mbpp.json.pth': 'mbpp',
    'lima.json.pth': 'LIMA',
    'dolly.json.pth': 'dolly',
    'oassist_train.json.pth': 'OASSIT',
    'alpaca_data_cleaned.json.pth': 'Alpaca',
    'wizardlm_alpaca.json.pth': 'WizardLM(Alpaca)',
    'wizardlm_sharegpt.json.pth': 'WizardLM(ShareGPT)',
    'sharegpt.json.pth': 'ShareGPT', 
}


def main(
    embedding_dir: str,
):
    embedding_dict = {}
    for embedding_path in label_map.keys():
        embeddings = torch.load(os.path.join(embedding_dir, embedding_path))
        embeddings = np.array([key.detach().numpy() for key in embeddings.keys()]).astype(np.float32)
        embedding_dict[embedding_path] = embeddings / embeddings.sum(axis=1, keepdims=True)

    num_datasets = len(list(embedding_dict.keys()))
    pairs = []
    for i in range(num_datasets-1):
        for j in range(i+1,num_datasets):
            pairs.append([i,j])

    mat1 = np.zeros((num_datasets, num_datasets))
    mat2 = np.zeros((num_datasets, num_datasets))
    mat3 = np.ones((num_datasets, num_datasets))
    keys = list(embedding_dict.keys())
    for (x, y) in tqdm(pairs):
        print(f"Current compare datasets\n\t{keys[x]}\n\t{keys[y]}")

        # jsd = calculate_jsd_histogram(embedding_dict[keys[x]], embedding_dict[keys[y]])
        # mat1[x][y] = jsd
        # mat1[y][x] = jsd

        wasserstein = calculate_wasserstein(embedding_dict[keys[x]], embedding_dict[keys[y]])
        mat2[x][y] = wasserstein
        mat2[y][x] = wasserstein

        # overlap_count, unique_dataset1_count, unique_dataset2_count = calculate_overlap_and_unique_in_batches(embedding_dict[keys[x]], embedding_dict[keys[y]], batch_size=1000, threshold=0.5)
        # mat3[x][y] = overlap_count / embedding_dict[keys[x]].shape[0]
        # mat3[y][x] = overlap_count / embedding_dict[keys[y]].shape[0]

        mat3[x][y] = max_cosine_similarity(embedding_dict[keys[x]], embedding_dict[keys[y]], block_size=1000).sum() / embedding_dict[keys[x]].shape[0]
        mat3[y][x] = max_cosine_similarity(embedding_dict[keys[y]], embedding_dict[keys[x]], block_size=1000).sum() / embedding_dict[keys[y]].shape[0]


    
    labels = list(embedding_dict.keys())
    labels = [label_map[item] for item in labels]

    # sns.heatmap(mat1, annot=True, cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    # plt.xticks(rotation=90)
    # plt.title("Heatmap Example")
    # plt.xlabel("X-axis Label")
    # plt.ylabel("Y-axis Label")
    # plt.savefig('inter_dataset_cosine_similarity.svg', format='svg')
    # plt.close()

    # plt.figure(figsize=(14, 12))
    plt.figure(figsize=(6, 3))
    sns.set(font='Times New Roman')
    # sns.heatmap(mat2, annot=True, cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    ax = sns.heatmap(mat2, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)
    # plt.title("Heatmap Example")
    # plt.xlabel("X-axis Label")
    # plt.ylabel("Y-axis Label")
    plt.tight_layout(pad=0.5)
    # plt.savefig(f'{embedding_dir}/wasserstein_distance_new.svg', format='svg')
    plt.savefig(f'{embedding_dir}/wasserstein_distance_normed.pdf', format='pdf')
    plt.close()

    # plt.figure(figsize=(16, 14))
    # sns.heatmap(mat3, annot=True, cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    # plt.xticks(rotation=45, ha='right')
    # # plt.title("Heatmap Example")
    # # plt.xlabel("X-axis Label")
    # # plt.ylabel("Y-axis Label")
    # plt.savefig(f'{embedding_dir}/max_cosine_similarity.svg', format='svg')
    # plt.close()

    plt.figure(figsize=(6, 3))
    sns.set(font='Times New Roman')
    ax = sns.heatmap(mat3, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=labels, yticklabels=labels, cbar=False)
    # plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)
    # plt.title("Heatmap Example")
    # plt.xlabel("X-axis Label")
    # plt.ylabel("Y-axis Label")
    plt.tight_layout(pad=0.5)
    plt.savefig(f'{embedding_dir}/max_cosine_similarity_normed.pdf', format='pdf')
    plt.close()

    
if __name__ == '__main__':
    fire.Fire(main)
