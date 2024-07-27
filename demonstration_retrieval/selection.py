import torch
import numpy as np
import json
from tqdm import tqdm
import random
import fire
        

def load_embedding(path):
    embeddings = torch.load(path)
    embeddings = np.array([key.detach().numpy() for key in embeddings.keys()])

    return torch.tensor(embeddings, dtype=torch.float32)


def find_demonstration(src_embedding, embedding_pool, num_demo):
    similarity = torch.cosine_similarity(embedding_pool, src_embedding.unsqueeze(0), dim=1).flatten()
    indices = torch.topk(similarity, k=min(num_demo, similarity.shape[0]), dim=0).indices.numpy().tolist()

    return indices


def main(
    data_pool_embedding_path: str,
    data_test_embedding_path: str,
    demonstration_save_path: str,
    num_demo: int=3,
):
    print(
            f"Select demonstrations with params:\n"
            f"data_pool_embedding_path: {data_pool_embedding_path}\n"
            f"demonstration_save_path: {demonstration_save_path}\n"
            f"data_test_embedding_path: {data_test_embedding_path}\n"
            f"num_demo: {num_demo}\n"
        )
    
    demonstration_map = {}

    data_pool_embedding = load_embedding(data_pool_embedding_path)
    data_test_embedding = load_embedding(data_test_embedding_path)

    for index in tqdm(range(data_test_embedding.shape[0])):
        src_embedding = data_test_embedding[index]
        indices = find_demonstration(src_embedding, data_pool_embedding, num_demo)
        demonstration_map[index] = indices

    with open(demonstration_save_path, 'w') as f:
        json.dump(demonstration_map, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
