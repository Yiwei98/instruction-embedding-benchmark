from torch.utils.data import Dataset
from transformers import LlamaTokenizer
import json

class DatasetForCL(Dataset):
    def __init__(self, data_path, prompt=None, use_prompt=False):
        super().__init__()
        self.data = self.jload(data_path)
        self.prompt = prompt
        self.use_prompt = use_prompt

    def jload(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
            f.close()
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index].strip()
        if self.use_prompt:
            item = self.prompt.format(item)

        return item
    

class DataCollatorForCL:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sentences = []
        for sent in batch:
            # add the same sentence into list twice to construct positive pairs in unsupervised training
            sentences.append(sent.strip())
            sentences.append(sent.strip())

            features = self.tokenizer(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )
        return features
