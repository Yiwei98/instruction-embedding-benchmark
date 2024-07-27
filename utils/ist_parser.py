import json
from torch.utils.data import Dataset

class ITSDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.data = self.jload(path)
    
    def jload(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

class ITSDataCollator:
    def __init__(self, tokenizer, template) -> None:
        self.tokenizer = tokenizer
        self.template = template

    def __call__(self, batch):
        if self.template is None:
            instruction_pairs = [[item['instruction1'], item['instruction2']] for item in batch]
        else:
            instruction_pairs = [[self.template.replace('*sent*', item['instruction1']), self.template.replace('*sent*', item['instruction2'])] for item in batch]
        
        labels = [item['label'] for item in batch]
        instructions = []
        for pair in instruction_pairs:
            instructions += pair

        inputs = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True)
        inputs['labels'] = labels
        
        return inputs


class MaskDataCollator:
    def __init__(self, tokenizer, template) -> None:
        self.tokenizer = tokenizer
        self.template = template
        if template is not None:
            self.template = template.replace('*mask*', tokenizer.mask_token) \
                                    .replace('_', ' ').replace('*sep+*', '') \
                                    .replace('*cls*', '')

    def __call__(self, batch):
        if self.template is None:
            instruction_pairs = [[item['instruction1'], item['instruction2']] for item in batch]
        else:
            instruction_pairs = [[self.template.replace('*sent 0*', item['instruction1']), self.template.replace('*sent 0*', item['instruction2'])] for item in batch]
        
        labels = [item['label'] for item in batch]
        instructions = []
        for pair in instruction_pairs:
            instructions += pair

        inputs = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True)
        inputs['labels'] = labels
        
        return inputs