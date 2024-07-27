import json
import random

from torch.utils.data import Dataset
import torch


def make_index(path: str):
    with open(path, 'r') as f:
        data = json.load(f)

    idx2data = {}
    for label in data.keys():
        samples = data[label]['samples']
        for sample in samples:
            idx2data[len(idx2data.keys())] = sample

    return idx2data


class IEBTrainDataset(Dataset):
    def __init__(self, path):
        super(IEBTrainDataset, self).__init__()
        self.data = self.load_file(path)

    def load_file(self, path):
        samples_loaded = []
        with open(path, 'r') as f:
            data = json.load(f)
        for label in data.keys():
            samples = data[label]['samples']
            for sample in samples:
                sample['label'] = label
                samples_loaded.append(sample)
        return samples_loaded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # def load_file(self, path, train_num, hard_neg=False):
    #     with open(path, 'r') as f:
    #         data = json.load(f)
    #     labels = list(data.keys())
    #     if hard_neg:
    #         vn_dict = {}
    #         nv_dict = {}
    #         vn2label = {}
    #         hard_negative_map = {}
    #         for label in labels:
    #             vn_pairs = label.strip('##').split('##')
    #             for vn in vn_pairs:
    #                 vn2label[vn] = label
    #                 verb, noun = vn.split("|")
    #                 if verb not in vn_dict.keys():
    #                     vn_dict[verb] = []
    #                 vn_dict[verb].append(noun)
    #                 if noun not in nv_dict.keys():
    #                     nv_dict[noun] = []
    #                 nv_dict[noun].append(verb)
    #         for vn in vn2label.keys():
    #             hard_negative_map[vn] = []
    #             verb, noun = vn.split("|")
    #             for hard_noun in vn_dict[verb]:
    #                 if vn2label[f"{verb}|{noun}"] != vn2label[f"{verb}|{hard_noun}"]:
    #                     hard_negative_map[vn].append(f"{verb}|{hard_noun}")
    #             for hard_verb in nv_dict[noun]:
    #                 if vn2label[f"{verb}|{noun}"] != vn2label[f"{hard_verb}|{noun}"]:
    #                     hard_negative_map[vn].append(f"{hard_verb}|{noun}")
    #     nums = [data[label]['number'] for label in labels]
    #     total = sum(nums)
    #     probs = [num/total for num in nums]
    #     sent_pairs = []
    #     while len(sent_pairs) < train_num:
    #         label = random.choices(labels, probs, k=1)[0]
    #         pair = random.choices(data[label]['samples'], k=2)
    #         if hard_neg:
    #             category = '|'.join(pair[0]['category'])
    #             if len(hard_negative_map[category]) > 0:
    #                 hard_category = random.choice(hard_negative_map[category])
    #                 hard_label = vn2label[hard_category]
    #                 hard_case_list = []
    #                 for sample in data[hard_label]['samples']:
    #                     if '|'.join(sample['category']) == hard_category:
    #                         hard_case_list.append(sample)
    #                 hard_case = random.choice(hard_case_list)
    #                 pair.append(hard_case)
    #                 pair.append(1)
    #             else:
    #                 category = '|'.join(pair[1]['category'])
    #                 if len(hard_negative_map[category]) > 0:
    #                     hard_category = random.choice(hard_negative_map[category])
    #                     hard_label = vn2label[hard_category]
    #                     hard_case_list = []
    #                     for sample in data[hard_label]['samples']:
    #                         if '|'.join(sample['category']) == hard_category:
    #                             hard_case_list.append(sample)
    #                     hard_case = random.choice(hard_case_list)
    #                     pair.reverse()
    #                     pair.append(hard_case)
    #                     pair.append(1)
    #                 else:
    #                     hard_label = label
    #                     while hard_label == label:
    #                         hard_label = random.choice(labels)
    #                     hard_case = random.choice(data[hard_label]['samples'])
    #                     pair.append(hard_case)
    #                     pair.append(0)
    #         sent_pairs.append(pair)
    #     if hard_neg:
    #         sent_pairs = [[item[0]['instruction'].strip(), item[1]['instruction'].strip(), item[2]['instruction'].strip(), item[3]] for item in sent_pairs]
    #     else:
    #         sent_pairs = [[item[0]['instruction'].strip(), item[1]['instruction'].strip()] for item in sent_pairs]
    #     if hard_neg:
    #         return data, sent_pairs, hard_negative_map
    #     else:
    #         return data, sent_pairs, None


class IEBTrainDataCollator:
    def __init__(self, tokenizer, template, path, hard_neg=False):
        self.tokenizer = tokenizer
        self.template = template
        self.hard_neg = hard_neg
        self.data, self.vn2label, self.hard_negative_map = self.load_file(path, hard_neg)

    def __call__(self, batch):
        all_labels = list(self.data.keys())
        new_batch = []
        hard_neg_flag = []
        sent_num = 3 if self.hard_neg else 2
        batch_size = len(batch)
        label_matrix = torch.zeros((batch_size, 2*batch_size if self.hard_neg else batch_size)).long()
        for i in range(batch_size):
            for j in range(batch_size):
                if batch[i]['label'] == batch[j]['label']:
                    label_matrix[i][j] = 1
        cur_idx = 0      
        for sample in batch:
            cur_label = sample['label']
            cur_instruction = sample['instruction']
            cur_category = sample['category']
            pos_instruction = cur_instruction
            while pos_instruction == cur_instruction:
                pos_case = random.choice(self.data[cur_label]['samples'])
                pos_instruction = pos_case['instruction']
                pos_category = pos_case['category']
                if len(self.data[cur_label]['samples']) == 1:
                    break
            if self.hard_neg:
                category = '|'.join(cur_category)
                if len(self.hard_negative_map[category]) > 0:
                    hard_category = random.choice(self.hard_negative_map[category])
                    hard_label = self.vn2label[hard_category]
                    hard_case_list = []
                    for sample in self.data[hard_label]['samples']:
                        if '|'.join(sample['category']) == hard_category:
                            hard_case_list.append(sample['instruction'])
                    hard_case = random.choice(hard_case_list)
                    new_batch.append([cur_instruction.strip(), pos_instruction.strip(), hard_case.strip()])
                    hard_neg_flag.append(1)
                    for i in range(batch_size):
                        if batch[i]['label'] == hard_label:
                            label_matrix[i][batch_size+cur_idx] = 1
                else:
                    category = '|'.join(pos_category)
                    if len(self.hard_negative_map[category]) > 0:
                        hard_category = random.choice(self.hard_negative_map[category])
                        hard_label = self.vn2label[hard_category]
                        hard_case_list = []
                        for sample in self.data[hard_label]['samples']['instruction']:
                            if '|'.join(sample['category']) == hard_category:
                                hard_case_list.append(sample['instruction'])
                        hard_case = random.choice(hard_case_list)
                        new_batch.append([pos_instruction.strip(), cur_instruction.strip(), hard_case.strip()])
                        hard_neg_flag.append(1)
                        for i in range(batch_size):
                            if batch[i]['label'] == hard_label:
                                label_matrix[i][batch_size+cur_idx] = 1
                    else:
                        hard_label = cur_label
                        while hard_label == cur_label:
                            hard_label = random.choice(all_labels)
                        hard_case = random.choice(self.data[hard_label]['samples'])['instruction']
                        new_batch.append([cur_instruction.strip(), pos_instruction.strip(), hard_case.strip()])
                        hard_neg_flag.append(0)
                        for i in range(batch_size):
                            if batch[i]['label'] == hard_label:
                                label_matrix[i][batch_size+cur_idx] = 1
                cur_idx += 1
            else:
                new_batch.append([cur_instruction.strip(), pos_instruction.strip()])

        for i in range(len(new_batch)):
            for j in range(sent_num):
                if len(new_batch[i][j]) > 0 and new_batch[i][j][-1] not in '.?"\'': 
                    new_batch[i][j] += '.'
                if self.template:
                    new_batch[i][j] = self.template.replace('*sent*', new_batch[i][j]).strip()
            if self.hard_neg:
                hard_neg_flag.append(new_batch[i][-1])
        new_batch = [item for sublist in new_batch for item in sublist]

        if self.tokenizer is not None:
            new_batch = self.tokenizer(
                new_batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            for key in new_batch:
                new_batch[key] = new_batch[key].reshape(batch_size, sent_num, -1)
            new_batch['hard_neg_flag'] = hard_neg_flag
            new_batch['label_matrix'] = label_matrix
        return new_batch
    
    def load_file(self, path, hard_neg=False):
        with open(path, 'r') as f:
            data = json.load(f)
        labels = list(data.keys())

        vn_dict = {}
        nv_dict = {}
        vn2label = {}
        hard_negative_map = {}
        for label in labels:
            vn_pairs = label.strip('##').split('##')
            for vn in vn_pairs:
                vn2label[vn] = label
                verb, noun = vn.split("|")
                if verb not in vn_dict.keys():
                    vn_dict[verb] = []
                vn_dict[verb].append(noun)
                if noun not in nv_dict.keys():
                    nv_dict[noun] = []
                nv_dict[noun].append(verb)
        for vn in vn2label.keys():
            hard_negative_map[vn] = []
            verb, noun = vn.split("|")
            for hard_noun in vn_dict[verb]:
                if vn2label[f"{verb}|{noun}"] != vn2label[f"{verb}|{hard_noun}"]:
                    hard_negative_map[vn].append(f"{verb}|{hard_noun}")
            for hard_verb in nv_dict[noun]:
                if vn2label[f"{verb}|{noun}"] != vn2label[f"{hard_verb}|{noun}"]:
                    hard_negative_map[vn].append(f"{hard_verb}|{noun}")

        return data, vn2label, hard_negative_map


class IEBTestDataset(Dataset):
    def __init__(self, path):
        super(IEBTestDataset, self).__init__()
        self.data = self.load_file(path)

    def load_file(self, path):
        idx2data = make_index(path)
        data = [(idx, instance['instruction']) for idx, instance in idx2data.items()]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class IEBTestDataCollator:
    def __init__(self, tokenizer=None, template=None):
        self.tokenizer = tokenizer
        self.template = template

    def __call__(self, batch):
        index = [idx for (idx, _) in batch]
        instructions = [instruction for (_, instruction) in batch]
        raw_instructions = instructions.copy()

        for i, s in enumerate(instructions):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            if self.template is not None:
                instructions[i] = self.template.replace('*sent*', s).strip()

        if self.tokenizer is not None:
            batch = self.tokenizer(
                instructions,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
        batch['instructions'] = raw_instructions
        batch['index'] = index

        return batch


class ListStyleDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = self.jload(path)
    
    def jload(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (index, self.data[index]['instruction'])


class ListStyleDataCollator:
    def __init__(self, tokenizer=None, template=None):
        self.tokenizer = tokenizer
        self.template = template

    def __call__(self, batch):
        index = [idx for (idx, _) in batch]
        instructions = [instruction for (_, instruction) in batch]
        raw_instructions = instructions.copy()

        for i, s in enumerate(instructions):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            if self.template is not None:
                instructions[i] = self.template.replace('*sent*', s).strip()

        if self.tokenizer is not None:
            batch = self.tokenizer(
                instructions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )
        batch['instructions'] = raw_instructions
        batch['index'] = index

        return batch
    