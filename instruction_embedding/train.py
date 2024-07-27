import os

import torch
import torch.nn as nn
from tqdm import tqdm
from utils.ieb_parser import *


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    

class EmbeddingTrainer():
    def __init__(self, train_loader, eval_loader, model, optimizer, save_path, epochs=3, temperature=0.05, device="cuda", pooling='last-2', template=None, hard_negative_weight=0.5):
        super().__init__()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.epochs = epochs
        self.temperature = temperature
        self.device = device
        self.pooling = pooling
        self.template = template
        self.hard_negative_weight = hard_negative_weight

        self.sim = Similarity(temp=0.05)
        self.embedding_dict = {}

    def train_epoch(self):
        loss_list = []
        train_loader = tqdm(self.train_loader, desc="Iteration")
        for step, inputs in enumerate(train_loader):
            hard_neg_flag = inputs['hard_neg_flag']
            label_matrix = inputs['label_matrix']
            del inputs['hard_neg_flag']
            del inputs['label_matrix']
            batch_size = inputs['input_ids'].shape[0]
            num_sent = inputs['input_ids'].shape[1]
            for key in inputs.keys():
                inputs[key] = inputs[key].reshape(batch_size*num_sent, -1).to(self.device)
            label_matrix = label_matrix.to(self.device)
            
            hidden_states = self.model(**inputs, output_hidden_states=True, return_dict=True)['hidden_states']
            embeddings_last = hidden_states[-1]

            # last 2 layers
            if self.pooling == 'last-2':
                embeddings_second_last = hidden_states[-2]
                embeddings = (embeddings_last[:, -1, :] + embeddings_second_last[:, -1, :]) / 2.
            # last 1 layer
            elif self.pooling == 'last-1':
                embeddings = embeddings_last[:, -1, :]
            # first and last layer
            elif self.pooling == 'last-and-first':
                embedings_first = hidden_states[0]
                embeddings = (embeddings_last[:, -1, :] + embedings_first[:, -1, :]) / 2.
            # mid layer
            elif self.pooling == 'mid':
                embeddings_mid = hidden_states[len(hidden_states) // 2]
                embeddings = embeddings_mid[:, -1, :]
            elif self.pooling == 'cls':
                embeddings = embeddings_last[:, 0, :]
            else:
                raise NotImplementedError

            # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)
            embeddings = embeddings.reshape(batch_size, num_sent, -1)
            # Separate representation
            z1, z2 = embeddings[:,0,:], embeddings[:,1,:]
            if num_sent == 3:
                z3 = embeddings[:,2,:]

            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            if num_sent == 3:
                z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            # if num_sent == 3:
            #     # Note that weights are actually logits of weights
            #     z3_weight = self.hard_negative_weight
                # weights = torch.tensor(
                #     [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight if hard_neg_flag[i] == 0 else 0.] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
                # ).to(cos_sim.device)

                # cos_sim = cos_sim + weights

            loss = torch.nn.functional.cross_entropy(cos_sim, labels)
            # softmax_cos_sim = torch.nn.functional.softmax(cos_sim, dim=1)
            # loss = torch.nn.functional.binary_cross_entropy(softmax_cos_sim, label_matrix.float())

            assert not torch.isnan(loss).any().item()
            loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

            if (step + 1) % 1000 == 0:
                print(f"Step [{step+1}], Loss: {sum(loss_list) / len(loss_list)}")
                loss_list.clear()

        return sum(loss_list) / len(loss_list)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            loss = self.train_epoch()
            os.makedirs("../checkpoints", exist_ok=True)
            if epoch < self.epochs-1:
                self.model.save_pretrained(f'{self.save_path.strip()}_epoch{epoch+1}')
            else:
                self.model.save_pretrained(f'{self.save_path.strip()}')
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss:.4f}")

        #     self.model.eval()
        #     eval_loader = tqdm(self.eval_loader, desc="Iteration")
        #     for _, batch in enumerate(eval_loader):
        #         instructions = batch['instructions'].copy()
        #         index = batch['index'].copy()

        #         del batch['instructions']
        #         del batch['index']

        #         inputs = {key: value.to(self.device) for key, value in batch.items()}
        #         with torch.autocast(self.device):
        #             # Get the embeddings
        #             with torch.no_grad():
        #                 hidden_states = self.model.forward(**inputs, output_hidden_states=True, return_dict=True)['hidden_states']
        #                 embeddings_last = hidden_states[-1]
        #                 # last 2 layers
        #                 if self.pooling == 'last-2':
        #                     embeddings_second_last = hidden_states[-2]
        #                     embeddings = (embeddings_last[:, -1, :] + embeddings_second_last[:, -1, :]) / 2.
        #                 # last 1 layer
        #                 elif self.pooling == 'last-1':
        #                     embeddings = embeddings_last[:, -1, :]
        #                 # first and last layer
        #                 elif self.pooling == 'last-and-first':
        #                     embedings_first = hidden_states[0]
        #                     embeddings = (embeddings_last[:, -1, :] + embedings_first[:, -1, :]) / 2.
        #                 # mid layer
        #                 elif self.pooling == 'mid':
        #                     embeddings_mid = hidden_states[len(hidden_states) // 2]
        #                     embeddings = embeddings_mid[:, -1, :]
        #                 else:
        #                     raise NotImplementedError

        #                 for i in range(len(instructions)):
        #                     self.embedding_dict[embeddings[i].cpu()] = index[i]

        #     os.makedirs("../saved_instruction_embeddings", exist_ok=True)
        #     torch.save(self.embedding_dict, f"{self.save_path.strip()}_epoch{epoch+1}.pth")
        # torch.save(self.embedding_dict, f"{self.save_path.strip()}.pth")
