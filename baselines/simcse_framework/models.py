import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers

import fire
import accelerate

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaPreTrainedModel, BertPreTrainedModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


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


class BertForCL(BertPreTrainedModel):
    def __init__(self, config: AutoConfig, model_path: str):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
        ).cuda()
        self.model.config.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.train()
        self.model.config.use_cache = False
        self.sim = Similarity(temp=0.05)

    def forward(self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    ):
        batch_size = input_ids.size(0)

        # Get raw embeddings
        outputs = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs['hidden_states']
        embeddings_last = hidden_states[-1]
        embeddings_second_last = hidden_states[-2]
        embeddings = (embeddings_last[:, -1, :] + embeddings_second_last[:, -1, :]) / 2.
        embeddings = embeddings.reshape(batch_size//2, 2, -1)

        # Separate representation
        z1, z2 = embeddings[:,0], embeddings[:,1]

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss = F.cross_entropy(cos_sim, labels)

        return loss

    def save_checkpoint(self, output_dir):
        self.model.save_pretrained(output_dir)


class LlamaForCL(LlamaPreTrainedModel):
    def __init__(self, config: AutoConfig, model_path: str, lora=False, load_in_8bit=False):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            # torch_dtype=torch.float16,
            device_map="auto",
            config=config
        )
        self.model.config.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        if lora:
            config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=['q_proj','v_proj'],
                lora_dropout=0.05,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

        self.model.train()
        self.model.config.use_cache = False
        self.sim = Similarity(temp=0.05)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    ):
        batch_size = input_ids.size(0)

        # Get raw embeddings
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs['hidden_states']
        embeddings_last = hidden_states[-1]
        embeddings = embeddings_last[:, 0, :]
        # embeddings_second_last = hidden_states[-2]
        # embeddings = (embeddings_last[:, -1, :] + embeddings_second_last[:, -1, :]) / 2.
        embeddings = embeddings.reshape(batch_size//2, 2, -1)

        # Separate representation
        z1, z2 = embeddings[:,0], embeddings[:,1]

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss = F.cross_entropy(cos_sim, labels)

        return loss

    def save_checkpoint(self, output_dir):
        self.model.save_pretrained(output_dir)
