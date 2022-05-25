import numpy
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import BertModel


class PromptBERT(nn.Module):
    def __init__(self, pretrained_model_path):
        super().__init__()
        conf = AutoConfig.from_pretrained(pretrained_model_path)
        self.encoder = BertModel.from_pretrained(pretrained_model_path, config=conf)
        self.predict = nn.Linear(conf.hidden_size, 40524).cuda()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids, attention_mask, token_type_ids)
        # seq_relationship_score = self.cls(pooled_output)
        mask_index = (input_ids == 103)
        # input_mask_expanded = mask_index.long().unsqueeze(-1).expand(output[0].size()).float()
        # mask_embedding = torch.sum(output[0] * input_mask_expanded, 1)
        mask_embedding = output[0][mask_index]
        mask_predict = self.predict(mask_embedding)
        return mask_predict
