import torch.nn as nn
from transformers import BartForConditionalGeneration

from config import model_name

class QspellModel(nn.Module):
    def __init__(self, pretrained_model_name=model_name):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def save_pretrained(self, save_directory):
        self.bart.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        return cls(pretrained_model_name_or_path)

def get_model(pretrained_model_name=model_name):
    return QspellModel(pretrained_model_name)