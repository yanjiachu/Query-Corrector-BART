from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging

from config import batch_size, max_length, model_name

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def load_data(file_path):
    src_texts = []
    tgt_texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                logging.warning(f"Skipping invalid line: {line}")
                continue
            src_texts.append(parts[0])
            tgt_texts.append(parts[1])
    return src_texts, tgt_texts

class QspellDataset(Dataset):
    def __init__(self, src_texts, tgt_texts):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.tgt_texts[idx]

def collate_fn(batch, tokenizer, max_length):
    src_texts, tgt_texts = zip(*batch)
    model_inputs = tokenizer(
        list(src_texts),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            list(tgt_texts),
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': labels
    }

def create_dataloader(file_path, tokenizer, batch_size, max_length, shuffle):
    src_texts, tgt_texts = load_data(file_path)
    dataset = QspellDataset(src_texts, tgt_texts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length)
    )
    return dataloader

def get_train_dataloader(file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return create_dataloader(file, tokenizer, batch_size, max_length, shuffle=True)

def get_test_dataloader(file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return create_dataloader(file, tokenizer, batch_size, max_length, shuffle=False)

if __name__ == "__main__":
    print("run data.py directly...")
    train_loader = get_train_dataloader()
    for batch in train_loader:
        print("input_ids shape:", batch['input_ids'].shape)
        print("attention_mask shape:", batch['attention_mask'].shape)
        print("labels shape:", batch['labels'].shape)
        break