import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
from tqdm import tqdm

from config import  model_name, save_path, epoch_num, learning_rate, device, train_file
from data import get_train_dataloader
from model import QspellModel


def train(epochs=epoch_num, lr=learning_rate, device=device, save_path=save_path, log_interval=100):
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading training data...")
    train_loader = get_train_dataloader(train_file)

    logging.info(f"Loading model {model_name}...")
    model = QspellModel().to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

    model.train()
    global_step = 0
    total_loss = 0

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                progress_bar.set_postfix({'loss': avg_loss})
                total_loss = 0

    torch.save(model.state_dict(), save_path)
    logging.info(f"Training completed. Model saved to {save_path}")

    return model