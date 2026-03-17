import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import model_name, max_length, device, save_path, pred_path
from data import get_test_dataloader
from model import QspellModel

def inference(model_path, output_file, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = QspellModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = get_test_dataloader()

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            generated_ids = model.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=1,
                early_stopping=True,
                no_repeat_ngram_size=0
            )

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(pred_texts)

    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            f.write(pred + '\n')

    print(f"Predicting completed. Results saved to {output_file}")

if __name__ == "__main__":
    inference(model_path=save_path, output_file=pred_path, device=device)