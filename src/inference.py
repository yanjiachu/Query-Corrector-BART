import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import model_name, max_length, device, save_path, pred_path, test_file, test_file_sample
from data import get_test_dataloader
from model import QspellModel


def remove_chinese_internal_spaces(text):
    chars = list(text)
    i = 0
    while i < len(chars):
        if chars[i] == ' ':
            prev_is_letter = (i > 0) and chars[i - 1].isalpha() and chars[i - 1].isascii()
            next_is_letter = (i < len(chars) - 1) and chars[i + 1].isalpha() and chars[i + 1].isascii()
            if not (prev_is_letter and next_is_letter):
                del chars[i]
                continue
        i += 1
    return ''.join(chars)

def inference(model_path, test_file, output_file, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = QspellModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = get_test_dataloader(test_file)

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

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred_texts = [remove_chinese_internal_spaces(text) for text in pred_texts]
            all_predictions.extend(pred_texts)

    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            f.write(pred + '\n')

    print(f"Predicting completed. Results saved to {output_file}")

if __name__ == "__main__":
    # inference(model_path=save_path, output_file=pred_path, device=device)
    inference(model_path=save_path, test_file= test_file, output_file='../output/bart.txt', device=device)