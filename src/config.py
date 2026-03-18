import torch

train_file = '../data/qspell_250k_train.txt'
test_file = '../data/qspell_250k_test.txt'
test_file_sample = '../data/sample_test.txt'

batch_size = 16
max_length = 128
learning_rate = 1e-5
epoch_num = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42

# model_name = 'fnlp/bart-base-chinese'
model_name = '../bart'
save_path = '../model/qspell_bart_3.pth'
pred_path = '../output/qspell_bart_3.txt'
pred_path_sample = '../output/bart_3.txt'
csv_path = '../eval/qspell.csv'
csv_path_sample = '../eval/sample_eval.csv'