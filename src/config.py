import torch

train_file = '../data/qspell_250k_train.txt'
test_file = '../data/qspell_250k_test.txt'
# test_file = '../data/sample_test.txt'

batch_size = 32
max_length = 128
learning_rate = 1e-5
epoch_num = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42

# model_name = 'fnlp/bart-base-chinese'
model_name = '../bart'
save_path = '../model/qspell_bart_base.pth'
pred_path = '../output/qspell_bart_base.txt'
csv_path = '../eval/qspell.csv'