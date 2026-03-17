import random
import os


def sample_lines(input_file, output_file, sample_size=1000, seed=42):
    if not os.path.exists(input_file):
        print(f"找不到文件 '{input_file}'")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    if sample_size > total_lines:
        print(f"请求抽样 {sample_size} 行，文件仅 {total_lines} 行。将抽取全部数据。")
        sample_size = total_lines

    if sample_size == 0:
        print("文件为空或抽样数量为0")
        return

    random.seed(seed)
    sampled_lines = random.sample(all_lines, sample_size)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    print(f"抽样结果已保存至: {output_file}")


if __name__ == "__main__":
    INPUT_FILE = "../data/qspell_250k_test.txt"
    OUTPUT_FILE = "../data/sample_test.txt"
    SAMPLE_COUNT = 1000

    sample_lines(INPUT_FILE, OUTPUT_FILE, SAMPLE_COUNT)