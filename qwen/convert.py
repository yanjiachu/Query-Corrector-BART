import json

def convert(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            content = data.get('predict', '')
            f_out.write(str(content) + '\n')



if __name__ == '__main__':
    input_file = '../output/generated_predictions.jsonl'
    output_file = '../output/qwen2.5-1.5B-sft.txt'

    convert(input_file, output_file)