import json


def convert_txt_to_jsonl(txt_file_path, output_jsonl_path):
    count = 0
    fixed_instruction = (
        "你是一个智能助手，负责检查改写用户输入的查询，纠正其中的错别字、术语错误或语法错误，"
        "使其成为准确、通顺的查询。若原查询已正确，则直接输出原句。"
        "请仅输出改写后的查询，严禁输出任何解释或额外内容。"
    )

    with open(txt_file_path, 'r', encoding='utf-8') as f_in, \
            open(output_jsonl_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                print(f"跳过格式错误的行: {line}")
                continue

            input_text = parts[0]
            output_text = parts[1]

            data_obj = {
                "instruction": fixed_instruction,
                "input": input_text,
                "output": ""
            }
            json_str = json.dumps(data_obj, ensure_ascii=False)
            f_out.write(json_str + '\n')

            count += 1

    print(f"成功转换 {count} 条数据到 JSONL 格式: {output_jsonl_path}")


if __name__ == "__main__":
    input_txt = "../data/sample_test.txt"
    output_jsonl = "../data/inference.jsonl"

    convert_txt_to_jsonl(input_txt, output_jsonl)