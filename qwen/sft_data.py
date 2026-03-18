import json

def convert_txt_to_json(txt_file_path, output_json_path):
    dataset = []
    fixed_instruction = ("你是一个智能助手，负责检查改写用户输入的查询，纠正其中的错别字、术语错误或语法错误，"
                         "使其成为准确、通顺的查询。若原查询已正确，则直接输出原句。"
                         "请仅输出改写后的查询，严禁输出任何解释或额外内容。")

    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            input_text = parts[0]
            output_text = parts[1]

            data_obj = {
                "instruction": fixed_instruction,
                "input": input_text,
                "output": output_text
            }
            dataset.append(data_obj)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"成功转换 {len(dataset)} 条数据到 {output_json_path}")


if __name__ == "__main__":
    input_txt = "../data/qspell_250k_test.txt"
    output_json = "../data/qspell.json"

    convert_txt_to_json(input_txt, output_json)
