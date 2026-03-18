import os
from openai import OpenAI

def rewrite_query(original_query):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能助手，负责检查改写用户输入的查询，纠正其中的错别字、术语错误或语法错误，"
                "使其成为准确、通顺的查询。若原查询已正确，则直接输出原句。"
                "示例："
                "输入：唐森照片"
                "输出：唐僧照片"
                "请仅输出改写后的查询，严禁输出任何解释或额外内容。"
            )
        },
        {
            "role": "user",
            "content": original_query
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="qwen3-max",
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=True
        )

        rewritten = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                rewritten += chunk.choices[0].delta.content
        return rewritten.strip()
    except Exception as e:
        print(f"调用 API 时出错：{e}")
        return original_query

def main():
    input_file = "../data/sample_test.txt"
    output_file = "../output/qwen3-max.txt"

    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，请检查路径。")
        return

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.rstrip("\n")
            if not line:
                f_out.write("\n")
                continue

            parts = line.split("\t")
            original = parts[0]

            if line_num % 10 == 0:
                print(f"已处理 {line_num} 行")

            rewritten = rewrite_query(original)
            f_out.write(rewritten + "\n")


    print(f"处理完成，已保存至 {output_file}")

if __name__ == "__main__":
    main()