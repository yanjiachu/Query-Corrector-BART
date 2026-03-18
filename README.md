# Query-Corrector-BART

## 项目介绍

在日常使用搜索引擎、电商平台或各类内部检索系统时，用户输入的Query（查询词）经常会出现错别字、形近字混淆或拼音输入错误等情况。这些错误会直接导致检索结果匹配度下降，甚至返回无关内容，严重影响用户体验。

本项目基于 BART 预训练模型（`fnlp/bart-base-chinese`）进行微调，实现 Query 错别字自动纠正。我们使用 **Qspell** 数据集（25 万规模）进行训练和测试，以**字符错误率（CER）**、**字符级精确率（Precision）**、**召回率（Recall）** 和 **F1 分数** 作为核心评估指标。同时，我们将微调后的模型与通用大模型（如 GPT 系列、文心一言等）在相同测试集上进行对比实验，验证专用微调模型在 Query 纠错任务上的优越性。

## 目录

- [数据准备](#数据准备)
- [快速开始](#快速开始)
  - [训练](#训练)
  - [推理](#推理)
  - [评估](#评估)
- [仓库结构](#仓库结构)
- [实验结果](#实验结果)
- [参考](#参考)


## 数据准备

数据集为 **Qspell** 格式的 TSV 文件，每行包含 `原始查询\t改写后查询`。  
示例：
```
ai文字怎么左右对齐	ai文字怎么左右对齐
马来西亚流血签证多少钱	马来西亚留学签证多少钱
苏波尔电火锅可以换锅吗	苏泊尔电火锅可以换锅吗
中国地震局 郑凯	中国地震局 郑凯
天寒孟泽	天寒梦泽
```

将训练集和测试集置于 `data/` 目录下：
- `data/qspell_250k_train.txt`
- `data/qspell_250k_test.txt`

## 快速开始

所有核心代码位于 `src/` 目录。请先进入该目录：
```bash
cd src
```

### 训练

训练参数在 `config.py` 中配置，包括模型名称、数据路径、批次大小、学习率等。默认配置已适配 Qspell 数据集。

启动训练（可覆盖默认超参数）：
```bash
python main.py --epochs 3 --lr 2e-5
```

训练过程中，模型检查点会保存在 `model/` 目录（默认 `best_model.pt`），日志保存在 `logs/`。

### 推理

使用训练好的模型对测试集进行改写，结果逐行写入输出文件：
```bash
python main.py --skip_train --model_save_path ../model/best_model.pt --output_file ../output/predictions.txt
```
或直接运行 `inference.py`：
```bash
python inference.py --model_path ../model/best_model.pt --output_file ../output/predictions.txt
```

推理时会自动去除中文间的多余空格（保留英文单词间空格），确保输出格式干净。

### 评估

评估脚本 `eval.py` 计算 CER、精确率、召回率和 F1 分数。  
运行：
```bash
python eval.py --pred_file ../output/predictions.txt --ref_file ../data/qspell_250k_test.txt
```
评估结果会保存至 `eval/` 目录下的 CSV 文件（例如 `qspell.csv`）。

## 仓库结构

```
Query-Corrector-BART/
├── data/
│   ├── qspell_250k_train.txt  # 训练集
│   ├── qspell_250k_test.txt   # 测试集
│   └── sample_test.txt        # 示例测试数据
├── src/
│   ├── config.py              # 配置文件
│   ├── data.py                # 数据加载器
│   ├── model.py               # 模型封装
│   ├── train.py               # 训练逻辑
│   ├── inference.py           # 推理逻辑
│   ├── eval.py                # 评估指标
│   └── main.py                # 主流程入口
├── eval/
│   ├── qspell.csv             # 测试集评估结果
│   └── sample_eval.csv        # 示例评估结果
├── logs/                      # 训练日志
├── model/                     # 保存的模型权重
├── output/                    # 推理输出文件
└── qwen/
    ├── api.py                 # 调用Qwen API
    └── sample_data.py         # 示例采样代码
```

## 实验结果

### 微调模型效果
在 Qspell 测试集上，使用默认配置训练后得到以下指标：


| 模型              | CER      | Precision | Recall   | F1       |
|-----------------|----------|-----------|----------|----------|
| BART-FT-epoch_1 | 0.070367 | 0.932762  | 0.929633 | 0.931269 |
| BART-FT-epoch_3 | 0.063073 | 0.939686  | 0.936927 | 0.938391 |

详细结果见 `eval/qspell.csv`。

### 与通用大模型对比
由于成本限制，我们选取了多个通用大模型在相同小样本示例测试集上进行对比（零样本提示）。对比结果如下：

| 模型                  | CER             | Precision       | Recall          | F1              |
|---------------------|-----------------|-----------------|-----------------|-----------------|
| Qwen2.5-7b-instruct | 0.293166        | 0.742665        | 0.706834        | 0.760337        |
| Qwen3-8b            | 0.170618        | 0.849017        | 0.829382        | 0.850478        |
| Qwen3-max           | 0.179424        | 0.816344        | 0.820576        | 0.825507        |
| Qwen3.5-plus        | 0.166046        | 0.839108        | 0.833954        | 0.834763        |
| BART-FT-epoch_1     | <u>0.069800</u> | <u>0.932849</u> | <u>0.930200</u> | <u>0.931162</u> |
| BART-FT-epoch_3     | **0.065056**    | **0.937860**    | **0.934944**    | **0.936057**    |

详细结果见 `eval/sample_eval.csv`。

实验表明，针对 Query 纠错任务进行微调的 BART 模型在各项指标上均显著优于通用大模型，验证了专用微调的必要性和有效性。

## 参考

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- BART中文预训练模型 [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese)
- [QSpell 250K: A Large-Scale, Practical Dataset for Chinese Search Query Spell Correction](https://aclanthology.org/2025.naacl-industry.13/)
---

欢迎使用本项目，如有问题请提交 Issue。