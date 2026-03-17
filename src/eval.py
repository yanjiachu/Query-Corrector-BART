import csv
import os


def levenshtein_distance(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

    return dp[len1][len2]


def calculate_metrics(pred, ref):
    edit_dist = levenshtein_distance(ref, pred)
    ref_len = len(ref)

    if ref_len == 0:
        return {
            'cer': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    cer = edit_dist / ref_len

    correct_chars = ref_len - edit_dist
    precision = correct_chars / len(pred) if len(pred) > 0 else 0.0
    recall = correct_chars / ref_len
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        'cer': cer,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def eval(test_file, pred_file, csv_file):
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                test_data.append((parts[0], parts[1]))

    predictions = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(line.strip())

    n_samples = min(len(test_data), len(predictions))
    test_data = test_data[:n_samples]
    predictions = predictions[:n_samples]

    total_cer = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for (original, correct), pred in zip(test_data, predictions):
        metrics = calculate_metrics(pred, correct)
        total_cer += metrics['cer']
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1']

    avg_cer = total_cer / n_samples
    avg_precision = total_precision / n_samples
    avg_recall = total_recall / n_samples
    avg_f1 = total_f1 / n_samples

    dataset_name = os.path.splitext(os.path.basename(pred_file))[0]
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'dataset_name',
                'avg_cer',
                'avg_precision',
                'avg_recall',
                'avg_f1'
            ])
        writer.writerow([
            dataset_name,
            f"{avg_cer:.6f}",
            f"{avg_precision:.6f}",
            f"{avg_recall:.6f}",
            f"{avg_f1:.6f}"
        ])

    print(f"CER: {avg_cer:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")


if __name__ == '__main__':
    test_file = '../data/sample_test.txt'
    pred_file = '../output/qwen3.5-plus.txt'
    csv_file = '../eval/sample_eval.csv'
    eval(test_file, pred_file, csv_file)