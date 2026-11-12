import json
import os
from collections import Counter

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "results",
        "prompt_sensitivity",
    )
)

FILES = {
    "P1": "qwen27b_prompt_P1_100.json",
    "P2": "qwen27b_prompt_P2_100.json",
    "P3": "qwen27b_prompt_P3_100.json",
}

LABELS = ["B", "C", "G", "Y"]


def load_results(filename):
    path = os.path.join(BASE_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    if isinstance(results, dict):
        results = list(results.values())

    results = sorted(results, key=lambda x: x["index"])

    return results


def migration_matrix(pred_a, pred_b):
    matrix = {
        a: {b: 0 for b in LABELS}
        for a in LABELS
    }

    for a, b in zip(pred_a, pred_b):
        matrix[a][b] += 1

    return matrix


def print_matrix(title, matrix):
    print(f"\n【{title}】")
    print("-" * 65)

    print(
        f"{'From / To':<12}"
        f"{'B':<10}"
        f"{'C':<10}"
        f"{'G':<10}"
        f"{'Y':<10}"
    )

    for label in LABELS:
        print(
            f"{label:<12}"
            f"{matrix[label]['B']:<10}"
            f"{matrix[label]['C']:<10}"
            f"{matrix[label]['G']:<10}"
            f"{matrix[label]['Y']:<10}"
        )


def correctness_transition(results_a, results_b):
    """
    分析 A -> B：
    correct -> correct
    correct -> wrong
    wrong -> correct
    wrong -> wrong
    """

    counter = Counter()

    details = {
        "correct_to_correct": [],
        "correct_to_wrong": [],
        "wrong_to_correct": [],
        "wrong_to_wrong": [],
    }

    for a, b in zip(results_a, results_b):

        gold = a["gold"]
        pred_a = a["prediction"]
        pred_b = b["prediction"]

        correct_a = pred_a == gold
        correct_b = pred_b == gold

        if correct_a and correct_b:
            key = "correct_to_correct"
        elif correct_a and not correct_b:
            key = "correct_to_wrong"
        elif not correct_a and correct_b:
            key = "wrong_to_correct"
        else:
            key = "wrong_to_wrong"

        counter[key] += 1

        details[key].append({
            "index": a["index"],
            "gold": gold,
            "prediction_a": pred_a,
            "prediction_b": pred_b,
        })

    return counter, details


def main():

    print("=" * 90)
    print("Prompt Sensitivity —— 预测迁移与正确性迁移分析")
    print("模型：Qwen3.8-27B")
    print("样本：固定100条")
    print("=" * 90)

    experiments = {
        prompt: load_results(filename)
        for prompt, filename in FILES.items()
    }

    # ---------------------------------------------------------
    # 1. 检查样本是否完全一致
    # ---------------------------------------------------------

    print("\n【一、样本一致性检查】")
    print("-" * 70)

    base_indices = [
        x["index"]
        for x in experiments["P1"]
    ]

    for prompt, results in experiments.items():

        indices = [
            x["index"]
            for x in results
        ]

        print(
            f"{prompt}: "
            f"{len(indices)} 条，"
            f"与 P1 样本索引一致 = {indices == base_indices}"
        )

    # ---------------------------------------------------------
    # 2. 两两类别迁移
    # ---------------------------------------------------------

    pair_results = {}

    pairs = [
        ("P1", "P2"),
        ("P1", "P3"),
        ("P2", "P3"),
    ]

    for prompt_a, prompt_b in pairs:

        results_a = experiments[prompt_a]
        results_b = experiments[prompt_b]

        pred_a = [
            x["prediction"]
            for x in results_a
        ]

        pred_b = [
            x["prediction"]
            for x in results_b
        ]

        matrix = migration_matrix(
            pred_a,
            pred_b,
        )

        key = f"{prompt_a}_to_{prompt_b}"

        pair_results[key] = {
            "migration_matrix": matrix,
        }

        print_matrix(
            f"{prompt_a} → {prompt_b} 类别迁移",
            matrix,
        )

        # -----------------------------------------------------
        # 迁移总量
        # -----------------------------------------------------

        changes = sum(
            pred_a[i] != pred_b[i]
            for i in range(len(pred_a))
        )

        same = len(pred_a) - changes

        print(
            f"\n保持相同预测：{same}/100"
        )
        print(
            f"发生预测变化：{changes}/100"
        )

    # ---------------------------------------------------------
    # 3. 正确性迁移
    # ---------------------------------------------------------

    print("\n【二、正确性迁移】")
    print("-" * 80)

    correctness_results = {}

    for prompt_a, prompt_b in pairs:

        counter, details = correctness_transition(
            experiments[prompt_a],
            experiments[prompt_b],
        )

        key = f"{prompt_a}_to_{prompt_b}"

        correctness_results[key] = {
            "counts": dict(counter),
            "details": details,
        }

        print(f"\n--- {prompt_a} → {prompt_b} ---")

        print(
            f"正确 → 正确：{counter['correct_to_correct']}"
        )

        print(
            f"正确 → 错误：{counter['correct_to_wrong']}"
        )

        print(
            f"错误 → 正确：{counter['wrong_to_correct']}"
        )

        print(
            f"错误 → 错误：{counter['wrong_to_wrong']}"
        )

        net_change = (
            counter["wrong_to_correct"]
            - counter["correct_to_wrong"]
        )

        print(
            f"净正确数变化：{net_change:+d}"
        )

    # ---------------------------------------------------------
    # 4. 找出最重要的类别迁移
    # ---------------------------------------------------------

    print("\n【三、主要类别迁移】")
    print("-" * 80)

    major_transitions = {}

    for prompt_a, prompt_b in pairs:

        results_a = experiments[prompt_a]
        results_b = experiments[prompt_b]

        counter = Counter()

        for a, b in zip(results_a, results_b):

            pred_a = a["prediction"]
            pred_b = b["prediction"]

            if pred_a != pred_b:
                counter[f"{pred_a}→{pred_b}"] += 1

        key = f"{prompt_a}_to_{prompt_b}"

        major_transitions[key] = dict(
            counter.most_common()
        )

        print(f"\n--- {prompt_a} → {prompt_b} ---")

        for transition, count in counter.most_common():

            print(
                f"{transition:<10} {count:>3} 次"
            )

    # ---------------------------------------------------------
    # 5. 保存详细结果
    # ---------------------------------------------------------

    output = {
        "experiment": "prompt_prediction_migration",
        "model": "qwen3.8-27b",
        "sample_size": 100,
        "label_order": LABELS,
        "pairs": pair_results,
        "correctness_transitions": correctness_results,
        "major_transitions": major_transitions,
    }

    output_path = os.path.join(
        BASE_DIR,
        "qwen27b_prompt_prediction_migration.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("分析完成")
    print(
        "结果已保存："
        "02_llm_inference\\results\\prompt_sensitivity\\qwen27b_prompt_prediction_migration.json"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()