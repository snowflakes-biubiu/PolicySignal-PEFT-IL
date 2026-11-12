import json
import os

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

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


def load_result(prompt_name, filename):
    path = os.path.join(BASE_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    # 兼容 results 为 list 或 dict 两种结构
    if isinstance(results, dict):
        results = list(results.values())

    results = sorted(results, key=lambda x: x["index"])

    y_true = [x["gold"] for x in results]
    y_pred = [x["prediction"] for x in results]

    return {
        "prompt": prompt_name,
        "data": data,
        "results": results,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():

    print("=" * 90)
    print("Prompt Sensitivity 正式统计分析")
    print("模型：Qwen3.8-27B")
    print("样本：固定100条，B/C/G/Y各25条")
    print("=" * 90)

    experiments = {}

    for prompt_name, filename in FILES.items():
        experiments[prompt_name] = load_result(prompt_name, filename)

    # ---------------------------------------------------------
    # 1. 总体指标
    # ---------------------------------------------------------

    print("\n【一、总体指标】")
    print("-" * 90)

    print(
        f"{'Prompt':<10}"
        f"{'Accuracy':<15}"
        f"{'Macro-F1':<15}"
        f"{'Kappa':<15}"
    )

    print("-" * 90)

    summary = {}

    for prompt_name, exp in experiments.items():

        y_true = exp["y_true"]
        y_pred = exp["y_pred"]

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        kappa = cohen_kappa_score(y_true, y_pred, labels=LABELS)

        summary[prompt_name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "kappa": kappa,
        }

        print(
            f"{prompt_name:<10}"
            f"{acc:<15.4f}"
            f"{macro_f1:<15.4f}"
            f"{kappa:<15.4f}"
        )

    # ---------------------------------------------------------
    # 2. 每个类别的 Precision / Recall / F1
    # ---------------------------------------------------------

    print("\n【二、各类别 Precision / Recall / F1】")
    print("-" * 110)

    print(
        f"{'Prompt':<10}"
        f"{'Class':<10}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1':<15}"
    )

    print("-" * 110)

    per_class = {}

    for prompt_name, exp in experiments.items():

        y_true = exp["y_true"]
        y_pred = exp["y_pred"]

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )

        per_class[prompt_name] = {}

        for label in LABELS:

            precision = report[label]["precision"]
            recall = report[label]["recall"]
            f1 = report[label]["f1-score"]

            per_class[prompt_name][label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            print(
                f"{prompt_name:<10}"
                f"{label:<10}"
                f"{precision:<15.4f}"
                f"{recall:<15.4f}"
                f"{f1:<15.4f}"
            )

    # ---------------------------------------------------------
    # 3. 混淆矩阵
    # ---------------------------------------------------------

    print("\n【三、混淆矩阵】")
    print("行 = Gold，列 = Prediction")
    print("标签顺序：B C G Y")

    confusion_matrices = {}

    for prompt_name, exp in experiments.items():

        cm = confusion_matrix(
            exp["y_true"],
            exp["y_pred"],
            labels=LABELS,
        )

        confusion_matrices[prompt_name] = cm.tolist()

        print(f"\n--- {prompt_name} ---")

        print(
            f"{'Gold\\\\Pred':<12}"
            f"{'B':<8}"
            f"{'C':<8}"
            f"{'G':<8}"
            f"{'Y':<8}"
        )

        for i, label in enumerate(LABELS):

            print(
                f"{label:<12}"
                f"{cm[i, 0]:<8}"
                f"{cm[i, 1]:<8}"
                f"{cm[i, 2]:<8}"
                f"{cm[i, 3]:<8}"
            )

    # ---------------------------------------------------------
    # 4. 预测类别分布
    # ---------------------------------------------------------

    print("\n【四、预测类别分布】")
    print("-" * 80)

    print(
        f"{'Prompt':<10}"
        f"{'B':<10}"
        f"{'C':<10}"
        f"{'G':<10}"
        f"{'Y':<10}"
    )

    prediction_distribution = {}

    for prompt_name, exp in experiments.items():

        counts = {
            label: exp["y_pred"].count(label)
            for label in LABELS
        }

        prediction_distribution[prompt_name] = counts

        print(
            f"{prompt_name:<10}"
            f"{counts['B']:<10}"
            f"{counts['C']:<10}"
            f"{counts['G']:<10}"
            f"{counts['Y']:<10}"
        )

    # ---------------------------------------------------------
    # 5. 两两预测一致率
    # ---------------------------------------------------------

    print("\n【五、Prompt 两两预测一致率】")
    print("-" * 80)

    pairwise_agreement = {}

    prompt_names = list(experiments.keys())

    for i in range(len(prompt_names)):
        for j in range(i + 1, len(prompt_names)):

            p1 = prompt_names[i]
            p2 = prompt_names[j]

            pred1 = experiments[p1]["y_pred"]
            pred2 = experiments[p2]["y_pred"]

            same = sum(
                a == b
                for a, b in zip(pred1, pred2)
            )

            agreement = same / len(pred1)

            key = f"{p1}_vs_{p2}"
            pairwise_agreement[key] = agreement

            print(
                f"{p1} vs {p2}: "
                f"{same}/100 = {agreement:.4f}"
            )

    # ---------------------------------------------------------
    # 6. 两两预测变化数量
    # ---------------------------------------------------------

    print("\n【六、Prompt 两两预测变化】")
    print("-" * 80)

    prediction_changes = {}

    for i in range(len(prompt_names)):
        for j in range(i + 1, len(prompt_names)):

            p1 = prompt_names[i]
            p2 = prompt_names[j]

            pred1 = experiments[p1]["y_pred"]
            pred2 = experiments[p2]["y_pred"]

            changes = sum(
                a != b
                for a, b in zip(pred1, pred2)
            )

            key = f"{p1}_to_{p2}"
            prediction_changes[key] = changes

            print(
                f"{p1} -> {p2}: "
                f"{changes}/100 条发生预测变化"
            )

    # ---------------------------------------------------------
    # 7. 每个 Prompt 的正确样本数
    # ---------------------------------------------------------

    print("\n【七、正确性重叠情况】")
    print("-" * 80)

    correctness = {}

    for prompt_name, exp in experiments.items():

        correct_flags = [
            gold == pred
            for gold, pred in zip(
                exp["y_true"],
                exp["y_pred"],
            )
        ]

        correctness[prompt_name] = correct_flags

        print(
            f"{prompt_name}: "
            f"{sum(correct_flags)}/100 正确"
        )

    all_three_correct = sum(
        a and b and c
        for a, b, c in zip(
            correctness["P1"],
            correctness["P2"],
            correctness["P3"],
        )
    )

    none_correct = sum(
        not a and not b and not c
        for a, b, c in zip(
            correctness["P1"],
            correctness["P2"],
            correctness["P3"],
        )
    )

    at_least_one_correct = sum(
        a or b or c
        for a, b, c in zip(
            correctness["P1"],
            correctness["P2"],
            correctness["P3"],
        )
    )

    print(f"\nP1/P2/P3 全部正确：{all_three_correct}/100")
    print(f"P1/P2/P3 全部错误：{none_correct}/100")
    print(f"至少一个 Prompt 正确：{at_least_one_correct}/100")

    # ---------------------------------------------------------
    # 8. 保存完整分析结果
    # ---------------------------------------------------------

    output = {
        "experiment": "prompt_sensitivity_analysis",
        "model": "qwen3.8-27b",
        "sample_size": 100,
        "label_order": LABELS,
        "summary": summary,
        "per_class": per_class,
        "confusion_matrices": confusion_matrices,
        "prediction_distribution": prediction_distribution,
        "pairwise_agreement": pairwise_agreement,
        "prediction_changes": prediction_changes,
        "all_three_correct": all_three_correct,
        "none_correct": none_correct,
        "at_least_one_correct": at_least_one_correct,
    }

    output_path = os.path.join(
        BASE_DIR,
        "qwen27b_prompt_sensitivity_analysis.json",
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
    print(f"结果已保存：{output_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()