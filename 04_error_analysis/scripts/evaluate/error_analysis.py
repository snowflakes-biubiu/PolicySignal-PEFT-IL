import os
import json
import numpy as np
import pandas as pd


# ============================================================
# 基础配置
# ============================================================

OUTPUT_DIR = "04_error_analysis/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 已完成实验文件
# ============================================================

EXPERIMENTS = {
    # -----------------------------
    # Level-1
    # -----------------------------
    "DeepSeek_Level1_ZS": {
        "file": "02_llm_inference/results/deepseek/deepseek_level1_zero_shot_300.json",
        "labels": ["N", "R", "W"],
    },

    "DeepSeek_Level1_k3": {
        "file": "02_llm_inference/results/deepseek/deepseek_level1_fewshot3_300.json",
        "labels": ["N", "R", "W"],
    },

    "DeepSeek_Level1_k6": {
        "file": "02_llm_inference/results/deepseek/deepseek_level1_fewshot6_300.json",
        "labels": ["N", "R", "W"],
    },

    "QwenMax_Level1_ZS": {
        "file": "02_llm_inference/results/qwen/qwen_level1_zero_shot_300_fixed.json",
        "labels": ["N", "R", "W"],
    },

    "QwenMax_Level1_k3": {
        "file": "02_llm_inference/results/qwen/qwen_level1_fewshot3_300.json",
        "labels": ["N", "R", "W"],
    },

    "QwenMax_Level1_k6": {
        "file": "02_llm_inference/results/qwen/qwen_level1_fewshot6_300_fixed.json",
        "labels": ["N", "R", "W"],
    },

    # -----------------------------
    # Level-2
    # -----------------------------
    "DeepSeek_Level2_ZS": {
        "file": "02_llm_inference/results/deepseek/deepseek_level2_zero_shot_400.json",
        "labels": ["B", "C", "G", "Y"],
    },

    "DeepSeek_Level2_k3": {
        "file": "02_llm_inference/results/deepseek/deepseek_level2_fewshot3_400.json",
        "labels": ["B", "C", "G", "Y"],
    },

    "DeepSeek_Level2_k6": {
        "file": "02_llm_inference/results/deepseek/deepseek_level2_fewshot6_400.json",
        "labels": ["B", "C", "G", "Y"],
    },

    "QwenFlash_Level2_ZS": {
        "file": "02_llm_inference/results/qwen/qwen_level2_zero_shot_400.json",
        "labels": ["B", "C", "G", "Y"],
    },

    "QwenFlash_Level2_k3": {
        "file": "02_llm_inference/results/qwen/qwen_level2_fewshot3_400.json",
        "labels": ["B", "C", "G", "Y"],
    },

    "QwenFlash_Level2_k6": {
        "file": "02_llm_inference/results/qwen/qwen_level2_fewshot6_400.json",
        "labels": ["B", "C", "G", "Y"],
    },

    # -----------------------------
    # Prompt Sensitivity
    # -----------------------------
    "Qwen27B_P1": {
        "file": "02_llm_inference/results/prompt_sensitivity/qwen27b_prompt_P1_100.json",
        "labels": ["B", "C", "G", "Y"],
    },

    "Qwen27B_P2": {
        "file": "02_llm_inference/results/prompt_sensitivity/qwen27b_prompt_P2_100.json",
        "labels": ["B", "C", "G", "Y"],
    },

    "Qwen27B_P3": {
        "file": "02_llm_inference/results/prompt_sensitivity/qwen27b_prompt_P3_100.json",
        "labels": ["B", "C", "G", "Y"],
    },
}


# ============================================================
# 工具函数
# ============================================================

def load_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    if isinstance(results, dict):
        rows = list(results.values())
    elif isinstance(results, list):
        rows = results
    else:
        raise ValueError(f"无法识别 results 格式: {file_path}")

    rows = [
        r for r in rows
        if r.get("status", "success") == "success"
        and r.get("gold") is not None
        and r.get("prediction") is not None
    ]

    return data, rows


def calculate_error_statistics(rows, labels):

    gold = [str(r["gold"]).upper() for r in rows]
    pred = [str(r["prediction"]).upper() for r in rows]

    cm = pd.crosstab(
        pd.Series(gold, name="gold"),
        pd.Series(pred, name="prediction"),
        dropna=False
    )

    cm = cm.reindex(
        index=labels,
        columns=labels,
        fill_value=0
    )

    total = len(rows)
    correct = sum(g == p for g, p in zip(gold, pred))

    per_class = {}

    for label in labels:
        tp = int(cm.loc[label, label])

        gold_total = int(cm.loc[label].sum())
        pred_total = int(cm[label].sum())

        fp = pred_total - tp
        fn = gold_total - tp

        precision = (
            tp / pred_total
            if pred_total > 0
            else 0
        )

        recall = (
            tp / gold_total
            if gold_total > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        per_class[label] = {
            "gold_count": gold_total,
            "predicted_count": pred_total,
            "correct": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "error_rate": (
                1 - recall
                if gold_total > 0
                else 0
            ),
        }

    # --------------------------------------------------------
    # 错误迁移
    # --------------------------------------------------------

    migrations = []

    for gold_label in labels:
        for pred_label in labels:

            if gold_label == pred_label:
                continue

            count = int(cm.loc[gold_label, pred_label])

            if count > 0:
                migrations.append({
                    "gold": gold_label,
                    "prediction": pred_label,
                    "count": count,
                    "rate_within_gold": (
                        count / int(cm.loc[gold_label].sum())
                        if int(cm.loc[gold_label].sum()) > 0
                        else 0
                    )
                })

    migrations.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    return {
        "sample_count": total,
        "correct_count": correct,
        "accuracy": correct / total if total else 0,
        "confusion_matrix": cm.to_dict(),
        "label_order": labels,
        "per_class": per_class,
        "error_migrations": migrations,
    }


# ============================================================
# 主分析
# ============================================================

all_results = {}

print("=" * 80)
print("CAPC-CG 模型错误模式统一分析")
print("=" * 80)

for experiment_name, config in EXPERIMENTS.items():

    file_path = config["file"]
    labels = config["labels"]

    if not os.path.exists(file_path):
        print(f"\n[跳过] 文件不存在: {file_path}")
        continue

    print(f"\n正在分析：{experiment_name}")

    try:
        raw_data, rows = load_results(file_path)

        stats = calculate_error_statistics(
            rows,
            labels
        )

        stats["experiment"] = experiment_name
        stats["source_file"] = file_path

        # 保留已有指标
        for key in [
            "accuracy",
            "macro_f1",
            "kappa"
        ]:
            if key in raw_data:
                stats[f"original_{key}"] = raw_data[key]

        all_results[experiment_name] = stats

        print(
            f"  样本={stats['sample_count']} "
            f"Accuracy={stats['accuracy']:.4f}"
        )

        if stats["error_migrations"]:
            top = stats["error_migrations"][0]

            print(
                f"  最大错误迁移："
                f"{top['gold']} → {top['prediction']} "
                f"({top['count']})"
            )

    except Exception as e:
        print(f"  [错误] {e}")


# ============================================================
# 汇总表
# ============================================================

summary_rows = []

for name, result in all_results.items():

    row = {
        "experiment": name,
        "sample_count": result["sample_count"],
        "accuracy": result["accuracy"],
        "original_macro_f1": result.get(
            "original_macro_f1"
        ),
        "original_kappa": result.get(
            "original_kappa"
        ),
    }

    for label, info in result["per_class"].items():
        row[f"{label}_precision"] = info["precision"]
        row[f"{label}_recall"] = info["recall"]
        row[f"{label}_f1"] = info["f1"]
        row[f"{label}_error_rate"] = info["error_rate"]

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_file = os.path.join(
    OUTPUT_DIR,
    "error_analysis_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 保存完整 JSON
# ============================================================

json_file = os.path.join(
    OUTPUT_DIR,
    "error_analysis_full.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        all_results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 输出最主要错误
# ============================================================

print("\n" + "=" * 80)
print("各实验 Top-5 错误迁移")
print("=" * 80)

for name, result in all_results.items():

    print(f"\n{name}")

    migrations = result["error_migrations"][:5]

    if not migrations:
        print("  无错误")
        continue

    for item in migrations:
        print(
            f"  {item['gold']} → {item['prediction']}: "
            f"{item['count']} "
            f"({item['rate_within_gold']:.2%})"
        )


# ============================================================
# Level-1 / Level-2 汇总
# ============================================================

level1 = {
    k: v
    for k, v in all_results.items()
    if "Level1" in k
}

level2 = {
    k: v
    for k, v in all_results.items()
    if "Level2" in k
}


def average_class_error(results, label):

    values = []

    for result in results.values():

        if label in result["per_class"]:
            values.append(
                result["per_class"][label]["error_rate"]
            )

    return (
        sum(values) / len(values)
        if values
        else None
    )


comparison = {
    "level1": {
        label: average_class_error(
            level1,
            label
        )
        for label in ["N", "R", "W"]
    },
    "level2": {
        label: average_class_error(
            level2,
            label
        )
        for label in ["B", "C", "G", "Y"]
    }
}


comparison_file = os.path.join(
    OUTPUT_DIR,
    "level1_level2_error_comparison.json"
)

with open(
    comparison_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        comparison,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)

print(f"\n汇总 CSV：{summary_file}")
print(f"完整 JSON：{json_file}")
print(f"Level-1/Level-2 对比：{comparison_file}")