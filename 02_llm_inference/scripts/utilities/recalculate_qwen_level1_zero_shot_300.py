# ============================================================
# 实验标题：Qwen Level-1 Zero-shot 结果重算
# ============================================================

import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 基本配置
# ============================================================

EXPERIMENT_TITLE = "Qwen Level-1 Zero-shot 实验"

INPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300_fixed.json"

VALID_LABELS = ["N", "R", "W"]


# ============================================================
# 读取原始实验结果
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


results = data["results"]


# ============================================================
# 筛选有效预测
# ============================================================

valid_results = [
    item
    for item in results
    if item.get("gold") in VALID_LABELS
    and item.get("prediction") in VALID_LABELS
]


failed_results = [
    item
    for item in results
    if item.get("prediction") not in VALID_LABELS
]


y_true = [
    item["gold"]
    for item in valid_results
]

y_pred = [
    item["prediction"]
    for item in valid_results
]


# ============================================================
# 计算指标
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    average="macro",
    zero_division=0,
)

kappa = cohen_kappa_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
)


report = classification_report(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    zero_division=0,
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=VALID_LABELS,
)


# ============================================================
# 输出结果
# ============================================================

print()
print("=" * 70)
print(f"{EXPERIMENT_TITLE} 结果重算")
print("=" * 70)

print("原始样本数：", len(results))
print("有效预测：", len(valid_results), "/", len(results))
print("失败预测：", len(failed_results))

print()
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro-F1 : {macro_f1:.4f}")
print(f"Kappa    : {kappa:.4f}")

print()
print("Classification Report:")
print(report)

print("Confusion Matrix")
print("标签顺序：N / R / W")
print(cm)


# ============================================================
# 输出失败样本
# ============================================================

if failed_results:

    print()
    print("=" * 70)
    print("失败样本")
    print("=" * 70)

    for item in failed_results:
        print()
        print("index：", item.get("index"))
        print("真实标签：", item.get("gold"))
        print("prediction：", item.get("prediction"))
        print("error：", item.get("error"))
        print("error_detail：", item.get("error_detail"))


# ============================================================
# 保存重算结果
# ============================================================

output = {
    "experiment": EXPERIMENT_TITLE,
    "source_file": INPUT_FILE,
    "total_samples": len(results),
    "valid_predictions": len(valid_results),
    "failed_predictions": len(failed_results),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "kappa": kappa,
    "classification_report": report,
    "confusion_matrix_labels": VALID_LABELS,
    "confusion_matrix": cm.tolist(),
    "failed_indices": [
        item.get("index")
        for item in failed_results
    ],
}


OUTPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300_recalculated.json"

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 70)
print("重算完成")
print("=" * 70)
print("结果文件：", OUTPUT_FILE)
print("=" * 70)