# ============================================================
# 实验标题：Qwen Level-1 Few-shot k=6 最终结果重算
# ============================================================

import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


INPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_fewshot6_300_fixed.json"

LABELS = ["N", "R", "W"]


# ============================================================
# 读取结果
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]


# ============================================================
# 提取有效预测
# ============================================================

valid_results = [
    x for x in results
    if x.get("prediction") in LABELS
]

y_true = [
    x["gold"]
    for x in valid_results
]

y_pred = [
    x["prediction"]
    for x in valid_results
]


# ============================================================
# 计算指标
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=LABELS,
    average="macro",
)

kappa = cohen_kappa_score(
    y_true,
    y_pred,
    labels=LABELS,
)


# ============================================================
# 输出
# ============================================================

print("=" * 70)
print("Qwen Level-1 Few-shot k=6 实验 最终结果")
print("=" * 70)

print(f"总样本数： {len(results)}")
print(f"有效预测： {len(valid_results)} / {len(results)}")
print(f"失败预测： {len(results) - len(valid_results)}")

print()
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro-F1 : {macro_f1:.4f}")
print(f"Kappa    : {kappa:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
)

print("Confusion Matrix")
print("标签顺序：N / R / W")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )
)

print("=" * 70)