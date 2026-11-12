"""CAPC-CG 复现评测指标 (与论文一致，纯 Python 实现，无需 sklearn).

论文报告的核心指标：
  - Cohen's Kappa (多级/二级分类的一致性)
  - Accuracy (整体准确率)
  - Macro-F1      (各类 F1 的算术平均，论文称 M-F1)
  - 各类 Precision / Recall / F1 (误差分析用)

参考：Table 3 / Table 4 (Level-1, R/W/N) 与 Table 5 (Level-2, B/Y/C/G)。
"""
from __future__ import annotations

from collections import Counter


def accuracy(y_true, y_pred) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def cohen_kappa(y_true, y_pred, labels=None) -> float:
    """Cohen's Kappa，衡量预测与金标的一致性（排除随机一致）。"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    n = len(y_true)
    if n == 0:
        return 0.0
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    # 行=真值，列=预测
    conf = {l: Counter() for l in labels}
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    po = sum(conf[l][l] for l in labels) / n
    pe = 0.0
    for l in labels:
        p_true = sum(conf[l].values()) / n
        p_pred = sum(conf[r][l] for r in labels) / n
        pe += p_true * p_pred
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def per_class_f1(y_true, y_pred, labels):
    out = {}
    for l in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[l] = {"precision": prec, "recall": rec, "f1": f1}
    return out


def macro_f1(y_true, y_pred, labels) -> float:
    pc = per_class_f1(y_true, y_pred, labels)
    return sum(v["f1"] for v in pc.values()) / len(labels)


def evaluate(y_true, y_pred, labels) -> dict:
    """一次性返回论文所用的指标包。

    例：evaluate(y_true, y_pred, labels=["W","R","N"])
    -> {'accuracy','cohen_kappa','macro_f1','per_class'}
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "cohen_kappa": cohen_kappa(y_true, y_pred, labels),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class": per_class_f1(y_true, y_pred, labels),
    }


def print_report(y_true, y_pred, labels, title: str = ""):
    rep = evaluate(y_true, y_pred, labels)
    print(f"\n=== {title} ===")
    print(f"Accuracy   : {rep['accuracy']:.3f}")
    print(f"Cohen Kappa: {rep['cohen_kappa']:.3f}")
    print(f"Macro-F1   : {rep['macro_f1']:.3f}")
    print("Per-class F1:")
    for l in labels:
        v = rep["per_class"][l]
        print(f"  {l}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")
    return rep


if __name__ == "__main__":
    # 自测：完美预测应得 1.0；随机应低于 1.0
    yt = ["W", "R", "N", "W", "R", "N"]
    yp = ["W", "R", "N", "W", "R", "N"]
    print_report(yt, yp, ["W", "R", "N"], "self-test (perfect)")
