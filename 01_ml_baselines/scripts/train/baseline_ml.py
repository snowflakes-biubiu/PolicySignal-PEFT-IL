"""CAPC-CG 基线1：传统机器学习 (SVM / XGBoost + TF-IDF).

本脚本同时跑两套配置，便于诚实复现 + 对照分析：

【A. 论文忠实复现配置】 —— 对照论文 Table 3 / Appendix 传统 ML 基线
  论文原文(Appendix)：Traditional ML baselines used TF-IDF features with
  1–2 grams and a 5,000-feature cap; the SVM baseline used a linear kernel
  with C=1, and XGBoost used 200 trees with maximum depth 6.
  论文 Table 3 报告(L1, R/W/N)：
    SVM     Kappa=0.175  Acc=0.508  M-F1=0.442
    XGBoost Kappa=0.153  Acc=0.487  M-F1=0.420
  注意：中文无空格，word 级 1–2 元在未经分词时近乎失效；本脚本用
  char(1,2)+max_features=5000 作为最贴近“1–2 grams + 5000 cap”的忠实实现。
  由于发布划分(2500条)≠论文内部6000/1901，数字不会逐位相等，但应落在
  同一量级(低 Kappa)，以验证“传统 ML 弱、LLM/BERT 强”的论文结论。

【B. 增强配置】 —— 仅作对照分析，说明“特征工程”能极大缩小差距
  char_wb(2,4)+无上限+类别平衡。该配置不属于论文复现，结果是“传统 ML
  在更好特征下也能不错”的工程性旁证，不应与论文数字直接比较。

运行：<venv>/python 01_ml_baselines/scripts/train/baseline_ml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "common"))

import preprocess
import metrics as M

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RESULTS_MD = PROJECT_ROOT / "01_ml_baselines" / "results" / "results.md"


# ---------- 配置定义 ----------
def vec_paper():
    # 论文忠实：TF-IDF 1–2 元，5000 特征上限（中文以 char 级近似 word 1–2 元）
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=5000,
                           sublinear_tf=True)


def vec_enhanced():
    # 增强：字符 2–4 元 + 词 1–2 元（无上限），对中文政策文本高度判别
    from sklearn.pipeline import FeatureUnion
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return FeatureUnion([("char", char), ("word", word)])


def svm_paper():
    return LinearSVC(C=1.0, random_state=42)  # 论文：linear kernel, C=1, 无 class_weight


def svm_enhanced():
    return LinearSVC(C=1.0, class_weight="balanced", random_state=42)


def xgb_paper():
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.3,
                         tree_method="auto", random_state=42, verbosity=0)


def xgb_enhanced():
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9,
                         tree_method="auto", random_state=42, verbosity=0)


# 论文 Table 3 参考(L1 R/W/N)
PAPER_REF_L1 = {
    "SVM(LinearSVC)|paper": (0.175, 0.508, 0.442),
    "XGBoost|paper": (0.153, 0.487, 0.420),
}


def run_one(task, vec_factory, clf_factory, tag, paper_ref=None):
    labels = preprocess.TASK_LABELS[task]
    train = preprocess.load_split(task, "train")
    val = preprocess.load_split(task, "validation")
    X_tr = [r["text"] for r in train]
    y_tr = [r["label"] for r in train]
    X_va = [r["text"] for r in val]
    y_va = [r["label"] for r in val]

    le = LabelEncoder().fit(y_tr)
    vec = vec_factory()
    clf = clf_factory()
    pipe = Pipeline([("vec", vec), ("clf", clf)])
    pipe.fit(X_tr, le.transform(y_tr))
    y_pred = le.inverse_transform(pipe.predict(X_va))
    rep = M.evaluate(y_va, list(y_pred), labels)

    print(f"\n[{tag}] {task}")
    print(f"  Accuracy   : {rep['accuracy']:.3f}")
    print(f"  Cohen Kappa: {rep['cohen_kappa']:.3f}")
    print(f"  Macro-F1   : {rep['macro_f1']:.3f}")
    for l in labels:
        v = rep["per_class"][l]
        print(f"    {l}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")

    ref_s = ""
    if paper_ref:
        ref_s = f" | 论文参考(Acc/Kappa/MF1)={paper_ref[1]}/{paper_ref[0]}/{paper_ref[2]}"
    line = (f"| {task} | {tag} | {rep['accuracy']:.3f} | "
            f"{rep['cohen_kappa']:.3f} | {rep['macro_f1']:.3f} |{ref_s}\n")
    return line


def main():
    with RESULTS_MD.open("a", encoding="utf-8") as f:
        f.write("\n\n## 基线1：传统机器学习 (SVM / XGBoost + TF-IDF)\n")
        f.write("> 发布划分(2500条)上的可复现近似；论文 exact 数字基于未公开的内部6000/1901。\n")
        f.write("> A=论文忠实配置(对照 Table 3)；B=增强配置(仅对照分析，非复现)。\n\n")
        f.write("| Task | Config | Acc | CohenKappa | MacroF1 | 备注 |\n")
        f.write("|---|---|---:|---:|---:|---|\n")

        # ---- A. 论文忠实复现 ----
        for task in preprocess.TASK_LABELS:
            ref = PAPER_REF_L1.get(f"SVM(LinearSVC)|paper") if task == "task1_level1" else None
            l1 = run_one(task, vec_paper, svm_paper, "SVM|paper", ref)
            f.write(l1)
            ref2 = PAPER_REF_L1.get("XGBoost|paper") if task == "task1_level1" else None
            l2 = run_one(task, vec_paper, xgb_paper, "XGBoost|paper", ref2)
            f.write(l2)

        # ---- B. 增强配置(对照分析) ----
        for task in preprocess.TASK_LABELS:
            l3 = run_one(task, vec_enhanced, svm_enhanced, "SVM|enhanced")
            f.write(l3)
            l4 = run_one(task, vec_enhanced, xgb_enhanced, "XGBoost|enhanced")
            f.write(l4)

    print(f"\n结果已追加到 {RESULTS_MD}")


if __name__ == "__main__":
    main()
