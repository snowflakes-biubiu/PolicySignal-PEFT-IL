from pathlib import Path
import ast
import jieba
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. 找到 CAPC-CG 数据集
# ============================================================

root = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--Baron-Sun--CAPC-CG_V1.0"
    / "snapshots"
)

snapshots = [p for p in root.iterdir() if p.is_dir()]
snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)

print("数据集目录：")
print(snapshot)


# ============================================================
# 2. 读取 Level-1 数据
# ============================================================

train_path = snapshot / "task1_level1" / "train.parquet"
val_path = snapshot / "task1_level1" / "validation.parquet"

train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)

print()
print("训练集数量：", len(train_df))
print("验证集数量：", len(val_df))


# ============================================================
# 3. 解析 messages
# ============================================================

def parse_messages(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return value


# ============================================================
# 4. 提取政策文本和标签
# ============================================================

def extract_text_and_label(value):
    messages = parse_messages(value)

    if not isinstance(messages, list):
        return None, None

    user_text = None
    label = None

    for item in messages:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = str(item.get("content", "")).strip()

        if role == "user":
            user_text = content
            user_text = user_text.replace("待判断政策段落：", "")
            user_text = user_text.replace("\n", "")
            user_text = user_text.replace("\r", "")
            user_text = user_text.replace("\t", "")
            user_text = user_text.strip()

        elif role == "assistant":
            label = content

    return user_text, label


# ============================================================
# 5. 提取训练集
# ============================================================

train_result = train_df["messages"].apply(extract_text_and_label)

X_train = train_result.apply(lambda x: x[0])
y_train = train_result.apply(lambda x: x[1])


# ============================================================
# 6. 提取验证集
# ============================================================

val_result = val_df["messages"].apply(extract_text_and_label)

X_val = val_result.apply(lambda x: x[0])
y_val = val_result.apply(lambda x: x[1])


# ============================================================
# 7. TF-IDF
# ============================================================

print()
print("=" * 60)
print("开始进行 TF-IDF 特征提取")
print("=" * 60)

vectorizer = TfidfVectorizer(
    analyzer="word",
    tokenizer=jieba.lcut,
    token_pattern=None,
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("训练集 TF-IDF 形状：", X_train_tfidf.shape)
print("验证集 TF-IDF 形状：", X_val_tfidf.shape)

print("特征数量：", len(vectorizer.get_feature_names_out()))

print()
print("前20个 TF-IDF 特征：")
print(vectorizer.get_feature_names_out()[:20])
# ============================================================
# 8. Linear SVM 分类
# ============================================================

from sklearn.svm import LinearSVC


print()
print("=" * 60)
print("开始训练 Linear SVM")
print("=" * 60)

svm_model = LinearSVC(
    C=1
)

svm_model.fit(X_train_tfidf, y_train)


# ============================================================
# 9. 在验证集上进行预测
# ============================================================

y_pred = svm_model.predict(X_val_tfidf)

print()
print("SVM 训练完成！")
print("验证集预测完成！")

print()
print("前20条预测结果：")

for i in range(20):
    print(
        f"{i + 1:02d}. "
        f"预测={y_pred[i]}    "
        f"真实={y_val.iloc[i]}"
    )
# ============================================================
# 10. 计算评价指标
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score
)

accuracy = accuracy_score(y_val, y_pred)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

kappa = cohen_kappa_score(
    y_val,
    y_pred
)

print()
print("=" * 60)
print("Word-level TF-IDF + SVM")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro-F1 : {macro_f1:.4f}")
print(f"Kappa    : {kappa:.4f}")
# ============================================================
# 11. 分类报告
# ============================================================

from sklearn.metrics import classification_report

print()
print("=" * 60)
print("分类报告")
print("=" * 60)

print(
    classification_report(
        y_val,
        y_pred,
        labels=["N", "R", "W"],
        digits=4
    )
)
# ============================================================
# 12. 混淆矩阵
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=["N", "R", "W"]
)

print()
print("=" * 60)
print("混淆矩阵")
print("=" * 60)

print("          预测")
print("          N    R    W")
print(
    f"真实 N   {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}"
)
print(
    f"     R   {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}"
)
print(
    f"     W   {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}"
)