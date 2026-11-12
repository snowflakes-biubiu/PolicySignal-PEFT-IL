import os
import glob
import numpy as np
import pandas as pd
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. 找到 Hugging Face 数据集
# ============================================================

cache_root = os.path.expanduser(
    r"~\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
)

snapshots = glob.glob(os.path.join(cache_root, "snapshots", "*"))

if not snapshots:
    raise FileNotFoundError("没有找到 CAPC-CG_V1.0 数据集缓存")

snapshot_dir = max(snapshots, key=os.path.getmtime)

train_path = os.path.join(
    snapshot_dir,
    "task1_level1",
    "train.parquet"
)

val_path = os.path.join(
    snapshot_dir,
    "task1_level1",
    "validation.parquet"
)

print("数据集目录：")
print(snapshot_dir)
print()


# ============================================================
# 2. 解析 messages
# ============================================================

def parse_messages(messages):

    if isinstance(messages, np.ndarray):
        messages = messages.tolist()

    user_text = None
    label = None

    for message in messages:

        if isinstance(message, dict):

            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                user_text = content

            elif role == "assistant":
                label = content.strip()

    if user_text is None or label is None:
        raise ValueError("无法正确解析 messages")

    # 与之前实验保持一致：
    # 只删除提示语，不删除换行、制表符等格式字符
    user_text = user_text.replace("待判断政策段落：", "")

    return user_text, label


def load_dataset(path):

    df = pd.read_parquet(path)

    texts = []
    labels = []

    for messages in df["messages"]:

        text, label = parse_messages(messages)

        texts.append(text)
        labels.append(label)

    return texts, labels


# ============================================================
# 3. 读取数据
# ============================================================

print("读取训练集...")
train_texts, train_labels = load_dataset(train_path)

print("读取验证集...")
val_texts, val_labels = load_dataset(val_path)

print()

print("训练集：", len(train_texts))
print("验证集：", len(val_texts))

print(
    "训练标签：",
    pd.Series(train_labels).value_counts().to_dict()
)

print(
    "验证标签：",
    pd.Series(val_labels).value_counts().to_dict()
)

print()


# ============================================================
# 4. 词级 TF-IDF
# ============================================================

print("正在进行中文分词...")

def jieba_tokenizer(text):
    return jieba.lcut(text)


print("正在进行词级 TF-IDF 特征提取...")

vectorizer = TfidfVectorizer(
    analyzer="word",
    tokenizer=jieba_tokenizer,
    token_pattern=None,
    ngram_range=(1, 2),
    max_features=5000
)

X_train = vectorizer.fit_transform(train_texts)

X_val = vectorizer.transform(val_texts)

print("训练特征维度：", X_train.shape)
print("验证特征维度：", X_val.shape)
print()


# ============================================================
# 5. 标签编码
# ============================================================

label_to_id = {
    "N": 0,
    "R": 1,
    "W": 2
}

y_train = np.array(
    [label_to_id[x] for x in train_labels]
)

y_val = np.array(
    [label_to_id[x] for x in val_labels]
)


# ============================================================
# 6. XGBoost
# ============================================================

print("开始训练词级 XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("训练完成。")
print()


# ============================================================
# 7. 预测
# ============================================================

print("正在进行预测...")

y_pred = model.predict(X_val)

print()


# ============================================================
# 8. 评价指标
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

kappa = cohen_kappa_score(
    y_val,
    y_pred
)


print("=" * 60)
print("Word-level TF-IDF + XGBoost")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro-F1 : {macro_f1:.4f}")
print(f"Kappa    : {kappa:.4f}")

print()

print("Classification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=["N", "R", "W"],
        digits=4
    )
)

print("Confusion Matrix:")
print()

cm = confusion_matrix(
    y_val,
    y_pred
)

print("       N    R    W")

print(
    f"N   {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}"
)

print(
    f"R   {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}"
)

print(
    f"W   {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}"
)