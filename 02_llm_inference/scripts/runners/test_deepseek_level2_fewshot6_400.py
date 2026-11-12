# ============================================================
# 实验标题：DeepSeek Level-2 Few-shot k=6 实验
# ============================================================

import os
import json
import time
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 基础配置
# ============================================================

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"

DATA_DIR = (
    r"C:\Users\snowf\.cache\huggingface\hub"
    r"\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task2_level2"
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train.parquet"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "validation.parquet"
)

PROJECT_DIR = r"C:\Users\snowf\CAPC-CG-Project"

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "02_llm_inference/results/deepseek/deepseek_level2_fewshot6_400.json"
)

CHECKPOINT_FILE = os.path.join(
    PROJECT_DIR,
    "02_llm_inference/results/deepseek/deepseek_level2_fewshot6_400_checkpoint.json"
)

VALID_LABELS = ["B", "C", "G", "Y"]

# ============================================================
# API
# ============================================================

api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("未找到 DEEPSEEK_API_KEY 环境变量")

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
)

# ============================================================
# 数据解析
# ============================================================

def parse_messages(messages):
    system_prompt = ""
    user_text = ""
    assistant_label = ""

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            system_prompt = content

        elif role == "user":
            user_text = content

        elif role == "assistant":
            assistant_label = content.strip()

    return system_prompt, user_text, assistant_label


def clean_text(text):
    prefix = "待判断政策段落："

    if text.startswith(prefix):
        text = text[len(prefix):]

    return text.strip()


# ============================================================
# 实验开始
# ============================================================

print("=" * 70)
print("DeepSeek Level-2 Few-shot k=6 实验")
print("=" * 70)

# ============================================================
# 读取训练集
# ============================================================

print("\n读取训练集...")

train_df = pd.read_parquet(TRAIN_FILE)

print(f"训练集样本：{len(train_df)}")

# ============================================================
# 选择 Few-shot 示例
#
# B/C/G/Y 每类6条
# 共24条示例
# ============================================================

examples = {
    "B": [],
    "C": [],
    "G": [],
    "Y": []
}

for _, row in train_df.iterrows():

    messages = row["messages"]

    system_prompt, user_text, label = parse_messages(messages)

    label = label.strip().upper()

    if label in VALID_LABELS and len(examples[label]) < 6:

        examples[label].append({
            "text": clean_text(user_text),
            "label": label
        })

    if all(len(examples[x]) == 6 for x in VALID_LABELS):
        break

# ============================================================
# 检查示例数量
# ============================================================

for label in VALID_LABELS:

    if len(examples[label]) != 6:

        raise RuntimeError(
            f"类别 {label} 的 Few-shot 示例不足6条，"
            f"实际只有 {len(examples[label])} 条。"
        )

print("\nFew-shot 示例选择完成：")

for label in VALID_LABELS:

    print(
        f"{label}: "
        f"{len(examples[label])} 条"
    )

# ============================================================
# 使用训练集中的正式 System Prompt
# ============================================================

system_prompt = parse_messages(
    train_df.iloc[0]["messages"]
)[0]

# ============================================================
# 构造 Few-shot 消息
# ============================================================

fewshot_messages = [
    {
        "role": "system",
        "content": system_prompt
    }
]

for label in VALID_LABELS:

    for example in examples[label]:

        fewshot_messages.append({
            "role": "user",
            "content": "待判断政策段落：" + example["text"]
        })

        fewshot_messages.append({
            "role": "assistant",
            "content": example["label"]
        })

# ============================================================
# 读取验证集
# ============================================================

print("\n读取验证集...")

val_df = pd.read_parquet(VAL_FILE)

print(f"验证集样本：{len(val_df)}")

val_df = val_df.iloc[:400].copy()

# ============================================================
# Checkpoint恢复
# ============================================================

results = []

if os.path.exists(CHECKPOINT_FILE):

    print("\n发现已有Checkpoint，尝试恢复...")

    with open(
        CHECKPOINT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    print(
        f"已恢复 {len(results)} 条结果"
    )

else:

    print("\n未发现Checkpoint，从头开始。")

# ============================================================
# 正式预测
# ============================================================

start_index = len(results)

print("\n开始正式预测...")
print(f"剩余样本：{400 - start_index}")

for idx in range(start_index, 400):

    row = val_df.iloc[idx]

    system_prompt_row, user_text, gold = parse_messages(
        row["messages"]
    )

    user_text = clean_text(user_text)

    gold = gold.strip().upper()

    # --------------------------------------------------------
    # 构造当前请求
    # --------------------------------------------------------

    messages = list(fewshot_messages)

    messages.append({
        "role": "user",
        "content": "待判断政策段落：" + user_text
    })

    # --------------------------------------------------------
    # API调用
    # --------------------------------------------------------

    prediction = None

    for attempt in range(3):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=4,
                stream=False,
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                }
            )

            prediction = (
                response.choices[0]
                .message.content
                .strip()
            )

            break

        except Exception as e:

            print(
                f"\n样本 {idx + 1} API失败 "
                f"(第{attempt + 1}次)：{e}"
            )

            if attempt < 2:
                time.sleep(2)

    # --------------------------------------------------------
    # API最终失败
    # --------------------------------------------------------

    if prediction is None:

        print(
            f"\n样本 {idx + 1} 最终失败，跳过。"
        )

        continue

    # --------------------------------------------------------
    # 解析标签
    # --------------------------------------------------------

    prediction_upper = prediction.upper()

    pred_label = None

    for label in VALID_LABELS:

        if label in prediction_upper:

            pred_label = label
            break

    # --------------------------------------------------------
    # 无效输出
    # --------------------------------------------------------

    if pred_label is None:

        print(
            f"\n样本 {idx + 1} 无效输出："
            f"{prediction}"
        )

        continue

    # --------------------------------------------------------
    # 保存单条结果
    # --------------------------------------------------------

    result = {
        "index": idx,
        "gold": gold,
        "prediction": pred_label,
        "raw_prediction": prediction,
        "text": user_text
    }

    results.append(result)

    # --------------------------------------------------------
    # 每条成功结果立即保存
    # --------------------------------------------------------

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"[{idx + 1:03d}/400] "
        f"Gold={gold} "
        f"Pred={pred_label}"
    )

# ============================================================
# 检查有效结果
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "没有获得有效预测结果。"
    )

# ============================================================
# 计算指标
# ============================================================

y_true = [
    item["gold"]
    for item in results
]

y_pred = [
    item["prediction"]
    for item in results
]

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    average="macro"
)

kappa = cohen_kappa_score(
    y_true,
    y_pred,
    labels=VALID_LABELS
)

report = classification_report(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    target_names=VALID_LABELS,
    digits=4
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=VALID_LABELS
)

# ============================================================
# 保存最终结果
# ============================================================

output = {
    "experiment": "DeepSeek Level-2 Few-shot k=6 实验",
    "model": MODEL,
    "dataset": "Baron-Sun/CAPC-CG_V1.0",
    "task": "Level-2",
    "method": "Few-shot",
    "k": 6,
    "fewshot_total": 24,
    "labels": VALID_LABELS,
    "num_samples": len(results),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "kappa": kappa,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "results": results
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

# ============================================================
# 输出最终结果
# ============================================================

print("\n")
print("=" * 70)
print("DeepSeek Level-2 Few-shot k=6 实验")
print("=" * 70)

print(f"有效样本：{len(results)}")
print(f"Accuracy ：{accuracy:.4f}")
print(f"Macro-F1 ：{macro_f1:.4f}")
print(f"Kappa    ：{kappa:.4f}")

print("\nClassification Report")
print(report)

print("Confusion Matrix [B, C, G, Y]")
print(cm)

print(
    f"\n结果已保存：{OUTPUT_FILE}"
)

print(
    f"Checkpoint：{CHECKPOINT_FILE}"
)