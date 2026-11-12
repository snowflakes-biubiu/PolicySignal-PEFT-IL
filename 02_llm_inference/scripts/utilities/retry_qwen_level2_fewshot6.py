# -*- coding: utf-8 -*-

"""
Qwen Level-2 Few-shot k=6 Retry 实验

仅补跑：
397 -> G
398 -> Y
399 -> Y

不会重新调用 0-396。
"""

import os
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. 配置
# ============================================================

MODEL = "qwen3.8-flash"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

API_KEY_ENV = "DASHSCOPE_API_KEY"

DATA_DIR = Path(
    r"C:\Users\snowf\.cache\huggingface\hub"
    r"\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
)

TRAIN_FILE = (
    DATA_DIR
    / "task2_level2"
    / "train.parquet"
)

VALIDATION_FILE = (
    DATA_DIR
    / "task2_level2"
    / "validation.parquet"
)

CHECKPOINT_FILE = Path(
    "02_llm_inference/results/qwen/qwen_level2_fewshot6_400_checkpoint.json"
)

FINAL_RESULT_FILE = Path(
    "02_llm_inference/results/qwen/qwen_level2_fewshot6_400_final.json"
)

RETRY_INDICES = [397, 398, 399]

VALID_LABELS = ["B", "C", "G", "Y"]


# ============================================================
# 2. API
# ============================================================

api_key = os.getenv(API_KEY_ENV)

if not api_key:
    raise RuntimeError(
        f"环境变量 {API_KEY_ENV} 未找到。"
    )

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
)


# ============================================================
# 3. 辅助函数
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    return str(text).strip()


def parse_prediction(content):

    if not content:
        return None

    content = str(content).strip().upper()

    if content in VALID_LABELS:
        return content

    for label in VALID_LABELS:
        if label in content:
            return label

    return None


# ============================================================
# 4. 读取数据
# ============================================================

print("=" * 70)
print("Qwen Level-2 Few-shot k=6 Retry 实验")
print("=" * 70)

print("\n[1/6] 读取数据...")

train_df = pd.read_parquet(
    TRAIN_FILE
)

valid_df = pd.read_parquet(
    VALIDATION_FILE
)

print(f"训练集：{len(train_df)}")
print(f"验证集：{len(valid_df)}")


# ============================================================
# 5. 构造 24 个 few-shot 示例
#
# 注意：
# messages 实际类型为 numpy.ndarray
#
# 每个样本：
# messages[0] = system
# messages[1] = user
# messages[2] = assistant
# ============================================================

print(
    "\n[2/6] 构造与原实验一致的 24 个 few-shot 示例..."
)

examples = {
    "B": [],
    "C": [],
    "G": [],
    "Y": [],
}

system_prompt = None

for _, row in train_df.iterrows():

    messages = row["messages"]

    if len(messages) != 3:
        continue

    system_msg = messages[0]
    user_msg = messages[1]
    assistant_msg = messages[2]

    # 获取 system prompt
    if system_prompt is None:
        if system_msg["role"] == "system":
            system_prompt = str(
                system_msg["content"]
            )

    # 获取政策文本
    if user_msg["role"] != "user":
        continue

    # 获取 gold
    if assistant_msg["role"] != "assistant":
        continue

    text = clean_text(
        user_msg["content"]
    )

    label = clean_text(
        assistant_msg["content"]
    ).upper()

    if label not in VALID_LABELS:
        continue

    if not text:
        continue

    # 每类只取前 6 条
    if len(examples[label]) < 6:

        examples[label].append(
            {
                "text": text,
                "label": label,
            }
        )

    # 四类全部达到 6 条
    if all(
        len(examples[x]) == 6
        for x in VALID_LABELS
    ):
        break


print("\nFew-shot 示例数量：")

for label in VALID_LABELS:
    print(
        f"{label}: {len(examples[label])}"
    )

if not all(
    len(examples[x]) == 6
    for x in VALID_LABELS
):
    raise RuntimeError(
        "无法构造完整的 24 条 few-shot 示例。"
    )

if not system_prompt:
    raise RuntimeError(
        "无法读取 Level-2 system prompt。"
    )

print(
    "\n成功构造 24 条 few-shot 示例。"
)


# ============================================================
# 6. 构造 Few-shot messages
# ============================================================

fewshot_messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

for label in VALID_LABELS:

    for example in examples[label]:

        fewshot_messages.append(
            {
                "role": "user",
                "content": example["text"],
            }
        )

        fewshot_messages.append(
            {
                "role": "assistant",
                "content": example["label"],
            }
        )


print(
    f"Few-shot message 数："
    f"{len(fewshot_messages)}"
)


# ============================================================
# 7. 读取 checkpoint
# ============================================================

print(
    "\n[3/6] 读取已有 checkpoint..."
)

if not CHECKPOINT_FILE.exists():
    raise FileNotFoundError(
        f"找不到：{CHECKPOINT_FILE}"
    )

with open(
    CHECKPOINT_FILE,
    "r",
    encoding="utf-8",
) as f:

    old_results = json.load(f)

print(
    f"已有成功结果："
    f"{len(old_results)}"
)

old_indices = {
    item["index"]
    for item in old_results
}

missing_indices = [
    i
    for i in range(400)
    if i not in old_indices
]

print(
    f"当前缺失索引："
    f"{missing_indices}"
)

if missing_indices != RETRY_INDICES:
    raise RuntimeError(
        "checkpoint 状态与预期不一致。\n"
        f"预期缺失：{RETRY_INDICES}\n"
        f"实际缺失：{missing_indices}"
    )


# ============================================================
# 8. 补跑 397 / 398 / 399
# ============================================================

print(
    "\n[4/6] 开始补跑 3 条失败样本..."
)

retry_results = []

for index in RETRY_INDICES:

    row = valid_df.iloc[index]

    messages = row["messages"]

    # validation 结构同样是：
    # 0 system
    # 1 user
    # 2 assistant

    user_msg = messages[1]
    assistant_msg = messages[2]

    user_text = clean_text(
        user_msg["content"]
    )

    gold = clean_text(
        assistant_msg["content"]
    ).upper()

    print("\n" + "-" * 70)
    print(f"index = {index}")
    print(f"Gold  = {gold}")
    print("-" * 70)

    if gold not in VALID_LABELS:
        raise RuntimeError(
            f"index={index} gold={gold} 非法"
        )

    request_messages = list(
        fewshot_messages
    )

    request_messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    success = False

    for attempt in range(1, 4):

        try:

            start_time = time.time()

            response = client.chat.completions.create(
                model=MODEL,
                messages=request_messages,
                max_tokens=10,
                extra_body={
                    "enable_thinking": False
                },
                timeout=60,
            )

            elapsed = (
                time.time()
                - start_time
            )

            raw_response = (
                response
                .choices[0]
                .message
                .content
            )

            prediction = parse_prediction(
                raw_response
            )

            if prediction is None:
                raise ValueError(
                    f"无法解析模型输出："
                    f"{raw_response!r}"
                )

            result = {
                "index": index,
                "gold": gold,
                "prediction": prediction,
                "raw_response": raw_response,
            }

            retry_results.append(
                result
            )

            print(
                f"模型输出："
                f"{raw_response!r}"
            )

            print(
                f"预测：{prediction}"
            )

            print(
                f"Gold：{gold}"
            )

            print(
                f"耗时："
                f"{elapsed:.2f} 秒"
            )

            success = True
            break

        except Exception as e:

            print(
                f"第 {attempt}/3 次失败："
                f"{type(e).__name__}: {e}"
            )

            if attempt < 3:
                time.sleep(2)

    if not success:

        raise RuntimeError(
            f"index={index} 连续 3 次失败，"
            f"停止实验。"
        )


# ============================================================
# 9. 合并
# ============================================================

print(
    "\n[5/6] 合并结果..."
)

all_results = (
    old_results
    + retry_results
)

all_results = sorted(
    all_results,
    key=lambda x: x["index"]
)

all_indices = [
    x["index"]
    for x in all_results
]

expected_indices = list(
    range(400)
)

if all_indices != expected_indices:

    missing = [
        i
        for i in expected_indices
        if i not in set(all_indices)
    ]

    duplicated = [
        i
        for i in set(all_indices)
        if all_indices.count(i) > 1
    ]

    raise RuntimeError(
        f"结果完整性检查失败。\n"
        f"缺失：{missing}\n"
        f"重复：{duplicated}"
    )


print(
    "结果完整性检查通过："
    "0-399，共 400 条。"
)


# ============================================================
# 10. 保存完整结果
# ============================================================

with open(
    FINAL_RESULT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_results,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(
    f"完整结果已保存："
    f"{FINAL_RESULT_FILE}"
)


# ============================================================
# 11. 计算指标
# ============================================================

print(
    "\n[6/6] 计算完整 400 条指标..."
)

y_true = [
    x["gold"]
    for x in all_results
]

y_pred = [
    x["prediction"]
    for x in all_results
]

accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    average="macro",
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
    target_names=VALID_LABELS,
    digits=4,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=VALID_LABELS,
)


# ============================================================
# 12. 最终输出
# ============================================================

print("\n")
print("=" * 70)
print("Qwen Level-2 Few-shot k=6 最终结果")
print("=" * 70)

print(f"模型：{MODEL}")
print("Few-shot：k=6")
print("Few-shot 示例：24 条")
print("验证集：400")
print(
    f"有效样本：{len(all_results)}"
)

print(
    f"\nAccuracy ：{accuracy:.4f}"
)

print(
    f"Macro-F1 ：{macro_f1:.4f}"
)

print(
    f"Kappa    ：{kappa:.4f}"
)

print(
    "\nClassification Report"
)

print(report)

print(
    "Confusion Matrix [B, C, G, Y]"
)

print(cm)

print("\n" + "=" * 70)
print("实验完成")
print("=" * 70)

print(
    f"\n完整结果："
    f"{FINAL_RESULT_FILE}"
)

print(
    "\n本次新增 API 调用：最多 3×3 次"
)