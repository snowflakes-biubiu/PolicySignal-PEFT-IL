# ============================================================
# 实验标题：Qwen Level-2 Zero-shot 实验
# ============================================================

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
# 1. 基础配置
# ============================================================

EXPERIMENT_TITLE = "Qwen Level-2 Zero-shot 实验"

MODEL_NAME = "qwen3.8-flash"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

PROJECT_DIR = Path(r"C:\Users\snowf\CAPC-CG-Project")

DATA_FILE = (
    Path(r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0")
    / "snapshots"
    / "d05965f169e6f7da33a25f667edf39add0339e2b"
    / "task2_level2"
    / "validation.parquet"
)

OUTPUT_FILE = PROJECT_DIR / "02_llm_inference/results/qwen/qwen_level2_zero_shot_400.json"

CHECKPOINT_FILE = (
    PROJECT_DIR / "02_llm_inference/results/qwen/qwen_level2_zero_shot_400_checkpoint.json"
)

TARGET_N = 400

VALID_LABELS = {"B", "C", "G", "Y"}

LABEL_ORDER = ["B", "C", "G", "Y"]


# ============================================================
# 2. API 配置
# ============================================================

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise RuntimeError(
        "未检测到 DASHSCOPE_API_KEY 环境变量。"
    )

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
)


# ============================================================
# 3. 读取 Level-2 验证集
# ============================================================

print("=" * 72)
print(EXPERIMENT_TITLE)
print("=" * 72)

print(f"模型：{MODEL_NAME}")
print(f"数据集：{DATA_FILE}")
print()

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"找不到验证集文件：\n{DATA_FILE}"
    )

df = pd.read_parquet(DATA_FILE)

print(f"验证集总数：{len(df)}")

if len(df) != 400:
    raise RuntimeError(
        f"验证集数量异常，预期 400，实际 {len(df)}"
    )

if "messages" not in df.columns:
    raise RuntimeError(
        f"找不到 messages 字段，当前字段：{df.columns.tolist()}"
    )

print("数据结构检查：通过")
print()


# ============================================================
# 4. 解析 messages
# ============================================================

def parse_sample(messages):
    """
    从数据集 messages 中提取：

    system_prompt
    user_prompt
    gold_label
    """

    if isinstance(messages, str):
        messages = json.loads(messages)

    system_prompt = None
    user_content = None
    gold_label = None

    for message in messages:

        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            system_prompt = content

        elif role == "user":
            user_content = content

        elif role == "assistant":
            gold_label = content.strip().upper()

    if system_prompt is None:
        raise ValueError("样本缺少 system message")

    if user_content is None:
        raise ValueError("样本缺少 user message")

    if gold_label not in VALID_LABELS:
        raise ValueError(
            f"Gold 标签异常：{gold_label}"
        )

    return system_prompt, user_content, gold_label


# ============================================================
# 5. 预解析全部 400 条
# ============================================================

samples = []

for i, row in df.iloc[:TARGET_N].iterrows():

    system_prompt, user_content, gold_label = parse_sample(
        row["messages"]
    )

    samples.append(
        {
            "index": int(i),
            "system_prompt": system_prompt,
            "user_content": user_content,
            "gold": gold_label,
        }
    )

print(f"解析样本：{len(samples)} 条")

# 检查标签分布
gold_distribution = {}

for sample in samples:
    gold = sample["gold"]
    gold_distribution[gold] = gold_distribution.get(gold, 0) + 1

print("Gold 标签分布：")

for label in LABEL_ORDER:
    print(
        f"  {label}: {gold_distribution.get(label, 0)}"
    )

print()


# ============================================================
# 6. 读取已有 checkpoint
# ============================================================

results = {}

if CHECKPOINT_FILE.exists():

    print(
        f"发现已有 checkpoint：{CHECKPOINT_FILE}"
    )

    with open(
        CHECKPOINT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        checkpoint = json.load(f)

    results = checkpoint.get("results", {})

    completed = sum(
        1
        for r in results.values()
        if r.get("status") == "success"
        and r.get("prediction") in VALID_LABELS
    )

    print(f"已有成功结果：{completed} 条")
    print()

else:

    print("未发现已有 checkpoint，从第 1 条开始。")
    print()


# ============================================================
# 7. API 调用函数
# ============================================================

def call_model(system_prompt, user_content, max_retries=3):

    for attempt in range(1, max_retries + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                max_tokens=10,
                timeout=60,
                extra_body={
                    "enable_thinking": False
                },
            )

            answer = (
                response.choices[0]
                .message
                .content
                .strip()
                .upper()
            )

            # 严格接受四个合法标签
            if answer in VALID_LABELS:
                return answer

            print(
                f"    ⚠ 无效模型输出：{answer!r}"
            )

        except Exception as e:

            print(
                f"    ⚠ API 调用失败 "
                f"({attempt}/{max_retries})："
                f"{type(e).__name__}: {e}"
            )

        if attempt < max_retries:
            time.sleep(2 * attempt)

    return None


# ============================================================
# 8. 保存 checkpoint
# ============================================================

def save_checkpoint():

    checkpoint_data = {
        "experiment": EXPERIMENT_TITLE,
        "model": MODEL_NAME,
        "target_samples": TARGET_N,
        "results": results,
    }

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint_data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 9. 正式开始实验
# ============================================================

print("=" * 72)
print("开始 Qwen3.8-Flash Level-2 Zero-shot 实验")
print("=" * 72)
print()

start_time = time.time()

for sample in samples:

    index = sample["index"]
    key = str(index)

    # --------------------------------------------------------
    # 已经成功的样本不重复调用
    # --------------------------------------------------------

    if (
        key in results
        and results[key].get("status") == "success"
        and results[key].get("prediction") in VALID_LABELS
    ):
        continue

    gold = sample["gold"]

    print(
        f"[{index + 1:03d}/{TARGET_N}] "
        f"Gold={gold} | "
        f"调用 Qwen3.8-Flash...",
        end=" ",
        flush=True,
    )

    prediction = call_model(
        sample["system_prompt"],
        sample["user_content"],
    )

    if prediction is None:

        print("失败")

        results[key] = {
            "index": index,
            "gold": gold,
            "prediction": None,
            "status": "failed",
        }

    else:

        correct = prediction == gold

        print(
            f"Pred={prediction} "
            f"{'✓' if correct else '✗'}"
        )

        results[key] = {
            "index": index,
            "gold": gold,
            "prediction": prediction,
            "status": "success",
            "correct": correct,
        }

    # 每完成一条立即保存
    save_checkpoint()


# ============================================================
# 10. 汇总有效结果
# ============================================================

valid_results = [
    r
    for r in results.values()
    if r.get("status") == "success"
    and r.get("prediction") in VALID_LABELS
    and r.get("gold") in VALID_LABELS
]

failed_results = [
    r
    for r in results.values()
    if r.get("status") != "success"
]


# ============================================================
# 11. 计算评价指标
# ============================================================

if valid_results:

    y_true = [
        r["gold"]
        for r in valid_results
    ]

    y_pred = [
        r["prediction"]
        for r in valid_results
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

else:

    accuracy = None
    macro_f1 = None
    kappa = None
    report = ""
    cm = []


# ============================================================
# 12. 保存最终实验结果
# ============================================================

final_output = {
    "experiment": EXPERIMENT_TITLE,
    "model": MODEL_NAME,

    "dataset": "Baron-Sun/CAPC-CG_V1.0",
    "task": "Level-2",
    "setting": "Zero-shot",

    "target_samples": TARGET_N,
    "valid_samples": len(valid_results),
    "failed_samples": len(failed_results),

    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "kappa": kappa,

    "classification_report": report,

    "confusion_matrix": (
        cm.tolist()
        if hasattr(cm, "tolist")
        else cm
    ),

    "label_order": LABEL_ORDER,

    "results": results,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_output,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 13. 最终输出
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 72)
print("Qwen Level-2 Zero-shot 实验完成")
print("=" * 72)

print(f"模型：{MODEL_NAME}")
print(f"目标数量：{TARGET_N}")
print(f"有效数量：{len(valid_results)}")
print(f"失败数量：{len(failed_results)}")

if valid_results:

    print()
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")

    print()
    print("Classification Report:")
    print(report)

    print("Confusion Matrix")
    print("行 = Gold，列 = Prediction")
    print("标签顺序：B / C / G / Y")
    print(cm)

print()
print(f"耗时：{elapsed / 60:.2f} 分钟")
print()
print(f"最终结果：{OUTPUT_FILE}")
print(f"Checkpoint：{CHECKPOINT_FILE}")

print("=" * 72)