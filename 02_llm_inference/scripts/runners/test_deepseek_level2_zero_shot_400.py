import json
import os
import time
from openai import OpenAI
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

VAL_FILE = r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\task2_level2\validation.parquet"

CHECKPOINT_FILE = "02_llm_inference/results/deepseek/deepseek_level2_zero_shot_400_checkpoint.json"
RESULT_FILE = "02_llm_inference/results/deepseek/deepseek_level2_zero_shot_400.json"

MODEL = "deepseek-flash"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)


def parse_messages(messages):
    if isinstance(messages, str):
        messages = json.loads(messages)

    system_prompt = ""
    user_text = ""
    label = ""

    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]
        elif msg["role"] == "assistant":
            label = msg["content"].strip()

    prefix = "待判断政策段落："
    if user_text.startswith(prefix):
        user_text = user_text[len(prefix):]

    return system_prompt, user_text.strip(), label


# ============================================================
# 读取验证集
# ============================================================

df = pd.read_parquet(VAL_FILE)

samples = []

for i, row in df.iterrows():
    system_prompt, text, gold = parse_messages(row["messages"])

    samples.append({
        "index": int(i),
        "system_prompt": system_prompt,
        "text": text,
        "gold": gold
    })

samples = samples[:400]


print("=" * 70)
print("DeepSeek Level-2 Zero-shot 正式实验")
print("=" * 70)
print(f"模型：{MODEL}")
print("验证集：400")
print("类别：B / C / G / Y")
print("=" * 70)


# ============================================================
# 加载 checkpoint
# ============================================================

results = []

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

print(f"已有 checkpoint：{len(results)} 条")

completed_indices = {x["index"] for x in results}


# ============================================================
# API 调用
# ============================================================

for n, sample in enumerate(samples, start=1):

    if sample["index"] in completed_indices:
        continue

    print(f"[{n}/400] 正在处理验证集索引 {sample['index']} ...")

    messages = [
        {
            "role": "system",
            "content": sample["system_prompt"]
        },
        {
            "role": "user",
            "content": "待判断政策段落：" + sample["text"]
        }
    ]

    success = False

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

            prediction = response.choices[0].message.content.strip()

            if prediction not in ["B", "C", "G", "Y"]:
                for label in ["B", "C", "G", "Y"]:
                    if label in prediction:
                        prediction = label
                        break
                else:
                    prediction = ""

            result = {
                "index": sample["index"],
                "text": sample["text"],
                "gold": sample["gold"],
                "prediction": prediction
            }

            results.append(result)

            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    results,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            success = True
            break

        except Exception as e:
            print(
                f"  API失败，第 {attempt + 1}/3 次：{e}"
            )
            time.sleep(2)

    if not success:
        print("  本条连续失败，跳过。")

    time.sleep(0.1)


# ============================================================
# 统计
# ============================================================

valid_results = [
    x for x in results
    if x["prediction"] in ["B", "C", "G", "Y"]
]

y_true = [x["gold"] for x in valid_results]
y_pred = [x["prediction"] for x in valid_results]

labels = ["B", "C", "G", "Y"]

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=labels,
    average="macro"
)

kappa = cohen_kappa_score(
    y_true,
    y_pred,
    labels=labels
)


print()
print("=" * 70)
print("DeepSeek Level-2 Zero-shot 实验")
print("=" * 70)

print(f"有效样本：{len(valid_results)}")
print(f"Accuracy ：{accuracy:.4f}")
print(f"Macro-F1 ：{macro_f1:.4f}")
print(f"Kappa    ：{kappa:.4f}")

print()
print("Classification Report")

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4
    )
)

print("Confusion Matrix [B, C, G, Y]")

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )
)


# ============================================================
# 保存结果
# ============================================================

output = {
    "model": MODEL,
    "experiment": "Level-2 Zero-shot",
    "n_samples": len(valid_results),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "kappa": kappa,
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True
    ),
    "confusion_matrix": confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    ).tolist(),
    "results": valid_results
}

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print(f"结果已保存：{RESULT_FILE}")
print(f"Checkpoint：{CHECKPOINT_FILE}")