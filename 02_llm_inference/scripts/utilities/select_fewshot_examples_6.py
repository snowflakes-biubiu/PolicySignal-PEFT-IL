import json
import pandas as pd

TRAIN_FILE = r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\task1_level1\train.parquet"


def parse_messages(messages):
    if isinstance(messages, str):
        messages = json.loads(messages)

    user_text = ""
    label = ""

    for msg in messages:
        if msg["role"] == "user":
            user_text = msg["content"]
        elif msg["role"] == "assistant":
            label = msg["content"].strip()

    prefix = "待判断政策段落："

    if user_text.startswith(prefix):
        user_text = user_text[len(prefix):]

    return user_text.strip(), label


df = pd.read_parquet(TRAIN_FILE)

counts = {
    "N": 0,
    "R": 0,
    "W": 0,
}

selected = []

for i, row in df.iterrows():

    text, label = parse_messages(row["messages"])

    if label not in counts:
        continue

    counts[label] += 1

    # 选择每个类别的第2个样本
    if counts[label] == 2:
        selected.append({
            "source_index": int(i),
            "label": label,
            "text": text,
        })

    if len(selected) == 3:
        break


print("=" * 70)
print("新增 Few-shot k=6 示例")
print("=" * 70)

for item in sorted(selected, key=lambda x: x["label"]):

    print()
    print("-" * 70)
    print(f"标签：{item['label']}")
    print(f"训练集索引：{item['source_index']}")
    print("政策内容：")
    print(item["text"])

print()
print("=" * 70)