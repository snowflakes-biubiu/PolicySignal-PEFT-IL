import json
import pandas as pd

TRAIN_FILE = r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\task1_level1\train.parquet"


def parse_messages(messages):
    if isinstance(messages, str):
        messages = json.loads(messages)

    user_text = ""
    assistant_label = ""

    for msg in messages:
        if msg["role"] == "user":
            user_text = msg["content"]

        elif msg["role"] == "assistant":
            assistant_label = msg["content"].strip()

    prefix = "待判断政策段落："
    if user_text.startswith(prefix):
        user_text = user_text[len(prefix):]

    return user_text.strip(), assistant_label


df = pd.read_parquet(TRAIN_FILE)

selected = {}

for i, row in df.iterrows():

    text, label = parse_messages(row["messages"])

    if label in {"N", "R", "W"} and label not in selected:
        selected[label] = {
            "source_index": int(i),
            "label": label,
            "text": text,
        }

    if len(selected) == 3:
        break


print("=" * 70)
print("Few-shot k=3 示例")
print("=" * 70)

for label in ["N", "R", "W"]:

    item = selected[label]

    print()
    print("-" * 70)
    print(f"标签：{label}")
    print(f"训练集索引：{item['source_index']}")
    print("政策内容：")
    print(item["text"])

print()
print("=" * 70)