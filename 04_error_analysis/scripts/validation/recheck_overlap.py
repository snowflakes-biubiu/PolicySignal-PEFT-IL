import json
import re
from pathlib import Path
from collections import defaultdict

root = Path(".")
train_file = root / "shared/datasets/processed_readable/task1_level1__train.jsonl"
valid_file = root / "shared/datasets/processed_readable/task1_level1__validation.jsonl"
out_dir = root / "error_analysis"
out_dir.mkdir(exist_ok=True)

def extract_record(line):
    row = json.loads(line)
    messages = row["messages"]
    user_text = next(
        m["content"] for m in messages if m["role"] == "user"
    )
    assistant_text = next(
        m["content"] for m in messages if m["role"] == "assistant"
    )
    text = re.sub(r"^待判断政策段落：\s*", "", user_text)
    label = assistant_text.strip()
    return text, label

def normalize(text):
    return re.sub(r"\s+", "", text).strip()

def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            text, label = extract_record(line)
            records.append({
                "index": i,
                "text": text,
                "normalized_text": normalize(text),
                "label": label
            })
    return records

train = read_jsonl(train_file)
valid = read_jsonl(valid_file)

train_map = defaultdict(list)
for row in train:
    train_map[row["normalized_text"]].append(row)

matches = []
for v in valid:
    found = train_map.get(v["normalized_text"], [])
    if found:
        matches.append({
            "validation_index": v["index"],
            "validation_label": v["label"],
            "train_indices": ",".join(str(x["index"]) for x in found),
            "train_labels": ",".join(x["label"] for x in found),
            "match_count": len(found),
            "text": v["text"]
        })

import csv
output = out_dir / "train_validation_overlap_rechecked.csv"
with output.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "validation_index", "validation_label", "train_indices",
        "train_labels", "match_count", "text"
    ])
    writer.writeheader()
    writer.writerows(matches)

print("训练集样本数：", len(train))
print("验证集样本数：", len(valid))
print("完全重复的验证集样本数：", sum(
    1 for v in valid if any(
        t["text"] == v["text"] for t in train_map[v["normalized_text"]]
    )
))
print("标准化后重复的验证集样本数：", len(matches))
print("涉及的训练集匹配记录数：", sum(x["match_count"] for x in matches))
print("标签不一致的重复组数：", sum(
    1 for x in matches
    if any(label != x["validation_label"] for label in x["train_labels"].split(","))
))
print("详细报告：", output)
