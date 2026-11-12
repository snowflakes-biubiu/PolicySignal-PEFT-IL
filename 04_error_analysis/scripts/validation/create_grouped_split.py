import json
import re
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(".")
OUT = ROOT / "04_error_analysis" / "data" / "grouped_split_v2"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = ROOT / "shared/datasets/processed_readable/task1_level1__train.jsonl"
VALID_FILE = ROOT / "shared/datasets/processed_readable/task1_level1__validation.jsonl"

SEED = 42
VALID_RATIO = 0.25

def normalize_text(text):
    # 去除空白差异；不删除标点或汉字，避免过度合并
    return re.sub(r"\s+", "", text).strip()

def extract_record(line, source, index):
    row = json.loads(line)
    messages = row["messages"]

    user_text = next(
        m["content"] for m in messages if m["role"] == "user"
    )
    label = next(
        m["content"].strip()
        for m in messages if m["role"] == "assistant"
    )

    text = re.sub(r"^待判断政策段落：\s*", "", user_text).strip()

    if label not in {"N", "R", "W"}:
        raise ValueError(f"发现无效标签：{label!r}")

    norm = normalize_text(text)
    group_id = hashlib.sha256(norm.encode("utf-8")).hexdigest()

    return {
        "source_split": source,
        "source_index": index,
        "text": text,
        "label": label,
        "group_id": group_id,
        "original_row": row,
    }

def read_jsonl(path, source):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            records.append(extract_record(line, source, i))
    return records

records = (
    read_jsonl(TRAIN_FILE, "old_train")
    + read_jsonl(VALID_FILE, "old_validation")
)

# 同一规范化文本只能进入一个分组
groups = defaultdict(list)
for row in records:
    groups[row["group_id"]].append(row)

group_items = list(groups.items())
rng = random.Random(SEED)
rng.shuffle(group_items)

# 以组为单位分配验证集，尽量接近各类别目标数量
total_labels = Counter(r["label"] for r in records)
target_valid = {
    label: round(total_labels[label] * VALID_RATIO)
    for label in ("N", "R", "W")
}

valid_groups = []
valid_counts = Counter()

# 优先分配能减少类别目标差距的组；超长组可能导致轻微偏差
remaining = list(group_items)

while remaining:
    best_idx = None
    best_score = None

    for idx, (gid, members) in enumerate(remaining):
        group_counts = Counter(r["label"] for r in members)

        # 已达到目标的类别不再优先增加
        gain = sum(
            min(target_valid[c], valid_counts[c] + group_counts[c])
            - min(target_valid[c], valid_counts[c])
            for c in ("N", "R", "W")
        )

        # 控制验证集总规模，并避免超出目标太多
        new_total = sum(valid_counts.values()) + len(members)
        size_penalty = abs(new_total - round(len(records) * VALID_RATIO))

        score = (gain, -size_penalty, rng.random())
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx

    gid, members = remaining.pop(best_idx)
    current_total = sum(valid_counts.values())

    # 若加入该组会明显超过验证集目标规模，且已经有足够样本，则停止
    if (
        current_total >= round(len(records) * VALID_RATIO)
        and sum(valid_counts.values()) >= sum(target_valid.values())
    ):
        break

    valid_groups.append(gid)
    valid_counts.update(r["label"] for r in members)

valid_group_set = set(valid_groups)

new_train = []
new_valid = []

for row in records:
    if row["group_id"] in valid_group_set:
        new_valid.append(row)
    else:
        new_train.append(row)

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row["original_row"], ensure_ascii=False) + "\n")

write_jsonl(OUT / "train.jsonl", new_train)
write_jsonl(OUT / "validation.jsonl", new_valid)

# 保存索引映射，供后续追踪样本
import csv

with (OUT / "split_manifest.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_split", "source_index", "label", "group_id", "new_split"
        ],
    )
    writer.writeheader()

    for row in records:
        writer.writerow({
            "source_split": row["source_split"],
            "source_index": row["source_index"],
            "label": row["label"],
            "group_id": row["group_id"],
            "new_split": (
                "validation"
                if row["group_id"] in valid_group_set
                else "train"
            ),
        })

train_groups = {r["group_id"] for r in new_train}
valid_groups_final = {r["group_id"] for r in new_valid}

print("原始记录数：", len(records))
print("规范化文本分组数：", len(groups))
print("新训练集记录数：", len(new_train))
print("新验证集记录数：", len(new_valid))
print("训练集标签分布：", dict(Counter(r["label"] for r in new_train)))
print("验证集标签分布：", dict(Counter(r["label"] for r in new_valid)))
print("跨集重复分组数：", len(train_groups & valid_groups_final))
print("分组清单：", OUT / "split_manifest.csv")
print("新训练集：", OUT / "train.jsonl")
print("新验证集：", OUT / "validation.jsonl")
