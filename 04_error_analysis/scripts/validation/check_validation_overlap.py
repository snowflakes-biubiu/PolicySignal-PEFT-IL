import os
import json
import pandas as pd
from huggingface_hub import snapshot_download


# ============================================================
# 配置
# ============================================================

SNAPSHOT_DIR = snapshot_download(
    "Baron-Sun/CAPC-CG_V1.0",
    repo_type="dataset",
    local_files_only=True
)

TASK1_FILE = os.path.join(
    SNAPSHOT_DIR,
    "task1_level1",
    "validation.parquet"
)

TASK2_FILE = os.path.join(
    SNAPSHOT_DIR,
    "task2_level2",
    "validation.parquet"
)


# ============================================================
# 数据解析
# ============================================================

def parse_messages(messages):
    user_text = ""
    label = ""

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            user_text = content

        elif role == "assistant":
            label = content.strip()

    return user_text, label


def clean_text(text):
    text = str(text).strip()

    prefixes = [
        "待判断政策段落：",
        "政策段落：",
        "Policy paragraph to classify:"
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text


# ============================================================
# 读取
# ============================================================

print("=" * 70)
print("层级实验：Level-1 / Level-2 验证集重合检查")
print("=" * 70)

task1 = pd.read_parquet(TASK1_FILE)
task2 = pd.read_parquet(TASK2_FILE)

print(f"\nTask1 validation: {len(task1)}")
print(f"Task2 validation: {len(task2)}")

print("\nTask1 columns:")
print(list(task1.columns))

print("\nTask2 columns:")
print(list(task2.columns))


# ============================================================
# 提取文本和标签
# ============================================================

task1_records = []

for idx, row in task1.iterrows():

    user_text, label = parse_messages(row["messages"])

    task1_records.append({
        "task1_index": idx,
        "text": clean_text(user_text),
        "level1_gold": label.upper()
    })


task2_records = []

for idx, row in task2.iterrows():

    user_text, label = parse_messages(row["messages"])

    task2_records.append({
        "task2_index": idx,
        "text": clean_text(user_text),
        "level2_gold": label.upper()
    })


task1_df = pd.DataFrame(task1_records)
task2_df = pd.DataFrame(task2_records)


# ============================================================
# 标签统计
# ============================================================

print("\n" + "=" * 70)
print("Task1 Level-1 标签")
print("=" * 70)

print(task1_df["level1_gold"].value_counts())


print("\n" + "=" * 70)
print("Task2 Level-2 标签")
print("=" * 70)

print(task2_df["level2_gold"].value_counts())


# ============================================================
# 文本精确匹配
# ============================================================

merged = task1_df.merge(
    task2_df,
    on="text",
    how="inner"
)

print("\n" + "=" * 70)
print("验证集精确文本重合")
print("=" * 70)

print(f"重合样本数：{len(merged)}")

if len(merged) > 0:

    print("\n前10条重合样本：")

    print(
        merged[
            [
                "task1_index",
                "task2_index",
                "level1_gold",
                "level2_gold"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nLevel-1 × Level-2：")

    print(
        pd.crosstab(
            merged["level1_gold"],
            merged["level2_gold"]
        )
    )


# ============================================================
# 输出结果
# ============================================================

os.makedirs("hierarchical", exist_ok=True)

output_file = os.path.join(
    "hierarchical",
    "validation_overlap_check.json"
)

result = {
    "task1_validation": len(task1),
    "task2_validation": len(task2),
    "overlap_count": len(merged),
    "overlap_ratio_task1": (
        len(merged) / len(task1)
        if len(task1) else 0
    ),
    "overlap_ratio_task2": (
        len(merged) / len(task2)
        if len(task2) else 0
    )
}

if len(merged) > 0:

    result["overlap_level1_distribution"] = (
        merged["level1_gold"]
        .value_counts()
        .to_dict()
    )

    result["overlap_level2_distribution"] = (
        merged["level2_gold"]
        .value_counts()
        .to_dict()
    )

    result["level1_level2_crosstab"] = (
        pd.crosstab(
            merged["level1_gold"],
            merged["level2_gold"]
        )
        .to_dict()
    )

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n结果已保存：")
print(output_file)

print("\n检查完成。")