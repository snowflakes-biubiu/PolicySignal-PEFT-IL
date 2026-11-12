
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download


comparison_file = Path(
    "03_qlora_finetuning/results/rank_comparison_checkpoint300/all_sample_comparison.csv"
)

comparison = pd.read_csv(comparison_file, encoding="utf-8-sig")

print("文件列名：")
for column in comparison.columns:
    print(repr(column))

print("\n前5行数据：")
print(comparison.head().to_string(index=False))


# 2. 从本地 Hugging Face 缓存读取验证集
snapshot_dir = Path(
    snapshot_download(
        "Baron-Sun/CAPC-CG_V1.0",
        repo_type="dataset",
        local_files_only=True,
    )
)

validation_file = snapshot_dir / "task1_level1" / "validation.parquet"
data = pd.read_parquet(validation_file)

# 3. 读取第119条原始样本
messages = data.iloc[119]["messages"]

print("\n" + "=" * 70)
print("第119条样本的原始内容")
print("=" * 70)

for message in messages:
    role = message.get("role")
    content = message.get("content", "")

    if role == "user":
        marker = "待判断政策段落："
        if marker in content:
            content = content.split(marker, 1)[-1].strip()

        print("\n【政策原文】")
        print(content)

    elif role == "assistant":
        print("\n【数据集原始标签】")
        print(content)


sample = comparison[comparison["index"] == 119]

if sample.empty:
    print("未找到第119条样本")
else:
    columns = [
        "index",
        "gold",
        "pred_rank4",
        "pred_rank8",
        "pred_rank16",
        "correct_rank4",
        "correct_rank8",
        "correct_rank16",
    ]
    print(sample[columns].to_string(index=False))
