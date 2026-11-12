

from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

base_dir = Path("03_qlora_finetuning/results/rank_comparison_checkpoint300")
analysis_dir = base_dir / "sample_analysis"

# 读取共同错误样本
common_file = analysis_dir / "common_errors.csv"
common = pd.read_csv(common_file, encoding="utf-8-sig")

# 从本地缓存读取验证集，不重新下载
snapshot_dir = Path(
    snapshot_download(
        "Baron-Sun/CAPC-CG_V1.0",
        repo_type="dataset",
        local_files_only=True,
    )
)
validation_file = snapshot_dir / "task1_level1" / "validation.parquet"
data = pd.read_parquet(validation_file)

records = []

for _, row in common.iterrows():
    index = int(row["index"])
    messages = data.iloc[index]["messages"]

    policy_text = ""
    dataset_label = ""

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            marker = "待判断政策段落："
            if marker in content:
                policy_text = content.split(marker, 1)[-1].strip()
            else:
                policy_text = content.strip()

        elif role == "assistant":
            dataset_label = str(content).strip()

    records.append({
        "index": index,
        "policy_text": policy_text,
        "dataset_label": dataset_label,
        "gold": row["gold"],
        "pred_rank4": row["pred_rank4"],
        "pred_rank8": row["pred_rank8"],
        "pred_rank16": row["pred_rank16"],
        "correct_rank4": row["correct_rank4"],
        "correct_rank8": row["correct_rank8"],
        "correct_rank16": row["correct_rank16"],
    })

result = pd.DataFrame(records)
output_file = analysis_dir / "common_errors_with_text.csv"
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"共同错误样本数：{len(result)}")
print(f"数据集标签与对比文件标签一致：{(result['dataset_label'] == result['gold']).all()}")
print(f"输出文件：{output_file.resolve()}")

print("\n前3条样本预览：")
for _, row in result.head(3).iterrows():
    print("\n" + "=" * 70)
    print(f"样本编号：{row['index']}")
    print(f"真实标签：{row['gold']}")
    print(
        f"Rank4={row['pred_rank4']}，"
        f"Rank8={row['pred_rank8']}，"
        f"Rank16={row['pred_rank16']}"
    )
    print("政策原文：")
    print(row["policy_text"])


print("\n" + "=" * 70)
print("共同错误样本：第4至第8条")
print("=" * 70)

for _, row in result.iloc[3:8].iterrows():
    print("\n" + "-" * 70)
    print(f"样本编号：{row['index']}")
    print(f"真实标签：{row['gold']}")
    print(
        f"Rank4={row['pred_rank4']}，"
        f"Rank8={row['pred_rank8']}，"
        f"Rank16={row['pred_rank16']}"
    )
    print("政策原文：")
    print(row["policy_text"])
