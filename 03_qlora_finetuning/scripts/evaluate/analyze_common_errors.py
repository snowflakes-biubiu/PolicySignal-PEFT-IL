
from pathlib import Path
import pandas as pd

# 读取已有对比结果
input_file = Path(
    "03_qlora_finetuning/results/rank_comparison_checkpoint300/all_sample_comparison.csv"
)
output_dir = Path(
    "03_qlora_finetuning/results/rank_comparison_checkpoint300/sample_analysis"
)
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, encoding="utf-8-sig")

# 检查必要列，避免列名不匹配
required_columns = [
    "index",
    "gold",
    "pred_rank4",
    "pred_rank8",
    "pred_rank16",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"缺少必要列：{missing}")

# 分别判断三个模型是否预测错误
df["wrong_rank4"] = df["pred_rank4"] != df["gold"]
df["wrong_rank8"] = df["pred_rank8"] != df["gold"]
df["wrong_rank16"] = df["pred_rank16"] != df["gold"]

# 三种模型共同预测错误
common_errors = df[
    df["wrong_rank4"] & df["wrong_rank8"] & df["wrong_rank16"]
].copy()

# 只有一个模型预测错误
rank4_only_errors = df[
    df["wrong_rank4"] & ~df["wrong_rank8"] & ~df["wrong_rank16"]
].copy()

rank8_only_errors = df[
    ~df["wrong_rank4"] & df["wrong_rank8"] & ~df["wrong_rank16"]
].copy()

rank16_only_errors = df[
    ~df["wrong_rank4"] & ~df["wrong_rank8"] & df["wrong_rank16"]
].copy()

# 汇总每种情况的数量
summary = pd.DataFrame([
    {
        "error_type": "三个模型共同错误",
        "sample_count": len(common_errors),
    },
    {
        "error_type": "仅Rank4错误",
        "sample_count": len(rank4_only_errors),
    },
    {
        "error_type": "仅Rank8错误",
        "sample_count": len(rank8_only_errors),
    },
    {
        "error_type": "仅Rank16错误",
        "sample_count": len(rank16_only_errors),
    },
])

print("=" * 65)
print("三种 LoRA Rank 错误样本统计")
print("=" * 65)
print(summary.to_string(index=False))

print("\n三个模型共同错误的样本编号：")
print(common_errors["index"].tolist())

# 保存结果，后续用于原文提取和案例分析
summary.to_csv(
    output_dir / "common_error_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

common_errors.to_csv(
    output_dir / "common_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

rank4_only_errors.to_csv(
    output_dir / "rank4_only_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

rank8_only_errors.to_csv(
    output_dir / "rank8_only_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

rank16_only_errors.to_csv(
    output_dir / "rank16_only_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n分析结果已保存到：")
print(output_dir.resolve())


from itertools import product

# 读取已经生成的三模型样本对比表
df = pd.read_csv(
    "03_qlora_finetuning/results/rank_comparison_checkpoint300/all_sample_comparison.csv"
)

cols = ["correct_rank4", "correct_rank8", "correct_rank16"]

# 兼容 CSV 中的布尔值或 0/1
for col in cols:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip().str.lower().map({
            "true": True,
            "false": False,
            "1": True,
            "0": False
        })

print("\n三种 Rank 的八种正确/错误组合")
print("=" * 60)

for pattern in product([True, False], repeat=3):
    mask = True
    for col, value in zip(cols, pattern):
        mask = mask & (df[col] == value)

    count = int(mask.sum())
    status = "，".join(
        f"{name}{'正确' if value else '错误'}"
        for name, value in zip(["Rank4", "Rank8", "Rank16"], pattern)
    )
    print(f"{status}：{count} 条，占 {count / len(df):.2%}")

print(f"\n样本总数：{len(df)}")