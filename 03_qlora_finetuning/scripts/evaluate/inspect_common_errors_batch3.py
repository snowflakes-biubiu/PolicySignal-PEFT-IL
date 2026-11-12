
from pathlib import Path
import pandas as pd

file = Path(
    "03_qlora_finetuning/results/rank_comparison_checkpoint300/"
    "sample_analysis/common_errors_with_text.csv"
)

df = pd.read_csv(file, encoding="utf-8-sig")

print(f"共同错误样本总数：{len(df)}")
print("本批输出第9至第13条样本")

for _, row in df.iloc[13:17].iterrows():
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
