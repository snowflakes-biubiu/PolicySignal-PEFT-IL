from pathlib import Path
import pandas as pd

RESULT_DIR = Path("03_qlora_finetuning/results")

FILES = {
    "rank8": RESULT_DIR / "level1_qlora_checkpoint300_validation_300.csv",
    "rank4": RESULT_DIR / "level1_qlora_rank4_validation_300.csv",
    "rank16": RESULT_DIR / "level1_qlora_rank16_validation_300.csv",
}

OUT_DIR = RESULT_DIR / "rank_comparison_checkpoint300"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["N", "R", "W"]

def load_result(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    df = pd.read_csv(path)

    required = {"index", "gold", "prediction", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段：{missing}")

    if df["index"].duplicated().any():
        raise ValueError(f"{path} 的 index 存在重复")

    return df[["index", "gold", "prediction", "correct"]].copy()

data = {name: load_result(path) for name, path in FILES.items()}

# 以 Rank8 为基准，按 index 对齐并验证真实标签一致
base = data["rank8"].rename(columns={
    "gold": "gold_rank8",
    "prediction": "pred_rank8",
    "correct": "correct_rank8",
})

for name in ["rank4", "rank16"]:
    other = data[name].rename(columns={
        "gold": f"gold_{name}",
        "prediction": f"pred_{name}",
        "correct": f"correct_{name}",
    })
    base = base.merge(other, on="index", how="outer", validate="one_to_one")

if len(base) != 300 or base.isna().any().any():
    raise ValueError("三个结果文件样本数量不一致或 index 无法完整匹配")

for name in ["rank4", "rank16"]:
    if not (base["gold_rank8"] == base[f"gold_{name}"]).all():
        raise ValueError(f"Rank8 与 {name} 的真实标签不一致")

base["gold"] = base["gold_rank8"]
base["rank8_to_rank16"] = (
    base["correct_rank8"].astype(bool).map({True: "正确", False: "错误"})
    + " -> "
    + base["correct_rank16"].astype(bool).map({True: "正确", False: "错误"})
)

def error_type(row, pred_col):
    pred = row[pred_col]
    gold = row["gold"]
    if pred == gold:
        return "正确"
    return f"{gold}->{pred}"

base["error_rank8"] = base.apply(
    lambda row: error_type(row, "pred_rank8"), axis=1
)
base["error_rank16"] = base.apply(
    lambda row: error_type(row, "pred_rank16"), axis=1
)

# 汇总总体指标
summary = []
for name in ["rank4", "rank8", "rank16"]:
    pred_col = f"pred_{name}"
    correct_col = f"correct_{name}"
    correct = base[correct_col].astype(bool)
    summary.append({
        "model": name,
        "accuracy": correct.mean(),
        "correct_count": int(correct.sum()),
        "error_count": int((~correct).sum()),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    OUT_DIR / "rank_overall_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# Rank8 与 Rank16 的变化样本
improved = base[
    (~base["correct_rank8"].astype(bool))
    & base["correct_rank16"].astype(bool)
].copy()

regressed = base[
    base["correct_rank8"].astype(bool)
    & (~base["correct_rank16"].astype(bool))
].copy()

both_wrong = base[
    (~base["correct_rank8"].astype(bool))
    & (~base["correct_rank16"].astype(bool))
].copy()

improved.to_csv(OUT_DIR / "rank8_wrong_rank16_correct.csv",
                index=False, encoding="utf-8-sig")
regressed.to_csv(OUT_DIR / "rank8_correct_rank16_wrong.csv",
                 index=False, encoding="utf-8-sig")
both_wrong.to_csv(OUT_DIR / "both_rank8_rank16_wrong.csv",
                  index=False, encoding="utf-8-sig")
base.to_csv(OUT_DIR / "all_sample_comparison.csv",
            index=False, encoding="utf-8-sig")

# 按真实类别统计正确数、错误数，以及 Rank8->Rank16 的变化
by_class = []
for label in LABELS:
    subset = base[base["gold"] == label]
    for name in ["rank4", "rank8", "rank16"]:
        correct = subset[f"correct_{name}"].astype(bool)
        by_class.append({
            "gold_label": label,
            "model": name,
            "samples": len(subset),
            "correct": int(correct.sum()),
            "wrong": int((~correct).sum()),
            "accuracy": correct.mean() if len(subset) else float("nan"),
        })

pd.DataFrame(by_class).to_csv(
    OUT_DIR / "rank_by_class_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=" * 65)
print("Rank4 / Rank8 / Rank16 逐样本对比完成")
print("=" * 65)
print("\n总体正确率：")
print(summary_df.to_string(index=False))

print("\nRank8 错、Rank16 对：", len(improved))
print("Rank8 对、Rank16 错：", len(regressed))
print("两者都错：", len(both_wrong))

print("\nRank8 -> Rank16 变化，按真实类别统计：")
print(pd.crosstab(
    base["gold"],
    base["rank8_to_rank16"],
    margins=True,
).to_string())

print("\n输出目录：", OUT_DIR)
print("主要文件：")
print("- rank_overall_summary.csv")
print("- rank_by_class_summary.csv")
print("- rank8_wrong_rank16_correct.csv")
print("- rank8_correct_rank16_wrong.csv")
print("- both_rank8_rank16_wrong.csv")
print("- all_sample_comparison.csv")
