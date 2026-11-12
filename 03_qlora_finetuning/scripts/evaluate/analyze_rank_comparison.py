from pathlib import Path
import pandas as pd

file = Path(
    "03_qlora_finetuning/results/rank_comparison_checkpoint300/"
    "all_sample_comparison.csv"
)

df = pd.read_csv(file, encoding="utf-8-sig")

df["gold"] = df["gold"].astype(str).str.strip()
df["pred_rank8"] = df["pred_rank8"].astype(str).str.strip()
df["pred_rank16"] = df["pred_rank16"].astype(str).str.strip()

df["correct8"] = df["gold"] == df["pred_rank8"]
df["correct16"] = df["gold"] == df["pred_rank16"]

print("=" * 70)
print("Class-level accuracy comparison")
print("=" * 70)

for label in ["N", "R", "W"]:
    part = df[df["gold"] == label]
    n = len(part)
    c8 = int(part["correct8"].sum())
    c16 = int(part["correct16"].sum())

    print(f"\nClass: {label}")
    print(f"Samples: {n}")
    print(f"Rank8 correct: {c8}, accuracy: {c8 / n:.4f}")
    print(f"Rank16 correct: {c16}, accuracy: {c16 / n:.4f}")
    print(f"Correct-count change: {c16 - c8}")

fixed = int((~df["correct8"] & df["correct16"]).sum())
regressed = int((df["correct8"] & ~df["correct16"]).sum())

print("\n" + "=" * 70)
print("Overall comparison")
print("=" * 70)
print(f"Total samples: {len(df)}")
print(f"Rank8 accuracy: {df['correct8'].mean():.4f}")
print(f"Rank16 accuracy: {df['correct16'].mean():.4f}")
print(f"Rank8 wrong -> Rank16 correct: {fixed}")
print(f"Rank8 correct -> Rank16 wrong: {regressed}")
print(f"Net corrected samples: {fixed - regressed}")

print("\n" + "=" * 70)
print("Prediction transition matrices")
print("=" * 70)

labels = ["N", "R", "W"]

for label in labels:
    part = df[df["gold"] == label]

    matrix = pd.crosstab(
        part["pred_rank8"],
        part["pred_rank16"],
        rownames=["Rank8 prediction"],
        colnames=["Rank16 prediction"],
    ).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )

    print(f"\nGold label: {label}")
    print(matrix.to_string())

print("\n" + "=" * 70)
print("Samples whose predictions changed")
print("=" * 70)

changed = df[df["pred_rank8"] != df["pred_rank16"]]

print("Changed samples:", len(changed))
print(
    changed[
        [
            "index",
            "gold",
            "pred_rank8",
            "pred_rank16",
            "correct8",
            "correct16",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("Sample 119: prediction changed but remains incorrect")
print("=" * 70)

sample_file = Path(
    "03_qlora_finetuning/results/rank_comparison_checkpoint300/"
    "rank8_rank16_change_samples_with_text.csv"
)

samples = pd.read_csv(sample_file, encoding="utf-8-sig")

# 如果第119条不在15条样本文件中，先明确提示
if "sample_index" in samples.columns:
    sample = samples[samples["sample_index"] == 119]
else:
    sample = pd.DataFrame()

if sample.empty:
    print("Sample 119 is not in the existing 15-sample file.")
    print("It must be extracted from the original validation data.")
else:
    print(sample.to_string(index=False))