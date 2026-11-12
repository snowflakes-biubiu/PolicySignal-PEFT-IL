import os
import json
import pandas as pd


OUTPUT_DIR = "04_error_analysis/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 实验配置
# ============================================================

EXPERIMENTS = {
    # Level-1
    "DeepSeek_Level1_ZS": (
        "02_llm_inference/results/deepseek/deepseek_level1_zero_shot_300.json",
        ["N", "R", "W"]
    ),
    "DeepSeek_Level1_k3": (
        "02_llm_inference/results/deepseek/deepseek_level1_fewshot3_300.json",
        ["N", "R", "W"]
    ),
    "DeepSeek_Level1_k6": (
        "02_llm_inference/results/deepseek/deepseek_level1_fewshot6_300.json",
        ["N", "R", "W"]
    ),

    "QwenMax_Level1_ZS": (
        "02_llm_inference/results/qwen/qwen_level1_zero_shot_300_fixed.json",
        ["N", "R", "W"]
    ),
    "QwenMax_Level1_k3": (
        "02_llm_inference/results/qwen/qwen_level1_fewshot3_300.json",
        ["N", "R", "W"]
    ),
    "QwenMax_Level1_k6": (
        "02_llm_inference/results/qwen/qwen_level1_fewshot6_300_fixed.json",
        ["N", "R", "W"]
    ),

    # Level-2
    "DeepSeek_Level2_ZS": (
        "02_llm_inference/results/deepseek/deepseek_level2_zero_shot_400.json",
        ["B", "C", "G", "Y"]
    ),
    "DeepSeek_Level2_k3": (
        "02_llm_inference/results/deepseek/deepseek_level2_fewshot3_400.json",
        ["B", "C", "G", "Y"]
    ),
    "DeepSeek_Level2_k6": (
        "02_llm_inference/results/deepseek/deepseek_level2_fewshot6_400.json",
        ["B", "C", "G", "Y"]
    ),

    "QwenFlash_Level2_ZS": (
        "02_llm_inference/results/qwen/qwen_level2_zero_shot_400.json",
        ["B", "C", "G", "Y"]
    ),
    "QwenFlash_Level2_k3": (
        "02_llm_inference/results/qwen/qwen_level2_fewshot3_400.json",
        ["B", "C", "G", "Y"]
    ),

    # 注意：
    # QwenFlash_Level2_k6 不在这里重新读取。
    # 之前正式结果已经锁定为 400/400：
    # Accuracy = 0.6125
    # 因此这里不使用本次统计脚本错误读出的 397 条。
}


# ============================================================
# 加载
# ============================================================

def load_rows(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    if isinstance(results, dict):
        rows = list(results.values())
    else:
        rows = results

    rows = [
        r for r in rows
        if r.get("status", "success") == "success"
    ]

    return data, rows


# ============================================================
# 指定错误结构
# ============================================================

ERROR_GROUPS_LEVEL1 = {
    "W_to_N": ("W", "N"),
    "N_to_W": ("N", "W"),
    "R_to_N": ("R", "N"),
    "N_to_R": ("N", "R"),
    "W_to_R": ("W", "R"),
    "R_to_W": ("R", "W"),
}

ERROR_GROUPS_LEVEL2 = {
    "B_Y_bidirectional": None,
    "B_C_bidirectional": None,
    "C_G_bidirectional": None,
    "B_G_bidirectional": None,
    "C_Y_bidirectional": None,
    "G_Y_bidirectional": None,
}


def count_pair(rows, gold, pred):

    return sum(
        1
        for r in rows
        if str(r["gold"]).upper() == gold
        and str(r["prediction"]).upper() == pred
    )


def total_errors(rows):

    return sum(
        1
        for r in rows
        if str(r["gold"]).upper()
        != str(r["prediction"]).upper()
    )


def bidirectional_count(rows, a, b):

    return (
        count_pair(rows, a, b)
        +
        count_pair(rows, b, a)
    )


# ============================================================
# 分析
# ============================================================

records = []


for name, (path, labels) in EXPERIMENTS.items():

    if not os.path.exists(path):
        print(f"[跳过] {path}")
        continue

    data, rows = load_rows(path)

    n = len(rows)
    errors = total_errors(rows)

    record = {
        "experiment": name,
        "sample_count": n,
        "error_count": errors,
        "error_rate": errors / n if n else 0,
    }

    # --------------------------------------------------------
    # Level-1
    # --------------------------------------------------------

    if labels == ["N", "R", "W"]:

        for group_name, pair in ERROR_GROUPS_LEVEL1.items():

            count = count_pair(
                rows,
                pair[0],
                pair[1]
            )

            record[f"{group_name}_count"] = count

            record[f"{group_name}_of_all_errors"] = (
                count / errors
                if errors
                else 0
            )

            record[f"{group_name}_of_gold"] = (
                count /
                sum(
                    1
                    for r in rows
                    if str(r["gold"]).upper() == pair[0]
                )
                if sum(
                    1
                    for r in rows
                    if str(r["gold"]).upper() == pair[0]
                )
                else 0
            )

    # --------------------------------------------------------
    # Level-2
    # --------------------------------------------------------

    else:

        pairs = [
            ("B_Y", "B", "Y"),
            ("B_C", "B", "C"),
            ("C_G", "C", "G"),
            ("B_G", "B", "G"),
            ("C_Y", "C", "Y"),
            ("G_Y", "G", "Y"),
        ]

        for pair_name, a, b in pairs:

            count = bidirectional_count(
                rows,
                a,
                b
            )

            record[
                f"{pair_name}_bidirectional_count"
            ] = count

            record[
                f"{pair_name}_of_all_errors"
            ] = (
                count / errors
                if errors
                else 0
            )

    records.append(record)


df = pd.DataFrame(records)


# ============================================================
# 保存
# ============================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "error_structure_quantitative.csv"
)

df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 打印 Level-1
# ============================================================

print("\n" + "=" * 80)
print("Level-1 错误结构")
print("=" * 80)

level1_df = df[
    df["experiment"].str.contains("Level1")
]

for _, row in level1_df.iterrows():

    print(
        f"\n{row['experiment']}"
        f"  总错误={row['error_count']}"
        f"  错误率={row['error_rate']:.2%}"
    )

    for key in [
        "W_to_N",
        "R_to_N",
        "N_to_W",
        "R_to_W",
        "W_to_R",
        "N_to_R",
    ]:

        count_col = f"{key}_count"

        if count_col in row:

            print(
                f"  {key}: "
                f"{int(row[count_col])} "
                f"= {row[f'{key}_of_all_errors']:.2%}"
                f" of all errors"
            )


# ============================================================
# 打印 Level-2
# ============================================================

print("\n" + "=" * 80)
print("Level-2 错误结构")
print("=" * 80)

level2_df = df[
    df["experiment"].str.contains("Level2")
]

for _, row in level2_df.iterrows():

    print(
        f"\n{row['experiment']}"
        f"  总错误={row['error_count']}"
        f"  错误率={row['error_rate']:.2%}"
    )

    for pair in [
        "B_Y",
        "B_C",
        "C_G",
        "B_G",
        "C_Y",
        "G_Y",
    ]:

        col = f"{pair}_bidirectional_count"

        if col in row:

            print(
                f"  {pair} 双向错误: "
                f"{int(row[col])} "
                f"= {row[f'{pair}_of_all_errors']:.2%}"
                f" of all errors"
            )


# ============================================================
# Few-shot变化
# ============================================================

print("\n" + "=" * 80)
print("Few-shot 错误结构变化")
print("=" * 80)


def show_change(zs_name, k3_name, k6_name, columns):

    available = df.set_index("experiment")

    if (
        zs_name not in available.index
        or k3_name not in available.index
        or k6_name not in available.index
    ):
        return

    print(f"\n{zs_name.split('_Level')[0]}")

    for col in columns:

        if col not in available.columns:
            continue

        zs = available.loc[zs_name, col]
        k3 = available.loc[k3_name, col]
        k6 = available.loc[k6_name, col]

        print(
            f"  {col}: "
            f"ZS={zs:.2%}, "
            f"k3={k3:.2%}, "
            f"k6={k6:.2%}"
        )


show_change(
    "DeepSeek_Level1_ZS",
    "DeepSeek_Level1_k3",
    "DeepSeek_Level1_k6",
    [
        "W_to_N_of_all_errors",
        "R_to_N_of_all_errors",
    ]
)

show_change(
    "QwenMax_Level1_ZS",
    "QwenMax_Level1_k3",
    "QwenMax_Level1_k6",
    [
        "W_to_N_of_all_errors",
        "R_to_N_of_all_errors",
    ]
)

show_change(
    "DeepSeek_Level2_ZS",
    "DeepSeek_Level2_k3",
    "DeepSeek_Level2_k6",
    [
        "B_Y_of_all_errors",
        "B_C_of_all_errors",
        "C_G_of_all_errors",
        "B_G_of_all_errors",
    ]
)


# ============================================================
# 完成
# ============================================================

print("\n" + "=" * 80)
print("定量错误分析完成")
print("=" * 80)

print(f"\n结果文件：{output_csv}")