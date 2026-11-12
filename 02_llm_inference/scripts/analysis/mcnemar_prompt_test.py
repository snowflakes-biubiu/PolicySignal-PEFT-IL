import json
import os

from scipy.stats import binomtest


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "results",
        "prompt_sensitivity",
    )
)

FILES = {
    "P1": "qwen27b_prompt_P1_100.json",
    "P2": "qwen27b_prompt_P2_100.json",
    "P3": "qwen27b_prompt_P3_100.json",
}


def load_results(filename):
    path = os.path.join(BASE_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    if isinstance(results, dict):
        results = list(results.values())

    return sorted(results, key=lambda x: x["index"])


def compare(a, b):
    """
    McNemar 配对比较：

    b = A正确、B错误
    c = A错误、B正确
    """

    b = 0
    c = 0

    for x, y in zip(a, b_results):

        gold = x["gold"]

        correct_a = x["prediction"] == gold
        correct_b = y["prediction"] == gold

        if correct_a and not correct_b:
            b += 1

        elif not correct_a and correct_b:
            c += 1

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        # 双侧精确二项检验
        p_value = binomtest(
            b,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return b, c, p_value


def main():

    experiments = {
        name: load_results(filename)
        for name, filename in FILES.items()
    }

    pairs = [
        ("P1", "P2"),
        ("P1", "P3"),
        ("P2", "P3"),
    ]

    print("=" * 80)
    print("Prompt Sensitivity —— McNemar 配对显著性检验")
    print("模型：Qwen3.8-27B")
    print("样本：100")
    print("=" * 80)

    output = {}

    for name_a, name_b in pairs:

        global b_results
        b_results = experiments[name_b]

        a = experiments[name_a]

        b_count, c_count, p_value = compare(
            a,
            b_results,
        )

        output[f"{name_a}_vs_{name_b}"] = {
            "correct_to_wrong": b_count,
            "wrong_to_correct": c_count,
            "p_value": p_value,
        }

        print(f"\n{name_a} vs {name_b}")
        print("-" * 50)

        print(
            f"{name_a}正确 → {name_b}错误："
            f"{b_count}"
        )

        print(
            f"{name_a}错误 → {name_b}正确："
            f"{c_count}"
        )

        print(
            f"McNemar exact p-value："
            f"{p_value:.6f}"
        )

        if p_value < 0.05:
            print("结论：差异具有统计显著性（p < 0.05）")
        else:
            print("结论：未发现统计显著差异（p >= 0.05）")

    output_path = os.path.join(
        BASE_DIR,
        "qwen27b_prompt_mcnemar_test.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("检验完成")
    print(
        "结果：02_llm_inference\\results\\prompt_sensitivity\\"
        "qwen27b_prompt_mcnemar_test.json"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()