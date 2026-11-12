import json
from pathlib import Path
import pandas as pd


# ============================================================
# Qwen3.8-27B — Level-2 Prompt Sensitivity
# Step 1: Prepare fixed 100-sample evaluation set
# NOTE: This step DOES NOT call the API.
# ============================================================

BASE_DIR = Path(r"C:\Users\snowf\CAPC-CG-Project")

DATASET_PATH = Path(
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task2_level2\validation.parquet"
)

P0_RESULT_PATH = BASE_DIR / "02_llm_inference/results/qwen/qwen_level2_zero_shot_400.json"

OUTPUT_DIR = BASE_DIR / "02_llm_inference" / "results" / "prompt_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = OUTPUT_DIR / "prompt_sensitivity_100_samples.json"


# ============================================================
# Prompt definitions
# ============================================================

P1 = """角色：你是一名“政策信号标注员”。

任务：对给定的中文政策文档段落进行分类标注。该段落已经被判定为含有肯定性政策指令，请判断它属于以下哪一类：

B（Black，你可以）：
明确同意、批准或中性部署，例如“建立”“推进”“构建”“实现”等。指令明确，但没有明显强制性，也没有因地制宜的灵活路径或明显条件限制。

Y（Yellow，你必须）：
带有明显压力或紧迫性，要求全面落实或严格执行，例如“抓紧”“尽快”“及时”“必须”“严格执行”“全面落实”等。纳入考核、问责、审计也属于 Y。

C（Charcoal，你应当灵活地做）：
指令明确，但强调因地制宜、多种形式、试点示范、先行先试、借鉴与反馈等灵活路径或差异化方式。

G（Grey，你可能做也可能不做）：
指令含糊、模糊或带有保留条件，例如“在……前提下”“适度”“具备条件的”“可”“鼓励探索”“逐步推进”“协商解决”等，主体或执行强度不清晰。

输出要求：只输出最终的结果 B/C/G/Y。

待判断政策内容如下：
"""


P2 = """角色：你是一名“政策信号标注员”。

任务：对给定的中文政策文档段落进行分类标注。该段落已经被判定为含有肯定性政策指令，请判断它属于 B、C、G、Y 中的哪一类。

【标签定义】

B（Black，你可以）：
明确同意、批准或中性部署，例如“建立”“推进”“构建”“实现”等。指令明确，但没有明显强制性、灵活性或条件限制。

Y（Yellow，你必须）：
带有明显压力或紧迫性，要求全面落实或严格执行，例如“抓紧”“尽快”“及时”“必须”“严格执行”“全面落实”等。纳入考核、问责、审计也属于 Y。

C（Charcoal，你应当灵活地做）：
指令明确，但强调因地制宜、多种形式、试点示范、先行先试、借鉴与反馈等灵活路径或差异化方式。

G（Grey，你可能做也可能不做）：
指令含糊、模糊或带有保留条件，例如“在……前提下”“适度”“具备条件的”“可”“鼓励探索”“逐步推进”“协商解决”等，主体或执行强度不清晰。

【重要边界】

1. B 与 Y：
如果只是明确部署、建立、推进或实施，而没有明显强制性、紧迫性或严格执行要求，则优先判断为 B。
如果存在“必须、严格、抓紧、尽快、全面落实、考核、问责”等明显压力或强制执行要求，则优先判断为 Y。

2. B 与 C：
如果要求明确实施，但没有强调差异化路径、试点或因地制宜，则判断为 B。
如果明确要求因地制宜、试点示范、先行先试、采用多种形式或根据实际情况调整，则判断为 C。

3. B 与 G：
如果指令明确、主体和执行要求清晰，则判断为 B。
如果存在明显条件限制、可能性表达、适度要求、逐步推进或主体/强度不清，则判断为 G。

4. C 与 G：
如果已经明确要求采用灵活、差异化、试点或因地制宜的方式，则判断为 C。
如果只是表达“可以”“鼓励”“逐步”等不确定或保留条件，而没有明确的灵活执行路径，则判断为 G。

5. “必须”不能机械判断为 Y。
如果“必须”只是资格、条件或前置要求，例如“必须符合条件方可申请”，而不是对政策执行提出强制要求，则不能仅凭“必须”判断为 Y。

【判断原则】

不要只根据关键词进行分类。必须结合：
- 政策主体；
- 指令强度；
- 是否存在强制性；
- 是否存在灵活执行路径；
- 是否存在条件限制；
- 整个句子的实际政策功能。

输出要求：只输出最终的结果 B/C/G/Y。

待判断政策内容如下：
"""


P3 = """角色：你是一名“政策信号标注员”。

任务：对给定的中文政策文档段落进行分类标注。该段落已经被判定为含有肯定性政策指令，请根据下面的决策流程判断其属于 B、C、G、Y 中的哪一类。

【第一步：判断是否存在明显强制或严格执行要求】

如果段落要求全面落实、严格执行、必须完成、抓紧实施、尽快完成，或者明确涉及考核、问责、审计等强制执行压力，则判断为：

Y

但如果“必须”只是资格条件或前置条件，例如“必须符合条件方可申请”，而不是要求政策执行，则不要仅凭“必须”判断为 Y。

如果不属于 Y，继续第二步。

【第二步：判断是否存在明确的灵活执行路径】

如果段落明确要求：
- 因地制宜；
- 试点示范；
- 先行先试；
- 多种形式；
- 根据实际情况调整；
- 借鉴经验并根据反馈调整；

且政策指令本身是明确的，则判断为：

C

如果不属于 C，继续第三步。

【第三步：判断指令是否明确】

如果政策要求明确，主体明确，执行方向明确，并且没有明显强制压力、灵活路径或条件限制，则判断为：

B

如果不满足上述条件，继续第四步。

【第四步：判断是否存在模糊、条件性或保留性表达】

如果存在：
- “可以”
- “鼓励”
- “逐步”
- “适度”
- “在……前提下”
- “具备条件的”
- “协商解决”
- “视情况”
- 或其他导致主体、执行强度或执行条件不明确的表达，

则判断为：

G

【最终判断原则】

Y：强制、紧迫、严格执行。
C：明确指令 + 灵活/差异化执行路径。
B：明确、一般性的中性部署。
G：模糊、条件性或保留性的指令。

不要只看关键词，要结合整个政策段落的实际含义。

输出要求：只输出最终的结果 B/C/G/Y。

待判断政策内容如下：
"""


PROMPTS = {
    "P1": P1,
    "P2": P2,
    "P3": P3,
}


# ============================================================
# Helper
# ============================================================

def extract_messages(row):
    messages = row["messages"]

    system_prompt = str(messages[0]["content"])
    user_content = str(messages[1]["content"])
    gold = str(messages[2]["content"]).strip()

    return system_prompt, user_content, gold


def load_p0_results():
    if not P0_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"\n找不到已有 P0 结果文件：\n{P0_RESULT_PATH}\n"
            "请确认 02_llm_inference/results/qwen/qwen_level2_zero_shot_400.json 在项目目录中。"
        )

    with open(P0_RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"P0 JSON 顶层结构异常：{type(data)}"
        )

    if "results" not in data:
        raise ValueError(
            "P0 JSON 中没有找到 'results' 字段。"
        )

    results = data["results"]

    # 实际文件结构：
    # "results": {
    #     "0": {"index": 0, "gold": "B", "prediction": "G", ...},
    #     "1": {...},
    #     ...
    # }

    if isinstance(results, dict):
        results = list(results.values())

    if not isinstance(results, list):
        raise ValueError(
            f"P0 的 results 字段类型异常：{type(results)}"
        )

    print(f"P0 results records: {len(results)}")

    return results


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Qwen3.8-27B — Level-2 Prompt Sensitivity")
    print("Step 1: 固定 100 条测试样本 + 对齐已有 P0")
    print("本步骤不会调用 API")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load validation set
    # --------------------------------------------------------
    print("\n[1/4] 读取 Level-2 validation...")

    df = pd.read_parquet(DATASET_PATH)

    print(f"validation shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Select first 25 of each class
    # --------------------------------------------------------
    print("\n[2/4] 构造固定 100 条分层样本...")

    selected = []
    class_counts = {"B": 0, "C": 0, "G": 0, "Y": 0}

    for idx, row in df.iterrows():

        _, user_content, gold = extract_messages(row)

        if gold not in class_counts:
            continue

        if class_counts[gold] < 25:
            selected.append({
                "index": int(idx),
                "gold": gold,
                "user_content": user_content,
            })
            class_counts[gold] += 1

        if all(v == 25 for v in class_counts.values()):
            break

    if len(selected) != 100:
        raise RuntimeError(
            f"样本数量错误：{len(selected)}，"
            f"类别统计：{class_counts}"
        )

    print("固定样本类别分布：")
    for label, count in class_counts.items():
        print(f"  {label}: {count}")

    # --------------------------------------------------------
    # 3. Load existing P0
    # --------------------------------------------------------
    print("\n[3/4] 读取已有 Qwen3.8-Flash P0 结果...")

    p0_results = load_p0_results()

    print(f"P0 result records: {len(p0_results)}")

    # 建立 index -> result 映射
    p0_by_index = {}

    for item in p0_results:

        if not isinstance(item, dict):
            continue

        if "index" not in item:
            continue

        try:
            idx = int(item["index"])
        except Exception:
            continue

        p0_by_index[idx] = item

    print(f"P0 可按 index 对齐记录: {len(p0_by_index)}")

    # --------------------------------------------------------
    # 4. Verify alignment
    # --------------------------------------------------------
    print("\n[4/4] 检查 100 条样本与 P0 是否完全对齐...")

    missing_p0 = []

    final_samples = []

    for item in selected:

        idx = item["index"]

        if idx not in p0_by_index:
            missing_p0.append(idx)
            continue

        p0 = p0_by_index[idx]

        sample = {
            "index": idx,
            "gold": item["gold"],
            "user_content": item["user_content"],

            # Existing P0 result
            "P0_prediction": p0.get("prediction"),
            "P0_raw_response": p0.get("raw_response"),

            # New prompts
            "P1_prompt": P1,
            "P2_prompt": P2,
            "P3_prompt": P3,

            # Reserved for API experiment
            "P1_prediction": None,
            "P2_prediction": None,
            "P3_prediction": None,
        }

        final_samples.append(sample)

    if missing_p0:
        print("\nWARNING：以下样本没有找到 P0：")
        print(missing_p0)

        raise RuntimeError(
            f"\nP0 无法与固定 100 条样本完全对齐。"
            f"\n缺失数量：{len(missing_p0)}"
            "\n为了避免实验样本不一致，本步骤停止，不调用 API。"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            final_samples,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("数据准备完成")
    print("=" * 70)

    print(f"固定样本：{len(final_samples)}")
    print(f"P0 对齐：100/100")
    print(f"P1：准备完成")
    print(f"P2：准备完成")
    print(f"P3：准备完成")

    print("\n输出文件：")
    print(SAMPLE_PATH)

    print("\n类别分布：")
    for label in ["B", "C", "G", "Y"]:
        count = sum(x["gold"] == label for x in final_samples)
        print(f"  {label}: {count}")

    print("\n前 5 条样本检查：")
    for x in final_samples[:5]:
        print(
            f"index={x['index']} | "
            f"gold={x['gold']} | "
            f"P0={x['P0_prediction']}"
        )

    print("\n下一步：")
    print("确认这个文件生成正常后，再运行 P1/P2/P3 API 实验。")
    print("本次没有产生新的 API 调用。")


if __name__ == "__main__":
    main()