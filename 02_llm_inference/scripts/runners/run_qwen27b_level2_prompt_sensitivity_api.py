import os
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# Qwen3.8-27B — Level-2 Prompt Sensitivity
# Formal Experiment
#
# P0: Existing Qwen3.8-Flash zero-shot result
# P1: Definition-only
# P2: Boundary-enhanced
# P3: Decision-tree
#
# 100 fixed validation samples:
# B/C/G/Y = 25/25/25/25
#
# New API calls:
# P1 × 100 + P2 × 100 + P3 × 100 = 300
#
# Checkpoint after every successful sample.
# ============================================================


# ============================================================
# Basic configuration
# ============================================================

BASE_DIR = Path(r"C:\Users\snowf\CAPC-CG-Project")

SAMPLE_PATH = (
    BASE_DIR
    / "02_llm_inference"
    / "results"
    / "prompt_sensitivity"
    / "prompt_sensitivity_100_samples.json"
)

OUTPUT_DIR = BASE_DIR / "02_llm_inference" / "results" / "prompt_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen3.8-27b"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

API_KEY_ENV = "DASHSCOPE_API_KEY"

MAX_TOKENS = 10
TIMEOUT = 60
MAX_RETRIES = 3

LABELS = ["B", "C", "G", "Y"]


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
# File paths
# ============================================================

CHECKPOINT_PATHS = {
    prompt_name: OUTPUT_DIR / f"qwen27b_prompt_{prompt_name}_checkpoint.json"
    for prompt_name in PROMPTS
}

RESULT_PATHS = {
    prompt_name: OUTPUT_DIR / f"qwen27b_prompt_{prompt_name}_100.json"
    for prompt_name in PROMPTS
}


# ============================================================
# API client
# ============================================================

def create_client():
    api_key = os.environ.get(API_KEY_ENV)

    if not api_key:
        raise RuntimeError(
            f"环境变量 {API_KEY_ENV} 不存在。"
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=TIMEOUT,
    )


# ============================================================
# Load fixed samples
# ============================================================

def load_samples():

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"找不到固定样本文件：\n{SAMPLE_PATH}"
        )

    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("固定样本文件结构不是 list。")

    if len(samples) != 100:
        raise ValueError(
            f"固定样本数量错误：{len(samples)}"
        )

    counts = {label: 0 for label in LABELS}

    for sample in samples:
        gold = sample["gold"]

        if gold not in counts:
            raise ValueError(
                f"发现未知标签：{gold}"
            )

        counts[gold] += 1

    if counts != {
        "B": 25,
        "C": 25,
        "G": 25,
        "Y": 25,
    }:
        raise ValueError(
            f"类别分布异常：{counts}"
        )

    return samples


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(path):

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        return data

    except Exception as e:
        print(f"WARNING: checkpoint 读取失败：{e}")
        return {}


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(path, results):

    tmp_path = path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp_path.replace(path)


# ============================================================
# Parse model response
# ============================================================

def parse_prediction(text):

    if text is None:
        return None

    text = str(text).strip().upper()

    # 去掉常见 Markdown / 标点
    cleaned = (
        text
        .replace("`", "")
        .replace("。", "")
        .replace(".", "")
        .replace("：", "")
        .replace(":", "")
        .strip()
    )

    # 最严格优先：完整等于标签
    if cleaned in LABELS:
        return cleaned

    # 如果模型多说了内容，只接受非常明确的最终标签形式
    tokens = []

    for label in LABELS:
        if label in cleaned:
            tokens.append(label)

    if len(tokens) == 1:
        return tokens[0]

    return None


# ============================================================
# Single API call
# ============================================================

def call_model(client, system_prompt, user_content):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        max_tokens=MAX_TOKENS,
        extra_body={
            "enable_thinking": False,
        },
    )

    raw_response = response.choices[0].message.content

    prediction = parse_prediction(raw_response)

    return raw_response, prediction


# ============================================================
# Run one prompt
# ============================================================

def run_prompt(prompt_name, prompt_text, samples, client):

    checkpoint_path = CHECKPOINT_PATHS[prompt_name]
    result_path = RESULT_PATHS[prompt_name]

    checkpoint = load_checkpoint(checkpoint_path)

    print("\n" + "=" * 70)
    print(f"开始实验：{prompt_name}")
    print(f"模型：{MODEL}")
    print(f"样本：100")
    print("=" * 70)

    success_count = 0
    failed_count = 0

    for sample_no, sample in enumerate(samples, start=1):

        idx = str(sample["index"])

        # ----------------------------------------------------
        # 已成功样本直接跳过
        # ----------------------------------------------------
        if idx in checkpoint:
            existing = checkpoint[idx]

            if existing.get("status") == "success":
                success_count += 1
                continue

        gold = sample["gold"]
        user_content = sample["user_content"]

        result = {
            "index": sample["index"],
            "gold": gold,
            "prediction": None,
            "raw_response": None,
            "status": "failed",
            "correct": False,
            "attempts": 0,
        }

        success = False

        for attempt in range(1, MAX_RETRIES + 1):

            result["attempts"] = attempt

            try:

                raw_response, prediction = call_model(
                    client,
                    prompt_text,
                    user_content,
                )

                result["raw_response"] = raw_response
                result["prediction"] = prediction

                if prediction in LABELS:

                    result["status"] = "success"
                    result["correct"] = (
                        prediction == gold
                    )

                    checkpoint[idx] = result

                    # 每成功一条立即保存
                    save_checkpoint(
                        checkpoint_path,
                        checkpoint,
                    )

                    success_count += 1
                    success = True

                    print(
                        f"[{sample_no:03d}/100] "
                        f"index={sample['index']} "
                        f"gold={gold} "
                        f"pred={prediction} "
                        f"{'✓' if prediction == gold else '✗'}"
                    )

                    break

                else:

                    print(
                        f"[{sample_no:03d}/100] "
                        f"index={sample['index']} "
                        f"无法解析标签："
                        f"{repr(raw_response)} "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )

            except Exception as e:

                print(
                    f"[{sample_no:03d}/100] "
                    f"index={sample['index']} "
                    f"API失败：{e} "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )

            if attempt < MAX_RETRIES:
                time.sleep(2)

        if not success:

            result["status"] = "failed"

            checkpoint[idx] = result

            # 失败也保存，便于后续识别
            save_checkpoint(
                checkpoint_path,
                checkpoint,
            )

            failed_count += 1

    # ========================================================
    # Build final results
    # ========================================================

    final_results = []

    for sample in samples:

        idx = str(sample["index"])

        if idx not in checkpoint:
            continue

        final_results.append(
            checkpoint[idx]
        )

    final_results.sort(
        key=lambda x: int(x["index"])
    )

    # ========================================================
    # Metrics
    # ========================================================

    valid_results = [
        x for x in final_results
        if x.get("status") == "success"
        and x.get("prediction") in LABELS
    ]

    failed_results = [
        x for x in final_results
        if x.get("status") != "success"
    ]

    y_true = [
        x["gold"]
        for x in valid_results
    ]

    y_pred = [
        x["prediction"]
        for x in valid_results
    ]

    metrics = {
        "valid_samples": len(valid_results),
        "failed_samples": len(failed_results),
        "accuracy": None,
        "macro_f1": None,
        "kappa": None,
        "classification_report": None,
        "confusion_matrix": None,
    }

    if valid_results:

        metrics["accuracy"] = accuracy_score(
            y_true,
            y_pred,
        )

        metrics["macro_f1"] = f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        metrics["kappa"] = cohen_kappa_score(
            y_true,
            y_pred,
            labels=LABELS,
        )

        metrics["classification_report"] = (
            classification_report(
                y_true,
                y_pred,
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            )
        )

        metrics["confusion_matrix"] = (
            confusion_matrix(
                y_true,
                y_pred,
                labels=LABELS,
            ).tolist()
        )

    output = {
        "experiment": f"Qwen3.8-27B Level-2 Prompt Sensitivity {prompt_name}",
        "model": MODEL,
        "prompt": prompt_name,
        "dataset": "Baron-Sun/CAPC-CG_V1.0",
        "task": "Level-2 B/C/G/Y",
        "setting": "zero-shot prompt sensitivity",
        "target_samples": 100,
        "valid_samples": metrics["valid_samples"],
        "failed_samples": metrics["failed_samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "kappa": metrics["kappa"],
        "classification_report": metrics[
            "classification_report"
        ],
        "confusion_matrix": metrics[
            "confusion_matrix"
        ],
        "label_order": LABELS,
        "results": final_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Print summary
    # ========================================================

    print("\n" + "-" * 70)
    print(f"{prompt_name} 完成")
    print("-" * 70)

    print(
        f"Valid: {metrics['valid_samples']}/100"
    )
    print(
        f"Failed: {metrics['failed_samples']}"
    )

    if metrics["accuracy"] is not None:

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Macro-F1 : {metrics['macro_f1']:.4f}"
        )

        print(
            f"Kappa    : {metrics['kappa']:.4f}"
        )

        print("\nConfusion Matrix")
        print("Labels:", LABELS)

        for row in metrics["confusion_matrix"]:
            print(row)

        print("\nPer-class F1")

        report = metrics["classification_report"]

        for label in LABELS:
            print(
                f"{label}: "
                f"P={report[label]['precision']:.4f} "
                f"R={report[label]['recall']:.4f} "
                f"F1={report[label]['f1-score']:.4f}"
            )

    print(
        f"\n结果文件：{result_path}"
    )


# ============================================================
# Compare P0/P1/P2/P3
# ============================================================

def build_comparison(samples):

    rows = []

    # --------------------------------------------------------
    # Existing P0
    # --------------------------------------------------------

    p0_true = [
        sample["gold"]
        for sample in samples
        if sample.get("P0_prediction") in LABELS
    ]

    p0_pred = [
        sample["P0_prediction"]
        for sample in samples
        if sample.get("P0_prediction") in LABELS
    ]

    if p0_true:

        rows.append({
            "Prompt": "P0",
            "Model": "Qwen3.8-Flash",
            "Valid": len(p0_true),
            "Accuracy": accuracy_score(
                p0_true,
                p0_pred,
            ),
            "Macro-F1": f1_score(
                p0_true,
                p0_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            "Kappa": cohen_kappa_score(
                p0_true,
                p0_pred,
                labels=LABELS,
            ),
        })

    # --------------------------------------------------------
    # P1/P2/P3
    # --------------------------------------------------------

    for prompt_name in ["P1", "P2", "P3"]:

        path = RESULT_PATHS[prompt_name]

        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])

        valid = [
            x for x in results
            if x.get("status") == "success"
            and x.get("prediction") in LABELS
        ]

        if not valid:
            continue

        y_true = [
            x["gold"]
            for x in valid
        ]

        y_pred = [
            x["prediction"]
            for x in valid
        ]

        rows.append({
            "Prompt": prompt_name,
            "Model": "Qwen3.8-27B",
            "Valid": len(valid),
            "Accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "Macro-F1": f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            "Kappa": cohen_kappa_score(
                y_true,
                y_pred,
                labels=LABELS,
            ),
        })

    return rows


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Qwen3.8-27B — Level-2 Prompt Sensitivity")
    print("正式实验：P1 / P2 / P3")
    print("=" * 70)

    samples = load_samples()

    print(
        f"\n固定测试集：{len(samples)}"
    )

    print(
        "类别：B=25, C=25, G=25, Y=25"
    )

    client = create_client()

    # --------------------------------------------------------
    # Run P1
    # --------------------------------------------------------

    run_prompt(
        "P1",
        P1,
        samples,
        client,
    )

    # --------------------------------------------------------
    # Run P2
    # --------------------------------------------------------

    run_prompt(
        "P2",
        P2,
        samples,
        client,
    )

    # --------------------------------------------------------
    # Run P3
    # --------------------------------------------------------

    run_prompt(
        "P3",
        P3,
        samples,
        client,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    rows = build_comparison(samples)

    if rows:

        comparison_path = (
            OUTPUT_DIR
            / "qwen27b_prompt_sensitivity_comparison.json"
        )

        with open(
            comparison_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                rows,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n" + "=" * 70)
        print("P0 / P1 / P2 / P3 对比")
        print("=" * 70)

        print(
            f"{'Prompt':<8}"
            f"{'Model':<18}"
            f"{'Valid':<8}"
            f"{'Accuracy':<12}"
            f"{'Macro-F1':<12}"
            f"{'Kappa':<10}"
        )

        print("-" * 70)

        for row in rows:

            print(
                f"{row['Prompt']:<8}"
                f"{row['Model']:<18}"
                f"{row['Valid']:<8}"
                f"{row['Accuracy']:<12.4f}"
                f"{row['Macro-F1']:<12.4f}"
                f"{row['Kappa']:<10.4f}"
            )

        print(
            f"\n对比文件：{comparison_path}"
        )


if __name__ == "__main__":
    main()