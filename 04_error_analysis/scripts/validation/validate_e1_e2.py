from pathlib import Path
import gc
import json
import re

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# 1. 配置
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["N", "R", "W"]
MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 4

DATA_DIR = Path("04_error_analysis") / "data" / "split_v2"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"
RESULTS_ROOT = Path("04_error_analysis") / "results" / "validation_results"

EXPERIMENTS = {
    "E1_baseline_split_v2": (
        Path("04_error_analysis")
        / "artifacts"
        / "baseline_split_v2"
        / "qwen15b_level1_qlora_rank16"
    ),
    "E2_A_N_weighted": (
        Path("04_error_analysis")
        / "artifacts"
        / "e2_class_weighted"
        / "E2_A_N_weighted"
        / "adapter"
    ),
    "E2_B_NW_weighted": (
        Path("04_error_analysis")
        / "artifacts"
        / "e2_class_weighted"
        / "E2_B_NW_weighted"
        / "adapter"
    ),
}


# ============================================================
# 2. 读取并检查验证集
# ============================================================

def get_gold_label(messages):
    return str(messages[-1]["content"]).strip()


def load_validation_data():
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"验证集不存在：{VALIDATION_FILE}"
        )

    df = pd.read_json(VALIDATION_FILE, lines=True)

    if len(df) != 225:
        raise ValueError(
            f"验证集应为 225 条，实际为 {len(df)} 条"
        )

    labels = [
        get_gold_label(messages)
        for messages in df["messages"]
    ]

    counts = pd.Series(labels).value_counts().to_dict()
    expected = {"N": 75, "R": 75, "W": 75}

    print(f"验证集样本数：{len(df)}")
    print(f"验证集标签分布：{counts}")

    if counts != expected:
        raise ValueError(
            f"验证集标签分布不符合预期：{counts}"
        )

    if not set(labels).issubset(set(LABELS)):
        raise ValueError("验证集存在 N/R/W 之外的标签")

    return df


# ============================================================
# 3. 加载 Tokenizer 和基础模型
# ============================================================

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer 缺少 pad_token 和 eos_token"
            )
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA GPU，请确认当前 Python 环境和显卡配置"
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = True
    return model


# ============================================================
# 4. 生成预测
# ============================================================

def get_prompt_messages(messages):
    return [
        {
            "role": "system",
            "content": messages[0]["content"],
        },
        {
            "role": "user",
            "content": messages[1]["content"],
        },
    ]


def parse_prediction(text):
    # 只接受独立出现的 N、R、W，避免从其他单词中误识别。
    matches = re.findall(
        r"(?<![A-Za-z])[NRW](?![A-Za-z])",
        text.strip(),
    )

    if not matches:
        return "INVALID"

    # 模型应只输出一个标签；如果输出多个标签，不擅自猜测。
    if len(matches) != 1:
        return "INVALID"

    return matches[0]


def predict_one(model, tokenizer, messages):
    prompt_messages = get_prompt_messages(messages)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        add_special_tokens=False,
    )

    input_device = model.get_input_embeddings().weight.device
    encoded = {
        key: value.to(input_device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = encoded["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_length:]

    raw_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return parse_prediction(raw_text), raw_text


# ============================================================
# 5. 评估单个实验
# ============================================================

def evaluate_experiment(name, adapter_dir, df, tokenizer):
    adapter_config = adapter_dir / "adapter_config.json"

    if not adapter_config.exists():
        raise FileNotFoundError(
            f"{name} 的 Adapter 不存在或未保存完整：{adapter_dir}"
        )

    print("\n" + "=" * 70)
    print(f"开始评估：{name}")
    print(f"Adapter 路径：{adapter_dir}")
    print("=" * 70)

    result_dir = RESULTS_ROOT / name
    result_dir.mkdir(parents=True, exist_ok=True)

    model = load_base_model()

    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
    )
    model.eval()

    records = []
    y_true = []
    y_pred = []

    for index, messages in enumerate(df["messages"], start=1):
        gold = get_gold_label(messages)
        pred, raw_output = predict_one(
            model,
            tokenizer,
            messages,
        )

        y_true.append(gold)
        y_pred.append(pred)

        records.append({
            "sample_index": index - 1,
            "gold_label": gold,
            "predicted_label": pred,
            "correct": pred == gold,
            "raw_generated_text": raw_output,
        })

        if index % 25 == 0 or index == len(df):
            print(f"推理进度：{index}/{len(df)}")

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=LABELS,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    prediction_df = pd.DataFrame(records)
    prediction_df.to_csv(
        result_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )

    confusion_df.to_csv(
        result_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    summary = {
        "experiment": name,
        "adapter_dir": str(adapter_dir),
        "validation_file": str(VALIDATION_FILE),
        "sample_count": len(y_true),
        "invalid_prediction_count": sum(
            label == "INVALID" for label in y_pred
        ),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "cohen_kappa": float(kappa),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix_label_order": LABELS,
        "confusion_matrix": cm.tolist(),
    }

    with open(
        result_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nAccuracy：{accuracy:.4%}")
    print(f"Macro-F1：{macro_f1:.4f}")
    print(f"Cohen's Kappa：{kappa:.4f}")
    print(f"非法输出数量：{summary['invalid_prediction_count']}")

    print("\n各类别指标：")
    for label in LABELS:
        item = summary["per_class"][label]
        print(
            f"{label}: "
            f"Precision={item['precision']:.4f}, "
            f"Recall={item['recall']:.4f}, "
            f"F1={item['f1']:.4f}"
        )

    print("\n混淆矩阵（行=真实标签，列=预测标签，顺序 N/R/W）：")
    print(confusion_df)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ============================================================
# 6. 依次评估 E1、E2-A、E2-B
# ============================================================

def main():
    print("=" * 70)
    print("E1 / E2 统一验证")
    print("=" * 70)

    df = load_validation_data()
    tokenizer = load_tokenizer()

    summaries = []

    for name, adapter_dir in EXPERIMENTS.items():
        if not adapter_dir.exists():
            print(f"\n[跳过] {name} 尚未找到 Adapter：{adapter_dir}")
            continue

        summary = evaluate_experiment(
            name=name,
            adapter_dir=adapter_dir,
            df=df,
            tokenizer=tokenizer,
        )
        summaries.append(summary)

    if not summaries:
        raise RuntimeError(
            "没有找到可评估的 Adapter。请检查训练是否完成及保存路径。"
        )

    summary_rows = []

    for item in summaries:
        row = {
            "experiment": item["experiment"],
            "sample_count": item["sample_count"],
            "accuracy": item["accuracy"],
            "macro_f1": item["macro_f1"],
            "cohen_kappa": item["cohen_kappa"],
            "invalid_prediction_count": item["invalid_prediction_count"],
        }

        for label in LABELS:
            row[f"{label}_precision"] = item["per_class"][label]["precision"]
            row[f"{label}_recall"] = item["per_class"][label]["recall"]
            row[f"{label}_f1"] = item["per_class"][label]["f1"]

        summary_rows.append(row)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        RESULTS_ROOT / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("所有可用实验评估完成")
    print(summary_df.to_string(index=False))
    print(f"\n汇总文件：{RESULTS_ROOT / 'comparison_summary.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()