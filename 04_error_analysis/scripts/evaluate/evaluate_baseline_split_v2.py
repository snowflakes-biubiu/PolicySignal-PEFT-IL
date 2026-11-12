
from pathlib import Path
import json

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# 0. 实验配置
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# 新数据划分
DATA_DIR = Path("04_error_analysis") / "data" / "split_v2"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"

# 新基准模型的 Adapter 路径
# 注意：必须与新训练脚本中的 OUTPUT_DIR 一致
ADAPTER_DIR = (
    Path("04_error_analysis")
    / "artifacts"
    / "baseline_split_v2"
    / "qwen15b_level1_qlora_rank16"
)

MAX_LENGTH = 768

LABELS = ["N", "R", "W"]

# 新实验独立保存评估结果
RESULT_DIR = (
    Path("04_error_analysis")
    / "results"
    / "baseline_split_v2"
    / "evaluation"
)

RESULT_JSON = (
    RESULT_DIR
    / "level1_rank16_split_v2_validation_225.json"
)

RESULT_CSV = (
    RESULT_DIR
    / "level1_rank16_split_v2_validation_225.csv"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 环境与文件检查
# ============================================================

print("=" * 72)
print("Level-1 QLoRA 基准模型评估：split_v2")
print("=" * 72)

print(f"基础模型：{BASE_MODEL}")
print(f"验证集：{VALIDATION_FILE}")
print(f"Adapter：{ADAPTER_DIR}")
print(f"输出目录：{RESULT_DIR}")
print(f"设备：{DEVICE}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "未检测到 CUDA GPU。本脚本按当前 QLoRA 配置要求使用 GPU，"
        "请检查 PyTorch 和 CUDA 环境。"
    )

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"找不到验证集：{VALIDATION_FILE}"
    )

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"找不到新基准模型 Adapter：{ADAPTER_DIR}\n"
        "请先完成 split_v2 基准模型训练。"
    )


# ============================================================
# 2. 加载并检查验证集
# ============================================================

print("\n[1/6] 加载验证集...")

df = pd.read_json(
    VALIDATION_FILE,
    lines=True,
)

if len(df) != 225:
    raise ValueError(
        f"验证集应为 225 条，实际为 {len(df)} 条。"
    )

required_columns = {"messages"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"验证集缺少必要字段：{required_columns - set(df.columns)}"
    )


def extract_label(messages):
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(
            f"消息格式异常：{messages}"
        )

    return str(
        messages[-1]["content"]
    ).strip()


df["gold"] = df["messages"].apply(extract_label)

label_counts = df["gold"].value_counts().to_dict()

print(f"验证集样本数：{len(df)}")
print(f"标签分布：{label_counts}")

expected_counts = {
    "N": 75,
    "R": 75,
    "W": 75,
}

if label_counts != expected_counts:
    raise ValueError(
        f"验证集标签分布不符合预期：{label_counts}"
    )

if not set(df["gold"]).issubset(set(LABELS)):
    raise ValueError("验证集中发现未知标签。")


# ============================================================
# 3. 加载 Tokenizer 并检查标签 Token
# ============================================================

print("\n[2/6] 加载 Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

label_token_ids = {}

for label in LABELS:
    token_ids = tokenizer(
        label,
        add_special_tokens=False,
    )["input_ids"]

    if len(token_ids) != 1:
        raise ValueError(
            f"标签 {label} 不是单个 Token：{token_ids}"
        )

    label_token_ids[label] = token_ids[0]

    print(
        f"标签 {label}：token_id={token_ids[0]}"
    )


# ============================================================
# 4. 加载 Base Model + LoRA Adapter
# ============================================================

print("\n[3/6] 加载 Qwen2.5-1.5B 与 Adapter...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

model.eval()

print("模型加载完成。")


# ============================================================
# 5. 单条样本预测
# ============================================================

def predict_one(row):
    messages = row["messages"]

    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    prompt_messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    # 与训练时保持一致：
    # 保留 Prompt 尾部，并为预测标签预留位置。
    # 本任务的 N/R/W 标签均为单 Token。
    if len(prompt_ids) > MAX_LENGTH - 1:
        prompt_ids = prompt_ids[-(MAX_LENGTH - 1):]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=model.get_input_embeddings().weight.device,
    )

    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    next_token_logits = outputs.logits[0, -1, :]

    scores = {
        label: float(
            next_token_logits[token_id].float().item()
        )
        for label, token_id in label_token_ids.items()
    }

    predicted_label = max(
        scores,
        key=scores.get,
    )

    return predicted_label, scores


# ============================================================
# 6. 执行预测
# ============================================================

print("\n[4/6] 开始 225 条样本推理...")

results = []

for position, (_, row) in enumerate(
    df.iterrows(),
    start=1,
):
    gold = row["gold"]

    predicted, scores = predict_one(row)

    results.append(
        {
            "index": int(position - 1),
            "gold": gold,
            "prediction": predicted,
            "correct": predicted == gold,
            "score_N": scores["N"],
            "score_R": scores["R"],
            "score_W": scores["W"],
        }
    )

    if position % 25 == 0 or position == len(df):
        current_correct = sum(
            item["correct"] for item in results
        )
        current_accuracy = current_correct / position

        print(
            f"[{position:3d}/{len(df)}] "
            f"当前 Accuracy={current_accuracy:.4f}"
        )


# ============================================================
# 7. 计算指标
# ============================================================

print("\n[5/6] 计算分类指标...")

y_true = [item["gold"] for item in results]
y_pred = [item["prediction"] for item in results]

accuracy = accuracy_score(
    y_true,
    y_pred,
)

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

correct_count = sum(
    item["correct"] for item in results
)

error_count = len(results) - correct_count


# ============================================================
# 8. 保存结果
# ============================================================

print("\n[6/6] 保存评估结果...")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

result_data = {
    "experiment": "Level-1 QLoRA Rank16 split_v2 baseline",
    "base_model": BASE_MODEL,
    "adapter": str(ADAPTER_DIR),
    "validation_file": str(VALIDATION_FILE),
    "validation_samples": len(df),
    "label_distribution": label_counts,
    "max_length": MAX_LENGTH,
    "labels": LABELS,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "cohen_kappa": kappa,
    "confusion_matrix_label_order": LABELS,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "correct_count": correct_count,
    "error_count": error_count,
    "predictions": results,
}

with RESULT_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result_data,
        f,
        ensure_ascii=False,
        indent=2,
    )

pd.DataFrame(results).to_csv(
    RESULT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 9. 打印结果
# ============================================================

print("\n" + "=" * 72)
print("Level-1 QLoRA Rank16 split_v2 基准评估结果")
print("=" * 72)

print(f"验证集样本数：{len(df)}")
print(f"正确分类：{correct_count}")
print(f"错误分类：{error_count}")
print(f"Accuracy：{accuracy:.4f}")
print(f"Macro-F1：{macro_f1:.4f}")
print(f"Cohen's Kappa：{kappa:.4f}")

print("\n混淆矩阵（行=真实标签，列=预测标签）")
print(f"标签顺序：{LABELS}")
print(cm)

print("\n各类别指标")

for label in LABELS:
    print(
        f"{label}: "
        f"Precision={report[label]['precision']:.4f}, "
        f"Recall={report[label]['recall']:.4f}, "
        f"F1={report[label]['f1-score']:.4f}"
    )

print("\n结果文件：")
print(f"JSON：{RESULT_JSON}")
print(f"CSV： {RESULT_CSV}")

print("=" * 72)
print("评估完成")
print("=" * 72)
