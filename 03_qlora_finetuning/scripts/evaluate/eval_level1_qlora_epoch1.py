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

from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# 配置
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


ADAPTER_DIR = (
    Path("03_qlora_finetuning/artifacts/models")
    / "qwen15b_level1_qlora"
    / "checkpoint-150"
)

MAX_LENGTH = 768

LABELS = ["N", "R", "W"]

RESULT_DIR = (
    Path("03_qlora_finetuning/artifacts/models")
    / "results"
)


RESULT_JSON = (
    RESULT_DIR
    / "level1_qlora_epoch1_validation_300.json"
)


RESULT_CSV = (
    RESULT_DIR
    / "level1_qlora_epoch1_validation_300.csv"
)


# ============================================================
# 1. 数据
# ============================================================

print("=" * 70)
print("Level-1 QLoRA 正式 Validation 评估")
print("=" * 70)

print()
print("[1/6] 定位验证集...")

snapshot_dir = Path(
    snapshot_download(
        "Baron-Sun/CAPC-CG_V1.0",
        repo_type="dataset",
        local_files_only=True,
    )
)

validation_file = (
    snapshot_dir
    / "task1_level1"
    / "validation.parquet"
)

df = pd.read_parquet(validation_file)

print(f"验证集样本数: {len(df)}")

if len(df) != 300:
    raise ValueError(
        f"验证集不是预期的300条，而是 {len(df)} 条"
    )


# ============================================================
# 2. Tokenizer
# ============================================================

print()
print("[2/6] 加载 Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 3. 检查标签 token
# ============================================================

print()
print("[3/6] 检查标签 Token...")

label_token_ids = {}

for label in LABELS:

    ids = tokenizer(
        label,
        add_special_tokens=False,
    )["input_ids"]

    if len(ids) != 1:
        raise ValueError(
            f"标签 {label} 不是单 token: {ids}"
        )

    label_token_ids[label] = ids[0]

    print(
        f"{label}: token_id={ids[0]}"
    )


# ============================================================
# 4. 加载 Base Model + LoRA Adapter
# ============================================================

print()
print("[4/6] 加载 Qwen2.5-1.5B + LoRA Adapter...")

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"找不到 Adapter: {ADAPTER_DIR}"
    )

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

print("模型加载完成")


# ============================================================
# 5. 单条预测函数
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

    # 与训练时保持一致
    if len(prompt_ids) > MAX_LENGTH:
        prompt_ids = prompt_ids[-MAX_LENGTH:]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=model.device,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # 最后一个位置预测下一个 token
    next_token_logits = outputs.logits[
        0,
        -1,
    ]

    scores = {
        label: float(
            next_token_logits[token_id].item()
        )
        for label, token_id
        in label_token_ids.items()
    }

    predicted_label = max(
        scores,
        key=scores.get,
    )

    return predicted_label, scores


# ============================================================
# 6. 正式推理
# ============================================================

print()
print("[5/6] 开始 300 条正式推理...")
print()

results = []

correct = 0

for i, row in df.iterrows():

    gold = str(
        row["messages"][2]["content"]
    ).strip()

    predicted, scores = predict_one(row)

    is_correct = (
        predicted == gold
    )

    if is_correct:
        correct += 1

    results.append(
        {
            "index": int(i),
            "gold": gold,
            "prediction": predicted,
            "correct": is_correct,
            "score_N": scores["N"],
            "score_R": scores["R"],
            "score_W": scores["W"],
        }
    )

    if (i + 1) % 25 == 0:

        current_acc = (
            correct / (i + 1)
        )

        print(
            f"[{i + 1:3d}/300] "
            f"当前 Accuracy = "
            f"{current_acc:.4f}"
        )


# ============================================================
# 7. 计算正式指标
# ============================================================

print()
print("[6/6] 计算正式指标...")

y_true = [
    x["gold"]
    for x in results
]

y_pred = [
    x["prediction"]
    for x in results
]

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


# ============================================================
# 8. 输出结果
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

result_data = {
    "experiment": "Qwen2.5-1.5B QLoRA Level-1",
    "model": BASE_MODEL,
    "adapter": str(ADAPTER_DIR),
    "validation_samples": len(df),
    "max_length": MAX_LENGTH,
    "labels": LABELS,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "cohen_kappa": kappa,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "predictions": results,
}

with open(
    RESULT_JSON,
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
# 9. 打印正式结果
# ============================================================

print()
print("=" * 70)
print("Level-1 QLoRA Validation 正式结果")
print("=" * 70)

print(
    f"Validation: {len(df)}"
)

print(
    f"Accuracy:   {accuracy:.4f}"
)

print(
    f"Macro-F1:   {macro_f1:.4f}"
)

print(
    f"Kappa:      {kappa:.4f}"
)

print()
print("Confusion Matrix")
print(
    "行 = Gold，列 = Prediction"
)

print(
    "Labels:",
    LABELS,
)

print(cm)

print()
print("Classification Report")
print()

for label in LABELS:

    print(
        f"{label}: "
        f"Precision={report[label]['precision']:.4f}, "
        f"Recall={report[label]['recall']:.4f}, "
        f"F1={report[label]['f1-score']:.4f}"
    )

print()
print(
    f"正确: {sum(y_true[i] == y_pred[i] for i in range(len(y_true)))}"
)

print(
    f"错误: {sum(y_true[i] != y_pred[i] for i in range(len(y_true)))}"
)

print()
print(
    f"JSON结果: {RESULT_JSON}"
)

print(
    f"CSV结果:  {RESULT_CSV}"
)

print("=" * 70)
print("评估完成")
print("=" * 70)