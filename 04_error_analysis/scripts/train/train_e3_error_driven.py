from pathlib import Path

import pandas as pd
import torch

from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 实验配置
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_LENGTH = 768

#LORA_R = 8
#LORA_ALPHA = 16
#LORA_DROPOUT = 0.05
#
#LEARNING_RATE = 2e-4
#NUM_EPOCHS = 3
#
#BATCH_SIZE = 1
#GRADIENT_ACCUMULATION = 4
#
#SEED = 42
#
#OUTPUT_DIR = Path(
#    "04_error_analysis/artifacts/e3_error_driven"
#) / "qwen15b_level1_qlora_3epoch"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

SEED = 42

OUTPUT_DIR = (
    Path("04_error_analysis")
    / "artifacts"
    / "baseline_split_v2"
    / "qwen15b_level1_qlora_rank16"
)

# ============================================================
# 1. 基础环境
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("Qwen2.5-1.5B Level-1 QLoRA 正式训练")
print("=" * 70)

print(f"模型: {MODEL_NAME}")
print(f"MAX_LENGTH: {MAX_LENGTH}")
print(f"LoRA r: {LORA_R}")
print(f"LoRA alpha: {LORA_ALPHA}")
print(f"LoRA dropout: {LORA_DROPOUT}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"Seed: {SEED}")

# ============================================================
# 2. 数据路径
# ============================================================

#print()
#print("[1/7] 定位数据集...")
#
#snapshot_dir = Path(
#    snapshot_download(
#        "Baron-Sun/CAPC-CG_V1.0",
#        repo_type="dataset",
#        local_files_only=True,
#    )
#)
#
#train_file = (
#    snapshot_dir
#    / "task1_level1"
#    / "train.parquet"
#)
#
#validation_file = (
#    snapshot_dir
#    / "task1_level1"
#    / "validation.parquet"
#)
#
#train_df = pd.read_parquet(train_file)
#validation_df = pd.read_parquet(validation_file)
#
#print(f"训练集: {len(train_df)}")
#print(f"验证集: {len(validation_df)}")
#

# ============================================================
# 2. 数据路径：split_v2
# ============================================================

print()
print("[1/7] 加载 split_v2 数据集...")

DATA_DIR = Path("04_error_analysis") / "data" / "split_v2"

train_file = DATA_DIR / "train.jsonl"
validation_file = DATA_DIR / "validation.jsonl"

if not train_file.exists():
    raise FileNotFoundError(f"训练集不存在：{train_file}")

if not validation_file.exists():
    raise FileNotFoundError(f"验证集不存在：{validation_file}")

train_df = pd.read_json(train_file, lines=True)
validation_df = pd.read_json(validation_file, lines=True)

print(f"训练集：{len(train_df)}")
print(f"验证集：{len(validation_df)}")

def extract_label(messages):
    return str(messages[-1]["content"]).strip()

train_counts = train_df["messages"].apply(extract_label).value_counts().to_dict()
validation_counts = validation_df["messages"].apply(extract_label).value_counts().to_dict()

print(f"训练集标签分布：{train_counts}")
print(f"验证集标签分布：{validation_counts}")

if len(train_df) != 675 or train_counts != {"W": 225, "R": 225, "N": 225}:
    raise ValueError("训练集数量或标签分布不符合预期")

if len(validation_df) != 225 or validation_counts != {"W": 75, "R": 75, "N": 75}:
    raise ValueError("验证集数量或标签分布不符合预期")

## ============================================================
## 3. Tokenizer
## ============================================================
#
#print()
#print("[2/7] 加载 Tokenizer...")
#
#tokenizer = AutoTokenizer.from_pretrained(
#    MODEL_NAME,
#    trust_remote_code=True,
#)
#
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
#
#print("Tokenizer 加载成功")
# ============================================================
# 3. 加载 Tokenizer
# ============================================================

print()
print("[2/7] 加载 Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer 加载成功")

LABELS = ["N", "R", "W"]

for label in LABELS:
    token_ids = tokenizer(
        label,
        add_special_tokens=False,
    )["input_ids"]

    if len(token_ids) != 1:
        raise ValueError(
            f"标签 {label} 不是单 Token：{token_ids}"
        )

print("N/R/W 均通过单 Token 检查")

# ============================================================
# 4. 构造 Causal LM 数据
# ============================================================

def encode_example(row):

    messages = row["messages"]

    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    label = str(messages[2]["content"]).strip()

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Prompt token
    # --------------------------------------------------------

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    # --------------------------------------------------------
    # Assistant label
    # --------------------------------------------------------

    answer_ids = tokenizer(
        label,
        add_special_tokens=False,
    )["input_ids"]

    if len(answer_ids) == 0:
        raise ValueError(
            f"标签编码为空: {label}"
        )

    # --------------------------------------------------------
    # 给 assistant label 预留空间
    # --------------------------------------------------------

    max_prompt_length = (
        MAX_LENGTH - len(answer_ids)
    )

    if max_prompt_length <= 0:
        raise ValueError(
            "MAX_LENGTH 不足以容纳监督标签"
        )

    # --------------------------------------------------------
    # 超长时：
    # 只截断 prompt
    # 永远保留 assistant label
    # --------------------------------------------------------

    if len(prompt_ids) > max_prompt_length:

        prompt_ids = prompt_ids[
            -max_prompt_length:
        ]

    # --------------------------------------------------------
    # 最终输入
    # --------------------------------------------------------

    input_ids = (
        prompt_ids + answer_ids
    )

    attention_mask = [
        1
    ] * len(input_ids)

    # --------------------------------------------------------
    # Loss mask
    #
    # Prompt:
    #   -100，不计算loss
    #
    # Assistant:
    #   W/R/N，计算loss
    # --------------------------------------------------------

    labels = (
        [-100] * len(prompt_ids)
        + answer_ids
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# 5. 编码数据
# ============================================================

print()
print("[3/7] Tokenizing 训练集...")

train_encoded = [
    encode_example(row)
    for _, row in train_df.iterrows()
]

print(
    f"训练样本编码完成: "
    f"{len(train_encoded)}"
)

print()
print("[4/7] Tokenizing 验证集...")

validation_encoded = [
    encode_example(row)
    for _, row in validation_df.iterrows()
]

print(
    f"验证样本编码完成: "
    f"{len(validation_encoded)}"
)

# ============================================================
# 6. 数据集包装
# ============================================================

from torch.utils.data import Dataset


class PolicyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


train_dataset = PolicyDataset(
    train_encoded
)

validation_dataset = PolicyDataset(
    validation_encoded
)

# ============================================================
# 7. 4-bit QLoRA 模型
# ============================================================

print()
print("[5/7] 加载 Qwen2.5-1.5B...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

# ============================================================
# 8. 准备 QLoRA
# ============================================================

print()
print("[6/7] 配置 QLoRA...")

model = prepare_model_for_kbit_training(
    model
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()

# ============================================================
# 9. Data Collator
# ============================================================

def collate_fn(features):

    max_len = max(
        len(x["input_ids"])
        for x in features
    )

    input_ids = []
    attention_mask = []
    labels = []

    for feature in features:

        pad_len = (
            max_len
            - len(feature["input_ids"])
        )

        input_ids.append(
            feature["input_ids"]
            + [tokenizer.pad_token_id] * pad_len
        )

        attention_mask.append(
            feature["attention_mask"]
            + [0] * pad_len
        )

        labels.append(
            feature["labels"]
            + [-100] * pad_len
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
    }


# ============================================================
# 10. Training Arguments
# ============================================================

training_args = TrainingArguments(

    output_dir=str(OUTPUT_DIR),

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=
        GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    logging_steps=10,

    save_strategy="epoch",

    eval_strategy="epoch",

    fp16=True,

    gradient_checkpointing=True,

    optim="paged_adamw_8bit",

    report_to="none",

    seed=SEED,

    remove_unused_columns=False,

)

# ============================================================
# 11. Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    data_collator=collate_fn,
)

# ============================================================
# 12. 正式训练
# ============================================================

print()
print("=" * 70)
print("[7/7] 开始正式 QLoRA 训练")
print("=" * 70)

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"输出目录非空，为避免覆盖实验结果，停止训练：{OUTPUT_DIR}"
    )

trainer.train()

# ============================================================
# 13. 保存 Adapter
# ============================================================

print()
print("保存 LoRA Adapter...")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print()
print("=" * 70)
print("训练完成")
print(f"Adapter 已保存到: {OUTPUT_DIR}")
print("=" * 70)