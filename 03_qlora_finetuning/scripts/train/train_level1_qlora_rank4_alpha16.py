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
# Experiment configuration
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
#    "03_qlora_finetuning/artifacts/models"
#) / "qwen15b_level1_qlora_3epoch"

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

SEED = 42

OUTPUT_DIR = Path("03_qlora_finetuning/artifacts/models") / "qwen15b_level1_qlora_rank4_alpha16"
# ============================================================
# 1. 鍩虹鐜
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("Qwen2.5-1.5B Level-1 QLoRA 姝ｅ紡璁粌")
print("=" * 70)

print(f"Model: {MODEL_NAME}")
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
# 2. 鏁版嵁璺緞
# ============================================================

print()
print("[1/7] Locating dataset...")

snapshot_dir = Path(
    snapshot_download(
        "Baron-Sun/CAPC-CG_V1.0",
        repo_type="dataset",
        local_files_only=True,
    )
)

train_file = (
    snapshot_dir
    / "task1_level1"
    / "train.parquet"
)

validation_file = (
    snapshot_dir
    / "task1_level1"
    / "validation.parquet"
)

train_df = pd.read_parquet(train_file)
validation_df = pd.read_parquet(validation_file)

print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")

# ============================================================
# 3. Tokenizer
# ============================================================

print()
print("[2/7] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully.")

# ============================================================
# 4. Build causal-LM training data
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
            f"Encoded label is empty: {label}"
        )

    # --------------------------------------------------------
    # Reserve space for the assistant label
    # --------------------------------------------------------

    max_prompt_length = (
        MAX_LENGTH - len(answer_ids)
    )

    if max_prompt_length <= 0:
        raise ValueError(
            "MAX_LENGTH is too small to fit the label."
        )

    # --------------------------------------------------------
    # If the prompt is too long:
    # truncate only the prompt
    # always preserve the assistant label
    # --------------------------------------------------------

    if len(prompt_ids) > max_prompt_length:

        prompt_ids = prompt_ids[
            -max_prompt_length:
        ]

    # --------------------------------------------------------
    # Final input
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
    #   -100: do not calculate loss
    #
    # Assistant:
    #   W/R/N: calculate loss
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
# 5. 缂栫爜鏁版嵁
# ============================================================

print()
print("[3/7] Tokenizing training data...")

train_encoded = [
    encode_example(row)
    for _, row in train_df.iterrows()
]

print(
    f"Encoded training examples: "
    f"{len(train_encoded)}"
)

print()
print("[4/7] Tokenizing validation data...")

validation_encoded = [
    encode_example(row)
    for _, row in validation_df.iterrows()
]

print(
    f"Encoded validation examples: "
    f"{len(validation_encoded)}"
)

# ============================================================
# 6. Dataset wrapper
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
# 7. 4-bit QLoRA model
# ============================================================

print()
print("[5/7] Loading Qwen2.5-1.5B...")

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
# 8. Prepare QLoRA
# ============================================================

print()
print("[6/7] Configuring QLoRA...")

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
# 12. Train
# ============================================================

print()
print("=" * 70)
print("[7/7] Starting QLoRA training")
print("=" * 70)

trainer.train()

# ============================================================
# 13. Save adapter
# ============================================================

print()
print("Saving LoRA adapter...")

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
print("Training complete.")
print(f"Adapter saved to: {OUTPUT_DIR}")
print("=" * 70)
