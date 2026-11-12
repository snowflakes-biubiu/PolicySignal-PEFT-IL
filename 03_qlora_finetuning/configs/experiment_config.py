from pathlib import Path
from huggingface_hub import snapshot_download


# ============================================================
# CAPC-CG PEFT 实验统一配置
# ============================================================

# ---------- 项目路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------- Hugging Face 数据集 ----------
SNAPSHOT_DIR = Path(
    snapshot_download(
        "Baron-Sun/CAPC-CG_V1.0",
        repo_type="dataset",
        local_files_only=True,
    )
)

# ---------- 数据 ----------
LEVEL1_TRAIN_FILE = SNAPSHOT_DIR / "task1_level1" / "train.parquet"
LEVEL1_VALID_FILE = SNAPSHOT_DIR / "task1_level1" / "validation.parquet"

LEVEL2_TRAIN_FILE = SNAPSHOT_DIR / "task2_level2" / "train.parquet"
LEVEL2_VALID_FILE = SNAPSHOT_DIR / "task2_level2" / "validation.parquet"


# ---------- 模型 ----------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ---------- Level-1 ----------
LEVEL1_LABELS = ["N", "R", "W"]

LEVEL1_LABEL_MEANING = {
    "N": "中立/非指令性",
    "R": "限制性/禁止性",
    "W": "肯定指令",
}


# ---------- Level-2 ----------
LEVEL2_LABELS = ["B", "C", "G", "Y"]

LEVEL2_LABEL_MEANING = {
    "B": "授权",
    "C": "灵活",
    "G": "模糊不清",
    "Y": "加压",
}


# ---------- Tokenizer ----------
MAX_LENGTH = 256


# ---------- QLoRA ----------
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = "float16"


# ---------- LoRA ----------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


# ---------- Training ----------
SEED = 42

NUM_EPOCHS = 1

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.0

WARMUP_RATIO = 0.05

FP16 = True


# ---------- 输出 ----------
OUTPUT_ROOT = PROJECT_ROOT / "03_qlora_finetuning" / "artifacts" / "outputs"

LEVEL1_QLORA_OUTPUT = OUTPUT_ROOT / "level1_qwen15b_qlora"

LEVEL2_QLORA_OUTPUT = OUTPUT_ROOT / "level2_qwen15b_qlora"


# ---------- 实验记录 ----------
EXPERIMENT_NAME = "Qwen2.5-1.5B-QLoRA"

EXPERIMENT_VERSION = "v1.0"


def print_config():
    print("=" * 70)
    print("CAPC-CG PEFT 实验配置")
    print("=" * 70)

    print(f"模型: {MODEL_NAME}")

    print()
    print("Level-1:")
    print(f"  Train: {LEVEL1_TRAIN_FILE}")
    print(f"  Validation: {LEVEL1_VALID_FILE}")
    print(f"  Labels: {LEVEL1_LABELS}")

    print()
    print("Level-2:")
    print(f"  Train: {LEVEL2_TRAIN_FILE}")
    print(f"  Validation: {LEVEL2_VALID_FILE}")
    print(f"  Labels: {LEVEL2_LABELS}")

    print()
    print("QLoRA:")
    print(f"  4-bit: {LOAD_IN_4BIT}")
    print(f"  Quant type: {BNB_4BIT_QUANT_TYPE}")
    print(f"  Double quant: {BNB_4BIT_USE_DOUBLE_QUANT}")
    print(f"  Compute dtype: {BNB_4BIT_COMPUTE_DTYPE}")

    print()
    print("LoRA:")
    print(f"  r: {LORA_R}")
    print(f"  alpha: {LORA_ALPHA}")
    print(f"  dropout: {LORA_DROPOUT}")
    print(f"  targets: {LORA_TARGET_MODULES}")

    print()
    print("Training:")
    print(f"  seed: {SEED}")
    print(f"  epochs: {NUM_EPOCHS}")
    print(f"  train batch: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  learning rate: {LEARNING_RATE}")
    print(f"  max length: {MAX_LENGTH}")

    print("=" * 70)


if __name__ == "__main__":
    print_config()