# -*- coding: utf-8 -*-
"""
Qwen2.5-1.5B-Instruct Level-1 QLoRA 双实验连续训练

实验一：Rank=4, Alpha=16, LR=2e-4, Epochs=2
实验二：Rank=16, Alpha=16, LR=2e-4, Epochs=2

日志：
    logs/qlora_two_experiments.log
    logs/rank4_alpha16.log
    logs/rank16_alpha16.log
    logs/qlora_crash.log

注意：
    不覆盖已有的非空实验输出目录。
    模型和数据仅从本地缓存加载。
"""

from pathlib import Path
import sys
import time
import traceback
import gc
import logging

import pandas as pd
import torch

from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset


# ============================================================
# 1. 通用训练参数
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_LENGTH = 768
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

SEED = 42

LOG_DIR = Path("logs")
MAIN_LOG = LOG_DIR / "qlora_two_experiments.log"
CRASH_LOG = LOG_DIR / "qlora_crash.log"

EXPERIMENTS = [
    {
        "name": "Rank4_Alpha16",
        "rank": 4,
        "alpha": 16,
        "output_dir": (
            Path("03_qlora_finetuning/artifacts/models") / "qwen15b_level1_qlora_rank4_alpha16"
        ),
        "log_file": LOG_DIR / "rank4_alpha16.log",
    },
    {
        "name": "Rank16_Alpha16",
        "rank": 16,
        "alpha": 16,
        "output_dir": (
            Path("03_qlora_finetuning/artifacts/models") / "qwen15b_level1_qlora_rank16_alpha16"
        ),
        "log_file": LOG_DIR / "rank16_alpha16.log",
    },
]


# ============================================================
# 2. UTF-8 日志
# ============================================================

LOG_STREAMS = []
CURRENT_EXPERIMENT_LOG = None


def log(message=""):
    """统一输出到终端、总日志和当前实验日志。"""
    text = str(message)

    # 使用 print 的 flush 行为，尽量及时刷新日志。
    print(text, flush=True)

    for stream in list(LOG_STREAMS):
        try:
            stream.write(text + "\n")
            stream.flush()
        except Exception:
            # 日志写入失败时不掩盖原始训练错误。
            pass


def log_exception(title, exc_info=None):
    """记录完整异常到终端、当前日志和独立崩溃日志。"""
    if exc_info is None:
        exc_info = sys.exc_info()

    error_text = "".join(traceback.format_exception(*exc_info))

    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)
    log(error_text)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        with CRASH_LOG.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(title + "\n")
            f.write("=" * 70 + "\n")
            f.write(error_text)
            f.write("\n")
    except Exception as log_error:
        print(
            f"Failed to write crash log: {log_error!r}",
            file=sys.stderr,
            flush=True,
        )


class TeeStream:
    """将 stdout/stderr 同时写入终端和 UTF-8 日志文件。"""

    def __init__(self, original_stream, *log_streams):
        self.original_stream = original_stream
        self.log_streams = log_streams

    @property
    def encoding(self):
        return "utf-8"

    def write(self, text):
        if not text:
            return 0

        try:
            self.original_stream.write(text)
            self.original_stream.flush()
        except UnicodeEncodeError:
            # 某些终端不能编码中文时，用可显示的替代形式输出。
            safe_text = text.encode(
                "ascii",
                errors="backslashreplace",
            ).decode("ascii")
            self.original_stream.write(safe_text)
            self.original_stream.flush()

        for stream in self.log_streams:
            try:
                stream.write(text)
                stream.flush()
            except Exception:
                pass

        return len(text)

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass

        for stream in self.log_streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        return self.original_stream.fileno()

    def writable(self):
        return True


# ============================================================
# 3. 训练进度回调
# ============================================================

class ProgressCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        log(
            f"Training started. "
            f"Estimated total steps: {state.max_steps}"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}

        message = (
            f"[Progress] Step "
            f"{state.global_step}/{state.max_steps}"
        )

        if "epoch" in logs:
            message += f" | Epoch={logs['epoch']:.2f}"

        if "loss" in logs:
            message += f" | Loss={logs['loss']:.6f}"

        if "eval_loss" in logs:
            message += f" | Eval Loss={logs['eval_loss']:.6f}"

        if "learning_rate" in logs:
            message += f" | LR={logs['learning_rate']:.2e}"

        log(message)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        log(f"[Progress] Epoch {epoch:.2f} ended.")


# ============================================================
# 4. 数据集
# ============================================================

class PolicyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# ============================================================
# 5. 数据编码
# ============================================================

def encode_example(row, tokenizer):
    """
    根据 messages 中的 system、user、assistant 内容编码。
    Prompt 标签设为 -100，只对答案部分计算监督损失。
    """

    messages = row["messages"]

    if len(messages) < 3:
        raise ValueError(
            f"Expected at least 3 messages, got {len(messages)}"
        )

    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    label = str(messages[2]["content"]).strip()

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

    answer_ids = tokenizer(
        label,
        add_special_tokens=False,
    )["input_ids"]

    if len(answer_ids) == 0:
        raise ValueError(f"Empty encoded label: {label!r}")

    if len(answer_ids) >= MAX_LENGTH:
        raise ValueError(
            f"Answer is too long for MAX_LENGTH={MAX_LENGTH}: "
            f"{label!r}"
        )

    max_prompt_length = MAX_LENGTH - len(answer_ids)

    # 超长时只截断 Prompt，保留完整标签。
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    input_ids = prompt_ids + answer_ids

    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# 6. 批次填充
# ============================================================

def build_collator(tokenizer):

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None.")

    def collate_fn(features):
        max_len = max(
            len(feature["input_ids"])
            for feature in features
        )

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])

            input_ids.append(
                feature["input_ids"]
                + [pad_token_id] * pad_len
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

    return collate_fn


# ============================================================
# 7. 输出目录安全检查
# ============================================================

def check_output_directories():
    for experiment in EXPERIMENTS:
        output_dir = experiment["output_dir"]

        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is not empty: "
                f"{output_dir}\n"
                "Training stopped to avoid overwriting existing results. "
                "Inspect the directory and select a new output directory "
                "if a new run is intended."
            )


# ============================================================
# 8. 单组实验
# ============================================================

def run_experiment(experiment, tokenizer, train_df, validation_df):

    name = experiment["name"]
    lora_rank = experiment["rank"]
    lora_alpha = experiment["alpha"]
    output_dir = experiment["output_dir"]

    log("\n" + "=" * 70)
    log(f"Experiment: {name}")
    log(f"Model: {MODEL_NAME}")
    log(f"Rank: {lora_rank}")
    log(f"Alpha: {lora_alpha}")
    log(f"Dropout: {LORA_DROPOUT}")
    log(f"Learning rate: {LEARNING_RATE}")
    log(f"Epochs: {NUM_EPOCHS}")
    log(f"Max length: {MAX_LENGTH}")
    log(f"Output directory: {output_dir}")
    log("=" * 70)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: "
            f"{output_dir}"
        )

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # 编码数据
    # --------------------------------------------------------

    log("Encoding training dataset...")

    train_encoded = [
        encode_example(row, tokenizer)
        for _, row in train_df.iterrows()
    ]

    log(f"Training examples: {len(train_encoded)}")

    log("Encoding validation dataset...")

    validation_encoded = [
        encode_example(row, tokenizer)
        for _, row in validation_df.iterrows()
    ]

    log(f"Validation examples: {len(validation_encoded)}")

    train_dataset = PolicyDataset(train_encoded)
    validation_dataset = PolicyDataset(validation_encoded)

    # --------------------------------------------------------
    # 加载 4-bit 模型
    # --------------------------------------------------------

    log("Loading Qwen2.5-1.5B model...")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script is configured for GPU QLoRA."
        )

    torch.cuda.empty_cache()

    gpu = torch.cuda.get_device_properties(0)

    log(f"GPU: {gpu.name}")
    log(f"Total GPU memory: {gpu.total_memory / (1024 ** 3):.2f} GiB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    log("Calling from_pretrained()...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    log("Model loaded successfully.")

    log(
        f"GPU memory allocated after loading: "
        f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB"
    )

    # --------------------------------------------------------
    # 准备 k-bit 模型
    # --------------------------------------------------------

    model.config.use_cache = False

    log("Step 1/3: Preparing k-bit model...")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    log("Step 1/3 complete.")

    # --------------------------------------------------------
    # 配置 LoRA
    # --------------------------------------------------------

    log("Step 2/3: Creating LoRA configuration...")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
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

    log("Step 2/3 complete.")

    log("Step 3/3: Injecting LoRA adapters...")

    model = get_peft_model(model, lora_config)

    log("Step 3/3 complete.")

    model.print_trainable_parameters()
    log("QLoRA preparation complete.")

    log(
        f"GPU memory allocated after QLoRA preparation: "
        f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB"
    )

    # --------------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,

        logging_steps=10,
        logging_first_step=True,
        logging_strategy="steps",

        save_strategy="epoch",
        eval_strategy="epoch",

        fp16=True,
        gradient_checkpointing=True,

        optim="paged_adamw_8bit",

        report_to="none",
        seed=SEED,

        remove_unused_columns=False,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=build_collator(tokenizer),
        callbacks=[ProgressCallback()],
    )

    # --------------------------------------------------------
    # 正式训练
    # --------------------------------------------------------

    log("Starting training...")
    log("Wait for the first loss log; the first step may take longer.")

    start_time = time.time()

    trainer.train()

    # --------------------------------------------------------
    # 保存 Adapter
    # --------------------------------------------------------

    log("Saving LoRA adapter...")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed_minutes = (time.time() - start_time) / 60

    log("=" * 70)
    log(f"Experiment completed: {name}")
    log(f"Adapter directory: {output_dir}")
    log(f"Training time: {elapsed_minutes:.1f} minutes")
    log("=" * 70)

    # --------------------------------------------------------
    # 释放显存
    # --------------------------------------------------------

    del trainer
    del model
    del train_dataset
    del validation_dataset
    del train_encoded
    del validation_encoded

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    log("Model and training resources released.")


# ============================================================
# 9. 主流程
# ============================================================

def main():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_output_directories()

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    log("=" * 70)
    log("Qwen2.5-1.5B Level-1 QLoRA: Two Experiments")
    log("Order: Rank=4 Alpha=16 -> Rank=16 Alpha=16")
    log("=" * 70)

    # --------------------------------------------------------
    # 加载数据集
    # --------------------------------------------------------

    log("[1/4] Locating local dataset cache...")

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

    if not train_file.exists() or not validation_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found:\n"
            f"{train_file}\n"
            f"{validation_file}"
        )

    train_df = pd.read_parquet(train_file)
    validation_df = pd.read_parquet(validation_file)

    log(f"Training set: {len(train_df)}")
    log(f"Validation set: {len(validation_df)}")

    # --------------------------------------------------------
    # 加载 Tokenizer
    # --------------------------------------------------------

    log("[2/4] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has neither pad_token nor eos_token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    log("Tokenizer loaded successfully.")

    # --------------------------------------------------------
    # 连续执行两个实验
    # --------------------------------------------------------

    for index, experiment in enumerate(EXPERIMENTS, start=1):

        log("\n" + "#" * 70)
        log(f"Experiment {index}/{len(EXPERIMENTS)}")
        log(f"Current experiment: {experiment['name']}")
        log("#" * 70)

        log_file = experiment["log_file"]

        # 每个实验单独写日志；主日志仍保留所有输出。
        with log_file.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as experiment_log:

            LOG_STREAMS.append(experiment_log)

            try:
                run_experiment(
                    experiment,
                    tokenizer,
                    train_df,
                    validation_df,
                )

            except Exception:
                log_exception(
                    f"Experiment failed: {experiment['name']}"
                )
                raise

            finally:
                if experiment_log in LOG_STREAMS:
                    LOG_STREAMS.remove(experiment_log)

    log("=" * 70)
    log("Both experiments completed successfully.")
    log("=" * 70)


# ============================================================
# 10. 程序入口、总日志和异常记录
# ============================================================

if __name__ == "__main__":

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # 总日志使用 UTF-8；以追加方式保留历史诊断记录。
        with MAIN_LOG.open(
            "a",
            encoding="utf-8",
            buffering=1,
        ) as main_log_file:

            LOG_STREAMS.append(main_log_file)

            sys.stdout = TeeStream(
                original_stdout,
                main_log_file,
            )

            sys.stderr = TeeStream(
                original_stderr,
                main_log_file,
            )

            log("\n" + "#" * 70)
            log(f"Run started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            log("#" * 70)

            try:
                main()

            except KeyboardInterrupt:
                log("Training interrupted by user.")
                log("Inspect the output directories for saved checkpoints.")
                raise

            except Exception:
                log_exception("Program exited with an exception.")
                sys.exit(1)

            finally:
                log(f"Run ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        LOG_STREAMS.clear()