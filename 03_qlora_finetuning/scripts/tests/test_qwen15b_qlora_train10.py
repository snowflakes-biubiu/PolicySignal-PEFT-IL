import os
import torch
import pandas as pd
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from huggingface_hub import snapshot_download

SNAPSHOT_DIR = snapshot_download(
    "Baron-Sun/CAPC-CG_V1.0",
    repo_type="dataset",
    local_files_only=True
)

TRAIN_FILE = str(Path(SNAPSHOT_DIR) / "task1_level1" / "train.parquet")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_DIR = "03_qlora_finetuning/artifacts/smoke_tests"

LABELS = {"N", "R", "W"}


def extract_messages(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    return x


def build_text(row, tokenizer):
    messages = extract_messages(row["messages"])

    # 数据格式：
    # messages[0] = system
    # messages[1] = user
    # messages[2] = assistant gold label

    system_text = str(messages[0]["content"])
    user_text = str(messages[1]["content"])
    gold = str(messages[2]["content"]).strip()

    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": gold},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    print("=" * 60)
    print("QLoRA 10条样本训练冒烟测试")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    print("GPU:", torch.cuda.get_device_name(0))

    # ---------------------------------------------------------
    # 1. 读取10条训练数据
    # ---------------------------------------------------------
    print("\n[1/6] 读取训练数据...")

    df = pd.read_parquet(TRAIN_FILE).head(10).copy()

    print("样本数:", len(df))

    # ---------------------------------------------------------
    # 2. tokenizer
    # ---------------------------------------------------------
    print("\n[2/6] 加载 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # 3. 构造训练文本
    # ---------------------------------------------------------
    print("\n[3/6] 构造训练样本...")

    texts = []

    for _, row in df.iterrows():
        text = build_text(row, tokenizer)
        texts.append(text)

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # ---------------------------------------------------------
    # 4. 4-bit QLoRA模型
    # ---------------------------------------------------------
    print("\n[4/6] 加载4-bit模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # ---------------------------------------------------------
    # 5. LoRA
    # ---------------------------------------------------------
    print("\n[5/6] 配置LoRA...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # 6. 训练
    # ---------------------------------------------------------
    print("\n[6/6] 开始10条样本训练...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    before = torch.cuda.memory_allocated() / 1024**3

    result = trainer.train()

    after = torch.cuda.memory_allocated() / 1024**3

    print("\n" + "=" * 60)
    print("QLoRA冒烟测试完成")
    print("=" * 60)

    print("训练步数:", result.global_step)
    print("训练损失:", result.training_loss)
    print(f"训练前显存: {before:.2f} GB")
    print(f"训练后显存: {after:.2f} GB")

    # 测试adapter保存
    adapter_dir = os.path.join(OUTPUT_DIR, "adapter_test")

    print("\n保存LoRA Adapter...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("Adapter保存成功:", adapter_dir)


if __name__ == "__main__":
    main()