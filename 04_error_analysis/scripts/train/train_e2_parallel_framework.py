from pathlib import Path
import gc
import json
import random

import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

from torch.utils.data import Dataset

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
# 0. 全局实验配置
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_LENGTH = 768

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

SEED = 42

LABELS = ["N", "R", "W"]
ALL_PREDICTION_LABELS = ["N", "R", "W", "INVALID"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "04_error_analysis" / "data" / "split_v2"

EXPERIMENT_ROOT = (
    PROJECT_ROOT / "04_error_analysis" / "artifacts" / "e2_class_weighted"
)

# E1 混淆矩阵提供的类别召回率。
# N: 72/75 = 0.96
# R: 68/75 ≈ 0.9067，但本轮对照实验固定 R 权重为 1.0
# W: 63/75 = 0.84
#
# 权重公式：1 + (1 - Recall)
# N 权重 = 1.04
# W 权重 = 1.16
#
# 注意：E1 混淆矩阵没有包含两条 INVALID 预测，
# 因此本权重是依据现有矩阵得到的启发式实验设置，
# 不是理论最优权重。

BASELINE_RECALL = {
    "N": 0.96,
    "R": 68 / 75,
    "W": 0.84,
}

N_WEIGHT = 1.0 + (1.0 - BASELINE_RECALL["N"])
W_WEIGHT = 1.0 + (1.0 - BASELINE_RECALL["W"])

EXPERIMENTS = [
    {
        "name": "E2_A_N_weighted",
        "class_weights": {
            "N": N_WEIGHT,
            "R": 1.0,
            "W": 1.0,
        },
    },
    {
        "name": "E2_B_NW_weighted",
        "class_weights": {
            "N": N_WEIGHT,
            "R": 1.0,
            "W": W_WEIGHT,
        },
    },
]


# ============================================================
# 1. 随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 自定义类别加权 Trainer
# ============================================================

class ClassWeightedTrainer(Trainer):

    def __init__(
        self,
        *args,
        class_token_weights,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_token_weights = class_token_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Causal LM：位置 t 的 logits 预测位置 t+1 的 token
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.reshape(
            -1, shift_logits.size(-1)
        )
        flat_labels = shift_labels.reshape(-1)

        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
        ).reshape_as(shift_labels)

        valid = shift_labels != -100

        if not valid.any():
            raise ValueError("当前 batch 没有有效监督标签")

        # 默认每个 token 的权重为 1
        weights = torch.ones_like(token_losses)

        # 仅训练时启用类别权重，验证损失保持未加权
        if model.training:
            for token_id, weight in self.class_token_weights.items():
                weights[shift_labels == token_id] = float(weight)

        # 关键修复：
        # 分母为有效监督 token 数量，而不是权重之和。
        # 否则每个样本只有一个标签 token 时，类别权重会被抵消。

        loss = (
            token_losses[valid] * weights[valid]
        ).sum() / valid.sum().to(
            token_losses.dtype
        ).clamp_min(1)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 3. 数据集与编码
# ============================================================

class PolicyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def extract_label(messages):
    return str(messages[-1]["content"]).strip()


def build_prompt_ids(messages, tokenizer):
    """
    构造模型生成标签前的 prompt。
    只包含 system 和 user 消息，不包含真实标签。
    """

    prompt_messages = [
        {
            "role": "system",
            "content": messages[0]["content"],
        },
        {
            "role": "user",
            "content": messages[1]["content"],
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

    # 预留目标标签 token 的位置
    max_prompt_length = MAX_LENGTH - 1

    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    return prompt_ids


def encode_example(row, tokenizer):
    messages = row["messages"]
    label = extract_label(messages)

    if label not in LABELS:
        raise ValueError(f"发现非法标签：{label}")

    prompt_ids = build_prompt_ids(messages, tokenizer)

    answer_ids = tokenizer(
        label,
        add_special_tokens=False,
    )["input_ids"]

    if len(answer_ids) != 1:
        raise ValueError(
            f"标签 {label} 不是单 Token：{answer_ids}"
        )

    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)

    # Prompt 不计算 loss，只监督 assistant 的标签 token
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# 4. 数据整理器
# ============================================================

def make_collate_fn(tokenizer):

    def collate_fn(features):
        max_len = max(
            len(item["input_ids"])
            for item in features
        )

        input_ids = []
        attention_masks = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"]
                + [tokenizer.pad_token_id] * pad_len
            )

            attention_masks.append(
                item["attention_mask"]
                + [0] * pad_len
            )

            labels.append(
                item["labels"]
                + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels, dtype=torch.long
            ),
        }

    return collate_fn


# ============================================================
# 5. 加载并检查数据
# ============================================================

def load_data():

    train_file = DATA_DIR / "train.jsonl"
    validation_file = DATA_DIR / "validation.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"训练集不存在：{train_file}"
        )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"验证集不存在：{validation_file}"
        )

    train_df = pd.read_json(train_file, lines=True)
    validation_df = pd.read_json(
        validation_file, lines=True
    )

    train_counts = (
        train_df["messages"]
        .apply(extract_label)
        .value_counts()
        .to_dict()
    )

    validation_counts = (
        validation_df["messages"]
        .apply(extract_label)
        .value_counts()
        .to_dict()
    )

    print(f"训练集数量：{len(train_df)}")
    print(f"训练集标签分布：{train_counts}")
    print(f"验证集数量：{len(validation_df)}")
    print(f"验证集标签分布：{validation_counts}")

    expected_train = {"N": 225, "R": 225, "W": 225}
    expected_validation = {"N": 75, "R": 75, "W": 75}

    if (
        len(train_df) != 675
        or train_counts != expected_train
    ):
        raise ValueError(
            "训练集数量或标签分布不符合预期"
        )

    if (
        len(validation_df) != 225
        or validation_counts != expected_validation
    ):
        raise ValueError(
            "验证集数量或标签分布不符合预期"
        )

    return train_df, validation_df


# ============================================================
# 6. 加载 Tokenizer 并检查标签 Token
# ============================================================

def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer 没有 pad_token 或 eos_token"
            )
        tokenizer.pad_token = tokenizer.eos_token

    label_token_ids = {}

    for label in LABELS:
        token_ids = tokenizer(
            label,
            add_special_tokens=False,
        )["input_ids"]

        if len(token_ids) != 1:
            raise ValueError(
                f"标签 {label} 不是单 Token：{token_ids}"
            )

        label_token_ids[label] = token_ids[0]

    if len(set(label_token_ids.values())) != len(LABELS):
        raise ValueError(
            "N/R/W 标签映射到了重复的 Token ID"
        )

    print(f"标签 Token ID：{label_token_ids}")

    return tokenizer, label_token_ids


# ============================================================
# 7. 加载基础模型并配置 QLoRA
# ============================================================

def load_qlora_model():

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

    model = prepare_model_for_kbit_training(model)

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

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ============================================================
# 8. 生成式验证与指标评估
# ============================================================

def predict_and_evaluate(
    model,
    tokenizer,
    validation_df,
    output_dir,
):

    model.eval()

    true_labels = []
    predictions = []
    raw_outputs = []

    for position in range(len(validation_df)):
        messages = validation_df.iloc[position]["messages"]
        true_label = extract_label(messages)

        prompt_ids = build_prompt_ids(
            messages, tokenizer
        )

        device = model.get_input_embeddings().weight.device

        input_tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.ones_like(input_tensor)

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0, input_tensor.shape[1]:]

        pred_text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        # 严格解析：只接受独立出现的一个 N/R/W 标签。
        # 不猜测多标签或非预期输出。

        if pred_text in LABELS:
            pred_label = pred_text
        else:
            pred_label = "INVALID"

        true_labels.append(true_label)
        predictions.append(pred_label)
        raw_outputs.append(pred_text)

        if (position + 1) % 25 == 0 or position + 1 == len(validation_df):
            print(
                f"验证推理进度：{position + 1}/{len(validation_df)}"
            )

    accuracy = accuracy_score(
        true_labels, predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    # 将 INVALID 作为独立预测类别纳入 Kappa，
    # 避免从一致性统计中忽略非法输出。
    kappa = cohen_kappa_score(
        true_labels,
        predictions,
        labels=ALL_PREDICTION_LABELS,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    metrics = {
        "sample_count": len(true_labels),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "cohen_kappa": float(kappa),
        "invalid_predictions": int(
            sum(p == "INVALID" for p in predictions)
        ),
        "n_precision": float(precision[0]),
        "n_recall": float(recall[0]),
        "n_f1": float(f1[0]),
        "r_precision": float(precision[1]),
        "r_recall": float(recall[1]),
        "r_f1": float(f1[1]),
        "w_precision": float(precision[2]),
        "w_recall": float(recall[2]),
        "w_f1": float(f1[2]),
    }

    
    # 混淆矩阵保留全部样本：
    # 行为真实标签 N/R/W，列为预测标签 N/R/W/INVALID。
    # INVALID 只作为预测类别，不作为真实类别。
    true_class_labels = LABELS
    pred_class_labels = ALL_PREDICTION_LABELS

    cm = [
        [
            sum(
                1
                for true_label, pred_label in zip(true_labels, predictions)
                if true_label == true_class
                and pred_label == pred_class
            )
            for pred_class in pred_class_labels
        ]
        for true_class in true_class_labels
    ]

    confusion_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in true_class_labels],
        columns=[f"pred_{label}" for label in pred_class_labels],
    )
    confusion_df.index.name = "true_label"



    prediction_rows = []

    for idx, (true_label, pred_label, raw_text) in enumerate(
        zip(true_labels, predictions, raw_outputs)
    ):
        messages = validation_df.iloc[idx]["messages"]

        prediction_rows.append({
            "sample_index": idx,
            "true_label": true_label,
            "predicted_label": pred_label,
            "correct": true_label == pred_label,
            "raw_generated_text": raw_text,
            "text": messages[1]["content"],
        })

    predictions_df = pd.DataFrame(prediction_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df.to_csv(
        output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    report = classification_report(
        true_labels,
        predictions,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )

    with open(
        output_dir / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n验证集指标：")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n混淆矩阵（行=真实标签，列=预测标签）：")
    print(confusion_df.to_string())

    return metrics


# ============================================================
# 9. 单次实验执行函数
# ============================================================

def run_experiment(
    exp,
    train_df,
    validation_df,
    tokenizer,
    label_token_ids,
):

    experiment_name = exp["name"]
    class_weights = exp["class_weights"]

    experiment_dir = EXPERIMENT_ROOT / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    adapter_dir = experiment_dir / "adapter"
    results_dir = experiment_dir / "results"

    # 防止覆盖历史实验
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            "实验目录已存在且非空，为避免覆盖结果，停止执行："
            f"{experiment_dir}\n"
            "如需重跑，请先备份并移走该实验目录。"
        )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_token_weights = {
        label_token_ids[label]: float(weight)
        for label, weight in class_weights.items()
    }

    config = {
        "experiment_name": experiment_name,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "seed": SEED,
        "baseline_recall": BASELINE_RECALL,
        "class_weights": class_weights,
        "label_token_ids": {
            label: int(token_id)
            for label, token_id in label_token_ids.items()
        },
    }

    with open(
        experiment_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 75)
    print(f"开始实验：{experiment_name}")
    print(f"类别权重：{class_weights}")
    print(f"实验目录：{experiment_dir}")
    print("=" * 75)

    set_seed(SEED)

    train_encoded = [
        encode_example(row, tokenizer)
        for _, row in train_df.iterrows()
    ]

    validation_encoded = [
        encode_example(row, tokenizer)
        for _, row in validation_df.iterrows()
    ]

    train_dataset = PolicyDataset(train_encoded)
    validation_dataset = PolicyDataset(validation_encoded)

    collate_fn = make_collate_fn(tokenizer)

    # 每组实验都从原始基础模型独立初始化
    model = load_qlora_model()

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        save_total_limit=2,
    )

    trainer = ClassWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
        class_token_weights=class_token_weights,
    )

    print(f"\n训练 {experiment_name} ...")
    trainer.train()

    # 保存最终 Adapter
    adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # 保存训练日志
    with open(
        experiment_dir / "trainer_log_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            trainer.state.log_history,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # 用生成式分类对验证集进行评估
    metrics = predict_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        validation_df=validation_df,
        output_dir=results_dir,
    )

    metrics["experiment_name"] = experiment_name
    metrics["n_weight"] = float(class_weights["N"])
    metrics["r_weight"] = float(class_weights["R"])
    metrics["w_weight"] = float(class_weights["W"])

    with open(
        results_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n实验完成：{experiment_name}")
    print(f"Adapter：{adapter_dir}")
    print(f"评估结果：{results_dir}")

    del trainer
    del model
    del train_dataset
    del validation_dataset
    del train_encoded
    del validation_encoded

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ============================================================
# 10. 主程序：顺序运行两组实验
# ============================================================

def main():

    print("=" * 75)
    print("E2 类别加权双实验")
    print("=" * 75)

    print(f"模型：{MODEL_NAME}")
    print(f"训练集：{DATA_DIR / 'train.jsonl'}")
    print(f"验证集：{DATA_DIR / 'validation.jsonl'}")
    print(f"实验输出：{EXPERIMENT_ROOT}")

    print("\n实验配置：")
    for exp in EXPERIMENTS:
        print(f"{exp['name']}: {exp['class_weights']}")

    # 控制变量检查
    if (
        EXPERIMENTS[0]["class_weights"]["N"]
        != EXPERIMENTS[1]["class_weights"]["N"]
    ):
        raise ValueError(
            "E2-A 与 E2-B 的 N 权重不一致"
        )

    if EXPERIMENTS[0]["class_weights"]["W"] != 1.0:
        raise ValueError(
            "E2-A 的 W 权重必须为 1.0"
        )

    if EXPERIMENTS[1]["class_weights"]["W"] <= 1.0:
        raise ValueError(
            "E2-B 的 W 权重必须大于 1.0"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有可用 CUDA GPU，请检查 PyTorch 与显卡环境"
        )

    print(f"\nGPU：{torch.cuda.get_device_name(0)}")

    train_df, validation_df = load_data()
    tokenizer, label_token_ids = load_tokenizer()

    # 在任何训练开始前检查所有目录，防止只完成一组后才发现
    # 第二组目录已存在，造成实验流程中断。
    for exp in EXPERIMENTS:
        exp_dir = EXPERIMENT_ROOT / exp["name"]

        if exp_dir.exists() and any(exp_dir.iterdir()):
            raise FileExistsError(
                "发现已有非空实验目录，为避免覆盖而停止："
                f"{exp_dir}"
            )

    all_metrics = []

    for index, exp in enumerate(EXPERIMENTS, start=1):
        print(
            f"\n开始第 {index}/{len(EXPERIMENTS)} 组实验"
        )

        metrics = run_experiment(
            exp=exp,
            train_df=train_df,
            validation_df=validation_df,
            tokenizer=tokenizer,
            label_token_ids=label_token_ids,
        )

        all_metrics.append(metrics)

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df = pd.DataFrame(all_metrics)

    comparison_path = (
        EXPERIMENT_ROOT / "e2_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 75)
    print("E2 两组实验全部完成")
    print("=" * 75)

    display_columns = [
        "experiment_name",
        "accuracy",
        "macro_f1",
        "cohen_kappa",
        "n_f1",
        "r_f1",
        "w_f1",
        "n_recall",
        "r_recall",
        "w_recall",
        "n_weight",
        "r_weight",
        "w_weight",
        "invalid_predictions",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in comparison_df.columns
    ]

    print(
        comparison_df[available_columns].to_string(
            index=False
        )
    )

    print(f"\n汇总文件：{comparison_path}")


if __name__ == "__main__":
    main()