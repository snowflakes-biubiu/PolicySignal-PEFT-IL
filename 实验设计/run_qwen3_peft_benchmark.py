"""G1 / G2 阶段统一训练、评估与效率统计脚本 (Qwen2.5 / Qwen3 + LoRA / QLoRA).

适配 CAPC-CG 仓库结构：
- 自动定位仓库根目录下的 shared/common/{preprocess.py, metrics.py}
- 支持 task1_level1 (W/R/N) 与 task2_level2 (B/Y/C/G)
- 支持 bf16 LoRA 与 4-bit QLoRA，支持 attn vs attn_mlp 目标模块消融
- 显式处理 Qwen3 的 enable_thinking=False 非思考模板，避免 <think> 破坏单 token 监督
- 统一记录效果指标 (Acc / Macro-F1 / Kappa) 与工程效率 (耗时 / 峰值显存 / 推理吞吐量)

用法示例（在仓库根目录执行）：
  python run_qwen3_peft_benchmark.py --exp-id G1-4_qwen3_1.7b_bf16_r16 \
      --model Qwen/Qwen3-1.7B --task task1_level1 --precision bf16 \
      --lora-r 16 --lora-alpha 32 --target-scope attn_mlp --lr 2e-4
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def resolve_project_root() -> Path:
    """定位包含 shared/common/preprocess.py 的仓库根目录。"""
    candidates = []
    if os.environ.get("CAPC_PROJECT_ROOT"):
        candidates.append(Path(os.environ["CAPC_PROJECT_ROOT"]).resolve())
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    for path in candidates:
        if (path / "shared" / "common" / "preprocess.py").is_file():
            return path
    raise FileNotFoundError(
        "未找到 shared/common/preprocess.py，请在仓库根目录运行或设置环境变量 CAPC_PROJECT_ROOT。"
    )


PROJECT_ROOT = resolve_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "common"))
import metrics as M  # noqa: E402
import preprocess  # noqa: E402


TARGET_MODULES_MAP = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truncate_head_tail(
    token_ids: list[int], max_length: int, head_ratio: float = 0.5
) -> tuple[list[int], bool]:
    """超长文本保留头部 50% + 尾部 50%，训练与评估共用同一截断函数。"""
    if len(token_ids) <= max_length:
        return list(token_ids), False
    head = max(1, int(max_length * head_ratio))
    tail = max(1, max_length - head)
    return list(token_ids[:head]) + list(token_ids[-tail:]), True


def build_prompt_ids(
    tokenizer: Any, system_prompt: str, user_raw: str, enable_thinking: bool
) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_raw},
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    sig = inspect.signature(tokenizer.apply_chat_template)
    if "enable_thinking" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        kwargs["enable_thinking"] = enable_thinking
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def encode_record(
    record: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    max_length: int,
    enable_thinking: bool,
) -> tuple[dict[str, list[int]], bool]:
    label_str = str(record["label"]).strip()
    label_ids = tokenizer(label_str, add_special_tokens=False)["input_ids"]
    if len(label_ids) != 1:
        raise ValueError(f"标签 {label_str!r} 不是单 token: {label_ids}")
    prompt_ids = build_prompt_ids(
        tokenizer, system_prompt, record["user_raw"], enable_thinking=enable_thinking
    )
    prompt_ids, truncated = truncate_head_tail(prompt_ids, max_length - 1)
    input_ids = prompt_ids + label_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + label_ids,
    }, truncated


class PolicyDataset(Dataset):
    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def make_collate_fn(pad_token_id: int):
    def collate_fn(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def load_base_model(model_name: str, precision: str, gradient_checkpointing: bool):
    if precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif precision == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    else:
        raise ValueError(f"不支持的精度模式: {precision}")
    return model


@torch.no_grad()
def evaluate_model(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    system_prompt: str,
    labels: list[str],
    max_length: int,
    eval_batch_size: int,
    enable_thinking: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    model.eval()
    label_token_ids = {}
    for lb in labels:
        ids = tokenizer(lb, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"评估标签 {lb!r} 非单 token: {ids}")
        label_token_ids[lb] = ids[0]

    encoded_prompts = []
    truncated_count = 0
    for rec in records:
        pids = build_prompt_ids(
            tokenizer, system_prompt, rec["user_raw"], enable_thinking=enable_thinking
        )
        pids, hit = truncate_head_tail(pids, max_length - 1)
        truncated_count += int(hit)
        encoded_prompts.append(pids)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    preds: list[str] = []
    rows_out: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    candidate_ids = [label_token_ids[lb] for lb in labels]

    for start in range(0, len(records), eval_batch_size):
        batch_pids = encoded_prompts[start : start + eval_batch_size]
        batch_recs = records[start : start + eval_batch_size]
        max_l = max(len(x) for x in batch_pids)
        # 左填充以便最后一个非 pad 位置对齐序列末端
        input_ids = []
        attn_mask = []
        for x in batch_pids:
            pad_n = max_l - len(x)
            input_ids.append([pad_id] * pad_n + x)
            attn_mask.append([0] * pad_n + [1] * len(x))

        inp = torch.tensor(input_ids, dtype=torch.long, device=device)
        msk = torch.tensor(attn_mask, dtype=torch.long, device=device)
        logits = model(input_ids=inp, attention_mask=msk).logits[:, -1, :]
        cand_logits = logits[:, candidate_ids].float()
        probs = torch.softmax(cand_logits, dim=-1).cpu().numpy()

        for rec, p_vec in zip(batch_recs, probs):
            pred_idx = int(np.argmax(p_vec))
            pred_label = labels[pred_idx]
            preds.append(pred_label)
            row = {
                "gold": rec["label"],
                "pred": pred_label,
                "correct": int(pred_label == rec["label"]),
                "text": rec["text"],
            }
            for lb, prob in zip(labels, p_vec):
                row[f"prob_{lb}"] = round(float(prob), 6)
            rows_out.append(row)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_sec = time.perf_counter() - t0
    peak_eval_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    y_true = [r["label"] for r in records]
    report = M.evaluate(y_true, preds, labels)
    efficiency = {
        "eval_time_sec": round(eval_sec, 2),
        "infer_samples_per_sec": round(len(records) / eval_sec, 2) if eval_sec > 0 else 0.0,
        "peak_eval_vram_gb": round(peak_eval_vram_gb, 3),
        "eval_truncated_count": truncated_count,
    }
    return report, rows_out, efficiency


def main() -> None:
    parser = argparse.ArgumentParser(description="G1/G2 阶段 Qwen3/Qwen2.5 LoRA 对比实验")
    parser.add_argument("--exp-id", required=True, help="实验唯一标识，如 G2-3_qwen3_4b_l1")
    parser.add_argument("--model", required=True, help="HuggingFace 模型名或本地路径")
    parser.add_argument("--task", choices=list(preprocess.TASK_LABELS), required=True)
    parser.add_argument("--precision", choices=["bf16", "4bit"], default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-scope", choices=["attn", "attn_mlp"], default="attn_mlp")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", help="默认关闭 Qwen3 思考模式")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_qwen3_benchmark"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2 条样本 1 步冒烟测试")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / args.exp_id
    summary_path = out_dir / "summary.json"
    if args.skip_existing and summary_path.is_file():
        print(f"[SKIP] 已存在结果: {summary_path}")
        return

    set_global_seed(args.seed)
    labels = preprocess.TASK_LABELS[args.task]
    system_prompt = preprocess.get_system_prompt(args.task, "train")
    train_records = preprocess.load_split(args.task, "train")
    val_records = preprocess.load_split(args.task, "validation")

    if args.smoke:
        train_records = train_records[:2]
        val_records = val_records[:4]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_train = []
    train_truncated = 0
    for rec in train_records:
        item, hit = encode_record(
            rec, tokenizer, system_prompt, args.max_length, args.enable_thinking
        )
        encoded_train.append(item)
        train_truncated += int(hit)

    base_model = load_base_model(args.model, args.precision, args.gradient_checkpointing)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES_MAP[args.target_scope],
    )
    model = get_peft_model(base_model, lora_cfg)
    trainable_params, total_params = model.get_nb_trainable_parameters()

    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=1 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "4bit"),
        optim="paged_adamw_8bit" if args.precision == "4bit" else "adamw_torch",
        logging_steps=1 if args.smoke else 10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_train0 = time.perf_counter()

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=PolicyDataset(encoded_train),
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )
    train_result = trainer.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t_train0
    peak_train_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    report, pred_rows, eval_eff = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        records=val_records,
        system_prompt=system_prompt,
        labels=labels,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        enable_thinking=args.enable_thinking,
    )

    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    summary = {
        "exp_id": args.exp_id,
        "model": args.model,
        "task": args.task,
        "precision": args.precision,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_scope": args.target_scope,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_truncated_count": train_truncated,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_params_pct": round(100.0 * trainable_params / total_params, 4),
        "train_loss": round(float(train_result.training_loss), 4),
        "train_time_sec": round(train_sec, 2),
        "peak_train_vram_gb": round(peak_train_vram_gb, 3),
        **eval_eff,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro_f1"], 4),
        "cohen_kappa": round(report["cohen_kappa"], 4),
        "metrics_detail": report,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[DONE] {args.exp_id} | Acc={summary['accuracy']:.4f} | "
        f"MF1={summary['macro_f1']:.4f} | Kappa={summary['cohen_kappa']:.4f} | "
        f"Train={summary['train_time_sec']}s ({summary['peak_train_vram_gb']}GB) | "
        f"Infer={summary['infer_samples_per_sec']} samples/s"
    )


if __name__ == "__main__":
    main()