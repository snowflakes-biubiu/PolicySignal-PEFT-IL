# -*- coding: utf-8 -*-
"""06 期 G2-1：中文双向编码器（chinese-roberta-wwm-ext-large）全参微调对照。

用途：检验「小模型 + LoRA」相对 0.33B 全参编码器的差距。除了模型结构与训练方式，
数据划分、评估集、指标口径与 G1/G2 的 PEFT 组完全一致，产物 schema 对齐
`run_peft_benchmark.py`，可直接并入同一张对比表。

用法（云端 4090D，仓库根目录）：
  python 06_generation_scale_benchmark/scripts/train/finetune_encoder.py \
      --exp-id G2-1_chinese_roberta_large_ft --model hfl/chinese-roberta-wwm-ext-large \
      --task task1_level1 --precision bf16 --lr 2e-5 --max-length 512
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

import benchmark_common as C  # noqa: E402
import gpu_guard  # noqa: E402


class TextDataset(Dataset):
    """模块级 Dataset，Windows spawn 下可 pickle。"""

    def __init__(self, encodings: dict[str, Any], label_ids: list[int]):
        self.encodings = encodings
        self.label_ids = list(label_ids)

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="06 期 G2-1 编码器全参微调对照")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("--task", choices=list(C.preprocess.TASK_LABELS), required=True)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    parser.add_argument("--model-root", default=str(C.MODEL_ROOT))
    parser.add_argument("--group", default="G2")
    parser.add_argument("--label", default="编码器对照")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root: str | Path = args.results_root
    model_root: str | Path = args.model_root
    if args.smoke:
        results_root = C.SMOKE_ROOT / "results"
        model_root = C.SMOKE_ROOT / "models"
    paths = C.run_paths(args.exp_id, args.task, results_root, model_root)

    code_version = C.code_fingerprint(
        [SCRIPTS_DIR / "benchmark_common.py", Path(__file__).resolve()]
    )
    plan = {
        "exp_id": args.exp_id,
        "group": args.group,
        "label": args.label,
        "runner": "encoder",
        "model": args.model,
        "task": args.task,
        "precision": args.precision,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "smoke": args.smoke,
    }
    fingerprint = C.experiment_fingerprint({**plan, "code_version": code_version})

    if paths["summary"].is_file() and args.skip_existing:
        stored = C.read_json(paths["summary"])
        if stored.get("fingerprint") == fingerprint:
            print(f"[SKIP] 结果已存在且指纹一致: {paths['summary']}")
            return 0
        if not args.overwrite:
            raise SystemExit(
                f"[冲突] {paths['summary']} 指纹不一致，如需重跑请加 --overwrite"
            )
        backup = paths["results_dir"].with_name(
            paths["results_dir"].name + f".bak-{C.now_stamp()}"
        )
        paths["results_dir"].rename(backup)
        print(f"[归档] 旧结果已备份到 {backup}")

    if args.dry_run:
        print("[DRY-RUN] 计划:")
        for key, value in plan.items():
            print(f"  {key:24s} {value}")
        print(f"  {'results_dir':24s} {paths['results_dir']}")
        print(f"  {'model_dir':24s} {paths['encoder_dir']}")
        counts = C.dataset_info()["splits"]
        for name, info in counts.items():
            print(f"  data[{name}] n={info['n']} labels={info['labels']}")
        return 0

    gpu_guard.assert_cloud_gpu_allowed("06 期编码器训练/评估")
    C.set_global_seed(args.seed)

    labels = C.preprocess.TASK_LABELS[args.task]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    train_records = C.preprocess.load_split(args.task, "train")
    val_records = C.preprocess.load_split(args.task, "validation")
    if args.smoke:
        train_records = gpu_guard.limit_smoke_samples(train_records)
        val_records = gpu_guard.limit_smoke_samples(val_records)
        print(f"[SMOKE] 训练 {len(train_records)} 条 / 评估 {len(val_records)} 条，1 步")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_enc = tokenizer(
        [r["text"] for r in train_records],
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    val_enc = tokenizer(
        [r["text"] for r in val_records],
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    train_ds = TextDataset(train_enc, [label_to_id[r["label"]] for r in train_records])
    val_ds = TextDataset(val_enc, [label_to_id[r["label"]] for r in val_records])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
    )
    if torch.cuda.is_available():
        model.to("cuda")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] {args.model} 全参 {total_params:,}，精度={args.precision}")

    paths["encoder_dir"].mkdir(parents=True, exist_ok=True)
    warmup_steps = C.compute_warmup_steps(
        len(train_records), args.batch_size, args.grad_accum, args.epochs, args.warmup_ratio
    )
    train_args = C.make_training_args(
        output_dir=paths["checkpoints"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        precision=args.precision,
        warmup_steps=warmup_steps,
        seed=args.seed,
        max_steps=1 if args.smoke else -1,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak_train_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    trainer.save_model(str(paths["encoder_dir"]))
    tokenizer.save_pretrained(str(paths["encoder_dir"]))
    print(f"[保存] 模型 -> {paths['encoder_dir']}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    prediction = trainer.predict(val_ds)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_sec = time.perf_counter() - t1
    peak_eval_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    probs = torch.softmax(torch.tensor(prediction.predictions, dtype=torch.float32), dim=-1)
    pred_ids = probs.argmax(dim=-1).tolist()
    y_true = [r["label"] for r in val_records]
    y_pred = [labels[i] for i in pred_ids]
    report = C.M.evaluate(y_true, y_pred, labels)

    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    rows = []
    for record, pred_id, prob_vec in zip(val_records, pred_ids, probs.tolist()):
        row: dict[str, Any] = {
            "gold": record["label"],
            "pred": labels[pred_id],
            "correct": int(labels[pred_id] == record["label"]),
            "text": record["text"],
        }
        for label, prob in zip(labels, prob_vec):
            row[f"prob_{label}"] = round(float(prob), 6)
        rows.append(row)
    with paths["predictions"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    train_loss = getattr(train_result, "training_loss", None)
    if train_loss is None:
        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        train_loss = sum(losses) / len(losses) if losses else float("nan")

    summary = {
        **plan,
        "lora_r": 0,
        "lora_alpha": 0,
        "lora_dropout": 0.0,
        "target_scope": "full",
        "gradient_checkpointing": False,
        "task_samples": {"train": len(train_records), "val": len(val_records)},
        "train_truncated_count": 0,
        "warmup_steps": warmup_steps,
        "global_step": int(train_result.global_step),
        "examples_seen": len(train_records) * args.epochs,
        "trainable_params": total_params,
        "total_params": total_params,
        "trainable_params_pct": 100.0,
        "train_loss": round(float(train_loss), 4),
        "train_time_sec": round(train_sec, 2),
        "peak_train_vram_gb": round(peak_train_vram_gb, 3),
        "eval_time_sec": round(eval_sec, 2),
        "infer_samples_per_sec": round(len(val_records) / eval_sec, 2) if eval_sec > 0 else 0.0,
        "peak_eval_vram_gb": round(peak_eval_vram_gb, 3),
        "eval_truncated_count": 0,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro_f1"], 4),
        "cohen_kappa": round(report["cohen_kappa"], 4),
        "labels": labels,
        "metrics_detail": report,
        "predictions_csv": str(paths["predictions"]),
        "model_dir": str(paths["encoder_dir"]),
        "fingerprint": fingerprint,
        "code_version": code_version,
        "data": C.dataset_info() if not args.smoke else "smoke-run",
        "provenance": C.provenance(args.model),
        "finished_at": C.now_stamp(),
    }
    C.write_json(paths["summary"], summary)

    print(
        f"[DONE] {args.exp_id} | {args.task} | Acc={summary['accuracy']:.4f} | "
        f"MF1={summary['macro_f1']:.4f} | Kappa={summary['cohen_kappa']:.4f} | "
        f"Train={summary['train_time_sec']}s ({summary['peak_train_vram_gb']}GB) | "
        f"Infer={summary['infer_samples_per_sec']} samples/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
