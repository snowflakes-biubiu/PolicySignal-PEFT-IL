# -*- coding: utf-8 -*-
"""06 期 G1/G2：Qwen PEFT（LoRA / QLoRA）单次训练 + 评估 + 效率统计。

一次运行 = 一个 (实验编号 × 任务)。产物：
  results/<exp_id>__<task>/summary.json      全部指标、效率、环境与指纹
  results/<exp_id>__<task>/predictions.csv   逐条预测（gold/pred/prob_*）
  assets/06_generation_scale_benchmark/models/<exp_id>__<task>/adapter/

用法（云端 4090D，仓库根目录）：
  python 06_generation_scale_benchmark/scripts/train/run_peft_benchmark.py \
      --exp-id G1-4_qwen3_1.7b_bf16_r16 --model Qwen/Qwen3-1.7B \
      --task task1_level1 --precision bf16 --lora-r 16 --lora-alpha 32 \
      --target-scope attn_mlp --lr 2e-4 --skip-existing
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
from transformers import AutoTokenizer, Trainer  # noqa: E402

import benchmark_common as C  # noqa: E402
import gpu_guard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="06 期 G1/G2 PEFT 基准：训练 + 评估")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=list(C.preprocess.TASK_LABELS), required=True)
    parser.add_argument("--precision", choices=["bf16", "4bit"], default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-scope", choices=list(C.TARGET_MODULES_MAP), default="attn_mlp")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", help="默认关闭 Qwen3 思考模式")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    parser.add_argument("--model-root", default=str(C.MODEL_ROOT))
    parser.add_argument("--group", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="指纹不一致时归档旧结果并重跑")
    parser.add_argument("--smoke", action="store_true", help="2 条训练样本 + 1 步")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不加载模型")
    return parser.parse_args()


def plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "exp_id": args.exp_id,
        "group": args.group,
        "label": args.label,
        "model": args.model,
        "task": args.task,
        "precision": args.precision,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_scope": args.target_scope,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "gradient_checkpointing": args.gradient_checkpointing,
        "smoke": args.smoke,
    }


def main() -> int:
    args = parse_args()
    results_root: str | Path = args.results_root
    model_root: str | Path = args.model_root
    if args.smoke:
        # 冒烟产物与正式结果、正式 adapter 完全隔离
        results_root = C.SMOKE_ROOT / "results"
        model_root = C.SMOKE_ROOT / "models"
    paths = C.run_paths(args.exp_id, args.task, results_root, model_root)
    code_version = C.code_fingerprint(
        [
            SCRIPTS_DIR / "benchmark_common.py",
            SCRIPTS_DIR / "gpu_guard.py",
            Path(__file__).resolve(),
        ]
    )
    plan = plan_payload(args)
    fingerprint = C.experiment_fingerprint({**plan, "code_version": code_version})

    if paths["summary"].is_file() and args.skip_existing:
        stored = C.read_json(paths["summary"])
        stored_fp = stored.get("fingerprint")
        if stored_fp == fingerprint:
            print(f"[SKIP] 结果已存在且指纹一致: {paths['summary']}")
            return 0
        message = (
            f"[冲突] {paths['summary']} 已存在但指纹不一致\n"
            f"  已存: {stored_fp}\n  本次: {fingerprint}\n"
            f"  如需重跑请加 --overwrite（旧目录会被改名备份）"
        )
        if not args.overwrite:
            raise SystemExit(message)
        backup = paths["results_dir"].with_name(
            paths["results_dir"].name + f".bak-{C.now_stamp()}"
        )
        paths["results_dir"].rename(backup)
        print(f"[归档] 旧结果已备份到 {backup}")

    if args.dry_run:
        print("[DRY-RUN] 计划:")
        for key, value in plan.items():
            print(f"  {key:24s} {value}")
        print(f"  {'fingerprint':24s} {fingerprint}")
        print(f"  {'results_dir':24s} {paths['results_dir']}")
        print(f"  {'adapter_dir':24s} {paths['adapter_dir']}")
        counts = C.dataset_info()["splits"]
        for name, info in counts.items():
            print(f"  data[{name}] n={info['n']} labels={info['labels']}")
        if torch.cuda.is_available():
            print(f"  gpu: {torch.cuda.get_device_name(0)}")
        return 0

    gpu_guard.assert_cloud_gpu_allowed("06 期训练/评估")
    C.set_global_seed(args.seed)

    labels = C.preprocess.TASK_LABELS[args.task]
    system_prompt = C.preprocess.get_system_prompt(args.task, "train")
    train_records = C.preprocess.load_split(args.task, "train")
    val_records = C.preprocess.load_split(args.task, "validation")
    if args.smoke:
        train_records = gpu_guard.limit_smoke_samples(train_records)
        val_records = gpu_guard.limit_smoke_samples(val_records)
        print(f"[SMOKE] 训练 {len(train_records)} 条 / 评估 {len(val_records)} 条，1 步")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_train = []
    train_truncated = 0
    for record in train_records:
        item, truncated = C.encode_record(
            record, tokenizer, system_prompt, args.max_length, args.enable_thinking
        )
        encoded_train.append(item)
        train_truncated += int(truncated)
    print(
        f"[数据] task={args.task} train={len(encoded_train)} val={len(val_records)} "
        f"标签={labels} 训练截断={train_truncated}"
    )

    model, trainable_params, total_params = C.build_peft_model(
        model_name=args.model,
        precision=args.precision,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_scope=args.target_scope,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(
        f"[模型] {args.model} 精度={args.precision} 可训练={trainable_params:,} "
        f"({100.0 * trainable_params / total_params:.4f}%)"
    )

    paths["adapter_dir"].mkdir(parents=True, exist_ok=True)
    warmup_steps = C.compute_warmup_steps(
        len(encoded_train), args.batch_size, args.grad_accum, args.epochs, args.warmup_ratio
    )
    max_steps = 1 if args.smoke else -1
    train_args = C.make_training_args(
        output_dir=paths["checkpoints"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        precision=args.precision,
        warmup_steps=warmup_steps,
        seed=args.seed,
        max_steps=max_steps,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_train0 = time.perf_counter()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=C.PolicyDataset(encoded_train),
        data_collator=C.make_collate_fn(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t_train0
    peak_train_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    # Trainer 用 accelerate 包装了 forward，会把整段 [B, L, V] logits 转 fp32；
    # 8B 在 bs8/seq768 下需要 3.7GB，会在评估阶段 OOM。解包后按原始 dtype 前向。
    model = trainer.accelerator.unwrap_model(model)
    torch.cuda.empty_cache()

    model.save_pretrained(paths["adapter_dir"])
    tokenizer.save_pretrained(paths["adapter_dir"])
    print(f"[保存] adapter -> {paths['adapter_dir']}")

    report, pred_rows, eval_eff = C.evaluate_label_scores(
        model=model,
        tokenizer=tokenizer,
        records=val_records,
        system_prompt=system_prompt,
        labels=labels,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        enable_thinking=args.enable_thinking,
    )

    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    with paths["predictions"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    train_loss = getattr(train_result, "training_loss", None)
    if train_loss is None:
        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        train_loss = sum(losses) / len(losses) if losses else float("nan")

    summary = {
        **plan,
        "runner": "peft",
        "task_samples": {"train": len(train_records), "val": len(val_records)},
        "train_truncated_count": train_truncated,
        "warmup_steps": warmup_steps,
        "global_step": int(train_result.global_step),
        "examples_seen": len(train_records) * args.epochs,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_params_pct": round(100.0 * trainable_params / total_params, 4),
        "train_loss": round(float(train_loss), 4),
        "train_time_sec": round(train_sec, 2),
        "peak_train_vram_gb": round(peak_train_vram_gb, 3),
        **eval_eff,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro_f1"], 4),
        "cohen_kappa": round(report["cohen_kappa"], 4),
        "labels": labels,
        "metrics_detail": report,
        "predictions_csv": str(paths["predictions"]),
        "adapter_dir": str(paths["adapter_dir"]),
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
