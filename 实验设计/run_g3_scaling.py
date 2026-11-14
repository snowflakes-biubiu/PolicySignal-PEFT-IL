# -*- coding: utf-8 -*-
"""06 期 G3 阶段样本量 Scaling 唯一编排与训练脚本：放入 06_generation_scale_benchmark/scripts/ 下运行。

核心特性：
1. 全面对齐 G1/G2 胜出配置：
   - 对比三大代表性架构：chinese-roberta-wwm-ext-large (0.33B 全参) vs Qwen3-1.7B (bf16 LoRA r8) vs Qwen3-4B (bf16 LoRA r8 + GC)
   - 保持有效 batch_size=4 (bs=4, grad_accum=1)，消除步数混淆
2. 自动适配 AutoDL 本地模型路径（自动检测 /root/autodl-tmp/ms/<name> 与 HF 缓存，无需手工改 json）
3. 自动复用 G1/G2 已跑完的金标准全量点（G2-1 与 G1-3），节省 4 次重复训练
4. 支持 --tier-mode {all,gold_only,silver_only}，自动生成 CSV 与 Markdown 阶段报告（含 C/G 与 W/R/N 逐类 F1 追踪）

用法（仓库根目录）：
  python 06_generation_scale_benchmark/scripts/prepare/build_g3_datasets.py
  python 06_generation_scale_benchmark/scripts/run_g3_scaling.py --mode plan
  python 06_generation_scale_benchmark/scripts/run_g3_scaling.py --mode run --smoke
  nohup python 06_generation_scale_benchmark/scripts/run_g3_scaling.py --mode all > 06_g3.log 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_common as C  # noqa: E402
import gpu_guard  # noqa: E402

G3_CONFIG = C.BENCH_ROOT / "configs" / "g3_matrix.json"
G3_DATA_ROOT = C.BENCH_ROOT / "data" / "g3"
G3_RESULTS_ROOT = C.BENCH_ROOT / "results_g3"
G3_MODEL_ROOT = C.ASSETS_ROOT / "models_g3"


def resolve_local_model_path(model_name: str) -> str:
    """在 AutoDL 或本地缓存中自动解析模型目录；若未命中本地目录则回退为原 HuggingFace ID。"""
    p = Path(model_name)
    if p.is_dir():
        return str(p)
    short_name = model_name.split("/")[-1]
    autodl_ms = Path("/root/autodl-tmp/ms") / short_name
    if autodl_ms.is_dir():
        return str(autodl_ms)
    cache_dir_name = "models--" + model_name.replace("/", "--")
    for hub_root in (
        Path("/root/autodl-tmp/hf_models/hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ):
        snap_root = hub_root / cache_dir_name / "snapshots"
        if snap_root.is_dir():
            snaps = [d for d in snap_root.iterdir() if d.is_dir()]
            if snaps:
                return str(max(snaps, key=lambda d: d.stat().st_mtime))
    return model_name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def try_reuse_from_g1_g2(
    model_spec: dict[str, Any],
    task: str,
    per_class: int,
    gold_per_class: int,
    regime: str,
    target_summary_path: Path,
) -> bool:
    """若当前单元恰好是金标准全量档位（L1=200 或 L2=300）且 G1/G2 已有同配置结果，则直接复用。"""
    reuse_exp_id = model_spec.get("reuse_gold_exp_id")
    if not reuse_exp_id or per_class != gold_per_class:
        return False
    src_summary_path = C.RESULTS_ROOT / f"{reuse_exp_id}__{task}" / "summary.json"
    if not src_summary_path.is_file():
        return False
    src = C.read_json(src_summary_path)
    exp_id = f"G3_{model_spec['id']}_n{per_class}"
    payload = {
        **src,
        "exp_id": exp_id,
        "model_id": model_spec["id"],
        "model_label": model_spec.get("label", model_spec["id"]),
        "per_class": per_class,
        "regime": regime,
        "train_samples": src.get("task_samples", {}).get("train", per_class * len(src.get("labels", []))),
        "val_samples": src.get("task_samples", {}).get("val", 0),
        "source": f"reuse:{reuse_exp_id}",
    }
    C.write_json(target_summary_path, payload)
    print(f"[REUSE] {exp_id} ({task}) <- 复用 {reuse_exp_id} (Acc={payload['accuracy']:.4f})")
    return True


def run_worker(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    cfg = C.read_json(args.config)
    model_spec = next(m for m in cfg["models"] if m["id"] == args.model_id)
    task = args.task
    task_spec = cfg["tasks"][task]
    per_class = int(args.per_class)
    gold_per_class = int(task_spec["gold_per_class"])
    regime = "pure_gold" if per_class <= gold_per_class else "gold_plus_silver"
    exp_id = f"G3_{model_spec['id']}_n{per_class}"

    results_root = C.SMOKE_ROOT / "results_g3" if args.smoke else G3_RESULTS_ROOT
    model_root = C.SMOKE_ROOT / "models_g3" if args.smoke else G3_MODEL_ROOT
    paths = C.run_paths(exp_id, task, results_root, model_root)

    if paths["summary"].is_file() and not args.no_skip_existing:
        print(f"[SKIP] 已存在结果: {paths['summary']}")
        return 0

    if not args.smoke and not args.no_skip_existing:
        if try_reuse_from_g1_g2(
            model_spec, task, per_class, gold_per_class, regime, paths["summary"]
        ):
            return 0

    train_jsonl = G3_DATA_ROOT / task / f"train_per_class_{per_class}.jsonl"
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"缺少 G3 训练集，请先运行 build_g3_datasets.py: {train_jsonl}")

    gpu_guard.assert_cloud_gpu_allowed(f"G3 {exp_id} ({task})")
    seed = int(cfg["seed"])
    C.set_global_seed(seed)

    resolved_model = resolve_local_model_path(model_spec["model"])
    labels = C.preprocess.TASK_LABELS[task]
    train_records = read_jsonl(train_jsonl)
    val_records = C.preprocess.load_split(task, "validation")
    if args.smoke:
        train_records = gpu_guard.limit_smoke_samples(train_records)
        val_records = gpu_guard.limit_smoke_samples(val_records)

    if model_spec["runner"] == "encoder":
        class EncDataset(Dataset):
            def __init__(self, enc, label_ids):
                self.enc = enc
                self.label_ids = label_ids

            def __len__(self):
                return len(self.label_ids)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
                return item

        label_to_id = {lb: i for i, lb in enumerate(labels)}
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
        tr_enc = tokenizer(
            [r["text"] for r in train_records],
            truncation=True,
            max_length=model_spec["max_length"],
            padding="max_length",
            return_tensors="pt",
        )
        va_enc = tokenizer(
            [r["text"] for r in val_records],
            truncation=True,
            max_length=model_spec["max_length"],
            padding="max_length",
            return_tensors="pt",
        )
        train_ds = EncDataset(tr_enc, [label_to_id[r["label"]] for r in train_records])
        val_ds = EncDataset(va_enc, [label_to_id[r["label"]] for r in val_records])

        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model, num_labels=len(labels)
        )
        if torch.cuda.is_available():
            model.to("cuda")
        trainable_params = total_params = sum(p.numel() for p in model.parameters())
        warmup = C.compute_warmup_steps(
            len(train_records),
            model_spec["batch_size"],
            model_spec["grad_accum"],
            cfg["epochs"],
            cfg["warmup_ratio"],
        )
        targs = C.make_training_args(
            output_dir=paths["checkpoints"],
            epochs=cfg["epochs"],
            batch_size=model_spec["batch_size"],
            grad_accum=model_spec["grad_accum"],
            lr=model_spec["lr"],
            precision=model_spec["precision"],
            warmup_steps=warmup,
            seed=seed,
            max_steps=1 if args.smoke else -1,
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr_out = trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0
        peak_train_gb = (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred_out = trainer.predict(val_ds)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_sec = time.perf_counter() - t1
        peak_eval_gb = (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        )
        probs = torch.softmax(torch.tensor(pred_out.predictions, dtype=torch.float32), dim=-1)
        pred_ids = probs.argmax(dim=-1).tolist()
        y_pred = [labels[i] for i in pred_ids]
        report = C.M.evaluate([r["label"] for r in val_records], y_pred, labels)
        pred_rows = []
        for rec, pid, pvec in zip(val_records, pred_ids, probs.tolist()):
            row = {
                "gold": rec["label"],
                "pred": labels[pid],
                "correct": int(labels[pid] == rec["label"]),
                "text": rec["text"],
            }
            for lb, pv in zip(labels, pvec):
                row[f"prob_{lb}"] = round(float(pv), 6)
            pred_rows.append(row)
        eval_eff = {
            "eval_time_sec": round(eval_sec, 2),
            "infer_samples_per_sec": round(len(val_records) / eval_sec, 2) if eval_sec > 0 else 0.0,
            "peak_eval_vram_gb": round(peak_eval_gb, 3),
            "eval_truncated_count": 0,
        }
    else:
        sys_prompt = C.preprocess.get_system_prompt(task, "train")
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded_train = [
            C.encode_record(r, tokenizer, sys_prompt, model_spec["max_length"], False)[0]
            for r in train_records
        ]
        model, trainable_params, total_params = C.build_peft_model(
            model_name=resolved_model,
            precision=model_spec["precision"],
            lora_r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_scope=cfg["target_scope"],
            gradient_checkpointing=model_spec["gradient_checkpointing"],
        )
        warmup = C.compute_warmup_steps(
            len(encoded_train),
            model_spec["batch_size"],
            model_spec["grad_accum"],
            cfg["epochs"],
            cfg["warmup_ratio"],
        )
        targs = C.make_training_args(
            output_dir=paths["checkpoints"],
            epochs=cfg["epochs"],
            batch_size=model_spec["batch_size"],
            grad_accum=model_spec["grad_accum"],
            lr=model_spec["lr"],
            precision=model_spec["precision"],
            warmup_steps=warmup,
            seed=seed,
            max_steps=1 if args.smoke else -1,
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=C.PolicyDataset(encoded_train),
            data_collator=C.make_collate_fn(tokenizer.pad_token_id),
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr_out = trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0
        peak_train_gb = (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        )
        paths["adapter_dir"].mkdir(parents=True, exist_ok=True)
        model.save_pretrained(paths["adapter_dir"])
        tokenizer.save_pretrained(paths["adapter_dir"])

        report, pred_rows, eval_eff = C.evaluate_label_scores(
            model=model,
            tokenizer=tokenizer,
            records=val_records,
            system_prompt=sys_prompt,
            labels=labels,
            max_length=model_spec["max_length"],
            eval_batch_size=model_spec["eval_batch_size"],
            enable_thinking=False,
        )

    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    with paths["predictions"].open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)

    train_loss = getattr(tr_out, "training_loss", None)
    if train_loss is None:
        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        train_loss = sum(losses) / len(losses) if losses else 0.0

    summary = {
        "exp_id": exp_id,
        "model_id": model_spec["id"],
        "model_label": model_spec.get("label", model_spec["id"]),
        "model": resolved_model,
        "runner": model_spec["runner"],
        "task": task,
        "regime": regime,
        "per_class": per_class,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "lora_r": 0 if model_spec["runner"] == "encoder" else cfg["lora_r"],
        "lora_alpha": 0 if model_spec["runner"] == "encoder" else cfg["lora_alpha"],
        "target_scope": "full" if model_spec["runner"] == "encoder" else cfg["target_scope"],
        "trainable_params_pct": round(100.0 * trainable_params / total_params, 4),
        "train_time_sec": round(train_sec, 2),
        "peak_train_vram_gb": round(peak_train_gb, 3),
        **eval_eff,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro_f1"], 4),
        "cohen_kappa": round(report["cohen_kappa"], 4),
        "train_loss": round(float(train_loss), 4),
        "global_step": int(tr_out.global_step),
        "labels": labels,
        "metrics_detail": report,
        "seed": seed,
        "source": "run",
        "finished_at": C.now_stamp(),
    }
    C.write_json(paths["summary"], summary)
    print(
        f"[DONE] {exp_id} ({task}, {regime}) | Acc={summary['accuracy']:.4f} | "
        f"MF1={summary['macro_f1']:.4f} | Kappa={summary['cohen_kappa']:.4f} | "
        f"Train={summary['train_time_sec']}s ({summary['peak_train_vram_gb']}GB)"
    )
    return 0


def summarize_g3(results_root: Path) -> None:
    rows = []
    for p in sorted(results_root.glob("*/summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        per_cls = d.get("metrics_detail", {}).get("per_class", {})
        per_class_n = int(d.get("per_class", 0))
        task = d["task"]
        regime = d.get("regime") or (
            "pure_gold"
            if (task == "task1_level1" and per_class_n <= 200)
            or (task == "task2_level2" and per_class_n <= 300)
            else "gold_plus_silver"
        )
        rows.append(
            {
                "exp_id": d["exp_id"],
                "task": task,
                "regime": regime,
                "model_id": d["model_id"],
                "model_label": d.get("model_label", d["model_id"]),
                "per_class": per_class_n,
                "train_samples": d.get("train_samples", 0),
                "accuracy": d["accuracy"],
                "macro_f1": d["macro_f1"],
                "cohen_kappa": d["cohen_kappa"],
                "bottleneck_f1": round(
                    per_cls.get("W", per_cls.get("G", {})).get("f1", 0.0), 4
                ),
                "train_time_sec": d["train_time_sec"],
                "peak_train_vram_gb": d["peak_train_vram_gb"],
                "infer_samples_per_sec": d["infer_samples_per_sec"],
                "train_loss": d.get("train_loss", 0.0),
                "source": d.get("source", "run"),
            }
        )
    if not rows:
        print("暂无 G3 结果可汇总")
        return

    rows.sort(key=lambda r: (r["task"], r["per_class"], r["model_id"]))
    results_root.mkdir(parents=True, exist_ok=True)
    out_csv = results_root / "g3_scaling_summary.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_lines = [
        "# 06 期 G3 样本量 Scaling 实验报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 汇总记录数：**{len(rows)}**",
        "",
    ]
    for task_key, task_title, b_name in (
        ("task1_level1", "Level-1 (W/R/N，瓶颈类 W-F1)", "W-F1"),
        ("task2_level2", "Level-2 (B/Y/C/G，瓶颈类 G-F1)", "G-F1"),
    ):
        sub = [r for r in rows if r["task"] == task_key]
        if not sub:
            continue
        md_lines.extend(
            [
                f"## {task_title}",
                "",
                f"| 数据段 | 单类样本 | 总训练量 | 模型 | Acc | Macro-F1 | Kappa | {b_name} | 训练(s) | 峰值显存(GB) | 推理(样本/s) | 来源 |",
                "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for r in sub:
            md_lines.append(
                f"| {r['regime']} | {r['per_class']} | {r['train_samples']} | {r['model_label']} | "
                f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['cohen_kappa']:.4f} | "
                f"{r['bottleneck_f1']:.4f} | {r['train_time_sec']:.1f} | "
                f"{r['peak_train_vram_gb']:.2f} | {r['infer_samples_per_sec']:.1f} | {r['source']} |"
            )
        md_lines.append("")

    out_md = results_root / "g3_scaling_report.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[汇总完成] CSV -> {out_csv}\n[汇总完成] MD  -> {out_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description="06 期 G3 样本量 Scaling 编排入口")
    parser.add_argument("--mode", choices=["plan", "run", "summarize", "all", "worker"], default="plan")
    parser.add_argument("--config", default=str(G3_CONFIG))
    parser.add_argument("--tier-mode", choices=["all", "gold_only", "silver_only"], default="all")
    parser.add_argument("--task", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="按模型 id 子串过滤")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    if args.mode == "worker":
        return run_worker(args)

    cfg = C.read_json(args.config)
    manifest_file = G3_DATA_ROOT / "g3_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"未找到 {manifest_file}，请先运行: python 06_generation_scale_benchmark/scripts/prepare/build_g3_datasets.py"
        )
    available = {
        (item["task"], int(item["per_class"])): item
        for item in C.read_json(manifest_file)["splits"]
    }

    jobs = []
    for task, spec in cfg["tasks"].items():
        if args.task and task not in args.task:
            continue
        tiers: list[int] = []
        if args.tier_mode in {"all", "gold_only"}:
            tiers.extend(spec.get("gold_tiers", []))
        if args.tier_mode in {"all", "silver_only"}:
            tiers.extend(spec.get("silver_tiers", []))
        for per_class in sorted(set(tiers)):
            if (task, per_class) not in available:
                continue
            regime = available[(task, per_class)]["regime"]
            for m in cfg["models"]:
                if args.only and not any(k in m["id"] for k in args.only):
                    continue
                jobs.append((task, per_class, regime, m))

    if args.mode == "plan":
        print(f"G3 计划共 {len(jobs)} 个单元 (tier_mode={args.tier_mode}):")
        print("-" * 105)
        for task, per_class, regime, m in jobs:
            reuse_tag = (
                f" [可复用 {m['reuse_gold_exp_id']}]"
                if m.get("reuse_gold_exp_id") and per_class == cfg["tasks"][task]["gold_per_class"]
                else ""
            )
            print(
                f"  {task:14s} | {regime:16s} | n/class={per_class:4d} | "
                f"{m['id']:22s} | {m['model']}{reuse_tag}"
            )
        print("-" * 105)
        return 0

    if args.mode in {"run", "all"}:
        for task, per_class, regime, m in jobs:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--config",
                args.config,
                "--task",
                task,
                "--model-id",
                m["id"],
                "--per-class",
                str(per_class),
            ]
            if args.smoke:
                cmd.append("--smoke")
            if args.no_skip_existing:
                cmd.append("--no-skip-existing")
            print(f"\n>>> [G3 RUN] {task} | {regime} | per_class={per_class} | {m['id']}")
            subprocess.run(cmd, check=True)

    if args.mode in {"summarize", "all"}:
        res_root = C.SMOKE_ROOT / "results_g3" if args.smoke else G3_RESULTS_ROOT
        summarize_g3(res_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())