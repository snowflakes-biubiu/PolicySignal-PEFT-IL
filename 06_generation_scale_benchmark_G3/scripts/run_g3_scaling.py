# -*- coding: utf-8 -*-
"""G3 样本量 Scaling 唯一编排入口：plan / run / all / summarize / worker。

单元 = 模型 × 任务 × 每类样本量。数据由 `scripts/prepare/build_g3_datasets.py` 构建，
纯金标准段（L1 ≤200/类、L2 ≤300/类）与银标扩增段（L1 >200、L2 >300）在 manifest 中标记。

复用：金标准全量档位（L1 200、L2 300）若与 G1/G2 配置一致，直接复用 06_generation_scale_benchmark_G1_G2/results，
并在 summary 中标记 `source=reuse:<exp>`；4B 因 G2 用 r16、G3 用 r8，需重训。

用法（仓库根目录）：
  python 06_generation_scale_benchmark_G3/scripts/prepare/build_g3_datasets.py
  python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode plan
  python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode run --smoke --only qwen3_4b
  nohup python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode all > 06_generation_scale_benchmark_G3/logs/run_all.log 2>&1 &
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

import g3_common as G  # noqa: E402
import gpu_guard  # noqa: E402

G3_RESULTS_ROOT = G.G3_RESULTS_ROOT
G3_MODEL_ROOT = G.G3_MODEL_ROOT
G3_SMOKE_ROOT = G.G3_SMOKE_ROOT
G3_DATA_ROOT = G.G3_DATA_ROOT
G1G2_RESULTS_ROOT = G.G1G2_RESULTS_ROOT
CONFIG_PATH = G.G3_CONFIG


def code_version() -> str:
    return G.code_fingerprint(
        [
            Path(__file__).resolve(),
            SCRIPTS_DIR / "g3_common.py",
            G.G1G2_SCRIPTS / "benchmark_common.py",
        ]
    )


def unit_fingerprint(
    cfg: dict[str, Any],
    model_spec: dict[str, Any],
    task: str,
    per_class: int,
    data_sha: str,
) -> str:
    return G.experiment_fingerprint(
        {
            "stage": "G3",
            "task": task,
            "per_class": per_class,
            "model_id": model_spec["id"],
            "model": model_spec["model"],
            "runner": model_spec["runner"],
            "precision": model_spec["precision"],
            "lr": model_spec["lr"],
            "batch_size": model_spec["batch_size"],
            "grad_accum": model_spec["grad_accum"],
            "max_length": model_spec["max_length"],
            "gradient_checkpointing": model_spec["gradient_checkpointing"],
            "epochs": cfg["epochs"],
            "lora_r": cfg["lora_r"],
            "lora_alpha": cfg["lora_alpha"],
            "target_scope": cfg["target_scope"],
            "seed": cfg["seed"],
            "data_sha": data_sha,
            "code": code_version(),
        }
    )


def load_manifest() -> dict[str, Any]:
    path = G3_DATA_ROOT / "g3_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}，请先运行 06_generation_scale_benchmark_G3/scripts/prepare/build_g3_datasets.py"
        )
    return G.read_json(path)


def load_reuse_source(
    model_spec: dict[str, Any],
    task: str,
    per_class: int,
    gold_per_class: int,
    cfg: dict[str, Any],
    label_count: int,
) -> dict[str, Any] | None:
    """仅复用「金标准全量档位」，且逐项校验配置一致。"""
    exp_id = model_spec.get("reuse_gold_exp_id")
    if not exp_id or per_class != gold_per_class:
        return None
    path = G1G2_RESULTS_ROOT / f"{exp_id}__{task}" / "summary.json"
    if not path.is_file():
        print(f"[REUSE-MISS] {exp_id}__{task} 不存在，改为重训")
        return None
    source = G.read_json(path)

    expected_train = gold_per_class * label_count
    actual_train = source.get("task_samples", {}).get("train", source.get("train_samples"))
    if actual_train != expected_train:
        print(f"[REUSE-MISS] {exp_id}: train={actual_train} != {expected_train}")
        return None
    if source.get("precision") != model_spec["precision"]:
        print(f"[REUSE-MISS] {exp_id}: precision 不一致")
        return None
    if model_spec["runner"] == "encoder":
        if source.get("target_scope") != "full":
            print(f"[REUSE-MISS] {exp_id}: 不是全参编码器")
            return None
    else:
        if (
            source.get("lora_r") != cfg["lora_r"]
            or source.get("lora_alpha") != cfg["lora_alpha"]
            or source.get("target_scope") != cfg["target_scope"]
        ):
            print(f"[REUSE-MISS] {exp_id}: LoRA 配比与 G3 不一致")
            return None
    return source


def run_worker(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    cfg = G.read_json(args.config)
    model_spec = next(m for m in cfg["models"] if m["id"] == args.model_id)
    task = args.task[0] if isinstance(args.task, (list, tuple)) else args.task
    task_spec = cfg["tasks"][task]
    per_class = int(args.per_class)
    gold_per_class = int(task_spec["gold_per_class"])
    labels = G.preprocess.TASK_LABELS[task]
    regime = "pure_gold" if per_class <= gold_per_class else "gold_plus_silver"
    exp_id = f"G3_{model_spec['id']}_n{per_class}"

    manifest = load_manifest()
    unit = next(
        (s for s in manifest["splits"] if s["task"] == task and int(s["per_class"]) == per_class),
        None,
    )
    if unit is None:
        raise FileNotFoundError(f"manifest 中缺少 {task} n={per_class}，请重建数据")
    data_sha = unit["sha256"]

    results_root = G3_SMOKE_ROOT / "results" if args.smoke else G3_RESULTS_ROOT
    model_root = G3_SMOKE_ROOT / "models" if args.smoke else G3_MODEL_ROOT
    paths = G.run_paths(exp_id, task, results_root, model_root)
    fingerprint = unit_fingerprint(cfg, model_spec, task, per_class, data_sha)

    if paths["summary"].is_file() and not args.no_skip_existing:
        stored = G.read_json(paths["summary"])
        if stored.get("fingerprint") == fingerprint:
            print(f"[SKIP] {paths['summary']}")
            return 0
        print(f"[CONFLICT] 指纹不一致，需 --no-skip-existing 覆盖: {paths['summary']}")

    if not args.smoke and not args.no_skip_existing:
        source = load_reuse_source(
            model_spec, task, per_class, gold_per_class, cfg, len(labels)
        )
        if source is not None:
            payload = {
                **source,
                "exp_id": exp_id,
                "model_id": model_spec["id"],
                "model_label": model_spec.get("label", model_spec["id"]),
                "per_class": per_class,
                "regime": regime,
                "train_samples": unit["total"],
                "val_samples": source.get("task_samples", {}).get("val", 0),
                "source": f"reuse:{model_spec['reuse_gold_exp_id']}",
                "data_sha": data_sha,
                "fingerprint": fingerprint,
                "stage": "G3",
            }
            G.write_json(paths["summary"], payload)
            print(
                f"[REUSE] {exp_id} ({task}) <- {model_spec['reuse_gold_exp_id']} "
                f"Acc={payload['accuracy']:.4f}"
            )
            return 0

    train_path = G3_DATA_ROOT / task / f"train_per_class_{per_class}.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"缺少训练集: {train_path}")
    train_records = G.read_jsonl(train_path)
    val_records = G.preprocess.load_split(task, "validation")
    if args.smoke:
        train_records = gpu_guard.limit_smoke_samples(train_records)
        val_records = gpu_guard.limit_smoke_samples(val_records)

    gpu_guard.assert_cloud_gpu_allowed(f"G3 {exp_id} ({task})")
    seed = int(cfg["seed"])
    G.set_global_seed(seed)
    resolved_model = G.resolve_local_model_path(model_spec["model"])
    print(
        f"[START] {exp_id} | {task} | n/class={per_class} | {regime} | "
        f"train={len(train_records)} | model={resolved_model}"
    )

    if model_spec["runner"] == "encoder":

        class EncDataset(Dataset):
            def __init__(self, enc, label_ids):
                self.enc = enc
                self.label_ids = label_ids

            def __len__(self) -> int:
                return len(self.label_ids)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                item = {k: v[idx] for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
                return item

        label_to_id = {lb: i for i, lb in enumerate(labels)}
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
        max_len = model_spec["max_length"]
        train_enc = tokenizer(
            [r["text"] for r in train_records],
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        val_enc = tokenizer(
            [r["text"] for r in val_records],
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        train_ds = EncDataset(train_enc, [label_to_id[r["label"]] for r in train_records])
        val_ds = EncDataset(val_enc, [label_to_id[r["label"]] for r in val_records])
        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model, num_labels=len(labels)
        )
        if torch.cuda.is_available():
            model.to("cuda")
        trainable_params = total_params = sum(p.numel() for p in model.parameters())
        warmup = G.compute_warmup_steps(
            len(train_records),
            model_spec["batch_size"],
            model_spec["grad_accum"],
            cfg["epochs"],
            cfg["warmup_ratio"],
        )
        targs = G.make_training_args(
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
        peak_train = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred_out = trainer.predict(val_ds)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_sec = time.perf_counter() - t1
        peak_eval = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        probs = torch.softmax(torch.tensor(pred_out.predictions, dtype=torch.float32), dim=-1)
        pred_ids = probs.argmax(dim=-1).tolist()
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
        y_pred = [labels[i] for i in pred_ids]
        report = G.M.evaluate([r["label"] for r in val_records], y_pred, labels)
        eval_eff = {
            "eval_time_sec": round(eval_sec, 2),
            "infer_samples_per_sec": round(len(val_records) / eval_sec, 2) if eval_sec else 0.0,
            "peak_eval_vram_gb": round(peak_eval, 3),
            "eval_truncated_count": 0,
        }
        if args.save_model:
            paths["encoder_dir"].mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(paths["encoder_dir"]))
            tokenizer.save_pretrained(str(paths["encoder_dir"]))
    else:
        sys_prompt = G.preprocess.get_system_prompt(task, "train")
        max_len = model_spec["max_length"]
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded_train = [
            G.encode_record(r, tokenizer, sys_prompt, max_len, False)[0]
            for r in train_records
        ]
        model, trainable_params, total_params = G.build_peft_model(
            model_name=resolved_model,
            precision=model_spec["precision"],
            lora_r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_scope=cfg["target_scope"],
            gradient_checkpointing=model_spec["gradient_checkpointing"],
        )
        warmup = G.compute_warmup_steps(
            len(encoded_train),
            model_spec["batch_size"],
            model_spec["grad_accum"],
            cfg["epochs"],
            cfg["warmup_ratio"],
        )
        targs = G.make_training_args(
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
            train_dataset=G.PolicyDataset(encoded_train),
            data_collator=G.make_collate_fn(tokenizer.pad_token_id),
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr_out = trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0
        peak_train = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        # 解包 accelerate 包装，避免评估时把全量 logits 转 fp32
        model = trainer.accelerator.unwrap_model(model)
        torch.cuda.empty_cache()
        if args.save_model:
            paths["adapter_dir"].mkdir(parents=True, exist_ok=True)
            model.save_pretrained(paths["adapter_dir"])
            tokenizer.save_pretrained(paths["adapter_dir"])
        report, pred_rows, eval_eff = G.evaluate_label_scores(
            model=model,
            tokenizer=tokenizer,
            records=val_records,
            system_prompt=sys_prompt,
            labels=labels,
            max_length=max_len,
            eval_batch_size=model_spec["eval_batch_size"],
            enable_thinking=False,
        )

    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    with paths["predictions"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    train_loss = getattr(tr_out, "training_loss", None)
    if train_loss is None:
        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        train_loss = sum(losses) / len(losses) if losses else 0.0

    summary = {
        "exp_id": exp_id,
        "stage": "G3",
        "model_id": model_spec["id"],
        "model_label": model_spec.get("label", model_spec["id"]),
        "model": resolved_model,
        "runner": model_spec["runner"],
        "task": task,
        "regime": regime,
        "per_class": per_class,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "gold_samples": gold_per_class * len(labels),
        "lora_r": 0 if model_spec["runner"] == "encoder" else cfg["lora_r"],
        "lora_alpha": 0 if model_spec["runner"] == "encoder" else cfg["lora_alpha"],
        "target_scope": "full" if model_spec["runner"] == "encoder" else cfg["target_scope"],
        "precision": model_spec["precision"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_params_pct": round(100.0 * trainable_params / total_params, 4),
        "train_time_sec": round(train_sec, 2),
        "peak_train_vram_gb": round(peak_train, 3),
        **eval_eff,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro_f1"], 4),
        "cohen_kappa": round(report["cohen_kappa"], 4),
        "train_loss": round(float(train_loss), 4),
        "global_step": int(tr_out.global_step),
        "labels": labels,
        "metrics_detail": report,
        "seed": seed,
        "data_sha": data_sha,
        "fingerprint": fingerprint,
        "source": "run",
        "finished_at": G.now_stamp(),
    }
    G.write_json(paths["summary"], summary)
    print(
        f"[DONE] {exp_id} ({task}, {regime}) Acc={summary['accuracy']:.4f} "
        f"G-F1/W-F1={report['per_class'].get('G', report['per_class'].get('W', {})).get('f1', 0):.4f} "
        f"Train={summary['train_time_sec']}s"
    )
    return 0


def summarize_g3(results_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        per_cls = data.get("metrics_detail", {}).get("per_class", {})
        task = data["task"]
        bottleneck = "G" if task == "task2_level2" else "W"
        rows.append(
            {
                "exp_id": data["exp_id"],
                "task": task,
                "regime": data.get("regime", ""),
                "model_id": data["model_id"],
                "model_label": data.get("model_label", data["model_id"]),
                "per_class": int(data.get("per_class", 0)),
                "train_samples": data.get("train_samples", 0),
                "accuracy": data["accuracy"],
                "macro_f1": data["macro_f1"],
                "cohen_kappa": data["cohen_kappa"],
                "bottleneck_class": bottleneck,
                "bottleneck_f1": round(per_cls.get(bottleneck, {}).get("f1", 0.0), 4),
                "train_time_sec": data.get("train_time_sec", 0.0),
                "peak_train_vram_gb": data.get("peak_train_vram_gb", 0.0),
                "infer_samples_per_sec": data.get("infer_samples_per_sec", 0.0),
                "train_loss": data.get("train_loss", 0.0),
                "source": data.get("source", "run"),
            }
        )
    if not rows:
        print("暂无 G3 结果")
        return

    rows.sort(key=lambda r: (r["task"], r["per_class"], r["model_id"]))
    baseline: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["task"], row["model_id"])
        prev = baseline.get(key)
        if prev is None:
            row["delta_acc"] = ""
            row["delta_bottleneck_f1"] = ""
        else:
            row["delta_acc"] = round(row["accuracy"] - prev["accuracy"], 4)
            row["delta_bottleneck_f1"] = round(
                row["bottleneck_f1"] - prev["bottleneck_f1"], 4
            )
        baseline[key] = row

    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "g3_scaling_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 06 G3 样本量 Scaling 报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 记录数：**{len(rows)}**",
        "- 纯金标准段：L1 50/100/150/200，L2 75/150/225/300；银标扩增段：L1 400/600/1000，L2 500/800/1200",
        "",
    ]
    for task, title, bottleneck in (
        ("task1_level1", "Level-1（W/R/N，瓶颈 W-F1）", "W"),
        ("task2_level2", "Level-2（B/Y/C/G，瓶颈 G-F1）", "G"),
    ):
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                f"| 数据段 | 单类 | 总量 | 模型 | Acc | ΔAcc | {bottleneck}-F1 | Δ{bottleneck}-F1 | MF1 | Kappa | 训练(s) | 显存(GB) | 来源 |",
                "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for r in sub:
            delta_acc = f"{r['delta_acc']:+.4f}" if r["delta_acc"] != "" else "—"
            delta_f1 = (
                f"{r['delta_bottleneck_f1']:+.4f}"
                if r["delta_bottleneck_f1"] != ""
                else "—"
            )
            lines.append(
                f"| {r['regime']} | {r['per_class']} | {r['train_samples']} | "
                f"{r['model_label']} | {r['accuracy']:.4f} | {delta_acc} | "
                f"{r['bottleneck_f1']:.4f} | {delta_f1} | {r['macro_f1']:.4f} | "
                f"{r['cohen_kappa']:.4f} | {r['train_time_sec']:.1f} | "
                f"{r['peak_train_vram_gb']:.2f} | {r['source']} |"
            )
        lines.append("")

    md_path = results_root / "g3_scaling_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[汇总] {len(rows)} 条 -> {csv_path}\n[汇总] {md_path}")


def build_jobs(
    cfg: dict[str, Any], manifest: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    available = {(s["task"], int(s["per_class"])): s for s in manifest["splits"]}
    jobs: list[dict[str, Any]] = []
    for task, spec in cfg["tasks"].items():
        if args.task and task not in args.task:
            continue
        tiers: list[int] = []
        if args.tier_mode in {"all", "gold_only"}:
            tiers.extend(int(x) for x in spec.get("gold_tiers", []))
        if args.tier_mode in {"all", "silver_only"}:
            tiers.extend(int(x) for x in spec.get("silver_tiers", []))
        for per_class in sorted(set(tiers)):
            unit = available.get((task, per_class))
            if unit is None:
                continue
            for model in cfg["models"]:
                if args.only and not any(k in model["id"] for k in args.only):
                    continue
                jobs.append(
                    {
                        "task": task,
                        "per_class": per_class,
                        "regime": unit["regime"],
                        "total": unit["total"],
                        "model": model,
                    }
                )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="G3 样本量 Scaling 编排")
    parser.add_argument("--mode", choices=["plan", "run", "all", "summarize", "worker"], default="plan")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--tier-mode", choices=["all", "gold_only", "silver_only"], default="all")
    parser.add_argument("--task", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="按模型 id 子串过滤")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--save-model", action="store_true", help="保存每个档位的 adapter/全参模型")
    args = parser.parse_args()

    if args.mode == "worker":
        return run_worker(args)

    cfg = G.read_json(args.config)
    manifest = load_manifest()
    jobs = build_jobs(cfg, manifest, args)

    if args.mode == "plan":
        total = len(jobs)
        reusable = sum(
            1
            for job in jobs
            if job["model"].get("reuse_gold_exp_id")
            and job["per_class"] == cfg["tasks"][job["task"]]["gold_per_class"]
        )
        print(f"G3 计划 {total} 个单元（可复用 {reusable}，需训练 {total - reusable}）")
        print("-" * 120)
        for job in jobs:
            model = job["model"]
            tag = (
                f"  [复用 {model['reuse_gold_exp_id']}]"
                if model.get("reuse_gold_exp_id")
                and job["per_class"] == cfg["tasks"][job["task"]]["gold_per_class"]
                else ""
            )
            print(
                f"  {job['task']:14s} | {job['regime']:16s} | n/class={job['per_class']:4d} "
                f"| total={job['total']:4d} | {model['id']:24s} | {model['model']}{tag}"
            )
        print("-" * 120)
        return 0

    if args.mode in {"run", "all"}:
        for job in jobs:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--config",
                args.config,
                "--task",
                job["task"],
                "--model-id",
                job["model"]["id"],
                "--per-class",
                str(job["per_class"]),
            ]
            if args.smoke:
                command.append("--smoke")
            if args.no_skip_existing:
                command.append("--no-skip-existing")
            if args.save_model:
                command.append("--save-model")
            print(
                f"\n>>> [{job['task']} | {job['regime']} | n={job['per_class']} | {job['model']['id']}]"
            )
            subprocess.run(command, check=True)

    if args.mode in {"summarize", "all"}:
        summarize_g3(G3_SMOKE_ROOT / "results" if args.smoke else G3_RESULTS_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
