# -*- coding: utf-8 -*-
"""G3-SSL 半监督 Scaling 唯一编排入口：plan / run / all / summarize / worker。

单次 worker 按 (model_id, per_class) 顺序完成两个阶段，节省 50% 4B 算力：
  1. Stage 1 (`ssl_co_mix`)    : 300 金 (w=1.0) + N 一致伪标签 (w=0.5)，3 epoch 混合训练并评测；
  2. Stage 2 (`ssl_co_polish`) : 复用 Stage 1 权重，在 300/类纯金标准上以 0.2×lr 精调 1 epoch 收尾并评测。

用法（仓库根目录）：
  python 06_generation_scale_benchmark_G3_SSL/scripts/run_g3_ssl.py --mode plan
  python 06_generation_scale_benchmark_G3_SSL/scripts/run_g3_ssl.py --mode run --smoke --only qwen3_4b
  nohup python 06_generation_scale_benchmark_G3_SSL/scripts/run_g3_ssl.py --mode all > 06_generation_scale_benchmark_G3_SSL/logs/run_ssl.log 2>&1 &
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
G1G2_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "06_generation_scale_benchmark_G1_G2"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(G1G2_SCRIPTS))

import gpu_guard  # noqa: E402
import ssl_common as S  # noqa: E402


def code_version() -> str:
    return S.code_fingerprint(
        [
            Path(__file__).resolve(),
            SCRIPTS_DIR / "ssl_common.py",
            S.G1G2_SCRIPTS / "benchmark_common.py",
        ]
    )


def load_manifest() -> dict[str, Any]:
    path = S.SSL_DATA_ROOT / "g3_ssl_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}，请先运行 06_generation_scale_benchmark_G3_SSL/scripts/prepare/build_ssl_datasets.py"
        )
    return S.read_json(path)


def unit_fingerprint(
    cfg: dict[str, Any],
    model_spec: dict[str, Any],
    regime_id: str,
    per_class: int,
    data_sha: str,
) -> str:
    return S.experiment_fingerprint(
        {
            "stage": "G3-SSL",
            "task": cfg["task"],
            "regime": regime_id,
            "per_class": per_class,
            "model_id": model_spec["id"],
            "model": model_spec["model"],
            "runner": model_spec["runner"],
            "precision": model_spec["precision"],
            "lr": model_spec["lr"],
            "silver_weight": cfg["silver_weight"],
            "polish_epochs": cfg["polish_epochs"] if regime_id == "ssl_co_polish" else 0,
            "polish_lr_factor": cfg["polish_lr_factor"] if regime_id == "ssl_co_polish" else 0,
            "epochs": cfg["epochs"],
            "lora_r": cfg["lora_r"],
            "lora_alpha": cfg["lora_alpha"],
            "seed": cfg["seed"],
            "data_sha": data_sha,
            "code": code_version(),
        }
    )


def save_stage_outputs(
    exp_id: str,
    regime_id: str,
    cfg: dict[str, Any],
    model_spec: dict[str, Any],
    resolved_model: str,
    per_class: int,
    train_count: int,
    val_count: int,
    labels: list[str],
    report: dict[str, Any],
    pred_rows: list[dict[str, Any]],
    eval_eff: dict[str, Any],
    train_sec: float,
    peak_train: float,
    train_loss: float,
    global_step: int,
    trainable_params: int,
    total_params: int,
    data_sha: str,
    fingerprint: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    with paths["predictions"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    summary = {
        "exp_id": exp_id,
        "stage": "G3-SSL",
        "regime": regime_id,
        "model_id": model_spec["id"],
        "model_label": model_spec.get("label", model_spec["id"]),
        "model": resolved_model,
        "runner": model_spec["runner"],
        "task": cfg["task"],
        "per_class": per_class,
        "train_samples": train_count,
        "val_samples": val_count,
        "gold_samples": int(cfg["gold_per_class"]) * len(labels),
        "silver_weight": cfg["silver_weight"],
        "lora_r": 0 if model_spec["runner"] == "encoder" else cfg["lora_r"],
        "lora_alpha": 0 if model_spec["runner"] == "encoder" else cfg["lora_alpha"],
        "precision": model_spec["precision"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_time_sec": round(train_sec, 2),
        "peak_train_vram_gb": round(peak_train, 3),
        **eval_eff,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro_f1"], 4),
        "cohen_kappa": round(report["cohen_kappa"], 4),
        "train_loss": round(float(train_loss), 4),
        "global_step": int(global_step),
        "labels": labels,
        "metrics_detail": report,
        "seed": int(cfg["seed"]),
        "data_sha": data_sha,
        "fingerprint": fingerprint,
        "finished_at": S.now_stamp(),
    }
    S.write_json(paths["summary"], summary)
    g_f1 = report["per_class"].get("G", {}).get("f1", 0.0)
    print(
        f"[DONE] {exp_id} ({regime_id}) Acc={summary['accuracy']:.4f} "
        f"G-F1={g_f1:.4f} MF1={summary['macro_f1']:.4f} Train={summary['train_time_sec']}s"
    )
    return summary


def run_worker(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = S.read_json(args.config)
    task = cfg["task"]
    labels = cfg["labels"]
    per_class = int(args.per_class)
    model_spec = next(m for m in cfg["models"] if m["id"] == args.model_id)
    seed = int(cfg["seed"])
    silver_weight = float(cfg["silver_weight"])

    results_root = (S.SSL_SMOKE_ROOT / "results") if args.smoke else S.SSL_RESULTS_ROOT
    model_root = (S.SSL_SMOKE_ROOT / "models") if args.smoke else S.SSL_MODEL_ROOT

    if args.smoke:
        data_sha = "smoke"
        train_records = [
            {**r, "source": "gold"} for r in gpu_guard.limit_smoke_samples(S.preprocess.load_split(task, "train"))
        ]
        val_records = gpu_guard.limit_smoke_samples(S.preprocess.load_split(task, "validation"))
    else:
        manifest = load_manifest()
        unit = next((s for s in manifest["splits"] if int(s["per_class"]) == per_class), None)
        if unit is None:
            raise FileNotFoundError(f"manifest 中缺少 per_class={per_class}")
        data_sha = unit["sha256"]
        train_records = S.read_jsonl(S.SSL_DATA_ROOT / task / f"train_per_class_{per_class}.jsonl")
        val_records = S.preprocess.load_split(task, "validation")

    gold_records = [r for r in train_records if r.get("source") == "gold"]
    exp_mix = f"G3SSL_mix_{model_spec['id']}_n{per_class}"
    exp_polish = f"G3SSL_polish_{model_spec['id']}_n{per_class}"
    paths_mix = S.run_paths(exp_mix, task, results_root, model_root)
    paths_polish = S.run_paths(exp_polish, task, results_root, model_root)
    fp_mix = unit_fingerprint(cfg, model_spec, "ssl_co_mix", per_class, data_sha)
    fp_polish = unit_fingerprint(cfg, model_spec, "ssl_co_polish", per_class, data_sha)

    skip_mix = (
        paths_mix["summary"].is_file()
        and not args.no_skip_existing
        and S.read_json(paths_mix["summary"]).get("fingerprint") == fp_mix
    )
    skip_polish = (
        paths_polish["summary"].is_file()
        and not args.no_skip_existing
        and S.read_json(paths_polish["summary"]).get("fingerprint") == fp_polish
    )
    if skip_mix and skip_polish:
        print(f"[SKIP] {exp_mix} & {exp_polish} 均已完成且指纹一致")
        return 0

    gpu_guard.assert_cloud_gpu_allowed(f"G3-SSL {model_spec['id']} n={per_class}")
    S.set_global_seed(seed)
    resolved_model = S.resolve_local_model_path(model_spec["model"])
    print(
        f"[START] {model_spec['id']} n/class={per_class} | train={len(train_records)} "
        f"(gold={len(gold_records)}, ssl={len(train_records) - len(gold_records)}, w={silver_weight})"
    )

    if model_spec["runner"] == "encoder":

        class WeightedEncDataset(Dataset):
            def __init__(self, enc, label_ids, weights):
                self.enc = enc
                self.label_ids = label_ids
                self.weights = weights

            def __len__(self) -> int:
                return len(self.label_ids)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                item = {k: v[idx] for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
                item["sample_weight"] = torch.tensor(self.weights[idx], dtype=torch.float32)
                return item

        label_to_id = {lb: i for i, lb in enumerate(labels)}
        tok = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
        max_len = model_spec["max_length"]

        def build_enc_ds(recs):
            enc = tok(
                [r["text"] for r in recs],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            w = [1.0 if r.get("source") == "gold" else silver_weight for r in recs]
            return WeightedEncDataset(enc, [label_to_id[r["label"]] for r in recs], w)

        tr_ds = build_enc_ds(train_records)
        gold_ds = build_enc_ds(gold_records)
        val_ds = build_enc_ds(val_records)

        model = AutoModelForSequenceClassification.from_pretrained(resolved_model, num_labels=len(labels))
        if torch.cuda.is_available():
            model.to("cuda")
        trainable_params = total_params = sum(p.numel() for p in model.parameters())

        def eval_encoder(trainer_obj):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            pred_out = trainer_obj.predict(val_ds)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            eval_sec = time.perf_counter() - t1
            peak_eval = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            probs = torch.softmax(torch.tensor(pred_out.predictions, dtype=torch.float32), dim=-1)
            pred_ids = probs.argmax(dim=-1).tolist()
            pred_rows = []
            for rec, pid, pvec in zip(val_records, pred_ids, probs.tolist()):
                row = {"gold": rec["label"], "pred": labels[pid], "correct": int(labels[pid] == rec["label"]), "text": rec["text"]}
                for lb, pv in zip(labels, pvec):
                    row[f"prob_{lb}"] = round(float(pv), 6)
                pred_rows.append(row)
            report = S.M.evaluate([r["label"] for r in val_records], [labels[i] for i in pred_ids], labels)
            eff = {
                "eval_time_sec": round(eval_sec, 2),
                "infer_samples_per_sec": round(len(val_records) / eval_sec, 2) if eval_sec else 0.0,
                "peak_eval_vram_gb": round(peak_eval, 3),
                "eval_truncated_count": 0,
            }
            return report, pred_rows, eff

        # Stage 1: Mix
        warmup1 = S.compute_warmup_steps(len(train_records), model_spec["batch_size"], model_spec["grad_accum"], cfg["epochs"], cfg["warmup_ratio"])
        targs1 = S.make_training_args(
            output_dir=paths_mix["checkpoints"],
            epochs=cfg["epochs"],
            batch_size=model_spec["batch_size"],
            grad_accum=model_spec["grad_accum"],
            lr=model_spec["lr"],
            precision=model_spec["precision"],
            warmup_steps=warmup1,
            seed=seed,
            max_steps=1 if args.smoke else -1,
        )
        trainer1 = S.WeightedTrainer(model=model, args=targs1, train_dataset=tr_ds, processing_class=tok)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr1 = trainer1.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec1 = time.perf_counter() - t0
        vram1 = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        rep1, prows1, eff1 = eval_encoder(trainer1)
        loss1 = getattr(tr1, "training_loss", 0.0) or 0.0
        save_stage_outputs(
            exp_mix, "ssl_co_mix", cfg, model_spec, resolved_model, per_class,
            len(train_records), len(val_records), labels, rep1, prows1, eff1,
            sec1, vram1, loss1, tr1.global_step, trainable_params, total_params,
            data_sha, fp_mix, paths_mix,
        )

        # Stage 2: Gold Polish
        polish_lr = float(model_spec["lr"]) * float(cfg["polish_lr_factor"])
        warmup2 = S.compute_warmup_steps(len(gold_records), model_spec["batch_size"], model_spec["grad_accum"], cfg["polish_epochs"], cfg["warmup_ratio"])
        targs2 = S.make_training_args(
            output_dir=paths_polish["checkpoints"],
            epochs=cfg["polish_epochs"],
            batch_size=model_spec["batch_size"],
            grad_accum=model_spec["grad_accum"],
            lr=polish_lr,
            precision=model_spec["precision"],
            warmup_steps=warmup2,
            seed=seed,
            max_steps=1 if args.smoke else -1,
        )
        model.train()
        trainer2 = S.WeightedTrainer(model=model, args=targs2, train_dataset=gold_ds, processing_class=tok)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        tr2 = trainer2.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec2 = sec1 + (time.perf_counter() - t2)
        vram2 = max(vram1, torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0)
        rep2, prows2, eff2 = eval_encoder(trainer2)
        loss2 = getattr(tr2, "training_loss", 0.0) or 0.0
        save_stage_outputs(
            exp_polish, "ssl_co_polish", cfg, model_spec, resolved_model, per_class,
            len(train_records), len(val_records), labels, rep2, prows2, eff2,
            sec2, vram2, loss2, tr1.global_step + tr2.global_step, trainable_params, total_params,
            data_sha, fp_polish, paths_polish,
        )
    else:

        class WeightedCausalDataset(Dataset):
            def __init__(self, examples: list[dict[str, Any]]):
                self.examples = examples

            def __len__(self) -> int:
                return len(self.examples)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return self.examples[idx]

        sys_prompt = S.preprocess.get_system_prompt(task, "train")
        max_len = model_spec["max_length"]
        tok = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def encode_weighted(recs):
            out = []
            for r in recs:
                ex = S.encode_record(r, tok, sys_prompt, max_len, False)[0]
                ex["sample_weight"] = 1.0 if r.get("source") == "gold" else silver_weight
                out.append(ex)
            return out

        encoded_mix = encode_weighted(train_records)
        encoded_gold = encode_weighted(gold_records)

        model, trainable_params, total_params = S.build_peft_model(
            model_name=resolved_model,
            precision=model_spec["precision"],
            lora_r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_scope=cfg["target_scope"],
            gradient_checkpointing=model_spec["gradient_checkpointing"],
        )

        # Stage 1: Mix
        warmup1 = S.compute_warmup_steps(len(encoded_mix), model_spec["batch_size"], model_spec["grad_accum"], cfg["epochs"], cfg["warmup_ratio"])
        targs1 = S.make_training_args(
            output_dir=paths_mix["checkpoints"],
            epochs=cfg["epochs"],
            batch_size=model_spec["batch_size"],
            grad_accum=model_spec["grad_accum"],
            lr=model_spec["lr"],
            precision=model_spec["precision"],
            warmup_steps=warmup1,
            seed=seed,
            max_steps=1 if args.smoke else -1,
        )
        trainer1 = S.WeightedTrainer(
            model=model,
            args=targs1,
            train_dataset=WeightedCausalDataset(encoded_mix),
            data_collator=S.make_weighted_collate_fn(tok.pad_token_id),
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr1 = trainer1.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec1 = time.perf_counter() - t0
        vram1 = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        eval_model = trainer1.accelerator.unwrap_model(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rep1, prows1, eff1 = S.evaluate_label_scores(
            model=eval_model,
            tokenizer=tok,
            records=val_records,
            system_prompt=sys_prompt,
            labels=labels,
            max_length=max_len,
            eval_batch_size=model_spec["eval_batch_size"],
            enable_thinking=False,
        )
        loss1 = getattr(tr1, "training_loss", 0.0) or 0.0
        save_stage_outputs(
            exp_mix, "ssl_co_mix", cfg, model_spec, resolved_model, per_class,
            len(train_records), len(val_records), labels, rep1, prows1, eff1,
            sec1, vram1, loss1, tr1.global_step, trainable_params, total_params,
            data_sha, fp_mix, paths_mix,
        )

        # Stage 2: Gold Polish
        polish_lr = float(model_spec["lr"]) * float(cfg["polish_lr_factor"])
        warmup2 = S.compute_warmup_steps(len(encoded_gold), model_spec["batch_size"], model_spec["grad_accum"], cfg["polish_epochs"], cfg["warmup_ratio"])
        targs2 = S.make_training_args(
            output_dir=paths_polish["checkpoints"],
            epochs=cfg["polish_epochs"],
            batch_size=model_spec["batch_size"],
            grad_accum=model_spec["grad_accum"],
            lr=polish_lr,
            precision=model_spec["precision"],
            warmup_steps=warmup2,
            seed=seed,
            max_steps=1 if args.smoke else -1,
        )
        eval_model.train()
        trainer2 = S.WeightedTrainer(
            model=eval_model,
            args=targs2,
            train_dataset=WeightedCausalDataset(encoded_gold),
            data_collator=S.make_weighted_collate_fn(tok.pad_token_id),
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        tr2 = trainer2.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec2 = sec1 + (time.perf_counter() - t2)
        vram2 = max(vram1, torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0)

        eval_model = trainer2.accelerator.unwrap_model(eval_model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rep2, prows2, eff2 = S.evaluate_label_scores(
            model=eval_model,
            tokenizer=tok,
            records=val_records,
            system_prompt=sys_prompt,
            labels=labels,
            max_length=max_len,
            eval_batch_size=model_spec["eval_batch_size"],
            enable_thinking=False,
        )
        loss2 = getattr(tr2, "training_loss", 0.0) or 0.0
        save_stage_outputs(
            exp_polish, "ssl_co_polish", cfg, model_spec, resolved_model, per_class,
            len(train_records), len(val_records), labels, rep2, prows2, eff2,
            sec2, vram2, loss2, tr1.global_step + tr2.global_step, trainable_params, total_params,
            data_sha, fp_polish, paths_polish,
        )

    return 0


def summarize_ssl(results_root: Path, cfg: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    # 1. 拉取 G3 原生 R_2 银标结果作横向参照（n=300, 500, 800, 1200）
    for p in sorted(S.G3_RESULTS_ROOT.glob("G3_*__task2_level2/summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("model_id") not in {m["id"] for m in cfg["models"]}:
            continue
        if int(d.get("per_class", 0)) not in {300, 500, 800, 1200}:
            continue
        pc = d["metrics_detail"]["per_class"]
        rows.append(
            {
                "regime": "g3_native_r2" if int(d["per_class"]) > 300 else "g3_pure_gold",
                "model_id": d["model_id"],
                "model_label": d.get("model_label", d["model_id"]),
                "per_class": int(d["per_class"]),
                "train_samples": int(d["train_samples"]),
                "accuracy": d["accuracy"],
                "macro_f1": d["macro_f1"],
                "cohen_kappa": d["cohen_kappa"],
                "G_f1": round(pc["G"]["f1"], 4),
                "G_p": round(pc["G"]["precision"], 4),
                "G_r": round(pc["G"]["recall"], 4),
                "train_time_sec": d.get("train_time_sec", 0.0),
                "peak_train_vram_gb": d.get("peak_train_vram_gb", 0.0),
            }
        )

    # 2. 收集 G3-SSL 结果
    for p in sorted(results_root.glob("G3SSL_*__task2_level2/summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        pc = d["metrics_detail"]["per_class"]
        rows.append(
            {
                "regime": d["regime"],
                "model_id": d["model_id"],
                "model_label": d.get("model_label", d["model_id"]),
                "per_class": int(d["per_class"]),
                "train_samples": int(d["train_samples"]),
                "accuracy": d["accuracy"],
                "macro_f1": d["macro_f1"],
                "cohen_kappa": d["cohen_kappa"],
                "G_f1": round(pc["G"]["f1"], 4),
                "G_p": round(pc["G"]["precision"], 4),
                "G_r": round(pc["G"]["recall"], 4),
                "train_time_sec": d.get("train_time_sec", 0.0),
                "peak_train_vram_gb": d.get("peak_train_vram_gb", 0.0),
            }
        )

    if not rows:
        print("暂无 G3-SSL 结果")
        return

    rows.sort(key=lambda r: (r["model_id"], r["per_class"], r["regime"]))
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "g3_ssl_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 06 G3-SSL Level-2 半监督去噪 Scaling 对照报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 记录数：**{len(rows)}**（含 G3 原生 R_2 基线横向对照）",
        "- 方案：`g3_native_r2`（原 R_2 等权混合） vs `ssl_co_mix`（双模型一致性伪标签 w=0.5） vs `ssl_co_polish`（+1 epoch 纯金标准精调）",
        "",
        "| 模型 | 单类 | 总量 | 策略 | Acc | MF1 | Kappa | G-F1 | G-P | G-R | 耗时(s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_label']} | {r['per_class']} | {r['train_samples']} | {r['regime']} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['cohen_kappa']:.4f} | "
            f"{r['G_f1']:.4f} | {r['G_p']:.4f} | {r['G_r']:.4f} | {r['train_time_sec']:.1f} |"
        )
    md_path = results_root / "g3_ssl_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[汇总] {len(rows)} 条 -> {csv_path}\n[汇总] {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="G3-SSL 半监督编排")
    parser.add_argument("--mode", choices=["plan", "run", "all", "summarize", "worker"], default="plan")
    parser.add_argument("--config", default=str(S.SSL_CONFIG))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    if args.mode == "worker":
        return run_worker(args)

    cfg = S.read_json(args.config)
    models = [
        m for m in cfg["models"]
        if not args.only or any(k in m["id"] for k in args.only)
    ]
    tiers = sorted(int(x) for x in cfg["ssl_tiers"])
    jobs = [{"model": m, "per_class": n} for n in tiers for m in models]

    if args.mode == "plan":
        print(f"G3-SSL 计划 {len(jobs)} 个训练单元（每单元连续产出 ssl_co_mix + ssl_co_polish，共 {len(jobs)*2} 条结果）")
        print("-" * 96)
        for j in jobs:
            m = j["model"]
            print(f"  {cfg['task']:14s} | n/class={j['per_class']:4d} | {m['id']:24s} | {m['model']}")
        print("-" * 96)
        return 0

    if args.mode in {"run", "all"}:
        for j in jobs:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode", "worker",
                "--config", args.config,
                "--model-id", j["model"]["id"],
                "--per-class", str(j["per_class"]),
            ]
            if args.smoke:
                cmd.append("--smoke")
            if args.no_skip_existing:
                cmd.append("--no-skip-existing")
            print(f"\n>>> [G3-SSL | n={j['per_class']} | {j['model']['id']}]")
            subprocess.run(cmd, check=True)

    if args.mode in {"summarize", "all"}:
        res_root = (S.SSL_SMOKE_ROOT / "results") if args.smoke else S.SSL_RESULTS_ROOT
        summarize_ssl(res_root, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
