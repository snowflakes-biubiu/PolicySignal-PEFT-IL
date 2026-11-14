# -*- coding: utf-8 -*-
"""G3-SSL 数据构建：纯金标准双 Teacher 训练 → 无标注池打分 → 异构一致性 + 难度分层伪标签。

核心流程（默认仅针对 task2_level2）：
1. 读取全量语料 `CN_DOCS_FINAL_complete.parquet`，按文档级零泄漏规则排除命中金标准
   train/validation 的全部文档，仅取 `R_1 == 'W'` 的肯定型政策指令段落作为无标注候选池
   （完全不使用 `R_2` 伪标签）；
2. 在 300/类纯金标准训练集上训练（或加载已缓存的）两个异构 Teacher：
   - Teacher A: `hfl/chinese-roberta-wwm-ext-large` (bf16 全参)
   - Teacher B: `Qwen/Qwen3-4B` (bf16 LoRA r=8/α=16, attn_mlp, GC)
3. 双 Teacher 对无标注候选池逐条做 4 类受限概率预测，仅保留预测类别一致且达到置信度
   门槛的样本；
4. 在一致样本中按 70% 高置信度 (≥0.90) + 30% 中置信度甜区 (<0.90) 分层抽样，构建严格嵌套的
   500 / 800 / 1200 每类半监督训练集（300 金 + 200/500/900 伪标签）。

用法（仓库根目录）：
  python 06_generation_scale_benchmark_G3_SSL/scripts/prepare/build_ssl_datasets.py
  python 06_generation_scale_benchmark_G3_SSL/scripts/prepare/build_ssl_datasets.py --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
G1G2_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "06_generation_scale_benchmark_G1_G2"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(G1G2_SCRIPTS))

import gpu_guard  # noqa: E402
import ssl_common as S  # noqa: E402


def norm_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def find_corpus(override: str | None, cfg: dict[str, Any]) -> Path:
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"指定的语料文件不存在: {p}")
    filename = cfg["pool"]["corpus_file"]
    candidates = [
        Path("/root/autodl-tmp/hf_models/hub/datasets--Baron-Sun--CAPC-CG_V1.0/snapshots/local/data") / filename,
        S.PROJECT_ROOT / "assets" / "shared" / "datasets" / "raw" / "CAPC-CG_V1.0" / "data" / filename,
    ]
    try:
        candidates.insert(0, S.preprocess.find_snapshot() / "data" / filename)
    except Exception:
        pass
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"未找到全量语料 {filename}，请通过 --corpus-parquet 指定路径")


def extract_unlabeled_pool(
    corpus: Path, cfg: dict[str, Any], smoke: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    pool_cfg = cfg["pool"]
    text_col = pool_cfg["text_column"]
    doc_col = pool_cfg["doc_column"]
    filter_col = pool_cfg["filter_column"]
    filter_val = pool_cfg["filter_value"]
    max_para = int(pool_cfg["max_para_per_doc"])
    min_len = int(pool_cfg["min_char_len"])
    max_len = int(pool_cfg["max_char_len"])
    target_size = 200 if smoke else int(pool_cfg["candidate_pool_size"])
    seed = int(cfg["seed"])

    forbidden: set[str] = set()
    for t in ("task1_level1", "task2_level2"):
        for split in ("train", "validation"):
            for r in S.preprocess.load_split(t, split):
                forbidden.add(norm_text(r["text"]))

    print(f"[候选池] 读取 {corpus.name} ({filter_col} == {filter_val!r}) ...")
    df = pd.read_parquet(corpus, columns=[doc_col, text_col, filter_col])
    df = df[df[filter_col].astype(str).str.strip() == filter_val].copy()
    df["text"] = df[text_col].fillna("").astype(str).str.strip()
    lengths = df["text"].str.len()
    df = df[(lengths >= min_len) & (lengths <= max_len)].copy()
    df["_norm"] = df["text"].map(norm_text)

    hit_docs = set(df.loc[df["_norm"].isin(forbidden), doc_col].astype(str))
    excluded_rows = int(df[doc_col].astype(str).isin(hit_docs).sum())
    df = df[~df[doc_col].astype(str).isin(hit_docs)].copy()
    print(f"[候选池] 文档级排除命中金标准文档 {len(hit_docs)} 个 / {excluded_rows} 行")

    rng = random.Random(seed)
    by_doc: dict[str, list[int]] = defaultdict(list)
    for idx, doc_id in enumerate(df[doc_col].astype(str)):
        by_doc[doc_id].append(idx)

    kept: list[int] = []
    for idxs in by_doc.values():
        if len(idxs) > max_para:
            idxs = rng.sample(idxs, max_para)
        kept.extend(idxs)

    sub = df.iloc[kept]
    pool: list[dict[str, Any]] = []
    seen = set(forbidden)
    for doc_id, text, nt in zip(
        sub[doc_col].astype(str).tolist(),
        sub["text"].tolist(),
        sub["_norm"].tolist(),
    ):
        if not nt or nt in seen:
            continue
        seen.add(nt)
        pool.append(
            {
                "doc_id": doc_id,
                "text": text,
                "user_raw": f"{S.preprocess.USER_PREFIX}\n{text}",
            }
        )

    rng.shuffle(pool)
    if len(pool) > target_size:
        pool = pool[:target_size]

    stats = {
        "corpus": str(corpus),
        "excluded_docs": len(hit_docs),
        "excluded_rows": excluded_rows,
        "pool_size": len(pool),
    }
    print(f"[候选池] 零泄漏无标注候选池就绪: {len(pool)} 条")
    return pool, stats


class _EncDataset(Dataset):
    def __init__(self, enc: dict[str, Any], label_ids: list[int]):
        self.enc = enc
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
        return item


class _CausalDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


@torch.no_grad()
def predict_roberta_probs(
    model_dir: Path,
    records: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    device = next(model.parameters()).device

    probs_all: list[list[float]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        enc = tokenizer(
            [r["text"] for r in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        probs_all.extend(probs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs_all


@torch.no_grad()
def predict_qwen_probs(
    base_model_path: str,
    adapter_dir: Path,
    records: list[dict[str, Any]],
    sys_prompt: str,
    labels: list[str],
    batch_size: int,
    max_length: int,
) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    candidate_ids = []
    for lb in labels:
        ids = tokenizer(lb, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"标签 {lb!r} 非单 token: {ids}")
        candidate_ids.append(ids[0])

    encoded_prompts = []
    for r in records:
        pids = S.build_prompt_ids(tokenizer, sys_prompt, r["user_raw"], enable_thinking=False)
        pids, _ = S.truncate_head_tail(pids, max_length - 1)
        encoded_prompts.append(pids)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = getattr(model, "device", None) or next(model.parameters()).device

    probs_all: list[list[float]] = []
    t0 = time.perf_counter()
    for start in range(0, len(encoded_prompts), batch_size):
        batch_p = encoded_prompts[start : start + batch_size]
        max_len = max(len(x) for x in batch_p)
        input_ids, attn_mask, last_pos = [], [], []
        for ids in batch_p:
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attn_mask.append([1] * len(ids) + [0] * pad_n)
            last_pos.append(len(ids) - 1)
        inp = torch.tensor(input_ids, dtype=torch.long, device=device)
        msk = torch.tensor(attn_mask, dtype=torch.long, device=device)
        logits = model(input_ids=inp, attention_mask=msk).logits
        row_idx = torch.arange(len(batch_p), device=device)
        pos_idx = torch.tensor(last_pos, device=device)
        next_logits = logits[row_idx, pos_idx][:, candidate_ids].float()
        probs = torch.softmax(next_logits, dim=-1).cpu().tolist()
        probs_all.extend(probs)
        if (start // batch_size + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [Qwen3-4B 打分] {start + len(batch_p)}/{len(encoded_prompts)} ({elapsed:.1f}s)")

    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs_all


def ensure_teachers(
    cfg: dict[str, Any],
    gold_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    smoke: bool,
) -> dict[str, Path]:
    labels = cfg["labels"]
    seed = int(cfg["seed"])
    S.set_global_seed(seed)
    teacher_root = (S.SSL_SMOKE_ROOT / "teachers") if smoke else S.SSL_TEACHER_ROOT
    teacher_dirs: dict[str, Path] = {}

    for mspec in cfg["models"]:
        mid = mspec["id"]
        tdir = teacher_root / f"teacher_{mid}_{cfg['task']}"
        teacher_dirs[mid] = tdir
        done_flag = tdir / "teacher_meta.json"
        if done_flag.is_file():
            print(f"[Teacher] 复用已训练金标准 Teacher: {tdir}")
            continue

        gpu_guard.assert_cloud_gpu_allowed(f"G3-SSL Teacher {mid}")
        resolved = S.resolve_local_model_path(mspec["model"])
        print(f"[Teacher] 训练纯金标准 Teacher ({mid}, n={len(gold_records)}) ...")
        t0 = time.perf_counter()

        if mspec["runner"] == "encoder":
            label_to_id = {lb: i for i, lb in enumerate(labels)}
            tok = AutoTokenizer.from_pretrained(resolved, use_fast=True)
            tr_enc = tok(
                [r["text"] for r in gold_records],
                truncation=True,
                max_length=mspec["max_length"],
                padding="max_length",
                return_tensors="pt",
            )
            tr_ds = _EncDataset(tr_enc, [label_to_id[r["label"]] for r in gold_records])
            model = AutoModelForSequenceClassification.from_pretrained(resolved, num_labels=len(labels))
            if torch.cuda.is_available():
                model.to("cuda")
            warmup = S.compute_warmup_steps(len(gold_records), mspec["batch_size"], mspec["grad_accum"], cfg["epochs"], cfg["warmup_ratio"])
            targs = S.make_training_args(
                output_dir=tdir / "ckpts",
                epochs=cfg["epochs"],
                batch_size=mspec["batch_size"],
                grad_accum=mspec["grad_accum"],
                lr=mspec["lr"],
                precision=mspec["precision"],
                warmup_steps=warmup,
                seed=seed,
                max_steps=1 if smoke else -1,
            )
            trainer = Trainer(model=model, args=targs, train_dataset=tr_ds, processing_class=tok)
            trainer.train()
            tdir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(tdir))
            tok.save_pretrained(str(tdir))
            del model, trainer
        else:
            sys_prompt = S.preprocess.get_system_prompt(cfg["task"], "train")
            tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            encoded = [S.encode_record(r, tok, sys_prompt, mspec["max_length"], False)[0] for r in gold_records]
            model, _, _ = S.build_peft_model(
                model_name=resolved,
                precision=mspec["precision"],
                lora_r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_scope=cfg["target_scope"],
                gradient_checkpointing=mspec["gradient_checkpointing"],
            )
            warmup = S.compute_warmup_steps(len(encoded), mspec["batch_size"], mspec["grad_accum"], cfg["epochs"], cfg["warmup_ratio"])
            targs = S.make_training_args(
                output_dir=tdir / "ckpts",
                epochs=cfg["epochs"],
                batch_size=mspec["batch_size"],
                grad_accum=mspec["grad_accum"],
                lr=mspec["lr"],
                precision=mspec["precision"],
                warmup_steps=warmup,
                seed=seed,
                max_steps=1 if smoke else -1,
            )
            trainer = Trainer(
                model=model,
                args=targs,
                train_dataset=_CausalDataset(encoded),
                data_collator=S.C.make_collate_fn(tok.pad_token_id),
            )
            trainer.train()
            model = trainer.accelerator.unwrap_model(model)
            tdir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(tdir)
            tok.save_pretrained(tdir)
            del model, trainer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        S.write_json(done_flag, {"model_id": mid, "train_sec": round(time.perf_counter() - t0, 2), "n": len(gold_records)})
        print(f"[Teacher] {mid} 保存完成 -> {tdir}")

    return teacher_dirs


def build_nested_ssl_tiers(
    cfg: dict[str, Any],
    gold_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    scored_pool: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    labels = cfg["labels"]
    seed = int(cfg["seed"])
    gold_per_class = int(cfg["gold_per_class"])
    pool_cfg = cfg["pool"]
    min_single = float(pool_cfg["min_single_prob"])
    min_mean = float(pool_cfg["min_mean_prob"])
    high_thresh = float(pool_cfg["high_conf_threshold"])
    high_ratio = float(pool_cfg["high_conf_ratio"])

    val_norms = {norm_text(r["text"]) for r in val_records}

    # 按类别整理双模型一致性高置信伪标签
    agreed_by_label: dict[str, list[dict[str, Any]]] = {lb: [] for lb in labels}
    agree_total = 0
    for rec in scored_pool:
        if rec["pred_roberta"] != rec["pred_qwen"]:
            continue
        agree_total += 1
        if min(rec["prob_roberta"], rec["prob_qwen"]) < min_single:
            continue
        if rec["mean_prob"] < min_mean:
            continue
        if norm_text(rec["text"]) in val_norms:
            raise AssertionError("SSL 候选池意外命中验证集")
        agreed_by_label[rec["pred_qwen"]].append(
            {
                "text": rec["text"],
                "user_raw": rec["user_raw"],
                "label": rec["pred_qwen"],
                "source": "ssl_co",
                "prob_roberta": rec["prob_roberta"],
                "prob_qwen": rec["prob_qwen"],
                "mean_prob": rec["mean_prob"],
            }
        )

    rng = random.Random(seed)
    ordered_pseudo: dict[str, list[dict[str, Any]]] = {}
    pool_stats: dict[str, dict[str, int]] = {}
    for lb in labels:
        items = agreed_by_label[lb]
        high_bucket = [x for x in items if x["mean_prob"] >= high_thresh]
        mid_bucket = [x for x in items if x["mean_prob"] < high_thresh]
        rng.shuffle(high_bucket)
        rng.shuffle(mid_bucket)

        # 按 7:3 交错排列，保证前 K 条切片天然保持高置信与边界甜区比例，同时严格嵌套
        interleaved: list[dict[str, Any]] = []
        hi, mi = 0, 0
        while hi < len(high_bucket) or mi < len(mid_bucket):
            want_high = int((len(interleaved) + 1) * high_ratio)
            cur_high = sum(1 for x in interleaved if x["mean_prob"] >= high_thresh)
            if (cur_high < want_high and hi < len(high_bucket)) or mi >= len(mid_bucket):
                interleaved.append(high_bucket[hi])
                hi += 1
            else:
                interleaved.append(mid_bucket[mi])
                mi += 1
        ordered_pseudo[lb] = interleaved
        pool_stats[lb] = {
            "total_agreed_filtered": len(items),
            "high_conf": len(high_bucket),
            "mid_conf": len(mid_bucket),
        }
    print(f"[伪标签筛选] 双模型一致 {agree_total}/{len(scored_pool)}；过滤后分桶: {pool_stats}")

    gold_by_label: dict[str, list[dict[str, Any]]] = {lb: [] for lb in labels}
    for r in gold_records:
        gold_by_label[r["label"]].append(
            {
                "text": r["text"],
                "user_raw": r["user_raw"],
                "label": r["label"],
                "source": "gold",
            }
        )
    for lb in labels:
        assert len(gold_by_label[lb]) == gold_per_class

    splits: list[dict[str, Any]] = []
    for per_class in sorted(int(x) for x in cfg["ssl_tiers"]):
        need_pseudo = per_class - gold_per_class
        avail = min(len(ordered_pseudo[lb]) for lb in labels)
        if avail < need_pseudo:
            raise RuntimeError(
                f"per_class={per_class} 每类需 {need_pseudo} 条一致伪标签，当前最小类仅 {avail} 条。"
                f"请增大 candidate_pool_size 或略降 min_mean_prob。"
            )
        selected: list[dict[str, Any]] = []
        for lb in labels:
            selected.extend(gold_by_label[lb])
            selected.extend(ordered_pseudo[lb][:need_pseudo])
        rng.shuffle(selected)

        out_path = out_dir / f"train_per_class_{per_class}.jsonl"
        S.write_jsonl(out_path, selected)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
        gold_n = gold_per_class * len(labels)
        pseudo_n = need_pseudo * len(labels)
        splits.append(
            {
                "task": cfg["task"],
                "per_class": per_class,
                "total": len(selected),
                "gold_count": gold_n,
                "pseudo_count": pseudo_n,
                "file": out_path.relative_to(S.PROJECT_ROOT).as_posix(),
                "sha256": digest,
            }
        )
        print(f"[OK] n/class={per_class:4d} total={len(selected):4d} gold={gold_n} pseudo={pseudo_n} -> {out_path.name}")

    return {
        "agree_total": agree_total,
        "pool_stats": pool_stats,
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 G3-SSL 半监督嵌套训练集")
    parser.add_argument("--config", default=str(S.SSL_CONFIG))
    parser.add_argument("--corpus-parquet", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = S.read_json(args.config)
    task = cfg["task"]
    labels = cfg["labels"]
    gold_records = S.preprocess.load_split(task, "train")
    val_records = S.preprocess.load_split(task, "validation")
    if args.smoke:
        gold_records = gpu_guard.limit_smoke_samples(gold_records)
        val_records = gpu_guard.limit_smoke_samples(val_records)

    corpus = find_corpus(args.corpus_parquet, cfg)
    pool_records, pool_meta = extract_unlabeled_pool(corpus, cfg, args.smoke)
    teacher_dirs = ensure_teachers(cfg, gold_records, val_records, args.smoke)

    rob_spec = next(m for m in cfg["models"] if m["runner"] == "encoder")
    qwen_spec = next(m for m in cfg["models"] if m["runner"] == "peft")

    print("[打分] Teacher-A (RoBERTa-large) 推理无标注候选池 ...")
    rob_probs = predict_roberta_probs(
        teacher_dirs[rob_spec["id"]],
        pool_records,
        rob_spec["eval_batch_size"],
        rob_spec["max_length"],
    )

    print("[打分] Teacher-B (Qwen3-4B) 推理无标注候选池 ...")
    sys_prompt = S.preprocess.get_system_prompt(task, "train")
    qwen_probs = predict_qwen_probs(
        S.resolve_local_model_path(qwen_spec["model"]),
        teacher_dirs[qwen_spec["id"]],
        pool_records,
        sys_prompt,
        labels,
        qwen_spec["eval_batch_size"],
        qwen_spec["max_length"],
    )

    scored_pool: list[dict[str, Any]] = []
    for rec, rp, qp in zip(pool_records, rob_probs, qwen_probs):
        ri = int(max(range(len(labels)), key=lambda i: rp[i]))
        qi = int(max(range(len(labels)), key=lambda i: qp[i]))
        scored_pool.append(
            {
                **rec,
                "pred_roberta": labels[ri],
                "prob_roberta": round(float(rp[ri]), 6),
                "pred_qwen": labels[qi],
                "prob_qwen": round(float(qp[qi]), 6),
                "mean_prob": round(0.5 * (float(rp[ri]) + float(qp[qi])), 6),
            }
        )

    data_root = (S.SSL_SMOKE_ROOT / "data") if args.smoke else S.SSL_DATA_ROOT
    out_dir = data_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    S.write_jsonl(out_dir / "scored_pool.jsonl", scored_pool)

    if args.smoke:
        print("[SMOKE] 打分与候选池流水线验证完成")
        return 0

    tier_info = build_nested_ssl_tiers(cfg, gold_records, val_records, scored_pool, out_dir)
    manifest_path = data_root / "g3_ssl_manifest.json"
    S.write_json(
        manifest_path,
        {
            "version": cfg["version"],
            "stage": "G3-SSL",
            "task": task,
            "seed": int(cfg["seed"]),
            "pool_meta": pool_meta,
            "agree_total": tier_info["agree_total"],
            "pool_stats": tier_info["pool_stats"],
            "splits": tier_info["splits"],
        },
    )
    print(f"[清单] 已写入 {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
