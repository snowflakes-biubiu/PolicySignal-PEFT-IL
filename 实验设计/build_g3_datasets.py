# -*- coding: utf-8 -*-
"""06 期 G3 阶段数据集构建脚本：放入 06_generation_scale_benchmark/scripts/prepare/ 下运行。

核心设计：
1. 将 G3 样本量梯度拆分为两段，避免「人工金标准」与「R1/R2 伪标签噪声」混杂：
   - 纯金标准段 (pure_gold)：
     * Level-1 (W/R/N)：每类 50 / 100 / 150 / 200 条（总计 150 / 300 / 450 / 600 条）
     * Level-2 (B/Y/C/G)：每类 75 / 150 / 225 / 300 条（总计 300 / 600 / 900 / 1200 条）
     严格采用嵌套抽样（50 ⊂ 100 ⊂ 150 ⊂ 200），保证相邻档位只增加样本、不替换样本。
   - 银标扩增段 (gold_plus_silver)：
     * Level-1：每类 400 / 600 / 1000 条（在 200 条金标准基础上，从 R_1 语料补充）
     * Level-2：每类 500 / 800 / 1200 条（在 300 条金标准基础上，从全量语料 R_2 列补充）
2. 零泄漏守卫：
   - 所有银标补充池严格排除与 task1_level1 / task2_level2 的 train.parquet 和 validation.parquet
     文本相同的段落及所属文档；单文档最多抽 4 段（MAX_PARA_PER_DOC=4），防止长文档主导。

用法（仓库根目录）：
  python 06_generation_scale_benchmark/scripts/prepare/build_g3_datasets.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_common as C  # noqa: E402

G3_DATA_ROOT = C.BENCH_ROOT / "data" / "g3"
G3_CONFIG_PATH = C.BENCH_ROOT / "configs" / "g3_matrix.json"
MAX_PARA_PER_DOC = 4
MIN_CHAR_LEN = 20
MAX_CHAR_LEN = 768


def norm_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def find_full_corpus_parquet() -> Path | None:
    """自动查找 CN_DOCS_FINAL_complete.parquet（支持 HF 缓存、AutoDL 缓存与 assets 目录）。"""
    candidates: list[Path] = []
    try:
        snap = C.preprocess.find_snapshot()
        candidates.append(snap / "data" / "CN_DOCS_FINAL_complete.parquet")
    except Exception:
        pass
    candidates.extend(
        [
            Path("/root/autodl-tmp/hf_models/hub/datasets--Baron-Sun--CAPC-CG_V1.0/snapshots/local/data/CN_DOCS_FINAL_complete.parquet"),
            C.PROJECT_ROOT / "assets" / "shared" / "datasets" / "raw" / "CAPC-CG_V1.0" / "data" / "CN_DOCS_FINAL_complete.parquet",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_silver_from_full_parquet(
    parquet_path: Path,
    label_col: str,
    valid_labels: set[str],
    forbidden_texts: set[str],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    print(f"[读取全量语料] {parquet_path} (列={label_col}) ...")
    cols = ["DocumentID", "Paragraph_content", "IssueDate", "Title", label_col]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["text"] = df["Paragraph_content"].fillna("").astype(str).str.strip()
    df["label"] = df[label_col].fillna("").astype(str).str.strip()
    df = df[df["label"].isin(valid_labels)]
    tl = df["text"].str.len()
    df = df[(tl >= MIN_CHAR_LEN) & (tl <= MAX_CHAR_LEN)]
    df["_norm"] = df["text"].map(norm_text)

    # 整篇文档级排除：凡命中金标准 train/val 的文档一律剔除
    hit_docs = set(df.loc[df["_norm"].isin(forbidden_texts), "DocumentID"].astype(str))
    df = df[~df["DocumentID"].astype(str).isin(hit_docs)].reset_index(drop=True)

    rng = random.Random(seed)
    by_doc: dict[str, list[int]] = defaultdict(list)
    for idx, doc_id in enumerate(df["DocumentID"].astype(str)):
        by_doc[doc_id].append(idx)

    kept_indices: list[int] = []
    for doc_id, idxs in by_doc.items():
        if len(idxs) > MAX_PARA_PER_DOC:
            idxs = rng.sample(idxs, MAX_PARA_PER_DOC)
        kept_indices.extend(idxs)

    sub = df.iloc[kept_indices]
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set(forbidden_texts)
    for row in sub.itertuples(index=False):
        nt = row._norm
        if not nt or nt in seen:
            continue
        seen.add(nt)
        pool[row.label].append(
            {
                "text": row.text,
                "user_raw": f"{C.preprocess.USER_PREFIX}\n{row.text}",
                "label": row.label,
                "source": f"corpus:{label_col}",
            }
        )
    for lb in valid_labels:
        rng.shuffle(pool[lb])
    return pool


def load_l1_from_stage05_jsonl(
    forbidden_texts: set[str], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """当未挂载 500MB 全量 parquet 时，直接从 05 期已入库的 T1-T5 / D1-D6 训练集抽取 L1 银标样本。"""
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set(forbidden_texts)
    patterns = [
        C.PROJECT_ROOT / "05_incremental_learning_time" / "data" / "temporal" / "*_train.jsonl",
        C.PROJECT_ROOT / "05_incremental_learning_domain" / "data" / "domain_incremental" / "*_train.jsonl",
    ]
    for pattern in patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    text = str(rec.get("text", "")).strip()
                    label = str(rec.get("label", "")).strip()
                    nt = norm_text(text)
                    if not text or label not in {"W", "R", "N"} or nt in seen:
                        continue
                    seen.add(nt)
                    pool[label].append(
                        {
                            "text": text,
                            "user_raw": f"{C.preprocess.USER_PREFIX}\n{text}",
                            "label": label,
                            "source": f"stage05:{path.name}",
                        }
                    )
    rng = random.Random(seed)
    for lb in ("W", "R", "N"):
        rng.shuffle(pool[lb])
    return pool


def build_task_tiers(
    task: str,
    spec: dict[str, Any],
    gold_train: list[dict[str, Any]],
    silver_pool: dict[str, list[dict[str, Any]]],
    val_norms: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    labels = spec["labels"]
    gold_per_class = int(spec["gold_per_class"])
    all_tiers = sorted(set(spec.get("gold_tiers", []) + spec.get("silver_tiers", [])))

    rng = random.Random(seed)
    gold_by_label: dict[str, list[dict[str, Any]]] = {lb: [] for lb in labels}
    for rec in gold_train:
        gold_by_label[rec["label"]].append({**rec, "source": "gold"})
    for lb in labels:
        rng.shuffle(gold_by_label[lb])

    ordered_by_label = {
        lb: gold_by_label[lb] + silver_pool.get(lb, []) for lb in labels
    }

    out_dir = G3_DATA_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    for per_class in all_tiers:
        avail_min = min(len(ordered_by_label[lb]) for lb in labels)
        if avail_min < per_class:
            print(
                f"[SKIP] {task} per_class={per_class}: 可用最小类仅 {avail_min} 条 "
                f"(如需构建该档位，请挂载 CN_DOCS_FINAL_complete.parquet)"
            )
            continue

        selected: list[dict[str, Any]] = []
        gold_n = 0
        silver_n = 0
        for lb in labels:
            chunk = ordered_by_label[lb][:per_class]
            for item in chunk:
                if norm_text(item["text"]) in val_norms:
                    raise AssertionError(f"检测到与验证集重叠的样本: {task} {lb}")
            selected.extend(chunk)
            gold_n += sum(1 for x in chunk if x["source"] == "gold")
            silver_n += sum(1 for x in chunk if x["source"] != "gold")

        rng.shuffle(selected)
        regime = "pure_gold" if per_class <= gold_per_class else "gold_plus_silver"
        out_path = out_dir / f"train_per_class_{per_class}.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            for item in selected:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()[:12]
        row = {
            "task": task,
            "per_class": per_class,
            "total": len(selected),
            "regime": regime,
            "gold_count": gold_n,
            "silver_count": silver_n,
            "file": str(out_path),
            "sha256": sha,
        }
        manifest_rows.append(row)
        print(
            f"[OK] {task:12s} | {regime:16s} | per_class={per_class:4d} | "
            f"total={len(selected):4d} (gold={gold_n}, silver={silver_n}) -> {out_path.name}"
        )
    return manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 G3 阶段样本量 Scaling 嵌套训练集")
    parser.add_argument("--config", default=str(G3_CONFIG_PATH))
    parser.add_argument("--corpus-parquet", default=None, help="可选：显式指定 CN_DOCS_FINAL_complete.parquet 路径")
    args = parser.parse_args()

    cfg = C.read_json(args.config)
    seed = int(cfg.get("seed", 42))
    corpus_parquet = (
        Path(args.corpus_parquet) if args.corpus_parquet else find_full_corpus_parquet()
    )
    if corpus_parquet:
        print(f"[数据源] 检测到全量语料: {corpus_parquet}")
    else:
        print("[数据源] 未检测到全量 parquet，L1 扩增使用 05 期 jsonl，L2 仅构建纯金标准梯度 (75/150/225/300)")

    # 汇总所有金标准 train/val 文本，严禁银标池泄漏
    all_forbidden: set[str] = set()
    task_gold: dict[str, tuple[list[dict[str, Any]], set[str]]] = {}
    for task in cfg["tasks"]:
        tr = C.preprocess.load_split(task, "train")
        va = C.preprocess.load_split(task, "validation")
        va_norms = {norm_text(r["text"]) for r in va}
        all_forbidden.update(norm_text(r["text"]) for r in tr + va)
        task_gold[task] = (tr, va_norms)

    manifest_splits: list[dict[str, Any]] = []
    for task, spec in cfg["tasks"].items():
        labels_set = set(spec["labels"])
        gold_train, va_norms = task_gold[task]
        if task == "task1_level1":
            if corpus_parquet:
                silver_pool = load_silver_from_full_parquet(
                    corpus_parquet, "R_1", labels_set, all_forbidden, seed
                )
            else:
                silver_pool = load_l1_from_stage05_jsonl(all_forbidden, seed)
        else:
            if corpus_parquet:
                silver_pool = load_silver_from_full_parquet(
                    corpus_parquet, "R_2", labels_set, all_forbidden, seed
                )
            else:
                silver_pool = {lb: [] for lb in labels_set}

        manifest_splits.extend(
            build_task_tiers(task, spec, gold_train, silver_pool, va_norms, seed)
        )

    G3_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = G3_DATA_ROOT / "g3_manifest.json"
    C.write_json(
        manifest_path,
        {
            "version": cfg.get("version", "2.0"),
            "seed": seed,
            "corpus_parquet": str(corpus_parquet) if corpus_parquet else None,
            "splits": manifest_splits,
        },
    )
    print(f"\n[完成] G3 数据清单已写入: {manifest_path}（共 {len(manifest_splits)} 个划分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())