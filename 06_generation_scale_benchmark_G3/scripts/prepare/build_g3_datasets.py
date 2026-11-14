# -*- coding: utf-8 -*-
"""G3 数据构建：纯金标准嵌套曲线 + R1/R2 银标扩增曲线。

边界（与设计一致，由 `validate_config()` 强制校验）：
- Level-1：每类 ≤ 200 条取自金标准 train（600 条）；> 200 条从全量语料 `R_1` 列补银标。
- Level-2：每类 ≤ 300 条取自金标准 train（1200 条）；> 300 条从全量语料 `R_2` 列补银标。

零泄漏：
- 银标池排除所有金标准 train/validation 文本，并整篇文档排除命中金标准的文档；
- 单文档最多 `max_para_per_doc` 段，避免长文档主导；
- 构建后对每个档位断言与验证集文本零重叠。

用法（仓库根目录）：
  python 06_generation_scale_benchmark_G3/scripts/prepare/build_g3_datasets.py
  python 06_generation_scale_benchmark_G3/scripts/prepare/build_g3_datasets.py --corpus-parquet <path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import g3_common as G  # noqa: E402

DATA_ROOT = G.G3_DATA_ROOT
CONFIG_PATH = G.G3_CONFIG


def norm_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def validate_config(cfg: dict[str, Any]) -> None:
    """把用户声明的「金标准边界」固定下来，配置写错直接报错。"""
    for task, spec in cfg["tasks"].items():
        labels = spec["labels"]
        gold = int(spec["gold_per_class"])
        gold_tiers = [int(x) for x in spec.get("gold_tiers", [])]
        silver_tiers = [int(x) for x in spec.get("silver_tiers", [])]
        if not gold_tiers:
            raise ValueError(f"{task}: gold_tiers 不能为空")
        if max(gold_tiers) != gold:
            raise ValueError(
                f"{task}: gold_tiers 最大值 {max(gold_tiers)} 必须等于 gold_per_class {gold}"
            )
        if any(x > gold for x in gold_tiers):
            raise ValueError(f"{task}: gold_tiers 不得越过金标准边界 {gold}")
        if silver_tiers and min(silver_tiers) <= gold:
            raise ValueError(
                f"{task}: silver_tiers 最小值 {min(silver_tiers)} 必须大于金标准边界 {gold}"
            )
        if silver_tiers and len(set(gold_tiers) & set(silver_tiers)):
            raise ValueError(f"{task}: gold_tiers 与 silver_tiers 有重叠")
        if spec.get("silver_column") not in {"R_1", "R_2"}:
            raise ValueError(f"{task}: silver_column 必须是 R_1 或 R_2")
        if spec["val_total"] <= 0 or not labels:
            raise ValueError(f"{task}: labels/val_total 配置非法")


def find_corpus(override: str | None, cfg: dict[str, Any]) -> Path | None:
    if override:
        path = Path(override)
        return path if path.is_file() else None
    filename = cfg.get("corpus", {}).get("file", "CN_DOCS_FINAL_complete.parquet")
    candidates: list[Path] = []
    try:
        candidates.append(G.preprocess.find_snapshot() / "data" / filename)
    except Exception:
        pass
    candidates.extend(
        [
            Path("/root/autodl-tmp/hf_models/hub/datasets--Baron-Sun--CAPC-CG_V1.0/snapshots/local/data") / filename,
            G.PROJECT_ROOT / "assets" / "shared" / "datasets" / "raw" / "CAPC-CG_V1.0" / "data" / filename,
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def load_gold(
    cfg: dict[str, Any],
) -> tuple[dict[str, list[dict]], dict[str, set[str]], dict[str, dict[str, int]]]:
    gold: dict[str, list[dict]] = {}
    val_norms: dict[str, set[str]] = {}
    overlap_by_task: dict[str, dict[str, int]] = {}
    for task in cfg["tasks"]:
        train = G.preprocess.load_split(task, "train")
        validation = G.preprocess.load_split(task, "validation")
        gold[task] = [
            {
                "text": rec["text"],
                "user_raw": rec["user_raw"],
                "label": rec["label"],
                "source": "gold",
            }
            for rec in train
        ]
        val_norms[task] = {norm_text(rec["text"]) for rec in validation}
        counts = Counter(rec["label"] for rec in gold[task])
        overlap = Counter(
            rec["label"] for rec in gold[task] if norm_text(rec["text"]) in val_norms[task]
        )
        overlap_by_task[task] = {
            lb: int(overlap.get(lb, 0)) for lb in cfg["tasks"][task]["labels"]
        }
        print(f"[金标准] {task}: n={len(train)} 每类={dict(sorted(counts.items()))}")
        print(
            f"[金标准] {task}: 与验证集文本重叠 {sum(overlap.values())} 条 "
            f"{overlap_by_task[task]}（数据集固有，G1/G2 同口径，不剔除、只记录）"
        )
    return gold, val_norms, overlap_by_task


def load_silver_pool(
    corpus: Path,
    label_col: str,
    labels: set[str],
    forbidden_texts: set[str],
    cfg: dict[str, Any],
    seed: int,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    import pandas as pd

    corpus_cfg = cfg.get("corpus", {})
    text_col = corpus_cfg.get("text_column", "Paragraph_content")
    doc_col = corpus_cfg.get("doc_column", "DocumentID")
    max_para = int(corpus_cfg.get("max_para_per_doc", 4))
    min_len = int(corpus_cfg.get("min_char_len", 20))
    max_len = int(corpus_cfg.get("max_char_len", 768))

    print(f"[银标] 读取 {corpus.name}，列={label_col} ...")
    df = pd.read_parquet(corpus, columns=[doc_col, text_col, label_col])
    missing = [c for c in (doc_col, text_col, label_col) if c not in df.columns]
    if missing:
        raise KeyError(f"全量语料缺少列 {missing}，实际列={list(df.columns)}")

    df = df[df[label_col].astype(str).isin(labels)].copy()
    df["text"] = df[text_col].fillna("").astype(str).str.strip()
    df["label"] = df[label_col].astype(str).str.strip()
    lengths = df["text"].str.len()
    df = df[(lengths >= min_len) & (lengths <= max_len)].copy()
    print(f"[银标] 标签/长度过滤后 {len(df)} 行")

    if forbidden_texts:
        df["_norm"] = df["text"].map(norm_text)
        hit_docs = set(
            df.loc[df["_norm"].isin(forbidden_texts), doc_col].astype(str)
        )
        excluded_docs = int(df[doc_col].astype(str).isin(hit_docs).sum())
        df = df[~df[doc_col].astype(str).isin(hit_docs)].copy()
        print(f"[银标] 文档级排除命中金标准文档 {len(hit_docs)} 个 / {excluded_docs} 行")
    else:
        df["_norm"] = df["text"].map(norm_text)

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
    pool: dict[str, list[dict]] = defaultdict(list)
    seen = set(forbidden_texts)
    for text, label, nt in zip(
        sub["text"].tolist(), sub["label"].tolist(), sub["_norm"].tolist()
    ):
        if not nt or nt in seen:
            continue
        seen.add(nt)
        pool[label].append(
            {
                "text": text,
                "user_raw": f"{G.preprocess.USER_PREFIX}\n{text}",
                "label": label,
                "source": f"corpus:{label_col}",
            }
        )
    for label in labels:
        rng.shuffle(pool[label])

    stats = {
        "corpus": str(corpus),
        "column": label_col,
        "documents": len(by_doc),
        "pool_per_class": {lb: len(pool[lb]) for lb in sorted(labels)},
    }
    print(f"[银标] 可用池: {stats['pool_per_class']}")
    return dict(pool), stats


def load_l1_from_stage05(
    forbidden_texts: set[str], cfg: dict[str, Any]
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """无全量 parquet 时的降级路径：从 05 期已入库的 T/D 训练集取得 L1 银标。"""
    pool: dict[str, list[dict]] = defaultdict(list)
    seen = set(forbidden_texts)
    patterns = [
        G.PROJECT_ROOT / "05_incremental_learning_time" / "data" / "temporal" / "*_train.jsonl",
        G.PROJECT_ROOT / "05_incremental_learning_domain" / "data" / "domain_incremental" / "*_train.jsonl",
    ]
    used: list[str] = []
    for pattern in patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
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
                            "user_raw": f"{G.preprocess.USER_PREFIX}\n{text}",
                            "label": label,
                            "source": f"stage05:{path.name}",
                        }
                    )
            used.append(path.name)
    rng = random.Random(int(cfg.get("seed", 42)))
    for label in ("W", "R", "N"):
        rng.shuffle(pool[label])
    stats = {
        "corpus": None,
        "column": "stage05_jsonl",
        "pool_per_class": {lb: len(pool[lb]) for lb in ("W", "R", "N")},
        "files": used,
    }
    print(f"[银标] 降级路径（05 jsonl）可用池: {stats['pool_per_class']}")
    return dict(pool), stats


def build_tiers(
    task: str,
    spec: dict[str, Any],
    gold_records: list[dict],
    silver_pool: dict[str, list[dict]],
    val_norms: set[str],
    cfg: dict[str, Any],
    gold_overlap: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    labels = spec["labels"]
    gold_boundary = int(spec["gold_per_class"])
    tiers = sorted(set(int(x) for x in spec.get("gold_tiers", []) + spec.get("silver_tiers", [])))
    seed = int(cfg.get("seed", 42))

    gold_by_label: dict[str, list[dict]] = {lb: [] for lb in labels}
    for rec in gold_records:
        gold_by_label[rec["label"]].append(rec)
    for lb in labels:
        if len(gold_by_label[lb]) < gold_boundary:
            raise ValueError(
                f"{task}: 金标准 {lb} 类只有 {len(gold_by_label[lb])} 条，"
                f"少于边界 {gold_boundary}，不能用银标填充纯金标准段"
            )

    rng = random.Random(seed)
    for lb in labels:
        rng.shuffle(gold_by_label[lb])

    # 同一有序列表切片，保证 50 ⊂ 100 ⊂ ... ⊂ 200 ⊂ 400 ⊂ ... 的嵌套关系
    ordered = {lb: gold_by_label[lb] + silver_pool.get(lb, []) for lb in labels}
    out_dir = DATA_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for per_class in tiers:
        available = min(len(ordered[lb]) for lb in labels)
        if available < per_class:
            print(
                f"[SKIP] {task} per_class={per_class}: 最小类只有 {available} 条"
                f"（银标不足，需挂载全量语料）"
            )
            continue

        selected: list[dict] = []
        gold_n = silver_n = tier_overlap = 0
        for lb in labels:
            chunk = ordered[lb][:per_class]
            for item in chunk:
                # 金标准与验证集的少量文本重叠是数据集固有（G1/G2 同口径，见 README）；
                # 银标必须严格零泄漏。
                normalized = norm_text(item["text"])
                if item["source"] != "gold" and normalized in val_norms:
                    raise AssertionError(f"{task}/{lb}: 银标样本与验证集重叠")
                if item["source"] == "gold" and normalized in val_norms:
                    tier_overlap += 1
            selected.extend(chunk)
            gold_n += sum(1 for item in chunk if item["source"] == "gold")
            silver_n += sum(1 for item in chunk if item["source"] != "gold")

        regime = "pure_gold" if per_class <= gold_boundary else "gold_plus_silver"
        if regime == "pure_gold" and silver_n != 0:
            raise AssertionError(
                f"{task} n={per_class}: 纯金标准档位混入 {silver_n} 条银标"
            )
        if regime == "gold_plus_silver" and gold_n != gold_boundary * len(labels):
            raise AssertionError(
                f"{task} n={per_class}: 银标档位未包含完整金标准 "
                f"({gold_n} != {gold_boundary * len(labels)})"
            )

        rng.shuffle(selected)
        out_path = out_dir / f"train_per_class_{per_class}.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in selected:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
        manifest.append(
            {
                "task": task,
                "per_class": per_class,
                "total": len(selected),
                "regime": regime,
                "gold_count": gold_n,
                "silver_count": silver_n,
                "silver_ratio": round(silver_n / len(selected), 4),
                "gold_val_overlap": tier_overlap,
                "file": out_path.relative_to(G.PROJECT_ROOT).as_posix(),
                "sha256": digest,
            }
        )
        print(
            f"[OK] {task:14s} {regime:16s} n/class={per_class:4d} "
            f"total={len(selected):4d} gold={gold_n:4d} silver={silver_n:4d} "
            f"-> {out_path.name}"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 G3 嵌套训练集")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--corpus-parquet", default=None)
    args = parser.parse_args()

    cfg = G.read_json(args.config)
    validate_config(cfg)
    seed = int(cfg.get("seed", 42))
    gold, val_norms, gold_overlap = load_gold(cfg)

    corpus = find_corpus(args.corpus_parquet, cfg)
    if corpus:
        print(f"[数据源] 全量语料: {corpus}")
    else:
        print("[数据源] 未找到全量 parquet：L1 退回 05 期 jsonl，L2 只能构建纯金标准段")

    forbidden = set()
    for task in cfg["tasks"]:
        forbidden.update(norm_text(rec["text"]) for rec in gold[task])
        forbidden.update(val_norms[task])

    splits: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for task, spec in cfg["tasks"].items():
        labels = set(spec["labels"])
        if corpus:
            pool, stats = load_silver_pool(
                corpus, spec["silver_column"], labels, forbidden, cfg, seed
            )
        elif task == "task1_level1":
            pool, stats = load_l1_from_stage05(forbidden, cfg)
        else:
            pool, stats = {}, {"corpus": None, "column": spec["silver_column"], "pool_per_class": {}}
        sources[task] = stats
        splits.extend(
            build_tiers(
                task, spec, gold[task], pool, val_norms[task], cfg, gold_overlap[task]
            )
        )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_ROOT / "g3_manifest.json"
    G.write_json(
        manifest_path,
        {
            "version": cfg.get("version", "2.0"),
            "stage": "G3",
            "seed": seed,
            "boundary": {
                task: int(spec["gold_per_class"]) for task, spec in cfg["tasks"].items()
            },
            "sources": sources,
            "gold_val_overlap": gold_overlap,
            "splits": splits,
        },
    )
    print(f"\n[完成] {len(splits)} 个划分 -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
