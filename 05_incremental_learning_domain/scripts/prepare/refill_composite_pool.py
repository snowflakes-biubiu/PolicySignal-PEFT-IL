"""按 gaps.json 从 27 万段母池补抽候选，供逐段复核后凑齐 5,400。

A1 确认的「按需补抽」：不预先盲目扩容（那样会多烧复核 token），只在复核后发现
某 (域×年代×WRN) 格子不足时才补，且只复核新增段。

用法：
    python 05_incremental_learning_domain/scripts/prepare/refill_composite_pool.py
    python ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))
from build_composite_pool import (  # noqa: E402
    CAP, CORPUS, DATA, ERAS, LABELS, OUT_DIR, PRIORITY, CANDIDATES, USED,
    load_jsonl, norm_text, pid_of, risk_flags, text_key_of,
)
from domain_taxonomy_v14 import TRAIN_DOMAINS  # noqa: E402
from title_utils import title_key  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gaps_path = args.out_dir / "gaps.json"
    if not gaps_path.is_file():
        print("没有 gaps.json，无需补抽")
        return 2
    gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
    if not gaps:
        print("gaps 为空，无需补抽")
        return 0
    pool_path = args.out_dir / "pool.jsonl"
    pool = load_jsonl(pool_path)
    seen_tk = {r.get("text_key") or text_key_of(r["text"]) for r in pool}
    seen_pid = {r["pid"] for r in pool}
    max_rank = defaultdict(int)
    for r in pool:
        k = (r["domain"], r["era"], r["wrn"])
        max_rank[k] = max(max_rank[k], r.get("rank", 0))

    # 缺口按「复核拒绝率上界」放大；高拒格子（经济域/T2/N 类）额外冗余
    plan = {}
    for g in gaps:
        p = 0.331
        if g["domain"] == "经济与财税金融" or g["era"] == "T2_1979_1992" or g["wrn"] == "N":
            p = 0.45
        need = g["missing"]
        plan[(g["domain"], g["era"], g["wrn"])] = {
            "take": min(2000, int(need / (1 - p) + 0.999) + 3),
            "p": p, "missing": need,
        }
    print(f"待补格子 {len(plan)} 个，计划补抽 {sum(v['take'] for v in plan.values()):,} 段", flush=True)
    if args.dry_run:
        for k, v in sorted(plan.items()):
            print(f"  {k}: 缺 {v['missing']} → 补 {v['take']}（p={v['p']}）")
        return 0

    lab = {}
    for r in load_jsonl(LABELS):
        if "error" not in r:
            lab[r.get("title_key") or title_key(r["Title"])] = r
    hits = {}
    for r in load_jsonl(CANDIDATES) + load_jsonl(PRIORITY):
        hits.setdefault(title_key(r["Title"]), r.get("hit_domains") or [])
    used = {str(x) for x in json.loads(USED.read_text(encoding="utf-8"))} if USED.is_file() else set()

    df = pd.read_parquet(CORPUS, columns=["DocumentID", "Title", "IssueDate", "R_1", "Paragraph_content"])
    df["text"] = df["Paragraph_content"].fillna("").astype(str)
    df["year"] = pd.to_numeric(df["IssueDate"].astype(str).str[:4], errors="coerce")
    tl = df["text"].str.len()
    df = df[df["R_1"].isin(["N", "R", "W"]) & df["year"].between(1949, 2023)
            & (tl >= 20) & (tl <= 768)]
    df = df[~df["DocumentID"].astype(str).isin(used)]
    print(f"母池合格段落 {len(df):,}", flush=True)

    doc_meta = {}
    for doc_id, title, year in zip(df["DocumentID"].astype(str), df["Title"], df["year"]):
        if doc_id not in doc_meta:
            k = title_key(str(title))
            r = lab.get(k)
            doc_meta[doc_id] = {"title": str(title), "year": int(year),
                                "domain": (r or {}).get("domain"),
                                "model": (r or {}).get("model"),
                                "conf": float((r or {}).get("confidence") or 0),
                                "hit": hits.get(k)}

    buckets = defaultdict(list)
    for doc_id, title, year, wrn, text in zip(
            df["DocumentID"].astype(str), df["Title"], df["year"], df["R_1"], df["text"]):
        meta = doc_meta[doc_id]
        dom = meta["domain"]
        if dom not in TRAIN_DOMAINS.values():
            continue
        y = int(year)
        era = next((e[0] for e in ERAS if e[1] <= y <= e[2]), None)
        key = (dom, era, wrn)
        if key not in plan or not era:
            continue
        tkey = text_key_of(text)
        if tkey in seen_tk:
            continue
        seen_tk.add(tkey)
        pid = pid_of(doc_id, text)
        if pid in seen_pid:
            continue
        seen_pid.add(pid)
        flags = risk_flags(text, meta["title"])
        hit = meta["hit"] or []
        tier = "A" if meta["conf"] >= 0.90 and not flags else (
            "B" if meta["conf"] >= 0.85 or dom in hit else "C")
        buckets[key].append({
            "pid": pid, "text_key": tkey, "doc_id": doc_id, "Title": meta["title"],
            "year": y, "era": era, "domain": dom, "wrn": wrn,
            "text": text, "text_len": len(text), "snippet": text[:220],
            "label_model": meta["model"], "label_conf": meta["conf"],
            "hit_domains": hit, "hit_agree": bool(hit) and dom in hit,
            "risk_flags": flags, "tier": tier,
            "cell": f"{dom}|{era}|{wrn}",
            "doc_split": "test" if int(__import__("hashlib").sha256(
                doc_id.encode()).hexdigest()[:8], 16) % 3 == 0 else "train",
        })

    rng = random.Random(args.seed)
    added, log = [], []
    for key, spec in sorted(plan.items()):
        cand = buckets.get(key, [])
        take = min(spec["take"], len(cand))
        picked = rng.sample(cand, take)
        start_rank = max_rank[key] + 1
        for i, rec in enumerate(sorted(picked, key=lambda r: (len(r["risk_flags"]),
                                                               {"A": 0, "B": 1, "C": 2}[r["tier"]],
                                                               -r["label_conf"]))):
            out = dict(rec)
            out["rank"] = start_rank + i
            out["cell_need"] = spec["missing"] + spec["take"]
            out["oversample_p"] = spec["p"]
            out["refill"] = True
            added.append(out)
        log.append({"domain": key[0], "era": key[1], "wrn": key[2],
                    "missing": spec["missing"], "available": len(cand), "added": take})
        print(f"  {key}: 缺 {spec['missing']} → 补 {take}（母池可用 {len(cand):,}）", flush=True)

    with pool_path.open("a", encoding="utf-8") as f:
        for r in added:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out_dir / "refill_log.json").write_text(
        json.dumps({"round": len(log), "added": len(added), "cells": log},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pool_size"] = len(pool) + len(added)
    manifest["unique_text_keys"] = len(seen_tk)
    manifest["refill_rounds"] = manifest.get("refill_rounds", 0) + 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已追加 {len(added):,} 段 → {pool_path}（池共 {manifest['pool_size']:,} 段）", flush=True)
    print("下一步：verify_composite.py --run（只复核新增段）→ --compose", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
