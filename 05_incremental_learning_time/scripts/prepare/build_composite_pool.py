"""构建「复合实验」候选池：>5,400 段，供逐段全查后筛出标准训练集。

设计要点（与用户确认的口径）：
  * 最终目标：6 训练域 × 900 段 = 5,400（每域 train W200/R200/N200 + test W100/R100/N100）；
  * 逐段复核会筛掉不合格段落，因此候选池按「估计拒绝率」放大采样；
  * T1 不强制每类 5 段，但尽量多取（early bonus），T1~T5 都要出现；
  * `candidates_priority.jsonl` / `hit_domains` **不是真理**，只作为排序用的弱证据，
    不参与 keep/reject 判定；判定标准见 composite/rubric.md；
  * 排除 used_documents.json 中已被 temporal 实验占用的文档；
  * 段落文本全局去重，剔除模板段风险标记（只标记、不预先删除，保证拒绝率估计无偏）。

用法：
    python 05_incremental_learning_time/scripts/prepare/build_composite_pool.py
    python ... --reject-rate 0.35 --out-dir ... --seed 42
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

import pandas as pd

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))
from domain_taxonomy_v14 import TRAIN_DOMAINS  # noqa: E402
from title_utils import title_key  # noqa: E402

PROJECT_ROOT = PREPARE_DIR.parents[2]
DATA = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "data" / "domain_v14"
LABELS = DATA / "title_domain_labels_v14.jsonl"
PRIORITY = DATA / "candidates_priority.jsonl"
CANDIDATES = DATA / "candidates.jsonl"
USED = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "data" / "used_documents.json"
CORPUS = Path(r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
              r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\data\CN_DOCS_FINAL_complete.parquet")
OUT_DIR = DATA / "composite"

ERAS = [("T1_1949_1978", 1949, 1978), ("T2_1979_1992", 1979, 1992),
        ("T3_1993_2002", 1993, 2002), ("T4_2003_2012", 2003, 2012),
        ("T5_2013_2023", 2013, 2023)]
CAP = {"N": 4, "R": 8, "W": 4}
NEED_PER_CLASS = 300          # train 200 + test 100
EARLY_T1 = 12                 # T1 每(域×类)尽量取到
EARLY_T2 = 30
SNIPPET_CHARS = 220

BOILERPLATE = re.compile(
    r"^(抄送|主题词|附件|印发|签发|校对|共印|联系人|责任编辑|打字|校对|份数)|"
    r"(现通知如下|请遵照执行|特此通知|特此批复|本办法自.*施行|本规定自.*施行|"
    r"以上报告如无不妥|附件：|联系人及电话|抄送：)")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).casefold()


def text_key_of(text: str) -> str:
    """段落内容级严格去重键：NFKC + 去所有标点/空白 + 小写。

    A3 要求「文档不互斥、段落内容严格互斥」——不同文档里复制粘贴的同一段话
    也必须只保留一份，故不能用 doc_id 参与去重。
    """
    import unicodedata
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]


def pid_of(doc_id: str, text: str) -> str:
    return hashlib.sha256(f"{doc_id}|{norm_text(text)}".encode("utf-8")).hexdigest()[:16]


def risk_flags(text: str, title: str) -> list[str]:
    flags = []
    n = len(text)
    if n < 40:
        flags.append("short_lt40")
    digits = sum(ch.isdigit() for ch in text)
    if digits / max(n, 1) > 0.30:
        flags.append("digit_heavy")
    if BOILERPLATE.search(text[:80]) or n < 25:
        flags.append("boilerplate")
    if re.search(r"[…\.]{3,}|（?\.\.\.", title):
        flags.append("truncated_title")
    if re.search(r"第\s*\d+\s*号(建议|提案|议案)?的?(答复|函)?$", title):
        flags.append("proposal_number_only")
    return flags


def allocate_eras(supply: dict[str, int], total: int) -> dict[str, int]:
    """按供给占比分配年代名额；T1/T2 先给 early bonus（不强制每类 5 段）。"""
    quota = {e[0]: 0 for e in ERAS}
    remain = total
    for era, cap in (("T1_1949_1978", EARLY_T1), ("T2_1979_1992", EARLY_T2)):
        take = min(cap, supply.get(era, 0), remain)
        quota[era] = take
        remain -= take
    late = [e[0] for e in ERAS[2:]]
    wsum = sum(supply.get(e, 0) for e in late)
    if wsum <= 0:
        return quota
    ideals = {e: remain * supply.get(e, 0) / wsum for e in late}
    for e in late:
        quota[e] = min(int(ideals[e]), supply.get(e, 0))
    # 最大余数法补齐
    diff = remain - sum(quota[e] for e in late)
    order = sorted(late, key=lambda e: ideals[e] - int(ideals[e]), reverse=True)
    for e in order:
        if diff <= 0:
            break
        if quota[e] < supply.get(e, 0):
            quota[e] += 1
            diff -= 1
    # 若仍缺（供给不足），回补给有存量的年代
    for e in late + [x[0] for x in ERAS[:2]]:
        if diff <= 0:
            break
        add = min(diff, supply.get(e, 0) - quota[e])
        if add > 0:
            quota[e] += add
            diff -= add
    return quota


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--reject-rate", type=float, default=0.30,
                    help="估计的逐段复核拒绝率，用于放大候选池")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    factor = 1.0 / max(1e-6, 1.0 - args.reject_rate)

    lab = {}
    for r in load_jsonl(LABELS):
        if "error" not in r:
            lab[r.get("title_key") or title_key(r["Title"])] = r
    hits = {}
    for r in load_jsonl(CANDIDATES) + load_jsonl(PRIORITY):
        hits.setdefault(title_key(r["Title"]), r.get("hit_domains") or [])
    used = {str(x) for x in json.loads(USED.read_text(encoding="utf-8"))} if USED.is_file() else set()

    print(f"标签 {len(lab):,}  弱证据 hit_domains {len(hits):,}  已占用文档 {len(used):,}", flush=True)
    df = pd.read_parquet(CORPUS, columns=["DocumentID", "Title", "IssueDate", "R_1", "Paragraph_content"])
    df["text"] = df["Paragraph_content"].fillna("").astype(str)
    df["year"] = pd.to_numeric(df["IssueDate"].astype(str).str[:4], errors="coerce")
    tl = df["text"].str.len()
    df = df[df["R_1"].isin(["N", "R", "W"]) & df["year"].between(1949, 2023)
            & (tl >= 20) & (tl <= 768)]
    df = df[~df["DocumentID"].astype(str).isin(used)]
    print(f"合格段落（已排除 temporal 占用）{len(df):,}", flush=True)

    # 文档级：标签/年代/证据；段落级：去重 + 风险标记
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
    seen_pid = set()
    seen_text_key = set()
    cells = defaultdict(list)
    for doc_id, title, year, wrn, text in zip(
            df["DocumentID"].astype(str), df["Title"], df["year"], df["R_1"], df["text"]):
        meta = doc_meta[doc_id]
        dom = meta["domain"]
        if dom not in TRAIN_DOMAINS.values():
            continue
        pid = pid_of(doc_id, text)
        if pid in seen_pid:
            continue
        tkey = text_key_of(text)
        if tkey in seen_text_key:
            continue
        seen_pid.add(pid)
        seen_text_key.add(tkey)
        y = int(year)
        era = next((e[0] for e in ERAS if e[1] <= y <= e[2]), None)
        if era is None:
            continue
        flags = risk_flags(text, meta["title"])
        hit = meta["hit"] or []
        strong = meta["conf"] >= 0.90 and not flags
        tier = "A" if strong else ("B" if meta["conf"] >= 0.85 or dom in hit else "C")
        rec = {
            "pid": pid, "text_key": tkey, "doc_id": doc_id, "Title": meta["title"],
            "year": y, "era": era,
            "domain": dom, "wrn": wrn, "text": text, "text_len": len(text),
            "snippet": text[:SNIPPET_CHARS],
            "label_model": meta["model"], "label_conf": meta["conf"],
            "hit_domains": hit, "hit_agree": bool(hit) and dom in hit,
            "risk_flags": flags, "tier": tier,
            "cell": f"{dom}|{era}|{wrn}",
            "doc_split": "test" if int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 3 == 0 else "train",
        }
        cells[rec["cell"]].append(rec)
    print(f"入池候选段落（去重后）{sum(len(v) for v in cells.values()):,}", flush=True)

    pool: list[dict] = []
    manifest_cells = []
    for idx, dom in TRAIN_DOMAINS.items():
        for wrn in "NRW":
            supply = {e[0]: len(cells[f"{dom}|{e[0]}|{wrn}"]) for e in ERAS}
            quota = allocate_eras(supply, NEED_PER_CLASS)
            for era, *_ in ERAS:
                cell = f"{dom}|{era}|{wrn}"
                cand = cells[cell]
                need = quota[era]
                if need <= 0:
                    continue
                p_eff = min(0.55, args.reject_rate + (0.10 if era.startswith(("T1", "T2")) or wrn == "R" else 0.0))
                want = min(len(cand), int(need / max(1e-6, 1 - p_eff) + 0.999) + 3)
                # 分层随机（保持自然风险分布，保证 pilot 拒绝率无偏）
                picked = rng.sample(cand, want)
                for rank, rec in enumerate(
                        sorted(picked, key=lambda r: (len(r["risk_flags"]),
                                                      {"A": 0, "B": 1, "C": 2}[r["tier"]],
                                                      -r["label_conf"]))):
                    out = dict(rec)
                    out["rank"] = rank
                    out["cell_need"] = need
                    out["oversample_p"] = round(p_eff, 3)
                    pool.append(out)
                manifest_cells.append({"domain": dom, "era": era, "wrn": wrn,
                                       "supply": len(cand), "need": need,
                                       "pool": want, "p_eff": round(p_eff, 3)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.out_dir / "pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for rec in pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "version": "composite-pool-v1",
        "created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "target_final_paragraphs": 6 * 3 * NEED_PER_CLASS,
        "need_per_class_per_domain": NEED_PER_CLASS,
        "assumed_reject_rate": args.reject_rate,
        "oversample_factor": round(factor, 3),
        "pool_size": len(pool),
        "unique_text_keys": len({r["text_key"] for r in pool}),
        "paragraph_exclusivity": "text_key 全局唯一（NFKC+去标点+小写 sha256），文档不互斥",
        "doc_split_policy": "A3：文档不要求互斥；doc_split 仅为参考字段，不用于分组",
        "wrn_policy": "A2：WRN 为训练数据内容，只保证每(域×类) 300 段数量，不复核标签语义",
        "pool_by_split_hint": dict(Counter(r["doc_split"] for r in pool)),
        "pool_by_tier": dict(Counter(r["tier"] for r in pool)),
        "pool_by_risk": dict(Counter(flag for r in pool for flag in r["risk_flags"])),
        "excluded_temporal_docs": len(used),
        "filter": {"R_1": ["N", "R", "W"], "year": [1949, 2023], "text_len": [20, 768],
                   "doc_cap": CAP, "paragraph_text_global_dedup": True},
        "era_policy": {"T1_floor_per_class": 0, "T1_bonus": EARLY_T1, "T2_bonus": EARLY_T2,
                       "note": "T1 不强制每类 5 段，尽量多取；T1~T5 均需出现"},
        "evidence_policy": "hit_domains 仅为弱排序信号，不作为 keep/reject 判据",
        "cells": manifest_cells,
        "api_models_available": ["deepseek-v4.1-flash", "kimi-k3.1-pro", "kimi-k3",
                                 "qwen3.8-max-0902", "deepseek-v4-pro-0813",
                                 "qwen3.8-2.4t-a95b"],
        "api_budget_per_model": 1_000_000,
        "planned_token_budget": [2_000_000, 3_000_000],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n候选池 {len(pool):,} 段 → {pool_path}")
    print("按域×类：")
    for idx, dom in TRAIN_DOMAINS.items():
        row = [c for c in manifest_cells if c["domain"] == dom]
        print(f"  {dom}: need={sum(c['need'] for c in row)}  pool={sum(c['pool'] for c in row)}  "
              f"supply={sum(c['supply'] for c in row):,}")
    print(f"年代: {dict(Counter(r['era'] for r in pool))}")
    print(f"tier: {manifest['pool_by_tier']}  risk: {manifest['pool_by_risk']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
