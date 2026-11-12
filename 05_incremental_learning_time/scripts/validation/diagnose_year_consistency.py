"""诊断：为什么按时期切出的 Task 里出现了区间外的年份？

怀疑根因：同一 DocumentID 内部不同段落的 IssueDate 不一致，
而 build 阶段用的是「文档首段年份」筛选文档，却把该文档的其它段落也带了进来。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPORAL_DIR = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "data" / "temporal"
SNAPSHOT = Path(
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b")
CORPUS = SNAPSHOT / "data" / "CN_DOCS_FINAL_complete.parquet"

TASKS = [
    ("T1_1949_1978", 1949, 1978),
    ("T2_1979_1992", 1979, 1992),
    ("T3_1993_2002", 1993, 2002),
    ("T4_2003_2012", 2003, 2012),
    ("T5_2013_2023", 2013, 2023),
]


def main():
    # ---------- 1. Task 层面的越界比例
    print("=" * 88)
    print("【A】各 Task 中「落到时期区间外」的样本占比")
    print("=" * 88)
    print(f"{'Task':<14}{'区间':>12}{'越界数':>9}{'占比':>8}   越界样本年份 top5")
    for name, lo, hi in TASKS:
        recs = []
        for sp in ("train", "test"):
            p = TEMPORAL_DIR / f"{name}_{sp}.jsonl"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    recs += [json.loads(l) for l in f if l.strip()]
        if not recs:
            continue
        out = [r for r in recs if not (lo <= r["year"] <= hi)]
        top = Counter(r["year"] for r in out).most_common(5)
        print(f"{name:<14}{f'{lo}-{hi}':>12}{len(out):>9}{len(out) / len(recs) * 100:>7.1f}%"
              f"   {top}")

    # ---------- 2. 根因：同一 DocumentID 内年份是否一致
    print("\n" + "=" * 88)
    print("【B】根因：语料中同一 DocumentID 内部年份一致性")
    print("=" * 88)
    df = pd.read_parquet(CORPUS, columns=["DocumentID", "IssueDate", "ID"])
    yr = pd.to_numeric(df["IssueDate"].astype(str).str[:4], errors="coerce")
    df["year"] = yr
    df = df[df["year"].between(1949, 2023)]
    g = df.groupby("DocumentID")["year"]
    nun = g.nunique()
    print(f"总文档数: {len(nun):,}")
    print(f"  内部年份唯一的文档: {(nun == 1).sum():,} ({(nun == 1).mean() * 100:.1f}%)")
    print(f"  内部年份不一致的文档: {(nun > 1).sum():,} ({(nun > 1).mean() * 100:.1f}%)")

    multi = nun[nun > 1].index
    if len(multi):
        sub = df[df["DocumentID"].isin(multi[:20000])]
        span = sub.groupby("DocumentID")["year"].agg(["min", "max"])
        span["gap"] = span["max"] - span["min"]
        print(f"\n  不一致文档的年份跨度分布:")
        print(f"    跨度=1 年: {(span['gap'] <= 1).mean() * 100:.1f}%")
        print(f"    跨度<=5 年: {(span['gap'] <= 5).mean() * 100:.1f}%")
        print(f"    跨度>10 年: {(span['gap'] > 10).mean() * 100:.1f}%")
        print(f"    最大跨度: {span['gap'].max():.0f} 年, 中位: {span['gap'].median():.0f} 年")

    # ---------- 3. 抽查一个具体例子
    print("\n" + "=" * 88)
    print("【C】实例抽查：T1 里年份越界的样本")
    print("=" * 88)
    with open(TEMPORAL_DIR / "T1_1949_1978_train.jsonl", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    bad = [r for r in recs if not (1949 <= r["year"] <= 1978)][:5]
    for r in bad:
        print(f"\n  DocumentID={r['DocumentID']}  year={r['year']}  label={r['label']}")
        print(f"  Title: {r['Title'][:60]}")
        print(f"  text: {r['text'][:80]}")
        rows = df[df["DocumentID"] == r["DocumentID"]]
        print(f"  该文档在语料中的全部年份: {sorted(set(rows['year'].astype(int)))}"
              f"  段落数={len(rows)}")


if __name__ == "__main__":
    main()
