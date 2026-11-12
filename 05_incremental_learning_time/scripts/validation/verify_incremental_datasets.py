"""校验增量学习数据集：类别均衡、文档互斥、全局无泄漏、文本无重复。

用法：
    python 05_incremental_learning_time/scripts/validation/verify_incremental_datasets.py temporal
    python 05_incremental_learning_time/scripts/validation/verify_incremental_datasets.py domain
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "data"
TASK_RANGES = {
    "T1_1949_1978": (1949, 1978),
    "T2_1979_1992": (1979, 1992),
    "T3_1993_2002": (1993, 2002),
    "T4_2003_2012": (2003, 2012),
    "T5_2013_2023": (2013, 2023),
}

GOLD_TRAIN = (r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
              r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
              r"\task1_level1\train.parquet")
GOLD_VAL = GOLD_TRAIN.replace("train.parquet", "validation.parquet")


def norm(s):
    return re.sub(r"\s+", "", str(s))


def load_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rep_candidate_counts(dirpath: Path, task: str, split: str):
    report_path = dirpath / "build_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("tasks", {}).get(task, {}).get(split, {}).get(
        "candidate_per_class_year", {})


def gold_texts():
    import ast
    import pandas as pd
    out = set()

    def flat(v):
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, str):
            v = ast.literal_eval(v)
        d = {m["role"]: m["content"] for m in v}
        return d.get("user", "").replace("待判断政策段落：\n", "").strip()

    for f in (GOLD_TRAIN, GOLD_VAL):
        df = pd.read_parquet(f)
        for v in df["messages"]:
            out.add(norm(flat(v)))
    return out


def check(dirpath: Path, tasks: list[str], gold: set):
    print("=" * 78)
    print(f"校验 {dirpath.name}")
    print("=" * 78)

    all_recs = {}
    problems = []

    for t in tasks:
        tr = load_jsonl(dirpath / f"{t}_train.jsonl")
        te = load_jsonl(dirpath / f"{t}_test.jsonl")
        all_recs[(t, "train")] = tr
        all_recs[(t, "test")] = te

        c_tr, c_te = Counter(r["label"] for r in tr), Counter(r["label"] for r in te)
        d_tr = {tuple(r.get("DocumentKey", [r["DocumentID"]])) for r in tr}
        d_te = {tuple(r.get("DocumentKey", [r["DocumentID"]])) for r in te}
        overlap = d_tr & d_te
        e_tr = Counter(r.get("era", "?") for r in tr)
        e_te = Counter(r.get("era", "?") for r in te)

        print(f"\n[{t}]  train {len(tr)}  test {len(te)}")
        print(f"   类别: train {dict(c_tr)}   test {dict(c_te)}")
        print(f"   年代: train {dict(sorted(e_tr.items()))}")
        print(f"         test  {dict(sorted(e_te.items()))}")
        print(f"   文档: train {len(d_tr)} 篇   test {len(d_te)} 篇   交叠 {len(overlap)} 篇")
        if len(tr) != 600:
            problems.append(f"{t} train 数量 {len(tr)} != 600")
        if len(te) != 300:
            problems.append(f"{t} test 数量 {len(te)} != 300")
        if any(c_tr[c] != 200 for c in ("N", "R", "W")):
            problems.append(f"{t} train 类别不均衡 {dict(c_tr)}")
        if any(c_te[c] != 100 for c in ("N", "R", "W")):
            problems.append(f"{t} test 类别不均衡 {dict(c_te)}")
        if overlap:
            problems.append(f"{t} train/test 文档泄漏 {len(overlap)} 篇")
        if t in TASK_RANGES:
            lo, hi = TASK_RANGES[t]
            task_years = {r["year"] for r in tr + te}
            if len(task_years) != hi - lo + 1:
                problems.append(f"{t} train/test 合计年份覆盖 {len(task_years)}/{hi - lo + 1}")
            for split, recs in (("train", tr), ("test", te)):
                years = Counter(r["year"] for r in recs)
                outside = [r for r in recs if not lo <= r["year"] <= hi]
                per_label_year = {
                    label: Counter(r["year"] for r in recs if r["label"] == label)
                    for label in ("N", "R", "W")
                }
                print(f"   {split} 年份: {min(years) if years else '-'}..{max(years) if years else '-'}; "
                      f"覆盖 {len(years)}/{hi - lo + 1}; 每类年度计数范围: "
                      f"{ {label: (min(counts.values()), max(counts.values())) if counts else (0, 0) for label, counts in per_label_year.items()} }")
                if outside:
                    problems.append(f"{t} {split} 有 {len(outside)} 条年份越界")
                candidates = rep_candidate_counts(dirpath, t, split)
                for label, counts in per_label_year.items():
                    for year, count in counts.items():
                        if candidates and count > int(candidates.get(label, {}).get(str(year), 0)):
                            problems.append(f"{t} {split} {label}/{year} 抽中数超过候选容量")

    # 全局交叉检查
    print("\n" + "-" * 78)
    print("全局交叉检查")
    print("-" * 78)

    seen_doc = defaultdict(set)
    for (t, s), recs in all_recs.items():
        for r in recs:
            key = tuple(r.get("DocumentKey", [r["DocumentID"]]))
            seen_doc[key].add(f"{t}/{s}")

    cross = {d: v for d, v in seen_doc.items() if len(v) > 1}
    print(f"  出现在多个 Task/split 的文档: {len(cross)}")
    if cross:
        ex = list(cross.items())[:3]
        for d, v in ex:
            print(f"    例: {d} -> {sorted(v)}")
        problems.append(f"{len(cross)} 篇文档跨 Task/split 复用（全局泄漏）")

    seen_txt = defaultdict(set)
    for (t, s), recs in all_recs.items():
        for r in recs:
            seen_txt[norm(r["text"])].add(f"{t}/{s}")
    dup = {k: v for k, v in seen_txt.items() if len(v) > 1}
    print(f"  文本重复出现在多处: {len(dup)}")
    if dup:
        problems.append(f"{len(dup)} 条文本重复出现")

    hit = sum(1 for k in seen_txt if k in gold)
    print(f"  与人工 gold 600/300 文本重合: {hit}")
    if hit:
        problems.append(f"{hit} 条样本与 gold 重合")

    doc_per_task = {f"{t}/{s}": len(set(r['DocumentID'] for r in recs))
                    for (t, s), recs in all_recs.items()}
    print(f"  各 split 文档数: {doc_per_task}")

    print("\n" + "=" * 78)
    if problems:
        print("发现的问题:")
        for p in problems:
            print(f"  [X] {p}")
    else:
        print("全部校验通过：类别均衡 / 文档互斥 / 全局无泄漏 / 无重复 / 未混入 gold")
    print("=" * 78)
    return len(problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scope", choices=["temporal", "domain"])
    args = ap.parse_args()

    d = DATA_ROOT / args.scope
    tasks = [p.name[: -len("_train.jsonl")]
             for p in sorted(d.glob("*_train.jsonl"))]
    if not tasks:
        print(f"{d} 下没有任务文件")
        return 1

    print("载入人工 gold 文本用于泄漏比对 ...", flush=True)
    gold = gold_texts()
    return 1 if check(d, tasks, gold) else 0


if __name__ == "__main__":
    sys.exit(main())
