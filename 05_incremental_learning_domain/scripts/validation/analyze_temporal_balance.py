"""时间增量 T1-T5 的均衡性与公平性体检。

关注维度（跨 Task 可比性才是「公平」的核心）：
  1. 抽样率  —— 每个 Task 用掉候选文档的比例，衡量资源紧张度
  2. 年份覆盖 —— Task 内部是否集中在少数年份
  3. 文本长度 —— 跨期系统性差异会成为伪特征
  4. 文体构成 —— Category / 发文机关集中度，混淆变量
  5. train/test 内部一致性 —— 两侧年份与长度分布是否同分布

用法：
    python 05_incremental_learning_domain/scripts/validation/analyze_temporal_balance.py
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPORAL_DIR = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "data" / "temporal"
REPORT = TEMPORAL_DIR / "build_report.json"

TASKS = [
    ("T1_1949_1978", 1949, 1978),
    ("T2_1979_1992", 1979, 1992),
    ("T3_1993_2002", 1993, 2002),
    ("T4_2003_2012", 2003, 2012),
    ("T5_2013_2023", 2013, 2023),
]


def load_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def describe(recs, years_range):
    """返回一组 Task 侧（train 或 test）的描述统计。"""
    years = [r["year"] for r in recs]
    lens = [len(r["text"]) for r in recs]
    by_doc = Counter(tuple(r.get("DocumentKey", [r["DocumentID"]])) for r in recs)
    yc = Counter(years)
    lo, hi = years_range
    covered = sum(1 for y in range(lo, hi + 1) if yc.get(y, 0) > 0)
    # 最大单年占比：衡量是否堆在个别年份
    top_year, top_n = yc.most_common(1)[0]
    return {
        "n": len(recs),
        "docs": len(by_doc),
        "max_para_per_doc": max(by_doc.values()),
        "label": dict(Counter(r["label"] for r in recs)),
        "years_min": min(years), "years_max": max(years),
        "years_covered": f"{covered}/{hi - lo + 1}",
        "top_year": (top_year, top_n, round(top_n / len(recs) * 100, 1)),
        "len_mean": round(statistics.mean(lens), 1),
        "len_median": statistics.median(lens),
        "len_p90": sorted(lens)[int(len(lens) * 0.9)],
        "len_by_label": {lab: round(statistics.mean(
            [len(r["text"]) for r in recs if r["label"] == lab]), 1)
            for lab in ("N", "R", "W")},
        "category_top3": Counter(r["Category"] for r in recs).most_common(3),
        "dept_unique": len(set(r["IssuingDepartment"] for r in recs)),
        "dept_top1_share": round(Counter(
            r["IssuingDepartment"] for r in recs).most_common(1)[0][1]
            / len(recs) * 100, 1),
    }


def main():
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    print("=" * 100)
    print("时间增量 T1-T5 均衡性 / 公平性体检")
    print("=" * 100)

    print(f"\n{'Task':<14}{'候选文档':>9}{'用掉文档':>9}{'抽样率':>9}"
          f"{'train n':>9}{'test n':>8}{'文档数 tr/te':>14}{'单文档最多段':>13}")
    print("-" * 100)
    rows = {}
    for name, lo, hi in TASKS:
        tr = load_jsonl(TEMPORAL_DIR / f"{name}_train.jsonl")
        te = load_jsonl(TEMPORAL_DIR / f"{name}_test.jsonl")
        cand = rep["tasks"][name]["candidate_docs"]
        d_tr = rep["tasks"][name]["train"]["docs"]
        d_te = rep["tasks"][name]["test"]["docs"]
        used = d_tr + d_te
        rows[name] = {"tr": describe(tr, (lo, hi)), "te": describe(te, (lo, hi)),
                      "cand": cand}
        print(f"{name:<14}{cand:>9,}{used:>9,}{used / cand * 100:>8.2f}%"
              f"{len(tr):>9}{len(te):>8}{f'{d_tr}/{d_te}':>14}"
              f"{max(rows[name]['tr']['max_para_per_doc'], rows[name]['te']['max_para_per_doc']):>13}")

    print("\n\n" + "=" * 100)
    print("【1】年份覆盖与集中度（Task 内部是否堆在个别年份）")
    print("=" * 100)
    print(f"{'Task':<14}{'年份区间实际':>16}{'覆盖年数':>12}{'最热年份占比':>16}"
          f"{'(train)':>10}")
    for name, lo, hi in TASKS:
        t = rows[name]["tr"]
        span = f"{t['years_min']}-{t['years_max']}"
        hot = f"{t['top_year'][0]}: {t['top_year'][2]}%"
        print(f"{name:<14}{span:>16}{t['years_covered']:>12}{hot:>16}{'(train)':>10}")

    print("\n\n" + "=" * 100)
    print("【2】文本长度（跨期系统性差异 = 伪特征风险）")
    print("=" * 100)
    print(f"{'Task':<14}{'mean':>8}{'median':>8}{'p90':>8}"
          f"{'N':>8}{'R':>8}{'W':>8}   (按类别的 mean)")
    for name, *_ in TASKS:
        t = rows[name]["tr"]
        lb = t["len_by_label"]
        print(f"{name:<14}{t['len_mean']:>8}{t['len_median']:>8}{t['len_p90']:>8}"
              f"{lb['N']:>8}{lb['R']:>8}{lb['W']:>8}")
    diffs = [rows[n]["tr"]["len_mean"] for n, *_ in TASKS]
    print(f"\n  最早 Task vs 最晚 Task 平均长度差: "
          f"{diffs[-1] - diffs[0]:+.1f} 字 "
          f"(相对 {diffs[-1] / diffs[0] - 1:+.1%})")

    print("\n\n" + "=" * 100)
    print("【3】文体构成（混淆变量：各部门规章占比、发文机关集中度）")
    print("=" * 100)
    print(f"{'Task':<14}{'Category top3':>46}{'机关数':>8}{'最热机关占比':>12}")
    for name, *_ in TASKS:
        t = rows[name]["tr"]
        cats = "; ".join(f"{k} {v}" for k, v in t["category_top3"])
        print(f"{name:<14}{cats:>46}{t['dept_unique']:>8}"
              f"{t['dept_top1_share']:>11}%")

    print("\n\n" + "=" * 100)
    print("【4】train / test 同分布检查（Task 内部一致性）")
    print("=" * 100)
    print(f"{'Task':<14}{'train mean':>11}{'test mean':>11}{'长度差':>9}"
          f"{'train top年':>13}{'test top年':>13}")
    for name, *_ in TASKS:
        tr, te = rows[name]["tr"], rows[name]["te"]
        print(f"{name:<14}{tr['len_mean']:>11}{te['len_mean']:>11}"
              f"{te['len_mean'] - tr['len_mean']:>+9.1f}"
              f"{str(tr['top_year'][0]):>13}{str(te['top_year'][0]):>13}")

    print("\n\n" + "=" * 100)
    print("【5】WRN 类别均衡（强制配额）")
    print("=" * 100)
    for name, *_ in TASKS:
        tr, te = rows[name]["tr"], rows[name]["te"]
        print(f"{name:<14} train {tr['label']}   test {te['label']}")


if __name__ == "__main__":
    main()
