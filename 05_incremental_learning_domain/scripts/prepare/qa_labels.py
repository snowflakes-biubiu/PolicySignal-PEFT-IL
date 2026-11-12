"""抽查领域标签质量，重点盯能力偏弱的模型（默认 qwen3.8-27b）。

参照口径（按用户确认）：
  * `candidates_priority.jsonl` 的 hit_domains 视为**可信参照**，用于算一致性；
  * `candidates.jsonl` 的 hit_domains 只作弱参考（已确认不够准），单列一栏；
  * LLM 预测与可信参照冲突的行会被单独列出，供人工判别是否误标。

做四件事（只读，不改任何数据）：
  1. 按模型汇总：条数、无法归类占比、平均置信度、需人工复核占比、两套一致性；
  2. 与其他模型加权基线对比，给出「差多少个百分点」；
  3. 导出可人工眼看的样本：随机样本、低置信/无法归类样本、**与可信参照冲突的疑似误标**；
  4. 可选 `--compare` 另一份标签文件，两模型在同一批标题上的逐条对比。

用法：
    python 05_incremental_learning_domain/scripts/prepare/qa_labels.py
    python 05_incremental_learning_domain/scripts/prepare/qa_labels.py --target qwen3.8-27b --sample 40
    python 05_incremental_learning_domain/scripts/prepare/qa_labels.py \
        --labels 05_incremental_learning_domain/data/domain_v14/_calibration/qa_27b.jsonl \
        --compare 05_incremental_learning_domain/data/domain_v14/title_domain_labels_v14.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))

from domain_taxonomy_v14 import DOMAIN_UNSURE  # noqa: E402
from title_utils import title_key  # noqa: E402

DATA = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_domain" / "data" / "domain_v14"
LABELS = DATA / "title_domain_labels_v14.jsonl"
PRIORITY = DATA / "candidates_priority.jsonl"
CANDIDATES = DATA / "candidates.jsonl"
OUT_MD = DATA / "_calibration" / "label_qa.md"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def index_ref(path: Path) -> dict[str, list[str]]:
    ref: dict[str, list[str]] = {}
    for rec in load_jsonl(path):
        key = title_key(str(rec.get("Title", "")))
        ref[key] = rec.get("hit_domains") or []
    return ref


def pct(num: int, den: int) -> str:
    return f"{num / den:.1%}" if den else "—"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--ref", type=Path, default=PRIORITY, help="可信参照（默认优先级候选）")
    ap.add_argument("--weak-ref", type=Path, default=CANDIDATES, help="弱参照（默认全体候选）")
    ap.add_argument("--compare", type=Path, default=None, help="另一份标签文件，逐条对比")
    ap.add_argument("--target", default="qwen3.8-27b")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.labels) if "error" not in r]
    if not rows:
        print(f"{args.labels} 还没有成功入库的记录")
        return 2
    ref = index_ref(args.ref)
    weak = index_ref(args.weak_ref)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r.get("model", "?")].append(r)

    def judge(items, table):
        hits = [r for r in items if table.get(r.get("title_key"))]
        agree = sum(1 for r in hits if r.get("domain") in table[r["title_key"]])
        return agree, len(hits)

    lines = ["# 领域标签抽查报告", "",
             f"标签文件：`{args.labels.name}`；总入库 {len(rows):,} 条，覆盖 {len(by_model)} 个模型。",
             f"可信参照：`{args.ref.name}`（{len(ref):,} 条）；弱参照：`{args.weak_ref.name}`。", "",
             "| 模型 | 条数 | 无法归类 | 平均置信度 | 需复核 | 可信参照一致性 | 弱参照一致性 |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for model, items in sorted(by_model.items(), key=lambda x: -len(x[1])):
        n = len(items)
        unsure = sum(1 for r in items if r.get("domain") == DOMAIN_UNSURE) / n
        conf = sum(float(r.get("confidence") or 0) for r in items) / n
        review = sum(1 for r in items if r.get("manual_review")) / n
        a_pri, j_pri = judge(items, ref)
        a_weak, j_weak = judge(items, weak)
        summary[model] = (n, unsure, conf, review, a_pri, j_pri)
        lines.append(f"| {model} | {n:,} | {unsure:.1%} | {conf:.3f} | {review:.1%} | "
                     f"{pct(a_pri, j_pri)}（{j_pri:,}） | {pct(a_weak, j_weak)}（{j_weak:,}） |")

    others = [(m, v) for m, v in summary.items() if m != args.target]
    base = None
    if others:
        den = sum(v[5] for _, v in others)
        num = sum(v[4] for _, v in others)
        if den:
            base = num / den
            lines += ["", f"其他模型对可信参照的加权一致性基线：{base:.1%}（{den:,} 条可比）"]

    target_rows = by_model.get(args.target, [])
    if target_rows:
        tn, tu, tc, tr, ta, tj = summary[args.target]
        lines += ["", f"## {args.target} 抽查样本", "",
                  f"该模型 {tn:,} 条；无法归类 {tu:.1%}；对可信参照一致性 {pct(ta, tj)}（{tj:,} 条可比）"]
        if base is not None and tj:
            gap = ta / tj - base
            verdict = "偏低，需人工重点复核" if gap < -0.10 else "差距可控"
            lines.append(f"与基线差 {gap:+.1%} —— {verdict}")

        rng = random.Random(args.seed)
        picks = rng.sample(target_rows, min(args.sample, len(target_rows)))
        low = sorted([r for r in target_rows
                      if r.get("domain") == DOMAIN_UNSURE or float(r.get("confidence") or 0) < 0.6],
                     key=lambda r: float(r.get("confidence") or 0))[:15]
        conflict = [r for r in target_rows
                    if ref.get(r.get("title_key")) and r.get("domain") not in ref[r["title_key"]]][:20]

        def fmt(r):
            hd = "、".join(ref.get(r.get("title_key")) or []) or "—"
            return (f"| {r.get('Title', '')[:48]} | {r.get('domain')} | "
                    f"{r.get('confidence')} | {hd} |")

        head = "| 标题 | 预测领域 | 置信度 | 可信参照 hit_domains |\n|---|---|---|---|"
        lines += ["", "### 随机样本（人工核对语义）", "", head]
        lines += [fmt(r) for r in picks]
        if conflict:
            lines += ["", "### 与可信参照冲突（疑似误标，人工判别）", "", head]
            lines += [fmt(r) for r in conflict]
        if low:
            lines += ["", "### 低置信 / 无法归类（错误高发区）", "", head]
            lines += [fmt(r) for r in low]
    else:
        lines += ["", f"## 暂无 {args.target} 的结果"]

    if args.compare:
        other = {r.get("title_key"): r for r in load_jsonl(args.compare) if "error" not in r}
        shared = [r for r in rows if r.get("title_key") in other]
        if shared:
            agree = sum(1 for r in shared
                        if r.get("domain") == other[r["title_key"]].get("domain"))
            lines += ["", f"## 与 `{args.compare.name}` 逐条对比", "",
                      f"共同标题 {len(shared):,} 条，领域完全一致 {agree:,} 条（{agree / len(shared):.1%}）"]
            diff = [r for r in shared
                    if r.get("domain") != other[r["title_key"]].get("domain")][:30]
            if diff:
                lines += ["", "| 标题 | 本文件 | 对比文件 | 可信参照 |", "|---|---|---|---|"]
                for r in diff:
                    o = other[r["title_key"]]
                    hd = "、".join(ref.get(r.get("title_key")) or []) or "—"
                    lines.append(f"| {r.get('Title', '')[:48]} | {r.get('domain')} | "
                                 f"{o.get('domain')} | {hd} |")
        else:
            lines += ["", f"## 与 `{args.compare.name}` 无共同标题，无法对比"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"\n完整报告：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
