"""独立校验 D1~D6 训练数据集（temporal 同格式契约 + A3 段落互斥）。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "assets" / "05_incremental_learning_time" / "data"
OUT = DATA / "domain_incremental"
TEMPORAL = DATA / "temporal"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain_taxonomy_v14 import TRAIN_DOMAINS  # noqa: E402

FIELDS = ["text", "label", "label_source", "level2_auto", "DocumentID", "DocumentKey",
          "ParagraphID", "year", "era", "Title", "IssuingDepartment", "Category",
          "Validity", "period", "domain", "split"]
ERAS = [(1949, 1978, "T1_1949_1978"), (1979, 1992, "T2_1979_1992"),
        (1993, 2002, "T3_1993_2002"), (2003, 2012, "T4_2003_2012"),
        (2013, 2023, "T5_2013_2023")]


def load(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def era_of(y):
    return next((n for a, b, n in ERAS if a <= y <= b), "T?")


def norm(s):
    return "".join(str(s).split()).casefold()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    problems = []
    all_rows = []
    print("任务  域  split  条数  N/R/W  文档数  文本长度范围")
    for k, dom in TRAIN_DOMAINS.items():
        stem = f"D{k}_{dom.replace('/', '_')}"
        for split, expect in (("train", 600), ("test", 300)):
            p = OUT / f"{stem}_{split}.jsonl"
            m = OUT / f"{stem}_{split}_manifest.csv"
            if not p.is_file() or not m.is_file():
                problems.append(f"缺文件 {stem}_{split}")
                continue
            recs = load(p)
            all_rows += [(stem, split, r) for r in recs]
            cnt = Counter(r["label"] for r in recs)
            lens = [len(r["text"]) for r in recs]
            docs = len({tuple(r["DocumentKey"]) for r in recs})
            print(f"D{k}   {dom[:6]:<7}{split:<6}{len(recs):>5}  "
                  f"{cnt['N']}/{cnt['R']}/{cnt['W']}  {docs:>5}  {min(lens)}~{max(lens)}")
            if len(recs) != expect:
                problems.append(f"{stem}/{split} 条数 {len(recs)} != {expect}")
            per = expect // 3
            if any(cnt[x] != per for x in "NRW"):
                problems.append(f"{stem}/{split} 类别不均 {dict(cnt)}")
            for i, r in enumerate(recs):
                miss = [f for f in FIELDS if f not in r]
                if miss:
                    problems.append(f"{stem}/{split}#{i} 缺字段 {miss}")
                if r.get("split") != split:
                    problems.append(f"{stem}/{split}#{i} split={r.get('split')}")
                if r.get("label") not in ("N", "R", "W"):
                    problems.append(f"{stem}/{split}#{i} label={r.get('label')}")
                if not isinstance(r.get("text"), str) or not (20 <= len(r["text"]) <= 768):
                    problems.append(f"{stem}/{split}#{i} 文本长度 {len(r.get('text', ''))}")
                y = int(r.get("year", -1))
                if not 1949 <= y <= 2023 or r.get("era") != era_of(y):
                    problems.append(f"{stem}/{split}#{i} 年份/年代 {y}/{r.get('era')}")

    # A3：段落内容全局严格互斥
    texts = [norm(r["text"]) for _, _, r in all_rows]
    dups = len(texts) - len(set(texts))
    print(f"\n段落文本全局唯一: {len(set(texts)):,}/{len(texts):,}（重复 {dups}）")
    if dups:
        problems.append(f"段落文本重复 {dups}")

    # 与 temporal 数据集不重叠
    t_texts = set()
    for p in TEMPORAL.glob("T*_*_*.jsonl"):
        if "_manifest" in p.name:
            continue
        for r in load(p):
            t_texts.add(norm(r["text"]))
    overlap = sum(1 for t in texts if t in t_texts)
    print(f"与 temporal 数据集文本重叠: {overlap}")
    if overlap:
        problems.append(f"与 temporal 重叠 {overlap} 段")

    # 文档跨 split / 跨任务情况（A3 允许，仅供训练侧知晓）
    doc_loc = defaultdict(set)
    for stem, split, r in all_rows:
        doc_loc[tuple(r["DocumentKey"])].add(f"{stem}/{split}")
    cross = sum(1 for v in doc_loc.values() if len(v) > 1)
    print(f"文档出现在多个 (任务/划分): {cross:,} / {len(doc_loc):,} 篇（A3 允许）")
    same_split_doc_dup = sum(1 for stem, split, _ in all_rows)
    print(f"总段数 {same_split_doc_dup:,}；独立文档 {len(doc_loc):,}")

    report = {"total": len(all_rows), "unique_texts": len(set(texts)),
              "overlap_with_temporal": overlap, "cross_location_docs": cross,
              "unique_docs": len(doc_loc), "problems": problems,
              "pass": not problems}
    (OUT / "validate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVALIDATE_PASS = {not problems}")
    for p in problems[:20]:
        print("  !", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
