"""把人工核对通过的 5,400 段整理成 D1~D6 训练数据集（与 temporal 同格式）。

来源：data/domain_v14/composite/final/D{1..6}_{train,test}.jsonl
输出：data/domain_incremental/D{k}_{域名}_{train,test}.jsonl + _manifest.csv + build_report.json

字段契约照抄 build_incremental_datasets.make_records()：text / label / label_source /
level2_auto / DocumentID / DocumentKey / ParagraphID / year / era / Title /
IssuingDepartment / Category / Validity / period / domain / split，另加 task。
元数据（IssueDate / ID / R_2 / 发文机关等）从原始语料 parquet 按 (DocumentID, 段落文本) 回查补齐。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))
from domain_taxonomy_v14 import TRAIN_DOMAINS  # noqa: E402

DATA = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_time" / "data"
SRC = DATA / "domain_v14" / "composite" / "final"
OUT = DATA / "domain_incremental"
CORPUS = Path(r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
              r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\data\CN_DOCS_FINAL_complete.parquet")
COLS = ["DocumentID", "ID", "Paragraph_content", "Title", "IssueDate",
        "IssuingDepartment", "Category", "Validity", "R_1", "R_2"]
ERAS = [(1949, 1978, "T1_1949_1978"), (1979, 1992, "T2_1979_1992"),
        (1993, 2002, "T3_1993_2002"), (2003, 2012, "T4_2003_2012"),
        (2013, 2023, "T5_2013_2023")]


def norm(s: str) -> str:
    return "".join(str(s).split()).casefold()


def era_of(year: int) -> str:
    for a, b, name in ERAS:
        if a <= year <= b:
            return name
    return "T?"


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = []
    for k, dom in TRAIN_DOMAINS.items():
        for split in ("train", "test"):
            for r in load_jsonl(SRC / f"D{k}_{split}.jsonl"):
                rows.append((k, dom, split, r))
    print(f"读入 {len(rows):,} 段", flush=True)

    targets = {(str(r["doc_id"]), norm(r["text"])): r for _, _, _, r in rows}
    doc_ids = {k[0] for k in targets}
    meta = {}
    pf = pq.ParquetFile(CORPUS)
    for batch in pf.iter_batches(batch_size=200_000, columns=COLS):
        d = batch.to_pandas()
        d = d[d["DocumentID"].astype(str).isin(doc_ids)]
        for rec in d.to_dict("records"):
            key = (str(rec["DocumentID"]), norm(rec["Paragraph_content"]))
            if key in targets and key not in meta:
                meta[key] = rec
    print(f"元数据回查命中 {len(meta):,}/{len(targets):,}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"version": "domain-incremental-v14",
              "source": "data/domain_v14/composite/final (人工核对通过的 5,400 段)",
              "task_order": {f"D{k}": d for k, d in TRAIN_DOMAINS.items()},
              "quotas": {"train": 600, "test": 300},
              "year_range": [1949, 2023],
              "design": {"paragraph_exclusive": True, "document_exclusive": False,
                         "note": "A3：文档不要求互斥，段落内容严格互斥；T1~T5 全覆盖"},
              "tasks": {}}
    manifest_fields = ["task", "split", "label", "DocumentID", "year",
                       "DocumentKey", "domain", "text_len"]

    for k, dom in TRAIN_DOMAINS.items():
        stem = f"D{k}_{dom.replace('/', '_')}"
        task_info = {"domain": dom, "stem": stem}
        for split in ("train", "test"):
            recs = []
            for kk, dd, sp, r in rows:
                if kk != k or sp != split:
                    continue
                m = meta.get((str(r["doc_id"]), norm(r["text"])), {})
                issue = str(m.get("IssueDate") or r["year"])
                recs.append({
                    "text": r["text"],
                    "label": r["wrn"],
                    "label_source": "auto_R1",
                    "level2_auto": str(m.get("R_2") or ""),
                    "DocumentID": str(r["doc_id"]),
                    "DocumentKey": [str(r["doc_id"]), issue, str(r["Title"])],
                    "ParagraphID": str(m.get("ID") or ""),
                    "year": int(r["year"]),
                    "era": era_of(int(r["year"])),
                    "Title": str(r["Title"]),
                    "IssuingDepartment": str(m.get("IssuingDepartment") or ""),
                    "Category": str(m.get("Category") or ""),
                    "Validity": str(m.get("Validity") or ""),
                    "period": stem,
                    "domain": dom,
                    "split": split,
                    "task": f"D{k}",
                })
            recs.sort(key=lambda x: (x["label"], x["era"], x["DocumentID"], x["ParagraphID"]))
            with (OUT / f"{stem}_{split}.jsonl").open("w", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with (OUT / f"{stem}_{split}_manifest.csv").open(
                    "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=manifest_fields)
                w.writeheader()
                for rec in recs:
                    w.writerow({"task": stem, "split": split, "label": rec["label"],
                                "DocumentID": rec["DocumentID"], "year": rec["year"],
                                "DocumentKey": json.dumps(rec["DocumentKey"], ensure_ascii=False),
                                "domain": dom, "text_len": len(rec["text"])})
            per_class = Counter(rec["label"] for rec in recs)
            task_info[split] = {
                "n": len(recs), "per_class": dict(per_class),
                "docs": len({tuple(rec["DocumentKey"]) for rec in recs}),
                "by_class_era": {lab: dict(Counter(rec["era"] for rec in recs
                                                   if rec["label"] == lab)) for lab in "NRW"},
            }
            print(f"  {stem}_{split}: {len(recs)} 条 {dict(per_class)}", flush=True)
        report["tasks"][f"D{k}"] = task_info

    (OUT / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
