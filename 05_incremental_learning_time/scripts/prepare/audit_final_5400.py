"""最终 5,400 段训练集的独立离线验收（不调用 API，不依赖 compose 的自报结果）。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "assets" / "05_incremental_learning_time" / "data" / "domain_v14"
FINAL = DATA / "composite" / "final"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_composite_pool import text_key_of  # noqa: E402
from domain_taxonomy_v14 import TRAIN_DOMAINS  # noqa: E402


def load(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    all_rows = []
    per_file = {}
    for idx, dom in TRAIN_DOMAINS.items():
        for split in ("train", "test"):
            p = FINAL / f"D{idx}_{split}.jsonl"
            rows = load(p) if p.is_file() else []
            per_file[p.name] = len(rows)
            for r in rows:
                r = dict(r)
                r["_split"] = split
                all_rows.append(r)
    print(f"文件行数: {per_file}")
    print(f"总行数: {len(all_rows):,}")

    # A3：段落内容严格互斥
    tkeys = [r.get("text_key") or text_key_of(r["text"]) for r in all_rows]
    tr = {r["text_key"] for r in all_rows if r["_split"] == "train"}
    te = {r["text_key"] for r in all_rows if r["_split"] == "test"}
    print(f"text_key 唯一: {len(set(tkeys)):,}/{len(tkeys):,}  "
          f"train∩test={len(tr & te)}  文档互斥=未要求")

    # A2：WRN 数量
    by_dw = Counter((r["domain"], r["wrn"]) for r in all_rows)
    by_split_wrn = Counter((r["domain"], r["_split"], r["wrn"]) for r in all_rows)
    print(f"每(域×WRN)=300: {all(v == 300 for v in by_dw.values())}  "
          f"组合数={len(by_dw)}  min={min(by_dw.values())} max={max(by_dw.values())}")
    print(f"每(域×split×WRN) train200/test100: "
          f"{all(v == (200 if s == 'train' else 100) for (_, s, _), v in by_split_wrn.items())}")

    # 年代与领域
    eras = Counter(r["era"] for r in all_rows)
    per_dom_era = Counter((r["domain"], r["era"]) for r in all_rows)
    era_names = ["T1_1949_1978", "T2_1979_1992", "T3_1993_2002", "T4_2003_2012", "T5_2013_2023"]
    coverage = {d: [per_dom_era[(d, e)] for e in era_names] for _, d in TRAIN_DOMAINS.items()}
    print(f"年代分布: {dict(eras)}")
    print("每域 T1~T5 覆盖:")
    for d, v in coverage.items():
        print(f"  {d}: {v}  (T1~T5 全覆盖={all(x > 0 for x in v)})")

    # 领域合法性 + 每域 train/test
    ok_domain = all(r["domain"] in TRAIN_DOMAINS.values() for r in all_rows)
    per_ds = Counter((r["domain"], r["_split"]) for r in all_rows)
    print(f"领域合法: {ok_domain}  每域 train600/test300: "
          f"{all(per_ds[(d, 'train')] == 600 and per_ds[(d, 'test')] == 300 for _, d in TRAIN_DOMAINS.items())}")

    # 与复核记录对照：所有最终段都应有 keep=true 的判定
    verdicts = {}
    for v in load(DATA / "composite" / "verdicts.jsonl"):
        if v.get("parsed") and (v["pid"] not in verdicts or v["round"] >= verdicts[v["pid"]]["round"]):
            verdicts[v["pid"]] = v
    matched = sum(1 for r in all_rows if (v := verdicts.get(r["pid"]))
                  and v["keep"] and v["verdict_domain"] == r["domain"])
    print(f"有 keep=true 且领域一致的复核记录: {matched:,}/{len(all_rows):,}")

    passed = (len(all_rows) == 5400 and len(set(tkeys)) == len(tkeys) and len(tr & te) == 0
              and all(v == 300 for v in by_dw.values()) and len(by_dw) == 18
              and all(v == (200 if s == "train" else 100) for (_, s, _), v in by_split_wrn.items())
              and ok_domain and matched == len(all_rows)
              and all(per_ds[(d, "train")] == 600 and per_ds[(d, "test")] == 300
                      for _, d in TRAIN_DOMAINS.items())
              and all(all(x > 0 for x in v) for v in coverage.values()))
    print(f"\nAUDIT_PASS = {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
