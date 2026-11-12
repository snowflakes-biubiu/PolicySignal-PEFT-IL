"""复合实验监督面板：判定进度、质量、token 账（只读）。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "assets" / "05_incremental_learning_domain" / "data" / "domain_v14" / "composite"


def load(p: Path):
    if not p.is_file():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pool = load(OUT / "pool.jsonl")
    vs = load(OUT / "verdicts.jsonl")
    budget = json.loads((OUT / "model_budget.json").read_text(encoding="utf-8"))
    print(f"池 {len(pool):,}  已判定 {len(vs):,}  完成度 {len(vs)/max(len(pool),1):.1%}")
    print("token: " + "  ".join(f"{m}={v:,}" for m, v in budget["tokens"].items()))
    if not vs:
        return 0
    by_round = Counter(v["round"] for v in vs)
    by_model = Counter(v["model"] for v in vs)
    parsed = [v for v in vs if v["parsed"]]
    print(f"轮次 {dict(by_round)}  模型 {dict(by_model)}")
    print(f"解析成功 {len(parsed):,}/{len(vs):,} = {len(parsed)/len(vs):.1%}")
    print(f"keep=true {sum(1 for v in parsed if v['keep']):,}/{len(parsed):,} = "
          f"{sum(1 for v in parsed if v['keep'])/max(len(parsed),1):.1%}"
          f"（pilot 拒绝率 23.3%）")
    print(f"领域一致 {sum(1 for v in parsed if v['verdict_domain']==v['domain']):,}/{len(parsed):,}")
    conf = [float(v["confidence"] or 0) for v in parsed]
    print(f"平均置信度 {sum(conf)/max(len(conf),1):.3f}")
    for field in ("domain", "era", "wrn"):
        d = defaultdict(list)
        for v in parsed:
            d[v[field]].append(v)
        parts = [f"{k}:{sum(1 for x in a if x['keep'])/len(a):.0%}" for k, a in sorted(d.items())]
        print(f"  keep by {field}: " + "  ".join(parts))

    # 按 --compose 的口径预测最终可接受量与格子缺口
    pool_by_pid = {r["pid"]: r for r in pool}
    best = {}
    for v in vs:
        cur = best.get(v["pid"])
        if v.get("parsed") and (cur is None or v["round"] >= cur["round"]):
            best[v["pid"]] = v
        elif cur is None:
            best[v["pid"]] = v
    accepted = [pool_by_pid[pid] for pid, v in best.items()
                if pid in pool_by_pid and v["parsed"] and v["keep"]
                and float(v["confidence"] or 0) >= 0.80
                and v["verdict_domain"] == v["domain"]]
    seen, cell_cnt = set(), Counter()
    for r in sorted(accepted, key=lambda r: r["rank"]):
        tk = r.get("text_key")
        if tk in seen:
            continue
        seen.add(tk)
        cell_cnt[(r["domain"], r["era"], r["wrn"])] += 1
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    short = []
    for c in manifest["cells"]:
        key = (c["domain"], c["era"], c["wrn"])
        if cell_cnt[key] < c["need"]:
            short.append((key, cell_cnt[key], c["need"]))
    print(f"\n[预测] 可接受 {len(accepted):,}/{len(pool):,}；"
          f"严格去重后 {len(seen):,}；目标 5,400")
    print(f"[预测] 不足格子 {len(short)}/90，合计缺口 {sum(n-h for _, h, n in short):,} 段")
    for key, have, need in sorted(short, key=lambda x: x[2] - x[1], reverse=True)[:8]:
        print(f"   {key}: {have}/{need}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
