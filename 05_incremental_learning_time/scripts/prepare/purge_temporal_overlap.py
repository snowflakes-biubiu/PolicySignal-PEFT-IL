"""替换 composite/final 中与 temporal 数据集文本重叠的段落（同格补位，配额不变）。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "assets" / "05_incremental_learning_time" / "data"
OUT = DATA / "domain_v14" / "composite"
FINAL = OUT / "final"
TEMPORAL = DATA / "temporal"


def load(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(s):
    return "".join(str(s).split()).casefold()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t_texts = set()
    for p in TEMPORAL.glob("T*_*_*.jsonl"):
        if "manifest" in p.name:
            continue
        for r in load(p):
            t_texts.add(norm(r["text"]))

    files = sorted(FINAL.glob("D*_*.jsonl"))
    final = {}
    for p in files:
        final[p.name] = load(p)
    all_final = [r for rows in final.values() for r in rows]
    bad_pids = {r["pid"] for r in all_final if norm(r["text"]) in t_texts}
    print(f"与 temporal 重叠 {len(bad_pids)} 段；最终集 {len(all_final):,} 段", flush=True)
    if not bad_pids:
        return 0

    # 已通过逐段复核的候选池
    verdicts = defaultdict(dict)
    for v in load(OUT / "verdicts.jsonl"):
        verdicts[v["pid"]][v["round"]] = v
    best = {}
    for pid, rounds in verdicts.items():
        ok = [r for k, r in rounds.items() if r.get("parsed")]
        if ok:
            best[pid] = max(ok, key=lambda r: r["round"])

    def disputed(pid):
        d = verdicts[pid]
        return (1 in d and 2 in d and 3 not in d
                and (d[1]["keep"] != d[2]["keep"]
                     or d[1]["verdict_domain"] != d[2]["verdict_domain"]))

    taken = {r["pid"] for r in all_final}
    taken_tk = {r["text_key"] for r in all_final}
    candidates = []
    for r in load(OUT / "pool.jsonl"):
        v = best.get(r["pid"])
        if (r["pid"] in taken or r["text_key"] in taken_tk or disputed(r["pid"])
                or norm(r["text"]) in t_texts or not v or not v["keep"]
                or float(v["confidence"] or 0) < 0.80
                or v["verdict_domain"] != r["domain"]):
            continue
        candidates.append(r)
    print(f"可用补位候选 {len(candidates):,} 段", flush=True)

    by_group = defaultdict(list)
    for r in candidates:
        by_group[(r["domain"], r["wrn"])].append(r)
    for v in by_group.values():
        v.sort(key=lambda r: r["rank"])

    # 记录被移除段的 (文件, 域, split, WRN)，逐组补位
    removed = []
    for fname, rows in final.items():
        split = "train" if "train" in fname else "test"
        for r in rows:
            if r["pid"] in bad_pids:
                removed.append((fname, split, r["domain"], r["wrn"]))
    need = Counter((d, w) for _, _, d, w in removed)
    print(f"待补位分组: {dict(need)}", flush=True)

    replacements = defaultdict(list)
    for (dom, wrn), n in need.items():
        pool = by_group[(dom, wrn)]
        picked = [r for r in pool if r["pid"] not in taken][:n]
        for r in picked:
            taken.add(r["pid"])
            taken_tk.add(r["text_key"])
        replacements[(dom, wrn)] = picked
        if len(picked) < n:
            print(f"[缺口] {dom}/{wrn} 只找到 {len(picked)}/{n}", flush=True)

    idx = Counter()
    changed = set()
    for fname, split, dom, wrn in removed:
        key = (dom, wrn)
        r = replacements[key][idx[key]]
        idx[key] += 1
        rows = final[fname]
        pos = next(i for i, x in enumerate(rows) if x["pid"] in bad_pids)
        rows[pos] = r
        changed.add(fname)
    for fname in changed:
        with (FINAL / fname).open("w", encoding="utf-8") as f:
            for r in final[fname]:
                f.write(json.dumps({k: r.get(k) for k in
                                    ("pid", "text_key", "doc_id", "Title", "year", "era",
                                     "domain", "wrn", "text", "label_model", "label_conf",
                                     "tier", "rank")}, ensure_ascii=False) + "\n")
    used = sum(idx.values())
    print(f"已替换 {used} 段，涉及 {len(changed)} 个文件", flush=True)
    (OUT / "purge_log.json").write_text(json.dumps({
        "removed": len(bad_pids), "replaced": used,
        "by_group": {f"{d}|{w}": n for (d, w), n in need.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if used == len(bad_pids) else 1


if __name__ == "__main__":
    sys.exit(main())
