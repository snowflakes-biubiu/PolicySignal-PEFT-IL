"""领域增量（14 类 taxonomy，boundary-v4）数据集构建工具集。

子命令：
    prepare  按 14 类关键词粗筛候选文档（仅放大候选池，最终以 LLM 标签为准）
    audit    领域标签落地后，统计「每域 × 年代 × WRN」的可用段落，判定哪些域够 900
    build    构建 6 个训练域（D1–D6，train/test/val）与 8 个未见域（U1–U8，test-only）
    verify   校验产物：配额、段落唯一、文件齐全
    report   打印 build_report.json 的「每域 × 年代 × WRN」矩阵

设计口径（与实验约定一致）：
  * 抽样单位 = 段落；train/test/val 之间只要求**段落不重复**，允许同一文档的段落
    分散到不同划分（A3 明确取消文档级互斥）。
  * 年代配额 = 该类总量 × 该年代在全语料中的段落占比（A4，近似即可）。
  * 每个「年代 × 类别」至少 5 条，比例配额不足 5 时补到 5（A6）；供给不足则取全部并记缺口。
  * 单文档贡献上限：R 类 8 段，N/W 类 4 段。
  * 标签取语料 R_1（论文 §5 自动标签），人工 gold 不混入。

用法（项目根目录执行）：
    python 05_incremental_learning_time/scripts/prepare/build_domain_v14.py prepare
    python 05_incremental_learning_time/scripts/prepare/classify_domains.py \
        --input 05_incremental_learning_time/data/domain_v14/candidates.jsonl \
        --output 05_incremental_learning_time/data/domain_v14/title_domain_labels_v14.jsonl
    python 05_incremental_learning_time/scripts/prepare/build_domain_v14.py audit
    python 05_incremental_learning_time/scripts/prepare/build_domain_v14.py build
    python 05_incremental_learning_time/scripts/prepare/build_domain_v14.py verify
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))

import build_incremental_datasets as B  # noqa: E402
from domain_taxonomy_v14 import (  # noqa: E402
    DOMAINS,
    DOMAIN_UNSURE,
    KEYWORDS,
    TRAIN_DOMAINS,
    UNSEEN_DOMAINS,
)
from title_utils import title_key  # noqa: E402

ROOT = B.PROJECT_ROOT
OUT_DIR = ROOT / "05_incremental_learning_time" / "data" / "domain_v14"
CANDIDATES = OUT_DIR / "candidates.jsonl"
LABELS = OUT_DIR / "title_domain_labels_v14.jsonl"
REPORT = OUT_DIR / "build_report.json"
AUDIT = OUT_DIR / "supply_audit.json"

SEED = 42
# 年代名与 build_incremental_datasets.era_of() 保持一致（T1_1949_1978 … T5_2013_2023）
ERAS = [name for name, _a, _b in B.PERIODS]
LABELS3 = ["N", "R", "W"]

N_TRAIN_PER_CLASS = 200
N_TEST_PER_CLASS = 100
N_VAL_PER_CLASS = 40
VAL_FALLBACK_PER_CLASS = 20
ERA_FLOOR = 5                  # 每个 年代×类别 至少 5 条
CAP_R = 8                      # R 类单文档上限
CAP_DEFAULT = 4                # N/W 类单文档上限

# prepare 阶段的候选配额
CAND_PER_DOMAIN = 3000
CAND_R_PER_DOMAIN = 800
CAND_EXTRA_RANDOM = 8000


# ------------------------------------------------------------------ 公共

def era_of(year: int) -> str:
    return B.era_of(year)


def load_labels(path: Path = LABELS) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue
            key = rec.get("title_key") or title_key(rec.get("Title", ""))
            dom = rec.get("domain")
            out[key] = dom if dom in DOMAINS else DOMAIN_UNSURE
    return out


def title_to_domain(df: pd.DataFrame, labels: dict[str, str]) -> pd.Series:
    """把标题标签映射到段落级：先按唯一标题算一次，再向量化映射。"""
    titles = df["Title"].fillna("").astype(str)
    uniq = pd.Index(titles.unique())
    mapping = {t: labels.get(title_key(t), DOMAIN_UNSURE) for t in uniq}
    return titles.map(mapping)


def era_weights(df: pd.DataFrame) -> dict[str, float]:
    counts = Counter(df["era"])
    total = sum(counts.values())
    return {e: counts.get(e, 0) / total for e in ERAS}


def allocate(total: int, weights: dict[str, float], supply: dict[str, int]) -> dict[str, int]:
    """按年代占比分配 total 条，地板 5，供给不足时向其它年代转移。"""
    ideals = {e: total * weights.get(e, 0.0) for e in ERAS}
    alloc = {e: max(ERA_FLOOR, int(round(ideals[e]))) for e in ERAS}
    # 按最大余数法校正到 total
    diff = total - sum(alloc.values())
    order = sorted(ERAS, key=lambda e: ideals[e] - int(ideals[e]), reverse=True)
    guard = 0
    while diff != 0 and guard < 10000:
        guard += 1
        for e in (order if diff > 0 else order[::-1]):
            if diff > 0:
                alloc[e] += 1
                diff -= 1
            elif alloc[e] > ERA_FLOOR:
                alloc[e] -= 1
                diff += 1
            if diff == 0:
                break
    # 供给不足的年代，把缺口转移给仍有存量的年代
    for e in ERAS:
        short = alloc[e] - supply.get(e, 0)
        if short > 0:
            alloc[e] = supply.get(e, 0)
            for donor in sorted(ERAS, key=lambda x: supply.get(x, 0) - alloc[x], reverse=True):
                take = min(short, max(0, supply.get(donor, 0) - alloc[donor]))
                if take:
                    alloc[donor] += take
                    short -= take
                if short <= 0:
                    break
    return alloc


def sample_split(rows, df, quota: int, weights, used_para: set, doc_picks: Counter,
                 rng: random.Random) -> tuple[list[int], dict]:
    """在给定候选行中按「年代比例 + 每类配额」抽样，返回选中行号与实得矩阵。"""
    picks: list[int] = []
    grid = {lab: Counter() for lab in LABELS3}
    taken: set[int] = set()

    for lab in LABELS3:
        lab_rows = [r for r in rows if df["R_1"].iat[r] == lab]
        supply = Counter(df["era"].iat[r] for r in lab_rows)
        alloc = allocate(quota, weights, supply)
        cap = CAP_R if lab == "R" else CAP_DEFAULT
        for era in ERAS:
            want = alloc.get(era, 0)
            bucket = [r for r in lab_rows if df["era"].iat[r] == era and r not in taken]
            rng.shuffle(bucket)
            got = 0
            for r in bucket:
                if got >= want:
                    break
                doc = int(df["_doc_key"].iat[r])
                if r in used_para or doc_picks[(doc, lab)] >= cap:
                    continue
                picks.append(r)
                taken.add(r)
                used_para.add(r)          # 跨划分、跨任务全局去重（段落级）
                doc_picks[(doc, lab)] += 1
                got += 1
            grid[lab][era] = got

        # 兜底：年代配额因「单文档上限」或该层供给枯竭而没凑满时，
        # 从全年代剩余候选里补足（仍受单文档上限约束）。
        filled = sum(grid[lab].values())
        if filled < quota:
            rest = [r for r in lab_rows if r not in taken]
            rng.shuffle(rest)
            for r in rest:
                if filled >= quota:
                    break
                doc = int(df["_doc_key"].iat[r])
                if r in used_para or doc_picks[(doc, lab)] >= cap:
                    continue
                picks.append(r)
                taken.add(r)
                used_para.add(r)
                doc_picks[(doc, lab)] += 1
                grid[lab][df["era"].iat[r]] += 1
                filled += 1
    return picks, grid


# ------------------------------------------------------------------ prepare

def cmd_prepare(args) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = B.load_pool()
    df["era"] = df["year"].map(era_of)
    rng = random.Random(SEED + 11)

    title_rows = df.groupby("Title", sort=False).indices  # Title -> 行号
    docs = []
    for title, positions in title_rows.items():
        positions = list(positions)
        year = int(df["year"].iat[positions[0]])
        labels = Counter(df["R_1"].iat[r] for r in positions)
        docs.append({
            "Title": str(title),
            "DocumentID": str(df["DocumentID"].iat[positions[0]]),
            "year": year,
            "era": era_of(year),
            "n_paragraphs": len(positions),
            "n_R_paras": labels.get("R", 0),
            "hit_domains": [],
        })
    print(f"候选文档（唯一标题）: {len(docs):,}")

    patterns = {d: re.compile("|".join(re.escape(k) for k in KEYWORDS[d])) for d in DOMAINS}
    hits = {d: [] for d in DOMAINS}
    for rec in docs:
        title = rec["Title"]
        for dom in DOMAINS:
            if patterns[dom].search(title):
                hits[dom].append(rec)
                rec["hit_domains"].append(dom)

    chosen: dict[str, dict] = {}
    for dom in DOMAINS:
        pool = hits[dom]
        r_docs = [r for r in pool if r["n_R_paras"] > 0]
        nr_docs = [r for r in pool if r["n_R_paras"] == 0]
        rng.shuffle(r_docs)
        rng.shuffle(nr_docs)
        picked = r_docs[:CAND_R_PER_DOMAIN] + nr_docs[:max(0, CAND_PER_DOMAIN - min(len(r_docs), CAND_R_PER_DOMAIN))]
        for rec in picked:
            chosen.setdefault(rec["Title"], rec)
        print(f"  {dom:<20} 命中 {len(pool):>6,}  取 {len(picked):>6,}")

    rest = [r for r in docs if r["Title"] not in chosen]
    rng.shuffle(rest)
    extra = rest[:CAND_EXTRA_RANDOM]
    for rec in extra:
        chosen.setdefault(rec["Title"], rec)
    print(f"随机补充: {len(extra):,}")

    recs = sorted(chosen.values(), key=lambda r: (r["year"], r["DocumentID"]))
    B.write_jsonl(CANDIDATES, recs)
    print(f"\n已写入 {CANDIDATES}  共 {len(recs):,} 篇")
    print(f"年代分布: {dict(sorted(Counter(r['era'] for r in recs).items()))}")
    print("\n下一步：classify_domains.py --input 该文件 --output data/domain_v14/title_domain_labels_v14.jsonl")
    return 0


# ------------------------------------------------------------------ priority

PRIORITY_TRAIN_DOCS = 1200      # 每个训练域优先分类的文档数
PRIORITY_UNSEEN_DOCS = 400      # 每个未见域优先分类的文档数
PRIORITY_ERA_FLOOR = 40         # 每个年代至少保留的候选文档数


def cmd_priority(args) -> int:
    """生成“先分类这一批”的优先清单，把 token 花在最可能凑齐配额的文档上。

    排序原则：域内先按年代配额分配名额（早年保底 PRIORITY_ERA_FLOOR 篇），
    层内优先取含 R 段落多的文档，再补其余。
    """
    src = OUT_DIR / "candidates.jsonl"
    if not src.is_file():
        print(f"[卡点] 缺少 {src}，请先运行 prepare")
        return 2
    records = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    weights = {}
    total = len(records)
    era_cnt = Counter(r["era"] for r in records)
    weights = {e: era_cnt.get(e, 0) / total for e in ERAS}
    rng = random.Random(SEED + 33)

    plan = [(d, PRIORITY_TRAIN_DOCS) for d in TRAIN_DOMAINS.values()]
    plan += [(d, PRIORITY_UNSEEN_DOCS) for d in UNSEEN_DOMAINS]

    out: list[dict] = []
    for dom, cap in plan:
        pool = [r for r in records if dom in (r.get("hit_domains") or [])]
        rng.shuffle(pool)
        by_era = defaultdict(list)
        for r in pool:
            by_era[r["era"]].append(r)
        for era in ERAS:
            by_era[era].sort(key=lambda r: r["n_R_paras"], reverse=True)

        quota = {e: max(PRIORITY_ERA_FLOOR, int(round(cap * weights.get(e, 0)))) for e in ERAS}
        picked = []
        seen = set()
        for era in ERAS:
            for rec in by_era[era][:quota[era]]:
                if rec["Title"] in seen:
                    continue
                seen.add(rec["Title"])
                picked.append(rec)
        # 名额没用满则在域内补齐
        for era in ERAS:
            if len(picked) >= cap:
                break
            for rec in by_era[era]:
                if len(picked) >= cap:
                    break
                if rec["Title"] in seen:
                    continue
                seen.add(rec["Title"])
                picked.append(rec)
        out.extend(picked[:cap])
        print(f"  {dom:<20} 命中 {len(pool):>6,}  优先取 {min(len(picked), cap):>5,}")

    # 去重（同一标题可能命中多个域）
    uniq, seen = [], set()
    for rec in out:
        if rec["Title"] in seen:
            continue
        seen.add(rec["Title"])
        uniq.append(rec)
    path = OUT_DIR / "candidates_priority.jsonl"
    B.write_jsonl(path, uniq)
    est = len(uniq) * 230 / 1_000_000
    print(f"\n已写入 {path}  共 {len(uniq):,} 篇 "
          f"（按 ~230 token/条估算约 {est:.2f}M token）")
    return 0


# ------------------------------------------------------------------ audit

def _load_corpus() -> pd.DataFrame:
    df = B.load_pool()
    df["era"] = df["year"].map(era_of)
    return df


def cmd_audit(args) -> int:
    labels_path = Path(args.labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_path)
    if not labels:
        print(f"[卡点] 标签文件不存在或为空：{labels_path}")
        print("请先运行 classify_domains.py 生成 14 类标题标签。")
        return 2
    df = _load_corpus()
    df["domain"] = title_to_domain(df, labels)
    known = df[df["domain"].isin(DOMAINS)]
    print(f"标签命中段落: {len(known):,} / {len(df):,}")

    need = {
        "train": N_TRAIN_PER_CLASS,
        "test": N_TEST_PER_CLASS,
        "val": N_VAL_PER_CLASS,
    }
    report = {"need_per_class": need, "era_floor": ERA_FLOOR, "domains": {}}
    print(f"\n{'领域':<20}{'N':>9}{'R':>9}{'W':>9}   {'状态'}")
    print("-" * 74)
    short = []
    for dom in DOMAINS:
        sub = known[known["domain"] == dom]
        cnt = Counter(sub["R_1"])
        need_all = sum(need.values())  # 200+100+40 = 340 per class
        ok = all(cnt.get(lab, 0) >= need_all for lab in LABELS3)
        if dom in TRAIN_DOMAINS.values() and not ok:
            short.append(dom)
        grid = {lab: dict(Counter(sub.loc[sub["R_1"] == lab, "era"])) for lab in LABELS3}
        report["domains"][dom] = {
            "role": "train" if dom in TRAIN_DOMAINS.values() else "unseen",
            "total": cnt,
            "by_class_era": grid,
            "meets_340": ok,
        }
        print(f"{dom:<20}{cnt.get('N',0):>9,}{cnt.get('R',0):>9,}{cnt.get('W',0):>9,}   "
              f"{'OK' if ok else f'缺口（每类需 {need_all}）'}")
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {AUDIT}")
    if short:
        print(f"\n[缺口] 训练域 R/W/N 供给不足: {short}")
        return 3
    print("\n6 个训练域供给均满足 340 段/类（train200+test100+val40）")
    return 0


# ------------------------------------------------------------------ build

def cmd_build(args) -> int:
    out_dir = Path(args.out)
    labels_path = Path(args.labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_path)
    if not labels:
        print(f"[卡点] 标签文件不存在或为空：{labels_path}")
        return 2
    df = _load_corpus()
    df["domain"] = title_to_domain(df, labels)
    weights = era_weights(df)
    print("全语料年代占比:", {k: round(v, 4) for k, v in weights.items()})

    rng = random.Random(SEED + 22)
    used_para: set[int] = set()
    report = {
        "taxonomy": "boundary-v4",
        "seed": SEED,
        "era_weights": weights,
        "era_floor": ERA_FLOOR,
        "caps": {"R": CAP_R, "N": CAP_DEFAULT, "W": CAP_DEFAULT},
        "quotas": {"train": N_TRAIN_PER_CLASS, "test": N_TEST_PER_CLASS, "val": N_VAL_PER_CLASS},
        "tasks": {},
        "unseen": {},
        "shortfalls": [],
    }

    plan = [(f"D{k}", dom, True) for k, dom in TRAIN_DOMAINS.items()]
    plan += [(f"U{i+1}", dom, False) for i, dom in enumerate(UNSEEN_DOMAINS)]

    print(f"\n{'编号':<5}{'领域':<20}{'train':>8}{'test':>7}{'val':>6}   状态")
    print("-" * 78)
    for tag, dom, is_train in plan:
        sub = df[df["domain"] == dom]
        rows = sub.index.tolist()
        doc_picks: Counter = Counter()
        splits = ["test", "val", "train"] if is_train else ["test"]
        result = {}
        for split in splits:
            quota = {"train": N_TRAIN_PER_CLASS, "test": N_TEST_PER_CLASS,
                     "val": N_VAL_PER_CLASS}[split]
            picks, grid = sample_split(rows, df, quota, weights, used_para, doc_picks, rng)
            recs = B.make_records(df, picks, domain=dom)
            for r in recs:
                r["split"] = split
                r["task"] = tag
            safe = dom.replace("/", "_")
            stem = f"{tag}_{safe}"
            B.write_jsonl(out_dir / f"{stem}_{split}.jsonl", recs)
            B.write_manifest(out_dir / f"{stem}_{split}_manifest.csv", recs, tag, split)
            counts = Counter(r["label"] for r in recs)
            result[split] = {"n": len(recs), "per_class": dict(counts),
                             "by_class_era": {k: dict(v) for k, v in grid.items()}}
        target = {"train": 600, "test": 300, "val": 120}
        ok = all(result[s]["n"] == target[s] for s in result)
        detail = "  ".join(f"{s} {result[s]['n']}/{target[s]}" for s in result)
        print(f"{tag:<5}{dom:<20}{result.get('train',{}).get('n','-'):>8}"
              f"{result['test']['n']:>7}{result.get('val',{}).get('n','-'):>6}   "
              f"{'OK' if ok else '不足'}  {detail}")
        if not ok:
            for s, info in result.items():
                for lab in LABELS3:
                    if info["per_class"].get(lab, 0) < {"train": 200, "test": 100, "val": 120}[s] / 3:
                        report["shortfalls"].append(
                            {"task": tag, "domain": dom, "split": s, "label": lab,
                             "got": info["per_class"].get(lab, 0)})
        (report["tasks"] if is_train else report["unseen"])[tag if is_train else dom] = {
            "domain": dom, **result,
        }

    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out_dir}")
    if report["shortfalls"]:
        print(f"\n[缺口] {len(report['shortfalls'])} 项未达配额，详见 {report_path.name}")
        for s in report["shortfalls"][:12]:
            print(f"  - {s['task']} {s['domain']} {s['split']}/{s['label']} 仅 {s['got']}")
        return 3
    print("全部任务达配额")
    return 0


# ------------------------------------------------------------------ verify

def cmd_verify(args) -> int:
    out_dir = Path(args.out)
    problems = []
    seen_para: dict[str, str] = {}
    files = sorted(out_dir.glob("*_*.jsonl"))
    if not files:
        print(f"[卡点] {out_dir} 下没有任何数据集文件")
        return 2
    for path in files:
        recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        counts = Counter(r["label"] for r in recs)
        for r in recs:
            pid = str(r["ParagraphID"])
            if pid in seen_para:
                problems.append(f"段落重复: {pid} 同时出现在 {seen_para[pid]} 与 {path.name}")
                break
            seen_para[pid] = path.name
            if r["label"] not in LABELS3:
                problems.append(f"非法标签 {r['label']} in {path.name}")
        print(f"{path.name:<44} n={len(recs):>4}  N/R/W="
              f"{counts.get('N',0)}/{counts.get('R',0)}/{counts.get('W',0)}")
    print()
    if problems:
        print(f"[失败] {len(problems)} 个问题，示例：")
        for p in problems[:10]:
            print("  -", p)
        return 3
    print(f"[通过] {len(files)} 个文件，段落全局唯一")
    return 0


# ------------------------------------------------------------------ report

def cmd_report(args) -> int:
    report_path = Path(args.out) / "build_report.json"
    if not report_path.is_file():
        print(f"[卡点] 缺少 {report_path}")
        return 2
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    for group in ("tasks", "unseen"):
        for tag, info in rep.get(group, {}).items():
            print(f"\n=== {tag} {info['domain']} ===")
            for split in ("train", "val", "test"):
                if split not in info:
                    continue
                grid = info[split]["by_class_era"]
                eras = sorted({e for g in grid.values() for e in g})
                print(f"  [{split}] n={info[split]['n']}  " + "  ".join(
                    f"{e}:{sum(grid[l].get(e, 0) for l in LABELS3)}" for e in eras))
                for lab in LABELS3:
                    print(f"    {lab}: " + "  ".join(f"{grid[lab].get(e, 0):>4}" for e in eras))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="领域增量（14 类）数据集工具")
    ap.add_argument("cmd", choices=["prepare", "priority", "audit", "build", "verify", "report"])
    ap.add_argument("--labels", default=str(LABELS), help="14 类标题标签文件（audit/build 用）")
    ap.add_argument("--out", default=str(OUT_DIR), help="数据集输出目录（build/verify/report 用）")
    args = ap.parse_args()
    return {
        "prepare": cmd_prepare,
        "priority": cmd_priority,
        "audit": cmd_audit,
        "build": cmd_build,
        "verify": cmd_verify,
        "report": cmd_report,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
