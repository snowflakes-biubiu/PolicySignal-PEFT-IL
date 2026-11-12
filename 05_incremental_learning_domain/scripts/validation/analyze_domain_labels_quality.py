"""title_domain_labels.jsonl 质量体检：标签质量 + 训练可行性 + 年份均衡。

输出统计到 stdout，供撰写质量报告使用。
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "05_incremental_learning_domain" / "scripts" / "prepare"))
from title_utils import title_key, normalize_title  # noqa: E402

DOMAIN_DIR = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "data" / "domain"
LABELS = DOMAIN_DIR / "title_domain_labels.jsonl"
USED = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "data" / "used_documents.json"
SNAPSHOT = Path(
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b")
CORPUS = SNAPSHOT / "data" / "CN_DOCS_FINAL_complete.parquet"

DOMAINS = ["经济与金融", "民生与教育", "自然与生态环境",
           "科技与工业发展", "法律与政治监管", "交通与基础设施"]
NEED = {"train": 200, "test": 100}   # 每类
MIN_LEN, MAX_LEN = 20, 768


def era_of(y):
    for name, a, b in [("T1_1949_1978", 1949, 1978), ("T2_1979_1992", 1979, 1992),
                       ("T3_1993_2002", 1993, 2002), ("T4_2003_2012", 2003, 2012),
                       ("T5_2013_2023", 2013, 2023)]:
        if a <= y <= b:
            return name
    return "T?"


def main():
    rng = random.Random(42)

    # ---------- 1. 标签文件统计
    lab_rows = [json.loads(l) for l in
                LABELS.read_text(encoding="utf-8").splitlines() if l.strip()]
    lab = {r["title_key"]: r for r in lab_rows}
    print(f"[快照] 标签行数: {len(lab_rows):,}  唯一 title_key: {len(lab):,}")
    print(f"  标签覆盖唯一标题: {len(lab):,} / 336,973 = {len(lab) / 336973 * 100:.1f}%")

    dist = Counter(r["domain"] for r in lab.values())
    print("\n[领域分布]")
    for k, v in dist.most_common():
        print(f"  {k:<10} {v:>6}  ({v / len(lab) * 100:.1f}%)")

    err = [r for r in lab.values() if r.get("error")]
    low_conf = [r for r in lab.values() if float(r.get("confidence", 0)) < 0.6]
    print(f"\n  带 error 的标签: {len(err)}")
    print(f"  confidence < 0.6: {len(low_conf)}")
    print(f"  manual_review=True: {sum(1 for r in lab.values() if r.get('manual_review'))}")
    print(f"  suggested != domain: {sum(1 for r in lab.values() if r.get('suggested_domain') and r['suggested_domain'] != r['domain'])}")

    # ---------- 2. 语料池
    print("\n[加载语料池 ...]", flush=True)
    cols = ["ID", "DocumentID", "Paragraph_content", "IssueDate", "Title", "R_1"]
    df = pd.read_parquet(CORPUS, columns=cols)
    df["text"] = df["Paragraph_content"].fillna("").astype(str)
    df["year"] = pd.to_numeric(df["IssueDate"].astype(str).str[:4], errors="coerce")
    tl = df["text"].str.len()
    df = df[df["R_1"].isin(["N", "R", "W"]) & df["year"].between(1949, 2023)
            & (tl >= MIN_LEN) & (tl <= MAX_LEN)]
    used = {str(x) for x in json.loads(USED.read_text(encoding="utf-8"))}
    print(f"  池段落 {len(df):,}  已被 temporal 占用文档 {len(used):,}")

    # 文档级：标题 / 年份 / 类别计数
    g = df.groupby("DocumentID")
    docs = pd.DataFrame({
        "title": g["Title"].first(),
        "year": g["year"].first().astype(int),
        "n": g.size(),
        "nN": g["R_1"].apply(lambda s: (s == "N").sum()),
        "nR": g["R_1"].apply(lambda s: (s == "R").sum()),
        "nW": g["R_1"].apply(lambda s: (s == "W").sum()),
    })
    docs["title_key"] = docs["title"].map(title_key)
    docs["used"] = docs.index.isin(used)
    docs["label"] = docs["title_key"].map(lambda k: lab.get(k, {}).get("domain"))
    docs["era"] = docs["year"].map(era_of)
    print(f"  池文档 {len(docs):,}  其中有领域标签 {docs['label'].notna().sum():,}"
          f"  未被占用且有标签 {((~docs['used']) & docs['label'].notna()).sum():,}")

    # ---------- 3. 每领域训练可行性（排除 temporal 已用文档）
    print("\n[训练可行性]  需求: 每领域 train 200×3 + test 100×3 = 900 段"
          "（单文档最多 4 段, train/test 文档互斥）")
    print(f"{'领域':<10}{'文档':>7}{'N段':>7}{'R段':>7}{'W段':>7}"
          f"{'R够300?':>9}{'N够300?':>9}{'W够300?':>9}")
    feas = {}
    for dom in DOMAINS:
        sub = docs[(docs["label"] == dom) & (~docs["used"])]
        nN, nR, nW = int(sub["nN"].sum()), int(sub["nR"].sum()), int(sub["nW"].sum())
        # 考虑单文档 4 段上限的粗略折减（长文档只取 4 段）
        cap = 4
        nN_c = int(sum(min(v, cap) if False else v for v in sub["nN"]))  # 上限影响按段粗估
        ok = lambda x: "OK" if x >= 300 else f"缺{300 - x}"
        print(f"{dom:<10}{len(sub):>7}{nN:>7}{nR:>7}{nW:>7}"
              f"{ok(nR):>9}{ok(nN):>9}{ok(nW):>9}")
        feas[dom] = {"docs": len(sub), "N": nN, "R": nR, "W": nW}

    # 按文档 4 段上限重估（抽样时每文档最多贡献 4 段）
    print("\n  [按单文档 4 段上限折减后]")
    print(f"{'领域':<10}{'N段':>7}{'R段':>7}{'W段':>7}{'R够300?':>9}")
    for dom in DOMAINS:
        sub = docs[(docs["label"] == dom) & (~docs["used"])]
        nN = int(sub["nN"].clip(upper=4).sum())  # 不精确：4段上限作用于总段数不是每类
        nR = int(sub["nR"].clip(upper=4).sum())
        nW = int(sub["nW"].clip(upper=4).sum())
        print(f"{dom:<10}{nN:>7}{nR:>7}{nW:>7}{('OK' if nR >= 300 else f'缺{300 - nR}'):>9}")

    # 未标注但含 R 段的文档 —— 优先补标候选
    unl = docs[(docs["label"].isna()) & (~docs["used"])]
    unl_r = unl[unl["nR"] > 0]
    print(f"\n  [优先补标候选]  未标注且未占用文档 {len(unl):,}，其中含 R 段的 "
          f"{len(unl_r):,} 篇（贡献 R 段约 {int(unl_r['nR'].clip(upper=4).sum()):,}）")

    # ---------- 4. 年份分布（era）
    print("\n[年份分布]  有标签且未占用文档，era × 领域")
    eras = ["T1_1949_1978", "T2_1979_1992", "T3_1993_2002",
            "T4_2003_2012", "T5_2013_2023"]
    sub_all = docs[(docs["label"].notna()) & (~docs["used"])]
    ct = pd.crosstab(sub_all["label"], sub_all["era"])
    print(f"{'领域':<10}" + "".join(f"{e:>16}" for e in eras))
    for dom in DOMAINS:
        row = [str(ct.loc[dom, e]) if (dom in ct.index and e in ct.columns) else "0"
               for e in eras]
        print(f"{dom:<10}" + "".join(f"{v:>16}" for v in row))

    # ---------- 5. 抽检样例（每领域 3 条：高置信 + 低置信混合）
    print("\n" + "=" * 100)
    print("[人工抽检样例]  （Title / 标注领域 / 置信度 / 语料正文片段）")
    print("=" * 100)
    text_by_doc = df.groupby("DocumentID")["text"].first()
    for dom in DOMAINS:
        sub = docs[(docs["label"] == dom) & (~docs["used"])]
        hi = sub[sub["title_key"].map(lambda k: float(lab[k]["confidence"])) >= 0.9]
        lo = sub[sub["title_key"].map(lambda k: float(lab[k]["confidence"])) < 0.8]
        picks = []
        if len(hi):
            picks += list(hi.sample(min(2, len(hi)), random_state=42).index)
        if len(lo):
            picks += list(lo.sample(min(1, len(lo)), random_state=42).index)
        if not picks:
            picks = list(sub.sample(min(3, len(sub)), random_state=42).index)
        print(f"\n--- {dom} ---")
        for docid in picks:
            r = docs.loc[docid]
            lb = lab[r["title_key"]]
            para = str(text_by_doc.get(docid, ""))[:120].replace("\n", " ")
            print(f"  [{lb['confidence']}] {r['title'][:70]}")
            print(f"      正文: {para}")

    # ---------- 6. 与关键词先验的一致性
    cin = DOMAIN_DIR / "domain_classification_input.jsonl"
    if cin.exists():
        print("\n[一致性]  与关键词粗筛 hit_domains 的对照（仅对双方都有的文档）")
        agree = total = 0
        confusion = Counter()
        for line in cin.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            k = title_key(o["Title"])
            if k not in lab or not o.get("hit_domains"):
                continue
            total += 1
            d = lab[k]["domain"]
            if d in o["hit_domains"]:
                agree += 1
            else:
                confusion[(o["hit_domains"][0], d)] += 1
        print(f"  对照文档: {total}  一致(命中集合内): {agree} ({agree / max(total, 1) * 100:.1f}%)")
        print(f"  最多分歧组合 top8: {confusion.most_common(8)}")


if __name__ == "__main__":
    main()
