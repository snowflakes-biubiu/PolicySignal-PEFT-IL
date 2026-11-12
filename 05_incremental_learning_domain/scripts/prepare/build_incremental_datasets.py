"""CAPC-CG 增量学习数据集构建。

从全量语料（3,275,474 段）按「时期」和「领域」两个维度构建增量学习任务：

    时间增量 5 个 Task：1949-1978 / 1979-1992 / 1993-2002 / 2003-2012 / 2013-2023
    领域增量 6 个 Task：经济与金融 / 民生与教育 / 自然与生态环境 /
                      科技与工业发展 / 法律与政治监管 / 交通与基础设施

每个 Task：train 200×3 类 (N/R/W)，test 100×3 类。

关键约束（务必保留）：
  1. 标签一律取自语料的 `R_1` 列（= 论文 §5 微调模型回打的自动标签）。
     人工 gold 600/300 不混入任何 Task，只作跨检查点的共享对照集，
     以保证「Task 之间唯一自变量是时期/领域」，不引入标签质量混淆。
    2. train / test 按文件键分组互斥 —— 同一文件的段落不会跨侧出现。
  3. 全局互斥 —— 被任一 Task 用过的文档从池中永久移除，避免
     T1 的测试文档出现在 T3 的训练集里（15/21 次回测协议要求测试集冻结）。
  4. 排除 gold 文本命中的文档，避免与人工 gold 对照集泄漏。

领域增量的构建流程（标题驱动）：
    1. domain-prepare : 对总语料按「标题关键词」粗筛 6 领域候选（R 富集 +
       年代分层），写 domain_classification_input.jsonl
    2. classify_domains.py 对 all_unique_titles.jsonl 做标题分类 → title_domain_labels.jsonl
    3. domain-build   : 按标签分组，每领域内 WRN 严格均分 + T1-T5 年代
       尽量均分地抽段落

用法（项目根目录执行）：
    python 05_incremental_learning_domain/scripts/prepare/build_incremental_datasets.py temporal
    python 05_incremental_learning_domain/scripts/prepare/build_incremental_datasets.py domain-prepare
    python 05_incremental_learning_domain/scripts/prepare/extract_unique_titles.py
    python 05_incremental_learning_domain/scripts/prepare/classify_domains.py
    python 05_incremental_learning_domain/scripts/prepare/build_incremental_datasets.py domain-build \
        --labels 05_incremental_learning_domain/data/domain/title_domain_labels.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from title_utils import title_key

# ---------------------------------------------------------------- 常量

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SNAPSHOT = Path(
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
)
CORPUS_FILE = SNAPSHOT / "data" / "CN_DOCS_FINAL_complete.parquet"
GOLD_TRAIN = SNAPSHOT / "task1_level1" / "train.parquet"
GOLD_VAL = SNAPSHOT / "task1_level1" / "validation.parquet"

DATA_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "data"
TEMPORAL_DIR = DATA_ROOT / "temporal"
DOMAIN_DIR = DATA_ROOT / "domain"

SEED = 42
LABELS = ["N", "R", "W"]
MIN_LEN = 20          # 论文: 排除 <20 字的片段与格式噪声
MAX_LEN = 768         # 与 03/04 阶段训练脚本的 MAX_LENGTH 一致
USER_PREFIX = "待判断政策段落：\n"

PERIODS = [
    ("T1_1949_1978", 1949, 1978),
    ("T2_1979_1992", 1979, 1992),
    ("T3_1993_2002", 1993, 2002),
    ("T4_2003_2012", 2003, 2012),
    ("T5_2013_2023", 2013, 2023),
]

DOMAINS = [
    "经济与金融",
    "民生与教育",
    "自然与生态环境",
    "科技与工业发展",
    "法律与政治监管",
    "交通与基础设施",
]
DOMAIN_UNSURE = "无法归类"

# 标题关键词只用于「候选放大」——错放没关系，最终领域一律以 LLM 标签为准。
# 目的：让稀缺领域（环境/交通）也有足量候选，避免纯随机抽样下凑不齐 R 配额。
DOMAIN_KEYWORDS = {
    "经济与金融": [
        "财政", "税收", "税务", "金融", "银行", "货币", "外汇", "国债",
        "预算", "物价", "价格", "会计", "审计", "海关", "关税", "经济",
        "投资", "国有资产管理", "国资", "企业", "商业", "外贸", "进出口",
        "统计", "发展改革", "体制改革",
    ],
    "民生与教育": [
        "教育", "学校", "教师", "卫生", "医疗", "医院", "药品", "食品",
        "劳动", "就业", "社保", "社会保险", "社会救助", "养老", "民政",
        "低保", "残疾", "人口", "计划生育", "住房", "文化", "体育",
        "旅游", "广播", "电视", "电影", "出版", "文物", "婚姻",
    ],
    "自然与生态环境": [
        "环境", "环保", "生态", "污染", "节能", "减排", "气候", "气象",
        "水利", "水电", "防汛", "抗旱", "森林", "草原", "湿地", "海洋",
        "地质", "矿产", "土地", "自然保护", "核安全", "辐射", "水", "水土",
    ],
    "科技与工业发展": [
        "科技", "科学", "技术", "创新", "研究", "工业", "制造", "信息化",
        "电子", "通信", "标准", "计量", "质量", "专利", "知识产权",
        "航天", "原子能", "能源", "电力", "煤炭", "石油", "化工", "装备",
    ],
    "法律与政治监管": [
        "法", "条例", "司法", "公安", "警察", "检察", "法院", "监察",
        "纪检", "廉政", "组织", "人事", "干部", "编制", "宣传",
        "意识形态", "民族", "宗教", "国家安全", "保密", "档案", "信访",
        "监督", "许可", "登记", "资格",
    ],
    "交通与基础设施": [
        "交通", "公路", "道路", "铁路", "民航", "机场", "港口", "航运",
        "运输", "邮政", "城市建设", "城建", "城乡规划", "建筑", "市政",
        "基础设施", "城镇化", "城市规划", "房屋",
    ],
}

N_TRAIN_PER_CLASS = 200
N_TEST_PER_CLASS = 100
VAL_PER_CLASS = 40         # 验证集每类配额（独立阶段抽取，不改动既有 train/test）
MAX_PARA_PER_DOC = 4      # 单文档最多贡献段落数，防止个别长文档主导某类

# domain-prepare 候选抽样参数
DOMAIN_PER_CAP = 2200     # 每领域候选文档上限（关键词命中后，R 富集优先）
DOMAIN_R_CAP = 700        # 每领域候选中含 R 段文档上限
DOMAIN_EXTRA_RANDOM = 1500  # 未命中任何关键词的随机补充（覆盖关键词表遗漏）


# ---------------------------------------------------------------- 工具


def norm_text(s: str) -> str:
    """归一化：去所有空白。用于去重与 gold 命中判定。"""
    return re.sub(r"\s+", "", str(s))


def load_gold_norm_texts() -> set[str]:
    """读取人工 gold 600/300 的归一化文本集合。"""

    def flat(v):
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, str):
            v = ast.literal_eval(v)
        d = {m["role"]: m["content"] for m in v}
        return d.get("user", "").replace(USER_PREFIX, "").strip()

    out: set[str] = set()
    for f in (GOLD_TRAIN, GOLD_VAL):
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        for v in df["messages"]:
            out.add(norm_text(flat(v)))
    return out


def load_pool(verbose: bool = True) -> pd.DataFrame:
    """加载语料并构建可抽样池：有 R_1 标签 + 年份可解析 + 长度合规 + 去重 + 排除 gold 文档。"""
    if verbose:
        print("读取全量语料 ...", flush=True)
    cols = ["ID", "DocumentID", "Paragraph_content", "IssueDate", "Title",
            "IssuingDepartment", "Category", "Validity", "R_1", "R_2"]
    df = pd.read_parquet(CORPUS_FILE, columns=cols)
    if verbose:
        print(f"  语料行数: {len(df):,}", flush=True)

    df["text"] = df["Paragraph_content"].fillna("").astype(str)
    df["year"] = pd.to_numeric(df["IssueDate"].astype(str).str[:4], errors="coerce")
    file_keys = pd.MultiIndex.from_arrays([
        df["DocumentID"].astype(str),
        df["IssueDate"].fillna("").astype(str),
        df["Title"].fillna("").astype(str),
    ])
    df["_doc_key"], _ = pd.factorize(file_keys, sort=False)

    n0 = len(df)
    df = df[df["R_1"].isin(LABELS)]
    if verbose:
        print(f"  有 L1 标签: {len(df):,}  (滤除 {n0 - len(df):,})", flush=True)

    n1 = len(df)
    df = df[df["year"].between(1949, 2023)]
    if verbose:
        print(f"  年份可解析且在 1949-2023: {len(df):,}  (滤除 {n1 - len(df):,})", flush=True)

    n2 = len(df)
    tl = df["text"].str.len()
    df = df[(tl >= MIN_LEN) & (tl <= MAX_LEN)]
    if verbose:
        print(f"  长度 {MIN_LEN}-{MAX_LEN} 字: {len(df):,}  (滤除 {n2 - len(df):,})", flush=True)

    # 排除 gold 命中的文档（整篇排除）
    gold_norm = load_gold_norm_texts()
    df["_norm"] = df["text"].map(norm_text)
    hit = df["_norm"].isin(gold_norm)
    gold_docs = set(df.loc[hit, "_doc_key"])
    n3 = len(df)
    df = df[~df["_doc_key"].isin(gold_docs)]
    if verbose:
        print(f"  排除 gold 命中文档 {len(gold_docs)} 篇 → 剩余 {len(df):,}  (滤除 {n3 - len(df):,})",
              flush=True)

    if verbose:
        print(f"  抽样池: {len(df):,}  (候选重复文本将在随机抽样时排除)", flush=True)

    df = df.reset_index(drop=True)
    return df


def build_doc_index(df: pd.DataFrame) -> dict:
    """文件键 -> 候选段落行号（最多 MAX_PARA_PER_DOC 条，随机取）。"""
    rng = random.Random(SEED)
    out = {}
    labels = df["R_1"].to_numpy()
    for doc, positions in df.groupby("_doc_key", sort=False).indices.items():
        poss = positions.tolist()
        if len(poss) > MAX_PARA_PER_DOC:
            by_label = defaultdict(list)
            for row in poss:
                by_label[labels[row]].append(row)
            for bucket in by_label.values():
                rng.shuffle(bucket)
            chosen = []
            for label in rng.sample(LABELS, len(LABELS)):
                if by_label[label] and len(chosen) < MAX_PARA_PER_DOC:
                    chosen.append(by_label[label].pop())
            remaining = [row for bucket in by_label.values() for row in bucket]
            rng.shuffle(remaining)
            poss = chosen + remaining[:MAX_PARA_PER_DOC - len(chosen)]
        out[int(doc)] = poss
    return out


# ---------------------------------------------------------------- 互斥状态

USED_DOCS_FILE = DATA_ROOT / "used_documents.json"


def load_used() -> set:
    if USED_DOCS_FILE.exists():
        return set(json.loads(USED_DOCS_FILE.read_text(encoding="utf-8")))
    return set()


def save_used(used: set) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    USED_DOCS_FILE.write_text(
        json.dumps(sorted(int(doc) for doc in used), ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- 采样核心


def collect_docs(docs, doc_index, df, need_per_class, used, rng):
    """按文档顺序贪心收集，直到三类各达到 need_per_class 条。

    返回 (选中的文档列表, 选中文档提供的行号列表)。
    """
    picked_docs, picked_rows = [], []
    cnt = Counter()
    for doc in docs:
        if doc in used:
            continue
        rows = doc_index.get(doc, [])
        if not rows:
            continue
        picked_docs.append(doc)
        picked_rows.extend(rows)
        for r in rows:
            cnt[df["R_1"].iat[r]] += 1
        if all(cnt[c] >= need_per_class for c in LABELS):
            break
    return picked_docs, picked_rows


def sample_balanced(rows, df, n_per_class, rng, per_doc_cap):
    """从给定行号中按类均衡采样，单文档最多 per_doc_cap 条。"""
    rows = list(rows)
    rng.shuffle(rows)
    out = []
    cnt = Counter()
    per_doc = Counter()
    # 先按「每文档 1 条」轮转，保证文档多样性
    for cap in range(1, per_doc_cap + 1):
        for r in rows:
            lab = df["R_1"].iat[r]
            if cnt[lab] >= n_per_class:
                continue
            doc = df["_doc_key"].iat[r]
            if per_doc[doc] >= cap:
                continue
            out.append(r)
            cnt[lab] += 1
            per_doc[doc] += 1
        if all(cnt[c] >= n_per_class for c in LABELS):
            break
    return out, cnt


def sample_year_balanced(rows, df, n_per_class, rng, per_doc_cap,
                         excluded_docs=None, selected_texts=None):
    """按类别逐年轮转随机抽样，候选不足时将配额分给其它年份。"""
    excluded_docs = excluded_docs or set()
    selected_texts = selected_texts if selected_texts is not None else set()
    years = sorted({int(df["year"].iat[row]) for row in rows})
    buckets = defaultdict(list)
    for row in rows:
        buckets[(df["R_1"].iat[row], int(df["year"].iat[row]))].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked = []
    per_doc = Counter()
    counts = Counter()
    by_label_year = {label: Counter() for label in LABELS}
    for label in rng.sample(LABELS, len(LABELS)):
        available = {year: len(buckets[(label, year)]) for year in years}
        quotas = era_quotas(available, n_per_class)

        def take_one(year):
            bucket = buckets[(label, year)]
            while bucket:
                row = bucket.pop()
                doc = df["_doc_key"].iat[row]
                if doc in excluded_docs or per_doc[doc] >= per_doc_cap:
                    continue
                text_key = df["_norm"].iat[row]
                if text_key in selected_texts:
                    continue
                picked.append(row)
                selected_texts.add(text_key)
                counts[label] += 1
                by_label_year[label][year] += 1
                per_doc[doc] += 1
                return True
            return False

        for year in years:
            while by_label_year[label][year] < quotas.get(year, 0):
                if not take_one(year):
                    break

        while counts[label] < n_per_class:
            remaining_years = [year for year in years if buckets[(label, year)]]
            if not remaining_years:
                break
            rng.shuffle(remaining_years)
            remaining_years.sort(key=lambda year: by_label_year[label][year])
            if not any(take_one(year) for year in remaining_years):
                break
    return picked, counts, by_label_year


def era_quotas(avail_by_era: dict, n_total: int) -> dict:
    """把 n_total 个配额按年代桶尽量均匀分配。

    轮转 +1：每桶每轮至多 +1，桶容量不足时自动把剩余配额摊给还有余量的桶
    （即「时间线上尽量均分」，稀缺年代有多少用多少）。
    """
    q = {e: 0 for e in avail_by_era}
    rest = n_total
    keys = sorted(avail_by_era)
    while rest > 0:
        progressed = False
        for e in keys:
            if q[e] < avail_by_era[e] and rest > 0:
                q[e] += 1
                rest -= 1
                progressed = True
        if not progressed:
            break
    return q


def sample_balanced_by_era(rows, df, n_per_class, rng, per_doc_cap):
    """WRN 严格均分 + T1-T5 年代尽量均分的段落抽样。

    对每个类别先用 era_quotas 在五个年代桶间分配 n_per_class，
    再逐桶选段（单文档上限 per_doc_cap 条）。
    返回 (选中行号, {label: {era: 实际数}}, 总计数字典)。
    """
    rows = list(rows)
    rng.shuffle(rows)
    by_le = defaultdict(list)          # (label, era) -> [row]
    for r in rows:
        by_le[(df["R_1"].iat[r],
               era_of(int(df["year"].iat[r])))].append(r)

    out = []
    picked = set()
    per_doc = Counter()
    quota_grid = {}                    # label -> {era: 实际选中数}
    for lab in LABELS:
        avail = {e: len(by_le.get((lab, e), [])) for e in
                 sorted({e for (l, e) in by_le if l == lab})}
        quotas = era_quotas(avail, n_per_class) if avail else {}
        quota_grid[lab] = {e: 0 for e in quotas}
        for e, need in quotas.items():
            for r in by_le[(lab, e)]:
                if quota_grid[lab][e] >= need:
                    break
                doc = df["DocumentID"].iat[r]
                if per_doc[doc] >= per_doc_cap:
                    continue
                out.append(r)
                picked.add(r)
                per_doc[doc] += 1
                quota_grid[lab][e] += 1

    # 第二轮：把因 per-doc cap / 桶内不足造成的缺额摊给还有余量的桶
    # （WRN 严格均分优先于时间均分；时间上已尽量均匀）
    for lab in LABELS:
        deficit = n_per_class - sum(quota_grid[lab].values())
        if deficit <= 0:
            continue
        for r in rows:
            if deficit <= 0:
                break
            if df["R_1"].iat[r] != lab or r in picked:
                continue
            doc = df["_doc_key"].iat[r]
            if per_doc[doc] >= per_doc_cap:
                continue
            out.append(r)
            picked.add(r)
            per_doc[doc] += 1
            deficit -= 1
            e = era_of(int(df["year"].iat[r]))
            quota_grid[lab][e] = quota_grid[lab].get(e, 0) + 1

    cnt = {lab: sum(quota_grid[lab].values()) for lab in LABELS}
    return out, quota_grid, cnt


def make_records(df, rows, *, period=None, domain=None):
    recs = []
    for r in rows:
        year = int(df["year"].iat[r])
        recs.append({
            "text": df["text"].iat[r],
            "label": df["R_1"].iat[r],
            "label_source": "auto_R1",
            "level2_auto": df["R_2"].iat[r],
            "DocumentID": str(df["DocumentID"].iat[r]),
                "DocumentKey": [
                    str(df["DocumentID"].iat[r]),
                    str(df["IssueDate"].iat[r]),
                    str(df["Title"].iat[r]),
                ],
            "ParagraphID": str(df["ID"].iat[r]),
            "year": year,
            "era": era_of(year),
            "Title": str(df["Title"].iat[r]),
            "IssuingDepartment": str(df["IssuingDepartment"].iat[r]),
            "Category": str(df["Category"].iat[r]),
            "Validity": str(df["Validity"].iat[r]),
            "period": period,
            "domain": domain,
        })
    return recs


def write_jsonl(path: Path, recs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_manifest(path: Path, recs, task_name, split):
    rows = []
    for r in recs:
        rows.append({
            "task": task_name, "split": split, "label": r["label"],
            "DocumentID": r["DocumentID"], "year": r["year"],
            "DocumentKey": json.dumps(r["DocumentKey"], ensure_ascii=False),
            "domain": r["domain"], "text_len": len(r["text"]),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 阶段：时间增量


def stage_temporal(df, doc_index, doc_meta, used):
    print("\n" + "=" * 74)
    print("构建时间增量数据集（5 个 Task）")
    print("=" * 74)

    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    report = {"tasks": {}, "params": {
        "seed": SEED, "min_len": MIN_LEN, "max_len": MAX_LEN,
        "n_train_per_class": N_TRAIN_PER_CLASS, "n_test_per_class": N_TEST_PER_CLASS,
        "max_para_per_doc": MAX_PARA_PER_DOC,
    }}
    used = set()
    selected_texts = set()
    print(f"{'时期':<16}{'候选文档':>10}{'候选段落':>10}   抽样结果")
    print("-" * 74)

    for name, a, b in PERIODS:
        eligible_docs = [doc for doc, year in doc_meta.items()
                         if a <= year <= b and doc not in used]
        docs = set(eligible_docs)
        split_docs = {"train": set(), "test": set()}
        docs_by_stratum = defaultdict(list)
        for doc in eligible_docs:
            signature = tuple(sorted({df["R_1"].iat[row] for row in doc_index[doc]}))
            docs_by_stratum[(doc_meta[doc], signature)].append(doc)
        for stratum_docs in docs_by_stratum.values():
            rng.shuffle(stratum_docs)
            n_test_docs = round(len(stratum_docs) / 3)
            if len(stratum_docs) > 1:
                n_test_docs = max(1, min(len(stratum_docs) - 1, n_test_docs))
            split_docs["test"].update(stratum_docs[:n_test_docs])
            split_docs["train"].update(stratum_docs[n_test_docs:])
        rows_by_split = {
            split: [row for doc in doc_set for row in doc_index[doc]]
            for split, doc_set in split_docs.items()
        }
        cand = sum(map(len, rows_by_split.values()))
        candidate_by_split = {split: {label: Counter() for label in LABELS}
                              for split in rows_by_split}
        for split, rows in rows_by_split.items():
            for row in rows:
                candidate_by_split[split][df["R_1"].iat[row]][int(df["year"].iat[row])] += 1

        test_pick, cnt_t, years_t = sample_year_balanced(
            rows_by_split["test"], df, N_TEST_PER_CLASS, rng, per_doc_cap=2,
            selected_texts=selected_texts)
        test_docs = {df["_doc_key"].iat[row] for row in test_pick}
        train_pick, cnt_r, years_r = sample_year_balanced(
            rows_by_split["train"], df, N_TRAIN_PER_CLASS, rng, per_doc_cap=3,
            selected_texts=selected_texts)
        used |= test_docs | {df["_doc_key"].iat[row] for row in train_pick}

        recs_te = make_records(df, test_pick, period=name)
        recs_tr = make_records(df, train_pick, period=name)
        for r in recs_te:
            r["split"] = "test"
        for r in recs_tr:
            r["split"] = "train"

        write_jsonl(TEMPORAL_DIR / f"{name}_train.jsonl", recs_tr)
        write_jsonl(TEMPORAL_DIR / f"{name}_test.jsonl", recs_te)
        write_manifest(TEMPORAL_DIR / f"{name}_train_manifest.csv", recs_tr, name, "train")
        write_manifest(TEMPORAL_DIR / f"{name}_test_manifest.csv", recs_te, name, "test")

        ok = len(recs_tr) == 600 and len(recs_te) == 300
        print(f"{name:<16}{len(docs):>10}{cand:>10}   "
              f"train {len(recs_tr)}/600  test {len(recs_te)}/300  "
              f"{'OK' if ok else '不足!'}")

        if not ok:
            raise RuntimeError(f"{name} 类别配额不足: train={dict(cnt_r)}, test={dict(cnt_t)}")
        report["tasks"][name] = {
            "year_range": [a, b],
            "candidate_docs": len(docs),
            "candidate_paragraphs": cand,
            "train": {"n": len(recs_tr),
                      "per_class": {c: cnt_r[c] for c in LABELS},
                      "docs": len({tuple(r["DocumentKey"]) for r in recs_tr}),
                      "candidate_per_class_year": {c: dict(sorted(candidate_by_split["train"][c].items())) for c in LABELS},
                      "per_class_year": {c: dict(sorted(years_r[c].items())) for c in LABELS}},
            "test": {"n": len(recs_te),
                     "per_class": {c: cnt_t[c] for c in LABELS},
                     "docs": len({tuple(r["DocumentKey"]) for r in recs_te}),
                     "candidate_per_class_year": {c: dict(sorted(candidate_by_split["test"][c].items())) for c in LABELS},
                     "per_class_year": {c: dict(sorted(years_t[c].items())) for c in LABELS}},
            "mean_text_len": round(sum(len(r["text"]) for r in recs_tr + recs_te)
                                   / max(1, len(recs_tr) + len(recs_te)), 1),
        }

    save_used(used)
    (TEMPORAL_DIR / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {TEMPORAL_DIR}")
    print(f"全局已用文档累计: {len(used):,}")


def read_jsonl(path: Path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def stage_temporal_val(df, doc_index, doc_meta, used):
    """为每个时间窗口单独划分验证集（每类 VAL_PER_CLASS 条）。

    约束（与《修改建议》一致）：
      - **不改动既有 train/test**：抽样池排除所有已用文档与已用文本；
      - 验证集与全部窗口的 train/test 文档互斥，可用于 λ / 回放比例 / adapter 权重调参；
      - 调参只允许使用「当前及过去窗口」的验证数据，因此按窗口独立落盘；
      - 验证集不参与任何最终测试指标。
    """
    print("\n" + "=" * 74)
    print("构建时间增量验证集（5 个 Task × 每类 %d 条）" % VAL_PER_CLASS)
    print("=" * 74)

    rng = random.Random(SEED + 777)
    selected_texts: set[str] = set()
    for name, _, _ in PERIODS:
        for split in ("train", "test"):
            path = TEMPORAL_DIR / f"{name}_{split}.jsonl"
            if not path.is_file():
                continue
            for record in read_jsonl(path):
                selected_texts.add(norm_text(record["text"]))
    print(f"已用文本集合（train/test）: {len(selected_texts):,} 条")

    report = {"params": {"seed": SEED + 777, "n_val_per_class": VAL_PER_CLASS,
                          "per_doc_cap": 2}, "tasks": {}}
    for name, a, b in PERIODS:
        eligible_docs = [doc for doc, year in doc_meta.items()
                         if a <= year <= b and doc not in used]
        docs = set(eligible_docs)
        rows = [row for doc in docs for row in doc_index[doc]]
        pick, cnt, years = sample_year_balanced(
            rows, df, VAL_PER_CLASS, rng, per_doc_cap=2,
            excluded_docs=used, selected_texts=selected_texts)
        if len(pick) != VAL_PER_CLASS * len(LABELS):
            raise RuntimeError(
                f"{name} 验证集配额不足: {len(pick)}/{VAL_PER_CLASS * len(LABELS)}"
                f"（{dict(cnt)}）")
        recs = make_records(df, pick, period=name)
        for record in recs:
            record["split"] = "val"
        write_jsonl(TEMPORAL_DIR / f"{name}_val.jsonl", recs)
        write_manifest(TEMPORAL_DIR / f"{name}_val_manifest.csv", recs, name, "val")
        used |= {df["_doc_key"].iat[row] for row in pick}
        report["tasks"][name] = {
            "year_range": [a, b],
            "candidate_docs": len(docs),
            "val": {"n": len(recs),
                    "per_class": {c: cnt[c] for c in LABELS},
                    "docs": len({tuple(r["DocumentKey"]) for r in recs}),
                    "per_class_year": {c: dict(sorted(years[c].items())) for c in LABELS}},
        }
        print(f"{name}: val {len(recs)} 条  {dict(cnt)}")

    save_used(used)
    (TEMPORAL_DIR / "build_report_val.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {TEMPORAL_DIR}/*_val.jsonl")
    print(f"全局已用文档累计: {len(used):,}")


# ---------------------------------------------------------------- 阶段：领域分类输入


def era_of(year: int) -> str:
    for name, a, b in PERIODS:
        if a <= year <= b:
            return name
    return "T?"


def spread_by_era(docs, doc_meta, n, rng):
    """按 T1-T5 五个年代桶轮转抽样，尽量均匀覆盖时间线。"""
    by_era = defaultdict(list)
    for d in docs:
        by_era[era_of(doc_meta[d])].append(d)
    for lst in by_era.values():
        rng.shuffle(lst)
    out = []
    eras = sorted(by_era)
    while len(out) < n and any(by_era.values()):
        for e in eras:
            if by_era[e] and len(out) < n:
                out.append(by_era[e].pop())
    return out


def stage_domain_prepare(df, doc_index, doc_meta, used):
    print("\n" + "=" * 74)
    print("准备领域分类输入（标题关键词粗筛 + R 富集 + 年代分层）")
    print("=" * 74)

    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + 1)

    # ---- 每文档全量 R 段计数（不受 MAX_PARA_PER_DOC 截断影响）
    doc_nR = df.loc[df["R_1"] == "R", "_doc_key"].value_counts()

    avail = [d for d in doc_meta if d not in used]
    print(f"可用文档池: {len(avail):,} 篇")

    # ---- 标题关键词命中
    title_by_doc = {}
    dept_by_doc = {}
    for d in avail:
        rows = doc_index.get(d, [])
        if not rows:
            continue
        r0 = rows[0]
        title_by_doc[d] = str(df["Title"].iat[r0])
        dept_by_doc[d] = str(df["IssuingDepartment"].iat[r0])

    hits = {dom: [] for dom in DOMAINS}
    hit_any = set()
    for d, t in title_by_doc.items():
        for dom, kws in DOMAIN_KEYWORDS.items():
            if any(k in t for k in kws):
                hits[dom].append(d)
                hit_any.add(d)
    print("标题关键词命中分布：")
    for dom in DOMAINS:
        print(f"  {dom}: {len(hits[dom]):,}")

    # ---- 每领域候选：含 R 文档优先 + 年代均匀
    chosen = {}   # doc -> set(domain) 命中领域（供参考，LLM 才是最终标签）
    for dom in DOMAINS:
        docs = hits[dom]
        r_docs = [d for d in docs if doc_nR.get(d, 0) > 0]
        nr_docs = [d for d in docs if doc_nR.get(d, 0) == 0]
        pick_r = spread_by_era(r_docs, doc_meta, DOMAIN_R_CAP, rng)
        pick_nr = spread_by_era(nr_docs, doc_meta,
                                max(0, DOMAIN_PER_CAP - len(pick_r)), rng)
        for d in pick_r + pick_nr:
            chosen.setdefault(d, set()).add(dom)

    # ---- 未命中关键词的随机补充（覆盖关键词表遗漏的领域）
    rest = [d for d in title_by_doc if d not in chosen]
    rest_r = [d for d in rest if doc_nR.get(d, 0) > 0]
    rest_nr = [d for d in rest if doc_nR.get(d, 0) == 0]
    extra = spread_by_era(rest_r, doc_meta, DOMAIN_R_CAP // 3, rng) + \
        spread_by_era(rest_nr, doc_meta,
                      DOMAIN_EXTRA_RANDOM - DOMAIN_R_CAP // 3, rng)
    for d in extra:
        chosen.setdefault(d, set())

    # ---- 写分类输入
    recs = []
    for d in sorted(chosen):
        rows = doc_index.get(d, [])
        if not rows:
            continue
        r0 = rows[0]
        recs.append({
            "DocumentID": str(df["DocumentID"].iat[r0]),
            "DocumentKey": [str(df["DocumentID"].iat[r0]),
                            str(df["IssueDate"].iat[r0]),
                            str(df["Title"].iat[r0])],
            "Title": title_by_doc[d],
            "IssuingDepartment": dept_by_doc[d],
            "year": int(df["year"].iat[r0]),
            "era": era_of(doc_meta[d]),
            "n_R_paras": int(doc_nR.get(d, 0)),
            "n_paragraphs": len(rows),
            "hit_domains": sorted(chosen[d]),
        })
    recs.sort(key=lambda r: (r["DocumentID"]))

    out = DOMAIN_DIR / "domain_classification_input.jsonl"
    write_jsonl(out, recs)
    n_r = sum(1 for r in recs if r["n_R_paras"] > 0)
    print(f"\n已写入 {out}")
    print(f"  候选文档: {len(recs):,} 篇（含 R 段文档 {n_r:,} 篇）")
    print(f"  覆盖候选段落: {sum(r['n_paragraphs'] for r in recs):,}")
    eras = Counter(r["era"] for r in recs)
    print(f"  年代分布: {dict(sorted(eras.items()))}")
    print("\n下一步：运行 classify_domains.py 用 LLM 按标题打领域标签 → domain_labels.jsonl")


# ---------------------------------------------------------------- 阶段：构建领域任务


def stage_domain_build(df, doc_index, doc_meta, used, labels_file):
    print("\n" + "=" * 74)
    print("构建领域增量数据集（6 个 Task）")
    print("=" * 74)

    title_labels = {}
    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            key = o.get("title_key")
            if key:
                title_labels[key] = o.get("domain")
    lab = {}
    for doc, rows in doc_index.items():
        key = title_key(df["Title"].iat[rows[0]])
        if key in title_labels:
            lab[doc] = title_labels[key]
    print(f"读入领域标签 {len(lab):,} 条")
    dist = Counter(lab.values())
    for k, v in dist.most_common():
        print(f"  {k}: {v}")

    rng = random.Random(SEED + 2)
    report = {"tasks": {}, "domain_doc_counts": dict(dist)}
    print(f"\n{'领域':<16}{'候选文档':>10}{'候选段落':>10}   抽样结果")
    print("-" * 74)

    for dom in DOMAINS:
        docs = [d for d in lab if lab[d] == dom and d in doc_meta and d not in used]
        rng.shuffle(docs)
        cand = sum(len(doc_index.get(d, [])) for d in docs)

        test_docs, test_rows = collect_docs(
            docs, doc_index, df, N_TEST_PER_CLASS * 6 // 5, used, rng)
        used |= set(test_docs)
        train_docs, train_rows = collect_docs(
            docs, doc_index, df, N_TRAIN_PER_CLASS * 6 // 5, used, rng)
        used |= set(train_docs)

        # WRN 严格均分 + T1-T5 年代尽量均分
        test_pick, grid_t, cnt_t = sample_balanced_by_era(
            test_rows, df, N_TEST_PER_CLASS, rng, per_doc_cap=2)
        train_pick, grid_r, cnt_r = sample_balanced_by_era(
            train_rows, df, N_TRAIN_PER_CLASS, rng, per_doc_cap=3)

        recs_te = make_records(df, test_pick, domain=dom)
        recs_tr = make_records(df, train_pick, domain=dom)
        for r in recs_te:
            r["split"] = "test"
        for r in recs_tr:
            r["split"] = "train"

        safe = dom.replace("/", "_")
        write_jsonl(DOMAIN_DIR / f"{safe}_train.jsonl", recs_tr)
        write_jsonl(DOMAIN_DIR / f"{safe}_test.jsonl", recs_te)
        write_manifest(DOMAIN_DIR / f"{safe}_train_manifest.csv", recs_tr, dom, "train")
        write_manifest(DOMAIN_DIR / f"{safe}_test_manifest.csv", recs_te, dom, "test")

        ok = len(recs_tr) == 600 and len(recs_te) == 300
        print(f"{dom:<16}{len(docs):>10}{cand:>10}   "
              f"train {len(recs_tr)}/600  test {len(recs_te)}/300  "
              f"{'OK' if ok else '不足!'}"
              + ("" if ok else f"  缺: {[c for c in LABELS if cnt_r[c] < N_TRAIN_PER_CLASS]}"))

        report["tasks"][dom] = {
            "train": {"n": len(recs_tr),
                      "per_class": cnt_r,
                      "by_class_era": grid_r},
            "test": {"n": len(recs_te),
                     "per_class": cnt_t,
                     "by_class_era": grid_t},
        }

    save_used(used)
    (DOMAIN_DIR / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {DOMAIN_DIR}")
    print(f"全局已用文档累计: {len(used):,}")


# ---------------------------------------------------------------- 入口


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["temporal", "temporal-val", "domain-prepare", "domain-build"])
    ap.add_argument("--labels", help="domain-build 所需的领域标签 jsonl")
    args = ap.parse_args()

    if args.stage == "domain-build" and not args.labels:
        ap.error("domain-build 需要 --labels")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_pool()
    doc_index = build_doc_index(df)
    doc_meta = {d: int(df["year"].iat[rows[0]])
                for d, rows in doc_index.items()}
    print(f"\n可抽样池: {len(df):,} 段 / {len(doc_index):,} 篇文档")

    used = set() if args.stage == "temporal" else load_used()
    print(f"已占用文档: {len(used):,}")

    if args.stage == "temporal":
        stage_temporal(df, doc_index, doc_meta, used)
    elif args.stage == "temporal-val":
        stage_temporal_val(df, doc_index, doc_meta, used)
    elif args.stage == "domain-prepare":
        stage_domain_prepare(df, doc_index, doc_meta, used)
    else:
        stage_domain_build(df, doc_index, doc_meta, used, args.labels)


if __name__ == "__main__":
    sys.exit(main())
