"""领域增量数据集契约审计（只读，不加载模型、不占 GPU、不调 API）。

对应修复方案 P1-1：把"代码无法自证"的契约变成可执行断言。

检查项（6 项）：
  1. 每 (域 × split) 的条数与标签分布：train 600（N/R/W 各 200）、test 300（各 100）
  2. 弱键（去空白）全局唯一
  3. 强键（NFKC + casefold + 去非文字字符）全局唯一 —— 比现有实现更严
  4. record["domain"] 与 TASKS[task]["domain"] 一致
  5. 每域 T1~T5 年代覆盖（--strict-era 时缺失即失败，默认仅告警）
  6. 文档跨 (任务/划分) 重叠比例（A3 允许，仅留痕）

可选 --truncation：用本地 tokenizer 统计 prompt 长度分布与截断率（对应 P1-2）。

用法：
    python 05_incremental_learning_domain/scripts/tests/audit_dataset_contract.py
    python 05_incremental_learning_domain/scripts/tests/audit_dataset_contract.py --truncation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from incremental_common import (  # noqa: E402
    DATA_ROOT,
    EVAL_MAX_LENGTH,
    MAX_LENGTH,
    TASKS,
    read_jsonl,
    task_file,
)

ERA_ALL = ("T1_1949_1978", "T2_1979_1992", "T3_1993_2002",
           "T4_2003_2012", "T5_2013_2023")
EXPECT_PER_CLASS = {"train": 200, "test": 100}

PROBLEMS: list[str] = []


def fail(msg: str) -> None:
    PROBLEMS.append(msg)


def warn(msg: str) -> None:
    print(f"  [警告] {msg}")


def text_key_weak(text: str) -> str:
    """现有实现口径：只去空白。"""
    return hashlib.sha256("".join(str(text).split()).encode("utf-8")).hexdigest()[:20]


def text_key_strong(text: str) -> str:
    """更严口径：NFKC + casefold + 去掉所有非文字/非汉字字符。"""
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]


def audit_rows() -> tuple[dict[str, list[dict]], dict]:
    print("=" * 92)
    print("[1] 每 (域 × split) 条数 / 标签 / 域字段 / 年代覆盖")
    print("=" * 92)
    data: dict[str, list[dict]] = {}
    for task, meta in TASKS.items():
        expect_domain = meta.get("domain")
        for split, per_class in EXPECT_PER_CLASS.items():
            path = task_file(task, split)
            if not path.is_file():
                fail(f"缺少数据文件: {path}")
                continue
            rows = read_jsonl(path)
            data[path.name] = rows
            labels = Counter(str(r.get("label")) for r in rows)
            eras = Counter(str(r.get("era")) for r in rows)
            bad_domain = sum(1 for r in rows
                             if expect_domain and str(r.get("domain", "")).strip() != expect_domain)
            missing_era = [e for e in ERA_ALL if not eras.get(e)]
            ok = (len(rows) == per_class * len(("N", "R", "W"))
                  and all(labels.get(lab) == per_class for lab in ("N", "R", "W"))
                  and bad_domain == 0)
            if not ok:
                fail(f"{path.name}: 条数 {len(rows)} / 标签 {dict(labels)} / 错域 {bad_domain}")
            if missing_era:
                msg = f"{path.name} 缺少年代 {missing_era}"
                if meta.get("start", 1949) == meta.get("end", 2023):
                    warn(msg) if not ARGS.strict_era else fail(msg)
                else:
                    fail(msg)
            print(f"  {'OK ' if ok else 'BAD'} D{task} {meta['name']:<20} {split:<5}"
                  f" {len(rows):>4} 条  N/R/W={labels.get('N', 0)}/{labels.get('R', 0)}/{labels.get('W', 0)}"
                  f"  错域={bad_domain}  年代={len(eras)}/5")
    return data, {}


def audit_uniqueness(data: dict[str, list[dict]]) -> None:
    print()
    print("=" * 92)
    print("[2][3] 文本唯一性（弱键 = 现有实现；强键 = NFKC + 去标点）")
    print("=" * 92)
    weak: dict[str, str] = {}
    strong: dict[str, str] = {}
    weak_dup = strong_dup = 0
    for name, rows in data.items():
        for r in rows:
            text = r.get("text", "")
            kw, ks = text_key_weak(text), text_key_strong(text)
            if kw in weak:
                weak_dup += 1
                fail(f"弱键重复: {weak[kw]} 与 {name}")
            else:
                weak[kw] = name
            if ks in strong:
                strong_dup += 1
                fail(f"强键重复: {strong[ks]} 与 {name}")
            else:
                strong[ks] = name
    total = sum(len(v) for v in data.values())
    print(f"  条目 {total:,}；弱键唯一 {len(weak):,}（重复 {weak_dup}）；"
          f"强键唯一 {len(strong):,}（重复 {strong_dup}）")


def doc_identity(record: dict) -> str:
    """与 incremental_common._document_key 同口径：优先 DocumentKey 列表。"""
    key = record.get("DocumentKey")
    if isinstance(key, list):
        return json.dumps(key, ensure_ascii=False)
    return str(record.get("DocumentID"))


def audit_docs(data: dict[str, list[dict]]) -> None:
    print()
    print("=" * 92)
    print("[6] 文档维度（A3：不要求互斥，仅留痕）")
    print("=" * 92)
    locs: dict[str, set[str]] = defaultdict(set)
    for name, rows in data.items():
        split = "train" if "_train" in name else "test"
        for r in rows:
            locs[doc_identity(r)].add(split)
    overlap = sum(1 for s in locs.values() if len(s) > 1)
    train_docs = sum(1 for s in locs.values() if "train" in s)
    test_docs = sum(1 for s in locs.values() if "test" in s)
    print(f"  独立文档 {len(locs):,}；train {train_docs:,}；test {test_docs:,}；"
          f"跨 train/test 重叠 {overlap:,} 篇（A3 允许）")


def audit_truncation(data: dict[str, list[dict]]) -> None:
    print()
    print("=" * 92)
    print("[7] prompt 长度与截断率（本地 tokenizer，CPU）")
    print("=" * 92)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        warn("未安装 transformers，跳过截断统计")
        return
    from incremental_common import MODEL_NAME, prompt_prefix_tokens, truncation_stats

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    prefix_tokens = prompt_prefix_tokens(tok)
    head_budget = int(EVAL_MAX_LENGTH * 0.5)
    print(f"  提示词前缀 {prefix_tokens} token；head 预算 {head_budget} token；"
          f"触发截断时正文头部剩余 {head_budget - prefix_tokens} token")

    total = truncated = 0
    lo, hi, worst_p95 = 10 ** 9, 0, 0
    per_file: dict[str, int] = {}
    for name, rows in sorted(data.items()):
        stats = truncation_stats(rows, tok, EVAL_MAX_LENGTH)
        per_file[name] = stats["truncated"]
        total += stats["total"]
        truncated += stats["truncated"]
        length = stats["length_stats"]
        lo = min(lo, length["min"])
        hi = max(hi, length["max"])
        worst_p95 = max(worst_p95, length["p95"])
    print(f"  prompt 长度（合并口径）min={lo} / 各文件 p95 最大值={worst_p95} / max={hi}")
    print(f"  截断 {truncated:,} / {total:,} = {truncated / total:.2%}")
    for name, count in sorted(per_file.items()):
        if count:
            print(f"    {name}: {count}")


def main() -> int:
    data, _ = audit_rows()
    audit_uniqueness(data)
    audit_docs(data)
    if ARGS.truncation:
        audit_truncation(data)

    print()
    print("=" * 92)
    if PROBLEMS:
        print(f"结论：FAIL（{len(PROBLEMS)} 个问题）")
        for msg in PROBLEMS[:30]:
            print(f"  - {msg}")
        return 1
    print("结论：PASS —— 条数/标签/域字段/弱键/强键/文档统计全部符合契约")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncation", action="store_true", help="统计 prompt 截断率")
    parser.add_argument("--strict-era", action="store_true", help="每域 T1~T5 缺失即失败")
    ARGS = parser.parse_args()
    sys.exit(main())
