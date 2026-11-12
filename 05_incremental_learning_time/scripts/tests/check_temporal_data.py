"""测试 1/3：T1-T5 时间增量数据完整性检查（纯标准库，无需 GPU/模型）。

检查项：
  1. 文件存在且 JSONL 可解析、记录数正确（train 600 / test 300）
  2. 字段齐全（text/label/DocumentID/year/era/period/split 等 16 个字段）
  3. 标签合法（N/R/W）且类别均衡（train 200×3 / test 100×3）
  4. 文本长度 20-768、非空白
  5. 年份/era/period/split 与所属 Task 一致
  6. Task 内 train/test 文档互斥；全局（跨 Task）文档不复用
  7. 全局文本去重（归一化后无重复）
  8. 与 build_report.json 的统计交叉核对（若存在）

文档身份口径（重要）：一律用 DocumentKey=[DocumentID, IssueDate, Title]。
语料的 DocumentID 不是唯一标识（同一 ID 可装多份无关文件），
按 DocumentID 归并会把不同文档误判为泄漏。

用法：
    python 05_incremental_learning_time/scripts/tests/check_temporal_data.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "data" / "temporal"

TASKS = [
    (1, "T1_1949_1978", 1949, 1978),
    (2, "T2_1979_1992", 1979, 1992),
    (3, "T3_1993_2002", 1993, 2002),
    (4, "T4_2003_2012", 2003, 2012),
    (5, "T5_2013_2023", 2013, 2023),
]
EXPECTED_N = {"train": 600, "test": 300}
EXPECTED_PER_CLASS = {"train": 200, "test": 100}
REQUIRED_FIELDS = (
    "text", "label", "label_source", "level2_auto", "DocumentID", "ParagraphID",
    "year", "era", "Title", "IssuingDepartment", "Category", "Validity",
    "period", "domain", "split",
)
MIN_LEN, MAX_LEN = 20, 768


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def doc_identity(r: dict) -> str:
    """真实文档身份：优先 DocumentKey=[DocumentID, IssueDate, Title]。"""
    key = r.get("DocumentKey")
    if key is not None:
        return json.dumps(key, ensure_ascii=False) if isinstance(key, list) else str(key)
    return str(r.get("DocumentID"))


def check_task(task_no: int, name: str, y0: int, y1: int,
               problems: list[str], doc_locations: dict, text_locations: dict) -> None:
    for split in ("train", "test"):
        path = DATA_ROOT / f"{name}_{split}.jsonl"
        tag = f"{name}/{split}"
        if not path.is_file():
            problems.append(f"{tag}: 文件不存在 {path}")
            continue
        records = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                problems.append(f"{tag}: 第 {i} 行 JSON 解析失败: {e}")
        n = len(records)
        print(f"  [{tag}] {n} 条")

        if n != EXPECTED_N[split]:
            problems.append(f"{tag}: 数量 {n} != {EXPECTED_N[split]}")

        missing_fields = Counter()
        for r in records:
            for f in REQUIRED_FIELDS:
                if f not in r:
                    missing_fields[f] += 1
        if missing_fields:
            problems.append(f"{tag}: 字段缺失 {dict(missing_fields)}")

        # 标签 / 类别均衡
        labels = Counter(r.get("label") for r in records)
        for lab in ("N", "R", "W"):
            if labels.get(lab, 0) != EXPECTED_PER_CLASS[split]:
                problems.append(f"{tag}: 类别 {lab}={labels.get(lab, 0)} != {EXPECTED_PER_CLASS[split]}")
        bad_label = [r.get("label") for r in records if r.get("label") not in ("N", "R", "W")]
        if bad_label:
            problems.append(f"{tag}: 非法标签 {bad_label[:5]}")

        # 文本长度 / 年份 / era / period / split 一致性
        for r in records:
            pid = r.get("ParagraphID", "?")
            text = str(r.get("text", ""))
            if not text.strip() or not (MIN_LEN <= len(text) <= MAX_LEN):
                problems.append(f"{tag}: ParagraphID={pid} 文本长度异常 len={len(text)}")
            year = r.get("year")
            if not isinstance(year, int) or not (y0 <= year <= y1):
                problems.append(f"{tag}: ParagraphID={pid} 年份越界 {year} (应 {y0}-{y1})")
            if r.get("era") != name:
                problems.append(f"{tag}: ParagraphID={pid} era={r.get('era')} != {name}")
            if r.get("period") != name:
                problems.append(f"{tag}: ParagraphID={pid} period={r.get('period')} != {name}")
            if r.get("split") != split:
                problems.append(f"{tag}: ParagraphID={pid} split={r.get('split')} != {split}")
            if r.get("label_source") != "auto_R1":
                problems.append(f"{tag}: ParagraphID={pid} label_source={r.get('label_source')}")
            doc_locations[doc_identity(r)].add(tag)
            text_locations[norm(text)].add(tag)

    # Task 内 train/test 文档互斥（DocumentKey 口径）
    tr = {doc_identity(r) for r in
          (json.loads(l) for l in (DATA_ROOT / f"{name}_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    te = {doc_identity(r) for r in
          (json.loads(l) for l in (DATA_ROOT / f"{name}_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    overlap = tr & te
    if overlap:
        problems.append(f"{name}: train/test 文档交叠 {len(overlap)} 篇: {sorted(overlap)[:5]}")
    print(f"  [{name}] 文档(DocumentKey): train {len(tr)} 篇 / test {len(te)} 篇 / 交叠 {len(overlap)}")


def main() -> int:
    print("=" * 78)
    print("测试 1/3：T1-T5 时间增量数据完整性检查")
    print("=" * 78)

    problems: list[str] = []
    doc_locations: dict[str, set] = defaultdict(set)
    text_locations: dict[str, set] = defaultdict(set)

    for task_no, name, y0, y1 in TASKS:
        print(f"\n[{name}]")
        check_task(task_no, name, y0, y1, problems, doc_locations, text_locations)

    print("\n" + "-" * 78)
    print("全局检查")
    cross_doc = {d: locs for d, locs in doc_locations.items() if len(locs) > 1}
    cross_txt = {t: locs for t, locs in text_locations.items() if len(locs) > 1}
    print(f"  跨 Task/split 复用的文档: {len(cross_doc)}")
    print(f"  跨 Task/split 重复的文本: {len(cross_txt)}")
    if cross_doc:
        problems.append(f"全局: {len(cross_doc)} 篇文档被多个 Task/split 复用，例: "
                        f"{list(cross_doc.items())[:3]}")
    if cross_txt:
        problems.append(f"全局: {len(cross_txt)} 条文本重复出现")

    # 与 build_report.json 交叉核对
    report_path = DATA_ROOT / "build_report.json"
    if report_path.is_file():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        for task_no, name, _, _ in TASKS:
            t = rep.get("tasks", {}).get(name)
            if not t:
                continue
            for split in ("train", "test"):
                actual = sum(1 for _ in
                             (DATA_ROOT / f"{name}_{split}.jsonl").open(encoding="utf-8"))
                if t[split]["n"] != actual:
                    problems.append(f"{name}/{split}: build_report 记录 {t[split]['n']} != 实际 {actual}")
        print("  build_report.json 交叉核对完成")

    print("\n" + "=" * 78)
    if problems:
        print(f"发现 {len(problems)} 个问题:")
        for p in problems:
            print(f"  [X] {p}")
    else:
        print("全部通过：数量 / 字段 / 类别均衡 / 年份一致 / 文档互斥 / 全局无泄漏")
    print("=" * 78)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
