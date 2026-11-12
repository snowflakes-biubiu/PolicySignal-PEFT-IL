"""E1 一致性自检（对应 P3-11）。

4 个方法在阶段 1 都评估**同一份共享 T1 模型**（`run_full_matrix.py` 的 `eval_command`
对 task==1 统一使用 `adapter_dir(1)`），因此四份结果必须逐位一致。
任何差异都说明：模型被换过、数据被改过、或某次评估走了不同代码路径。

只读，不加载模型。

用法：
    python 05_incremental_learning_domain/scripts/tests/check_e1_consistency.py
    python 05_incremental_learning_domain/scripts/tests/check_e1_consistency.py --strict
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from incremental_evaluation import result_path  # noqa: E402
from run_full_matrix import ALL_METHODS  # noqa: E402

ARGS = None


def load_pairs(json_path: Path) -> tuple[list[tuple[str, str]] | None, dict | None]:
    csv_path = json_path.with_name(json_path.stem + "_predictions.csv")
    if not csv_path.is_file():
        return None, None
    pairs: list[tuple[str, str]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("ParagraphID") or row.get("index") or ""
            pairs.append((str(key), str(row.get("prediction", ""))))
    report = None
    if json_path.is_file():
        try:
            report = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None
    return pairs, report


def main() -> int:
    print("=" * 88)
    print("E1 一致性自检：4 方法阶段 1 应逐位一致（同一份共享 T1 模型）")
    print("=" * 88)

    loaded: dict[str, list[tuple[str, str]]] = {}
    reports: dict[str, dict] = {}
    for method in ALL_METHODS:
        path = result_path(method, 1)
        pairs, report = load_pairs(path)
        if pairs is None:
            print(f"  -- {method:<14} 结果缺失（{path.name}）")
            continue
        loaded[method] = pairs
        if report:
            reports[method] = report
        acc = (report.get("continual_metrics") or {}).get("ACC") if report else None
        shown = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
        print(f"  OK {method:<14} {len(pairs):>4} 条  ACC={shown}  文件 {path.name}")

    if len(loaded) < 2:
        print("\n不足两份结果，暂无可比对象。")
        return 1 if (ARGS.strict and len(loaded) < len(ALL_METHODS)) else 0

    base_method, base = next(iter(loaded.items()))
    bad: list[str] = []
    for method, pairs in loaded.items():
        if method == base_method:
            continue
        if len(pairs) != len(base):
            bad.append(f"{method} 条数 {len(pairs)} != {base_method} {len(base)}")
            continue
        diff = [i for i, (a, b) in enumerate(zip(base, pairs)) if a != b]
        if diff:
            idx = diff[0]
            bad.append(f"{method} 与 {base_method} 有 {len(diff)} 处不同，"
                       f"首处 index={idx} {base[idx]} vs {pairs[idx]}")
        else:
            print(f"  == {method:<14} 与 {base_method} 逐位一致")

    missing = [m for m in ALL_METHODS if m not in loaded]
    print()
    if bad:
        print("结论：FAIL")
        for msg in bad:
            print(f"  - {msg}")
        return 1
    if missing:
        print(f"结论：部分通过（缺 {missing}）")
        return 1 if ARGS.strict else 0
    print("结论：PASS —— 4 份 E1 结果逐位一致")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="结果缺失也算失败")
    ARGS = parser.parse_args()
    sys.exit(main())
