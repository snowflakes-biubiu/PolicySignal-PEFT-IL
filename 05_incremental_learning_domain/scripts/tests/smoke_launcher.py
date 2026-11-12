"""测试 3/3：run_all_continual.py 无监管顺序启动的冒烟测试。

驱动真实的统一启动器以 --smoke 模式跑完整矩阵（默认 3 方法 × Task2-3 = 6 个运行），
验证无监管顺序启动的机制本身可靠：
    1. 严格按 方法(EWC→Replay→Adapter) × 任务(2→3…) 顺序执行
    2. 每个运行产出独立日志文件
    3. 汇总 smoke_summary_*.json 生成且记录完整
    4. 全部运行返回码为 0（失败会打断链并记入汇总）

正式启用无监管长跑（12 个实验）前必须先通过本测试。

用法：
    python 05_incremental_learning_domain/scripts/tests/smoke_launcher.py            # 3 方法 × Task2-3
    python 05_incremental_learning_domain/scripts/tests/smoke_launcher.py --full     # 3 方法 × Task2-6
    python 05_incremental_learning_domain/scripts/tests/smoke_launcher.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (PROJECT_ROOT / "05_incremental_learning_domain" / "scripts"
            / "run_all_continual.py")
LOG_ROOT = PROJECT_ROOT / "05_incremental_learning_domain" / "logs" / "continual_runs"

METHODS = ["train_ewc", "train_replay", "train_adapter_cl"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="覆盖 Task2-6 全部（默认 Task2-3）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    args = ap.parse_args()
    tasks = [2, 3, 4, 5, 6] if args.full else [2, 3]

    expected = [f"{m}_task{t}" for m in METHODS for t in tasks]
    print("=" * 78)
    print("测试 3/3：无监管顺序启动冒烟（run_all_continual.py --smoke）")
    print("=" * 78)
    print(f"预期 {len(expected)} 个运行，顺序：")
    for label in expected:
        print(f"  {label}")

    cmd = [sys.executable, str(LAUNCHER), "--smoke", "--tasks", *[str(t) for t in tasks]]
    if args.dry_run:
        print(f"\n[dry-run] 将执行: {' '.join(cmd)}")
        return 0

    print(f"\n启动: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    minutes = (time.time() - t0) / 60
    print(f"\n启动器退出码: {proc.returncode}  总耗时 {minutes:.1f} 分钟")

    problems: list[str] = []
    if proc.returncode != 0:
        problems.append(f"启动器退出码 {proc.returncode}（可能有运行失败，看汇总）")

    # 定位本次汇总（最新的 smoke_summary_*.json）
    summaries = sorted(LOG_ROOT.glob("smoke_summary_*.json"),
                       key=lambda p: p.stat().st_mtime)
    if not summaries:
        problems.append("未找到 smoke_summary_*.json 汇总文件")
        summary = None
    else:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        print(f"汇总文件: {summaries[-1].name}")

    if summary is not None:
        results = summary.get("results", [])
        labels = [r["label"] for r in results]

        if labels != expected:
            problems.append(f"执行顺序/数量不符: 实际 {labels} != 预期 {expected}")
        else:
            print("[OK] 执行顺序与数量完全符合预期")

        bad = [r for r in results if r.get("returncode") != 0]
        if bad:
            problems.append(f"{len(bad)} 个运行失败/被跳过: "
                            + ", ".join(f"{r['label']}({r.get('returncode', r.get('skipped'))})"
                                        for r in bad))
        else:
            print("[OK] 全部运行返回码 0")

        missing_logs = [r["label"] for r in results
                        if r.get("returncode") == 0 and
                        not Path(r["log"]).is_file()]
        if missing_logs:
            problems.append(f"日志文件缺失: {missing_logs}")
        else:
            print("[OK] 每个运行都有独立日志文件")

    print("\n" + "=" * 78)
    if problems:
        print(f"发现 {len(problems)} 个问题:")
        for p in problems:
            print(f"  [X] {p}")
    else:
        print("全部通过：顺序启动机制可靠，可以进行 12 个实验的无监管长跑")
        print("正式运行: python 05_incremental_learning_domain/scripts/run_all_continual.py")
    print("=" * 78)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
