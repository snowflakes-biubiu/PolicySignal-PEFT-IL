"""测试 2/3：三个持续学习实验框架（EWC / Replay / Adapter-CL）冒烟测试。

对每个方法跑 Task2 + Task3 的 --smoke 单步训练（Task3 覆盖链式 parent/状态加载），
验证三条代码路径全程无异常：
    EWC     : parent 加载 → Fisher 估计 → 惩罚损失 → Fisher 状态保存/续传
    Replay  : exemplar 采样 → 回放池混合训练 → 回放池读写
    Adapter : 多 Adapter 加载 → 新增 adapter → 冻结检查 → manifest 保存/续传

冒烟产物写入 artifacts/smoke_tests/ 下的临时目录（不污染正式模型/回放池/EWC 状态），
每次运行前自动清空。需要 GPU 与已训练的 Task1 adapter。

用法：
    python 05_incremental_learning_domain/scripts/tests/smoke_experiments.py
    python 05_incremental_learning_domain/scripts/tests/smoke_experiments.py --methods train_ewc --tasks 2
    python 05_incremental_learning_domain/scripts/tests/smoke_experiments.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "05_incremental_learning_domain" / "scripts"
TRAIN_DIR = SCRIPTS / "train"
SMOKE_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "smoke_tests"
LOG_ROOT = PROJECT_ROOT / "05_incremental_learning_domain" / "logs" / "continual_runs"
TASK1_ADAPTER = (PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "models"
                 / "qwen15b_domain_task1_e2_r16_a32_lr2e-4")

METHODS = ["train_ewc", "train_replay", "train_adapter_cl"]
TASKS = [2, 3]
# 各方法冒烟产物的完成标记
MARKERS = {
    "train_ewc": "adapter_config.json",
    "train_replay": "adapter_config.json",
    "train_adapter_cl": "adapter_manifest.json",
}
METHOD_DIRS = {"train_ewc": "ewc", "train_replay": "replay", "train_adapter_cl": "adapter"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--tasks", nargs="+", type=int, default=TASKS, choices=[2, 3, 4, 5, 6])
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    args = ap.parse_args()

    print("=" * 78)
    print("测试 2/3：三实验框架冒烟测试（单步训练 + 链式状态加载）")
    print("=" * 78)

    if not (TASK1_ADAPTER / "adapter_config.json").is_file():
        print(f"[X] 缺少 Task1 adapter（冒烟的 parent）: {TASK1_ADAPTER}")
        print("    请先正式运行 train_task1.py。")
        return 1

    plan = [(m, t) for m in args.methods for t in args.tasks]
    print(f"计划 {len(plan)} 个冒烟运行："
          + ", ".join(f"{m.removeprefix('train_')}#T{t}" for m, t in plan))
    if args.dry_run:
        return 0

    # 清空冒烟产物，保证可重复
    for sub in ("continual", "continual_state"):
        shutil.rmtree(SMOKE_ROOT / sub, ignore_errors=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    results = []
    broken_chain: set[str] = set()
    for method, task in plan:
        label = f"{method}_task{task}"
        if method in broken_chain:
            results.append((label, "SKIP(链已中断)"))
            print(f"\n>>> 跳过 {label}（{method} 前序冒烟失败）")
            continue

        cmd = [sys.executable, str(TRAIN_DIR / f"{method}.py"), "--task", str(task), "--smoke"]
        log_file = LOG_ROOT / f"smokeexp_{label}_{stamp}.log"
        print(f"\n>>> {label}: {' '.join(cmd[1:])}")
        t0 = time.time()
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(PROJECT_ROOT), text=True,
                                  encoding="utf-8", errors="replace")
        minutes = (time.time() - t0) / 60

        marker = SMOKE_ROOT / "continual" / f"{METHOD_DIRS[method]}_task{task}" / MARKERS[method]
        ok = proc.returncode == 0 and marker.is_file()
        state = "OK" if ok else f"FAIL(rc={proc.returncode}, marker={marker.is_file()})"
        results.append((label, state))
        print(f"    {state}  耗时 {minutes:.1f} 分钟  日志: {log_file.name}")
        if not ok:
            broken_chain.add(method)

    print("\n" + "=" * 78)
    n_ok = sum(1 for _, s in results if s == "OK")
    print(f"汇总: {n_ok}/{len(results)} 通过")
    print("=" * 78)
    for label, state in results:
        print(f"  {label:<26} {state}")
    print(f"\n日志目录: {LOG_ROOT}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
