"""一键无监管顺序启动 12 个持续学习实验（EWC / Replay / Adapter-CL × Task2-5）。

实验矩阵（每个方法内必须 2→3→4→5 顺序，链式 parent 依赖）：
    train_ewc.py        --task 2/3/4/5
    train_replay.py     --task 2/3/4/5
    train_adapter_cl.py --task 2/3/4/5

特性：
    - Task1 adapter 缺失时自动先补训 train_task1.py（Task2 的 parent 依赖）
    - 每个实验输出独立日志到 05_incremental_learning_time/logs/continual_runs/
    - 单个实验失败：跳过该方法的剩余任务（链依赖），其余方法继续
    - 结束打印汇总表 + 写 summary_*.json；有失败则退出码 1
    - --resume 透传给每个训练（从最近 Trainer checkpoint 续训）

用法（项目根目录执行）：
    python 05_incremental_learning_time/scripts/run_all_continual.py
    python 05_incremental_learning_time/scripts/run_all_continual.py --dry-run
    python 05_incremental_learning_time/scripts/run_all_continual.py --methods train_ewc --tasks 2 3
    python 05_incremental_learning_time/scripts/run_all_continual.py --resume
    python 05_incremental_learning_time/scripts/run_all_continual.py --smoke --tasks 2 3   # 冒烟顺序启动测试
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPTS / "train"
MODEL_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "models"
LOG_ROOT = PROJECT_ROOT / "05_incremental_learning_time" / "logs" / "continual_runs"
SMOKE_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "smoke_tests"

METHODS = ["train_ewc", "train_replay", "train_adapter_cl", "train_cumulative"]
METHOD_TO_EVAL = {
    "train_ewc": "ewc",
    "train_replay": "replay",
    "train_adapter_cl": "adapter",
    "train_cumulative": "cumulative",
}
TASK_NAMES = {
    1: "T1_1949_1978", 2: "T2_1979_1992", 3: "T3_1993_2002",
    4: "T4_2003_2012", 5: "T5_2013_2023",
}
TASKS = [2, 3, 4, 5]

sys.path.insert(0, str(SCRIPTS))
from gpu_guard import ALLOW_ENV as _ALLOW_ENV  # noqa: E402
from gpu_guard import assert_cloud_gpu_allowed  # noqa: E402
# 与 incremental_common.adapter_name(1) 保持一致
TASK1_ADAPTER = MODEL_ROOT / "stage1" / "qwen15b_temporal_task1_e2_r16_a32_lr2e-4"


def run_one(label: str, cmd: list[str], log_file: Path) -> dict:
    """跑一条命令，stdout/stderr 全量写日志，返回结果记录。"""
    print(f"[{datetime.now():%H:%M:%S}] 启动 {label}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT), text=True, encoding="utf-8", errors="replace")
    duration = time.time() - t0
    if proc.returncode != 0:
        try:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        guard_markers = ("按项目规范禁止在本地显卡", "需要 CUDA 云端 GPU", _ALLOW_ENV)
        if any(marker in log_text for marker in guard_markers):
            print(f"[{datetime.now():%H:%M:%S}] [中止] {label} 被 GPU 守卫拒绝"
                  "（本地显卡），终止整个流程，不再继续下一个单元。", flush=True)
            raise SystemExit(3)
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{datetime.now():%H:%M:%S}] {label} {status}  "
          f"耗时 {duration / 60:.1f} 分钟  日志: {log_file.name}", flush=True)
    return {"label": label, "cmd": " ".join(cmd), "returncode": proc.returncode,
            "minutes": round(duration / 60, 1), "log": str(log_file)}


def _gate_check(method: str, task: int, threshold: float) -> dict | None:
    """读取刚生成的评估报告，检查当前阶段准确率是否达到门禁阈值。

    用于「加入新窗口后其性能是否确实提升」的早停判断：
    threshold<=0 关闭门禁；报告缺失或缺指标时不阻断（passed=None）。
    """
    if threshold <= 0:
        return None
    path = (PROJECT_ROOT / "05_incremental_learning_time" / "results" / "temporal"
            / f"temporal_{method}_task{task}_cumulative_test.json")
    if not path.is_file():
        return {"passed": None, "reason": f"评估报告缺失: {path.name}"}
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = (report.get("metrics_by_test_task") or {}).get(TASK_NAMES[task]) or {}
    acc = metrics.get("acc")
    if acc is None:
        return {"passed": None, "reason": "评估报告缺少当前阶段指标"}
    return {"passed": float(acc) >= float(threshold),
            "current_acc": float(acc),
            "threshold": float(threshold)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS,
                    help="只跑指定方法（默认三个全跑）")
    ap.add_argument("--tasks", nargs="+", type=int, default=TASKS,
                    choices=TASKS, help="只跑指定任务（默认 2 3 4 5）")
    ap.add_argument("--resume", action="store_true",
                    help="透传 --resume，从最近 Trainer checkpoint 续训")
    ap.add_argument("--stop-on-failure", action="store_true",
                    help="任一实验失败即整体停止（默认: 只跳过该方法的剩余任务）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：每个实验单步训练，产物写入临时冒烟目录（不产出正式模型）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子；非 42 的种子写入 *_s{seed} 目录，与既有实验隔离")
    ap.add_argument("--evaluate", action="store_true",
                    help="每个阶段训练完成后立即做累计评估并保存逐条预测（默认只训练）")
    ap.add_argument("--gate-current-acc", type=float, default=0.0,
                    help="门禁阈值：当前阶段准确率低于该值则中断该方法后续阶段（<=0 关闭）")
    ap.add_argument("--evaluate-split", choices=("test", "val"), default="test",
                    help="训练后评估使用固定测试集(test)或验证集(val)；"
                         "val 用于模型选择（λ/权重/epoch），test 只做最终报告")
    args = ap.parse_args()

    # 编排器入口同样拦截本地 GPU：否则每个子进程被拒后仍会继续下一个单元
    if not args.dry_run:  # --dry-run 只打印计划，允许本地预览
        try:
            assert_cloud_gpu_allowed("持续学习编排（run_all_continual）")
        except RuntimeError as error:
            print(f"[拒绝] {error}")
            return 3

    if args.smoke and args.resume:
        print("--smoke 与 --resume 不可同用")
        return 1

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.smoke:
        # 冒烟产物集中存放且每次运行前清空，保证测试可重复
        for sub in ("continual", "continual_state"):
            shutil.rmtree(SMOKE_ROOT / sub, ignore_errors=True)

    plan: list[tuple[str, list[str]]] = []

    # Task1 adapter 是 Task2 的 parent，缺失则先补训
    if not (TASK1_ADAPTER / "adapter_config.json").is_file():
        if args.smoke:
            print(f"冒烟模式需要已存在的 Task1 adapter（冒烟不产出正式模型）: {TASK1_ADAPTER}")
            print("请先正式运行 train_task1.py，再重试。")
            return 1
        plan.append(("train_task1",
                     [sys.executable, str(TRAIN_DIR / "train_task1.py")]))

    for method in args.methods:
        for task in args.tasks:
            cmd = [sys.executable, str(TRAIN_DIR / f"{method}.py"),
                   "--task", str(task), "--seed", str(args.seed)]
            if args.resume:
                cmd.append("--resume")
            if args.smoke:
                cmd.append("--smoke")
            plan.append((f"{method}_task{task}", cmd))

    print(f"计划实验 {len(plan)} 个：")
    for label, cmd in plan:
        print(f"  {label:<26} {' '.join(cmd)}")
    if args.dry_run:
        return 0

    results: list[dict] = []
    failed_methods: set[str] = set()
    aborted = False

    for label, cmd in plan:
        method = label.rsplit("_task", 1)[0]
        if aborted:
            results.append({"label": label, "cmd": " ".join(cmd),
                            "returncode": None, "skipped": "aborted"})
            print(f"[{datetime.now():%H:%M:%S}] 跳过 {label}（--stop-on-failure）")
            continue
        if method in failed_methods:
            results.append({"label": label, "cmd": " ".join(cmd),
                            "returncode": None, "skipped": "chain-broken"})
            print(f"[{datetime.now():%H:%M:%S}] 跳过 {label}（{method} 链已中断）")
            continue

        prefix = "smoke_" if args.smoke else ""
        record = run_one(label, cmd, LOG_ROOT / f"{prefix}{label}_{stamp}.log")
        results.append(record)
        record["evaluation"] = None
        if record["returncode"] == 0 and args.evaluate and not args.smoke:
            eval_method = METHOD_TO_EVAL[method]
            eval_cmd = [sys.executable,
                        str(SCRIPTS / "evaluate" / f"evaluate_task{task}_cumulative.py"),
                        "--method", eval_method, "--model-seed", str(args.seed),
                        "--split", args.evaluate_split]
            record["evaluation"] = run_one(f"eval_{label}", eval_cmd,
                                           LOG_ROOT / f"{prefix}eval_{label}_{stamp}.log")
            gate = _gate_check(eval_method, task, args.gate_current_acc)
            record["gate"] = gate
            if gate is not None and gate.get("passed") is False:
                print(f"[门禁] {label} 当前阶段准确率 {gate['current_acc']:.4f} < "
                      f"{gate['threshold']:.4f}，中断该方法后续阶段")
                failed_methods.add(method)
                continue
        if record["returncode"] != 0:
            if label == "train_task1":
                print("Task1 补训失败，后续全部依赖它的实验无法进行，终止。")
                aborted = True
            elif args.stop_on_failure:
                aborted = True
            else:
                failed_methods.add(method)

    # 汇总
    n_ok = sum(1 for r in results if r.get("returncode") == 0)
    n_fail = sum(1 for r in results if r.get("returncode") not in (0, None))
    n_skip = sum(1 for r in results if r.get("returncode") is None)

    print("\n" + "=" * 72)
    print(f"汇总: OK {n_ok}  失败 {n_fail}  跳过 {n_skip}  (共 {len(plan)})")
    print("=" * 72)
    for r in results:
        rc = r.get("returncode")
        state = "OK" if rc == 0 else (f"FAIL({rc})" if rc is not None
                                      else f"SKIP({r.get('skipped')})")
        mins = f"{r['minutes']}min" if "minutes" in r else "-"
        print(f"  {r['label']:<26} {state:<12} {mins:>8}")

    summary_name = f"smoke_summary_{stamp}.json" if args.smoke else f"summary_{stamp}.json"
    summary_path = LOG_ROOT / summary_name
    summary_path.write_text(
        json.dumps({"stamp": stamp, "smoke": args.smoke, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n汇总已写入: {summary_path}")
    print(f"全部日志目录: {LOG_ROOT}")
    return 1 if (n_fail or (n_skip and not n_ok)) else 0


if __name__ == "__main__":
    sys.exit(main())
