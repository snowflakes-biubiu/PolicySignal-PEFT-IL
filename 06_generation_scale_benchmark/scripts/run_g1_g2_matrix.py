# -*- coding: utf-8 -*-
"""06 期 G1/G2 唯一编排入口：计划预览 / 批量训练 / 结果汇总。

矩阵来源：`configs/g1_g2_matrix.json`（9 条 run × 2 个任务 = 18 个单元，
其中 G2-2 复用 G1-4，实际训练 16 个单元）。

用法（仓库根目录）：
  # 1) 只打印计划，确认参数
  python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode plan
  # 2) 冒烟（2 条样本、1 步；产物隔离在 assets/.../smoke_tests/）
  python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode run --smoke
  # 3) 正式全量
  python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode all
  # 4) 只跑某个实验 / 某任务
  python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode all --only G1-3 G2-3
  python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode all --task task2_level2
  # 5) 断点续跑（已成功的单元自动跳过）
  python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode all --state logs/summary_<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_common as C  # noqa: E402

TRAIN_PEFT = SCRIPTS_DIR / "train" / "run_peft_benchmark.py"
TRAIN_ENCODER = SCRIPTS_DIR / "train" / "finetune_encoder.py"
SUMMARIZE = SCRIPTS_DIR / "evaluate" / "summarize_results.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="06 期 G1/G2 矩阵编排")
    parser.add_argument("--mode", choices=["plan", "run", "summarize", "all"], default="plan")
    parser.add_argument("--config", default=str(C.CONFIG_PATH))
    parser.add_argument("--only", nargs="*", default=None, help="按实验编号子串过滤，如 G1-3 G2-4")
    parser.add_argument("--group", nargs="*", default=None, help="G1 / G2")
    parser.add_argument("--task", nargs="*", default=None, help="task1_level1 / task2_level2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    parser.add_argument("--model-root", default=str(C.MODEL_ROOT))
    parser.add_argument("--log-root", default=str(C.LOG_ROOT))
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--state", default=None, help="上次台账 summary_<stamp>.json，跳过已成功单元")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true", help="不向子脚本传 --skip-existing")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_command(run: dict[str, Any], task: str, args: argparse.Namespace) -> list[str]:
    skip = [] if args.no_skip_existing else ["--skip-existing"]
    smoke = ["--smoke"] if args.smoke else []
    common = [
        "--exp-id", run["id"],
        "--model", run["model"],
        "--task", task,
        "--seed", str(run["seed"]),
        "--results-root", args.results_root,
        "--model-root", args.model_root,
        "--group", run.get("group", ""),
        "--label", run.get("label", ""),
    ]
    if run.get("runner", "peft") == "encoder":
        return [
            args.python,
            str(TRAIN_ENCODER),
            *common,
            "--precision", run["precision"],
            "--lr", str(run["lr"]),
            "--epochs", str(run["epochs"]),
            "--batch-size", str(run["batch_size"]),
            "--grad-accum", str(run["grad_accum"]),
            "--eval-batch-size", str(run["eval_batch_size"]),
            "--max-length", str(run["max_length"]),
            *skip,
            *smoke,
        ]
    command = [
        args.python,
        str(TRAIN_PEFT),
        *common,
        "--precision", run["precision"],
        "--lora-r", str(run["lora_r"]),
        "--lora-alpha", str(run["lora_alpha"]),
        "--lora-dropout", str(run["lora_dropout"]),
        "--target-scope", run["target_scope"],
        "--lr", str(run["lr"]),
        "--epochs", str(run["epochs"]),
        "--batch-size", str(run["batch_size"]),
        "--grad-accum", str(run["grad_accum"]),
        "--eval-batch-size", str(run["eval_batch_size"]),
        "--max-length", str(run["max_length"]),
        "--warmup-ratio", str(run["warmup_ratio"]),
        *skip,
        *smoke,
    ]
    if run.get("enable_thinking"):
        command.append("--enable-thinking")
    if run.get("gradient_checkpointing"):
        command.append("--gradient-checkpointing")
    return command


def load_state(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        unit["key"]: unit
        for unit in payload.get("units", [])
        if unit.get("status") in {"ok", "reused"}
    }


def print_plan(units: list[dict[str, Any]]) -> None:
    print(f"{'编号':46s} {'任务':14s} {'模型':30s} {'精度':6s} {'配置':22s} 备注")
    print("-" * 140)
    for unit in units:
        run = unit["run"]
        if run.get("runner", "peft") == "encoder":
            config = f"full/lr={run['lr']:g}"
        else:
            config = (
                f"r={run['lora_r']}/a={run['lora_alpha']}/"
                f"{run['target_scope']}/lr={run['lr']:g}"
            )
        note = run.get("note", "")
        if unit["reuse"]:
            note = f"复用 {unit['reuse']}｜" + note
        print(
            f"{unit['key']:46s} {unit['task']:14s} {run['model']:30s} "
            f"{run['precision']:6s} {config:22s} {note}"
        )
    total = len(units)
    reused = sum(1 for unit in units if unit["reuse"])
    print("-" * 140)
    print(f"单元总数 {total}；需训练 {total - reused}；复用 {reused}")


def run_unit(unit: dict[str, Any], args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    run = unit["run"]
    command = build_command(run, unit["task"], args)
    log_path = log_dir / f"{unit['key']}.log"
    started = time.perf_counter()
    returncode = -1
    attempts = 0
    for attempt in range(1, max(1, args.retries) + 1):
        attempts = attempt
        with log_path.open("wb") as handle:
            handle.write((" ".join(command) + "\n\n").encode("utf-8"))
            handle.flush()
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        returncode = completed.returncode
        if returncode == 0:
            break
        if attempt < max(1, args.retries):
            print(f"  [重试] {unit['key']} 第 {attempt} 次失败，{args.retry_wait}s 后重试")
            time.sleep(args.retry_wait)
    duration = round(time.perf_counter() - started, 2)

    summary_path = C.run_paths(
        run["id"], unit["task"], args.results_root, args.model_root
    )["summary"]
    accuracy = macro_f1 = None
    if returncode == 0 and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        accuracy = summary.get("accuracy")
        macro_f1 = summary.get("macro_f1")
    return {
        "key": unit["key"],
        "exp_id": run["id"],
        "task": unit["task"],
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "attempts": attempts,
        "duration_sec": duration,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "log": str(log_path),
        "command": command,
    }


def main() -> int:
    args = parse_args()
    matrix = C.load_matrix(args.config)
    units = list(
        C.iter_units(matrix, only=args.only, groups=args.group, tasks=args.task)
    )
    if not units:
        print("没有匹配到任何执行单元，请检查 --only / --group / --task")
        return 2

    print_plan(units)
    if args.mode == "plan":
        return 0

    if args.mode == "summarize":
        command = [
            args.python,
            str(SUMMARIZE),
            "--config", args.config,
            "--results-root", args.results_root,
        ]
        return subprocess.run(command).returncode

    stamp = args.stamp or C.now_stamp()
    log_dir = Path(args.log_root) / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    done = load_state(args.state)
    if done:
        print(f"[状态] 从 {args.state} 读取到 {len(done)} 个已成功单元，将跳过")

    records: list[dict[str, Any]] = []
    for index, unit in enumerate(units, 1):
        print(f"\n=== [{index}/{len(units)}] {unit['key']} ===")
        if unit["key"] in done:
            print("  [跳过] 台账记录已成功")
            records.append({**done[unit["key"]], "note": "carried-from-state"})
            continue
        if unit["reuse"]:
            print(f"  [复用] {unit['reuse']}，不重复训练")
            records.append(
                {
                    "key": unit["key"],
                    "exp_id": unit["run"]["id"],
                    "task": unit["task"],
                    "status": "reused",
                    "reuse": unit["reuse"],
                    "duration_sec": 0.0,
                }
            )
            continue
        record = run_unit(unit, args, log_dir)
        records.append(record)
        flag = "OK" if record["status"] == "ok" else f"FAILED(code={record['returncode']})"
        print(
            f"  [{flag}] {record['duration_sec']}s "
            f"Acc={record['accuracy']} MF1={record['macro_f1']} log={record['log']}"
        )
        if record["status"] != "ok" and args.stop_on_error:
            print("[中断] --stop-on-error 生效")
            break

    ledger = {
        "stamp": stamp,
        "mode": args.mode,
        "smoke": args.smoke,
        "config": str(Path(args.config).resolve()),
        "results_root": str(Path(args.results_root).resolve()),
        "model_root": str(Path(args.model_root).resolve()),
        "units": records,
        "counts": {
            "total": len(records),
            "ok": sum(1 for r in records if r["status"] == "ok"),
            "failed": sum(1 for r in records if r["status"] == "failed"),
            "reused": sum(1 for r in records if r["status"] == "reused"),
        },
    }
    ledger_path = log_dir / f"summary_{stamp}.json"
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[台账] {ledger_path}")

    if args.mode == "all":
        print("\n=== 汇总结果 ===")
        subprocess.run(
            [
                args.python,
                str(SUMMARIZE),
                "--config", args.config,
                "--results-root", args.results_root,
            ]
        )

    return 0 if ledger["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
