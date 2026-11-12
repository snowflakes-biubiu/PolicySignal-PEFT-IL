# -*- coding: utf-8 -*-
"""全量矩阵编排：4 方法 × 5 阶段训练 + 4×15=60 组评估，一次调用完成。

训练：4 方法 × 5 阶段 = 20 个方法-阶段单元；T1 为四方法共享基线，只训练一次 → 实际 17 次训练。
评估：每方法每阶段评估 T1..Ti，4 × (1+2+3+4+5) = 60 组（E2/E3/E4 的 T1 用共享基线评估）。
依赖：T1 → E1 T2..T5 → E2 T2..T5 → E3 T2..T5 → E4 T2..T5。

用法（云端 RTX 4090D；本地会被 gpu_guard 拒绝）：
    python 05_incremental_learning_time/scripts/run_full_matrix.py [--dry-run]
    python 05_incremental_learning_time/scripts/run_full_matrix.py --skip-existing
    python 05_incremental_learning_time/scripts/run_full_matrix.py --evaluate-split val --gate-current-acc 0.75
    python 05_incremental_learning_time/scripts/run_full_matrix.py --methods ewc replay --tasks 2 3
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))

from continual_training import (  # noqa: E402
    EWC_LAMBDA_DEFAULT,
    REPLAY_MEMORY_CAPACITY,
    REPLAY_MODE_DEFAULT,
    REPLAY_RATIO_DEFAULT,
)
from gpu_guard import ALLOW_ENV as ALLOW_ENV_MARKER  # noqa: E402
from gpu_guard import SMOKE_MAX_SAMPLES, assert_cloud_gpu_allowed  # noqa: E402
from incremental_common import (  # noqa: E402
    DEFAULT_SEED,
    LABELS,
    MODEL_NAME,
    MODEL_ROOT,
    model_group,
    TASKS,
    adapter_dir,
    experiment_fingerprint,
    task_file,
)

TRAIN_DIR = SCRIPTS / "train"
EVAL_DIR = SCRIPTS / "evaluate"
LOG_ROOT = PROJECT_ROOT / "05_incremental_learning_time" / "logs" / "full_matrix"
RESULT_ROOT = PROJECT_ROOT / "05_incremental_learning_time" / "results" / "temporal"
STATE_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "continual_state"
SMOKE_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "smoke_tests"
SMOKE_CONT = SMOKE_ROOT / "continual"
SMOKE_EVAL = SMOKE_ROOT / "eval"
MODES = ["smoke", "full", "train", "eval"]
ALL_METHODS = ["sequential_ft", "ewc", "replay", "adapter"]
CL_METHODS = {"ewc": "train_ewc", "replay": "train_replay", "adapter": "train_adapter_cl"}
TASK_NUMBERS = [1, 2, 3, 4, 5]


def fingerprint_conflict(model_dir: Path, method: str, task: int, args) -> str | None:
    """比对产物内的配置指纹与本次参数；返回冲突说明，None 表示一致。

    「目录存在」不等于「是本次参数跑出来的」——只要 seed / run_tag / 方法超参
    任一项不同，就必须重训，不得静默跳过旧产物。
    """
    meta = model_dir / "training_metadata.json"
    if not meta.is_file():
        return "缺少 training_metadata.json（无法确认配置）"
    try:
        metadata = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "training_metadata.json 损坏"
    if not isinstance(metadata, dict) or not isinstance(metadata.get("fingerprint"), dict):
        return "缺少有效实验指纹"
    stored = metadata["fingerprint"]
    params = {
        "method": method,
        "task": task,
        "seed": int(args.seed),
        "run_tag": args.run_tag or None,
        "base_model": MODEL_NAME,
        "max_steps": int(args.max_steps) if args.max_steps > 0 else None,
    }
    if method == "sequential_ft":
        params["run_tag"] = None
    if method == "ewc":
        params.update({"ewc_lambda": float(args.ewc_lambda),
                       "ewc_normalize": args.ewc_normalize})
    if method == "replay":
        params.update({"replay_capacity": args.replay_capacity,
                       "replay_ratio": args.replay_ratio, "replay_mode": args.replay_mode})
    if method == "adapter":
        params["adapter_inference_primary"] = "route"
    expected = experiment_fingerprint(**params)
    if stored == expected:
        return None
    diffs = []
    stored_params = stored.get("params") or {}
    for key in sorted(set(stored_params) | set(expected["params"])):
        if stored_params.get(key) != expected["params"].get(key):
            diffs.append(f"{key}: 产物={stored_params.get(key)} vs 本次={expected['params'].get(key)}")
    for key in ("code_version", "data_version"):
        if stored.get(key) != expected[key]:
            diffs.append(f"{key}: 产物={stored.get(key)} vs 本次={expected[key]}")
    return "；".join(diffs) if diffs else "实验指纹哈希不一致"


def model_complete(model_dir: Path, method: str) -> bool:
    if method != "adapter":
        return (model_dir / "adapter_config.json").is_file()
    manifest_path = model_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    names = manifest.get("adapters")
    adapter_dir = manifest.get("adapter_dir", "adapters")
    if (not isinstance(names, list) or not names
            or not isinstance(adapter_dir, str)
            or not all(isinstance(name, str) for name in names)):
        return False
    adapter_root = model_dir / adapter_dir
    return all((adapter_root / name / "adapter_config.json").is_file() for name in names)


def training_unit_complete(model_dir: Path, method: str, task: int, args) -> bool:
    if not model_complete(model_dir, method):
        return False
    if method == "ewc":
        state_path = state_dir_for("ewc", args) / f"ewc_state_after_task{task}.pt"
        metadata_path = model_dir / "training_metadata.json"
        if not state_path.is_file() or not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            import torch

            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except (OSError, json.JSONDecodeError, RuntimeError, TypeError, EOFError,
                pickle.UnpicklingError):
            return False
        return (
            isinstance(metadata, dict)
            and isinstance(state, dict)
            and state.get("completed_task") == task
            and state.get("fingerprint") == metadata.get("fingerprint")
        )
    if method == "replay":
        root = state_dir_for("replay", args)
        data_path = root / f"memory_after_task{task}.jsonl"
        meta_path = root / f"memory_after_task{task}_meta.json"
        if not data_path.is_file() or not meta_path.is_file():
            return False
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected = experiment_fingerprint(
            method="replay_memory",
            task=task,
            seed=int(args.seed),
            run_tag=args.run_tag or None,
            replay_capacity=args.replay_capacity,
            replay_mode=args.replay_mode,
        )
        return (
            isinstance(metadata, dict)
            and metadata.get("completed_task") == task
            and metadata.get("fingerprint") == expected
        )
    return True


def evaluation_result_matches(result_path: Path, method: str, task: int, args) -> bool:
    """只复用与本轮模型、种子、划分及完整样本数相符的评估结果。"""
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(report, dict):
        return False

    if args.mode == "smoke":
        model_dir = model_dir_for(method, task, args.seed, args.run_tag, "smoke")
        expected_samples = SMOKE_MAX_SAMPLES * task
    elif task == 1:
        model_dir = adapter_dir(1, args.seed)
        expected_samples = (120 if args.evaluate_split == "val" else 300) * task
    else:
        model_dir = model_dir_for(method, task, args.seed, args.run_tag, "full")
        expected_samples = (120 if args.evaluate_split == "val" else 300) * task
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(model_metadata, dict):
        return False

    try:
        report_model_path = Path(report["model_path"]).resolve()
        predictions_path = Path(report["predictions_csv"])
        if not predictions_path.is_absolute():
            predictions_path = result_path.parent / predictions_path
        predictions_path = predictions_path.resolve()
    except (KeyError, OSError, TypeError):
        return False

    return (
        report.get("method") == method
        and report.get("trained_through_task") == task
        and report.get("model_seed") == int(args.seed)
        and report.get("evaluated_split") == args.evaluate_split
        and str(report_model_path).casefold() == str(model_dir.resolve()).casefold()
        and report.get("model_fingerprint") == model_metadata.get("fingerprint")
        and report.get("total_samples") == expected_samples
        and (method != "adapter" or report.get("adapter_inference") == args.adapter_inference)
        and predictions_path.is_file()
    )


def run_one(label: str, cmd: list, log_file: Path) -> dict:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now():%H:%M:%S}] {label}", flush=True)
    print(f"    $ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.time()
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(str(c) for c in cmd)}\n\n")
        log.flush()
        proc = subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT),
                              stdout=log, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace")
    minutes = round((time.time() - started) / 60, 1)
    if proc.returncode != 0:
        try:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        guard_markers = ("按项目规范禁止在本地显卡", "需要 CUDA 云端 GPU", ALLOW_ENV_MARKER)
        if any(marker in log_text for marker in guard_markers):
            print("    [中止] 子进程被 GPU 守卫拒绝（本地显卡），终止整个矩阵，不再重试。",
                  flush=True)
            raise SystemExit(3)
    state = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"    -> {state}  {minutes} min  -> {log_file.name}", flush=True)
    return {"label": label, "cmd": " ".join(str(c) for c in cmd),
            "returncode": proc.returncode, "minutes": minutes, "log": str(log_file)}


def model_dir_for(method: str, task: int, seed: int, tag: str, mode: str = "full") -> Path:
    """与训练侧命名规则一致：{method}_task{N}[_{tag}][_s{seed}]；冒烟模式指向临时产物。"""
    if mode == "smoke":
        return SMOKE_CONT / f"{method}_task{task}"
    if method == "sequential_ft":
        return adapter_dir(task, seed)
    suffix = f"_{tag.strip('_')}" if tag else ""
    if int(seed) != DEFAULT_SEED:
        suffix += f"_s{int(seed)}"
    return MODEL_ROOT / model_group(method, task) / f"{method}_task{task}{suffix}"


def state_dir_for(kind: str, args) -> Path:
    """EWC / Replay 的状态目录（与训练侧命名一致）。"""
    name = "ewc" if kind == "ewc" else "replay_memory"
    if args.run_tag:
        name += f"_{args.run_tag.strip('_')}"
    if int(args.seed) != DEFAULT_SEED:
        name += f"_s{int(args.seed)}"
    root = SMOKE_ROOT / "continual_state" if args.mode == "smoke" else STATE_ROOT
    return root / name


def conflict_targets(method: str, task: int, args) -> list:
    """该单元可能与既有产物冲突的路径（用于错误修复）。"""
    targets = []
    if method == "sequential_ft":
        targets.append(adapter_dir(task, args.seed))
    else:
        targets.append(model_dir_for(method, task, args.seed, args.run_tag, args.mode))
        if method == "ewc":
            targets.append(state_dir_for("ewc", args) / f"ewc_state_after_task{task}.pt")
    return targets


def train_command(method: str, task: int, args) -> list:
    if method == "sequential_ft":
        cmd = [sys.executable, str(TRAIN_DIR / f"train_task{task}.py"), "--seed", str(args.seed)]
    else:
        cmd = [sys.executable, str(TRAIN_DIR / f"{CL_METHODS[method]}.py"),
               "--task", str(task), "--seed", str(args.seed)]
        if args.run_tag:
            cmd += ["--run-tag", args.run_tag]
        if method == "ewc":
            cmd += ["--ewc-lambda", str(args.ewc_lambda), "--ewc-normalize", args.ewc_normalize]
        if method == "replay":
            cmd += ["--replay-capacity", str(args.replay_capacity),
                    "--replay-ratio", str(args.replay_ratio), "--replay-mode", args.replay_mode]
    if args.max_steps and args.max_steps > 0:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.resume:
        cmd.append("--resume")
    if args.mode == "smoke":
        cmd.append("--smoke")
    return cmd


def eval_command(method: str, task: int, args) -> list:
    if args.mode == "smoke":
        model_path = model_dir_for(method, task, args.seed, args.run_tag, "smoke")
        output_root = SMOKE_EVAL
    elif task == 1:
        model_path = adapter_dir(1, args.seed)
        output_root = Path(args.results_dir) if args.results_dir else RESULT_ROOT
    else:
        model_path = model_dir_for(method, task, args.seed, args.run_tag, "full")
        output_root = Path(args.results_dir) if args.results_dir else RESULT_ROOT
    cmd = [sys.executable, str(EVAL_DIR / f"evaluate_task{task}_cumulative.py"),
           "--method", method, "--model-path", str(model_path),
           "--model-seed", str(args.seed), "--split", args.evaluate_split]
    if method == "adapter":
        cmd += ["--adapter-inference", args.adapter_inference]
        if args.allow_nonprimary_adapter_inference:
            cmd.append("--allow-nonprimary-adapter-inference")
    if args.mode == "smoke":
        cmd += ["--limit", str(SMOKE_MAX_SAMPLES)]
    cmd += ["--output-dir", str(output_root)]
    return cmd


def result_path_for(method: str, task: int, args) -> Path:
    if args.mode == "smoke":
        root = SMOKE_EVAL
    else:
        root = Path(args.results_dir) if args.results_dir else RESULT_ROOT
    suffix = "" if args.evaluate_split == "test" else f"_{args.evaluate_split}"
    return root / f"temporal_{method}_task{task}_cumulative_test{suffix}.json"


def classify_failure(log_file: Path) -> str:
    """从日志尾部判断失败类型，决定修复动作。"""
    text = ""
    if log_file.is_file():
        text = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
    if "FileExistsError" in text or "拒绝覆盖" in text:
        return "conflict"
    if "out of memory" in text.lower() or "OutOfMemoryError" in text:
        return "oom"
    if "KeyboardInterrupt" in text:
        return "interrupted"
    if "FileNotFoundError" in text or "缺少前阶段" in text or "需要先完成" in text:
        return "missing_dependency"
    return "unknown"


def repair_conflicts(method: str, task: int, args) -> str:
    """按冲突策略处理既有产物：archive（默认，备份不删）/ clean / skip / fail。"""
    if args.conflict_policy == "fail":
        return "fail"
    for target in conflict_targets(method, task, args):
        if not target.exists():
            continue
        if args.conflict_policy == "skip":
            return "skip"
        if args.conflict_policy == "clean":
            shutil.rmtree(target) if target.is_dir() else target.unlink()
            print(f"    [修复] 已清理冲突产物: {target}")
        else:
            backup = target.with_name(f"{target.name}_backup_{args.stamp}")
            suffix = 1
            while backup.exists():
                backup = target.with_name(f"{target.name}_backup_{args.stamp}_{suffix}")
                suffix += 1
            shutil.move(str(target), str(backup))
            print(f"    [修复] 冲突产物已备份: {target} -> {backup.name}")
    return "retry"


def run_with_retry(label: str, cmd: list, log_dir: Path, kind: str,
                   method: str, task: int, args) -> dict:
    """执行子命令；失败后按错误类型自动修复并重试（默认重试 1 次）。"""
    attempts = max(1, int(args.retries) + 1)
    record: dict = {}
    for attempt in range(1, attempts + 1):
        log = log_dir / (f"{label}.log" if attempt == 1 else f"{label}.retry{attempt}.log")
        record = run_one(label, cmd, log)
        record["attempts"] = attempt
        if record["returncode"] == 0:
            return record
        reason = classify_failure(log)
        record["reason"] = reason
        print(f"    [失败] {label} 第 {attempt}/{attempts} 次，原因: {reason}", flush=True)
        if attempt >= attempts:
            if reason == "oom":
                print("      显存不足：需减小批量或换卡；本单元标记失败后继续其它单元")
            elif reason == "interrupted":
                print("      进程被中断：可用 --resume 从 checkpoint 续训")
            elif reason == "missing_dependency":
                print("      依赖缺失：检查上一阶段模型/状态是否已生成")
            return record
        if reason == "conflict":
            action = repair_conflicts(method, task, args)
            if action in {"skip", "fail"}:
                record["skipped"] = f"repair-{action}"
                return record
            print("      [修复] 冲突已处理，重试本单元")
        else:
            print("      [修复] 该错误暂无自动修复手段，直接重试")
        if args.retry_wait > 0:
            time.sleep(args.retry_wait)
    return record


def current_accuracy(result_path: Path, task: int):
    if not result_path.is_file():
        return None
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(report, dict):
        return None
    metrics_by_task = report.get("metrics_by_test_task")
    if not isinstance(metrics_by_task, dict):
        return None
    metrics = metrics_by_task.get(TASKS[task]["name"])
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("acc")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="全量编排：4 方法 × 5 阶段训练 + 60 组评估")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--tasks", nargs="+", type=int, default=TASK_NUMBERS, choices=TASK_NUMBERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false")
    parser.add_argument("--evaluate-split", choices=("test", "val"), default="test")
    parser.add_argument("--adapter-inference", choices=("route", "weighted", "simultaneous"),
                        default="route")
    parser.add_argument("--allow-nonprimary-adapter-inference", action="store_true",
                        help="允许 Adapter-CL 的 weighted/simultaneous 扩展对照（结果不可用于正式排名）")
    parser.add_argument("--gate-current-acc", type=float, default=0.0)
    parser.add_argument("--ewc-lambda", type=float, default=EWC_LAMBDA_DEFAULT)
    parser.add_argument("--ewc-normalize", choices=("none", "param", "task"), default="param")
    parser.add_argument("--replay-capacity", type=int, default=REPLAY_MEMORY_CAPACITY)
    parser.add_argument("--replay-ratio", type=float, default=REPLAY_RATIO_DEFAULT)
    parser.add_argument("--replay-mode", choices=("reservoir", "class_balanced"),
                        default=REPLAY_MODE_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--backup-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=MODES, default="full",
                        help="smoke=冒烟（每模型≤2条，只验上下调用与流程）；"
                             "full=正式训练+测试；train=单跑训练；eval=单跑测试")
    parser.add_argument("--retries", type=int, default=1, help="失败后自动重试次数（默认 1）")
    parser.add_argument("--retry-wait", type=int, default=5, help="重试前等待秒数")
    parser.add_argument("--conflict-policy", choices=("fail", "archive", "clean", "skip"),
                        default="archive",
                        help="产物冲突修复策略：archive=备份改名（默认）/clean=清理/skip=跳过/fail=失败")
    parser.add_argument("--state", default="", help="状态输入：上次 summary_*.json，已成功单元自动跳过")
    parser.add_argument("--single", default="", help="单跑一个单元，如 ewc:3（只跑该方法该阶段）")
    args = parser.parse_args()
    args.methods = list(dict.fromkeys(args.methods))
    if "replay" in args.methods and (
        args.replay_capacity < len(LABELS) or args.replay_ratio <= 0
    ):
        print(f"[错误] Replay 需要 capacity >= {len(LABELS)} 且 replay-ratio > 0。")
        return 2
    if "ewc" in args.methods and args.ewc_lambda < 0:
        print("[错误] EWC lambda 不能为负数。")
        return 2
    if ("adapter" in args.methods and args.adapter_inference != "route"
            and not args.allow_nonprimary_adapter_inference):
        print("[错误] weighted/simultaneous 仅作为扩展对照；显式添加 "
              "--allow-nonprimary-adapter-inference 才会运行，且不得用于方法排名。")
        return 2
    if args.gate_current_acc > 0 and args.evaluate_split != "val":
        print("[错误] 门禁只允许基于验证集（--evaluate-split val）。"
              "用测试集准确率决定是否继续训练/重跑属于测试泄漏，会污染最终报告。")
        return 2
    if args.results_dir:
        args.results_dir = str(Path(args.results_dir).expanduser().resolve())

    # 编排器自身也必须在入口拦截本地 GPU：只靠子进程守卫时，失败会被 run_with_retry
    # 当成普通错误重试并继续下一个单元，看起来像「守卫没生效」且会空转整张矩阵。
    if not args.dry_run:  # --dry-run 只打印计划、不加载模型，允许本地预览
        try:
            assert_cloud_gpu_allowed("全矩阵编排（run_full_matrix）")
        except RuntimeError as error:
            print(f"[拒绝] {error}")
            return 3

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.stamp = stamp
    log_dir = LOG_ROOT / stamp
    results_root = SMOKE_EVAL if args.mode == "smoke" else (
        Path(args.results_dir).expanduser().resolve() if args.results_dir else RESULT_ROOT
    )

    done_labels: set[str] = set()
    if args.state:
        state_path = Path(args.state)
        if not state_path.is_file():
            print(f"[错误] 状态文件不存在: {state_path}")
            return 2
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        for item in previous.get("records", []):
            if item.get("returncode") == 0 and item.get("label"):
                done_labels.add(str(item["label"]))
        print(f"[状态] 已从 {state_path.name} 载入 {len(done_labels)} 个成功单元（自动跳过）")

    execution_methods = (
        (["sequential_ft"] if "sequential_ft" in args.methods else [])
        + [method for method in args.methods if method != "sequential_ft"]
    )
    train_plan = []
    for method in execution_methods:
        for task in TASK_NUMBERS:
            if task not in args.tasks or (task == 1 and method != "sequential_ft"):
                continue
            train_plan.append((f"train_{method}_task{task}", method, task,
                               model_dir_for(method, task, args.seed, args.run_tag),
                               train_command(method, task, args)))
    eval_plan = []
    for method in execution_methods:
        for task in TASK_NUMBERS:
            if task not in args.tasks:
                continue
            eval_plan.append((f"eval_{method}_task{task}", method, task,
                              result_path_for(method, task, args),
                              eval_command(method, task, args)))

    if args.mode == "train":
        eval_plan = []
    elif args.mode == "eval":
        train_plan = []
    elif args.mode == "smoke":
        # 冒烟只验链路：E1 的临时产物训练结束即删除，故不评估；E2/E3/E4 用冒烟产物评估
        eval_plan = [i for i in eval_plan if i[1] != "sequential_ft" and i[2] >= 2]
    if not args.evaluate:
        eval_plan = []
    if args.single:
        key_method, key_task = args.single.split(":")
        key_task = int(key_task)
        train_plan = [i for i in train_plan if i[1] == key_method and i[2] == key_task]
        eval_plan = [i for i in eval_plan if i[1] == key_method and i[2] == key_task]

    if args.evaluate and eval_plan and args.evaluate_split == "val":
        missing_val = [task_file(task, "val") for task in range(1, max(item[2] for item in eval_plan) + 1)
                       if not task_file(task, "val").is_file()]
        if missing_val:
            print("[错误] 验证集文件不完整，尚未启动任何训练/评估：")
            for path in missing_val:
                print(f"  缺少 {path}")
            return 2
    if args.backup_results and results_root.exists():
        backup = results_root.with_name(f"{results_root.name}_backup_{stamp}")
        shutil.move(str(results_root), str(backup))
        print(f"既有评估结果已备份到: {backup}")

    test_groups = sum(item[2] for item in eval_plan)
    print("=" * 78)
    print(f"矩阵计划（mode={args.mode}, seed={args.seed}, run_tag={args.run_tag or '-'}）")
    print(f"  训练单元 {len(train_plan)} 个（T1 只训练一次，四方法共享）")
    print(f"  评估调用 {len(eval_plan)} 次 → 覆盖测试组 {test_groups} 组"
          f"（每个评估单元测 T1..T_i）")
    print(f"  重试 {args.retries} 次，冲突策略 {args.conflict_policy}"
          + (f"，状态输入 {Path(args.state).name}" if args.state else "")
          + (f"，单跑 {args.single}" if args.single else ""))
    print("=" * 78)
    for label, _, _, _, cmd in train_plan:
        print(f"  {label:<28} {' '.join(str(c) for c in cmd)}")
    if args.dry_run:
        print("\n[dry-run] 仅打印计划，不执行。")
        return 0

    shared_t1 = adapter_dir(1, args.seed)
    t1_in_plan = any(item[1] == "sequential_ft" and item[2] == 1 for item in train_plan)
    if not t1_in_plan and not (shared_t1 / "adapter_config.json").is_file():
        print(f"[错误] 共享的 T1 基线不存在: {shared_t1}\n"
              f"       本次计划不含 T1 训练，请先训练 T1（sequential_ft Task1）再运行本编排；"
              f"或改用 --tasks 1 2 3 4 5 让本编排一并重训 T1。")
        return 2
    if t1_in_plan:
        print(f"注意：本轮将重训共享 T1 基线（{shared_t1.name}），"
              f"旧目录若存在会被训练脚本拒绝覆盖，请先备份或移走。")
    if args.mode == "smoke":
        print("提示：冒烟模式每个训练最多 2 条样本、只验证上下调用与流程；"
              "E1 的冒烟产物训练结束即删除，因此 T2 及以后阶段以「真实 T1 基线」为父模型"
              "（需已存在）。冒烟评估结果写在 artifacts/smoke_tests/eval，不污染正式结果。")

    records = []
    stopped = set()
    shared_t1_ready = (shared_t1 / "adapter_config.json").is_file() if not t1_in_plan else False
    for method in execution_methods:
        if method != "sequential_ft" and not shared_t1_ready:
            print(f"[依赖失败] 共享 T1 基线未成功训练，停止 {method} 矩阵")
            first_task = next((task for task in TASK_NUMBERS if task in args.tasks), None)
            if first_task is not None:
                records.append({
                    "label": f"train_{method}_task{first_task}",
                    "kind": "train",
                    "method": method,
                    "task": first_task,
                    "returncode": 1,
                    "reason": "missing_shared_t1",
                })
            stopped.add(method)
            continue
        print(f"\n===== 方法 {method} =====")
        for task in TASK_NUMBERS:
            if task not in args.tasks:
                continue
            if method in stopped:
                print(f"[门禁] {method} 已中断，跳过 T{task}")
                continue
            entry = next((i for i in train_plan if i[1] == method and i[2] == task), None)
            if entry is not None:
                label, _, _, model_dir, cmd = entry
                exists = training_unit_complete(model_dir, method, task, args)
                conflict = fingerprint_conflict(model_dir, method, task, args) if exists else None
                if label in done_labels and exists and not conflict:
                    print(f"[state] {label}：上次已成功且产物指纹一致，跳过训练")
                    records.append({"label": label, "kind": "train", "returncode": 0,
                                    "skipped": "state-done"})
                elif args.skip_existing and exists and not conflict:
                    print(f"[skip] {label}：模型已存在且实验指纹一致 {model_dir.name}")
                    records.append({"label": label, "kind": "train", "returncode": 0,
                                    "skipped": "model-exists"})
                else:
                    if label in done_labels:
                        print(f"[state] {label}：旧成功记录与当前产物不匹配，将重新训练")
                    if conflict:
                        print(f"[冲突] {label}：配置指纹不一致（{conflict}）→ 强制重训，不复用旧产物")
                    record = run_with_retry(label, cmd, log_dir, "train", method, task, args)
                    record.update({"kind": "train", "method": method, "task": task,
                                   "model_dir": str(model_dir)})
                    records.append(record)
                    if record["returncode"] != 0:
                        stopped.add(method)
                        if method == "sequential_ft" and task == 1:
                            shared_t1_ready = False
                        continue
                if method == "sequential_ft" and task == 1:
                    shared_t1_ready = model_complete(shared_t1, "sequential_ft")
            entry = next((i for i in eval_plan if i[1] == method and i[2] == task), None)
            if entry is None or not args.evaluate:
                continue
            label, _, _, result_path, cmd = entry
            if label in done_labels:
                if evaluation_result_matches(result_path, method, task, args):
                    print(f"[state] {label}：上次成功结果与当前模型/划分一致，跳过")
                    records.append({"label": label, "kind": "eval", "method": method,
                                    "task": task, "returncode": 0, "skipped": "state-done",
                                    "result": str(result_path)})
                    continue
                print(f"[state] {label}：上次结果与当前模型/划分不符，将重新评估")
            if args.skip_existing and result_path.is_file():
                if evaluation_result_matches(result_path, method, task, args):
                    print(f"[skip] {label}：评估结果与当前模型/划分一致")
                    records.append({"label": label, "kind": "eval", "method": method,
                                    "task": task, "returncode": 0, "skipped": "result-exists",
                                    "result": str(result_path)})
                    continue
                print(f"[冲突] {label}：已有评估结果与当前模型或划分不符，将重新评估")
            record = run_with_retry(label, cmd, log_dir, "eval", method, task, args)
            record.update({"kind": "eval", "method": method, "task": task,
                           "result": str(result_path)})
            records.append(record)
            if record["returncode"] != 0:
                stopped.add(method)
                continue
            if args.gate_current_acc > 0:
                acc = current_accuracy(result_path, task)
                if acc is None:
                    print(f"[错误] {method} T{task} 验证结果缺少有效准确率，停止该方法后续阶段")
                    record["returncode"] = 1
                    record["reason"] = "missing_validation_accuracy"
                    stopped.add(method)
                elif acc < args.gate_current_acc:
                    print(f"[门禁] {method} T{task} 准确率 {acc:.4f} < "
                          f"{args.gate_current_acc:.4f}，中断该方法后续阶段")
                    stopped.add(method)

    summary_path = LOG_ROOT / f"summary_{stamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "stamp": stamp, "seed": args.seed, "run_tag": args.run_tag or None,
        "methods": args.methods, "tasks": args.tasks,
        "train_units": len(train_plan), "eval_units": len(eval_plan),
        "test_groups": test_groups,
        "evaluate_split": args.evaluate_split,
        "adapter_inference": args.adapter_inference if "adapter" in args.methods else None,
        "allow_nonprimary_adapter_inference": args.allow_nonprimary_adapter_inference,
        "ewc": {"lambda": args.ewc_lambda, "normalize": args.ewc_normalize},
        "replay": {"capacity": args.replay_capacity, "ratio": args.replay_ratio,
                   "mode": args.replay_mode},
        "max_steps": args.max_steps or None,
        "gate_current_acc": args.gate_current_acc or None,
        "stopped_methods": sorted(stopped), "records": records,
        "mode": args.mode, "retries": int(args.retries),
        "retry_wait": int(args.retry_wait), "conflict_policy": args.conflict_policy,
        "state_input": args.state or None, "single": args.single or None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_train = sum(1 for r in records if r.get("kind") == "train" and r.get("returncode") == 0)
    ok_eval = sum(1 for r in records if r.get("kind") == "eval" and r.get("returncode") == 0)
    fail = [r for r in records if r.get("returncode") not in (0, None)]
    print("\n" + "=" * 78)
    print(f"训练成功 {ok_train}/{len(train_plan)}  评估成功 {ok_eval}/{len(eval_plan)}")
    if stopped:
        print(f"被门禁/失败中断的方法: {sorted(stopped)}")
    for item in fail:
        print(f"  失败: {item['label']}  rc={item.get('returncode')}  log={item.get('log')}")
    print(f"台账: {summary_path}")
    print("=" * 78)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
