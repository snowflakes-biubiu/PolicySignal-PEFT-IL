"""状态链只读体检（对应修复方案 P1-7）。

不训练、不评估、不加载模型权重，只检查 21 个训练单元与 24 个评估单元的
"产出 / 消费"是否首尾相接，以及父子产物是否配套。

检查内容：
  1. 顺序微调：`adapter_dir(k)` 存在性与 `training_metadata.json` 可解析性；
  2. EWC：`ewc_state_after_task{k}.pt` 的 `completed_task` 与模型指纹是否一致；
  3. Replay：`memory_after_task{k}.jsonl/_meta.json` 的 `completed_task` / 容量 / 模式；
  4. Adapter：bundle 内 adapter 数量是否等于 k、每个 adapter 是否齐全；
  5. 父子关系：metadata 里若记录了 `parent.fingerprint`，则与父目录实际指纹比对；
  6. 评估：24 份结果是否存在、`missing_history_stages` 是否为空。

退出码：0 = 无断链；1 = 发现断链（半完成 / 指纹不一致 / 数量不符）。
产物缺失（尚未训练）不算断链，记为 MISSING。

用法：
    python 05_incremental_learning_domain/scripts/tests/verify_chain.py
    python 05_incremental_learning_domain/scripts/tests/verify_chain.py --strict   # 缺失也算失败
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from incremental_common import DEFAULT_SEED, TASKS, adapter_dir  # noqa: E402
from incremental_evaluation import result_path  # noqa: E402
from run_full_matrix import ALL_METHODS, TASK_NUMBERS, model_dir_for  # noqa: E402
import continual_training as ct  # noqa: E402

BROKEN: list[str] = []
MISSING: list[str] = []
ARGS = None


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def mark(status: str, unit: str, detail: str) -> None:
    tag = {"OK": "OK ", "MISSING": "--", "BROKEN": "BAD"}[status]
    print(f"  {tag} {unit:<26} {detail}")
    if status == "BROKEN":
        BROKEN.append(f"{unit}: {detail}")
    elif status == "MISSING":
        MISSING.append(f"{unit}: {detail}")


def check_sequential(task: int) -> None:
    unit = f"seq T{task}"
    d = adapter_dir(task)
    if not (d / "adapter_config.json").is_file():
        return mark("MISSING", unit, "未训练")
    meta_path = d / "training_metadata.json"
    meta = read_json(meta_path)
    if not isinstance(meta, dict) or not isinstance(meta.get("fingerprint"), dict):
        return mark("BROKEN", unit, "有 adapter 但缺少有效 training_metadata.json")
    if task > 1 and meta.get("completed") is False:
        return mark("BROKEN", unit, "metadata.completed=false（半完成态）")
    mark("OK", unit, f"fingerprint {meta['fingerprint'].get('code_version', '?')}")


def check_ewc(task: int) -> None:
    unit = f"ewc T{task}"
    d = model_dir_for("ewc", task, DEFAULT_SEED, "")
    if not (d / "adapter_config.json").is_file():
        return mark("MISSING", unit, "未训练")
    meta = read_json(d / "training_metadata.json")
    state_path = ct._ewc_root(DEFAULT_SEED) / f"ewc_state_after_task{task}.pt"
    if not state_path.is_file():
        return mark("BROKEN", unit, f"缺少状态文件 {state_path.name}")
    try:
        import torch

        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        return mark("BROKEN", unit, f"状态文件不可读: {type(exc).__name__}")
    if not isinstance(state, dict) or state.get("completed_task") != task:
        return mark("BROKEN", unit, f"completed_task={getattr(state, 'get', lambda *_: None)('completed_task')}")
    if not isinstance(meta, dict) or state.get("fingerprint") != meta.get("fingerprint"):
        return mark("BROKEN", unit, "状态指纹与模型 metadata 指纹不一致")
    anchors = len(state.get("tasks") or [])
    mark("OK", unit, f"anchor 套数 {anchors}（期望 {task}）")
    if anchors != task:
        BROKEN.append(f"{unit}: anchor 套数 {anchors} != {task}")


def check_replay(task: int) -> None:
    unit = f"replay T{task}"
    d = model_dir_for("replay", task, DEFAULT_SEED, "")
    if not (d / "adapter_config.json").is_file():
        return mark("MISSING", unit, "未训练")
    data_path, meta_path = ct._replay_memory_files(DEFAULT_SEED, task)
    if not data_path.is_file() or not meta_path.is_file():
        return mark("BROKEN", unit, "缺少回放缓存文件")
    meta = read_json(meta_path) or {}
    if meta.get("completed_task") != task:
        return mark("BROKEN", unit, f"completed_task={meta.get('completed_task')}")
    mark("OK", unit, f"capacity={meta.get('capacity')} mode={meta.get('mode')}")


def check_adapter(task: int) -> None:
    unit = f"adapter T{task}"
    d = model_dir_for("adapter", task, DEFAULT_SEED, "")
    manifest_path = d / "adapter_manifest.json"
    if not manifest_path.is_file():
        return mark("MISSING", unit, "未训练")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        return mark("BROKEN", unit, "manifest 不可解析")
    names = manifest.get("adapters") or []
    root = d / manifest.get("adapter_dir", "adapters")
    missing = [n for n in names if not (root / n / "adapter_config.json").is_file()]
    if len(names) != task:
        return mark("BROKEN", unit, f"adapter 数量 {len(names)} != {task}")
    if missing:
        return mark("BROKEN", unit, f"缺少 adapter: {missing}")
    mark("OK", unit, f"bundle task1..task{task} 齐全")


def check_eval(method: str, task: int) -> None:
    unit = f"eval {method} T{task}"
    path = result_path(method, task)
    if not path.is_file():
        return mark("MISSING", unit, "未评估")
    report = read_json(path)
    if not isinstance(report, dict):
        return mark("BROKEN", unit, "结果文件不可解析")
    gaps = report.get("missing_history_stages") or []
    if gaps:
        return mark("BROKEN", unit, f"missing_history_stages={gaps}（指标不完整）")
    if not report.get("model_fingerprint"):
        return mark("BROKEN", unit, "结果缺少 model_fingerprint")
    acc = (report.get("continual_metrics") or {}).get("ACC")
    mark("OK", unit, f"ACC={acc:.4f}" if isinstance(acc, (int, float)) else "OK")


def verify_parents() -> None:
    """P1-7：校验每个单元记录的父产物指纹是否仍与实际父产物一致。"""
    print("\n[父产物指纹]")
    units: list[tuple[str, Path, Path | None]] = []
    for task in TASK_NUMBERS:
        units.append((f"seq T{task}", adapter_dir(task),
                      adapter_dir(task - 1) if task > 1 else None))
    for method in ("ewc", "replay", "adapter"):
        for task in TASK_NUMBERS:
            if task == 1:
                continue
            parent = (adapter_dir(1) if task == 2
                      else model_dir_for(method, task - 1, DEFAULT_SEED, ""))
            units.append((f"{method} T{task}", model_dir_for(method, task, DEFAULT_SEED, ""), parent))

    checked = legacy = 0
    for unit, model_dir, parent_dir in units:
        meta = read_json(model_dir / "training_metadata.json")
        if not isinstance(meta, dict):
            continue
        rec = meta.get("parent")
        if not isinstance(rec, dict):
            legacy += 1
            continue
        actual = None
        if parent_dir is not None:
            parent_meta = read_json(parent_dir / "training_metadata.json")
            actual = (parent_meta or {}).get("fingerprint")
        checked += 1
        if rec.get("fingerprint") != actual:
            mark("BROKEN", unit, "parent.fingerprint 与实际父产物不一致（父可能被重训）")
    print(f"  已校验 {checked} 个单元；无 parent 字段的旧产物 {legacy} 个")


def main() -> int:
    print("=" * 92)
    print("领域增量状态链体检（只读；21 训练单元 + 24 评估单元）")
    print("=" * 92)
    print("\n[训练单元]")
    for task in TASK_NUMBERS:
        check_sequential(task)
    for method, fn in (("ewc", check_ewc), ("replay", check_replay), ("adapter", check_adapter)):
        for task in TASK_NUMBERS:
            if task == 1:
                continue
            fn(task)

    verify_parents()

    print("\n[评估单元]")
    for method in ALL_METHODS:
        for task in TASK_NUMBERS:
            check_eval(method, task)

    print("\n" + "=" * 92)
    print(f"训练+评估单元：断链 {len(BROKEN)} 个，未开始 {len(MISSING)} 个")
    for item in BROKEN:
        print(f"  [断链] {item}")
    if BROKEN:
        return 1
    if MISSING and ARGS.strict:
        print("  [strict] 存在未开始单元，按失败处理")
        return 1
    if MISSING:
        print("  产物尚未生成（未开始训练），链路结构检查通过")
    else:
        print("  全部单元齐备且指纹一致")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="产物缺失也算失败")
    ARGS = parser.parse_args()
    sys.exit(main())
