"""训练前静态依赖检查：6 阶段 × 4 方法的「产出路径 / 消费路径」是否逐格对齐。

不加载模型、不写任何产物，只做四件事：
  1. 数据侧：6 域 × train 600 / test 300，N/R/W 各 200 / 100；
  2. 21 个训练单元：task{n} 的产出目录 == task{n+1} 的父目录；
  3. 方法专属状态链：EWC Fisher / Replay 缓存 / Adapter bundle 相邻阶段首尾相接；
  4. 24 次评估调用、84 个测试组：评估读的模型目录 == 训练写的目录。

用法：
    python 05_incremental_learning_domain/scripts/tests/plan_domain_matrix.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from incremental_common import (  # noqa: E402
    DEFAULT_SEED,
    TASKS,
    adapter_dir,
    read_jsonl,
    task_file,
)
from run_full_matrix import ALL_METHODS, TASK_NUMBERS, model_dir_for  # noqa: E402
import continual_training as ct  # noqa: E402

SEED = DEFAULT_SEED
TAG = ""
PROBLEMS: list[str] = []


def fail(msg: str) -> None:
    PROBLEMS.append(msg)
    print(f"  [FAIL] {msg}")


def head(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def check_data() -> None:
    head("[1] 数据侧：D1~D6 × train 600 / test 300（N/R/W 各 200 / 100）")
    total = 0
    for task, meta in TASKS.items():
        for split, per_class in (("train", 200), ("test", 100)):
            path = task_file(task, split)
            if not path.is_file():
                fail(f"缺少数据文件 {path}")
                continue
            rows = read_jsonl(path)
            labels = Counter(str(r["label"]) for r in rows)
            expect = Counter({"N": per_class, "R": per_class, "W": per_class})
            ok = labels == expect
            if not ok:
                fail(f"{path.name} 标签分布 {dict(labels)} != {dict(expect)}")
            print(f"  {'OK ' if ok else 'BAD'} D{task} {meta['name']:<20} {split:<5}"
                  f" {len(rows):>4} 条   N/R/W={labels['N']}/{labels['R']}/{labels['W']}")
            total += len(rows)
    print(f"  合计 {total} 段（目标 5,400）")
    if total != 5400:
        fail(f"总段数 {total} != 5400")


def check_train_chain() -> int:
    head("[2] 训练单元（目标 21）与链式父目录")
    units = 0
    for method in ALL_METHODS:
        print(f"\n  --- {method} ---")
        for task in TASK_NUMBERS:
            out_dir = model_dir_for(method, task, SEED, TAG)
            if task == 1 and method != "sequential_ft":
                print(f"   T{task} 复用共享 T1（不训练）")
                continue
            units += 1
            if task == 1:
                print(f"   T{task} 训练 → {out_dir.name}（基座起步）")
            elif task == 2:
                parent = adapter_dir(1, SEED)
                if parent != model_dir_for("sequential_ft", 1, SEED, TAG):
                    fail(f"{method} T2 的父目录与 sequential_ft T1 产出不一致")
                print(f"   T{task} 训练 → {out_dir.name}（父 = 共享 T1）")
            else:
                expect_parent = model_dir_for(method, task - 1, SEED, TAG)
                print(f"   T{task} 训练 → {out_dir.name}（父 = {expect_parent.name}）")
                if prev_out != expect_parent:
                    fail(f"{method} T{task} 父目录断链：{prev_out.name} != {expect_parent.name}")
            prev_out = out_dir
            if method == "ewc" and task >= 2:
                print(f"        产出 ewc_state_after_task{task}.pt")
            if method == "replay" and task >= 2:
                data_path, _ = ct._replay_memory_files(SEED, task)
                print(f"        产出 {data_path.name}")
            if method == "adapter":
                print(f"        产出 bundle：task1..task{task} 共 {task} 个 adapter")
    print(f"\n  训练单元合计 {units}（目标 21 = 6 + 3×5）")
    if units != 21:
        fail(f"训练单元 {units} != 21")
    return units


def check_eval_plan() -> tuple[int, int]:
    head("[3] 评估单元（目标 24）与测试组（目标 84）")
    calls = 0
    groups = 0
    for method in ALL_METHODS:
        print(f"\n  --- {method} ---")
        for task in TASK_NUMBERS:
            # 与 eval_command 一致：task1 一律评估共享 T1 基线（四方法同一份模型）
            model_dir = adapter_dir(1, SEED) if task == 1 else model_dir_for(method, task, SEED, TAG)
            missing = [t for t in range(1, task + 1) if not task_file(t, "test").is_file()]
            if missing:
                fail(f"{method} T{task} 缺少测试集 D{missing}")
            calls += 1
            groups += task
            print(f"   eval {method:<13} task{task} → 测 D1..D{task}"
                  f"（{task} 组 / {task * 300} 条）  模型 {model_dir.name}")
    print(f"\n  评估调用 {calls}（目标 24 = 4×6）；测试组 {groups}（目标 84 = 4×21）")
    if calls != 24:
        fail(f"评估调用 {calls} != 24")
    if groups != 84:
        fail(f"测试组 {groups} != 84")
    return calls, groups


def check_method_semantics() -> None:
    head("[4] 四方法语义与 6 阶段适配点")
    notes = [
        ("sequential_ft", "每阶段继承上一阶段 adapter，不存历史状态；T1 为四方法共享基线"),
        ("ewc", "task{n} 读 ewc_state_after_task{n-1}.pt，训练后把本阶段 Fisher 追加成 task{n} 状态"),
        ("replay", "task{n} 读 task{n-1} 的定容缓存（class_balanced，容量 300），训练后写 task{n}"),
        ("adapter", "每阶段新增 task{n} adapter 且只训当前 adapter；推理 route 按测试域激活 task{j}"),
    ]
    for method, note in notes:
        print(f"  {method:<14} {note}")
    print("\n  6 阶段新增依赖（相对 5 阶段）：")
    print("   - ewc   ：多一条 ewc_state_after_task5.pt 的消费链")
    print("   - replay：多一条 replay_memory task5 的消费链")
    print("   - adapter：bundle 从 5 个 adapter 扩到 6 个，route 需存在 task6")
    print("   - 数据  ：每阶段 train/test 均为 T1~T5 混合，年份窗口不再按阶段收窄")


def main() -> int:
    print("领域增量矩阵静态检查（不调用 GPU / 不调用 API）")
    print(f"tasks={TASK_NUMBERS}  methods={ALL_METHODS}  seed={SEED}")
    check_data()
    check_train_chain()
    check_eval_plan()
    check_method_semantics()
    head("结论")
    if PROBLEMS:
        print(f"  FAIL：{len(PROBLEMS)} 个问题")
        for msg in PROBLEMS:
            print(f"   - {msg}")
        return 1
    print("  PASS：数据、依赖链、评估计划全部自洽（21 训练单元 / 24 评估调用 / 84 测试组）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
