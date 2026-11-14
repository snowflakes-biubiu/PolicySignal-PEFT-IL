# -*- coding: utf-8 -*-
"""06 期纯逻辑自检：不加载模型、不触碰 GPU，可在本地秒级运行。

覆盖：矩阵展开与复用、标签单 token 之外的可静态校验项、截断函数、
warmup 步数、指纹稳定性、GPU 守卫判定、结果汇总的别名与完成度。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "evaluate"))

import benchmark_common as C  # noqa: E402
import gpu_guard  # noqa: E402
import summarize_results as S  # noqa: E402

CHECKS: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> None:
    CHECKS.append((bool(condition), message))
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")


def test_matrix() -> None:
    matrix = C.load_matrix()
    units = list(C.iter_units(matrix))
    trainable = [unit for unit in units if not unit["reuse"]]
    reused = [unit for unit in units if unit["reuse"]]
    check(len(matrix["runs"]) == 9, f"矩阵含 9 条 run（实际 {len(matrix['runs'])}）")
    check(len(units) == 18, f"run×task 展开为 18 个单元（实际 {len(units)}）")
    check(len(trainable) == 16, f"需训练 16 个单元（实际 {len(trainable)}）")
    check(len(reused) == 2, f"复用单元 2 个（实际 {len(reused)}）")
    check(
        all(unit["reuse"] == "G1-4_qwen3_1.7b_bf16_r16" for unit in reused),
        "复用单元均指向 G1-4",
    )
    keys = [unit["key"] for unit in units]
    check(len(keys) == len(set(keys)), "执行单元 key 唯一")

    for task, spec in matrix["tasks"].items():
        labels = C.preprocess.TASK_LABELS[task]
        check(sorted(labels) == sorted(spec["labels"]), f"{task} 标签与设计表一致: {labels}")
        check(
            spec["train_per_class"] * len(labels) == spec["train_total"],
            f"{task} 训练集配额自洽: {spec['train_per_class']}×{len(labels)}={spec['train_total']}",
        )


def test_config_resolution() -> None:
    matrix = C.load_matrix()
    run = C.resolve_run(matrix, matrix["runs"][0])
    check(run["max_length"] == 768 and run["epochs"] == 3, "默认超参从矩阵级继承")
    eight_b = next(r for r in matrix["runs"] if r["id"].startswith("G2-4"))
    resolved = C.resolve_run(matrix, eight_b)
    check(
        resolved["batch_size"] == 2 and resolved["grad_accum"] == 2,
        "G2-4 覆盖批量与累积（2×2，有效 batch 4）",
    )
    check(resolved["gradient_checkpointing"] is True, "G2-4 开启梯度检查点")


def test_truncation() -> None:
    ids = list(range(10))
    kept, hit = C.truncate_head_tail(ids, 10)
    check(kept == ids and hit is False, "未超长时原样返回")
    long_ids = list(range(1000))
    kept, hit = C.truncate_head_tail(long_ids, 100)
    check(hit is True and len(kept) == 100, "超长时截断到指定长度")
    check(kept[:50] == long_ids[:50], "保留头部 50%")
    check(kept[50:] == long_ids[-50:], "保留尾部 50%")


def test_warmup_and_steps() -> None:
    matrix = C.load_matrix()
    for unit in C.iter_units(matrix):
        run = unit["run"]
        total = math.ceil(
            matrix["tasks"][unit["task"]]["train_total"]
            / (run["batch_size"] * run["grad_accum"])
        ) * run["epochs"]
        warmup = C.compute_warmup_steps(
            matrix["tasks"][unit["task"]]["train_total"],
            run["batch_size"],
            run["grad_accum"],
            run["epochs"],
            run["warmup_ratio"],
        )
        if not (1 <= warmup <= total):
            check(False, f"{unit['key']} warmup 步数异常: {warmup}/{total}")
            return
    check(True, "所有单元的 warmup 步数落在 (0, total] 内")


def test_fingerprint() -> None:
    base = {"exp_id": "X", "lr": 2e-4, "seed": 42, "smoke": False}
    same = dict(base)
    other = {**base, "lr": 1e-4}
    check(
        C.experiment_fingerprint(base) == C.experiment_fingerprint(same),
        "同配置指纹稳定",
    )
    check(
        C.experiment_fingerprint(base) != C.experiment_fingerprint(other),
        "lr 变化时指纹改变",
    )


def test_gpu_guard() -> None:
    check(gpu_guard.is_cloud_gpu("NVIDIA GeForce RTX 4090 D"), "4090 D 判定为云端卡")
    check(not gpu_guard.is_cloud_gpu("NVIDIA GeForce GTX 1660 Ti with Max-Q Design"), "1660 Ti 被拒")
    check(not gpu_guard.is_cloud_gpu("NVIDIA GeForce RTX 4090"), "非 D 版 4090 被拒（避免本地误跑）")
    check(gpu_guard.SMOKE_MAX_SAMPLES == 2, "冒烟样本上限 = 2")


def test_summary_alias() -> None:
    matrix = C.load_matrix()
    fake = {
        "exp_id": "G1-4_qwen3_1.7b_bf16_r16",
        "group": "G1",
        "label": "去量化+原秩",
        "task": "task1_level1",
        "model": "Qwen/Qwen3-1.7B",
        "precision": "bf16",
        "lora_r": 16,
        "lora_alpha": 32,
        "target_scope": "attn_mlp",
        "lr": 2e-4,
        "epochs": 3,
        "trainable_params_pct": 1.0,
        "accuracy": 0.91,
        "macro_f1": 0.91,
        "cohen_kappa": 0.86,
        "train_time_sec": 100.0,
        "peak_train_vram_gb": 5.0,
        "infer_samples_per_sec": 100.0,
        "peak_eval_vram_gb": 5.0,
        "train_loss": 0.2,
        "global_step": 450,
        "seed": 42,
        "source": "run",
    }
    rows = S.add_aliases([fake], matrix)
    alias = next(
        (row for row in rows if row["exp_id"] == "G2-2_qwen3_1.7b_bf16_r16"),
        None,
    )
    check(alias is not None, "G2-2 生成复用别名行")
    check(
        alias is not None and alias["source"].startswith("reuse:G1-4"),
        "别名行标注来源为 G1-4",
    )
    hit, total, missing = S.completeness(rows, matrix)
    check(total == 18, f"完成度分母 = 18（实际 {total}）")
    check(hit == 2, f"G1-4/task1 + 别名 G2-2/task1 时完成度 = 2（实际 {hit}）")
    check(len(missing) == total - hit, "缺口数量与完成度互补")


def main() -> int:
    test_matrix()
    test_config_resolution()
    test_truncation()
    test_warmup_and_steps()
    test_fingerprint()
    test_gpu_guard()
    test_summary_alias()
    failed = [message for ok, message in CHECKS if not ok]
    print(f"\n=== 自检 {len(CHECKS) - len(failed)}/{len(CHECKS)} 通过 ===")
    if failed:
        for message in failed:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
