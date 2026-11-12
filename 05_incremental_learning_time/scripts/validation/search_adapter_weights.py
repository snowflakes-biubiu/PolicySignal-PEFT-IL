# -*- coding: utf-8 -*-
"""Adapter-CL 组合权重搜索：在验证集上比较候选权重，写出 adapter_weights.json。

依据 scripts/修改建议.md ①：旧实验训练时只激活当前 adapter、评估时却等权同时激活全部，
评估的并非训练出来的模型。本脚本在验证集上搜索线性组合权重，供评估脚本以
--adapter-inference weighted 复现；route 模式仍是与训练一致的方案。

前置：python .../prepare/build_incremental_datasets.py temporal-val
用法：python 05_incremental_learning_time/scripts/validation/search_adapter_weights.py --task 5
输出：artifacts/models/adapter_task{N}/adapter_weights.json
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "05_incremental_learning_time" / "scripts"
MODELS = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "models"
RESULTS = PROJECT_ROOT / "05_incremental_learning_time" / "results" / "temporal"
TASK_NAMES = {
    1: "T1_1949_1978", 2: "T2_1979_1992", 3: "T3_1993_2002",
    4: "T4_2003_2012", 5: "T5_2013_2023",
}


def run(cmd: list) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT), check=True)


def candidate_weights(names: list[str], random_count: int) -> list[dict[str, float]]:
    n = len(names)
    candidates: list[dict[str, float]] = [{name: 1.0 / n for name in names}]
    for name in names:
        candidates.append({adapter: (1.0 if adapter == name else 0.0) for adapter in names})
    rng = random.Random(42)
    for _ in range(random_count):
        raw = [rng.random() + 1e-3 for _ in range(n)]
        total = sum(raw)
        candidates.append({name: value / total for name, value in zip(names, raw)})
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter-CL 组合权重搜索（验证集）")
    parser.add_argument("--task", type=int, choices=range(2, 6), default=5)
    parser.add_argument("--random", type=int, default=6, help="额外随机候选权重数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    suffix = "" if int(args.seed) == 42 else f"_s{int(args.seed)}"
    model_dir = MODELS / f"adapter_task{args.task}{suffix}"
    manifest_path = model_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 adapter bundle: {manifest_path}")
    adapter_names = [str(name) for name in
                     json.loads(manifest_path.read_text(encoding="utf-8"))["adapters"]]

    weights_path = model_dir / "adapter_weights.json"
    eval_script = SCRIPTS / "evaluate" / f"evaluate_task{args.task}_cumulative.py"
    best_weights: dict[str, float] | None = None
    best_score = -1.0
    history: list[dict] = []

    for index, weights in enumerate(candidate_weights(adapter_names, args.random)):
        weights_path.write_text(
            json.dumps({"weights": weights, "note": "search_adapter_weights.py 候选"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[候选 {index + 1}] 权重: "
              f"{ {k: round(v, 4) for k, v in weights.items()} }")
        run([sys.executable, eval_script, "--method", "adapter", "--split", "val",
             "--adapter-inference", "weighted", "--model-path", model_dir])
        val_path = RESULTS / f"temporal_adapter_task{args.task}_cumulative_test_val.json"
        if not val_path.is_file():
            print(f"[警告] 未找到验证集评估结果: {val_path}")
            continue
        metrics = (json.loads(val_path.read_text(encoding="utf-8"))
                   .get("metrics_by_test_task") or {})
        current = metrics.get(TASK_NAMES[args.task]) or {}
        macro_values = [float(m.get("macro_f1", 0.0)) for m in metrics.values()]
        score = float(current.get("macro_f1") or 0.0)
        history.append({"weights": weights, "val_macro_f1_current": score,
                        "val_macro_f1_mean": round(sum(macro_values) / max(1, len(macro_values)), 4)})
        print(f"    当前阶段 val macro-F1={score}")
        if score > best_score:
            best_score = score
            best_weights = weights

    if best_weights is None:
        print("没有可用候选，未写入权重文件")
        return 1
    weights_path.write_text(
        json.dumps({"weights": best_weights,
                    "source_split": "val",
                    "selection_metric": "val_macro_f1_current",
                    "generated_by": "search_adapter_weights.py",
                    "val_macro_f1_current": best_score,
                    "candidates": history,
                    "note": "由 search_adapter_weights.py 在验证集上选出；"
                            "route 模式为与训练一致的方案，weighted 为组合方案；"
                            "source_split=val 表示权重未使用测试集"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最优权重（val macro-F1={best_score}）已写入: {weights_path}")
    print("评估时使用: --adapter-inference weighted（或保持默认 route）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
