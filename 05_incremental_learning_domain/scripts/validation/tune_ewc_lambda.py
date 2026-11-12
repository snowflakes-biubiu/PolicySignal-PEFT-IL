# -*- coding: utf-8 -*-
"""EWC λ 网格搜索：在验证集上比较不同 λ 的表现（训练 + 评估全自动编排）。

依据 scripts/修改建议.md ②：
    旧实验 λ=5000 且惩罚为未归一化全局 sum，无法判断正则是否过强；
    归一化口径改变后 λ 的尺度完全不同，必须整组重搜，不能沿用旧值。

前置：先构建验证集  python .../prepare/build_incremental_datasets.py temporal-val
用法（云端）：
    python 05_incremental_learning_domain/scripts/validation/tune_ewc_lambda.py --task 5
    python 05_incremental_learning_domain/scripts/validation/tune_ewc_lambda.py --task 5 --skip-training
输出：results/tuning/ewc_lambda_task{N}.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "05_incremental_learning_domain" / "scripts"
TRAIN = SCRIPTS / "train" / "train_ewc.py"
MODELS = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "models"
RESULTS = PROJECT_ROOT / "05_incremental_learning_domain" / "results" / "temporal"
TUNING = PROJECT_ROOT / "05_incremental_learning_domain" / "results" / "tuning"
TASK_NAMES = {
    1: "T1_1949_1978", 2: "T2_1979_1992", 3: "T3_1993_2002",
    4: "T4_2003_2012", 5: "T5_2013_2023",
}


def run(cmd: list) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="EWC λ 网格搜索（训练 + 验证集评估）")
    parser.add_argument("--task", type=int, choices=range(2, 6), default=5)
    parser.add_argument("--lambdas", default="0,1,10,100,500,1000",
                        help="逗号分隔的 λ 取值；归一化口径变化后需整组重搜")
    parser.add_argument("--normalize", choices=("none", "param", "task"), default="param")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-training", action="store_true", help="只评估已有模型目录")
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.replace("，", ",").split(",") if x.strip()]
    TUNING.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for value in lambdas:
        tag = f"lam{value:g}_{args.normalize}"
        suffix = "" if int(args.seed) == 42 else f"_s{int(args.seed)}"
        model_dir = MODELS / f"ewc_task{args.task}_{tag}{suffix}"
        if not args.skip_training:
            run([sys.executable, TRAIN, "--task", args.task,
                 "--ewc-lambda", value, "--ewc-normalize", args.normalize,
                 "--seed", args.seed, "--run-tag", tag])
        eval_script = SCRIPTS / "evaluate" / f"evaluate_task{args.task}_cumulative.py"
        run([sys.executable, eval_script, "--method", "ewc", "--split", "val",
             "--model-path", model_dir])

        val_path = RESULTS / f"temporal_ewc_task{args.task}_cumulative_test_val.json"
        if not val_path.is_file():
            print(f"[警告] 未找到验证集评估结果: {val_path}")
            continue
        report = json.loads(val_path.read_text(encoding="utf-8"))
        metrics = report.get("metrics_by_test_task") or {}
        current = metrics.get(TASK_NAMES[args.task]) or {}
        macro_values = [float(m.get("macro_f1", 0.0)) for m in metrics.values()]
        rows.append({
            "lambda": value,
            "normalize": args.normalize,
            "seed": args.seed,
            "model_dir": model_dir.name,
            "val_acc_current": current.get("acc"),
            "val_macro_f1_current": current.get("macro_f1"),
            "val_macro_f1_mean_all_windows": round(sum(macro_values) / max(1, len(macro_values)), 4),
            "val_windows": len(metrics),
        })
        print(f"λ={value:g}: 当前阶段 val acc={current.get('acc')} "
              f"macro-F1={current.get('macro_f1')}")

    out = TUNING / f"ewc_lambda_task{args.task}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["lambda"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n汇总已写入: {out}")
    if rows:
        best = max(rows, key=lambda r: (r["val_macro_f1_current"] or 0.0))
        print(f"验证集最优 λ: {best['lambda']}（macro-F1={best['val_macro_f1_current']}）")
        print("提示：最终选择应同时看历史窗口保持与当前阶段表现，并写回 --ewc-lambda 使用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
