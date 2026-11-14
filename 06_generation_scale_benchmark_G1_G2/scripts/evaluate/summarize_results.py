# -*- coding: utf-8 -*-
"""06 期 G1/G2 结果汇总：单表 CSV + 对比报告 Markdown。

只读 `results/*/summary.json`，纯标准库实现，可在本地（无 GPU）运行。
产出：
  results/g1_g2_comparison_summary.csv   一行一个 (实验 × 任务)
  results/g1_g2_comparison_report.md     分组对比表、关键对照差值、帕累托与缺失清单
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = BENCH_ROOT / "configs" / "g1_g2_matrix.json"
DEFAULT_RESULTS = BENCH_ROOT / "results"

CSV_FIELDS = [
    "exp_id",
    "group",
    "label",
    "task",
    "model",
    "precision",
    "lora_r",
    "lora_alpha",
    "target_scope",
    "lr",
    "epochs",
    "trainable_params_pct",
    "accuracy",
    "macro_f1",
    "cohen_kappa",
    "train_time_sec",
    "peak_train_vram_gb",
    "infer_samples_per_sec",
    "peak_eval_vram_gb",
    "train_loss",
    "global_step",
    "seed",
    "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="06 期 G1/G2 结果汇总")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(extract_row(payload))
    return rows


def extract_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "exp_id": summary.get("exp_id"),
        "group": summary.get("group"),
        "label": summary.get("label"),
        "task": summary.get("task"),
        "model": summary.get("model"),
        "precision": summary.get("precision"),
        "lora_r": summary.get("lora_r"),
        "lora_alpha": summary.get("lora_alpha"),
        "target_scope": summary.get("target_scope"),
        "lr": summary.get("lr"),
        "epochs": summary.get("epochs"),
        "trainable_params_pct": summary.get("trainable_params_pct"),
        "accuracy": summary.get("accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "cohen_kappa": summary.get("cohen_kappa"),
        "train_time_sec": summary.get("train_time_sec"),
        "peak_train_vram_gb": summary.get("peak_train_vram_gb"),
        "infer_samples_per_sec": summary.get("infer_samples_per_sec"),
        "peak_eval_vram_gb": summary.get("peak_eval_vram_gb"),
        "train_loss": summary.get("train_loss"),
        "global_step": summary.get("global_step"),
        "seed": summary.get("seed"),
        "source": "run",
    }


def add_aliases(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {(row["exp_id"], row["task"]): row for row in rows}
    extras: list[dict[str, Any]] = []
    for run in config["runs"]:
        base_id = run.get("reuse")
        if not base_id:
            continue
        for task in config["tasks"]:
            if (run["id"], task) in by_key:
                continue
            base = by_key.get((base_id, task))
            if not base:
                continue
            alias = dict(base)
            alias.update(
                {
                    "exp_id": run["id"],
                    "group": run.get("group", base.get("group")),
                    "label": run.get("label", base.get("label")),
                    "model": run.get("model", base.get("model")),
                    "source": f"reuse:{base_id}",
                }
            )
            extras.append(alias)
    return rows + extras


def sort_key(row: dict[str, Any]) -> tuple[str, str]:
    exp_id = row.get("exp_id") or ""
    prefix = exp_id.split("_")[0]
    return (prefix, exp_id)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def task_table(rows: list[dict[str, Any]], task: str) -> str:
    subset = sorted([row for row in rows if row["task"] == task], key=sort_key)
    headers = [
        "编号",
        "配置",
        "可训练%",
        "Acc",
        "Macro-F1",
        "Kappa",
        "训练(s)",
        "训练峰值(GB)",
        "推理(样本/s)",
        "来源",
    ]
    body = []
    for row in subset:
        if row.get("target_scope") == "full":
            config = "全参 / lr=" + str(row.get("lr"))
        else:
            config = (
                f"{row.get('precision')} r={row.get('lora_r')}/α={row.get('lora_alpha')}"
                f"/{row.get('target_scope')}"
            )
        body.append(
            [
                str(row.get("exp_id")),
                config,
                fmt(row.get("trainable_params_pct"), 2),
                fmt(row.get("accuracy")),
                fmt(row.get("macro_f1")),
                fmt(row.get("cohen_kappa")),
                fmt(row.get("train_time_sec"), 1),
                fmt(row.get("peak_train_vram_gb"), 2),
                fmt(row.get("infer_samples_per_sec"), 1),
                str(row.get("source")),
            ]
        )
    return table(headers, body)


def comparison_table(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[str, list[str]]:
    headers = ["对照", "任务", "基线 Acc", "处理组 Acc", "ΔAcc", "基线 MF1", "处理组 MF1", "ΔMF1"]
    body = []
    missing: list[str] = []
    for comparison in config.get("comparisons", []):
        for task in config["tasks"]:
            base = next(
                (r for r in rows if r["exp_id"] == comparison["baseline"] and r["task"] == task),
                None,
            )
            treat = next(
                (r for r in rows if r["exp_id"] == comparison["treatment"] and r["task"] == task),
                None,
            )
            if not base or not treat:
                missing.append(f"{comparison['name']} / {task}")
                continue
            delta_acc = round(treat["accuracy"] - base["accuracy"], 4)
            delta_f1 = round(treat["macro_f1"] - base["macro_f1"], 4)
            body.append(
                [
                    comparison["name"],
                    task,
                    fmt(base["accuracy"]),
                    fmt(treat["accuracy"]),
                    f"{delta_acc:+.4f}",
                    fmt(base["macro_f1"]),
                    fmt(treat["macro_f1"]),
                    f"{delta_f1:+.4f}",
                ]
            )
    return table(headers, body), missing


def pareto_table(rows: list[dict[str, Any]], task: str) -> str:
    subset = [
        row
        for row in rows
        if row["task"] == task
        and row.get("group") == "G2"
        and row.get("peak_train_vram_gb") is not None
        and row.get("accuracy") is not None
    ]
    subset.sort(key=lambda r: r["peak_train_vram_gb"])
    headers = ["编号", "Acc", "训练峰值(GB)", "训练(s)", "推理(样本/s)", "帕累托"]
    body = []
    for row in subset:
        dominated = any(
            other is not row
            and other["accuracy"] >= row["accuracy"]
            and other["peak_train_vram_gb"] <= row["peak_train_vram_gb"]
            and (
                other["accuracy"] > row["accuracy"]
                or other["peak_train_vram_gb"] < row["peak_train_vram_gb"]
            )
            for other in subset
        )
        body.append(
            [
                str(row["exp_id"]),
                fmt(row["accuracy"]),
                fmt(row["peak_train_vram_gb"], 2),
                fmt(row["train_time_sec"], 1),
                fmt(row["infer_samples_per_sec"], 1),
                "被支配" if dominated else "非支配",
            ]
        )
    if not body:
        return "（暂无 G2 结果）"
    return table(headers, body)


def completeness(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[int, int, list[str]]:
    present = {(row["exp_id"], row["task"]) for row in rows}
    total = 0
    hit = 0
    missing: list[str] = []
    for run in config["runs"]:
        base_id = run.get("reuse")
        for task in config["tasks"]:
            total += 1
            key = (run["id"], task)
            base_key = (base_id, task) if base_id else None
            if key in present or (base_key and base_key in present):
                hit += 1
            else:
                missing.append(f"{run['id']} / {task}")
    return hit, total, missing


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = add_aliases(collect(results_root), config)
    if not rows:
        print(f"[空] {results_root} 下没有可汇总的 summary.json")
        return 0

    ordered = sorted(rows, key=lambda r: (sort_key(r), r.get("task") or ""))
    csv_path = out_dir / "g1_g2_comparison_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    hit, total, missing_units = completeness(rows, config)
    comp_md, comp_missing = comparison_table(rows, config)

    sections = [
        "# 06 期 G1/G2 结果对比报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 结果目录：`{results_root}`",
        f"- 矩阵完成度：**{hit}/{total}**（含复用单元）",
        f"- 已汇总记录：{len(ordered)} 条（含 G2-2 复用 G1-4 的别名行）",
        "",
        "## G1：代际锚定与 LoRA 配比重标定",
        "",
        task_table([row for row in rows if row.get("group") == "G1"], "task1_level1"),
        "",
        "### Level-2（B/Y/C/G）",
        "",
        task_table([row for row in rows if row.get("group") == "G1"], "task2_level2"),
        "",
        "## G2：参数规模阶梯与编码器对照",
        "",
        "### Level-1（W/R/N）",
        "",
        task_table([row for row in rows if row.get("group") == "G2"], "task1_level1"),
        "",
        "### Level-2（B/Y/C/G）",
        "",
        task_table([row for row in rows if row.get("group") == "G2"], "task2_level2"),
        "",
        "## 关键对照（Δ = 处理组 − 基线）",
        "",
        comp_md if comp_md else "（对比项数据不完整）",
        "",
        "## G2 效果-效率帕累托",
        "",
        "### Level-1",
        "",
        pareto_table(rows, "task1_level1"),
        "",
        "### Level-2",
        "",
        pareto_table(rows, "task2_level2"),
        "",
        "## 缺口清单",
        "",
    ]
    if missing_units:
        sections.extend([f"- 未产出：{unit}" for unit in missing_units])
    else:
        sections.append("- 矩阵已齐全")
    if comp_missing:
        sections.extend([f"- 对照缺数据：{item}" for item in comp_missing])
    sections.append("")

    md_path = out_dir / "g1_g2_comparison_report.md"
    md_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"已汇总 {len(ordered)} 条记录 -> {csv_path}")
    print(f"对比报告 -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
