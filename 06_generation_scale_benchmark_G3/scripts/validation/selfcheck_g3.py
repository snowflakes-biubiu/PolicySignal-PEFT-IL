# -*- coding: utf-8 -*-
"""G3 纯逻辑自检：边界、嵌套、零泄漏、复用源，不加载模型。

用法：
  python 06_generation_scale_benchmark_G3/scripts/validation/selfcheck_g3.py
数据未构建时只检查配置与复用源，并提示先运行 build_g3_datasets.py。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import g3_common as G  # noqa: E402

CHECKS: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> None:
    CHECKS.append((bool(condition), message))
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def check_config(cfg: dict) -> None:
    for task, spec in cfg["tasks"].items():
        gold = int(spec["gold_per_class"])
        gold_tiers = [int(x) for x in spec["gold_tiers"]]
        silver_tiers = [int(x) for x in spec["silver_tiers"]]
        check(max(gold_tiers) == gold, f"{task}: 金标准边界 = {gold}（{gold_tiers}）")
        check(min(silver_tiers) > gold, f"{task}: 银标档位从 {min(silver_tiers)} 起，严格越过边界")
        check(spec["silver_column"] == ("R_1" if task == "task1_level1" else "R_2"),
              f"{task}: 银标列 = {spec['silver_column']}")
        check(
            spec["val_total"] % len(spec["labels"]) == 0,
            f"{task}: 验证集 {spec['val_total']} 按 {len(spec['labels'])} 类均分",
        )
    check(cfg["lora_r"] == 8 and cfg["lora_alpha"] == 16, "G3 统一 r=8 / α=16（G1 胜出配比）")
    check(cfg["grad_accum"] == 1 and cfg["batch_size"] == 4, "有效 batch = 4（与 G1/G2 对齐）")


def check_reuse_sources(cfg: dict) -> None:
    labels = G.preprocess.TASK_LABELS
    for model in cfg["models"]:
        exp_id = model.get("reuse_gold_exp_id")
        if not exp_id:
            check(True, f"{model['id']}: 无复用（需训练全量金标准点）")
            continue
        for task, spec in cfg["tasks"].items():
            path = G.G1G2_RESULTS_ROOT / f"{exp_id}__{task}" / "summary.json"
            if not path.is_file():
                check(False, f"{model['id']}/{task}: 复用源不存在 {path}")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            expected = int(spec["gold_per_class"]) * len(labels[task])
            actual = data.get("task_samples", {}).get("train", data.get("train_samples"))
            check(
                actual == expected,
                f"{model['id']}/{task}: 复用源 train={actual}（期望 {expected}）",
            )


def check_datasets(cfg: dict) -> None:
    manifest_path = G.G3_DATA_ROOT / "g3_manifest.json"
    if not manifest_path.is_file():
        print("[INFO] 尚未构建 G3 数据，跳过嵌套/泄漏校验")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    expected_units = sum(
        len(spec["gold_tiers"]) + len(spec["silver_tiers"]) for spec in cfg["tasks"].values()
    )
    check(len(splits) == expected_units, f"manifest 覆盖 {len(splits)}/{expected_units} 个档位")

    by_task: dict[str, list[dict]] = {}
    for item in splits:
        by_task.setdefault(item["task"], []).append(item)

    for task, spec in cfg["tasks"].items():
        items = sorted(by_task.get(task, []), key=lambda x: int(x["per_class"]))
        gold = int(spec["gold_per_class"])
        n_labels = len(spec["labels"])
        validation = {norm(r["text"]) for r in G.preprocess.load_split(task, "validation")}
        previous: set[str] | None = None
        for item in items:
            per_class = int(item["per_class"])
            path = G.PROJECT_ROOT / item["file"]
            if not path.is_file():
                check(False, f"{task} n={per_class}: 文件缺失 {path}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            check(digest == item["sha256"], f"{task} n={per_class}: sha256 与 manifest 一致")

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            texts = {norm(r["text"]) for r in records}
            silver_texts = {
                norm(r["text"]) for r in records if r["source"] != "gold"
            }
            gold_n = sum(1 for r in records if r["source"] == "gold")
            silver_n = len(records) - gold_n
            gold_overlap_n = sum(
                1
                for r in records
                if r["source"] == "gold" and norm(r["text"]) in validation
            )
            expected_overlap = int(item.get("gold_val_overlap", 0))
            check(len(records) == per_class * n_labels, f"{task} n={per_class}: 总量 {len(records)}")
            check(
                not (silver_texts & validation),
                f"{task} n={per_class}: 银标与验证集零重叠",
            )
            check(
                gold_overlap_n == expected_overlap,
                f"{task} n={per_class}: 金标准重叠 {gold_overlap_n} 与 manifest {expected_overlap} 一致",
            )
            if per_class <= gold:
                check(item["regime"] == "pure_gold" and silver_n == 0, f"{task} n={per_class}: 纯金标准段无银标")
            else:
                check(
                    item["regime"] == "gold_plus_silver" and gold_n == gold * n_labels,
                    f"{task} n={per_class}: 银标段保留全部 {gold * n_labels} 条金标准",
                )
            if previous is not None:
                check(previous <= texts, f"{task} n={per_class}: 与上一档嵌套（上一档是子集）")
            previous = texts


def main() -> int:
    cfg = G.read_json(G.G3_CONFIG)
    check_config(cfg)
    check_reuse_sources(cfg)
    check_datasets(cfg)
    failed = [message for ok, message in CHECKS if not ok]
    print(f"\n=== G3 自检 {len(CHECKS) - len(failed)}/{len(CHECKS)} 通过 ===")
    if failed:
        for message in failed:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
