# -*- coding: utf-8 -*-
"""06 期运行前预检：数据、标签单 token、模型可用性、GPU 与输出目录。

不加载模型权重、不做任何 GPU 计算，可在本地安全运行（只做条件判断与 tokenizer 编码）。

用法：
  python 06_generation_scale_benchmark/scripts/prepare/preflight_check.py --offline
  python 06_generation_scale_benchmark/scripts/prepare/preflight_check.py --strict   # 云端正式运行前
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_common as C  # noqa: E402
import gpu_guard  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []  # (level, check, detail)


def add(level: str, check: str, detail: str) -> None:
    RESULTS.append((level, check, detail))
    print(f"[{level:4s}] {check}: {detail}")


def check_data(config: dict[str, Any]) -> bool:
    ok = True
    try:
        snapshot = C.preprocess.find_snapshot()
        add("OK", "数据集快照", str(snapshot))
    except Exception as error:
        add("FAIL", "数据集快照", f"{error}")
        return False

    for task, spec in config["tasks"].items():
        for split, expected in (
            ("train", spec["train_total"]),
            ("validation", spec["val_total"]),
        ):
            try:
                rows = C.preprocess.load_split(task, split)
            except Exception as error:
                add("FAIL", f"{task}/{split}", f"读取失败: {error}")
                ok = False
                continue
            labels = sorted({r["label"] for r in rows})
            expected_labels = sorted(spec["labels"])
            if len(rows) != expected:
                add("FAIL", f"{task}/{split}", f"样本数 {len(rows)} != 期望 {expected}")
                ok = False
            elif labels != expected_labels:
                add("FAIL", f"{task}/{split}", f"标签 {labels} != 期望 {expected_labels}")
                ok = False
            else:
                add("OK", f"{task}/{split}", f"n={len(rows)} labels={labels}")

    for task in config["tasks"]:
        prompt = C.preprocess.get_system_prompt(task, "train")
        if prompt.strip():
            add("OK", f"{task} 系统提示", f"{len(prompt)} 字符")
        else:
            add("FAIL", f"{task} 系统提示", "为空")
            ok = False
    return ok


def check_tokenizers(config: dict[str, Any], offline: bool) -> bool:
    from transformers import AutoTokenizer

    ok = True
    # 单 token 约束只针对 PEFT 组的标签 token 打分；编码器用分类头，不适用
    models = sorted(
        {
            run["model"]
            for run in config["runs"]
            if run.get("runner", "peft") == "peft"
        }
    )
    for model in models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True, local_files_only=offline
            )
        except Exception as error:
            add("WARN", f"tokenizer {model}", f"不可用（{type(error).__name__}）")
            ok = False
            continue

        problems = []
        for task, spec in config["tasks"].items():
            for label in spec["labels"]:
                ids = tokenizer(label, add_special_tokens=False)["input_ids"]
                if len(ids) != 1:
                    problems.append(f"{task}:{label}→{ids}")
        if problems:
            add("FAIL", f"标签单 token {model}", "; ".join(problems))
            ok = False
        else:
            pad = tokenizer.pad_token or tokenizer.eos_token
            add("OK", f"标签单 token {model}", f"pad={pad!r}")
    return ok


def check_models(config: dict[str, Any]) -> None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for model in sorted({run["model"] for run in config["runs"]}):
        cache_name = "models--" + model.replace("/", "--")
        present = (hub / cache_name).is_dir()
        add(
            "OK" if present else "WARN",
            f"模型缓存 {model}",
            "已缓存" if present else "未缓存（云端首次运行需联网下载）",
        )


def check_gpu() -> None:
    try:
        import torch
    except Exception as error:  # pragma: no cover
        add("FAIL", "PyTorch", f"不可用: {error}")
        return
    add("OK", "PyTorch", f"{torch.__version__} cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        add("WARN", "GPU", "不可用（训练/评估必须云端 4090D）")
        return
    name = torch.cuda.get_device_name(0)
    allowed = gpu_guard.is_cloud_gpu(name)
    add(
        "OK" if allowed else "WARN",
        "GPU",
        f"{name}（{'符合云端要求' if allowed else '本地卡，禁止训练/评估'}）",
    )


def check_paths() -> bool:
    ok = True
    for label, path in (
        ("结果目录", C.RESULTS_ROOT),
        ("模型目录", C.MODEL_ROOT),
        ("日志目录", C.LOG_ROOT),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            add("OK", label, str(path))
        except Exception as error:
            add("FAIL", label, f"{path} 不可写: {error}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="06 期运行前预检")
    parser.add_argument("--config", default=str(C.CONFIG_PATH))
    parser.add_argument("--offline", action="store_true", help="tokenizer 只读本地缓存")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--strict", action="store_true", help="WARN 也视为失败（云端正式运行用）")
    parser.add_argument("--json", action="store_true", help="额外输出机器可读结果")
    args = parser.parse_args()

    config = C.load_matrix(args.config)
    print(f"=== 06 期预检：{config['title']} ===")

    data_ok = check_data(config)
    paths_ok = check_paths()
    check_gpu()
    check_models(config)
    tok_ok = True if args.skip_tokenizer else check_tokenizers(config, args.offline)

    failures = [item for item in RESULTS if item[0] == "FAIL"]
    warnings = [item for item in RESULTS if item[0] == "WARN"]
    print(
        f"\n=== 汇总：FAIL={len(failures)} WARN={len(warnings)} "
        f"OK={sum(1 for item in RESULTS if item[0] == 'OK')} ==="
    )
    if args.json:
        print(json.dumps({"results": RESULTS}, ensure_ascii=False, indent=2))

    if failures or not data_ok or not paths_ok:
        return 1
    if args.strict and (warnings or not tok_ok):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
