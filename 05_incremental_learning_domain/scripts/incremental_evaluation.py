"""Cumulative evaluation and continual-learning metrics for temporal tasks."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gpu_guard import SMOKE_MAX_SAMPLES, assert_cloud_gpu_allowed
from incremental_common import (
    DEFAULT_SEED,
    EVAL_MAX_LENGTH,
    LABELS,
    MAX_LENGTH,
    MODEL_NAME,
    TASKS,
    adapter_dir,
    build_prompt,
    build_prompt_fitted,
    environment_provenance,
    read_jsonl,
    run_preflight,
    set_global_seed,
    task_file,
    truncate_head_tail,
)


RESULT_ROOT = Path(__file__).resolve().parents[1] / "results" / "domain_incremental"
# 只保留有训练实现的方法（P3：lwf / packnet 无对应训练脚本，已移除）
METHODS = ("sequential_ft", "ewc", "replay", "adapter", "cumulative")


def result_stem(method: str, task_number: int, seed: int = DEFAULT_SEED,
                run_tag: str = "", split: str = "test") -> str:
    """结果文件名（P1-9）：带上 run_tag 与 seed，避免不同实验互相覆盖。"""
    suffix = "" if split == "test" else f"_{split}"
    tag_part = f"_{str(run_tag).strip('_')}" if run_tag else ""
    seed_part = "" if int(seed) == DEFAULT_SEED else f"_s{int(seed)}"
    return f"temporal_{method}_task{task_number}{tag_part}{seed_part}_cumulative_test{suffix}"


def result_path(method: str, task_number: int, result_root: Path = RESULT_ROOT,
                split: str = "test", seed: int = DEFAULT_SEED,
                run_tag: str = "") -> Path:
    return result_root / f"{result_stem(method, task_number, seed, run_tag, split)}.json"


def _read_stage_accuracy(method: str, stage: int, result_root: Path = RESULT_ROOT,
                         split: str = "test", seed: int = DEFAULT_SEED,
                         run_tag: str = "") -> dict[int, float] | None:
    path = result_path(method, stage, result_root, split, seed, run_tag)
    if not path.exists() and method == "sequential_ft" and split == "test":
        legacy_path = result_root / f"temporal_task{stage}_cumulative_test.json"
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as source:
        report = json.load(source)
    # 身份校验（P1-9）：不同 seed / tag 的历史结果不得混入
    if report.get("model_seed") is not None and int(report["model_seed"]) != int(seed):
        return None
    if run_tag and report.get("run_tag") not in (None, run_tag):
        return None
    if report.get("evaluated_split") not in (None, split):
        return None
    metrics = report.get("metrics_by_test_task", {})
    accuracies = {}
    for task in range(1, stage + 1):
        item = metrics.get(TASKS[task]["name"])
        if item is None:
            return None
        accuracy = item.get("acc", item.get("accuracy"))
        if accuracy is None:
            return None
        accuracies[task] = float(accuracy)
    return accuracies


def _continual_metrics(
    trained_through_task: int,
    task_metrics: dict[str, dict[str, Any]],
    method: str,
    result_root: Path = RESULT_ROOT,
    split: str = "test",
    seed: int = DEFAULT_SEED,
    run_tag: str = "",
) -> dict[str, Any]:
    current_accuracies = {
        task: float(task_metrics[TASKS[task]["name"]]["acc"])
        for task in range(1, trained_through_task + 1)
    }
    accuracy_matrix: dict[str, dict[str, float]] = {}
    previous_stages: dict[int, dict[int, float]] = {}
    missing_stages = []

    for stage in range(1, trained_through_task):
        stage_accuracies = _read_stage_accuracy(method, stage, result_root, split, seed, run_tag)
        if stage_accuracies is None:
            missing_stages.append(stage)
            continue
        previous_stages[stage] = stage_accuracies
        accuracy_matrix[f"after_task{stage}"] = {
            TASKS[task]["name"]: accuracy
            for task, accuracy in stage_accuracies.items()
        }

    accuracy_matrix[f"after_task{trained_through_task}"] = {
        TASKS[task]["name"]: accuracy
        for task, accuracy in current_accuracies.items()
    }
    acc = sum(current_accuracies.values()) / trained_through_task

    if trained_through_task == 1:
        bwt = None
        forgetting = None
        status = "BWT and Forgetting require at least two task stages."
    elif missing_stages:
        bwt = None
        forgetting = None
        status = f"Missing same-method historical evaluation results for stages: {missing_stages}."
    else:
        bwt_terms = []
        forgetting_terms = []
        for task in range(1, trained_through_task):
            initial_accuracy = previous_stages[task][task]
            final_accuracy = current_accuracies[task]
            bwt_terms.append(final_accuracy - initial_accuracy)
            previous_accuracies = [
                previous_stages[stage][task]
                for stage in range(task, trained_through_task)
            ]
            forgetting_terms.append(max(previous_accuracies) - final_accuracy)
        bwt = sum(bwt_terms) / len(bwt_terms)
        forgetting = sum(forgetting_terms) / len(forgetting_terms)
        status = "Complete: all earlier stages have same-method cumulative evaluations."

    # 与汇报表口径一致的补充指标：AF（含最终阶段）、KR、CSA、Final Avg Macro-F1
    extra: dict[str, Any] = {}
    # P1-6：前序阶段缺失时，csa / af 只用「有的」阶段会静默偏乐观或偏小，必须置 None
    partial = bool(missing_stages)
    extra["metrics_partial"] = partial
    diagonal = []
    for stage in range(1, trained_through_task + 1):
        if stage == trained_through_task:
            diagonal.append(current_accuracies[stage])
        elif stage in previous_stages:
            diagonal.append(previous_stages[stage][stage])
    extra["csa"] = None if partial else (
        sum(diagonal) / len(diagonal) if diagonal else None)
    af_terms = []
    for task in range(1, trained_through_task):
        if task not in previous_stages:
            continue
        history = [previous_stages[stage][task] for stage in range(task, trained_through_task)
                   if stage in previous_stages]
        history.append(current_accuracies[task])
        af_terms.append(max(history) - current_accuracies[task])
    extra["af_including_final"] = None if partial else (
        sum(af_terms) / len(af_terms) if af_terms else None)
    if trained_through_task > 1 and all(task in previous_stages for task in range(1, trained_through_task)):
        numerator = sum(current_accuracies[task] for task in range(1, trained_through_task))
        denominator = sum(previous_stages[task][task] for task in range(1, trained_through_task))
        if denominator:
            extra["knowledge_retention"] = numerator / denominator
    macro_values = [float(task_metrics[TASKS[task]["name"]]["macro_f1"])
                    for task in range(1, trained_through_task + 1)]
    extra["final_avg_macro_f1"] = sum(macro_values) / len(macro_values)

    return {
        "ACC": acc,
        "BWT": bwt,
        "Forgetting": forgetting,
        **extra,
        "accuracy_matrix": accuracy_matrix,
        "missing_history_stages": missing_stages,
        "status": status,
        "definitions": {
            "ACC": "Mean accuracy over all test tasks seen after the current training stage.",
            "BWT": "Mean over prior tasks of current accuracy minus accuracy immediately after first learning that task.",
            "Forgetting": "Mean over prior tasks of best recorded pre-final accuracy minus current accuracy; may be negative if performance improves beyond the prior best.",
            "af_including_final": "Mean over prior tasks of (best accuracy among stages j..t, including the final stage) minus the final accuracy; always >= 0. This is the metric used in the summary workbook.",
            "knowledge_retention": "Sum of final-stage accuracy on D1..D{n-1} divided by the sum of the accuracy right after each domain was first learned.",
            "csa": "Mean diagonal accuracy across stages (current-stage adaptation ability).",
            "metric_note": "Forgetting (script, can be negative) and af_including_final (workbook, >= 0) are different definitions; always state which one is reported.",
        },
    }


def _activate_adapters_for_task(
    model: Any,
    adapter_names: list[str],
    evaluated_task: int,
    mode: str,
    weights: dict[str, float] | None,
) -> list[str]:
    """按推理策略设置本测试窗口实际激活的 adapter，并返回可写入报告的激活说明。

    route        评估窗口 Tj 时只激活 taskj（与训练时该 adapter 的激活状态一致）
    weighted     用 add_weighted_adapter 线性组合（权重由验证集网格搜索给出）
    simultaneous 同时等权激活全部 adapter（与训练不一致，仅作扩展对照）
    """
    if mode == "route":
        target = f"task{evaluated_task}"
        if target not in adapter_names:
            raise ValueError(f"路由模式需要 adapter {target}，bundle 只有 {adapter_names}")
        model.set_adapter(target)
        return [target]
    if mode == "simultaneous":
        model.base_model.set_adapter(list(adapter_names))
        return list(adapter_names)
    if not weights:
        raise ValueError("weighted 模式需要权重（adapter_weights.json）")
    vector = [float(weights.get(name, 0.0)) for name in adapter_names]
    combined = "combined"
    try:
        if combined in getattr(model, "peft_config", {}):
            model.delete_adapter(combined)
        model.add_weighted_adapter(
            adapters=list(adapter_names),
            weights=vector,
            adapter_name=combined,
            combination_type="linear",
        )
    except Exception as error:
        raise RuntimeError(
            "add_weighted_adapter 失败，无法使用 weighted 推理；"
            "请改用 --adapter-inference route（与训练一致的方案）"
        ) from error
    model.set_adapter(combined)
    return [f"{name}*{weight:.3f}" for name, weight in zip(adapter_names, vector)]


def run_evaluation(
    task_number: int,
    method: str = "sequential_ft",
    model_path: Path | None = None,
    adapter_inference: str = "route",
    eval_seed: int = DEFAULT_SEED,
    model_seed: int = DEFAULT_SEED,
    split: str = "test",
    adapter_weights: dict[str, float] | None = None,
    output_dir: Path | None = None,
    frozen_baseline: bool = False,
    sample_limit: int | None = None,
    allow_nonprimary_adapter_inference: bool = False,
    run_tag: str = "",
) -> None:
    policy_mismatch: str | None = None
    import pandas as pd
    import torch
    from peft import PeftModel
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if task_number not in TASKS:
        raise ValueError(f"任务编号必须为 1 到 {len(TASKS)}，收到: {task_number}")
    if method not in METHODS:
        raise ValueError(f"未知持续学习方法 {method!r}；可选值: {', '.join(METHODS)}")
    # 守卫前置：本地卡直接退出，避免先加载数据 / 初始化 CUDA 再被拒绝
    assert_cloud_gpu_allowed(f"{method} Task{task_number} 评估")
    run_preflight(require_cuda=True)

    if frozen_baseline:
        # 冻结基线：始终用 T1 adapter 评估 T1..Ti，后续阶段不再更新任何参数
        model_path = adapter_dir(1, model_seed)
        method = "frozen_baseline"
        print("冻结基线模式：使用 T1 adapter 评估全部已见窗口（不做任何后续更新）")
    if model_path is None:
        if method == "sequential_ft":
            model_path = adapter_dir(task_number, model_seed)
        else:
            # 持续学习方法的默认模型目录：artifacts/models/{method}_task{N}
            suffix = "" if int(model_seed) == DEFAULT_SEED else f"_s{int(model_seed)}"
            group = "joint" if method.startswith("cumulative") else f"stage{task_number}"
            candidate = (Path(__file__).resolve().parents[2] / "assets" / "05_incremental_learning_domain" / "models"
                         / group / f"{method}_task{task_number}{suffix}")
            if not candidate.is_dir():
                raise ValueError(
                    f"方法 {method} 在 Task{task_number} 的默认模型目录不存在: {candidate}，"
                    "请通过 --model-path 指定该阶段训练好的模型")
            model_path = candidate
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    is_adapter_bundle = (model_path / "adapter_manifest.json").is_file()
    is_adapter = (model_path / "adapter_config.json").is_file()
    is_full_model = (model_path / "config.json").is_file()
    if not is_adapter_bundle and not is_adapter and not is_full_model:
        raise FileNotFoundError(
            "模型目录既无 adapter_manifest.json、adapter_config.json，也无 config.json，"
            f"无法加载: {model_path}"
        )
    adapter_manifest = None
    if is_adapter_bundle:
        with (model_path / "adapter_manifest.json").open("r", encoding="utf-8") as source:
            adapter_manifest = json.load(source)
        adapter_names = adapter_manifest.get("adapters")
        adapter_root = model_path / adapter_manifest.get("adapter_dir", "adapters")
        if not isinstance(adapter_names, list) or not adapter_names:
            raise ValueError(f"Adapter manifest 缺少有效 adapters 列表: {model_path}")
        missing_adapters = [name for name in adapter_names if not (adapter_root / name / "adapter_config.json").is_file()]
        if missing_adapters:
            raise FileNotFoundError(f"Adapter bundle 缺少 adapter: {missing_adapters} under {adapter_root}")

    if split not in ("test", "val"):
        raise ValueError(f"split 只能是 test 或 val，收到: {split}")
    test_sets = {}
    for previous_task in range(1, task_number + 1):
        data_path = task_file(previous_task, split)
        if not data_path.is_file():
            raise FileNotFoundError(f"缺少 {split} 数据集: {data_path}")
        test_sets[previous_task] = read_jsonl(data_path)
    # P3-10：统一从基座读 tokenizer，避免 bundle 自带 tokenizer 与基座不一致导致打分口径漂移
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )
    label_token_ids = {}
    for label in LABELS:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"标签 {label} 不是单 token: {token_ids}")
        label_token_ids[label] = token_ids[0]

    set_global_seed(eval_seed)
    if adapter_inference not in ("route", "weighted", "simultaneous"):
        raise ValueError(
            "adapter_inference 只能是 route / weighted / simultaneous，"
            f"收到: {adapter_inference}"
        )
    if (is_adapter_bundle and adapter_inference != "route"
            and not allow_nonprimary_adapter_inference):
        raise RuntimeError(
            "正式实验只允许 --adapter-inference route（评估窗口 Tj 只激活 task_j，"
            "与该 adapter 训练时的激活状态一致）。weighted / simultaneous 属扩展对照，"
            "须显式加 --allow-nonprimary-adapter-inference，且结果不得用于方法排名。"
        )
    weights_source = None
    if adapter_inference == "weighted" and not adapter_weights:
        weights_path = model_path / "adapter_weights.json"
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"weighted 模式需要 {weights_path}（由 scripts/validation/search_adapter_weights.py 生成）"
            )
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        adapter_weights = payload.get("weights")
        weights_source = payload.get("source_split", "unknown")
        if not adapter_weights:
            raise ValueError(f"{weights_path} 中没有 weights 字段")
        if weights_source not in ("val", "train_cv"):
            print(f"[警告] adapter 权重来源为 '{weights_source}'，不是验证集或训练集内部交叉验证；"
                  f"若来自测试集则该结果存在测试泄漏，不得作为正式结果")
        print(f"加载 adapter 组合权重（来源 {weights_source}）: {adapter_weights}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if is_adapter_bundle:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        first_adapter = str(adapter_names[0])
        model = PeftModel.from_pretrained(
            base_model,
            adapter_root / first_adapter,
            adapter_name=first_adapter,
            is_trainable=False,
        )
        for adapter_name in adapter_names[1:]:
            model.load_adapter(
                adapter_root / adapter_name,
                adapter_name=adapter_name,
                is_trainable=False,
            )
        # LoraModel.set_adapter 接受列表（可同时激活多个）；PeftModel.set_adapter 只接受单个名称
        model.base_model.set_adapter(adapter_names)
        print(f"已加载 adapters: {adapter_names}（实际激活策略由 --adapter-inference 决定）")
        declared = (adapter_manifest or {}).get("inference_policy_default")
        if declared is None:
            legacy = str((adapter_manifest or {}).get("inference_policy") or "")
            declared = "simultaneous" if "simultaneously" in legacy else None
        if declared and declared != adapter_inference:
            policy_mismatch = (
                f"manifest 声明默认推理策略为 {declared}，本次评估使用 {adapter_inference}；"
                f"不一致的结果仅作对照，不用于方法排名"
            )
            print(f"[警告] {policy_mismatch}")
    elif is_adapter:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    model.eval()

    all_results: list[dict[str, Any]] = []
    task_metrics: dict[str, Any] = {}
    activated_by_task: dict[str, list[str]] = {}
    trunc_total = 0
    trunc_hit = 0
    trunc_per_label: dict[str, dict[str, int]] = {}
    for evaluated_task, records in test_sets.items():
        if sample_limit is not None:
            records = records[:sample_limit]
        y_true, y_pred = [], []
        task_name = TASKS[evaluated_task]["name"]
        if is_adapter_bundle:
            active = _activate_adapters_for_task(
                model, adapter_names, evaluated_task, adapter_inference, adapter_weights
            )
            activated_by_task[task_name] = active
            print(f"[{method}] 评估 {task_name}：激活 adapter = {active}（策略 {adapter_inference}）")
        print(f"[{method}] 训练至 Task{task_number}，评估 {task_name} test ({len(records)} 条)")
        for index, record in enumerate(records):
            prompt_ids, hit = build_prompt_fitted(tokenizer, record["text"], EVAL_MAX_LENGTH)
            trunc_total += 1
            trunc_hit += int(hit)
            bucket = trunc_per_label.setdefault(str(record["label"]), {"n": 0, "truncated": 0})
            bucket["n"] += 1
            bucket["truncated"] += int(hit)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
            scores = {label: float(logits[token_id].float().item()) for label, token_id in label_token_ids.items()}
            prediction = max(scores, key=scores.get)
            gold = str(record["label"])
            y_true.append(gold)
            y_pred.append(prediction)
            all_results.append(
                {
                    "method": method,
                    "trained_through_task": task_number,
                    "test_task": evaluated_task,
                    "test_period": task_name,
                    "index": index,
                    "DocumentID": record.get("DocumentID"),
                    "ParagraphID": record.get("ParagraphID"),
                    "year": record.get("year"),
                    "gold": gold,
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "score_N": scores["N"],
                    "score_R": scores["R"],
                    "score_W": scores["W"],
                    "text": record["text"],
                }
            )
            if (index + 1) % 50 == 0:
                print(f"  {index + 1}/{len(records)}")

        report = classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        )
        acc = float(accuracy_score(y_true, y_pred))
        task_metrics[task_name] = {
            "samples": len(records),
            "acc": acc,
            "accuracy": acc,
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
            "confusion_matrix_labels": list(LABELS),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            "per_class": {
                label: {
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
                for label in LABELS
            },
            "classification_report": report,
        }

    result_root = Path(output_dir).expanduser().resolve() if output_dir else RESULT_ROOT
    continual_metrics = _continual_metrics(task_number, task_metrics, method, result_root, split,
                                           model_seed, run_tag)
    result_root.mkdir(parents=True, exist_ok=True)
    stem = result_stem(method, task_number, model_seed, run_tag, split)
    json_path = result_root / f"{stem}.json"
    csv_path = result_root / f"{stem}_predictions.csv"
    model_fingerprint = None
    meta_path = model_path / "training_metadata.json"
    if meta_path.is_file():
        try:
            model_fingerprint = (json.loads(meta_path.read_text(encoding="utf-8")) or {}).get("fingerprint")
        except json.JSONDecodeError:
            model_fingerprint = None

    # P1-10：adapter route 是「已知测试领域」的路由上限参照，BWT/Forgetting 由构造恒为 0
    route_mode = bool(is_adapter_bundle and adapter_inference == "route")
    deployment_setting = (
        "known_domain_route_oracle（评估时已知测试样本所属领域标签，按领域路由到对应 adapter；"
        "BWT/Forgetting 由构造恒为 0，不得与单模型多域方法同表排名）"
        if route_mode else
        "single_model_multi_domain（单个模型处理全部已见领域）" if not is_adapter_bundle else None)

    report = {
        "experiment": "Qwen2.5-1.5B domain-incremental continual-learning evaluation",
        "method": method,
        "trained_through_task": task_number,
            "base_model": MODEL_NAME,
        "provenance": environment_provenance(),
        "model_fingerprint": model_fingerprint,
        "adapter_weights_source": weights_source,
        "metric_scope": "label-token-restricted classification（对 N/R/W token 的 logits 取 argmax），"
                         "非自由生成后解析；非法输出被强制归入三类，故无非法输出率",
        "deployment_setting": deployment_setting,
        "route_requires_test_domain": route_mode,
        "bwt_zero_by_construction": route_mode,
        "model_path": str(model_path),
        "model_format": (
            "peft_adapter_bundle" if is_adapter_bundle else "peft_adapter" if is_adapter else "transformers_model"
        ),
        "active_adapters": adapter_names if is_adapter_bundle else None,
        "adapter_inference": adapter_inference if is_adapter_bundle else None,
        "activated_by_test_task": activated_by_task,
        "evaluated_split": split,
        "eval_seed": int(eval_seed),
        "model_seed": int(model_seed),
        "run_tag": run_tag or None,
        "truncation_length": EVAL_MAX_LENGTH,
        "truncation_policy": ("text_level（文本层头尾保留）：build_prompt_fitted()，"
                              "先扣掉 system prompt 固定开销再按 token 保留正文头 60%/尾 40%，"
                              "训练与评估同一函数/同方向/同长度"),
        "truncation": {
            "total": trunc_total,
            "truncated": trunc_hit,
            "ratio": round(trunc_hit / trunc_total, 4) if trunc_total else 0.0,
            "per_label": {
                label: {"n": bucket["n"], "truncated": bucket["truncated"],
                        "ratio": round(bucket["truncated"] / bucket["n"], 4) if bucket["n"] else 0.0}
                for label, bucket in sorted(trunc_per_label.items())
            },
        },
        "adapter_policy_declared": (adapter_manifest or {}).get("inference_policy_default"),
        "adapter_policy_mismatch": policy_mismatch,
        "adapter_inference_is_primary": (adapter_inference == "route") if is_adapter_bundle else None,
        "adapter_train_active_per_adapter": (adapter_manifest or {}).get("train_active_per_adapter"),
        "adapter_experiment_definition": (adapter_manifest or {}).get("experiment_definition"),
        "train_eval_consistency": (
            "评估窗口 Tj 激活 task_j，与该 adapter 训练时（单独激活 task_j）一致 → 训练/评估一致"
            if (is_adapter_bundle and adapter_inference == "route")
            else ("非 primary 推理策略：与训练时「只激活单个 adapter」的状态不一致，"
                  "结果仅作扩展对照，不得用于方法排名" if is_adapter_bundle else None)
        ),
        "invalid_output_policy": "forced_argmax_3way（无生成解析，无效输出率恒为 0）",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shared_baseline_task1": task_number == 1,
        "evaluated_tasks": [TASKS[number]["name"] for number in test_sets],
        "total_samples": len(all_results),
        "continual_metrics": continual_metrics,
        "metrics_by_test_task": task_metrics,
        "predictions_csv": str(csv_path),
    }
    with json_path.open("w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n累计测试结果")
    print(f"ACC={continual_metrics['ACC']:.4f}")
    print(f"BWT={continual_metrics['BWT'] if continual_metrics['BWT'] is not None else 'N/A'}")
    print(
        "Forgetting="
        f"{continual_metrics['Forgetting'] if continual_metrics['Forgetting'] is not None else 'N/A'}"
    )
    print(continual_metrics["status"])
    for test_period, metrics in task_metrics.items():
        print(
            f"{test_period}: ACC={metrics['acc']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, "
            f"Kappa={metrics['cohen_kappa']:.4f}, "
            f"N/R/W-F1={[round(metrics['per_class'][label]['f1'], 4) for label in LABELS]}"
        )
        print(f"  混淆矩阵 (labels={list(LABELS)}): {metrics['confusion_matrix']}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


def evaluation_main(task_number: int) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=f"用指定持续学习方法评估 Task{task_number} 模型在 T1 至 T{task_number} 上的表现"
    )
    parser.add_argument("--method", choices=METHODS, default="sequential_ft")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="该方法在当前训练阶段得到的 PEFT adapter 或 Transformers 完整模型目录",
    )
    parser.add_argument("--adapter-inference", choices=("route", "weighted", "simultaneous"),
                        default="route",
                        help="Adapter-CL 推理策略：route=按测试窗口路由（与训练一致，默认）；"
                             "weighted=验证集搜得的线性组合权重；simultaneous=同时激活（旧行为，不用于排名）")
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_SEED, help="评估随机种子")
    parser.add_argument("--model-seed", type=int, default=DEFAULT_SEED,
                        help="模型目录对应的训练种子（非 42 指向 *_s{seed} 目录）")
    parser.add_argument("--run-tag", default="",
                        help="实验标签：写入结果文件名与报告，避免不同 λ/配置的结果互相覆盖")
    parser.add_argument("--split", choices=("test", "val"), default="test",
                        help="评估用固定测试集(test)或验证集(val)；val 仅用于调参，结果单独命名")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="结果输出目录（默认 results/temporal）；用于冒烟或并行实验避免覆盖既有结果")
    parser.add_argument("--frozen-baseline", action="store_true",
                        help="冻结基线：只用 T1 adapter 评估 T1..Ti（判断后续更新是否有必要、"
                             "是否存在随时间下降）")
    parser.add_argument("--allow-nonprimary-adapter-inference", action="store_true",
                        help="允许 weighted / simultaneous 等非 primary 推理策略（扩展对照，不得用于方法排名）")
    parser.add_argument("--limit", type=int, default=None,
                        help=f"每个测试窗口最多评估多少条（冒烟用，上限 {SMOKE_MAX_SAMPLES}；"
                             "正式评估不限制）")
    args = parser.parse_args()
    run_evaluation(
        task_number,
        method=args.method,
        model_path=args.model_path,
        adapter_inference=args.adapter_inference,
        eval_seed=args.eval_seed,
        model_seed=args.model_seed,
        split=args.split,
        output_dir=args.output_dir,
        frozen_baseline=args.frozen_baseline,
        sample_limit=args.limit,
        allow_nonprimary_adapter_inference=args.allow_nonprimary_adapter_inference,
        run_tag=args.run_tag,
    )