# -*- coding: utf-8 -*-
"""G3-SSL 公共路径、加权 Trainer 与 Teacher 概率推理工具。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
G1G2_SCRIPTS = PROJECT_ROOT / "06_generation_scale_benchmark_G1_G2" / "scripts"
if str(G1G2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(G1G2_SCRIPTS))

import benchmark_common as C  # noqa: E402
import gpu_guard  # noqa: E402

SSL_ROOT = PROJECT_ROOT / "06_generation_scale_benchmark_G3_SSL"
SSL_CONFIG = SSL_ROOT / "configs" / "g3_ssl_matrix.json"
SSL_DATA_ROOT = SSL_ROOT / "data" / "g3_ssl"
SSL_RESULTS_ROOT = SSL_ROOT / "results"
SSL_LOG_ROOT = SSL_ROOT / "logs"
SSL_ASSETS_ROOT = PROJECT_ROOT / "assets" / "06_generation_scale_benchmark_G3_SSL"
SSL_TEACHER_ROOT = SSL_ASSETS_ROOT / "teachers"
SSL_MODEL_ROOT = SSL_ASSETS_ROOT / "models"
SSL_SMOKE_ROOT = SSL_ASSETS_ROOT / "smoke_tests"

G3_RESULTS_ROOT = PROJECT_ROOT / "06_generation_scale_benchmark_G3" / "results"

# 复用公共函数
preprocess = C.preprocess
M = C.M
read_json = C.read_json
write_json = C.write_json
now_stamp = C.now_stamp
set_global_seed = C.set_global_seed
run_paths = C.run_paths
encode_record = C.encode_record
build_prompt_ids = C.build_prompt_ids
truncate_head_tail = C.truncate_head_tail
build_peft_model = C.build_peft_model
make_training_args = C.make_training_args
compute_warmup_steps = C.compute_warmup_steps
evaluate_label_scores = C.evaluate_label_scores
code_fingerprint = C.code_fingerprint
experiment_fingerprint = C.experiment_fingerprint


def resolve_local_model_path(model_name: str) -> str:
    """解析模型目录：本地目录 → AutoDL /root/autodl-tmp/ms → HF 缓存 → 原 HF id。"""
    path = Path(model_name)
    if path.is_dir():
        return str(path)
    short_name = model_name.split("/")[-1]
    autodl = Path("/root/autodl-tmp/ms") / short_name
    if autodl.is_dir():
        return str(autodl)
    cache_dir = "models--" + model_name.replace("/", "--")
    for hub_root in (
        Path("/root/autodl-tmp/hf_models/hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ):
        snap_root = hub_root / cache_dir / "snapshots"
        if snap_root.is_dir():
            snaps = [d for d in snap_root.iterdir() if d.is_dir()]
            if snaps:
                return str(max(snaps, key=lambda d: d.stat().st_mtime))
    return model_name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


class WeightedTrainer(Trainer):
    """支持逐样本权重 (sample_weight) 的 Trainer，兼容 Encoder(2D) 与 CausalLM(3D)。"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if weights is None:
            loss = outputs.get("loss")
        else:
            if logits.dim() == 2:
                loss_vec = F.cross_entropy(logits, labels, reduction="none")
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_flat = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.size())
                valid_mask = (shift_labels != -100).float()
                loss_vec = (loss_flat * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
            weights = weights.to(loss_vec.device, dtype=loss_vec.dtype)
            loss = (loss_vec * weights).sum() / weights.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss


def make_weighted_collate_fn(pad_token_id: int):
    def collate_fn(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels, weights = [], [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            weights.append(float(f.get("sample_weight", 1.0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(weights, dtype=torch.float32),
        }

    return collate_fn
