# -*- coding: utf-8 -*-
"""06 期（G1/G2）公共工具：路径、数据编码、模型构建、评估与可复现元信息。

设计要点：
- 数据与指标复用 `shared/common/{preprocess,metrics}.py`，与 01–05 期同源同口径；
- 标签采用**单 token 监督**（prompt 只算 -100，标签 token 算 loss），
  评估为标签 token 的受限分类，不做自由生成；
- 训练与评估共用 `truncate_head_tail()`（保留头 50% + 尾 50%），避免两侧口径漂移；
- Qwen3 默认 `enable_thinking=False`，避免 `<think>` 段破坏单 token 监督；
- 所有产物路径集中在本模块，训练脚本不写死目录。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)


def resolve_project_root() -> Path:
    """定位包含 shared/common/preprocess.py 的仓库根目录。"""
    candidates: list[Path] = []
    if os.environ.get("CAPC_PROJECT_ROOT"):
        candidates.append(Path(os.environ["CAPC_PROJECT_ROOT"]).resolve())
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    for path in candidates:
        if (path / "shared" / "common" / "preprocess.py").is_file():
            return path
    raise FileNotFoundError(
        "未找到 shared/common/preprocess.py，请在仓库根目录运行或设置 CAPC_PROJECT_ROOT。"
    )


PROJECT_ROOT = resolve_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "common"))

import metrics as M  # noqa: E402
import preprocess  # noqa: E402

BENCH_ROOT = PROJECT_ROOT / "06_generation_scale_benchmark_G1_G2"
CONFIG_PATH = BENCH_ROOT / "configs" / "g1_g2_matrix.json"
RESULTS_ROOT = BENCH_ROOT / "results"
LOG_ROOT = BENCH_ROOT / "logs"
ASSETS_ROOT = PROJECT_ROOT / "assets" / "06_generation_scale_benchmark_G1_G2"
MODEL_ROOT = ASSETS_ROOT / "models"
SMOKE_ROOT = ASSETS_ROOT / "smoke_tests"

TARGET_MODULES_MAP = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def compute_warmup_steps(
    num_examples: int, batch_size: int, grad_accum: int, epochs: int, ratio: float
) -> int:
    """transformers v5 移除了 warmup_ratio，这里显式换算成步数。"""
    per_epoch = max(1, math.ceil(num_examples / max(1, batch_size * grad_accum)))
    total = max(1, per_epoch * max(1, epochs))
    return max(1, int(total * ratio))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def code_fingerprint(paths: Iterable[str | Path]) -> str:
    """核心代码内容哈希，用于判断旧产物是否可复用。"""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def experiment_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_bytes(canonical.encode("utf-8"))[:16]


def provenance(model_name: str | None = None) -> dict[str, Any]:
    """环境与基座版本，写入 summary.json，保证跨机可比性。"""
    def _version(pkg: str) -> str | None:
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:  # pragma: no cover - 依赖缺失时才触发
            return None

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "accelerate": _version("accelerate"),
        "bitsandbytes": _version("bitsandbytes"),
        "gpu": None,
        "gpu_total_gb": None,
        "base_model": model_name,
        "base_model_revision": None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    if model_name:
        try:
            from huggingface_hub import model_info

            info["base_model_revision"] = model_info(model_name).sha
        except Exception:
            info["base_model_revision"] = "unavailable"
    return info


def dataset_info() -> dict[str, Any]:
    """记录数据集快照与各任务/划分的规模、标签分布（summery 可复现用）。"""
    snapshot = preprocess.find_snapshot()
    counts: dict[str, Any] = {}
    for task in preprocess.TASK_LABELS:
        for split in ("train", "validation"):
            rows = preprocess.load_split(task, split)
            counts[f"{task}/{split}"] = {
                "n": len(rows),
                "labels": dict(sorted(Counter(r["label"] for r in rows).items())),
            }
    return {"snapshot": snapshot.name, "snapshot_path": str(snapshot), "splits": counts}


# ---------------------------------------------------------------------------
# 数据编码
# ---------------------------------------------------------------------------
def truncate_head_tail(
    token_ids: list[int], max_length: int, head_ratio: float = 0.5
) -> tuple[list[int], bool]:
    """超长时保留头部 50% + 尾部 50%；训练与评估共用同一函数。"""
    if len(token_ids) <= max_length:
        return list(token_ids), False
    head = max(1, int(max_length * head_ratio))
    tail = max(1, max_length - head)
    return list(token_ids[:head]) + list(token_ids[-tail:]), True


def build_prompt_ids(
    tokenizer: Any, system_prompt: str, user_raw: str, enable_thinking: bool
) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_raw},
    ]
    rendered: str
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Qwen2.5 等模板不接受 enable_thinking
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def encode_record(
    record: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    max_length: int,
    enable_thinking: bool,
) -> tuple[dict[str, list[int]], bool]:
    label_str = str(record["label"]).strip()
    label_ids = tokenizer(label_str, add_special_tokens=False)["input_ids"]
    if len(label_ids) != 1:
        raise ValueError(f"标签 {label_str!r} 不是单 token: {label_ids}")
    prompt_ids = build_prompt_ids(
        tokenizer, system_prompt, record["user_raw"], enable_thinking=enable_thinking
    )
    prompt_ids, truncated = truncate_head_tail(prompt_ids, max_length - 1)
    input_ids = prompt_ids + label_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + label_ids,
    }, truncated


class PolicyDataset(Dataset):
    """模块级 Dataset（Windows spawn 下可 pickle）。"""

    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def make_collate_fn(pad_token_id: int):
    def collate_fn(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def load_base_model(model_name: str, precision: str, gradient_checkpointing: bool):
    if precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
        return model

    if precision != "bf16":
        raise ValueError(f"不支持的精度模式: {precision}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,  # transformers v5 使用 dtype
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:  # 兼容 transformers 4.x
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def build_peft_model(
    model_name: str,
    precision: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_scope: str,
    gradient_checkpointing: bool,
):
    if target_scope not in TARGET_MODULES_MAP:
        raise ValueError(f"未知 target_scope: {target_scope}")
    base_model = load_base_model(model_name, precision, gradient_checkpointing)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES_MAP[target_scope],
    )
    model = get_peft_model(base_model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    return model, trainable, total


def make_training_args(
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    precision: str,
    warmup_steps: int,
    seed: int,
    max_steps: int = -1,
    weight_decay: float = 0.01,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        bf16=(precision == "bf16"),
        fp16=(precision == "4bit" or precision == "fp16"),
        optim="paged_adamw_8bit" if precision in {"4bit", "fp16"} else "adamw_torch",
        logging_steps=1 if max_steps == 1 else 10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
    )


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_label_scores(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    system_prompt: str,
    labels: list[str],
    max_length: int,
    eval_batch_size: int,
    enable_thinking: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    """标签 token 受限分类评估：只比较候选标签 token 的 logits。"""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 释放训练残留，避免 8B 评估阶段 OOM
    label_token_ids: dict[str, int] = {}
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"评估标签 {label!r} 非单 token: {ids}")
        label_token_ids[label] = ids[0]

    encoded_prompts: list[list[int]] = []
    truncated_count = 0
    for rec in records:
        prompt_ids = build_prompt_ids(
            tokenizer, system_prompt, rec["user_raw"], enable_thinking=enable_thinking
        )
        prompt_ids, hit = truncate_head_tail(prompt_ids, max_length - 1)
        truncated_count += int(hit)
        encoded_prompts.append(prompt_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    device = getattr(model, "device", None) or next(model.parameters()).device
    candidate_ids = [label_token_ids[label] for label in labels]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    preds: list[str] = []
    rows_out: list[dict[str, Any]] = []
    for start in range(0, len(records), eval_batch_size):
        batch_prompts = encoded_prompts[start : start + eval_batch_size]
        batch_records = records[start : start + eval_batch_size]
        max_len = max(len(x) for x in batch_prompts)
        input_ids, attn_mask = [], []
        for ids in batch_prompts:
            pad_n = max_len - len(ids)
            # 右填充：与训练 collate 完全一致，避免左填充导致 RoPE 位置整体偏移
            input_ids.append(ids + [pad_id] * pad_n)
            attn_mask.append([1] * len(ids) + [0] * pad_n)

        inp = torch.tensor(input_ids, dtype=torch.long, device=device)
        msk = torch.tensor(attn_mask, dtype=torch.long, device=device)
        logits = model(input_ids=inp, attention_mask=msk).logits
        last_idx = msk.sum(dim=1) - 1  # 每条最后一个真实 token
        last_logits = logits[torch.arange(logits.size(0), device=device), last_idx]
        cand_logits = last_logits[:, candidate_ids].float()
        probs = torch.softmax(cand_logits, dim=-1).cpu().numpy()

        for rec, prob_vec in zip(batch_records, probs):
            pred_idx = int(np.argmax(prob_vec))
            pred_label = labels[pred_idx]
            preds.append(pred_label)
            row: dict[str, Any] = {
                "gold": rec["label"],
                "pred": pred_label,
                "correct": int(pred_label == rec["label"]),
                "text": rec["text"],
            }
            for label, prob in zip(labels, prob_vec):
                row[f"prob_{label}"] = round(float(prob), 6)
            rows_out.append(row)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_sec = time.perf_counter() - t0
    peak_eval_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    y_true = [r["label"] for r in records]
    report = M.evaluate(y_true, preds, labels)
    efficiency = {
        "eval_time_sec": round(eval_sec, 2),
        "infer_samples_per_sec": round(len(records) / eval_sec, 2) if eval_sec > 0 else 0.0,
        "peak_eval_vram_gb": round(peak_eval_vram_gb, 3),
        "eval_truncated_count": truncated_count,
    }
    return report, rows_out, efficiency


# ---------------------------------------------------------------------------
# 矩阵配置
# ---------------------------------------------------------------------------
def load_matrix(path: str | Path | None = None) -> dict[str, Any]:
    matrix = read_json(path or CONFIG_PATH)
    ids = [run["id"] for run in matrix["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("configs/g1_g2_matrix.json 中 run id 重复")
    for run in matrix["runs"]:
        if run.get("runner", "peft") not in {"peft", "encoder"}:
            raise ValueError(f"{run['id']}: 未知 runner {run.get('runner')}")
        if run.get("precision") not in {"bf16", "4bit", "fp16"}:
            raise ValueError(f"{run['id']}: 未知 precision {run.get('precision')}")
        if run.get("runner", "peft") == "peft":
            if run.get("target_scope") not in TARGET_MODULES_MAP:
                raise ValueError(f"{run['id']}: 未知 target_scope {run.get('target_scope')}")
    return matrix


def resolve_run(matrix: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """把矩阵级默认值合并进单条 run，得到实际生效参数。"""
    defaults = {
        key: matrix[key]
        for key in (
            "seed",
            "max_length",
            "epochs",
            "batch_size",
            "grad_accum",
            "eval_batch_size",
            "lora_dropout",
        )
        if key in matrix
    }
    resolved = {**defaults, **run}
    resolved.setdefault("runner", "peft")
    resolved.setdefault("gradient_checkpointing", False)
    resolved.setdefault("warmup_ratio", matrix.get("warmup_ratio", 0.05))
    return resolved


def run_key(exp_id: str, task: str) -> str:
    return f"{exp_id}__{task}"


def run_paths(
    exp_id: str,
    task: str,
    results_root: str | Path | None = None,
    model_root: str | Path | None = None,
) -> dict[str, Path]:
    key = run_key(exp_id, task)
    results_dir = Path(results_root or RESULTS_ROOT) / key
    model_dir = Path(model_root or MODEL_ROOT) / key
    return {
        "key": Path(key),
        "results_dir": results_dir,
        "summary": results_dir / "summary.json",
        "predictions": results_dir / "predictions.csv",
        "model_dir": model_dir,
        "adapter_dir": model_dir / "adapter",
        "encoder_dir": model_dir / "model",
        "checkpoints": model_dir / "checkpoints",
    }


def iter_units(
    matrix: dict[str, Any],
    only: list[str] | None = None,
    groups: list[str] | None = None,
    tasks: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """展开「run × task」的执行单元；reuse 单元保留但标注，不重复训练。"""
    task_names = tasks or list(matrix["tasks"])
    for raw_run in matrix["runs"]:
        if groups and raw_run.get("group") not in groups:
            continue
        if only and not any(pattern in raw_run["id"] for pattern in only):
            continue
        run = resolve_run(matrix, raw_run)
        run_tasks = raw_run.get("tasks") or task_names
        for task in run_tasks:
            if task not in matrix["tasks"]:
                raise ValueError(f"{run['id']}: config 未定义任务 {task}")
            yield {
                "run": run,
                "task": task,
                "key": run_key(run["id"], task),
                "reuse": run.get("reuse"),
            }
