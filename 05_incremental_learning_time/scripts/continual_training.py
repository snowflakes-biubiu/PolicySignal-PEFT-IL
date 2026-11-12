"""Task 2-5 training implementations for EWC, replay, task adapters and the cumulative reference group.

依据 `scripts/修改建议.md` 的修订：
1. EWC：λ 可配（--ewc-lambda），惩罚支持归一化（--ewc-normalize none|param|task）。
   λ=0 用于校验「与顺序微调一致」；**归一化后 λ 的尺度与旧值（5000）不可互换，必须重搜**。
2. Replay：回放缓存定容（--replay-capacity，默认 300），两种模式
   （--replay-mode reservoir 流式蓄水池 / class_balanced 每类等额），
   回放比例独立可配（--replay-ratio，新样本:回放样本）；从 T1 样本初始化，
   并按阶段保存带数据/代码指纹的缓存快照。
3. Adapter-CL：训练侧保持「每窗口新增 adapter、旧 adapter 冻结」；
   **推理策略由评估脚本决定**（route / weighted / simultaneous），manifest 记录三种策略，
   避免训练-推理不一致被静默使用（旧实验的 simultaneously 结果已判定不可用）。
4. 新增累积训练参考组 cumulative：每阶段用 T1..Ti 全部训练数据从基座重训，作为参考上界。
5. 全部方法支持 --seed；非默认种子写入独立目录，避免与既有实验互相覆盖。
6. 训练元数据记录 steps / tokens / examples_seen，供「等步数、等算力」对照。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

from gpu_guard import SMOKE_MAX_SAMPLES, assert_cloud_gpu_allowed
from incremental_common import (
    ASSETS_ROOT,
    BATCH_SIZE,
    DEFAULT_SEED,
    environment_provenance,
    experiment_fingerprint,
    GRADIENT_ACCUMULATION,
    INCREMENTAL_ROOT,
    LABELS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_LENGTH,
    MODEL_NAME,
    MODEL_ROOT,
    model_group,
    NUM_EPOCHS,
    PolicyDataset,
    SEED,
    SMOKE_ROOT,
    TASKS,
    adapter_dir,
    encode_record,
    read_jsonl,
    run_preflight,
    set_global_seed,
    task_file,
    _collate_batch,
)

CONTINUAL_STATE_ROOT = ASSETS_ROOT / "continual_state"
REPLAY_MEMORY_ROOT = CONTINUAL_STATE_ROOT / "replay_memory"
EWC_STATE_ROOT = CONTINUAL_STATE_ROOT / "ewc"

LEGACY_EWC_LAMBDA = 5000.0        # 旧实验(2026-11-04)使用的 λ，仅留档；归一化后不可沿用
EWC_LAMBDA_DEFAULT = 100.0        # 归一化惩罚下的默认起点，建议在 {0,1,10,100,500,1000} 重搜
EWC_NORMALIZE_DEFAULT = "param"   # none | param | task
REPLAY_MEMORY_CAPACITY = 300      # 回放缓存总容量
REPLAY_RATIO_DEFAULT = 3.0        # 新样本 : 回放样本
REPLAY_MODE_DEFAULT = "class_balanced"   # class_balanced | reservoir

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
SMOKE_MODE = False          # 由 run_continual_training(smoke=True) 开启；重定向产物到冒烟目录
SMOKE_FISHER_LIMIT = SMOKE_MAX_SAMPLES   # 冒烟模式下 Fisher 估计的样本上限（每模型最多 2 条）
METHODS = ("ewc", "replay", "adapter", "cumulative")


RUN_TAG = ""            # 由 --run-tag 设置：隔离超参搜索产生的多组实验目录/状态


def method_model_dir(method: str, task_number: int, seed: int = DEFAULT_SEED) -> Path:
    if SMOKE_MODE:
        return SMOKE_ROOT / "continual" / f"{method}_task{task_number}"
    suffix = f"_{RUN_TAG.strip('_')}" if RUN_TAG else ""
    if int(seed) != DEFAULT_SEED:
        suffix += f"_s{int(seed)}"
    return MODEL_ROOT / model_group(method, task_number) / f"{method}_task{task_number}{suffix}"


def _replay_root(seed: int = DEFAULT_SEED) -> Path:
    """回放缓存根目录（默认种子沿用历史路径，其它种子/标签独立）。"""
    if SMOKE_MODE:
        return SMOKE_ROOT / "continual_state" / "replay_memory"
    name = "replay_memory"
    if RUN_TAG:
        name += f"_{RUN_TAG.strip('_')}"
    if int(seed) != DEFAULT_SEED:
        name += f"_s{int(seed)}"
    return REPLAY_MEMORY_ROOT if name == "replay_memory" else CONTINUAL_STATE_ROOT / name


def _ewc_root(seed: int = DEFAULT_SEED) -> Path:
    if SMOKE_MODE:
        return SMOKE_ROOT / "continual_state" / "ewc"
    name = "ewc"
    if RUN_TAG:
        name += f"_{RUN_TAG.strip('_')}"
    if int(seed) != DEFAULT_SEED:
        name += f"_s{int(seed)}"
    return EWC_STATE_ROOT if name == "ewc" else CONTINUAL_STATE_ROOT / name


def _read_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "adapter_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到多 Adapter manifest: {path}")
    with path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    names = manifest.get("adapters")
    if not isinstance(names, list) or not names:
        raise ValueError(f"manifest 中没有 adapters 列表: {path}")
    return manifest


class ReplayMemory:
    """定容回放缓存。

    mode="reservoir"：对跨阶段数据流做经典蓄水池抽样（总量固定，类别分布自然）。
    mode="class_balanced"：每类等额保留（类别均衡采样消融组）。
    """

    def __init__(self, capacity: int, mode: str = REPLAY_MODE_DEFAULT) -> None:
        self.capacity = int(capacity)
        self.mode = mode
        if self.capacity < len(LABELS):
            raise ValueError(f"回放容量必须至少为类别数 {len(LABELS)}，收到: {self.capacity}")
        if self.mode not in ("reservoir", "class_balanced"):
            raise ValueError(f"未知回放模式: {self.mode}")
        self.records: list[dict[str, Any]] = []
        self.seen = 0
        self._class_records: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
        self._class_seen: Counter = Counter()

    def update(self, new_records: list[dict[str, Any]], rng: random.Random) -> None:
        """把新数据并入缓存（流式蓄水池 / 每类等额）。"""
        if not new_records:
            return
        if self.mode == "class_balanced":
            per_class = max(1, self.capacity // len(LABELS))
            for record in new_records:
                label = record.get("label")
                if label not in self._class_records:
                    continue
                self._class_seen[label] += 1
                bucket = self._class_records[label]
                if len(bucket) < per_class:
                    bucket.append(record)
                else:
                    j = rng.randrange(self._class_seen[label])
                    if j < per_class:
                        bucket[j] = record
            self.records = [r for label in LABELS for r in self._class_records[label]]
            self.seen += len(new_records)
            return
        for record in new_records:
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
            else:
                j = rng.randrange(self.seen)
                if j < self.capacity:
                    self.records[j] = record

    def sample(self, count: int, rng: random.Random) -> list[dict[str, Any]]:
        """从缓存采样 count 条（不足时有放回补齐）。"""
        count = int(count)
        if count <= 0 or not self.records:
            return []
        if self.mode == "class_balanced":
            active_labels = [label for label in LABELS if self._class_records[label]]
            if not active_labels:
                return []
            per, remainder = divmod(count, len(active_labels))
            rng.shuffle(active_labels)
            extra_labels = set(active_labels[:remainder])
            picked: list[dict[str, Any]] = []
            for label in active_labels:
                bucket = self._class_records[label]
                sample_count = per + int(label in extra_labels)
                if len(bucket) >= sample_count:
                    picked.extend(rng.sample(bucket, sample_count))
                else:
                    picked.extend(bucket[rng.randrange(len(bucket))] for _ in range(sample_count))
            return picked
        if len(self.records) >= count:
            return rng.sample(self.records, count)
        return [self.records[rng.randrange(len(self.records))] for _ in range(count)]

    def state(self) -> dict[str, Any]:
        return {"seen": self.seen, "mode": self.mode, "capacity": self.capacity,
                "size": len(self.records), "class_seen": dict(self._class_seen)}

    def load(self, records: list[dict[str, Any]], seen: int,
             class_seen: dict[str, int] | None = None) -> None:
        self.records = list(records)
        self.seen = int(seen)
        self._class_records = {label: [] for label in LABELS}
        for record in self.records:
            label = record.get("label")
            if label in self._class_records:
                self._class_records[label].append(record)
        self._class_seen = Counter(
            {label: int(value) for label, value in (class_seen or {}).items()}
        )
        for label in LABELS:
            self._class_seen.setdefault(label, len(self._class_records[label]))


def _replay_memory_files(seed: int, completed_task: int) -> tuple[Path, Path]:
    root = _replay_root(seed)
    return (
        root / f"memory_after_task{completed_task}.jsonl",
        root / f"memory_after_task{completed_task}_meta.json",
    )


def _load_replay_memory(seed: int, capacity: int, mode: str,
                        completed_task: int) -> ReplayMemory:
    data_path, meta_path = _replay_memory_files(seed, completed_task)
    if not data_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"缺少 Task{completed_task} 之后的 Replay 缓存: {data_path} / {meta_path}"
        )
    memory = ReplayMemory(capacity=capacity, mode=mode)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("completed_task", -1)) != completed_task:
        raise ValueError(f"Replay 缓存阶段不匹配: 期望 Task{completed_task}, 元数据为 {meta}")
    if int(meta.get("capacity", -1)) != capacity or meta.get("mode") != mode:
        raise ValueError(
            f"Replay 缓存配置不匹配: 缓存 capacity/mode="
            f"{meta.get('capacity')}/{meta.get('mode')}，本次为 {capacity}/{mode}"
        )
    expected_fingerprint = experiment_fingerprint(
        method="replay_memory",
        task=completed_task,
        seed=int(seed),
        run_tag=RUN_TAG or None,
        replay_capacity=capacity,
        replay_mode=mode,
    )
    if meta.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Replay 缓存实验指纹与当前数据/代码/配置不匹配: {meta_path}")
    records = read_jsonl(data_path)
    if len(records) != int(meta.get("size", -1)):
        raise ValueError(f"Replay 缓存大小与元数据不符: {data_path}")
    if len(records) > capacity:
        raise ValueError(f"Replay 缓存超过配置容量 {capacity}: {data_path}")
    memory.load(
        records,
        seen=int(meta.get("seen", 0)),
        class_seen=meta.get("class_seen"),
    )
    if mode == "class_balanced" and sum(memory._class_seen.values()) != memory.seen:
        raise ValueError(f"Replay class_seen 与总 seen 数不一致: {meta_path}")
    return memory


def _save_replay_memory(memory: ReplayMemory, seed: int, completed_task: int) -> Path:
    data_path, meta_path = _replay_memory_files(seed, completed_task)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as target:
        for record in memory.records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        **memory.state(),
        "completed_task": completed_task,
        "fingerprint": experiment_fingerprint(
            method="replay_memory",
            task=completed_task,
            seed=int(seed),
            run_tag=RUN_TAG or None,
            replay_capacity=memory.capacity,
            replay_mode=memory.mode,
        ),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return data_path


def _load_parent_adapter(base_model: Any, method: str, task_number: int,
                         seed: int = DEFAULT_SEED) -> tuple[Any, list[str]]:
    from peft import PeftModel, get_peft_model

    if method == "cumulative":
        # 累积参考组：每阶段从基座重新训练，不继承任何历史 adapter
        model = get_peft_model(base_model, _make_lora_config())
        print("累积参考组：从基座新建 LoRA，不继承历史 adapter")
        return model, ["default"]

    if task_number == 2:
        parent_dir = adapter_dir(1, seed)
        if not (parent_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Task2 需要已完成的 Task1 adapter: {parent_dir}")
        if method == "adapter":
            model = PeftModel.from_pretrained(
                base_model,
                parent_dir,
                adapter_name="task1",
                is_trainable=False,
            )
            return model, ["task1"]
        return PeftModel.from_pretrained(base_model, parent_dir, is_trainable=True), ["default"]

    parent_dir = method_model_dir(method, task_number - 1, seed)
    if method != "adapter":
        if not (parent_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"缺少前一阶段 {method} adapter: {parent_dir}")
        return PeftModel.from_pretrained(base_model, parent_dir, is_trainable=True), ["default"]

    manifest = _read_manifest(parent_dir)
    adapter_root = parent_dir / manifest.get("adapter_dir", "adapters")
    names = [str(name) for name in manifest["adapters"]]
    first_name = names[0]
    first_path = adapter_root / first_name
    model = PeftModel.from_pretrained(
        base_model,
        first_path,
        adapter_name=first_name,
        is_trainable=False,
    )
    for name in names[1:]:
        model.load_adapter(adapter_root / name, adapter_name=name, is_trainable=False)
    return model, names


def _make_lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )


def _load_quantized_base() -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    base_model.config.use_cache = False
    return prepare_model_for_kbit_training(base_model)


def _encode_records(records: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, list[int]]]:
    return [encode_record(record, tokenizer) for record in records]


def _build_dataset_class():
    """返回模块级 PolicyDataset（Windows 的 dataloader spawn 无法 pickle 局部类）。"""
    return PolicyDataset


def _estimate_fisher(
    model: Any,
    examples: list[dict[str, list[int]]],
    tokenizer: Any,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """经验 Fisher = 逐样本梯度平方的均值（严格按样本累计，不是批平均梯度的平方）。"""
    import torch

    if max_examples is not None:
        examples = examples[:max_examples]
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    }
    if not parameters:
        raise RuntimeError("没有可估计 Fisher 的可训练 LoRA 参数")

    anchors = {name: parameter.detach().float().cpu().clone() for name, parameter in parameters.items()}
    fisher = {name: torch.zeros_like(value) for name, value in anchors.items()}
    was_training = model.training
    model.eval()
    model_device = model.device
    for index, example in enumerate(examples):
        batch = _collate_batch([example], tokenizer.pad_token_id)
        batch = {key: value.to(model_device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = model(**batch).loss
            loss.backward()
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                fisher[name].add_(parameter.grad.detach().float().cpu().square())
        if (index + 1) % 100 == 0:
            print(f"  Fisher: {index + 1}/{len(examples)}")

    divisor = max(1, len(examples))
    for value in fisher.values():
        value.div_(divisor)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {"anchor": anchors, "fisher": fisher}


def _check_ewc_alignment(model: Any, ewc_states: list[dict[str, Any]]) -> dict[str, Any]:
    """核对「可训练参数 / Fisher 参数 / anchor 参数」三者名称是否一致。

    建议要求：训练启动即打印并校验，避免 EWC 约束落在错误的参数子集上。
    """
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    fisher_names: set[str] = set()
    anchor_names: set[str] = set()
    for state in ewc_states:
        fisher_names |= set(state.get("fisher", {}).keys())
        anchor_names |= set(state.get("anchor", {}).keys())
    if not fisher_names and not anchor_names:
        return {"trainable_params": len(trainable), "fisher_params": 0,
                "anchor_params": 0, "overlap": 0, "name_mismatch": 0}
    overlap = trainable & fisher_names & anchor_names
    only_fisher = sorted(fisher_names - anchor_names)
    only_anchor = sorted(anchor_names - fisher_names)
    print(f"[EWC 参数对齐] 可训练={len(trainable)} Fisher={len(fisher_names)} "
          f"anchor={len(anchor_names)} 三者交集={len(overlap)}")
    if only_fisher or only_anchor:
        print(f"[EWC 参数对齐][警告] Fisher 与 anchor 名称不一致："
              f"仅 Fisher={only_fisher[:3]} 仅 anchor={only_anchor[:3]}")
    if fisher_names and not (trainable & fisher_names):
        raise RuntimeError(
            "EWC 状态参数名与当前可训练参数完全不匹配："
            f"Fisher={sorted(fisher_names)[:3]} vs trainable={sorted(trainable)[:3]}"
        )
    return {
        "trainable_params": len(trainable),
        "fisher_params": len(fisher_names),
        "anchor_params": len(anchor_names),
        "overlap": len(overlap),
        "name_mismatch": len(only_fisher) + len(only_anchor),
        "anchor_sets": len(ewc_states),
    }


def _load_ewc_states(
    task_number: int,
    model: Any,
    tokenizer: Any,
    fisher_limit: int | None = None,
    seed: int = DEFAULT_SEED,
    ewc_lambda: float = EWC_LAMBDA_DEFAULT,
    ewc_normalize: str = EWC_NORMALIZE_DEFAULT,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    import torch

    if task_number == 2:
        print("Task2 EWC：在 Task1 adapter 与 T1 train 上估计初始 Fisher…")
        task1_examples = _encode_records(read_jsonl(task_file(1, "train")), tokenizer)
        return [{"task": 1, **_estimate_fisher(model, task1_examples, tokenizer, max_examples=fisher_limit)}]

    state_path = _ewc_root(seed) / f"ewc_state_after_task{task_number - 1}.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"缺少前阶段 EWC Fisher 状态: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    expected_fingerprint = experiment_fingerprint(
        method="ewc",
        task=task_number - 1,
        seed=int(seed),
        run_tag=RUN_TAG or None,
        ewc_lambda=float(ewc_lambda),
        ewc_normalize=ewc_normalize,
        max_steps=(int(max_steps) if max_steps and int(max_steps) > 0 else None),
        base_model=MODEL_NAME,
    )
    if not isinstance(state, dict) or state.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"前阶段 EWC 状态与当前代码/数据/配置不匹配，需从 Task2 顺序重跑: {state_path}"
        )
    states = state.get("tasks")
    if not isinstance(states, list) or not states:
        raise ValueError(f"EWC 状态文件格式错误: {state_path}")
    return states


def _save_ewc_state(task_number: int, states: list[dict[str, Any]],
                    fingerprint: dict[str, Any], seed: int = DEFAULT_SEED) -> Path:
    import torch

    _ewc_root(seed).mkdir(parents=True, exist_ok=True)
    path = _ewc_root(seed) / f"ewc_state_after_task{task_number}.pt"
    if path.exists():
        raise FileExistsError(f"EWC 状态已存在，拒绝覆盖: {path}")
    torch.save({"method": "ewc", "completed_task": task_number,
                "fingerprint": fingerprint, "tasks": states}, path)
    return path


def _setup_adapter_model(model: Any, task_number: int, previous_names: list[str]) -> list[str]:
    adapter_name = f"task{task_number}"
    model.add_adapter(adapter_name, _make_lora_config())
    model.set_adapter(adapter_name)
    trainable_count = 0
    for name, parameter in model.named_parameters():
        is_current_adapter = f".{adapter_name}." in name
        parameter.requires_grad_(is_current_adapter)
        if is_current_adapter:
            trainable_count += parameter.numel()
    if trainable_count == 0:
        raise RuntimeError(f"新 Adapter {adapter_name} 没有可训练参数")
    print(f"Adapter 冻结检查：旧 adapters={previous_names}；当前可训练={adapter_name} ({trainable_count:,} params)")
    return previous_names + [adapter_name]


def _build_training_args(output_dir: Path, train_size: int, smoke: bool = False,
                         seed: int = DEFAULT_SEED, max_steps: int | None = None) -> Any:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        max_steps=1 if smoke else (int(max_steps) if max_steps and int(max_steps) > 0 else -1),
        per_device_train_batch_size=1 if smoke else BATCH_SIZE,
        gradient_accumulation_steps=1 if smoke else GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=1 if smoke else 10,
        save_strategy="no" if smoke else "epoch",
        eval_strategy="no",
        fp16=True,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=int(seed),
        data_seed=int(seed),
        remove_unused_columns=False,
        save_total_limit=2,
        dataloader_num_workers=2,
    )


def _ewc_penalty(model: Any, ewc_states: list[dict[str, Any]],
                 normalize: str = EWC_NORMALIZE_DEFAULT) -> Any:
    """EWC 二次惩罚。

    normalize="none"：旧行为（全局 sum），仅用于与 2026-11-04 的旧实验对照。
    normalize="param"：对单套 anchor 的参数量取平均（与任务数无关，λ 尺度稳定）。
    normalize="task"：对历史 anchor 套数取平均（online-EWC 风格）。
    """
    import torch

    named_parameters = dict(model.named_parameters())
    reference_parameter = next(
        (parameter for parameter in named_parameters.values() if parameter.requires_grad),
        None,
    )
    if reference_parameter is None:
        raise RuntimeError("EWC penalty 没有可训练参数")
    total = reference_parameter.new_zeros((), dtype=torch.float32)
    per_state_param_count = 0
    for state in ewc_states:
        state_total = reference_parameter.new_zeros((), dtype=torch.float32)
        for name, anchor in state["anchor"].items():
            parameter = named_parameters.get(name)
            if parameter is None or not parameter.requires_grad:
                continue
            fisher = state["fisher"][name].to(device=parameter.device, dtype=torch.float32)
            reference = anchor.to(device=parameter.device, dtype=torch.float32)
            state_total = state_total + (fisher * (parameter.float() - reference).square()).sum()
        if not per_state_param_count:
            per_state_param_count = sum(int(a.numel()) for a in state["anchor"].values())
        total = total + state_total
    if normalize == "param" and per_state_param_count:
        total = total / per_state_param_count
    elif normalize == "task":
        total = total / max(1, len(ewc_states))
    return total


def _save_method_output(
    method: str,
    task_number: int,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    adapter_names: list[str],
    train_size: int,
    loss: float,
    global_step: int,
    parent_path: Path | None,
    extra: dict[str, Any],
) -> None:
    if method == "adapter":
        adapters_dir = output_dir / "adapters"
        adapters_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapters_dir, selected_adapters=adapter_names)
        manifest = {
            "method": "adapter",
            "trained_through_task": task_number,
            "adapter_dir": "adapters",
            "adapters": adapter_names,
            "inference_policy_default": "route",
            "train_activation": "每个 adapter 仅在对应时间窗口单独训练：训练时只激活该 adapter，旧 adapter 全冻结",
            "train_active_per_adapter": {name: [name] for name in adapter_names},
            "primary_inference": "route",
            "experiment_definition": (
                "正式口径：测试样本所属时间窗口已知，评估窗口 Tj 时只激活该窗口对应的 adapter（route）。"
                "weighted / simultaneous 属扩展对照，不是本训练方案的自然结果，不得用于方法排名。"
                "若真实部署时未知窗口，需另建路由器并报告路由错误代价，不能用测试窗口编号冒充路由能力。"
            ),
            "inference_policies": {
                "route": "评估窗口 Tj 时只激活 taskj（与训练时该 adapter 的激活状态一致；"
                         "属「已知窗口」设置，需在报告中单独标注）",
                "weighted": "用 add_weighted_adapter 线性组合各 adapter，权重由验证集网格搜索确定"
                            "（adapter_weights.json）",
                "simultaneous": "同时等权激活全部 adapter（与训练不一致，仅作扩展对照，不用于方法排名）",
            },
            "note": "训练时每个窗口只训练自己的 adapter、旧 adapter 全程冻结；"
                    "推理必须显式选择策略，默认 route。",
        }
        with (output_dir / "adapter_manifest.json").open("w", encoding="utf-8") as target:
            json.dump(manifest, target, ensure_ascii=False, indent=2)
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "method": method,
        "task_number": task_number,
        "task": TASKS[task_number]["name"],
        "base_model": MODEL_NAME,
        "parent_model": str(parent_path) if parent_path else None,
        "train_file": str(task_file(task_number, "train")),
        "train_samples": train_size,
        "epochs": NUM_EPOCHS,
        "rank": LORA_R,
        "alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "global_step": global_step,
        "training_loss": loss,
        **extra,
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)


def run_continual_training(
    method: str,
    task_number: int,
    resume: bool = False,
    smoke: bool = False,
    seed: int = DEFAULT_SEED,
    ewc_lambda: float = EWC_LAMBDA_DEFAULT,
    ewc_normalize: str = EWC_NORMALIZE_DEFAULT,
    replay_capacity: int = REPLAY_MEMORY_CAPACITY,
    replay_ratio: float = REPLAY_RATIO_DEFAULT,
    replay_mode: str = REPLAY_MODE_DEFAULT,
    run_tag: str = "",
    max_steps: int | None = None,
) -> None:
    import torch
    from transformers import AutoTokenizer, Trainer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint

    if method not in METHODS:
        raise ValueError(f"此入口只支持 {'/'.join(METHODS)}，收到: {method}")
    if task_number not in range(2, 6):
        raise ValueError("三种持续学习方法从 Task2 开始，--task 只能是 2 到 5")
    if method == "replay" and replay_ratio <= 0:
        raise ValueError(f"Replay 比例必须大于 0，收到: {replay_ratio}")
    if method == "ewc" and ewc_lambda < 0:
        raise ValueError(f"EWC lambda 不能为负数，收到: {ewc_lambda}")
    if method == "ewc" and ewc_normalize not in ("none", "param", "task"):
        raise ValueError(f"未知 EWC 归一化方式: {ewc_normalize}")
    global SMOKE_MODE, RUN_TAG
    SMOKE_MODE = smoke
    RUN_TAG = run_tag
    fisher_limit = SMOKE_FISHER_LIMIT if smoke else None
    set_global_seed(seed)
    run_preflight(require_cuda=True)
    assert_cloud_gpu_allowed(f"{method} Task{task_number} 训练")
    output_dir = method_model_dir(method, task_number, seed)
    final_marker = output_dir / ("adapter_manifest.json" if method == "adapter" else "adapter_config.json")
    if not smoke:
        if final_marker.is_file():
            raise FileExistsError(f"目标模型已完成，拒绝覆盖: {output_dir}")
        if output_dir.exists() and any(output_dir.iterdir()) and not resume:
            raise FileExistsError(f"输出目录已存在；若是中断任务请加 --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed) + task_number)

    # ---------- 训练数据 ----------
    replay_records: list[dict[str, Any]] = []
    if method == "cumulative":
        task_data: list[dict[str, Any]] = []
        for previous in range(1, task_number + 1):
            task_data.extend(read_jsonl(task_file(previous, "train")))
    else:
        task_data = read_jsonl(task_file(task_number, "train"))

    memory: ReplayMemory | None = None
    if method == "replay":
        if task_number == 2:
            memory = ReplayMemory(capacity=replay_capacity, mode=replay_mode)
            memory.update(read_jsonl(task_file(1, "train")), rng)
        else:
            memory = _load_replay_memory(seed, replay_capacity, replay_mode, task_number - 1)
        if memory.records:
            wanted = int(round(len(task_data) / max(1e-6, replay_ratio)))
            replay_records = memory.sample(wanted, rng)
            print(f"Replay：从定容缓存(mode={memory.mode}, 容量 {memory.capacity}, "
                  f"现有 {len(memory.records)} 条)采样 {len(replay_records)} 条回放样本")
    train_records = task_data + replay_records
    rng.shuffle(train_records)
    if smoke:
        # 冒烟流程：每个模型最多 2 条样本（项目宗旨）
        train_records = train_records[:SMOKE_MAX_SAMPLES]
        print(f"冒烟模式：训练样本裁剪为 {len(train_records)} 条（上限 {SMOKE_MAX_SAMPLES}）")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = _encode_records(train_records, tokenizer)
    train_tokens = sum(len(example["input_ids"]) for example in encoded)
    PolicyDataset = _build_dataset_class()

    base_model = _load_quantized_base()
    if method == "cumulative":
        parent_path: Path | None = None
    else:
        parent_path = (adapter_dir(1, seed) if task_number == 2
                       else method_model_dir(method, task_number - 1, seed))
    model, adapter_names = _load_parent_adapter(base_model, method, task_number, seed)
    if method == "adapter":
        adapter_names = _setup_adapter_model(model, task_number, adapter_names)
    else:
        model.print_trainable_parameters()

    ewc_states = (
        _load_ewc_states(
            task_number,
            model,
            tokenizer,
            fisher_limit=fisher_limit,
            seed=seed,
            ewc_lambda=ewc_lambda,
            ewc_normalize=ewc_normalize,
            max_steps=max_steps,
        )
        if method == "ewc"
        else []
    )
    ewc_alignment = _check_ewc_alignment(model, ewc_states) if method == "ewc" else None

    class MetricsCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs:
                return
            path = output_dir / "training_metrics.jsonl"
            with path.open("a", encoding="utf-8") as target:
                target.write(json.dumps({"step": state.global_step, **logs}, ensure_ascii=False) + "\n")

    trainer_class = Trainer
    trainer_kwargs: dict[str, Any] = {}
    if method == "ewc":
        lambda_value = float(ewc_lambda)

        class EWCTrainer(Trainer):
            def __init__(self, *args: Any, ewc_states: list[dict[str, Any]],
                         ewc_normalize: str = EWC_NORMALIZE_DEFAULT, **kwargs: Any):
                self.ewc_states = ewc_states
                self.ewc_normalize = ewc_normalize
                super().__init__(*args, **kwargs)

            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any = None,
            ) -> Any:
                outputs = model(**inputs)
                if lambda_value == 0.0:
                    # λ=0 对照组：退化为纯任务损失，用于校验与顺序微调的一致性
                    return (outputs.loss, outputs) if return_outputs else outputs.loss
                penalty = _ewc_penalty(model, self.ewc_states, self.ewc_normalize)
                loss = outputs.loss + (lambda_value / 2.0) * penalty
                return (loss, outputs) if return_outputs else loss

        trainer_class = EWCTrainer
        trainer_kwargs["ewc_states"] = ewc_states
        trainer_kwargs["ewc_normalize"] = ewc_normalize

    training_args = _build_training_args(output_dir, len(train_records), smoke=smoke, seed=seed,
                                         max_steps=max_steps)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=PolicyDataset(encoded),
        data_collator=partial(_collate_batch, pad_token_id=tokenizer.pad_token_id),
        callbacks=[MetricsCallback()],
        **trainer_kwargs,
    )
    checkpoint = None
    if resume and not smoke:
        checkpoint = get_last_checkpoint(str(output_dir))
        if checkpoint is None:
            raise FileNotFoundError(f"没有可恢复的训练 checkpoint: {output_dir}")
        print(f"从 checkpoint 恢复: {checkpoint}")

    print(f"方法: {method}; 训练至 Task{task_number}; parent: {parent_path}; seed: {seed}")
    print(f"新增数据: {len(task_data)}; replay 数据: {len(replay_records)}; 合计: {len(train_records)}")
    if method == "ewc":
        print(f"EWC: lambda={float(ewc_lambda)}; normalize={ewc_normalize}; "
              f"历史 anchor 套数={len(ewc_states)}")
    result = trainer.train(resume_from_checkpoint=checkpoint)

    extra: dict[str, Any] = {
        "seed": int(seed),
        "run_tag": run_tag or None,
        "train_tokens": train_tokens,
        "examples_seen": len(train_records) * NUM_EPOCHS,
        "ewc_state_mode": "cumulative_all_tasks" if method == "ewc" else None,
        "ewc_anchor_sets": len(ewc_states) if method == "ewc" else 0,
        "ewc_param_alignment": ewc_alignment,
        "ewc_fisher_data_scope": "current_task_train（Fisher 在本阶段训练完成后、用本阶段 train 估计，供下一阶段使用）",
        "provenance": environment_provenance(),
        "fingerprint": experiment_fingerprint(
            method=method, task=task_number, seed=int(seed), run_tag=run_tag or None,
            ewc_lambda=float(ewc_lambda) if method == "ewc" else None,
            ewc_normalize=ewc_normalize if method == "ewc" else None,
            replay_capacity=replay_capacity if method == "replay" else None,
            replay_ratio=replay_ratio if method == "replay" else None,
            replay_mode=replay_mode if method == "replay" else None,
            max_steps=(int(max_steps) if max_steps and int(max_steps) > 0 else None),
            base_model=MODEL_NAME,
            adapter_inference_primary="route" if method == "adapter" else None,
        ),
    }
    if method == "ewc":
        print(f"Task{task_number} EWC：估计当前阶段 Fisher，以供后续任务使用…")
        task_examples = _encode_records(task_data, tokenizer)
        ewc_states.append({"task": task_number,
                           **_estimate_fisher(model, task_examples, tokenizer, max_examples=fisher_limit)})
        state_path = _save_ewc_state(
            task_number, ewc_states, fingerprint=extra["fingerprint"], seed=seed
        )
        extra.update({"ewc_lambda": float(ewc_lambda),
                      "ewc_normalize": ewc_normalize,
                      "ewc_anchor_sets": len(ewc_states),
                      "ewc_state": str(state_path)})
    elif method == "replay":
        assert memory is not None
        memory.update(task_data, rng)
        memory_path = _save_replay_memory(memory, seed, task_number)
        extra.update({"replay_capacity": int(replay_capacity),
                      "replay_mode": replay_mode,
                      "replay_ratio_new_to_old": float(replay_ratio),
                      "replay_samples": len(replay_records),
                      "replay_pool_size": len(memory.records),
                      "replay_pool_seen": memory.seen,
                      "replay_memory": str(memory_path)})
    elif method == "cumulative":
        extra.update({"cumulative_examples": len(train_records),
                      "cumulative_windows": list(range(1, task_number + 1)),
                      "note": "累积训练参考组：每阶段从基座用全部已见窗口训练数据重训，"
                              "用于给出「历史数据全部可用」时的参考表现，非持续学习方法"})

    _save_method_output(
        method,
        task_number,
        output_dir,
        model,
        tokenizer,
        adapter_names,
        len(train_records),
        float(result.training_loss),
        int(result.global_step),
        parent_path,
        extra,
    )
    if smoke:
        if result.global_step != 1:
            raise RuntimeError(f"冒烟训练未完成一步更新: global_step={result.global_step}")
        print(f"冒烟测试通过：{method} Task{task_number} 完成 1 步训练，产物已写入临时冒烟目录（不产出正式模型）。")
    print(f"完成：{output_dir}")


def method_main(method: str) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=f"训练时间增量 {method.upper()}，支持 Task2-Task5")
    parser.add_argument("--task", type=int, choices=range(2, 6), default=2)
    parser.add_argument("--resume", action="store_true", help="从本方法、本任务最近的 Trainer checkpoint 续训")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟模式：单步训练 + Fisher 截断，产物写入临时冒烟目录，不产出正式模型")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子；非 {DEFAULT_SEED} 的种子写入独立目录（_s{{seed}}），避免互相覆盖")
    parser.add_argument("--ewc-lambda", type=float, default=EWC_LAMBDA_DEFAULT,
                        help="EWC 正则系数（建议在 0/1/10/100/500/1000 上重搜；0 = 与顺序微调对照；"
                             f"旧实验用的 {LEGACY_EWC_LAMBDA:.0f} 属未归一化尺度，不可沿用）")
    parser.add_argument("--ewc-normalize", choices=("none", "param", "task"),
                        default=EWC_NORMALIZE_DEFAULT,
                        help="EWC 惩罚归一化：none=旧行为(全局sum)；param=按参数量平均；task=按 anchor 套数平均")
    parser.add_argument("--replay-capacity", type=int, default=REPLAY_MEMORY_CAPACITY,
                        help="回放缓存总容量（条）")
    parser.add_argument("--replay-ratio", type=float, default=REPLAY_RATIO_DEFAULT,
                        help="新样本:回放样本 比例（1 = 1:1，3 = 3:1）")
    parser.add_argument("--replay-mode", choices=("reservoir", "class_balanced"),
                        default=REPLAY_MODE_DEFAULT,
                        help="回放缓存填充方式：reservoir=流式蓄水池；class_balanced=每类等额")
    parser.add_argument("--run-tag", default="",
                        help="实验标签：隔离超参搜索产生的多组目录与状态（如 lambda100、ratio3）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="固定优化步数（等步数预算对照用；<=0 表示按 epochs 跑满）")
    args = parser.parse_args()
    run_continual_training(
        method,
        args.task,
        resume=args.resume,
        smoke=args.smoke,
        seed=args.seed,
        ewc_lambda=args.ewc_lambda,
        ewc_normalize=args.ewc_normalize,
        replay_capacity=args.replay_capacity,
        replay_ratio=args.replay_ratio,
        replay_mode=args.replay_mode,
        run_tag=args.run_tag,
        max_steps=args.max_steps,
    )
