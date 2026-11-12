"""Shared data, prompt, and QLoRA training helpers for temporal tasks."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

from gpu_guard import SMOKE_MAX_SAMPLES, assert_cloud_gpu_allowed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCREMENTAL_ROOT = PROJECT_ROOT / "05_incremental_learning_domain"
# 大文件集中区（模型 / 状态 / 冒烟产物），不入版本库，见 .gitignore
ASSETS_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_domain"
# 使用数据（D1–D6 训练/测试 jsonl）保留在主库，随代码一起入库
DATA_ROOT = INCREMENTAL_ROOT / "data" / "domain_incremental"
MODEL_ROOT = ASSETS_ROOT / "models"
LOG_ROOT = INCREMENTAL_ROOT / "logs" / "domain_incremental"
SMOKE_ROOT = ASSETS_ROOT / "smoke_tests"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ("N", "R", "W")
MAX_LENGTH = 768
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
BATCH_SIZE = 4          # 24G 显存直跑；有效 batch 4×1 == 旧版 1×4，与 Task1/03/04 结果可比
GRADIENT_ACCUMULATION = 1
DEFAULT_SEED = 42
SEED = DEFAULT_SEED          # 向后兼容：默认随机种子
EVAL_MAX_LENGTH = MAX_LENGTH - 1   # 评估截断与训练对齐（训练 = MAX_LENGTH-1 提示 + 1 标签 token）
VAL_PER_CLASS = 40           # 每窗口验证集每类配额（独立抽取，不改动既有 train/test）

# 领域增量：6 个阶段 = 6 个训练领域，每个领域内部再覆盖 T1~T5 全部年代。
# name 必须与 data/domain_incremental/ 下的文件前缀完全一致。
TASKS = {
    1: {"name": "D1_经济与财税金融", "start": 1949, "end": 2023, "domain": "经济与财税金融"},
    2: {"name": "D2_教育", "start": 1949, "end": 2023, "domain": "教育"},
    3: {"name": "D3_卫生健康与食药", "start": 1949, "end": 2023, "domain": "卫生健康与食药"},
    4: {"name": "D4_农业与农村", "start": 1949, "end": 2023, "domain": "农业与农村"},
    5: {"name": "D5_自然资源与生态环境", "start": 1949, "end": 2023, "domain": "自然资源与生态环境"},
    6: {"name": "D6_公共安全与应急管理", "start": 1949, "end": 2023, "domain": "公共安全与应急管理"},
}

# A3：文档不要求互斥（同一文档的段落可跨阶段/跨划分），但段落内容严格互斥。
DOCUMENT_EXCLUSIVE = False

ERA_ALL = ("T1_1949_1978", "T2_1979_1992", "T3_1993_2002",
           "T4_2003_2012", "T5_2013_2023")


def text_key(text: str) -> str:
    """段落严格互斥键（P1-1）：NFKC + casefold + 去所有非文字/非汉字字符。

    比「去空白后精确比对」更严：全半角、标点、大小写差异都会归一到同一键。
    """
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]

SYSTEM_PROMPT = """角色：你是一名“政策信号一级标注员”。

任务：对给定的中文政策文档段落进行分类标注：

W = 段落中存在对政府/公共部门的肯定性行动指令（包含任何“要/应/负责/建立/加强/推进/组织/指导/监督/落实/制定/完善/开展/实施/安排”等对政府主体布置的任务或授权）。

R = 段落中存在对政府/公共部门的禁止性指令（如“严禁/不得/禁止/停止……（由政府或其部门）”）。

N = 无指令（仅背景介绍、原则口号、定义释义、程序性条款、日期/效力条款、对非国家主体的要求而未同时对政府布置任务或责任的内容）。

受众原则（极其重要）：只统计指向政府及其附属公共机构（包括国务院各部委、地方政府、监管机构、事业单位等）的行动/禁止为“指令”。

仅限制或要求企业/个人/社会组织，而没有同时给政府安排任务或责任 → 记为 N。

若同时出现：例如“企业不得……；主管部门要加强监管……”，则本段视为存在对政府的指令，进入 W 或 R 的判断。

不要只看关键词：是否出现“必须/可以/鼓励/不得”等词不等于对应标签，必须结合主语是否为政府/公共机构与句子功能来判断。

例如：“企业必须符合××条件方可办理” → 若仅针对企业且未布置政府任务 → N。

输出要求：仅输出最终的结果W/N/R"""


def adapter_name(task_number: int) -> str:
    if task_number not in TASKS:
        raise ValueError(f"任务编号必须为 1 到 {len(TASKS)}，收到: {task_number}")
    return f"qwen15b_domain_task{task_number}_e2_r16_a32_lr2e-4"


def model_group(method: str, task_number: int) -> str:
    """模型分组目录：Joint（cumulative）系列归 joint/，其余按阶段归 stage{N}/。"""
    return "joint" if method.startswith("cumulative") else f"stage{task_number}"


def adapter_dir(task_number: int, seed: int = DEFAULT_SEED) -> Path:
    """adapter 目录；默认种子沿用历史命名，非默认种子加 _s{seed} 后缀以隔离实验。"""
    name = adapter_name(task_number)
    if int(seed) == DEFAULT_SEED:
        return MODEL_ROOT / f"stage{task_number}" / name
    return MODEL_ROOT / f"stage{task_number}" / f"{name}_s{int(seed)}"


def code_version() -> str:
    """关键脚本内容的短哈希：代码一改，实验指纹随之改变（禁止跨代码版本复用产物）。"""
    digest = hashlib.sha1()
    for name in ("incremental_common.py", "continual_training.py",
                 "incremental_evaluation.py", "gpu_guard.py", "run_full_matrix.py"):
        path = Path(__file__).resolve().parent / name
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:10]


def data_version() -> str:
    """数据集指纹（P1-7）：显式列出 12 个领域文件（6 域 × train/test）。

    刻意不 glob，避免将来生成 `*_val.jsonl` 时改变主指纹、把全部已完成产物判为冲突。
    """
    digest = hashlib.sha1()
    for task_number in sorted(TASKS):
        for split in ("train", "test"):
            path = DATA_ROOT / f"{TASKS[task_number]['name']}_{split}.jsonl"
            try:
                digest.update(path.name.encode("utf-8"))
                digest.update(path.read_bytes())
            except OSError:
                continue
    return digest.hexdigest()[:10]


def experiment_fingerprint(**params: Any) -> dict[str, Any]:
    """实验配置指纹：任一项不同即视为不同实验，产物不得互相复用。

    参与项：seed / run_tag / 数据与代码版本 / 基座版本 / 各方法超参 /
    评估策略（adapter_inference、split）等。
    """
    payload = {key: value for key, value in sorted(params.items()) if value is not None}
    payload["code_version"] = code_version()
    payload["data_version"] = data_version()
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return {"hash": hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10], "params": payload}


def environment_provenance(model_name: str = MODEL_NAME) -> dict[str, Any]:
    """记录软件/硬件/基座版本：跨环境结果不可直接合并，需逐项核对。"""
    info: dict[str, Any] = {"python": sys.version.split()[0], "base_model": model_name}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "unavailable"
    for name in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            info[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception:
            info[name] = "not-installed"
    try:
        from huggingface_hub import snapshot_download

        info["base_model_revision"] = Path(
            snapshot_download(model_name, local_files_only=True)).name
    except Exception:
        info["base_model_revision"] = "unknown"
    return info


def set_global_seed(seed: int) -> None:
    """统一设置 random / numpy / torch 种子（含 CUDA），供训练与评估共用。"""
    import random as _random

    import numpy as np
    import torch

    seed = int(seed)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_file(task_number: int, split: str) -> Path:
    if task_number not in TASKS or split not in ("train", "test", "val"):
        raise ValueError(f"无效任务或数据集划分: task={task_number}, split={split}")
    return DATA_ROOT / f"{TASKS[task_number]['name']}_{split}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 格式错误: {path}:{line_number}: {error}") from error
            records.append(record)
    return records


def atomic_write_json(path: Path, payload: Any) -> None:
    """原子写 JSON（P2-2）：temp + rename，避免中断留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parent_record(parent_dir: Path | None) -> dict[str, Any] | None:
    """P1-7：记录父产物的身份（路径 + 指纹），供下游校验"父是否被替换过"。"""
    if parent_dir is None:
        return None
    meta_path = Path(parent_dir) / "training_metadata.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    except (OSError, json.JSONDecodeError):
        meta = None
    return {"path": str(parent_dir),
            "fingerprint": (meta or {}).get("fingerprint"),
            "completed": (meta or {}).get("completed")}


ORCHESTRATOR_LOCK = ASSETS_ROOT / ".orchestrator.lock"


def _pid_alive(pid: int) -> bool:
    """判断 pid 是否仍存活（Windows 不做探测，保守视为存活）。"""
    import os

    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_orchestrator_lock() -> bool:
    """P2-7：同一时间只允许一个编排进程，避免两套编排互踩模型/结果目录。

    被 SIGKILL 时 atexit 不会执行，会留下残留锁；这里检测持有者是否存活，
    已死则自动接管，避免"杀过一次就再也起不来"。
    """
    import atexit
    import os

    ORCHESTRATOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    for _attempt in (1, 2, 3):
        try:
            fd = os.open(str(ORCHESTRATOR_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                holder = ORCHESTRATOR_LOCK.read_text(encoding="utf-8").strip()
            except OSError:
                holder = ""
            if holder.isdigit() and _pid_alive(int(holder)):
                print(f"[错误] 已有编排进程在运行（pid {holder}）：{ORCHESTRATOR_LOCK}\n"
                      f"       确认无进程占用后可手动删除该锁文件再重试。")
                return False
            print(f"[修复] 检测到残留锁（持有者 pid {holder or '未知'} 已不存在），自动清理后继续")
            try:
                ORCHESTRATOR_LOCK.unlink()
            except OSError:
                pass
    if fd is None:
        print(f"[错误] 无法获取编排锁：{ORCHESTRATOR_LOCK}")
        return False
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)

    def _release() -> None:
        try:
            ORCHESTRATOR_LOCK.unlink()
        except OSError:
            pass

    atexit.register(_release)
    return True


def _document_key(record: dict[str, Any]) -> tuple[str, ...]:
    key = record.get("DocumentKey")
    if isinstance(key, list):
        return tuple(str(part) for part in key)
    return (str(record.get("DocumentID", "")),)


def validate_task_data(task_number: int) -> dict[str, list[dict[str, Any]]]:
    task = TASKS[task_number]
    data = {split: read_jsonl(task_file(task_number, split)) for split in ("train", "test")}
    expected_per_class = {"train": 200, "test": 100}
    val_path = task_file(task_number, "val")
    if val_path.is_file():
        data["val"] = read_jsonl(val_path)
        expected_per_class["val"] = VAL_PER_CLASS

    for split, records in data.items():
        if len(records) != expected_per_class[split] * len(LABELS):
            raise ValueError(
                f"{task['name']} {split} 应为 {expected_per_class[split] * len(LABELS)} 条，"
                f"实际 {len(records)} 条"
            )
        counts = Counter(record.get("label") for record in records)
        if any(counts[label] != expected_per_class[split] for label in LABELS):
            raise ValueError(f"{task['name']} {split} 标签分布错误: {dict(counts)}")
        for index, record in enumerate(records):
            if not isinstance(record.get("text"), str) or not record["text"].strip():
                raise ValueError(f"{task['name']} {split} 第 {index} 条缺少有效 text")
            if record.get("split") != split:
                raise ValueError(
                    f"{task['name']} {split} 第 {index} 条 split 字段为 {record.get('split')!r}"
                )
            if record.get("label") not in LABELS:
                raise ValueError(f"{task['name']} {split} 第 {index} 条标签无效")
            year = int(record.get("year", -1))
            if not task["start"] <= year <= task["end"]:
                raise ValueError(f"{task['name']} {split} 第 {index} 条年份越界: {year}")

    if DOCUMENT_EXCLUSIVE:
        docs_by_split = {
            split: {_document_key(record) for record in records}
            for split, records in data.items()
        }
        splits = sorted(docs_by_split)
        for i, left in enumerate(splits):
            for right in splits[i + 1:]:
                overlap = docs_by_split[left] & docs_by_split[right]
                if overlap:
                    raise ValueError(
                        f"{task['name']} {left}/{right} 存在 {len(overlap)} 篇文档交叉"
                    )
    return data


def validate_all_temporal_data() -> dict[int, dict[str, list[dict[str, Any]]]]:
    """全局数据契约校验（P1-1）。

    1. 段落严格互斥：强键（NFKC + 去标点 + casefold）**全库唯一**，同 split 内重复也算错；
    2. 域字段一致：record["domain"] 必须等于 TASKS[task]["domain"]；
    3. 年代覆盖：每个域 T1~T5 必须齐全；
    4. 文档重叠：A3 允许，仅统计留痕，不报错。
    """
    datasets = {task: validate_task_data(task) for task in TASKS}
    seen_text: dict[str, str] = {}
    doc_locations: dict[str, set[str]] = {}
    for task_number, splits in datasets.items():
        meta = TASKS[task_number]
        expect_domain = meta.get("domain")
        eras: Counter = Counter()
        for split, records in splits.items():
            location = f"{meta['name']}/{split}"
            for record in records:
                if expect_domain and str(record.get("domain", "")).strip() != expect_domain:
                    raise ValueError(
                        f"{location} 含错域记录: domain={record.get('domain')!r}，"
                        f"应为 {expect_domain!r}")
                key = text_key(record["text"])
                previous = seen_text.get(key)
                if previous is not None:
                    raise ValueError(f"段落内容重复: {previous} 与 {location}")
                seen_text[key] = location
                eras[str(record.get("era"))] += 1
                doc = str(record.get("DocumentID") or record.get("DocumentKey"))
                doc_locations.setdefault(doc, set()).add(location)
        missing_eras = [era for era in ERA_ALL if not eras.get(era)]
        if missing_eras:
            raise ValueError(f"{meta['name']} 缺少年代: {missing_eras}")
    overlap_docs = sum(1 for locations in doc_locations.values() if len(locations) > 1)
    print(f"段落全局唯一 {len(seen_text):,} 条（强键）；"
          f"文档跨(任务/划分) {overlap_docs:,} 篇（A3 允许，仅留痕）")
    return datasets


def build_prompt(tokenizer: Any, text: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"待判断政策段落：\n{text}"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def truncate_head_tail(token_ids: list[int], max_length: int,
                      head_ratio: float = 0.5,
                      marker_ids: list[int] | None = None) -> tuple[list[int], bool]:
    """超长序列保留「头部 + 尾部」。

    政策文本的发文机关、标题、背景、指令主体与作用对象常分布在首尾两端，
    只保留尾部会丢掉关键前提。训练与评估必须调用同一函数、同一方向、同一长度。
    返回 (截断后的 token 列表, 是否发生截断)。
    """
    if max_length <= 0:
        raise ValueError("max_length 必须为正")
    if len(token_ids) <= max_length:
        return list(token_ids), False
    marker = list(marker_ids or [])
    budget = max_length - len(marker)
    if budget <= 0:
        return list(token_ids[:max_length]), True
    head = max(1, int(budget * head_ratio))
    tail = max(1, budget - head)
    return list(token_ids[:head]) + marker + list(token_ids[-tail:]), True


_PREFIX_TOKENS: dict[int, int] = {}
_TRUNC_STATS = {"total": 0, "truncated": 0}


def prompt_prefix_tokens(tokenizer: Any) -> int:
    """chat template + SYSTEM_PROMPT 的固定前缀 token 数（按 tokenizer 实例缓存）。"""
    key = id(tokenizer)
    if key not in _PREFIX_TOKENS:
        _PREFIX_TOKENS[key] = len(build_prompt(tokenizer, ""))
    return _PREFIX_TOKENS[key]


def reset_truncation_stats() -> None:
    _TRUNC_STATS["total"] = 0
    _TRUNC_STATS["truncated"] = 0


def truncation_summary() -> dict[str, Any]:
    total = _TRUNC_STATS["total"]
    truncated = _TRUNC_STATS["truncated"]
    return {"total": total, "truncated": truncated,
            "ratio": round(truncated / total, 4) if total else 0.0}


def fit_text(tokenizer: Any, text: str, budget: int) -> tuple[str, bool]:
    """在「文本层」截断：先扣掉 system prompt 的固定开销，再按 token 保留头 60% / 尾 40%。

    旧实现从 prompt 头部切，超长时头部几乎全被 SYSTEM_PROMPT 占满（实测正文头只剩 28 token）；
    文本层截断可保证提示词完整、正文头尾都保留。
    """
    ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return str(text), False
    head = max(1, int(budget * 0.6))
    tail = max(1, budget - head)
    left = tokenizer.decode(ids[:head], skip_special_tokens=True)
    right = tokenizer.decode(ids[-tail:], skip_special_tokens=True)
    return f"{left}……{right}", True


def build_prompt_fitted(tokenizer: Any, text: str,
                        max_length: int | None = None) -> tuple[list[int], bool]:
    """构造 prompt 并保证总长度 ≤ max_length；返回 (input_ids, 是否截断)。"""
    limit = EVAL_MAX_LENGTH if max_length is None else max_length
    budget = max(1, limit - prompt_prefix_tokens(tokenizer) - 4)
    fitted, truncated = fit_text(tokenizer, text, budget)
    ids = build_prompt(tokenizer, fitted)
    if len(ids) > limit:                      # 重编码偶发膨胀，兜底硬截断
        ids, _ = truncate_head_tail(ids, limit)
        truncated = True
    return ids, truncated


def truncation_stats(records: list[dict[str, Any]], tokenizer: Any,
                      max_prompt_length: int | None = None) -> dict[str, Any]:
    """统计 prompt 截断比例（总体与按类别），用于训练/评估留痕。"""
    limit = max_prompt_length if max_prompt_length is not None else MAX_LENGTH - 1
    total = 0
    truncated = 0
    lengths: list[int] = []
    per_label: dict[str, dict[str, int]] = {}
    for record in records:
        ids = build_prompt(tokenizer, record["text"])
        hit = len(ids) > limit
        total += 1
        truncated += int(hit)
        lengths.append(len(ids))
        bucket = per_label.setdefault(str(record["label"]), {"n": 0, "truncated": 0})
        bucket["n"] += 1
        bucket["truncated"] += int(hit)
    lengths.sort()
    return {
        "max_prompt_length": limit,
        "prompt_prefix_tokens": prompt_prefix_tokens(tokenizer),
        "total": total,
        "truncated": truncated,
        "ratio": round(truncated / total, 4) if total else 0.0,
        "length_stats": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "p95": lengths[int(len(lengths) * 0.95)] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "per_label": {
            label: {
                "n": bucket["n"],
                "truncated": bucket["truncated"],
                "ratio": round(bucket["truncated"] / bucket["n"], 4) if bucket["n"] else 0.0,
            }
            for label, bucket in sorted(per_label.items())
        },
    }


def encode_record(record: dict[str, Any], tokenizer: Any) -> dict[str, list[int]]:
    label_ids = tokenizer(str(record["label"]), add_special_tokens=False)["input_ids"]
    if len(label_ids) != 1:
        raise ValueError(f"标签 {record['label']} 不是单 token，无法进行稳定的标签打分: {label_ids}")
    max_prompt_length = MAX_LENGTH - len(label_ids)
    if max_prompt_length <= 0:
        raise ValueError("MAX_LENGTH 不足以容纳监督标签")
    prompt_ids, truncated = build_prompt_fitted(tokenizer, record["text"], max_prompt_length)
    _TRUNC_STATS["total"] += 1
    _TRUNC_STATS["truncated"] += int(truncated)
    input_ids = prompt_ids + label_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + label_ids,
    }


def select_balanced_samples(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets = {label: [record for record in records if record["label"] == label] for label in LABELS}
    selected = []
    while len(selected) < min(limit, len(records)):
        changed = False
        for label in LABELS:
            if buckets[label] and len(selected) < min(limit, len(records)):
                selected.append(buckets[label].pop(0))
                changed = True
        if not changed:
            break
    return selected


def run_preflight(require_cuda: bool = False) -> dict[int, dict[str, list[dict[str, Any]]]]:
    import importlib.util

    missing = [
        package for package in ("torch", "transformers", "peft", "bitsandbytes")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise RuntimeError(f"缺少训练依赖: {', '.join(missing)}")
    datasets = validate_all_temporal_data()
    has_val = all("val" in splits for splits in datasets.values())
    val_note = "，val=120（每类 40）" if has_val else "（尚未构建 val，调参需先跑 temporal-val）"
    print(f"数据检查通过：{len(TASKS)} 个领域任务均为 train=600 / test=300{val_note}，"
          "类别均衡、年份合法、段落全局唯一（文档重叠按 A3 仅留痕）。")
    print(f"基础模型: {MODEL_NAME}")
    print(f"训练参数: Epoch={NUM_EPOCHS}, Rank={LORA_R}, Alpha={LORA_ALPHA}, LR={LEARNING_RATE}")
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；QLoRA 训练预检需要可用的 NVIDIA GPU。")
        print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
    return datasets


def _collate_batch(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_length = max(len(feature["input_ids"]) for feature in features)
    input_ids, attention_mask, labels = [], [], []
    for feature in features:
        pad_length = max_length - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
        attention_mask.append(feature["attention_mask"] + [0] * pad_length)
        labels.append(feature["labels"] + [-100] * pad_length)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class PolicyDataset:
    """模块级数据集（不能定义在函数体内：Windows 的 dataloader spawn 无法 pickle 局部类）。"""

    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def run_training(
    task_number: int,
    *,
    resume: bool = False,
    smoke: bool = False,
    sample_limit: int | None = None,
    seed: int = DEFAULT_SEED,
    max_steps: int | None = None,
) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from torch.utils.data import Dataset

    if task_number not in TASKS:
        raise ValueError(f"任务编号必须为 1 到 {len(TASKS)}，收到: {task_number}")
    set_global_seed(seed)
    assert_cloud_gpu_allowed(f"Task{task_number} QLoRA 训练")

    data = validate_task_data(task_number)
    train_records = data["train"]
    if sample_limit is not None:
        train_records = select_balanced_samples(train_records, sample_limit)
    task_name = TASKS[task_number]["name"]
    output_dir = adapter_dir(task_number, seed)
    previous_adapter = adapter_dir(task_number - 1, seed) if task_number > 1 else None

    if not smoke:
        if output_dir.exists() and (output_dir / "adapter_config.json").exists():
            raise FileExistsError(
                f"目标 adapter 已存在，拒绝覆盖: {output_dir}. "
                "请先备份/移走现有目录，再重新运行。"
            )
        if output_dir.exists() and any(output_dir.iterdir()) and not resume:
            raise FileExistsError(f"输出目录非空；若要从中断的 checkpoint 续训，请添加 --resume: {output_dir}")
    if task_number > 1 and not (previous_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Task{task_number} 需要先完成 Task{task_number - 1}: {previous_adapter}"
        )

    print(f"任务: {task_name}")
    print(f"训练样本: {len(train_records)}")
    print(f"adapter 输出: {output_dir if not smoke else '临时冒烟测试目录'}")
    if previous_adapter:
        print(f"继承 adapter: {previous_adapter}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reset_truncation_stats()
    encoded = [encode_record(record, tokenizer) for record in train_records]
    train_tokens = sum(len(example["input_ids"]) for example in encoded)
    trunc_stats = truncation_summary()
    if trunc_stats["truncated"]:
        print(f"截断统计：{trunc_stats['truncated']}/{trunc_stats['total']} 条超长，"
              f"已在文本层保留头尾（提示词完整）")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    if previous_adapter is None:
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
    else:
        model = PeftModel.from_pretrained(base_model, previous_adapter, is_trainable=True)
    model.print_trainable_parameters()

    temporary_directory = None
    if smoke:
        SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
        temporary_directory = tempfile.TemporaryDirectory(prefix=f"task{task_number}_", dir=SMOKE_ROOT)
        run_dir = Path(temporary_directory.name)
    else:
        run_dir = output_dir
        run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "training_metrics.jsonl"

    class MetricsCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs:
                return
            row = {"step": state.global_step, **logs}
            with metrics_path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=NUM_EPOCHS,
        max_steps=1 if smoke else (int(max_steps) if max_steps and int(max_steps) > 0 else -1),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=1 if smoke else 10,
        save_strategy="no" if smoke else "epoch",
        eval_strategy="no",
        bf16=True,                 # 4090D 原生支持；避免 fp16 溢出与 Fisher 下溢（P2-4 / P1-4）
        fp16=False,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=2,
        seed=int(seed),
        data_seed=int(seed),
        remove_unused_columns=False,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=PolicyDataset(encoded),
        data_collator=partial(_collate_batch, pad_token_id=tokenizer.pad_token_id),
        callbacks=[MetricsCallback()],
    )

    checkpoint = None
    if resume:
        checkpoint = get_last_checkpoint(str(run_dir))
        if checkpoint is None:
            raise FileNotFoundError(f"未找到可恢复的 checkpoint: {run_dir}")
        print(f"从 checkpoint 恢复: {checkpoint}")
    result = trainer.train(resume_from_checkpoint=checkpoint)

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    if smoke:
        if result.global_step != 1 or not (run_dir / "adapter_config.json").exists():
            raise RuntimeError("冒烟训练未完成一步更新或 adapter 保存失败")
        print(f"冒烟测试通过：完成 {result.global_step} 个优化步骤并成功保存 adapter。")
        temporary_directory.cleanup()
        return

    metadata = {
        "task": task_name,
        "task_number": task_number,
        "method": "sequential_ft",
        "base_model": MODEL_NAME,
        "parent_adapter": str(previous_adapter) if previous_adapter else None,
        "train_file": str(task_file(task_number, "train")),
        "train_samples": len(train_records),
        "epochs": NUM_EPOCHS,
        "rank": LORA_R,
        "alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "global_step": result.global_step,
        "training_loss": result.training_loss,
        "training_metrics": str(metrics_path),
        "seed": int(seed),
        "train_tokens": train_tokens,
        "examples_seen": len(train_records) * NUM_EPOCHS,
        "train_truncation": trunc_stats,
        "compute_dtype": "bfloat16",
        "parent": parent_record(previous_adapter),
        "fingerprint": experiment_fingerprint(
            method="sequential_ft",
            task=task_number,
            seed=int(seed),
            run_tag=None,
            max_steps=(int(max_steps) if max_steps and int(max_steps) > 0 else None),
            base_model=MODEL_NAME,
        ),
    }
    metadata["completed"] = True          # P2-2：完成标记最后写入
    atomic_write_json(output_dir / "training_metadata.json", metadata)
    print(f"训练完成，adapter 已保存: {output_dir}")


def train_main(task_number: int) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=f"训练时间增量 Task{task_number} QLoRA adapter")
    parser.add_argument("--resume", action="store_true", help="从本任务输出目录中的最新 checkpoint 续训")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟模式：单步训练，产物写入临时目录，不产出正式 adapter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子；非 {DEFAULT_SEED} 的种子会写入独立目录，避免与既有实验互相覆盖")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="固定优化步数（等步数预算对照用；<=0 表示按 epochs 跑满）")
    args = parser.parse_args()
    run_training(task_number, resume=args.resume, smoke=args.smoke,
                 sample_limit=SMOKE_MAX_SAMPLES if args.smoke else None, seed=args.seed,
                 max_steps=args.max_steps)