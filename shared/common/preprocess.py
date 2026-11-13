"""CAPC-CG 数据加载与预处理 (复现基线脚本).

将 parquet 中 chat 格式的 `messages` 列转换为 (text, label) 对，
支持两个任务：
  - task1_level1 : 一级分类 W / R / N
  - task2_level2 : 二级分类 B / Y / C / G

注意：用 pandas/pyarrow 直接读取时，`messages` 是 numpy.ndarray(元素为 dict)，
需要先 .tolist() 再按角色取字段；用户文本带前缀 "待判断政策段落："，
做 sklearn 分类时应去掉该前缀，做 OpenAI 微调时应保留原样。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter

import pandas as pd

# 任务 -> 论文定义的标签集合（顺序固定，便于标签编码）
TASK_LABELS = {
    "task1_level1": ["W", "R", "N"],
    "task2_level2": ["B", "Y", "C", "G"],
}

USER_PREFIX = "待判断政策段落："


def hub_cache_roots() -> list[Path]:
    """HF hub 缓存根目录，按优先级去重：HF_HUB_CACHE → HF_HOME/hub → 默认。"""
    roots: list[Path] = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env, "").strip()
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def find_snapshot() -> Path:
    """定位 HuggingFace 本地缓存中的最新 snapshot 目录（支持 HF_HOME）。"""
    tried: list[str] = []
    for root in hub_cache_roots():
        snap_root = root / "datasets--Baron-Sun--CAPC-CG_V1.0" / "snapshots"
        tried.append(str(snap_root))
        if not snap_root.is_dir():
            continue
        snaps = [p for p in snap_root.iterdir() if p.is_dir()]
        if not snaps:
            continue
        return max(snaps, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        "未找到 CAPC-CG 数据集快照，已尝试:\n  " + "\n  ".join(tried)
        + "\n请先下载数据集，或设置 HF_HOME / HF_HUB_CACHE。"
    )


def _messages_to_list(messages):
    """把 numpy.ndarray / list / tuple 统一成 list[dict]。"""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, (tuple,)):
        messages = list(messages)
    if not isinstance(messages, list):
        raise TypeError(f"无法识别的 messages 类型: {type(messages)}")
    return messages


def _flatten(messages) -> dict:
    """从一条 messages 中提取 user 文本(原始+去前缀) 与 assistant 标签。"""
    msgs = _messages_to_list(messages)
    user_raw = None
    label = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            user_raw = content if isinstance(content, str) else str(content)
        elif role == "assistant":
            label = str(content).strip() if content is not None else None
    user_clean = user_raw or ""
    if user_clean.startswith(USER_PREFIX):
        user_clean = user_clean[len(USER_PREFIX):]
    user_clean = user_clean.strip()
    return {"text": user_clean, "user_raw": (user_raw or "").strip(), "label": label}


def load_split(task: str, split: str) -> list[dict]:
    """读取某个任务的某个划分，返回 [{'text','user_raw','label'}, ...]。"""
    if task not in TASK_LABELS:
        raise ValueError(f"未知任务: {task}，可选 {list(TASK_LABELS)}")
    if split not in ("train", "validation"):
        raise ValueError(f"未知划分: {split}")
    snap = find_snapshot()
    path = snap / task / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    rows = []
    for v in df["messages"]:
        rows.append(_flatten(v))
    return rows


def load_all() -> dict:
    """返回 {'task1_level1/train': [...], ...}。"""
    out = {}
    for task in TASK_LABELS:
        for split in ("train", "validation"):
            out[f"{task}/{split}"] = load_split(task, split)
    return out


def get_system_prompt(task: str, split: str = "train") -> str:
    """取该任务/划分下第一条样本的 system 提示（全表一致）。"""
    snap = find_snapshot()
    df = pd.read_parquet(snap / task / f"{split}.parquet")
    first = _messages_to_list(df.iloc[0]["messages"])
    for m in first:
        if isinstance(m, dict) and m.get("role") == "system":
            return m.get("content", "")
    return ""


def export_openai_jsonl(out_dir: str) -> Path:
    """导出 OpenAI 微调用的 chat-format JSONL（train/validation 各一份）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for task in TASK_LABELS:
        sys_prompt = get_system_prompt(task)
        for split in ("train", "validation"):
            rows = load_split(task, split)
            with open(out / f"{task}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    rec = {
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": r["user_raw"]},
                            {"role": "assistant", "content": r["label"]},
                        ]
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    print("snapshot:", find_snapshot())
    data = load_all()
    for key, rows in data.items():
        c = Counter(r["label"] for r in rows)
        missing = sum(1 for r in rows if not r["label"])
        print(f"{key:24s} n={len(rows):5d}  labels={dict(c)}  缺失标签={missing}")
