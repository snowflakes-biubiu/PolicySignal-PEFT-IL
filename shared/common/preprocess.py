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
from pathlib import Path
from collections import Counter

import pandas as pd

# 任务 -> 论文定义的标签集合（顺序固定，便于标签编码）
TASK_LABELS = {
    "task1_level1": ["W", "R", "N"],
    "task2_level2": ["B", "Y", "C", "G"],
}

USER_PREFIX = "待判断政策段落："


def find_snapshot() -> Path:
    """定位 HuggingFace 本地缓存中的最新 snapshot 目录。"""
    root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--Baron-Sun--CAPC-CG_V1.0"
        / "snapshots"
    )
    if not root.exists():
        raise FileNotFoundError(f"未找到 HF 缓存目录: {root}")
    snaps = [p for p in root.iterdir() if p.is_dir()]
    if not snaps:
        raise FileNotFoundError("未找到任何数据集 snapshot")
    return max(snaps, key=lambda p: p.stat().st_mtime)


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
