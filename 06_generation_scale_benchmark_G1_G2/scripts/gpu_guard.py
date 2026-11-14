# -*- coding: utf-8 -*-
"""06 期本地 GPU 禁用守卫（沿用项目 05 期的宗旨级约束）。

宗旨：任何时候都不得在本地显卡上做训练或测试；06 期的训练与评估只在云端
RTX 4090D 上执行。本地只允许：静态检查、纯逻辑自检、结果汇总。

人工调试可显式设置 `CAPC_ALLOW_LOCAL_GPU=1` 解除限制；该开关仅供人工使用。
"""
from __future__ import annotations

import os

ALLOW_ENV = "CAPC_ALLOW_LOCAL_GPU"
CLOUD_GPU_KEYWORDS = ("4090d",)
SMOKE_MAX_SAMPLES = 2


def is_cloud_gpu(name: str) -> bool:
    normalized = "".join(ch for ch in (name or "").casefold() if ch.isalnum())
    return any(key in normalized for key in CLOUD_GPU_KEYWORDS)


def assert_cloud_gpu_allowed(action: str = "本次运行") -> None:
    if os.environ.get(ALLOW_ENV, "").strip().lower() in {"1", "true", "yes"}:
        print(f"[人工调试豁免] {ALLOW_ENV} 已显式设置，跳过本地 GPU 限制")
        return
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(f"{action} 需要 PyTorch: {error}") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{action} 需要 CUDA 云端 GPU；本地按项目规范不得使用显卡训练/测试。"
        )
    name = torch.cuda.get_device_name(0)
    if is_cloud_gpu(name):
        return
    raise RuntimeError(
        f"检测到本地显卡 '{name}'，按项目规范禁止在本地显卡上训练/测试。\n"
        f"请在云端 RTX 4090D 运行；本地只做静态检查、单元自检与结果汇总。\n"
        f"（仅人工调试可显式设置 {ALLOW_ENV}=1）"
    )


def limit_smoke_samples(records, limit: int = SMOKE_MAX_SAMPLES):
    items = list(records)
    if limit is None or limit <= 0:
        return items
    return items[:limit]
