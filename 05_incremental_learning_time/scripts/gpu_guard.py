# -*- coding: utf-8 -*-
"""本地 GPU 禁用守卫（项目宗旨级约束）。

宗旨：**任何时候都不得在本地显卡上做训练或测试**。
本项目的训练与评估只能在云端 RTX 4090D 上执行；本地仅允许：
    - 代码静态检查（py_compile / lint）
    - 纯逻辑单元自检（scripts/validation/selfcheck_units.py，不加载模型）
    - 对既有结果文件的读取、汇总与制表

实现：训练与评估入口在启动时调用 assert_cloud_gpu_allowed()，
检测到非云端 GPU 时直接退出，不做任何计算。
若确需人工在本地调试，可显式设置环境变量 CAPC_ALLOW_LOCAL_GPU=1；
该开关仅供人工使用，自动化助手不得使用。
"""
from __future__ import annotations

import os

ALLOW_ENV = "CAPC_ALLOW_LOCAL_GPU"
CLOUD_GPU_KEYWORDS = ("4090d",)
SMOKE_MAX_SAMPLES = 2      # 冒烟流程：每个模型最多 2 条样本


def is_cloud_gpu(name: str) -> bool:
    """判断是否为项目允许的 RTX 4090 D GPU，避免放行普通本地 RTX 4090。"""
    normalized_name = "".join(character for character in (name or "").casefold()
                              if character.isalnum())
    return any(key in normalized_name for key in CLOUD_GPU_KEYWORDS)


def assert_cloud_gpu_allowed(action: str = "本次运行") -> None:
    """训练/评估入口守卫：本地显卡一律拒绝。"""
    if os.environ.get(ALLOW_ENV, "").strip().lower() in {"1", "true", "yes"}:
        print(f"[人工调试豁免] {ALLOW_ENV} 已显式设置，跳过本地 GPU 限制（自动化助手不得使用）")
        return
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(f"{action} 需要 PyTorch: {error}") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{action} 需要 CUDA 云端 GPU。本地按项目规范不得使用显卡训练/测试，请在云端 4090D 上运行。"
        )
    name = torch.cuda.get_device_name(0)
    if is_cloud_gpu(name):
        return
    raise RuntimeError(
        f"检测到本地显卡 '{name}'，按项目规范禁止在本地显卡上训练/测试。\n"
        f"请在云端 RTX 4090D 上运行；本地只做静态检查、单元自检与结果汇总。\n"
        f"（仅人工调试可显式设置 {ALLOW_ENV}=1）"
    )


def limit_smoke_samples(records, limit: int = SMOKE_MAX_SAMPLES):
    """冒烟流程取样上限：每个模型最多 limit 条样本。"""
    items = list(records)
    if limit is None or limit <= 0:
        return items
    return items[:limit]
