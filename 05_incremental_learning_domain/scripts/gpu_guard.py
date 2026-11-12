# -*- coding: utf-8 -*-
"""本地 GPU 禁用守卫（项目宗旨级约束）。

宗旨：**任何时候都不得在本地显卡上做训练或测试**。
本项目的训练与评估只能在 RTX 4090 / 4090D 上执行；其它环境仅允许：
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
# 云端 4090 / 4090D 均放行；其余（1660 Ti、3090 等）一律拒绝。
# 注：判据是"显卡够不够强"，不是"在不在云上"——本地 4090 同样会放行，请自行确认运行位置。
CLOUD_GPU_KEYWORDS = ("4090d", "4090")
SMOKE_MAX_SAMPLES = 2      # 冒烟流程：每个模型最多 2 条样本


def is_cloud_gpu(name: str) -> bool:
    """判断是否为项目允许的 4090 / 4090D 显卡。"""
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
            f"{action} 需要 CUDA 云端 GPU。本地按项目规范不得使用显卡训练/测试，请在 4090 / 4090D 机器上运行。"
        )
    # 多卡场景逐张校验：任何一张不是 4090/4090D 就拒绝，避免只查 device 0 漏检
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    blocked = [name for name in names if not is_cloud_gpu(name)]
    if not blocked:
        return
    raise RuntimeError(
        f"检测到不受支持的显卡 {blocked}，按项目规范禁止在此卡上训练/测试。\n"
        f"请在云端 RTX 4090 / 4090D 上运行；其它卡只做静态检查、单元自检与结果汇总。\n"
        f"（仅人工调试可显式设置 {ALLOW_ENV}=1）"
    )
