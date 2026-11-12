# -*- coding: utf-8 -*-
"""核心逻辑单元自检（无需 GPU）：Replay 定容、EWC 归一化、截断一致、Adapter 路由。

用法：python 05_incremental_learning_time/scripts/validation/selfcheck_units.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import continual_training
from continual_training import ReplayMemory, _ewc_penalty
from gpu_guard import SMOKE_MAX_SAMPLES, is_cloud_gpu
from incremental_common import (
    EVAL_MAX_LENGTH,
    LABELS,
    MAX_LENGTH,
    MODEL_NAME,
    TASKS,
    experiment_fingerprint,
    truncate_head_tail,
)
from incremental_evaluation import (
    _activate_adapters_for_task,
    _read_stage_accuracy,
    result_path as evaluation_result_path,
)
from run_full_matrix import model_complete
from unittest.mock import patch

fails = []


def ok(cond, msg):
    print(("  [OK] " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


rng = random.Random(42)
initial_replay = ReplayMemory(300, "class_balanced")
initial_replay.update(
    [{"label": lab, "text": f"T1-{lab}-{i}"} for lab in LABELS for i in range(200)],
    rng,
)
initial_sample = initial_replay.sample(200, rng)
ok(initial_replay.seen == 600 and len(initial_sample) == 200
   and all(record["text"].startswith("T1-") for record in initial_sample),
   "Replay 从 Task2 起包含 T1 基线样本")
m = ReplayMemory(300, "class_balanced")
for lab in LABELS:
    m.update([{"label": lab, "text": f"{lab}-{i}"} for i in range(600)], rng)
ok(len(m.records) == 300, f"class_balanced 定容300: {len(m.records)}")
ok({s: sum(r["label"] == s for r in m.records) for s in LABELS} == {s: 100 for s in LABELS},
   "每类等额 100")
ok(len(m.sample(200, rng)) == 200, "采样 200 条")
ok(len(m.sample(1, rng)) == 1, "class_balanced 小批量采样返回精确请求数量")
state = m.state()
restored = ReplayMemory(300, "class_balanced")
restored.load(m.records, m.seen, state["class_seen"])
next_records = [
    {"label": lab, "text": f"next-{lab}-{i}"}
    for lab in LABELS for i in range(200)
]
m_next_rng = random.Random(7)
restored_next_rng = random.Random(7)
m.update(next_records, m_next_rng)
restored.update(next_records, restored_next_rng)
ok(restored.records == m.records and restored._class_seen == m._class_seen,
   "class_balanced 状态恢复保留各类 seen 计数")
with tempfile.TemporaryDirectory() as temp_dir:
    with patch.object(continual_training, "_replay_root", return_value=Path(temp_dir)):
        saved_path = continual_training._save_replay_memory(m, 42, completed_task=2)
        loaded = continual_training._load_replay_memory(
            42, capacity=300, mode="class_balanced", completed_task=2
        )
        ok(saved_path.name == "memory_after_task2.jsonl"
           and loaded.records == m.records
           and loaded._class_seen == m._class_seen,
           "Replay 缓存按阶段持久化且可一致恢复")
with tempfile.TemporaryDirectory() as temp_dir:
    ewc_states = [{
        "task": 2,
        "anchor": {"lora_A.weight": torch.tensor([0.5])},
        "fisher": {"lora_A.weight": torch.tensor([1.0])},
    }]
    ewc_fingerprint = experiment_fingerprint(
        method="ewc",
        task=2,
        seed=42,
        run_tag=None,
        ewc_lambda=100.0,
        ewc_normalize="param",
        max_steps=None,
        base_model=MODEL_NAME,
    )
    with patch.object(continual_training, "_ewc_root", return_value=Path(temp_dir)):
        continual_training._save_ewc_state(
            2, ewc_states, fingerprint=ewc_fingerprint, seed=42
        )
        loaded_ewc_states = continual_training._load_ewc_states(
            3,
            model=None,
            tokenizer=None,
            seed=42,
            ewc_lambda=100.0,
            ewc_normalize="param",
        )
        ok(loaded_ewc_states[0]["task"] == 2,
           "EWC 前阶段 Fisher 状态可按当前配置指纹恢复")
r = ReplayMemory(300, "reservoir")
r.update([{"label": "N", "text": str(i)} for i in range(1000)], rng)
ok(len(r.records) == 300 and r.seen == 1000, f"reservoir: {len(r.records)}/{r.seen}")

p1 = torch.nn.Parameter(torch.tensor([0.5]))
p2 = torch.nn.Parameter(torch.tensor([0.25]))


class FM:
    def named_parameters(self):
        return [("lora_A.w", p1), ("lora_B.w", p2)]


st = [{"task": t,
       "anchor": {"lora_A.w": torch.tensor([v]), "lora_B.w": torch.tensor([v])},
       "fisher": {"lora_A.w": torch.tensor([1.0]), "lora_B.w": torch.tensor([1.0])}}
      for t, v in ((1, 0.0), (2, 1.0))]
raw = float(_ewc_penalty(FM(), st, "none"))
pp = float(_ewc_penalty(FM(), st, "param"))
pt = float(_ewc_penalty(FM(), st, "task"))
ok(abs(pp - raw / 2) < 1e-6, f"param 归一化=raw/2: {pp:.6f}")
ok(abs(pt - raw / 2) < 1e-6, f"task 归一化=raw/2: {pt:.6f}")
ok(EVAL_MAX_LENGTH == MAX_LENGTH - 1 == 767, f"截断一致: {EVAL_MAX_LENGTH}")


class FP:
    """模拟 PeftModel：PeftModel.set_adapter 收字符串，LoraModel.set_adapter 收列表。"""

    def __init__(self):
        self.active = None
        self.base_model = self

    def set_adapter(self, n):
        self.active = list(n) if isinstance(n, (list, tuple)) else [n]


fp = FP()
names = ["task1", "task2", "task3"]
route = _activate_adapters_for_task(fp, names, 2, "route", None)
ok(route == ["task2"] and fp.active == ["task2"], f"route 只激活对应窗口 adapter: {route}")
every = _activate_adapters_for_task(fp, names, 2, "simultaneous", None)
ok(every == names and fp.active == names, f"simultaneous 激活全部（旧行为）: {every}")

with tempfile.TemporaryDirectory() as temp_dir:
    results = Path(temp_dir)
    test_path = evaluation_result_path("ewc", 1, results, "test")
    val_path = evaluation_result_path("ewc", 1, results, "val")
    test_path.write_text(json.dumps({
        "metrics_by_test_task": {TASKS[1]["name"]: {"acc": 0.25}}
    }), encoding="utf-8")
    val_path.write_text(json.dumps({
        "metrics_by_test_task": {TASKS[1]["name"]: {"acc": 0.75}}
    }), encoding="utf-8")
    ok(_read_stage_accuracy("ewc", 1, results, "test") == {1: 0.25},
       "持续指标从指定结果目录读取 test 历史")
    ok(_read_stage_accuracy("ewc", 1, results, "val") == {1: 0.75},
       "验证集持续指标不混用 test 历史")

    adapter_bundle = results / "adapter"
    adapter_bundle.mkdir()
    (adapter_bundle / "adapter_manifest.json").write_text(json.dumps({
        "adapter_dir": "adapters",
        "adapters": ["task1"],
    }), encoding="utf-8")
    (adapter_bundle / "adapters" / "task1").mkdir(parents=True)
    (adapter_bundle / "adapters" / "task1" / "adapter_config.json").touch()
    ok(model_complete(adapter_bundle, "adapter"),
       "矩阵识别 adapter bundle manifest 为完成标记")

print("\n== 头尾保留截断 ==")
ids = list(range(1000))
out, hit = truncate_head_tail(ids, 100, head_ratio=0.5)
ok(hit is True and len(out) == 100, f"超长截断到目标长度: {len(out)}")
ok(out[:50] == list(range(50)), "头部 50 个 token 保留")
ok(out[-49:] == list(range(951, 1000)), "尾部 49 个 token 保留")
out2, hit2 = truncate_head_tail(ids, 2000)
ok(hit2 is False and out2 == ids, "未超长时原样返回")

print("\n== 本地 GPU 守卫判定（纯字符串逻辑，不初始化 CUDA） ==")
ok(is_cloud_gpu("NVIDIA GeForce RTX 4090 D") is True, "云端 4090 D 判定为允许")
ok(is_cloud_gpu("NVIDIA GeForce RTX 4090") is False, "普通 RTX 4090 不被误判为云端 4090 D")
ok(is_cloud_gpu("NVIDIA GeForce RTX 1660 Ti") is False, "本地 1660 Ti 判定为拒绝")
ok(is_cloud_gpu("NVIDIA GeForce RTX 3060") is False, "其他本地显卡判定为拒绝")
ok(SMOKE_MAX_SAMPLES == 2, f"冒烟样本上限 = {SMOKE_MAX_SAMPLES}")

print(f"\n失败项: {len(fails)}")
sys.exit(1 if fails else 0)
