# -*- coding: utf-8 -*-
"""G3 的路径与公共工具入口。

G3 复用 06_generation_scale_benchmark_G1_G2/scripts/benchmark_common.py 的编码、模型构建与评估实现，
只把「结果 / 数据 / 模型」三类路径指向 06_generation_scale_benchmark_G3，避免复制一份公共代码。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
G1G2_SCRIPTS = PROJECT_ROOT / "06_generation_scale_benchmark_G1_G2" / "scripts"
if str(G1G2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(G1G2_SCRIPTS))

import benchmark_common as C  # noqa: E402

G3_ROOT = PROJECT_ROOT / "06_generation_scale_benchmark_G3"
G3_CONFIG = G3_ROOT / "configs" / "g3_matrix.json"
G3_DATA_ROOT = G3_ROOT / "data" / "g3"
G3_RESULTS_ROOT = G3_ROOT / "results"
G3_LOG_ROOT = G3_ROOT / "logs"
G3_ASSETS_ROOT = PROJECT_ROOT / "assets" / "06_generation_scale_benchmark_G3"
G3_MODEL_ROOT = G3_ASSETS_ROOT / "models"
G3_SMOKE_ROOT = G3_ASSETS_ROOT / "smoke_tests"

G1G2_ROOT = PROJECT_ROOT / "06_generation_scale_benchmark_G1_G2"
G1G2_RESULTS_ROOT = G1G2_ROOT / "results"
G1G2_MODEL_ROOT = PROJECT_ROOT / "assets" / "06_generation_scale_benchmark_G1_G2" / "models"

# 复用的公共函数（显式导出，G3 脚本只依赖这里）
preprocess = C.preprocess
M = C.M
read_json = C.read_json
write_json = C.write_json
now_stamp = C.now_stamp
set_global_seed = C.set_global_seed
run_paths = C.run_paths
encode_record = C.encode_record
PolicyDataset = C.PolicyDataset
make_collate_fn = C.make_collate_fn
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


def read_jsonl(path: Path) -> list[dict]:
    import json

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
