"""Joint Training 上界参照：把全部窗口/领域训练数据混合，从基座一次性训练。

零侵入设计（重要）
------------------
本文件是**新增**文件，不修改 incremental_common.py / continual_training.py /
incremental_evaluation.py / gpu_guard.py / run_full_matrix.py 这五个被
code_version() 哈希的文件——否则历史训练产物的实验指纹会全部失效，
--skip-existing 将拒绝复用既有 E1–E4 结果。

实现方式：复用 continual_training.run_continual_training(method="cumulative")
的完整训练链路（tokenization / Trainer / 保存 / metadata / 指纹全部一致），
仅在本次进程内临时替换其 task_file()：让 T1 指向合并训练集、T2..TN 指向
空占位文件，从而「一次性」读入全部数据。合并数据写入 artifacts/joint_data/
（不在 data/ 下），避免 data_version() 因新增 jsonl 而变化。

两组实验
--------
--size full  全部训练数据（时间 3000 / 领域 3600 条），2 epoch = 1500 / 1800 步，
             与 E1 全链累计步数相同（等算力上界）
--size 600   每窗口/域按类别均衡抽足 600 条，2 epoch = 300 步，
             作为「数据量对照」，训练超参与 E1–E4 完全一致

时间增量：5 窗口 × 3 类 × 40 条 = 600
领域增量：6 领域 × 100 条，每类 34/33/33 且余数按域轮转 → 每类严格 200 条

用法
----
python scripts/train/train_joint.py --size 600 --prepare-only     # 离线验证抽样配额
python scripts/train/train_joint.py --size 600 --tag joint600      # 云端训练
python scripts/train/train_joint.py --size full --tag jointfull    # 云端训练
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import incremental_common as ic  # noqa: E402

JOINT_ROOT = ic.INCREMENTAL_ROOT / "data" / "joint"
TOTAL_SMALL = 600


def _sources() -> list[tuple[int, Path]]:
    return [(i, ic.task_file(i, "train")) for i in range(1, len(ic.TASKS) + 1)]


def build_records(size: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构造合并训练集；按 (任务, 类别) 分层抽样，全局类别严格均衡。"""
    rng = random.Random(seed)
    sources = _sources()
    by_task: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for i, path in sources:
        buckets = {label: [] for label in ic.LABELS}
        for record in ic.read_jsonl(path):
            buckets.setdefault(record.get("label"), []).append(record)
        for label in ic.LABELS:
            if len(buckets[label]) < TOTAL_SMALL // len(sources) // len(ic.LABELS):
                raise ValueError(f"{path.name} 的 {label} 类样本不足：{len(buckets[label])}")
            rng.shuffle(buckets[label])
        by_task[i] = buckets

    quota: dict[str, dict[str, int]] = {}
    selected: list[dict[str, Any]] = []
    if size == "full":
        for i, _ in sources:
            quota[str(i)] = {label: len(by_task[i][label]) for label in ic.LABELS}
            selected.extend(r for label in ic.LABELS for r in by_task[i][label])
    else:
        per_task, extra = divmod(TOTAL_SMALL, len(sources))
        n_labels = len(ic.LABELS)
        for idx, (i, _) in enumerate(sources):
            target = per_task + (1 if idx < extra else 0)
            base, rem = divmod(target, n_labels)
            # 余数按任务轮转：6 个领域时 N/R/W 各多取 2 次 → 每类严格 200 条
            extra_slots = {(idx + j) % n_labels for j in range(rem)}
            counts = {label: base + (1 if k in extra_slots else 0)
                      for k, label in enumerate(ic.LABELS)}
            for label in ic.LABELS:
                selected.extend(by_task[i][label][:counts[label]])
            quota[str(i)] = counts
    rng.shuffle(selected)

    dist = Counter(r.get("label") for r in selected)
    info = {
        "size": size,
        "seed": seed,
        "n_tasks": len(sources),
        "total": len(selected),
        "per_task_class_quota": quota,
        "label_totals": {label: dist[label] for label in ic.LABELS},
        "source_files": [str(p) for _, p in sources],
        "sample_ids": [r.get("DocumentID") or r.get("DocumentKey") for r in selected],
    }
    return selected, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("full", "600"), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=ic.DEFAULT_SEED)
    ap.add_argument("--prepare-only", action="store_true",
                    help="只生成合并训练集与抽样清单，不训练（可离线运行）")
    args = ap.parse_args()
    tag = args.tag or ("jointfull" if args.size == "full" else "joint600")

    JOINT_ROOT.mkdir(parents=True, exist_ok=True)
    joint_file = JOINT_ROOT / f"joint_{tag}.jsonl"
    empty_file = JOINT_ROOT / "_placeholder_empty.jsonl"
    empty_file.touch()

    records, info = build_records(args.size, args.seed)
    # 固定 LF：否则 Windows 会写 CRLF，与云端 Linux 生成的文件字节不同，
    # 导致跨环境 sha256 校验失败（训练数据本身解析后等价，但无法做一致性核对）
    with joint_file.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    info["joint_file"] = str(joint_file)
    info["sha256"] = hashlib.sha256(joint_file.read_bytes()).hexdigest()
    (JOINT_ROOT / f"joint_{tag}_manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{tag}] size={args.size}  {info['total']} 条  "
          f"类别分布={info['label_totals']}  步数(2 epoch)={info['total'] * ic.NUM_EPOCHS // ic.BATCH_SIZE}")
    print(f"  每任务×类别配额: {info['per_task_class_quota']}")
    print(f"  数据: {joint_file}")
    if args.prepare_only:
        return 0

    import continual_training as ct  # noqa: E402
    real_task_file = ct.task_file

    def joint_task_file(n: int, split: str) -> Path:
        if split == "train":
            return joint_file if n == 1 else empty_file
        return real_task_file(n, split)

    ct.task_file = joint_task_file
    ct.run_continual_training(method="cumulative", task_number=len(ic.TASKS),
                              seed=args.seed, run_tag=tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
