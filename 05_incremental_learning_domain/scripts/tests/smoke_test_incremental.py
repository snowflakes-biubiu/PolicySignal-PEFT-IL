from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incremental_common import (
    LABELS,
    TASKS,
    adapter_dir,
    encode_record,
    run_preflight,
    run_training,
    select_balanced_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="领域增量 QLoRA 数据/模型预检与单步训练冒烟测试")
    parser.add_argument("--task", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--samples", type=int, default=8, help="冒烟训练样本数，默认 8 条")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查数据、依赖和 CUDA，不加载 tokenizer 或模型权重",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples 必须大于 0")

    datasets = run_preflight(require_cuda=not args.check_only)
    if args.task > 1 and not (adapter_dir(args.task - 1) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Task{args.task} 冒烟测试需要已训练的 Task{args.task - 1} adapter: "
            f"{adapter_dir(args.task - 1)}"
        )

    if args.check_only:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA 不可用；静态预检通过，但正式 QLoRA 和单步冒烟训练无法运行。")
        print("静态预检通过。Tokenizer、权重加载和反向传播由不带 --check-only 的单步冒烟测试验证。")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    label_token_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"] for label in LABELS
    }
    invalid_labels = {label: ids for label, ids in label_token_ids.items() if len(ids) != 1}
    if invalid_labels:
        raise ValueError(f"N/R/W 必须各自为单一 token: {invalid_labels}")
    sample_records = select_balanced_samples(datasets[args.task]["train"], args.samples)
    for record in sample_records:
        encoded = encode_record(record, tokenizer)
        if not encoded["input_ids"] or all(label == -100 for label in encoded["labels"]):
            raise ValueError("监督样本编码失败：缺少输入或有效标签 token")
    print(f"Task{args.task} prompt/tokenizer 检查通过，抽样 {len(sample_records)} 条。")
    run_training(args.task, smoke=True, sample_limit=args.samples)


if __name__ == "__main__":
    main()