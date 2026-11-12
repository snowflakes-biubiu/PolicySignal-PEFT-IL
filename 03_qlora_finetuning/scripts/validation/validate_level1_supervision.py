from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 768

VALID_LABELS = {"W", "R", "N"}


def main():

    # ============================================================
    # 1. 数据
    # ============================================================

    snapshot_dir = Path(
        snapshot_download(
            "Baron-Sun/CAPC-CG_V1.0",
            repo_type="dataset",
            local_files_only=True,
        )
    )

    train_file = (
        snapshot_dir
        / "task1_level1"
        / "train.parquet"
    )

    df = pd.read_parquet(train_file)

    # ============================================================
    # 2. Tokenizer
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # ============================================================
    # 3. 逐条构造监督数据
    # ============================================================

    all_supervised_counts = []
    all_input_lengths = []

    invalid_samples = []

    for idx, row in df.iterrows():

        messages = row["messages"]

        system_content = messages[0]["content"]
        user_content = messages[1]["content"]
        label = str(messages[2]["content"]).strip()

        if label not in VALID_LABELS:
            invalid_samples.append({
                "index": int(idx),
                "label": label,
                "reason": "非法标签",
            })
            continue

        # --------------------------------------------------------
        # Prompt
        # --------------------------------------------------------

        prompt_messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --------------------------------------------------------
        # Assistant answer
        # --------------------------------------------------------

        answer_ids = tokenizer(
            label,
            add_special_tokens=False,
        )["input_ids"]

        # 标签必须存在
        if len(answer_ids) == 0:
            invalid_samples.append({
                "index": int(idx),
                "label": label,
                "reason": "标签编码后为空",
            })
            continue

        # --------------------------------------------------------
        # Prompt tokens
        # --------------------------------------------------------

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        # --------------------------------------------------------
        # 为答案预留空间
        # --------------------------------------------------------

        max_prompt_length = MAX_LENGTH - len(answer_ids)

        if max_prompt_length <= 0:
            invalid_samples.append({
                "index": int(idx),
                "label": label,
                "reason": "MAX_LENGTH不足以容纳标签",
            })
            continue

        # --------------------------------------------------------
        # 关键：
        # 只截断 prompt，绝不截断 answer
        # --------------------------------------------------------

        if len(prompt_ids) > max_prompt_length:

            prompt_ids = prompt_ids[-max_prompt_length:]

        # --------------------------------------------------------
        # 最终 input
        # --------------------------------------------------------

        input_ids = prompt_ids + answer_ids

        # --------------------------------------------------------
        # labels
        #
        # prompt全部不计算loss
        # answer全部计算loss
        # --------------------------------------------------------

        labels = (
            [-100] * len(prompt_ids)
            + answer_ids
        )

        supervised_count = sum(
            x != -100
            for x in labels
        )

        all_supervised_counts.append(
            supervised_count
        )

        all_input_lengths.append(
            len(input_ids)
        )

        # --------------------------------------------------------
        # 强制检查
        # --------------------------------------------------------

        if supervised_count == 0:

            invalid_samples.append({
                "index": int(idx),
                "label": label,
                "reason": "监督token为0",
            })

        if len(input_ids) > MAX_LENGTH:

            invalid_samples.append({
                "index": int(idx),
                "label": label,
                "reason": "最终input超过MAX_LENGTH",
            })

    # ============================================================
    # 4. 输出最终检查结果
    # ============================================================

    print("=" * 70)
    print("Level-1 QLoRA 最终监督数据验证")
    print("=" * 70)

    print(f"总样本数: {len(df)}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")

    print()
    print("监督 token 数量:")
    print(
        f"  最小值: {min(all_supervised_counts)}"
    )
    print(
        f"  最大值: {max(all_supervised_counts)}"
    )
    print(
        f"  平均值: "
        f"{sum(all_supervised_counts) / len(all_supervised_counts):.2f}"
    )

    print()
    print("监督 token = 0:")
    print(
        f"  {sum(x == 0 for x in all_supervised_counts)} "
        f"/ {len(all_supervised_counts)}"
    )

    print()
    print("最终 input 长度:")
    print(
        f"  最小值: {min(all_input_lengths)}"
    )
    print(
        f"  最大值: {max(all_input_lengths)}"
    )

    print()
    print("非法/异常样本:")
    print(
        f"  {len(invalid_samples)}"
    )

    if invalid_samples:

        print()
        print("前20个异常样本:")

        for item in invalid_samples[:20]:
            print(item)

    # ============================================================
    # 5. 最终判定
    # ============================================================

    print()
    print("=" * 70)

    if (
        len(all_supervised_counts) == len(df)
        and all(x > 0 for x in all_supervised_counts)
        and len(invalid_samples) == 0
        and all(x <= MAX_LENGTH for x in all_input_lengths)
    ):

        print("✅ 监督数据验证通过")
        print()
        print("600条样本全部存在有效监督token。")
        print("所有最终input均不超过MAX_LENGTH。")
        print("可以进入正式QLoRA训练。")

    else:

        print("❌ 监督数据验证失败")
        print()
        print("暂时不要开始正式训练。")

    print("=" * 70)


if __name__ == "__main__":
    main()