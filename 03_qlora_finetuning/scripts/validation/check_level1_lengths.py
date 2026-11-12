from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():

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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    prompt_lengths = []
    full_lengths = []
    label_lengths = []

    for _, row in df.iterrows():

        messages = row["messages"]

        system_content = messages[0]["content"]
        user_content = messages[1]["content"]
        label = str(messages[2]["content"]).strip()

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

        full_messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": label,
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        label_ids = tokenizer(
            label,
            add_special_tokens=False,
        )["input_ids"]

        prompt_lengths.append(len(prompt_ids))
        full_lengths.append(len(full_ids))
        label_lengths.append(len(label_ids))

    prompt_lengths.sort()
    full_lengths.sort()

    def percentile(values, p):
        index = int((len(values) - 1) * p)
        return values[index]

    print("=" * 70)
    print("Level-1 QLoRA 真实 Token 长度统计")
    print("=" * 70)

    print(f"总样本数: {len(df)}")

    print("\nPrompt token 长度:")
    print(f"  最小值: {min(prompt_lengths)}")
    print(f"  P50:    {percentile(prompt_lengths, 0.50)}")
    print(f"  P75:    {percentile(prompt_lengths, 0.75)}")
    print(f"  P90:    {percentile(prompt_lengths, 0.90)}")
    print(f"  P95:    {percentile(prompt_lengths, 0.95)}")
    print(f"  P99:    {percentile(prompt_lengths, 0.99)}")
    print(f"  最大值: {max(prompt_lengths)}")

    print("\nFull conversation token 长度:")
    print(f"  最小值: {min(full_lengths)}")
    print(f"  P50:    {percentile(full_lengths, 0.50)}")
    print(f"  P75:    {percentile(full_lengths, 0.75)}")
    print(f"  P90:    {percentile(full_lengths, 0.90)}")
    print(f"  P95:    {percentile(full_lengths, 0.95)}")
    print(f"  P99:    {percentile(full_lengths, 0.99)}")
    print(f"  最大值: {max(full_lengths)}")

    print("\n不同 MAX_LENGTH 下的截断情况:")

    for max_length in [256, 384, 512, 640, 768, 1024]:

        truncated = sum(
            x > max_length
            for x in full_lengths
        )

        ratio = truncated / len(full_lengths)

        print(
            f"  MAX_LENGTH={max_length:4d}: "
            f"{truncated:3d}/{len(full_lengths)} "
            f"({ratio:.2%}) 会被截断"
        )

    print("\n标签 token 长度:")
    print(f"  最小值: {min(label_lengths)}")
    print(f"  最大值: {max(label_lengths)}")

    print("=" * 70)


if __name__ == "__main__":
    main()