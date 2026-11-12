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

    row = df.iloc[0]
    messages = row["messages"]

    print("=" * 70)
    print("1. 原始 messages")
    print("=" * 70)

    print(type(messages))
    print(messages)

    print()
    print("=" * 70)
    print("2. 每条 message")
    print("=" * 70)

    for i, message in enumerate(messages):
        print(f"\n--- message {i} ---")
        print("type:", type(message))
        print(message)

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

    print()
    print("=" * 70)
    print("3. 提取出的内容")
    print("=" * 70)

    print("\nSystem:")
    print(repr(system_content))

    print("\nUser:")
    print(repr(user_content))

    print("\nLabel:")
    print(repr(label))

    print()
    print("=" * 70)
    print("4. Chat Template 渲染结果")
    print("=" * 70)

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

    print("\nPrompt text:")
    print(prompt_text)

    print("\nFull text:")
    print(full_text)

    print()
    print("=" * 70)
    print("5. Tokenizer 编码结果")
    print("=" * 70)

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

    print("\nPrompt token数:", len(prompt_ids))
    print("Full token数:", len(full_ids))
    print("Label token数:", len(label_ids))

    print("\nLabel token IDs:")
    print(label_ids)

    print("\nLabel token:")
    print(
        tokenizer.convert_ids_to_tokens(label_ids)
    )

    print()
    print("=" * 70)
    print("6. tokenizer 基本信息")
    print("=" * 70)

    print("vocab_size:", tokenizer.vocab_size)
    print("eos_token:", repr(tokenizer.eos_token))
    print("eos_token_id:", tokenizer.eos_token_id)
    print("pad_token:", repr(tokenizer.pad_token))
    print("pad_token_id:", tokenizer.pad_token_id)

    print("=" * 70)


if __name__ == "__main__":
    main()