import json
import os
from openai import OpenAI
import pandas as pd

TRAIN_FILE = r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\task2_level2\train.parquet"

MODEL = "deepseek-flash"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)


def parse_messages(messages):
    if isinstance(messages, str):
        messages = json.loads(messages)

    system_prompt = ""
    user_text = ""
    label = ""

    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]
        elif msg["role"] == "assistant":
            label = msg["content"].strip()

    prefix = "待判断政策段落："
    if user_text.startswith(prefix):
        user_text = user_text[len(prefix):]

    return system_prompt, user_text.strip(), label


df = pd.read_parquet(TRAIN_FILE)

print("=" * 70)
print("DeepSeek Level-2 预实验（5条）")
print("=" * 70)

for i in range(5):

    system_prompt, text, gold = parse_messages(
        df.iloc[i]["messages"]
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "待判断政策段落：" + text
            }
        ],
        max_tokens=4,
        stream=False,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    prediction = response.choices[0].message.content.strip()

    print()
    print("-" * 70)
    print(f"样本：{i}")
    print(f"Gold：{gold}")
    print(f"预测：{prediction}")
    print("政策内容：")
    print(text[:500])

print()
print("=" * 70)
print("预实验完成")
print("=" * 70)