# ============================================================
# 实验标题：Qwen API 连通性测试
# ============================================================

import os
from openai import OpenAI


API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("未检测到 DASHSCOPE_API_KEY")


client = OpenAI(
    api_key=API_KEY,
    base_url="https://maas.qianwenaiapi.com/compatible-mode/v1"
)


response = client.chat.completions.create(
    model="qwen3.8-max",
    messages=[
        {
            "role": "user",
            "content": "请只回答：API连接成功"
        }
    ],
    max_tokens=20
)


print("=" * 60)
print("Qwen API 连通性测试")
print("=" * 60)
print("模型：", response.model)
print("回答：", response.choices[0].message.content)
print("=" * 60)