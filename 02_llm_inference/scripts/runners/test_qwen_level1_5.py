# ============================================================
# 实验标题：Qwen Level-1 Zero-shot 预实验
# ============================================================

import os
import json
import time
import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 基本配置
# ============================================================

API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("未检测到 DASHSCOPE_API_KEY")

MODEL = "qwen3.8-max"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

# Level-1 验证集
VAL_PATH = (
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task1_level1\validation.parquet"
)

OUTPUT_PATH = "02_llm_inference/results/qwen/qwen_level1_zero_shot_5.json"


# ============================================================
# 2. Appendix D.1 原始 Prompt
# ============================================================

SYSTEM_PROMPT = """角色：你是一名“政策信号一级标注员”。
任务：对给定中文政策文档段落分类标注：

W = 段落中存在对政府/公共部门的肯定性行动指令（包含任何“要/应/负责/建立/加强/推进/组织/指导/监督/落实/制定/完善/开展/实施/安排”等对政府主体布置的任务或授权）。

R = 段落中存在对政府/公共部门的禁止性指令（如“严禁/不得/禁止/停止……”（由政府或其部门））或对政府及公职人员的违法行为进行处罚。

N = 无指令（仅背景介绍、原则口号、定义释义、程序性条款、日期/效力条款、对非国家主体的要求而未同时对政府布置任务或责任的内容）。

受众原则（极其重要）：只统计指向政府及其附属公共机构（包括国务院各部委、地方政府、监管机构、事业单位等）的行动/禁止为指令。

仅限制或要求企业、个人或社会组织，而没有同时对政府安排任务或责任，则记为 N

若同时出现：例如“企业不得……；主管部门要加强监管……”，则本段视为存在对政府的指令，进入 W 或 R 的判断。

政府/公共部门的定义为中央或地方政府、公职人员或承担了国家分配的任务的附属机构

不要只看关键词：是否出现“必须/可以/鼓励/不得”等词不等于对应标签，必须结合主语是否为政府/公共机构与句子功能来判断。

例如：“企业必须符合××条件方可办理”，若仅针对企业且未布置政府任务，则记为 N。

输出要求：只输出最终的结果W/N/R

待判断政策内容如下：
"""


# ============================================================
# 3. 初始化客户端
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ============================================================
# 4. 读取验证集
# ============================================================

df = pd.read_parquet(VAL_PATH)

print("=" * 70)
print("实验标题：Qwen Level-1 Zero-shot 预实验")
print("=" * 70)
print("模型：", MODEL)
print("验证集总数：", len(df))
print()


# ============================================================
# 5. 解析 messages
# ============================================================

def extract_user_text(messages):
    for message in messages:
        if message.get("role") == "user":
            text = message.get("content", "")
            return text.replace("待判断政策段落：", "", 1).strip()

    raise ValueError("未找到 user 消息")


# ============================================================
# 6. 标签清洗
# ============================================================

def normalize_label(text):
    if not text:
        return None

    text = text.strip().upper()

    # 只接受明确的单标签
    if text in {"W", "N", "R"}:
        return text

    # 处理少量常见格式
    text = text.replace("：", "").replace(":", "").strip()

    if text in {"W", "N", "R"}:
        return text

    return None


# ============================================================
# 7. 运行 5 条预实验
# ============================================================

results = []

for i in range(5):

    row = df.iloc[i]

    gold = None
    messages = row["messages"]

    for message in messages:
        if message.get("role") == "assistant":
            gold = message.get("content", "").strip().upper()
            break

    policy_text = extract_user_text(messages)

    print("-" * 70)
    print(f"样本 {i + 1}/5")
    print("真实标签：", gold)
    print("文本：", policy_text[:200].replace("\n", " "))

    user_prompt = SYSTEM_PROMPT + policy_text

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=10,
            stream=False
        )

        raw_output = response.choices[0].message.content
        prediction = normalize_label(raw_output)

        print("模型原始输出：", repr(raw_output))
        print("解析标签：", prediction)

        results.append({
            "index": int(i),
            "gold": gold,
            "prediction": prediction,
            "raw_output": raw_output,
            "text": policy_text
        })

    except Exception as e:

        print("API错误：", repr(e))

        results.append({
            "index": int(i),
            "gold": gold,
            "prediction": None,
            "raw_output": None,
            "text": policy_text,
            "error": repr(e)
        })

    time.sleep(0.5)


# ============================================================
# 8. 保存结果
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "experiment": "Qwen Level-1 Zero-shot 预实验",
            "model": MODEL,
            "num_samples": len(results),
            "results": results
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 9. 简单统计
# ============================================================

valid = [
    r for r in results
    if r.get("prediction") in {"W", "N", "R"}
]

correct = [
    r for r in valid
    if r["prediction"] == r["gold"]
]

print()
print("=" * 70)
print("预实验结果")
print("=" * 70)
print("有效输出：", len(valid), "/", len(results))
print("正确数量：", len(correct))
print(
    "5条样本准确率：",
    f"{len(correct) / len(valid):.4f}"
    if valid else "N/A"
)
print("=" * 70)