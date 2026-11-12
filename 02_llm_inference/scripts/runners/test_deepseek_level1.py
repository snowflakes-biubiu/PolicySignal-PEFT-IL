import os
import json
import pandas as pd
from openai import OpenAI


# =========================
# 1. 基本配置
# =========================

MODEL = "deepseek-flash"
TEST_N = 5

SNAPSHOT = (
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
)

VAL_FILE = SNAPSHOT + r"\task1_level1\validation.parquet"


# =========================
# 2. 论文 Appendix D Level-1 Prompt
# =========================

SYSTEM_PROMPT = """角色：你是一名“政策信号一级标注员”。

任务：对给定中文政策文档段落分类标注：

W = 段落中存在对政府/公共部门的肯定性行动指令（包含任何“要/应/负责/建立/加强/推进/组织/指导/监督/落实/制定/完善/开展/实施/安排”等对政府主体布置的任务或授权）。

R = 段落中存在对政府/公共部门的禁止性指令（如“严禁/不得/禁止/停止……”（由政府或其部门））或对政府及公职人员的违法行为进行处罚。

N = 无指令（仅背景介绍、原则口号、定义释义、程序性条款、日期/效力条款、对非国家主体的要求而未同时对政府布置任务或责任的内容）。

受众原则（极其重要）：只统计指向政府及其附属公共机构（包括国务院各部委、地方政府、监管机构、事业单位等）的行动/禁止为指令。

仅限制或要求企业、个人或社会组织，而没有同时给政府安排任务或责任，则记为 N。

若同时出现：例如“企业不得……；主管部门要加强监管……”，则本段视为存在对政府的指令，进入 W 或 R 的判断。

政府/公共部门的定义为中央或地方政府、公职人员或承担了国家分配的任务的附属机构。

不要只看关键词：是否出现“必须/可以/鼓励/不得”等词不等于对应标签，必须结合主语是否为政府/公共机构与句子功能来判断。

例如：“企业必须符合××条件方可办理”，若仅针对企业且未布置政府任务，则记为 N。

输出要求：只输出最终的结果W/N/R

待判断政策内容如下：
«待判断政策段落»
"""


# =========================
# 3. 创建 API 客户端
# =========================

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY 未设置")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# =========================
# 4. 读取验证集
# =========================

df = pd.read_parquet(VAL_FILE)


def parse_messages(x):
    if hasattr(x, "tolist"):
        x = x.tolist()

    if isinstance(x, str):
        x = json.loads(x)

    return x


def get_user_text(messages):
    messages = parse_messages(messages)

    for msg in messages:
        if msg.get("role") == "user":
            text = msg.get("content", "")

            # 与我们之前处理验证集时保持一致
            prefix = "待判断政策段落："
            if text.startswith(prefix):
                text = text[len(prefix):]

            return text

    raise ValueError("没有找到 user 消息")


def get_gold_label(messages):
    messages = parse_messages(messages)

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip().upper()

    raise ValueError("没有找到 assistant 标签")


# =========================
# 5. 只测试前 5 条
# =========================

test_df = df.head(TEST_N)

results = []

print("=" * 70)
print("DeepSeek Level-1 Zero-shot 小规模测试")
print("模型:", MODEL)
print("样本数:", TEST_N)
print("Thinking: disabled")
print("=" * 70)


for i, row in test_df.iterrows():

    text = get_user_text(row["messages"])
    gold = get_gold_label(row["messages"])

    user_prompt = f"""待判断政策段落：

{text}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        stream=False,
        max_tokens=4,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    prediction_raw = response.choices[0].message.content.strip()
    prediction = prediction_raw.upper()

    # 判断是否严格输出合法标签
    valid = prediction in {"W", "N", "R"}

    results.append({
        "index": int(i),
        "gold": gold,
        "prediction_raw": prediction_raw,
        "prediction": prediction if valid else "INVALID",
        "correct": prediction == gold,
        "text": text
    })

    print()
    print(f"样本 {i + 1}")
    print("人工标签:", gold)
    print("模型输出:", prediction_raw)
    print("是否正确:", prediction == gold)
    print("文本:", text[:150].replace("\n", " "))
    print("-" * 70)


# =========================
# 6. 保存结果
# =========================

result_file = "02_llm_inference/results/deepseek/deepseek_level1_test5.json"

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print()
print("=" * 70)
print("测试完成")
print("结果已保存:", result_file)

valid_results = [r for r in results if r["prediction"] != "INVALID"]

if valid_results:
    accuracy = sum(r["correct"] for r in valid_results) / len(valid_results)
    print("有效预测:", len(valid_results))
    print("5条测试中的 Accuracy:", f"{accuracy:.4f}")

print("=" * 70)