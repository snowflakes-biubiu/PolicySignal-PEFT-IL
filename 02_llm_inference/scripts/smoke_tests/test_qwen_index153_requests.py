# ============================================================
# 实验标题：Qwen Level-1 Zero-shot index=153 requests 诊断
# ============================================================

import os
import json
import requests


API_KEY = os.getenv("DASHSCOPE_API_KEY")

URL = (
    "https://maas.qianwenaiapi.com/"
    "compatible-mode/v1/chat/completions"
)

MODEL = "qwen3.8-max"

RESULT_PATH = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300.json"


PROMPT = """角色：你是一名“政策信号一级标注员”。
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
# 读取 index=153
# ============================================================

with open(
    RESULT_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


item = next(
    x for x in data["results"]
    if x["index"] == 153
)

text = item["text"]


print("=" * 70)
print("Qwen Level-1 Zero-shot index=153 requests 诊断")
print("=" * 70)

print("文本长度：", len(text))
print("开始直接 HTTP 请求...")
print()


# ============================================================
# HTTP 请求
# ============================================================

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": PROMPT + text
        }
    ],
    "max_tokens": 10,
    "stream": False
}


try:

    response = requests.post(
        URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    print("HTTP状态码：", response.status_code)

    print()
    print("服务器响应：")
    print(response.text[:2000])

except Exception as e:

    print("HTTP请求失败：")
    print(repr(e))


print()
print("=" * 70)