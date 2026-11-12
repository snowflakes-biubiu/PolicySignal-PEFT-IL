# ============================================================
# 实验标题：Qwen Level-1 Zero-shot 失败样本补跑
# ============================================================

import os
import json
import time

from openai import OpenAI


# ============================================================
# 1. 配置
# ============================================================

API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("未检测到 DASHSCOPE_API_KEY")

MODEL = "qwen3.8-max"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

RESULT_PATH = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300.json"

MAX_RETRIES = 5


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
# 3. 客户端
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ============================================================
# 4. 读取已有结果
# ============================================================

with open(
    RESULT_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


results = data["results"]


# ============================================================
# 5. 找出 API_FAILED 样本
# ============================================================

failed_items = [
    item
    for item in results
    if item.get("error") == "API_FAILED_AFTER_RETRIES"
]


print("=" * 70)
print("实验标题：Qwen Level-1 Zero-shot 失败样本补跑")
print("=" * 70)

print("发现失败样本：", len(failed_items))

for item in failed_items:

    print(
        f"index={item['index']}, "
        f"gold={item['gold']}"
    )

print()


# ============================================================
# 6. 补跑
# ============================================================

for item in failed_items:

    index = item["index"]

    policy_text = item["text"]

    print("-" * 70)
    print(f"正在补跑样本：{index}")
    print("真实标签：", item["gold"])

    success = False

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content":
                            SYSTEM_PROMPT + policy_text
                    }
                ],
                max_tokens=10,
                stream=False
            )

            raw_output = (
                response
                .choices[0]
                .message
                .content
            )

            prediction = raw_output.strip().upper()

            print(
                "模型原始输出：",
                repr(raw_output)
            )

            print(
                "解析标签：",
                prediction
            )

            if prediction not in {"W", "N", "R"}:

                print("输出不是有效标签。")

                time.sleep(1)

                continue

            # ------------------------------------------------
            # 更新原结果
            # ------------------------------------------------

            item["prediction"] = prediction
            item["raw_output"] = raw_output

            if "error" in item:
                del item["error"]

            success = True

            break

        except Exception as e:

            print(
                f"API错误，第 {attempt}/{MAX_RETRIES} 次：",
                repr(e)
            )

            if attempt < MAX_RETRIES:

                time.sleep(3)


    if not success:

        print(
            f"样本 {index} 本次仍然失败。"
        )

    else:

        print(
            f"样本 {index} 补跑成功。"
        )


# ============================================================
# 7. 保存
# ============================================================

data["results"] = results

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 70)
print("补跑完成")
print("=" * 70)

for item in results:

    if item["index"] in {
        x["index"]
        for x in failed_items
    }:

        print(
            "index:",
            item["index"],
            "| gold:",
            item["gold"],
            "| prediction:",
            item.get("prediction")
        )

print()
print("已更新：")
print(RESULT_PATH)
print("=" * 70)