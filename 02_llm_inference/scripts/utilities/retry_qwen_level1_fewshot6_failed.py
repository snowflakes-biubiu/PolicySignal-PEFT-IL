# ============================================================
# 实验标题：Qwen Level-1 Few-shot k=6 失败样本补跑实验
# ============================================================

import json
import os
import time
from openai import OpenAI


INPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_fewshot6_300.json"
OUTPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_fewshot6_300_fixed.json"

MODEL = "qwen3.8-max"
BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

VALID_LABELS = {"N", "R", "W"}

TARGET_INDICES = {103, 248}


# ============================================================
# Level-1 官方实验 Prompt
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
"""


# ============================================================
# Few-shot k=6 示例
# ============================================================

FEWSHOT_EXAMPLES = [
    (
        "N",
        "第二十一条　企业所得税法第六条第（八）项所称接受捐赠收入，是指企业接受的来自其他企业、组织或者个人无偿给予的货币性资产、非货币性资产。接受捐赠收入，按照实际收到捐赠资产的日期确认收入的实现。"
    ),
    (
        "N",
        "第二十八条　企业发生的支出应当区分收益性支出和资本性支出。收益性支出在发生当期直接扣除；资本性支出应当分期扣除或者计入有关资产成本，不得在发生当期直接扣除。企业的不征税收入用于支出所形成的费用或者财产，不得扣除或者计算对应的折旧、摊销扣除。除企业所得税法和本条例另有规定外，企业实际发生的成本、费用、税金、损失和其他支出，不得重复扣除。"
    ),
    (
        "R",
        "（一）这次清查主要是一九八八年和一九八九年列入“小金库”的各项收支，以及历年滚存的“小金库”的结余。自国务院国发（１９８９）７７号文件发布之日起，“小金库”的各项资金一律停止支付。违者从严查处。"
    ),
    (
        "R",
        "第二十六条　公安机关的工作人员在执行《中华人民共和国公民出境入境管理法》和本实施细则时，如有利用职权索取、收受贿赂或者有其他违法失职行为，情节轻微的，由主管部门酌情予以行政处分；情节严重，构成犯罪的，依照《中华人民共和国刑法》和《全国人民代表大会常务委员会关于严惩组织、运送他人偷越（国）边境犯罪的补充规定》的有关条款的规定追究刑事责任。"
    ),
    (
        "W",
        "第二十一条　登记事项自记载于不动产登记簿时完成登记。不动产登记机构完成登记，应当依法向申请人核发不动产权属证书或者登记证明。"
    ),
    (
        "W",
        "第三十五条　责任疫情报告人发现甲类传染病和乙类传染病中的艾滋病、肺炭疽的病人、病原携带者和疑似传染病病人时，城镇于六小时内，农村于十二小时内，以最快的通讯方式向发病地的卫生防疫机构报告，并同时报出传染病报告卡。责任疫情报告人在丙类传染病监测区内发现丙类传染病病人时，应当在二十四小时内向发病地的卫生防疫机构报出传染病报告卡。"
    ),
]


def build_user_prompt(text):
    examples = []

    for label, example_text in FEWSHOT_EXAMPLES:
        examples.append(
            f"示例：\n"
            f"政策内容：{example_text}\n"
            f"标注结果：{label}"
        )

    examples_text = "\n\n".join(examples)

    return (
        examples_text
        + "\n\n"
        + "现在请判断：\n"
        + "待判断政策段落："
        + text
    )


# ============================================================
# 初始化 API
# ============================================================

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise RuntimeError("未检测到 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
    timeout=60,
    max_retries=0,
)


# ============================================================
# 读取原始结果
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]

target_items = [
    item for item in results
    if item.get("index") in TARGET_INDICES
]

print("=" * 70)
print("实验标题：Qwen Level-1 Few-shot k=6 失败样本补跑实验")
print("=" * 70)

print("待补跑样本：", len(target_items))

for item in target_items:
    print(
        f"index={item.get('index')}, "
        f"gold={item.get('gold')}"
    )


# ============================================================
# 补跑
# ============================================================

for item in target_items:

    index = item["index"]
    text = item["text"]

    print()
    print("-" * 70)
    print(f"正在补跑 index={index}")
    print(f"Gold：{item['gold']}")

    success = False

    for attempt in range(1, 4):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(text),
                    },
                ],
                max_tokens=4,
                stream=False,
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                },
            )

            raw_output = (
                response.choices[0]
                .message
                .content
                .strip()
            )

            prediction = None

            for label in ["W", "R", "N"]:
                if label in raw_output:
                    prediction = label
                    break

            if prediction not in VALID_LABELS:
                raise ValueError(
                    f"无法解析有效标签：{raw_output!r}"
                )

            item["prediction"] = prediction
            item["raw_output"] = raw_output
            item["error"] = None
            item["error_detail"] = None
            item["attempt"] = f"retry_{attempt}"

            print(f"成功：{prediction}")
            print(f"原始输出：{raw_output!r}")

            success = True
            break

        except Exception as e:

            print(
                f"第 {attempt} 次失败："
                f"{type(e).__name__}: {e}"
            )

            if attempt < 3:
                time.sleep(2)


    if not success:

        print(f"index={index} 补跑仍然失败")


# ============================================================
# 保存修复后的完整结果
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 70)
print("补跑完成")
print("=" * 70)

print("输出文件：", OUTPUT_FILE)