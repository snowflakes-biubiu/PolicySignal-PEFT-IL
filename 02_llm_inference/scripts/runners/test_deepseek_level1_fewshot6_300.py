import json
import os
import time
from openai import OpenAI
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix


# ============================================================
# 配置
# ============================================================

TRAIN_FILE = r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\task1_level1\train.parquet"

VAL_FILE = r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b\task1_level1\validation.parquet"

CHECKPOINT_FILE = "02_llm_inference/results/deepseek/deepseek_level1_fewshot6_300_checkpoint.json"
RESULT_FILE = "02_llm_inference/results/deepseek/deepseek_level1_fewshot6_300.json"

MODEL = "deepseek-flash"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# Appendix D.1 原始 Prompt
# ============================================================

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
"""


# ============================================================
# 解析数据
# ============================================================

def parse_messages(messages):
    if isinstance(messages, str):
        messages = json.loads(messages)

    user_text = ""
    label = ""

    for msg in messages:
        if msg["role"] == "user":
            user_text = msg["content"]
        elif msg["role"] == "assistant":
            label = msg["content"].strip()

    prefix = "待判断政策段落："
    if user_text.startswith(prefix):
        user_text = user_text[len(prefix):]

    return user_text.strip(), label


# ============================================================
# k=6 Few-shot 示例
# 前3个保持之前 k=3 完全不变
# 新增第2个 N/R/W
# ============================================================

FEWSHOT_EXAMPLES = [
    {
        "label": "N",
        "text": """第二十一条　企业所得税法第六条第（八）项所称接受捐赠收入，是指企业接受的来自其他企业、组织或者个人无偿给予的货币性资产、非货币性资产。 接受捐赠收入，按照实际收到捐赠资产的日期确认收入的实现。"""
    },
    {
        "label": "N",
        "text": """第二十八条　企业发生的支出应当区分收益性支出和资本性支出。收益性支出在发生当期直接扣除；资本性支出应当分期扣除或者计入有关资产成本，不得在发生当期直接扣除。 企业的不征税收入用于支出所形成的费用或者财产，不得扣除或者计算对应的折旧、摊销扣除。
　　除企业所得税法和本条例另有规定外，企业实际发生的成本、费用、税金、损失和其他支出，不得重复扣除。"""
    },
    {
        "label": "R",
        "text": """（一）这次清查主要是一九八八年和一九八九年列入“小金库”的各项收支，以及历年滚存的“小金库”的结余。自国务院国发（１９８９）７７号文件发布之日起，“小金库”的各项资金一律停止支付。违者从严查处。"""
    },
    {
        "label": "R",
        "text": """第二十六条　公安机关的工作人员在执行《中华人民共和国公民出境入境管理法》和本实施细则时，如有利用职权索取、收受贿赂或者有其他违法失职行为，情节轻微的，由主管部门酌情予以行政处分；情节严重，构成犯罪的，依照《中华人民共和国刑法》和《全国人民代表大会常务委员会关于严惩组织、运送他人偷越（国）境犯罪的补充规定》的有关条款的规定追究刑事责任。"""
    },
    {
        "label": "W",
        "text": """第二十一条　登记事项自记载于不动产登记簿时完成登记。 不动产登记机构完成登记，应当依法向申请人核发不动产权属证书或者登记证明。"""
    },
    {
        "label": "W",
        "text": """第三十五条　责任疫情报告人发现甲类传染病和乙类传染病中的艾滋病、肺炭疽的病人、病原携带者和疑似传染病病人时，城镇于六小时内，农村于十二小时内，以最快的通讯方式向发病地的卫生防疫机构报告，并同时报出传染病报告卡。 责任疫情报告人发现乙类传染病病人、病原携带者和疑似传染病病人时，城镇于十二小时内，农村于二十四小时内向发病地的卫生防疫机构报出传染病报告卡。
　　责任疫情报告人在丙类传染病监测区内发现丙类传染病病人时，应当在二十四小时内向发病地的卫生防疫机构报出传染病报告卡。"""
    }
]


# ============================================================
# 构造 Few-shot messages
# ============================================================

def build_messages(text):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    for example in FEWSHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": "待判断政策段落：" + example["text"]
        })
        messages.append({
            "role": "assistant",
            "content": example["label"]
        })

    messages.append({
        "role": "user",
        "content": "待判断政策段落：" + text
    })

    return messages


# ============================================================
# 读取验证集
# ============================================================

df = pd.read_parquet(VAL_FILE)

samples = []

for i, row in df.iterrows():
    text, gold = parse_messages(row["messages"])

    samples.append({
        "index": int(i),
        "text": text,
        "gold": gold
    })

samples = samples[:300]

print("=" * 70)
print("DeepSeek Level-1 Few-shot k=6")
print("=" * 70)
print(f"模型：{MODEL}")
print(f"验证集：300")
print(f"Few-shot：6（N×2、R×2、W×2）")
print("=" * 70)


# ============================================================
# 加载 checkpoint
# ============================================================

results = []

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

print(f"已有 checkpoint：{len(results)} 条")

completed_indices = {x["index"] for x in results}


# ============================================================
# API 调用
# ============================================================

for n, sample in enumerate(samples, start=1):

    if sample["index"] in completed_indices:
        continue

    print(f"[{n}/300] 正在处理验证集索引 {sample['index']} ...")

    messages = build_messages(sample["text"])

    success = False

    for attempt in range(3):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=4,
                stream=False,
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                }
            )

            prediction = response.choices[0].message.content.strip()

            # 只接受 N/R/W
            if prediction not in ["N", "R", "W"]:
                if "N" in prediction:
                    prediction = "N"
                elif "R" in prediction:
                    prediction = "R"
                elif "W" in prediction:
                    prediction = "W"
                else:
                    prediction = ""

            result = {
                "index": sample["index"],
                "text": sample["text"],
                "gold": sample["gold"],
                "prediction": prediction
            }

            results.append(result)

            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            success = True
            break

        except Exception as e:
            print(f"  API失败，第 {attempt + 1}/3 次：{e}")
            time.sleep(2)

    if not success:
        print("  本条连续失败，跳过。")

    time.sleep(0.1)


# ============================================================
# 统计
# ============================================================

valid_results = [
    x for x in results
    if x["prediction"] in ["N", "R", "W"]
]

y_true = [x["gold"] for x in valid_results]
y_pred = [x["prediction"] for x in valid_results]

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=["N", "R", "W"],
    average="macro"
)
kappa = cohen_kappa_score(y_true, y_pred, labels=["N", "R", "W"])

print()
print("=" * 70)
print("DeepSeek Level-1 Few-shot k=6 实验")
print("=" * 70)

print(f"有效样本：{len(valid_results)}")
print(f"Accuracy ：{accuracy:.4f}")
print(f"Macro-F1 ：{macro_f1:.4f}")
print(f"Kappa    ：{kappa:.4f}")

print()
print("Classification Report")
print(
    classification_report(
        y_true,
        y_pred,
        labels=["N", "R", "W"],
        digits=4
    )
)

print("Confusion Matrix [N, R, W]")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=["N", "R", "W"]
    )
)


# ============================================================
# 保存最终结果
# ============================================================

output = {
    "model": MODEL,
    "experiment": "Level-1 Few-shot k=6",
    "n_samples": len(valid_results),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "kappa": kappa,
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=["N", "R", "W"],
        output_dict=True
    ),
    "confusion_matrix": confusion_matrix(
        y_true,
        y_pred,
        labels=["N", "R", "W"]
    ).tolist(),
    "results": valid_results
}

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print()
print(f"结果已保存：{RESULT_FILE}")
print(f"Checkpoint：{CHECKPOINT_FILE}")