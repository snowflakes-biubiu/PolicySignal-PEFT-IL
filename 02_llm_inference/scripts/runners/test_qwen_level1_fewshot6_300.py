# ============================================================
# 实验标题：Qwen Level-1 Few-shot k=6 实验
# ============================================================

import os
import json
import time
import re

import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 基本配置
# ============================================================

MODEL = "qwen3.8-max"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

API_KEY = os.getenv("DASHSCOPE_API_KEY")

VALIDATION_PATH = (
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task1_level1\validation.parquet"
)

OUTPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_fewshot6_300.json"

CHECKPOINT_FILE = "02_llm_inference/results/qwen/qwen_level1_fewshot6_300_checkpoint.json"

EXPERIMENT_TITLE = "Qwen Level-1 Few-shot k=6 实验"

VALID_LABELS = {"N", "R", "W"}

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2


# ============================================================
# API Key
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "未检测到 DASHSCOPE_API_KEY，请先设置 Qwen API Key。"
    )


# ============================================================
# OpenAI-compatible Client
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60,
    max_retries=0,
)


# ============================================================
# Level-1 官方 Appendix D.1 Prompt
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
# k=6 固定 Few-shot 示例
#
# 与之前 DeepSeek Level-1 Few-shot k=6 保持一致
# 每个类别 2 个，共 6 个
# ============================================================

FEWSHOT_EXAMPLES = [

    # ---------------- N ----------------

    {
        "label": "N",
        "text": (
            "第二十一条　企业所得税法第六条第（八）项所称接受捐赠收入，"
            "是指企业接受的来自其他企业、组织或者个人无偿给予的货币性资产、"
            "非货币性资产。接受捐赠收入，按照实际收到捐赠资产的日期确认收入的实现。"
        ),
    },

    {
        "label": "N",
        "text": (
            "第二十八条　企业发生的支出应当区分收益性支出和资本性支出。"
            "收益性支出在发生当期直接扣除；资本性支出应当分期扣除或者计入有关资产成本，"
            "不得在发生当期直接扣除。企业的不征税收入用于支出所形成的费用或者财产，"
            "不得扣除或者计算对应的折旧、摊销扣除。"
            "除企业所得税法和本条例另有规定外，企业实际发生的成本、费用、税金、"
            "损失和其他支出，不得重复扣除。"
        ),
    },

    # ---------------- R ----------------

    {
        "label": "R",
        "text": (
            "（一）这次清查主要是一九八八年和一九八九年列入“小金库”的各项收支，"
            "以及历年滚存的“小金库”的结余。自国务院国发（１９８９）７７号文件发布之日起，"
            "“小金库”的各项资金一律停止支付。违者从严查处。"
        ),
    },

    {
        "label": "R",
        "text": (
            "第二十六条　公安机关的工作人员在执行《中华人民共和国公民出境入境管理法》"
            "和本实施细则时，如有利用职权索取、收受贿赂或者有其他违法失职行为，"
            "情节轻微的，由主管部门酌情予以行政处分；情节严重，构成犯罪的，"
            "依照《中华人民共和国刑法》和《全国人民代表大会常务委员会关于严惩组织、"
            "运送他人偷越（国）边境犯罪的补充规定》的有关条款的规定追究刑事责任。"
        ),
    },

    # ---------------- W ----------------

    {
        "label": "W",
        "text": (
            "第二十一条　登记事项自记载于不动产登记簿时完成登记。"
            "不动产登记机构完成登记，应当依法向申请人核发不动产权属证书或者登记证明。"
        ),
    },

    {
        "label": "W",
        "text": (
            "第三十五条　责任疫情报告人发现甲类传染病和乙类传染病中的艾滋病、"
            "肺炭疽的病人、病原携带者和疑似传染病病人时，城镇于六小时内，"
            "农村于十二小时内，以最快的通讯方式向发病地的卫生防疫机构报告，"
            "并同时报出传染病报告卡。责任疫情报告人在丙类传染病监测区内发现"
            "丙类传染病病人时，应当在二十四小时内向发病地的卫生防疫机构报出传染病报告卡。"
        ),
    },
]


# ============================================================
# 构造 Few-shot Prompt
# ============================================================

def build_fewshot_system_prompt():

    parts = [
        SYSTEM_PROMPT,
        "",
        "下面给出六个已经标注的示例，请参考示例理解分类标准。",
        "",
    ]

    for i, example in enumerate(FEWSHOT_EXAMPLES, start=1):

        parts.append(f"示例{i}：")
        parts.append("政策内容：")
        parts.append(example["text"])
        parts.append(f"标签：{example['label']}")
        parts.append("")

    parts.append(
        "请对接下来的政策段落进行分类。"
        "只输出最终结果W/N/R，不要输出任何解释。"
    )

    return "\n".join(parts)


FEWSHOT_SYSTEM_PROMPT = build_fewshot_system_prompt()


# ============================================================
# 解析 messages
# ============================================================

def parse_messages(messages):

    if isinstance(messages, str):
        messages = json.loads(messages)

    user_content = ""
    assistant_content = ""

    for message in messages:

        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            user_content = content

        elif role == "assistant":
            assistant_content = content

    return user_content, assistant_content


# ============================================================
# 清理文本
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    prefix = "待判断政策段落："

    if text.startswith(prefix):
        text = text[len(prefix):]

    return text.strip()


# ============================================================
# 解析模型标签
# ============================================================

def parse_label(content):

    if content is None:
        return None

    text = str(content).strip().upper()

    if text in VALID_LABELS:
        return text

    cleaned = re.sub(
        r"[*`'\"“”‘’：:。.!！\s]",
        "",
        text,
    )

    if cleaned in VALID_LABELS:
        return cleaned

    patterns = [
        r"(?:答案|结果|标签|分类)[为是：:\s]*([NRW])",
        r"最终(?:答案|结果|标签)[为是：:\s]*([NRW])",
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            label = match.group(1)

            if label in VALID_LABELS:
                return label

    matches = re.findall(
        r"\b([NRW])\b",
        text,
    )

    if len(matches) == 1 and matches[0] in VALID_LABELS:
        return matches[0]

    return None


# ============================================================
# API 调用
# ============================================================

def call_model(text):

    user_content = "待判断政策段落：" + text

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": FEWSHOT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                max_tokens=10,
                stream=False,
            )

            raw_output = response.choices[0].message.content

            label = parse_label(raw_output)

            if label in VALID_LABELS:

                return {
                    "raw_output": raw_output,
                    "prediction": label,
                    "error": None,
                    "attempt": attempt,
                }

            return {
                "raw_output": raw_output,
                "prediction": None,
                "error": "INVALID_LABEL",
                "attempt": attempt,
            }

        except Exception as e:

            last_error = repr(e)

            print(
                f"    API调用失败 "
                f"(attempt {attempt}/{MAX_RETRIES})："
                f"{last_error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    return {
        "raw_output": None,
        "prediction": None,
        "error": "API_FAILED_AFTER_RETRIES",
        "error_detail": last_error,
        "attempt": MAX_RETRIES,
    }


# ============================================================
# 加载验证集
# ============================================================

print("=" * 70)
print(f"实验标题：{EXPERIMENT_TITLE}")
print("=" * 70)

print("模型：", MODEL)
print("API地址：", BASE_URL)

print()
print("加载验证集：")

val_df = pd.read_parquet(
    VALIDATION_PATH
)

print("验证集样本数：", len(val_df))


# ============================================================
# 显示六个 Few-shot 示例
# ============================================================

print()
print("=" * 70)
print("Few-shot k=6 示例")
print("=" * 70)

for i, example in enumerate(
    FEWSHOT_EXAMPLES,
    start=1,
):

    print()
    print(f"示例 {i}")
    print("标签：", example["label"])
    print("文本：", example["text"])


# ============================================================
# 读取 checkpoint
# ============================================================

results = []

if os.path.exists(CHECKPOINT_FILE):

    print()
    print("检测到已有 checkpoint，继续实验...")

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            checkpoint_data = json.load(f)

        results = checkpoint_data.get(
            "results",
            [],
        )

        print(
            "checkpoint 已有结果：",
            len(results),
        )

    except Exception as e:

        print(
            "读取 checkpoint 失败：",
            repr(e),
        )

        results = []


# ============================================================
# 已完成样本
# ============================================================

completed_indices = {
    item.get("index")
    for item in results
    if item.get("prediction") in VALID_LABELS
}


# ============================================================
# 正式实验
# ============================================================

print()
print("=" * 70)
print("开始 Qwen Level-1 Few-shot k=6 实验")
print("=" * 70)

for i, row in val_df.iterrows():

    index = int(i)

    if index in completed_indices:
        continue

    messages = row["messages"]

    user_content, gold_label = parse_messages(
        messages
    )

    text = clean_text(
        user_content
    )

    gold_label = str(
        gold_label
    ).strip().upper()

    print()
    print("-" * 70)
    print(
        f"样本 {index + 1}/{len(val_df)}"
    )
    print("真实标签：", gold_label)
    print("文本：", text)

    result = call_model(text)

    prediction = result.get(
        "prediction"
    )

    print(
        "模型原始输出：",
        repr(result.get("raw_output")),
    )

    print(
        "解析标签：",
        prediction,
    )

    record = {
        "index": index,
        "gold": gold_label,
        "text": text,
        "prediction": prediction,
        "raw_output": result.get("raw_output"),
        "error": result.get("error"),
        "error_detail": result.get("error_detail"),
        "attempt": result.get("attempt"),
        "experiment": EXPERIMENT_TITLE,
        "model": MODEL,
        "fewshot_k": 6,
    }

    results.append(record)

    # --------------------------------------------------------
    # 每条结果立即 checkpoint
    # --------------------------------------------------------

    checkpoint_data = {
        "experiment": EXPERIMENT_TITLE,
        "model": MODEL,
        "fewshot_k": 6,
        "total_samples": len(val_df),
        "results": results,
    }

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    if prediction in VALID_LABELS:
        completed_indices.add(index)

    time.sleep(0.3)


# ============================================================
# 计算有效预测
# ============================================================

valid_results = [
    item
    for item in results
    if item.get("gold") in VALID_LABELS
    and item.get("prediction") in VALID_LABELS
]


y_true = [
    item["gold"]
    for item in valid_results
]

y_pred = [
    item["prediction"]
    for item in valid_results
]


# ============================================================
# 计算指标
# ============================================================

if len(valid_results) > 0:

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=["N", "R", "W"],
        average="macro",
        zero_division=0,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=["N", "R", "W"],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=["N", "R", "W"],
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["N", "R", "W"],
    )

else:

    accuracy = None
    macro_f1 = None
    kappa = None
    report = ""
    cm = []


# ============================================================
# 保存最终结果
# ============================================================

final_data = {
    "experiment": EXPERIMENT_TITLE,
    "model": MODEL,
    "base_url": BASE_URL,
    "fewshot_k": 6,
    "total_samples": len(val_df),
    "valid_predictions": len(valid_results),
    "failed_predictions": (
        len(val_df) - len(valid_results)
    ),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "kappa": kappa,
    "classification_report": report,
    "confusion_matrix_labels": [
        "N",
        "R",
        "W",
    ],
    "confusion_matrix": (
        cm.tolist()
        if hasattr(cm, "tolist")
        else cm
    ),
    "fewshot_examples": FEWSHOT_EXAMPLES,
    "results": results,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_data,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 输出最终结果
# ============================================================

print()
print("=" * 70)
print(f"{EXPERIMENT_TITLE} 结果")
print("=" * 70)

print(
    "总样本数：",
    len(val_df),
)

print(
    "有效预测：",
    len(valid_results),
    "/",
    len(val_df),
)

print(
    "失败预测：",
    len(val_df) - len(valid_results),
)

if accuracy is not None:

    print()
    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro-F1 : {macro_f1:.4f}"
    )

    print(
        f"Kappa    : {kappa:.4f}"
    )

    print()
    print(
        "Classification Report:"
    )

    print(report)

    print(
        "Confusion Matrix"
    )

    print(
        "标签顺序：N / R / W"
    )

    print(cm)

else:

    print()
    print(
        "没有有效预测，无法计算指标。"
    )


print()
print("=" * 70)
print("结果文件：", OUTPUT_FILE)
print("Checkpoint：", CHECKPOINT_FILE)
print("=" * 70)