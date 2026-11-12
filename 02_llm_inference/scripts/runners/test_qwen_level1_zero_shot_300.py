# ============================================================
# 实验标题：Qwen Level-1 Zero-shot 实验
# ============================================================

import os
import json
import time
import pandas as pd

from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. 基本配置
# ============================================================

API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("未检测到 DASHSCOPE_API_KEY")

MODEL = "qwen3.8-max"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

VAL_PATH = (
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task1_level1\validation.parquet"
)

OUTPUT_PATH = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300.json"
CHECKPOINT_PATH = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300_checkpoint.json"

MAX_RETRIES = 3
SLEEP_SECONDS = 0.5


# ============================================================
# 2. Appendix D.1 原始 Level-1 Prompt
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
# 4. 标签解析
# ============================================================

def normalize_label(text):
    if not text:
        return None

    text = text.strip().upper()

    if text in {"W", "N", "R"}:
        return text

    text = text.replace("：", "").replace(":", "").strip()

    if text in {"W", "N", "R"}:
        return text

    return None


# ============================================================
# 5. 提取政策文本和真实标签
# ============================================================

def extract_user_text(messages):

    for message in messages:

        if message.get("role") == "user":

            text = message.get("content", "")

            return text.replace(
                "待判断政策段落：",
                "",
                1
            ).strip()

    raise ValueError("未找到 user 消息")


def extract_gold_label(messages):

    for message in messages:

        if message.get("role") == "assistant":

            label = message.get(
                "content",
                ""
            ).strip().upper()

            if label in {"W", "N", "R"}:
                return label

    raise ValueError("未找到有效 assistant 标签")


# ============================================================
# 6. 读取验证集
# ============================================================

df = pd.read_parquet(VAL_PATH)

print("=" * 70)
print("实验标题：Qwen Level-1 Zero-shot 实验")
print("=" * 70)
print("模型：", MODEL)
print("验证集总数：", len(df))
print("目标样本数：300")
print()


# ============================================================
# 7. 加载已有 checkpoint
# ============================================================

if os.path.exists(CHECKPOINT_PATH):

    print("检测到已有 checkpoint，正在恢复...")

    with open(
        CHECKPOINT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        checkpoint = json.load(f)

    results = checkpoint.get("results", [])

    completed_indices = {
        item["index"]
        for item in results
    }

    print(
        "已完成样本：",
        len(completed_indices)
    )

else:

    results = []
    completed_indices = set()

    print("未发现 checkpoint，从第 1 条开始。")


# ============================================================
# 8. 正式实验
# ============================================================

for i in range(min(300, len(df))):

    if i in completed_indices:

        continue

    row = df.iloc[i]

    messages = row["messages"]

    gold = extract_gold_label(messages)

    policy_text = extract_user_text(messages)

    user_prompt = SYSTEM_PROMPT + policy_text

    print("-" * 70)
    print(
        f"样本 {i + 1}/300"
    )
    print(
        "真实标签：",
        gold
    )

    success = False

    for attempt in range(1, MAX_RETRIES + 1):

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

            raw_output = (
                response
                .choices[0]
                .message
                .content
            )

            prediction = normalize_label(
                raw_output
            )

            print(
                "模型原始输出：",
                repr(raw_output)
            )

            print(
                "解析标签：",
                prediction
            )

            if prediction is None:

                print(
                    "警告：模型输出不是有效标签，"
                    f"第 {attempt} 次尝试。"
                )

                if attempt < MAX_RETRIES:

                    time.sleep(1)

                    continue

            result = {
                "index": int(i),
                "gold": gold,
                "prediction": prediction,
                "raw_output": raw_output,
                "text": policy_text
            }

            results.append(result)

            success = True

            break

        except Exception as e:

            print(
                f"API错误，第 {attempt}/{MAX_RETRIES} 次：",
                repr(e)
            )

            if attempt < MAX_RETRIES:

                time.sleep(2)

    # --------------------------------------------------------
    # 如果三次都失败，记录错误
    # --------------------------------------------------------

    if not success:

        print(
            "该样本最终失败，将记录为 API_ERROR。"
        )

        results.append(
            {
                "index": int(i),
                "gold": gold,
                "prediction": None,
                "raw_output": None,
                "text": policy_text,
                "error": "API_FAILED_AFTER_RETRIES"
            }
        )

    # --------------------------------------------------------
    # 每完成一个样本立即保存 checkpoint
    # --------------------------------------------------------

    checkpoint_data = {
        "experiment":
            "Qwen Level-1 Zero-shot 实验",

        "model":
            MODEL,

        "num_completed":
            len(results),

        "results":
            results
    }

    with open(
        CHECKPOINT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            checkpoint_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    time.sleep(SLEEP_SECONDS)


# ============================================================
# 9. 最终统计
# ============================================================

print()
print("=" * 70)
print("实验完成，开始计算指标")
print("=" * 70)


valid_results = [
    r
    for r in results
    if r.get("prediction")
    in {"W", "N", "R"}
]


y_true = [
    r["gold"]
    for r in valid_results
]

y_pred = [
    r["prediction"]
    for r in valid_results
]


accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=["N", "R", "W"],
    average="macro",
    zero_division=0
)

kappa = cohen_kappa_score(
    y_true,
    y_pred,
    labels=["N", "R", "W"]
)


report = classification_report(
    y_true,
    y_pred,
    labels=["N", "R", "W"],
    digits=4,
    zero_division=0
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["N", "R", "W"]
)


# ============================================================
# 10. 保存最终结果
# ============================================================

final_data = {

    "experiment":
        "Qwen Level-1 Zero-shot 实验",

    "model":
        MODEL,

    "num_samples":
        len(results),

    "valid_predictions":
        len(valid_results),

    "accuracy":
        accuracy,

    "macro_f1":
        macro_f1,

    "kappa":
        kappa,

    "classification_report":
        report,

    "confusion_matrix_labels":
        ["N", "R", "W"],

    "confusion_matrix":
        cm.tolist(),

    "results":
        results
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 11. 输出最终结果
# ============================================================

print()
print("=" * 70)
print("Qwen Level-1 Zero-shot 实验结果")
print("=" * 70)

print(
    f"有效预测：{len(valid_results)}/{len(results)}"
)

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
print("Classification Report:")
print(report)

print("Confusion Matrix")
print("标签顺序：N / R / W")
print(cm)

print()
print("结果文件：")
print(OUTPUT_PATH)

print("Checkpoint：")
print(CHECKPOINT_PATH)

print("=" * 70)