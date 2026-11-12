import os
import json
import time
import re
from pathlib import Path

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
# 1. 文件与模型配置
# ============================================================

TRAIN_FILE = Path(
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task1_level1\train.parquet"
)

VALIDATION_FILE = Path(
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task1_level1\validation.parquet"
)

CHECKPOINT_FILE = Path(
    "02_llm_inference/results/deepseek/deepseek_level1_fewshot3_300_checkpoint.json"
)

RESULT_FILE = Path(
    "02_llm_inference/results/deepseek/deepseek_level1_fewshot3_300.json"
)

MODEL_NAME = "deepseek-flash"


# ============================================================
# 2. Appendix D.1 一级分类 Prompt
# ============================================================

SYSTEM_PROMPT = """角色：你是一名“政策信号一级标注员”。
任务：对给定中文政策文档段落分类标注：
W = 段落中存在对政府/公共部门的肯定性行动指令（包含任何“要/应/负责/建立/加强/推进/组织/指导/监督/落实/制定/完善/开展/实施/安排”等对政府主体布置的任务或授权）。
R = 段落中存在对政府/公共部门的禁止性指令（如“严禁/不得/禁止/停止……”（由政府或其部门））或对政府及公职人员的违法行为进行处罚。
N = 无指令（仅背景介绍、原则口号、定义释义、程序性条款、日期/效力条款、对非国家主体的要求而未同时对政府布置任务或责任的内容）。

受众原则（极其重要）：只统计指向政府及其附属公共机构（包括国务院各部委、地方政府、监管机构、事业单位等）的行动/禁止为指令。

仅限制或要求企业、个人或社会组织，而没有同时给政府安排任务或责任，则记为 N

若同时出现：例如“企业不得……；主管部门要加强监管……”，则本段视为存在对政府的指令，进入 W 或 R 的判断。

政府/公共部门的定义为中央或地方政府、公职人员或承担了国家分配的任务的附属机构

不要只看关键词：是否出现“必须/可以/鼓励/不得”等词不等于对应标签，必须结合主语是否为政府/公共机构与句子功能来判断。

例如：“企业必须符合××条件方可办理”，若仅针对企业且未布置政府任务，则记为 N。

输出要求：只输出最终的结果W/N/R
"""


# ============================================================
# 3. API
# ============================================================

api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY。"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# ============================================================
# 4. 数据解析
# ============================================================

def parse_messages(messages):

    if isinstance(messages, str):
        messages = json.loads(messages)

    user_text = None
    label = None

    for msg in messages:

        if msg.get("role") == "user":
            user_text = msg.get("content", "")

        elif msg.get("role") == "assistant":
            label = msg.get("content", "").strip().upper()

    if user_text is None:
        raise ValueError("没有找到 user message")

    prefix = "待判断政策段落："

    if user_text.startswith(prefix):
        user_text = user_text[len(prefix):]

    if label not in {"N", "R", "W"}:
        raise ValueError(f"无效标签：{label}")

    return user_text.strip(), label


def parse_prediction(text):

    if not text:
        return None

    cleaned = text.strip().upper()

    if cleaned in {"N", "R", "W"}:
        return cleaned

    matches = re.findall(r"\b([NRW])\b", cleaned)

    if len(matches) == 1:
        return matches[0]

    for label in ["N", "R", "W"]:
        if cleaned in {label + "。", label + "."}:
            return label

    return None


# ============================================================
# 5. 固定 Few-shot 示例
# ============================================================

# 这里直接固定刚才从训练集选出的三个样本。
FEWSHOT_EXAMPLES = [
    {
        "label": "N",
        "text": (
            "第二十一条　企业所得税法第六条第（八）项所称接受捐赠收入，"
            "是指企业接受的来自其他企业、组织或者个人无偿给予的货币性资产、"
            "非货币性资产。接受捐赠收入，按照实际收到捐赠资产的日期确认收入的实现。"
        ),
    },
    {
        "label": "R",
        "text": (
            "（一）这次清查主要是一九八八年和一九八九年列入“小金库”的各项收支，"
            "以及历年滚存的“小金库”的结余。自国务院国发（１９８９）７７号文件发布之日起，"
            "“小金库”的各项资金一律停止支付。违者从严查处。"
        ),
    },
    {
        "label": "W",
        "text": (
            "第二十一条　登记事项自记载于不动产登记簿时完成登记。"
            "不动产登记机构完成登记，应当依法向申请人核发不动产权属证书或者登记证明。"
        ),
    },
]


def build_messages(policy_text):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # --------------------------------------------------------
    # Few-shot demonstrations
    # --------------------------------------------------------

    for example in FEWSHOT_EXAMPLES:

        messages.append(
            {
                "role": "user",
                "content": (
                    "待判断政策段落：\n"
                    + example["text"]
                ),
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": example["label"],
            }
        )

    # --------------------------------------------------------
    # 真正需要判断的测试样本
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": (
                "待判断政策段落：\n"
                + policy_text
            ),
        }
    )

    return messages


# ============================================================
# 6. 读取 validation
# ============================================================

print("=" * 70)
print("DeepSeek Level-1 Few-shot k=3 正式实验")
print("=" * 70)

print(f"模型：{MODEL_NAME}")
print(f"训练集：{TRAIN_FILE}")
print(f"验证集：{VALIDATION_FILE}")
print()

train_df = pd.read_parquet(TRAIN_FILE)
df = pd.read_parquet(VALIDATION_FILE)

df = df.iloc[:300].copy()

print(f"训练集样本数：{len(train_df)}")
print(f"验证集测试样本数：{len(df)}")
print()


# ============================================================
# 7. 加载 checkpoint
# ============================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():
        return {}

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        print("警告：checkpoint 读取失败，将重新开始。")
        return {}


def save_checkpoint(results):

    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_file.replace(CHECKPOINT_FILE)


results = load_checkpoint()

completed = sum(
    1
    for item in results.values()
    if item.get("status") == "success"
)

print(f"已有成功结果：{completed} / 300")
print()


# ============================================================
# 8. 正式实验
# ============================================================

for index in range(300):

    key = str(index)

    # 已成功的样本不重复调用
    if (
        key in results
        and results[key].get("status") == "success"
    ):
        continue

    row = df.iloc[index]

    try:

        policy_text, gold = parse_messages(
            row["messages"]
        )

    except Exception as e:

        results[key] = {
            "index": index,
            "status": "data_error",
            "error": str(e),
        }

        save_checkpoint(results)

        print(
            f"[{index + 1}/300] 数据错误：{e}"
        )

        continue

    success = False

    # API失败最多重试3次
    for attempt in range(1, 4):

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=build_messages(
                    policy_text
                ),

                stream=False,

                max_tokens=4,

                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                },
            )

            raw_output = (
                response.choices[0].message.content
                or ""
            )

            prediction = parse_prediction(
                raw_output
            )

            results[key] = {

                "index": index,

                "gold": gold,

                "prediction": prediction,

                "raw_output": raw_output,

                "text": policy_text,

                "status": "success",
            }

            save_checkpoint(results)

            mark = (
                "✓"
                if prediction == gold
                else "✗"
            )

            print(
                f"[{index + 1:3d}/300] "
                f"Gold={gold} "
                f"Pred={prediction} "
                f"{mark}"
            )

            success = True
            break

        except Exception as e:

            print(
                f"[{index + 1:3d}/300] "
                f"API失败，第 {attempt}/3 次：{e}"
            )

            if attempt < 3:
                time.sleep(2 ** attempt)

    if not success:

        results[key] = {

            "index": index,

            "gold": gold,

            "status": "api_error",

            "error": "3次API调用均失败",
        }

        save_checkpoint(results)

        print(
            f"[{index + 1:3d}/300] "
            f"最终失败，已保存 checkpoint"
        )


# ============================================================
# 9. 统计结果
# ============================================================

valid_results = []

for index in range(300):

    key = str(index)

    if key not in results:
        continue

    item = results[key]

    if (
        item.get("status") == "success"
        and item.get("prediction") in {"N", "R", "W"}
        and item.get("gold") in {"N", "R", "W"}
    ):

        valid_results.append(item)


if not valid_results:

    raise RuntimeError(
        "没有有效实验结果。"
    )


y_true = [
    item["gold"]
    for item in valid_results
]

y_pred = [
    item["prediction"]
    for item in valid_results
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

summary = {

    "experiment":
        "DeepSeek Level-1 Few-shot k=3",

    "model":
        MODEL_NAME,

    "dataset":
        "task1_level1/validation.parquet",

    "sample_count":
        300,

    "valid_result_count":
        len(valid_results),

    "fewshot_k":
        3,

    "fewshot_examples":
        FEWSHOT_EXAMPLES,

    "accuracy":
        accuracy,

    "macro_f1":
        macro_f1,

    "cohen_kappa":
        kappa,

    "labels":
        ["N", "R", "W"],

    "confusion_matrix":
        cm.tolist(),

    "classification_report":
        report,

    "results":
        results,
}


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 11. 输出
# ============================================================

print()
print("=" * 70)
print("DeepSeek Level-1 Few-shot k=3 实验")
print("=" * 70)

print(
    f"有效样本数："
    f"{len(valid_results)}/300"
)

print(
    f"Accuracy   ："
    f"{accuracy:.4f}"
)

print(
    f"Macro-F1   ："
    f"{macro_f1:.4f}"
)

print(
    f"Cohen Kappa："
    f"{kappa:.4f}"
)

print()
print("Classification Report:")
print(report)

print("Confusion Matrix [N, R, W]:")
print(cm)

print()
print(f"完整结果：{RESULT_FILE}")
print(f"断点文件：{CHECKPOINT_FILE}")

print("=" * 70)