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


# =========================
# 1. 基本配置
# =========================

DATA_FILE = Path(
    r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0"
    r"\snapshots\d05965f169e6f7da33a25f667edf39add0339e2b"
    r"\task1_level1\validation.parquet"
)

CHECKPOINT_FILE = Path("02_llm_inference/results/deepseek/deepseek_level1_zero_shot_300_checkpoint.json")
RESULT_FILE = Path("02_llm_inference/results/deepseek/deepseek_level1_zero_shot_300.json")

MODEL_NAME = "deepseek-flash"

# 论文 Appendix D.1 的一级分类 Prompt
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
待判断政策内容如下：
"""


# =========================
# 2. API
# =========================

api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY，请先在当前 CMD 设置 API Key。"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# =========================
# 3. 读取数据
# =========================

print("=" * 70)
print("DeepSeek Level-1 Zero-shot 正式实验")
print("=" * 70)

print(f"数据文件：{DATA_FILE}")
print(f"模型：{MODEL_NAME}")
print()

df = pd.read_parquet(DATA_FILE)

if len(df) < 300:
    raise RuntimeError(f"验证集只有 {len(df)} 条，不足 300 条。")

df = df.iloc[:300].copy()

print(f"本次实验样本数：{len(df)}")
print()


# =========================
# 4. 提取消息
# =========================

def extract_user_text(messages):
    """
    从 Hugging Face 数据集 messages 中提取 user 内容，
    并去掉数据中原有的“待判断政策段落：”提示。
    """

    if isinstance(messages, str):
        messages = json.loads(messages)

    user_content = None

    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    if user_content is None:
        raise ValueError("没有找到 user message")

    prefix = "待判断政策段落："

    if user_content.startswith(prefix):
        user_content = user_content[len(prefix):]

    return user_content.strip()


def extract_gold_label(messages):
    """
    从最后一个 assistant message 中提取金标准标签。
    """

    if isinstance(messages, str):
        messages = json.loads(messages)

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            label = msg.get("content", "").strip().upper()

            if label in {"W", "N", "R"}:
                return label

            # 防止存在少量格式变化
            match = re.search(r"\b([WNR])\b", label)
            if match:
                return match.group(1)

            raise ValueError(f"无法识别金标准标签：{label}")

    raise ValueError("没有找到 assistant message")


# =========================
# 5. 模型输出解析
# =========================

def parse_prediction(text):
    """
    论文要求只输出 W/N/R。
    这里允许模型出现轻微格式变化，但最终只接受 W/N/R。
    """

    if not text:
        return None

    cleaned = text.strip().upper()

    # 最理想情况
    if cleaned in {"W", "N", "R"}:
        return cleaned

    # 如果模型输出类似：
    # “W。”
    # “结果：W”
    # “W\n”
    # 则提取单独出现的标签
    matches = re.findall(r"\b([WNR])\b", cleaned)

    if len(matches) == 1:
        return matches[0]

    # 中文/标点紧邻的情况
    for label in ["W", "N", "R"]:
        if cleaned == label + "。" or cleaned == label + ".":
            return label

    return None


# =========================
# 6. checkpoint
# =========================

def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return {}

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print("警告：checkpoint 文件读取失败，将重新开始。")
        return {}


def save_checkpoint(results):
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
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


# =========================
# 7. 正式逐条调用
# =========================

for index in range(300):

    key = str(index)

    # 成功结果不重复调用
    if key in results and results[key].get("status") == "success":
        continue

    row = df.iloc[index]

    try:
        text = extract_user_text(row["messages"])
        gold = extract_gold_label(row["messages"])

    except Exception as e:
        results[key] = {
            "index": index,
            "status": "data_error",
            "error": str(e),
        }

        save_checkpoint(results)
        print(f"[{index + 1}/300] 数据错误：{e}")
        continue

    # 最多 3 次，仅用于 API 失败/网络失败
    success = False

    for attempt in range(1, 4):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                stream=False,
                max_tokens=4,
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                },
            )

            raw_output = response.choices[0].message.content or ""
            prediction = parse_prediction(raw_output)

            results[key] = {
                "index": index,
                "gold": gold,
                "prediction": prediction,
                "raw_output": raw_output,
                "text": text,
                "status": "success",
            }

            save_checkpoint(results)

            status_text = (
                "✓"
                if prediction == gold
                else "✗"
            )

            print(
                f"[{index + 1:3d}/300] "
                f"Gold={gold} "
                f"Pred={prediction} "
                f"{status_text}"
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


# =========================
# 8. 汇总结果
# =========================

valid_results = []

for index in range(300):

    key = str(index)

    if key not in results:
        continue

    item = results[key]

    if (
        item.get("status") == "success"
        and item.get("prediction") in {"W", "N", "R"}
        and item.get("gold") in {"W", "N", "R"}
    ):
        valid_results.append(item)


if not valid_results:
    raise RuntimeError("没有可用于统计的有效结果。")


y_true = [x["gold"] for x in valid_results]
y_pred = [x["prediction"] for x in valid_results]

accuracy = accuracy_score(y_true, y_pred)
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
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["N", "R", "W"],
)


summary = {
    "experiment": "DeepSeek Level-1 Zero-shot",
    "model": MODEL_NAME,
    "dataset": "task1_level1/validation.parquet",
    "sample_count": 300,
    "valid_result_count": len(valid_results),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "cohen_kappa": kappa,
    "labels": ["N", "R", "W"],
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "results": results,
}


with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 70)
print("DeepSeek Level-1 Zero-shot 正式实验")
print("=" * 70)
print(f"有效样本数：{len(valid_results)}/300")
print(f"Accuracy   ：{accuracy:.4f}")
print(f"Macro-F1   ：{macro_f1:.4f}")
print(f"Cohen Kappa：{kappa:.4f}")
print()
print("Classification Report:")
print(report)
print("Confusion Matrix [N, R, W]:")
print(cm)
print()
print(f"完整结果：{RESULT_FILE}")
print(f"断点文件：{CHECKPOINT_FILE}")
print("=" * 70)