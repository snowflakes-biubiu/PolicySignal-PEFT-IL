# ============================================================
# 实验标题：Qwen Level-2 Few-shot k=3 实验
# ============================================================

import os
import json
import time
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
# 1. 基础配置
# ============================================================

EXPERIMENT_TITLE = "Qwen Level-2 Few-shot k=3 实验"

MODEL_NAME = "qwen3.8-flash"

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"

PROJECT_DIR = Path(r"C:\Users\snowf\CAPC-CG-Project")

DATA_FILE = (
    Path(r"C:\Users\snowf\.cache\huggingface\hub\datasets--Baron-Sun--CAPC-CG_V1.0")
    / "snapshots"
    / "d05965f169e6f7da33a25f667edf39add0339e2b"
    / "task2_level2"
    / "validation.parquet"
)

OUTPUT_FILE = PROJECT_DIR / "02_llm_inference/results/qwen/qwen_level2_fewshot3_400.json"

CHECKPOINT_FILE = (
    PROJECT_DIR / "02_llm_inference/results/qwen/qwen_level2_fewshot3_400_checkpoint.json"
)

TARGET_N = 400

VALID_LABELS = {"B", "C", "G", "Y"}

LABEL_ORDER = ["B", "C", "G", "Y"]


# ============================================================
# 2. API 配置
# ============================================================

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise RuntimeError(
        "未检测到 DASHSCOPE_API_KEY 环境变量。"
    )

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
)


# ============================================================
# 3. 读取验证集
# ============================================================

print("=" * 72)
print(EXPERIMENT_TITLE)
print("=" * 72)

print(f"模型：{MODEL_NAME}")
print(f"数据集：{DATA_FILE}")
print()

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"找不到验证集文件：\n{DATA_FILE}"
    )

df = pd.read_parquet(DATA_FILE)

if len(df) != 400:
    raise RuntimeError(
        f"验证集数量异常，预期 400，实际 {len(df)}"
    )

if "messages" not in df.columns:
    raise RuntimeError(
        f"找不到 messages 字段：{df.columns.tolist()}"
    )

print(f"验证集：{len(df)} 条")
print("数据结构检查：通过")
print()


# ============================================================
# 4. 解析原始数据
# ============================================================

def parse_sample(messages):

    if isinstance(messages, str):
        messages = json.loads(messages)

    system_prompt = None
    user_content = None
    gold_label = None

    for message in messages:

        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            system_prompt = content

        elif role == "user":
            user_content = content

        elif role == "assistant":
            gold_label = content.strip().upper()

    if system_prompt is None:
        raise ValueError("缺少 system message")

    if user_content is None:
        raise ValueError("缺少 user message")

    if gold_label not in VALID_LABELS:
        raise ValueError(
            f"Gold 标签异常：{gold_label}"
        )

    return system_prompt, user_content, gold_label


samples = []

for i, row in df.iloc[:TARGET_N].iterrows():

    system_prompt, user_content, gold_label = parse_sample(
        row["messages"]
    )

    samples.append(
        {
            "index": int(i),
            "system_prompt": system_prompt,
            "user_content": user_content,
            "gold": gold_label,
        }
    )


# ============================================================
# 5. Few-shot 示例
#
# 与此前 DeepSeek Level-2 k=3 实验保持一致：
# 每个类别 B / C / Y 各提供一个示例。
#
# 注意：
# G 不作为 k=3 示例。
# ============================================================

FEWSHOT_EXAMPLES = [
    {
        "user": """待判断政策段落：
第二十二条　依照《土地管理法》第十九条规定收回使用权的国有土地，县级以上人民政府可以确定给农业集体经济组织耕种。农业集体经济组织在耕种期间，不得在该土地上兴建永久性建筑物或者种植多年生作物，并在国家建设需要使用时按时交还。交还时土地上有青苗的，建设单位应当付给青苗补偿费。""",
        "assistant": "B",
    },
    {
        "user": """待判断政策段落：
五、选举委员会设召集人制度，负责必要时召集选举委员会会议，办理有关事宜。总召集人由担任国家领导职务的选举委员会委员担任，总召集人在选举委员会每个界别各指定若干名召集人。""",
        "assistant": "G",
    },
    {
        "user": """待判断政策段落：
各有关部门要抓紧组织实施，严格按照有关规定做好相关工作，确保各项任务全面落实。""",
        "assistant": "Y",
    },
]


# ============================================================
# 6. 读取 checkpoint
# ============================================================

results = {}

if CHECKPOINT_FILE.exists():

    print(
        f"发现已有 checkpoint：{CHECKPOINT_FILE}"
    )

    with open(
        CHECKPOINT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        checkpoint = json.load(f)

    results = checkpoint.get("results", {})

    completed = sum(
        1
        for r in results.values()
        if r.get("status") == "success"
        and r.get("prediction") in VALID_LABELS
    )

    print(f"已有成功结果：{completed} 条")
    print()

else:

    print("未发现 checkpoint，从第 1 条开始。")
    print()


# ============================================================
# 7. API 调用
# ============================================================

def call_model(
    system_prompt,
    user_content,
    max_retries=3,
):

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # --------------------------------------------------------
    # 加入 3 个 Few-shot 示例
    # --------------------------------------------------------

    for example in FEWSHOT_EXAMPLES:

        messages.append(
            {
                "role": "user",
                "content": example["user"],
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": example["assistant"],
            }
        )

    # --------------------------------------------------------
    # 加入当前待判断样本
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    # --------------------------------------------------------
    # API 调用
    # --------------------------------------------------------

    for attempt in range(1, max_retries + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=10,
                timeout=60,
                extra_body={
                    "enable_thinking": False
                },
            )

            answer = (
                response.choices[0]
                .message
                .content
                .strip()
                .upper()
            )

            if answer in VALID_LABELS:
                return answer

            print(
                f"    ⚠ 无效模型输出：{answer!r}"
            )

        except Exception as e:

            print(
                f"    ⚠ API 调用失败 "
                f"({attempt}/{max_retries})："
                f"{type(e).__name__}: {e}"
            )

        if attempt < max_retries:
            time.sleep(2 * attempt)

    return None


# ============================================================
# 8. 保存 checkpoint
# ============================================================

def save_checkpoint():

    checkpoint_data = {
        "experiment": EXPERIMENT_TITLE,
        "model": MODEL_NAME,
        "fewshot_k": 3,
        "target_samples": TARGET_N,
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


# ============================================================
# 9. 开始正式实验
# ============================================================

print("=" * 72)
print("开始 Qwen3.8-Flash Level-2 Few-shot k=3")
print("=" * 72)

print("Few-shot 示例数量：3")
print("示例标签：B / G / Y")
print("Thinking：关闭")
print()

start_time = time.time()

for sample in samples:

    index = sample["index"]
    key = str(index)

    # 已成功样本不重复调用
    if (
        key in results
        and results[key].get("status") == "success"
        and results[key].get("prediction") in VALID_LABELS
    ):
        continue

    gold = sample["gold"]

    print(
        f"[{index + 1:03d}/{TARGET_N}] "
        f"Gold={gold} | "
        f"调用 Qwen3.8-Flash...",
        end=" ",
        flush=True,
    )

    prediction = call_model(
        sample["system_prompt"],
        sample["user_content"],
    )

    if prediction is None:

        print("失败")

        results[key] = {
            "index": index,
            "gold": gold,
            "prediction": None,
            "status": "failed",
        }

    else:

        correct = prediction == gold

        print(
            f"Pred={prediction} "
            f"{'✓' if correct else '✗'}"
        )

        results[key] = {
            "index": index,
            "gold": gold,
            "prediction": prediction,
            "status": "success",
            "correct": correct,
        }

    save_checkpoint()


# ============================================================
# 10. 汇总结果
# ============================================================

valid_results = [
    r
    for r in results.values()
    if r.get("status") == "success"
    and r.get("prediction") in VALID_LABELS
    and r.get("gold") in VALID_LABELS
]

failed_results = [
    r
    for r in results.values()
    if r.get("status") != "success"
]


# ============================================================
# 11. 计算指标
# ============================================================

if valid_results:

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
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

else:

    accuracy = None
    macro_f1 = None
    kappa = None
    report = ""
    cm = []


# ============================================================
# 12. 保存最终结果
# ============================================================

final_output = {
    "experiment": EXPERIMENT_TITLE,
    "model": MODEL_NAME,

    "dataset": "Baron-Sun/CAPC-CG_V1.0",
    "task": "Level-2",
    "setting": "Few-shot",
    "fewshot_k": 3,

    "target_samples": TARGET_N,
    "valid_samples": len(valid_results),
    "failed_samples": len(failed_results),

    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "kappa": kappa,

    "classification_report": report,

    "confusion_matrix": (
        cm.tolist()
        if hasattr(cm, "tolist")
        else cm
    ),

    "label_order": LABEL_ORDER,

    "results": results,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_output,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 13. 最终输出
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 72)
print("Qwen Level-2 Few-shot k=3 实验完成")
print("=" * 72)

print(f"模型：{MODEL_NAME}")
print(f"目标数量：{TARGET_N}")
print(f"有效数量：{len(valid_results)}")
print(f"失败数量：{len(failed_results)}")

if valid_results:

    print()
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")

    print()
    print("Classification Report:")
    print(report)

    print("Confusion Matrix")
    print("行 = Gold，列 = Prediction")
    print("标签顺序：B / C / G / Y")
    print(cm)

print()
print(f"耗时：{elapsed / 60:.2f} 分钟")
print()
print(f"最终结果：{OUTPUT_FILE}")
print(f"Checkpoint：{CHECKPOINT_FILE}")

print("=" * 72)