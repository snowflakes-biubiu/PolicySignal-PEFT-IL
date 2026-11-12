# -*- coding: utf-8 -*-
"""
Level-1 QLoRA 新模型评估脚本

评估模型：
1. Rank=4, Alpha=16
2. Rank=16, Alpha=16

验证集：CAPC-CG_V1.0/task1_level1/validation.parquet（预期 300 条）

输出指标：
- Accuracy
- Macro-F1
- Cohen's Kappa
- 混淆矩阵
- N/R/W 各类别 Precision、Recall、F1
- 每条样本的预测结果和分类 logits

运行位置：
C:\\Users\\snowf\\CAPC-CG-Project

运行方式：
python eval_level1_qlora_new_adapters.py
"""

from pathlib import Path
import gc
import json
import traceback

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

# ============================================================
# 1. 基本配置
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "Baron-Sun/CAPC-CG_V1.0"

MAX_LENGTH = 768
LABELS = ["N", "R", "W"]

EXPERIMENTS = [
    {
        "name": "rank4_alpha16",
        "adapter_dir": (
            Path("03_qlora_finetuning/artifacts/models") / "qwen15b_level1_qlora_rank4_alpha16"
        ),
        "output_stem": (
            "level1_qlora_rank4_alpha16_validation_300"
        ),
    },
    {
        "name": "rank16_alpha16",
        "adapter_dir": (
            Path("03_qlora_finetuning/artifacts/models") / "qwen15b_level1_qlora_rank16_alpha16"
        ),
        "output_stem": (
            "level1_qlora_rank16_alpha16_validation_300"
        ),
    },
]

RESULTS_DIR = Path("03_qlora_finetuning/results")


# ============================================================
# 2. 检查分类标签 Token
# ============================================================

def check_label_tokens(tokenizer):
    """
    当前评估采用下一个 Token 的 logits 进行分类。
    因此 N、R、W 必须各自对应唯一的单 Token。
    """

    token_ids = {}

    for label in LABELS:
        ids = tokenizer(
            label,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) != 1:
            raise ValueError(
                f"标签 {label!r} 被编码成 {len(ids)} 个 Token：{ids}。\n"
                "当前评估方法要求每个标签对应一个 Token，"
                "请停止评估并检查标签编码方式。"
            )

        token_ids[label] = ids[0]

    if len(set(token_ids.values())) != len(LABELS):
        raise ValueError(
            f"N/R/W 对应的 Token ID 不唯一：{token_ids}"
        )

    print("标签 Token ID：", token_ids)

    return token_ids


# ============================================================
# 3. 加载验证集
# ============================================================

def load_validation_data():

    snapshot_dir = Path(
        snapshot_download(
            DATASET_NAME,
            repo_type="dataset",
            local_files_only=True,
        )
    )

    validation_file = (
        snapshot_dir
        / "task1_level1"
        / "validation.parquet"
    )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"找不到验证集文件：{validation_file}"
        )

    df = pd.read_parquet(validation_file)

    if len(df) != 300:
        raise ValueError(
            f"预期验证集有 300 条样本，实际为 {len(df)} 条。"
        )

    gold_labels = {
        str(row["messages"][2]["content"]).strip()
        for _, row in df.iterrows()
    }

    unexpected = gold_labels - set(LABELS)

    if unexpected:
        raise ValueError(
            f"验证集中出现非 N/R/W 标签：{unexpected}"
        )

    print(f"验证集加载成功：{len(df)} 条")

    return df


# ============================================================
# 4. 构造 Prompt
# ============================================================

def encode_prompt(row, tokenizer):

    messages = row["messages"]

    if len(messages) < 3:
        raise ValueError(
            f"样本 messages 数量不足 3 条：{len(messages)}"
        )

    prompt_messages = [
        {
            "role": "system",
            "content": messages[0]["content"],
        },
        {
            "role": "user",
            "content": messages[1]["content"],
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    # 与训练脚本保持一致：
    # Prompt 超长时保留末尾 MAX_LENGTH 个 Token。
    if len(prompt_ids) > MAX_LENGTH:
        prompt_ids = prompt_ids[-MAX_LENGTH:]

    if not prompt_ids:
        raise ValueError("Prompt 编码后为空。")

    return prompt_ids


# ============================================================
# 5. 评估单个 Adapter
# ============================================================

def evaluate_one(experiment, tokenizer, df, token_ids):

    adapter_dir = experiment["adapter_dir"]

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter 目录不存在：{adapter_dir}"
        )

    config_file = adapter_dir / "adapter_config.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"缺少 adapter_config.json：{config_file}"
        )

    print("\n" + "=" * 70)
    print(f"开始评估：{experiment['name']}")
    print(f"Adapter：{adapter_dir}")
    print("=" * 70)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = None
    model = None

    try:
        # ----------------------------------------------------
        # 加载基础模型
        # ----------------------------------------------------

        print("正在加载基础模型……")

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )

        # ----------------------------------------------------
        # 加载 LoRA Adapter
        # ----------------------------------------------------

        print("正在加载 LoRA Adapter……")

        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=False,
        )

        model.eval()

        # 将输入放到模型输入 Embedding 所在的设备。
        input_device = model.get_input_embeddings().weight.device

        class_token_ids = torch.tensor(
            [token_ids[label] for label in LABELS],
            dtype=torch.long,
            device=input_device,
        )

        y_true = []
        y_pred = []
        prediction_rows = []

        # ----------------------------------------------------
        # 开始推理
        # ----------------------------------------------------

        with torch.inference_mode():

            for index, (_, row) in enumerate(
                df.iterrows(),
                start=1,
            ):

                gold = str(
                    row["messages"][2]["content"]
                ).strip()

                prompt_ids = encode_prompt(row, tokenizer)

                input_ids = torch.tensor(
                    [prompt_ids],
                    dtype=torch.long,
                    device=input_device,
                )

                attention_mask = torch.ones_like(input_ids)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                # 取 Prompt 最后一个位置的 logits，
                # 用于预测紧接 Prompt 的下一个 Token。
                next_logits = outputs.logits[0, -1, :]

                # 仅比较 N、R、W 三个候选标签的 logits。
                class_logits = next_logits.index_select(
                    0,
                    class_token_ids,
                )

                pred_index = int(
                    torch.argmax(class_logits).item()
                )

                pred = LABELS[pred_index]

                y_true.append(gold)
                y_pred.append(pred)

                prediction_rows.append(
                    {
                        "row_number": index,
                        "gold": gold,
                        "prediction": pred,
                        "correct": gold == pred,
                        "logit_N": float(
                            class_logits[0].float().item()
                        ),
                        "logit_R": float(
                            class_logits[1].float().item()
                        ),
                        "logit_W": float(
                            class_logits[2].float().item()
                        ),
                        "prompt_tokens": len(prompt_ids),
                    }
                )

                if index % 25 == 0 or index == len(df):
                    print(
                        f"评估进度：{index}/{len(df)}"
                    )

        # ----------------------------------------------------
        # 计算指标
        # ----------------------------------------------------

        accuracy = accuracy_score(y_true, y_pred)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        kappa = cohen_kappa_score(
            y_true,
            y_pred,
            labels=LABELS,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "experiment": experiment["name"],
            "base_model": BASE_MODEL,
            "adapter_dir": str(adapter_dir),
            "validation_samples": len(df),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "kappa": float(kappa),
            "confusion_matrix_label_order": LABELS,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "class_f1": {
                label: float(report[label]["f1-score"])
                for label in LABELS
            },
        }

        # ----------------------------------------------------
        # 保存结果，禁止静默覆盖
        # ----------------------------------------------------

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            RESULTS_DIR
            / f"{experiment['output_stem']}.json"
        )

        csv_path = (
            RESULTS_DIR
            / f"{experiment['output_stem']}.csv"
        )

        for output_path in [json_path, csv_path]:
            if output_path.exists():
                raise FileExistsError(
                    f"结果文件已存在，为保护旧结果而停止写入："
                    f"{output_path}\n"
                    "如需重新评估，请修改脚本中的 output_stem。"
                )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                ensure_ascii=False,
                indent=2,
            )

        pd.DataFrame(
            prediction_rows
        ).to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        # ----------------------------------------------------
        # 输出摘要
        # ----------------------------------------------------

        print("\n评估结果")
        print("-" * 50)
        print(f"样本数    ：{len(df)}")
        print(f"Accuracy  ：{accuracy:.4%}")
        print(f"Macro-F1   ：{macro_f1:.4%}")
        print(f"Kappa      ：{kappa:.4f}")

        print("\n各类别指标：")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=LABELS,
                target_names=LABELS,
                zero_division=0,
            )
        )

        print("混淆矩阵（行=真实标签，列=预测标签）")
        print("标签顺序：N、R、W")
        print(cm)

        print(f"\nJSON 结果：{json_path}")
        print(f"CSV 结果 ：{csv_path}")

        return metrics

    finally:
        if model is not None:
            del model

        if base_model is not None:
            del base_model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ============================================================
# 6. 主流程
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA。当前脚本使用 GPU 4-bit 量化评估。"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Level-1 QLoRA 新 Adapter 评估")
    print("=" * 70)

    print("\n正在加载本地 Tokenizer……")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    # 先验证标签 Token，失败则不加载大模型。
    token_ids = check_label_tokens(tokenizer)

    print("\n正在加载验证集……")
    df = load_validation_data()

    all_metrics = []

    for experiment in EXPERIMENTS:
        metrics = evaluate_one(
            experiment,
            tokenizer,
            df,
            token_ids,
        )

        all_metrics.append(metrics)

    # --------------------------------------------------------
    # 保存两组实验的汇总表
    # --------------------------------------------------------

    summary_csv = (
        RESULTS_DIR
        / "level1_qlora_rank4a16_vs_rank16a16_summary.csv"
    )

    summary_json = (
        RESULTS_DIR
        / "level1_qlora_rank4a16_vs_rank16a16_summary.json"
    )

    if summary_csv.exists() or summary_json.exists():
        raise FileExistsError(
            "汇总文件已存在。为保护已有结果，未覆盖汇总文件。\n"
            f"{summary_csv}\n{summary_json}\n"
            "如需重新评估，请修改脚本中的汇总文件名。"
        )

    summary_rows = []

    for result in all_metrics:
        summary_rows.append(
            {
                "experiment": result["experiment"],
                "validation_samples": result["validation_samples"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "kappa": result["kappa"],
                "N_F1": result["class_f1"]["N"],
                "R_F1": result["class_f1"]["R"],
                "W_F1": result["class_f1"]["W"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("两组模型评估完成")
    print("=" * 70)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n汇总 CSV ：{summary_csv}")
    print(f"汇总 JSON：{summary_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n评估失败，完整错误信息：")
        traceback.print_exc()
        raise