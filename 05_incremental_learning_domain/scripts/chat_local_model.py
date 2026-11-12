r"""在终端与本地 Qwen 模型对话或进行政策信号分类。

用法（从项目根目录运行）：
    python .\05_incremental_learning_domain\scripts\chat_local_model.py --model base --mode chat
    python 05_incremental_learning_domain/scripts/chat_local_model.py
    python 05_incremental_learning_domain/scripts/chat_local_model.py --mode classify
    python 05_incremental_learning_domain/scripts/chat_local_model.py --model base
    python 05_incremental_learning_domain/scripts/chat_local_model.py --model task1 --mode classify
    python 05_incremental_learning_domain/scripts/chat_local_model.py --adapter-path 05_incremental_learning_domain/artifacts/models/qwen15b_temporal_task2_e2_r16_a32_lr2e-4/checkpoint-150

直接加载 03 阶段的 QLoRA adapter（PowerShell）：
    python .\05_incremental_learning_domain\scripts\chat_local_model.py --adapter-path "C:\Users\snowf\CAPC-CG-Project\03_qlora_finetuning\artifacts\models\qwen15b_level1_qlora" --mode chat
    python .\05_incremental_learning_domain\scripts\chat_local_model.py --adapter-path "C:\Users\snowf\CAPC-CG-Project\03_qlora_finetuning\artifacts\models\qwen15b_level1_qlora" --mode classify

模型选择：--model base 使用本地 Qwen 基础模型；--model task1 至 task5 使用对应已保存的
增量 adapter；--model latest 自动选择编号最高且已有顶层 adapter_config.json 的完整任务模型。
训练中的 checkpoint 或其他目录中的 adapter 可通过 --adapter-path 显式指定；此参数优先级高于
--model。指定路径必须包含 adapter_config.json 和 adapter 权重。默认 --model latest。

模式选择：--mode chat 使用“你>”输入问题，模型通过“模型>”回复并保留多轮上下文；输入 /clear
清空上下文，输入 /exit 退出。请注意，Level-1 adapter 是分类微调模型，不保证具备通用聊天模型的
回答风格。--mode classify 使用与增量训练相同的政策分类提示词；在“段落>”后输入一条政策段落，
程序返回 N/R/W 预测和各标签分数，可连续输入多条，输入 /exit 退出。默认 --mode chat。

默认只读 Hugging Face 本地缓存。若模型文件尚未缓存，添加 --allow-download 后才会下载。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from incremental_common import (  # noqa: E402
    LABELS,
    MAX_LENGTH,
    MODEL_NAME,
    MODEL_ROOT,
    SYSTEM_PROMPT,
    TASKS,
    adapter_dir,
    build_prompt,
)


CHAT_SYSTEM_PROMPT = "你是一名中文助手。请直接、清楚地回答用户的问题。"
DEFAULT_MAX_NEW_TOKENS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在终端使用本地 Qwen 模型聊天或分类")
    parser.add_argument(
        "--model",
        choices=("base", "latest", "task1", "task2", "task3", "task4", "task5", "task6"),
        default="latest",
        help="base 基座模型，latest 最新完整 adapter，或指定 task1-task6",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="指定本地 PEFT adapter/checkpoint 路径；优先级高于 --model",
    )
    parser.add_argument(
        "--mode",
        choices=("chat", "classify"),
        default="chat",
        help="chat 多轮聊天；classify 使用 N/R/W 政策分类提示词",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--allow-download", action="store_true", help="允许从 Hugging Face 下载缺失文件")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_input_tokens < 128:
        parser.error("max-new-tokens 必须为正数，max-input-tokens 不得小于 128")
    return args


def resolve_adapter(model_choice: str, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"指定路径不是可加载的 PEFT adapter/checkpoint：{path}")
        return path

    if model_choice == "base":
        return None
    if model_choice == "latest":
        for task_number in range(len(TASKS), 0, -1):
            candidate = adapter_dir(task_number)
            if (candidate / "adapter_config.json").is_file():
                return candidate
        raise FileNotFoundError(f"未找到完整的 Task1-Task{len(TASKS)} adapter，请检查：{MODEL_ROOT}")

    task_number = int(model_choice.removeprefix("task"))
    candidate = adapter_dir(task_number)
    if not (candidate / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Task{task_number} 尚无最终 adapter：{candidate}\n"
            "若要加载某个已保存的中间 checkpoint，请改用 --adapter-path 指定其目录。"
        )
    return candidate


def load_local_model(adapter_path: Path | None, allow_download: bool) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA。当前本地聊天入口使用 4-bit bitsandbytes，需要 NVIDIA GPU。")

    local_files_only = not allow_download
    print(f"加载基础模型：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if adapter_path is not None:
        print(f"加载 adapter：{adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
    model.eval()
    return tokenizer, model


def prepare_chat_inputs(
    tokenizer: Any,
    history: list[dict[str, str]],
    max_input_tokens: int,
    device: Any,
) -> Any:
    messages = list(history)
    while True:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if input_ids.shape[-1] <= max_input_tokens:
            return input_ids.to(device)
        if len(messages) <= 2:
            return input_ids[:, -max_input_tokens:].to(device)
        del messages[1:3]


def run_chat(tokenizer: Any, model: Any, max_input_tokens: int, max_new_tokens: int) -> None:
    import torch

    history = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    print("\n聊天模式已就绪。输入 /clear 清空上下文，输入 /exit 退出。")
    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("已退出。")
            return
        if user_text.lower() == "/clear":
            history = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
            print("对话上下文已清空。")
            continue

        history.append({"role": "user", "content": user_text})
        input_ids = prepare_chat_inputs(tokenizer, history, max_input_tokens, model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
        print(f"\n模型> {answer}")
        history.append({"role": "assistant", "content": answer})


def run_classify(tokenizer: Any, model: Any, max_input_tokens: int) -> None:
    import torch

    label_token_ids: dict[str, int] = {}
    for label in LABELS:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"分类标签 {label} 不是单 token：{token_ids}")
        label_token_ids[label] = token_ids[0]

    print("\n分类模式已就绪。输入政策段落，输入 /exit 退出。")
    while True:
        try:
            text = input("\n段落> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if not text:
            continue
        if text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("已退出。")
            return

        prompt_ids = build_prompt(tokenizer, text)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
        if input_ids.shape[-1] > min(max_input_tokens, MAX_LENGTH):
            input_ids = input_ids[:, -min(max_input_tokens, MAX_LENGTH):]
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
        scores = {label: float(logits[token_id].float().item()) for label, token_id in label_token_ids.items()}
        prediction = max(scores, key=scores.get)
        print(f"\n分类> {prediction}    scores={scores}")


def main() -> None:
    args = parse_args()
    adapter_path = resolve_adapter(args.model, args.adapter_path)
    print(f"模式：{args.mode}")
    print(f"Adapter：{adapter_path if adapter_path else '无（基础模型）'}")
    tokenizer, model = load_local_model(adapter_path, args.allow_download)
    if args.mode == "chat":
        run_chat(tokenizer, model, args.max_input_tokens, args.max_new_tokens)
    else:
        if adapter_path is None:
            print("提示：当前选择的是基础模型，分类能力未经本项目增量训练。")
        run_classify(tokenizer, model, args.max_input_tokens)


if __name__ == "__main__":
    main()