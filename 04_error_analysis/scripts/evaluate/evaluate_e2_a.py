
from datetime import datetime
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

# 导入原实验脚本中的数据处理、Tokenizer 和评估函数。
# 注意：原脚本有 if __name__ == "__main__" 保护，
# 因此导入不会自动执行 main()，也不会启动训练。
import train_e2_parallel_framework as exp


def main():
    print("=" * 70)
    print("E2-A 独立评估：只加载已训练 Adapter，不进行训练")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU，请检查 PyTorch 与显卡环境")

    experiment_dir = exp.EXPERIMENT_ROOT / "E2_A_N_weighted"
    adapter_dir = experiment_dir / "adapter"
    validation_path = exp.DATA_DIR / "validation.jsonl"

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"找不到已训练 Adapter：{adapter_dir}")

    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"缺少 adapter_config.json：{adapter_dir}"
        )

    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"缺少 adapter_model.safetensors：{adapter_dir}"
        )

    if not validation_path.is_file():
        raise FileNotFoundError(f"找不到验证集：{validation_path}")

    # 使用独立输出目录，避免覆盖原有实验结果。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = experiment_dir / f"evaluation_recovery_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"基础模型：{exp.MODEL_NAME}")
    print(f"已训练 Adapter：{adapter_dir}")
    print(f"验证集：{validation_path}")
    print(f"本次评估输出：{output_dir}")

    # 读取数据。这里不会训练模型。
    _, validation_df = exp.load_data()
    tokenizer, _ = exp.load_tokenizer()

    print(f"验证样本数量：{len(validation_df)}")

    # 按原实验的 4-bit NF4 配置加载基础模型。
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        exp.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 评估使用生成式推理，启用 KV cache。
    base_model.config.use_cache = True

    # 加载 E2-A 已训练的 Adapter，而不是新建随机 Adapter。
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
    )

    model.eval()

    print("\n已加载 E2-A 训练权重，开始独立验证推理。")
    print("本脚本不会调用 trainer.train()。")

    # 只执行验证，不训练。
    metrics = exp.predict_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        validation_df=validation_df,
        output_dir=output_dir,
    )

    print("\nE2-A 独立评估完成。")
    print(f"评估样本数：{metrics['sample_count']}")
    print(f"Accuracy：{metrics['accuracy']:.4%}")
    print(f"Macro-F1：{metrics['macro_f1']:.4f}")
    print(f"Cohen's Kappa：{metrics['cohen_kappa']:.4f}")
    print(f"非法输出数：{metrics['invalid_predictions']}")
    print(f"结果目录：{output_dir}")

    # 释放显存。
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import gc
    main()
