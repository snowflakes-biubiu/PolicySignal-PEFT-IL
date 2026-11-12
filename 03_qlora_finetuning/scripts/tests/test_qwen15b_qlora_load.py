import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    print("=" * 60)
    print("Qwen2.5-1.5B QLoRA 环境测试")
    print("=" * 60)

    print(f"模型: {MODEL_NAME}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"显存: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # 4-bit QLoRA 配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n[1/3] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("[2/3] 加载 4-bit 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("[3/3] 执行一次小规模前向推理...")

    text = (
        "请判断下面这段政策文本是否包含明确的政策指令。"
        "只回答 W、R 或 N。\n"
        "各地区要积极推进数字化建设，提高政策执行效率。"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    print("\n" + "=" * 60)
    print("测试成功")
    print("=" * 60)

    print(f"输出 logits shape: {tuple(outputs.logits.shape)}")

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"当前显存占用: {allocated:.2f} GB")
    print(f"当前显存保留: {reserved:.2f} GB")

    print("\n模型设备分布:")
    print(model.hf_device_map)


if __name__ == "__main__":
    main()