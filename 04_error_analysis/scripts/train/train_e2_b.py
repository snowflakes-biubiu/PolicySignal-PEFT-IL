from pathlib import Path

import torch

import train_e2_parallel_framework as framework


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "E2_B_NW_weighted"
EXPERIMENT_ROOT = (
    PROJECT_ROOT / "04_error_analysis" / "artifacts" / "e2_class_weighted"
)


def main():
    experiment = next(
        (
            item
            for item in framework.EXPERIMENTS
            if item["name"] == EXPERIMENT_NAME
        ),
        None,
    )
    if experiment is None:
        raise ValueError(
            f"共享训练框架中找不到实验配置：{EXPERIMENT_NAME}"
        )

    class_weights = experiment["class_weights"]
    e2_a = next(
        (
            item
            for item in framework.EXPERIMENTS
            if item["name"] == "E2_A_N_weighted"
        ),
        None,
    )
    if e2_a is None:
        raise ValueError("共享训练框架中找不到 E2-A 配置")

    if class_weights["N"] != e2_a["class_weights"]["N"]:
        raise ValueError("E2-A 与 E2-B 的 N 权重必须一致")
    if class_weights["R"] != 1.0 or class_weights["W"] <= 1.0:
        raise ValueError("E2-B 必须保持 R 权重为 1.0，且 W 权重大于 1.0")

    experiment_dir = EXPERIMENT_ROOT / EXPERIMENT_NAME
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            "E2-B 输出目录已存在且非空，为避免覆盖历史结果，停止训练："
            f"{experiment_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有可用 CUDA GPU，请检查 PyTorch 与显卡环境"
        )

    # 共享框架使用相对路径；固定到项目根目录以支持从其他目录启动脚本。
    framework.DATA_DIR = PROJECT_ROOT / "04_error_analysis" / "data" / "split_v2"
    framework.EXPERIMENT_ROOT = EXPERIMENT_ROOT

    print("=" * 75)
    print("E2-B 单独训练：N/W 类别加权")
    print("=" * 75)
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"训练数据：{framework.DATA_DIR / 'train.jsonl'}")
    print(f"验证数据：{framework.DATA_DIR / 'validation.jsonl'}")
    print(f"类别权重：{class_weights}")
    print(f"输出目录：{experiment_dir}")
    print("E2-A 不会重新训练；本脚本仅执行 E2-B。")

    train_df, validation_df = framework.load_data()
    tokenizer, label_token_ids = framework.load_tokenizer()
    metrics = framework.run_experiment(
        exp=experiment,
        train_df=train_df,
        validation_df=validation_df,
        tokenizer=tokenizer,
        label_token_ids=label_token_ids,
    )

    print("\nE2-B 训练与验证完成。")
    print(f"Accuracy：{metrics['accuracy']:.4f}")
    print(f"Macro-F1：{metrics['macro_f1']:.4f}")
    print(f"Cohen's Kappa：{metrics['cohen_kappa']:.4f}")
    print(f"Adapter：{experiment_dir / 'adapter'}")
    print(f"评估结果：{experiment_dir / 'results'}")


if __name__ == "__main__":
    main()
