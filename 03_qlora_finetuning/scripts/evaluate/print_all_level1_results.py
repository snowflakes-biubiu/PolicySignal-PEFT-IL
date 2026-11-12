import json
from pathlib import Path

# 项目目录下的实验结果目录
ROOT = Path("03_qlora_finetuning") / "results"

# 待汇总的实验结果文件
RESULT_FILES = [
    "level1_qlora_validation_300.json",
    "level1_qlora_rank4_validation_300.json",
    "level1_qlora_rank16_validation_300.json",
    "level1_qlora_checkpoint300_validation_300.json",
    "rank16_lr1e4_validation_300/level1_qlora_rank16_lr1e4_validation_300.json",
    "level1_qlora_rank4_alpha16_validation_300.json",
    "level1_qlora_rank16_alpha16_validation_300.json",
]

LABELS = ["N", "R", "W"]


def get_value(data, *keys, default=None):
    """兼容不同实验 JSON 使用的字段名称。"""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def format_number(value):
    """所有浮点指标统一保留四位小数，不转换成百分数。"""
    if value is None:
        return "未提供"
    return f"{float(value):.4f}"


def format_result(relative_path):
    path = ROOT / relative_path
    lines = []

    lines.append("=" * 60)
    lines.append(f"实验结果文件：{relative_path}")

    if not path.exists():
        lines.append("状态：文件不存在")
        return "\n".join(lines)

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        lines.append(f"状态：读取失败，{exc}")
        return "\n".join(lines)

    lines.append("评估结果")
    lines.append("-" * 60)
    lines.append(
        f"样本数    ：{get_value(data, 'validation_samples', default='未提供')}"
    )
    lines.append(
        f"Accuracy  ：{format_number(get_value(data, 'accuracy'))}"
    )
    lines.append(
        f"Macro-F1  ：{format_number(get_value(data, 'macro_f1'))}"
    )
    lines.append(
        f"Kappa     ：{format_number(get_value(data, 'cohen_kappa', 'kappa'))}"
    )

    report = data.get("classification_report", {})
    class_f1 = data.get("class_f1", {})

    lines.append("")
    lines.append("各类别指标")
    lines.append("-" * 60)
    lines.append(
        f"{'类别':<8}{'Precision':>12}{'Recall':>12}"
        f"{'F1':>12}{'Support':>12}"
    )

    for label in LABELS:
        metrics = report.get(label, {})

        # 兼容仅保存 class_f1 的结果文件
        f1_value = get_value(
            metrics, "f1-score",
            default=class_f1.get(label)
        )

        if metrics:
            precision = format_number(metrics.get("precision"))
            recall = format_number(metrics.get("recall"))
            f1 = format_number(f1_value)
            support = metrics.get("support", "未提供")

            if isinstance(support, (float, int)):
                support = str(int(support))
        else:
            precision = recall = f1 = "未提供"
            support = "未提供"

        lines.append(
            f"{label:<8}{precision:>12}{recall:>12}"
            f"{f1:>12}{support:>12}"
        )

    matrix = data.get("confusion_matrix")
    matrix_labels = get_value(
        data, "confusion_matrix_label_order", "labels",
        default=LABELS
    )

    lines.append("")
    lines.append("混淆矩阵")
    lines.append("-" * 60)

    if matrix:
        lines.append("真实值\\预测值：" + "  ".join(matrix_labels))
        for label, row in zip(matrix_labels, matrix):
            lines.append(
                f"{label:<12}" + "  ".join(str(int(v)) for v in row)
            )
    else:
        lines.append("未提供混淆矩阵")

    return "\n".join(lines)


def main():
    all_results = []

    for filename in RESULT_FILES:
        all_results.append(format_result(filename))

    output = "\n\n".join(all_results) + "\n"

    # UTF-8 with BOM，便于 Windows 记事本识别中文
    output_path = ROOT / "level1_all_results_standardized.txt"
    output_path.write_text(output, encoding="utf-8-sig")

    # 终端只打印英文和路径，避免终端编码问题影响结果文件
    print("Completed.")
    print(f"Results saved to: {output_path}")
    print(f"Experiments listed: {len(RESULT_FILES)}")


if __name__ == "__main__":
    main()