import json
from pathlib import Path

ROOT = Path("03_qlora_finetuning") / "results"

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

# All Chinese strings are built from Unicode escapes.
TEXT = {
    "file": "\u5b9e\u9a8c\u7ed3\u679c\u6587\u4ef6",
    "evaluation": "\u8bc4\u4f30\u7ed3\u679c",
    "samples": "\u6837\u672c\u6570",
    "accuracy": "\u51c6\u786e\u7387",
    "macro_f1": "\u5b8f\u5e73\u5747F1",
    "kappa": "Kappa",
    "class_metrics": "\u5404\u7c7b\u522b\u6307\u6807",
    "class": "\u7c7b\u522b",
    "support": "\u6837\u672c\u652f\u6301\u6570",
    "matrix": "\u6df7\u6dc6\u77e9\u9635",
    "true_pred": "\u771f\u5b9e\u503c\\\u9884\u6d4b\u503c",
    "missing": "\u6587\u4ef6\u4e0d\u5b58\u5728",
    "read_failed": "\u8bfb\u53d6\u5931\u8d25",
    "no_matrix": "\u672a\u63d0\u4f9b\u6df7\u6dc6\u77e9\u9635",
}


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def get_value(data, *keys):
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    return None


def render_result(relative_path):
    path = ROOT / relative_path
    lines = [
        "=" * 68,
        f"{TEXT['file']}: {relative_path}",
    ]

    if not path.exists():
        lines.append(TEXT["missing"])
        return "\n".join(lines)

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        lines.append(f"{TEXT['read_failed']}: {exc}")
        return "\n".join(lines)

    lines.extend([
        TEXT["evaluation"],
        "-" * 68,
        f"{TEXT['samples']:<12}: {get_value(data, 'validation_samples')}",
        f"{TEXT['accuracy']:<12}: {fmt(get_value(data, 'accuracy'))}",
        f"{TEXT['macro_f1']:<12}: {fmt(get_value(data, 'macro_f1'))}",
        f"{TEXT['kappa']:<12}: {fmt(get_value(data, 'cohen_kappa', 'kappa'))}",
        "",
        TEXT["class_metrics"],
        "-" * 68,
        f"{TEXT['class']:<8}{'Precision':>12}{'Recall':>12}"
        f"{'F1':>12}{TEXT['support']:>14}",
    ])

    report = data.get("classification_report", {})
    class_f1 = data.get("class_f1", {})

    for label in LABELS:
        metrics = report.get(label, {})
        f1 = metrics.get("f1-score", class_f1.get(label))

        precision = fmt(metrics.get("precision"))
        recall = fmt(metrics.get("recall"))
        f1_text = fmt(f1)

        support = metrics.get("support")
        support_text = (
            str(int(support))
            if isinstance(support, (int, float))
            else "N/A"
        )

        lines.append(
            f"{label:<8}{precision:>12}{recall:>12}"
            f"{f1_text:>12}{support_text:>14}"
        )

    matrix = data.get("confusion_matrix")
    matrix_labels = (
        data.get("confusion_matrix_label_order")
        or data.get("labels")
        or LABELS
    )

    lines.extend(["", TEXT["matrix"], "-" * 68])

    if matrix:
        lines.append(
            f"{TEXT['true_pred']}: " + "  ".join(matrix_labels)
        )
        for label, row in zip(matrix_labels, matrix):
            lines.append(
                f"{label:<12}" + "  ".join(str(int(v)) for v in row)
            )
    else:
        lines.append(TEXT["no_matrix"])

    return "\n".join(lines)


def main():
    output = "\n\n".join(
        render_result(name) for name in RESULT_FILES
    ) + "\n"

    output_path = ROOT / "level1_all_results_standardized_utf8.txt"
    output_path.write_text(output, encoding="utf-8-sig")

    print("Completed.")
    print(f"Experiments listed: {len(RESULT_FILES)}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()