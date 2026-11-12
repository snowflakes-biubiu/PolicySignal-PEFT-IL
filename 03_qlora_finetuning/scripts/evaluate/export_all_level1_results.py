
from pathlib import Path
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter

# ============================================================
# 路径配置
# ============================================================

ROOT = Path("03_qlora_finetuning")
RESULT_DIR = ROOT / "results"

OUTPUT_FILE = RESULT_DIR / "Level1_QLoRA_全部实验结果汇总.xlsx"

# ============================================================
# 实验配置
# 按原始 Word 表格顺序排列
# ============================================================

EXPERIMENTS = [
    {
        "epoch": 1,
        "rank": 8,
        "alpha": 16,
        "lr": "2e-4",
        "result": "level1_qlora_epoch1_validation_300.json",
        "note": "独立的一轮训练；与三轮训练中的 Epoch 1 不应视为完全相同的训练过程",
    },
    {
        "epoch": 2,
        "rank": 8,
        "alpha": 16,
        "lr": "2e-4",
        "result": "level1_qlora_checkpoint300_validation_300.json",
        "note": "三轮训练的 checkpoint-300",
    },
    {
        "epoch": 3,
        "rank": 8,
        "alpha": 16,
        "lr": "2e-4",
        "result": "level1_qlora_validation_300.json",
        "note": "三轮训练的 checkpoint-450",
    },
    {
        "epoch": 2,
        "rank": 4,
        "alpha": 8,
        "lr": "2e-4",
        "result": "level1_qlora_rank4_validation_300.json",
        "note": "",
    },
    {
        "epoch": 2,
        "rank": 16,
        "alpha": 32,
        "lr": "2e-4",
        "result": "level1_qlora_rank16_validation_300.json",
        "note": "",
    },
    {
        "epoch": 2,
        "rank": 4,
        "alpha": 16,
        "lr": "2e-4",
        "result": "level1_qlora_rank4_alpha16_validation_300.json",
        "note": "",
    },
    {
        "epoch": 2,
        "rank": 16,
        "alpha": 16,
        "lr": "2e-4",
        "result": "level1_qlora_rank16_alpha16_validation_300.json",
        "note": "",
    },
    {
        "epoch": 2,
        "rank": 16,
        "alpha": 32,
        "lr": "1e-4",
        "result": (
            "rank16_lr1e4_validation_300/"
            "level1_qlora_rank16_lr1e4_validation_300.json"
        ),
        "note": "",
    },
]

# ============================================================
# JSON 读取与字段兼容
# ============================================================

def load_json(path):
    if not path.exists():
        print(f"[警告] 结果文件不存在：{path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[警告] 读取失败：{path}；原因：{exc}")
        return None


def find_value(data, possible_keys):
    """从多个可能的字段名称中查找指标。"""
    if not isinstance(data, dict):
        return None

    # 先检查顶层字段
    for key in possible_keys:
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])

    # 再检查常见的嵌套指标对象
    for container_key in (
        "metrics",
        "results",
        "classification_report",
        "summary",
    ):
        nested = data.get(container_key)
        if isinstance(nested, dict):
            for key in possible_keys:
                if key in nested and isinstance(
                    nested[key], (int, float)
                ):
                    return float(nested[key])

    return None


def find_class_f1(data, class_names):
    """读取分类报告中指定类别的 F1。"""
    if not isinstance(data, dict):
        return None

    possible_containers = [
        data.get("classification_report"),
        data.get("report"),
        data.get("per_class"),
        data.get("class_metrics"),
    ]

    for container in possible_containers:
        if not isinstance(container, dict):
            continue

        for name in class_names:
            item = container.get(name)
            if isinstance(item, dict):
                value = item.get("f1-score", item.get("f1"))
                if isinstance(value, (int, float)):
                    return float(value)

    return None


def extract_metrics(data):
    """
    返回指标的小数值，范围通常为 0 到 1。
    不将数值乘以 100，也不转换为百分比。
    """
    if data is None:
        return {
            "Kappa": None,
            "Acc.": None,
            "Macro-F1": None,
            "N-F1": None,
            "R-F1": None,
            "W-F1": None,
        }

    kappa = find_value(
        data, ["kappa", "cohen_kappa", "cohen_kappa_score"]
    )
    accuracy = find_value(
        data, ["accuracy", "accuracy_score", "acc"]
    )
    macro_f1 = find_value(
        data, ["macro_f1", "macro_f1_score", "f1_macro"]
    )

    # 兼容 classification_report 中的 macro avg
    report = data.get("classification_report")
    if macro_f1 is None and isinstance(report, dict):
        macro = report.get("macro avg", report.get("macro_avg"))
        if isinstance(macro, dict):
            macro_f1 = macro.get("f1-score", macro.get("f1"))

    # 优先读取独立类别 F1 字段
    n_f1 = find_value(data, ["N-F1", "N_f1", "n_f1", "f1_N"])
    r_f1 = find_value(data, ["R-F1", "R_f1", "r_f1", "f1_R"])
    w_f1 = find_value(data, ["W-F1", "W_f1", "w_f1", "f1_W"])

    if n_f1 is None:
        n_f1 = find_class_f1(data, ["N", "n"])
    if r_f1 is None:
        r_f1 = find_class_f1(data, ["R", "r"])
    if w_f1 is None:
        w_f1 = find_class_f1(data, ["W", "w"])

    return {
        "Kappa": kappa,
        "Acc.": accuracy,
        "Macro-F1": macro_f1,
        "N-F1": n_f1,
        "R-F1": r_f1,
        "W-F1": w_f1,
    }


def format_metric(value):
    """统一显示为四位小数；缺失时标记待补。"""
    if value is None:
        return "待补"
    return f"{float(value):.4f}"


# ============================================================
# 创建 Excel
# ============================================================

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Level-1汇总"

    headers = [
        "Epoch",
        "LoRA_Rank",
        "缩放比 alpha/rank",
        "LoRA_Alpha",
        "学习率",
        "Kappa",
        "Acc.",
        "Macro-F1",
        "N-F1",
        "R-F1",
        "W-F1",
        "结果文件",
        "备注",
    ]
    ws.append(headers)

    metric_names = [
        "Kappa",
        "Acc.",
        "Macro-F1",
        "N-F1",
        "R-F1",
        "W-F1",
    ]

    for exp in EXPERIMENTS:
        result_path = RESULT_DIR / exp["result"]
        data = load_json(result_path)
        metrics = extract_metrics(data)

        ratio = exp["alpha"] / exp["rank"]

        row = [
            exp["epoch"],
            exp["rank"],
            ratio,
            exp["alpha"],
            exp["lr"],
            *[format_metric(metrics[name]) for name in metric_names],
            exp["result"],
            exp["note"],
        ]
        ws.append(row)

    # ========================================================
    # 样式
    # ========================================================

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(
        name="Microsoft YaHei",
        bold=True,
        color="FFFFFF",
        size=10,
    )
    thin_gray = Side(style="thin", color="D9E2F3")

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )
        cell.border = Border(
            bottom=Side(style="medium", color="17365D")
        )

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Microsoft YaHei", size=10)
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )
            cell.border = Border(bottom=thin_gray)

    # 最高值标记：仅对存在有效数值的指标列处理
    best_fill = PatternFill("solid", fgColor="FFF2CC")

    for col in range(6, 12):
        numeric_cells = []
        for r in range(2, ws.max_row + 1):
            cell = ws.cell(r, col)
            try:
                numeric_cells.append((r, float(cell.value)))
            except (ValueError, TypeError):
                continue

        if not numeric_cells:
            continue

        best_value = max(value for _, value in numeric_cells)

        for r, value in numeric_cells:
            if value == best_value:
                cell = ws.cell(r, col)
                cell.font = Font(
                    name="Microsoft YaHei",
                    size=10,
                    bold=True,
                    color="C00000",
                )
                cell.fill = best_fill

    widths = {
        1: 9,
        2: 12,
        3: 18,
        4: 12,
        5: 12,
        6: 12,
        7: 12,
        8: 13,
        9: 12,
        10: 12,
        11: 12,
        12: 52,
        13: 58,
    }

    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width

    ws.row_dimensions[1].height = 32
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 38

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False

    # ========================================================
    # 说明工作表
    # ========================================================

    notes = wb.create_sheet("说明")
    notes_data = [
        ["项目", "说明"],
        ["任务层级", "Level-1 指令性分类任务，类别为 N / R / W。"],
        ["验证集", "验证集规模为 300 个样本，每类 100 个样本。"],
        [
            "指标格式",
            "指标以 0 到 1 范围的小数显示，统一保留四位小数；不使用百分比格式。",
        ],
        [
            "实验顺序",
            "按照原始 Word 表格中约定的 8 组实验顺序排列。",
        ],
        [
            "Epoch 1 注意事项",
            "Epoch 1 结果来自独立的一轮训练；Epoch 2 和 Epoch 3 基准结果来自三轮训练中的 checkpoint。比较时应注意训练过程差异。",
        ],
        [
            "最高值标记",
            "各指标列中的最高数值使用浅黄色背景及红色加粗标记。",
        ],
        [
            "缺失数据",
            "如果结果文件不存在或未找到对应指标，单元格显示“待补”，请核对 JSON 字段后补充。",
        ],
    ]

    for row in notes_data:
        notes.append(row)

    for cell in notes[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    for row in notes.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Microsoft YaHei", size=10)
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )
            cell.border = Border(bottom=thin_gray)

    notes.column_dimensions["A"].width = 24
    notes.column_dimensions["B"].width = 110
    notes.row_dimensions[1].height = 28

    for r in range(2, notes.max_row + 1):
        notes.row_dimensions[r].height = 38

    notes.freeze_panes = "A2"
    notes.sheet_view.showGridLines = False

    wb.save(OUTPUT_FILE)

    print("=" * 60)
    print("Level-1 QLoRA 实验结果汇总完成")
    print(f"输出文件：{OUTPUT_FILE}")
    print(f"实验组数：{len(EXPERIMENTS)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
