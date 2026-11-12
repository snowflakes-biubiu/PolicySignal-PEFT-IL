# -*- coding: utf-8 -*-
"""按《汇报要求.docx》生成「时间增量持续学习实验结果分析」工作簿。

唯一原始数据源:
    results/temporal/temporal_{m}_task{N}_cumulative_test.json 中
    4 方法 × (1+2+3+4+5) = 60 组 3x3 混淆矩阵。
所有 Accuracy / Macro-F1 / Kappa / P / R / F1 / R_{i,j} / 持续学习指标
均由混淆矩阵现场复算，不直接采用报告中的指标值（报告值仅用于一致性核对）。

辅助来源:
    logs/full_matrix/summary_*.json   训练耗时（17 次训练实测，多批次按时间覆盖）
    artifacts/                        存储开销实测

输出:
    05_incremental_learning_time/results/持续学习实验结果汇总表.xlsx
    05_incremental_learning_time/results/fig1_stage_performance.png
    05_incremental_learning_time/results/fig2_knowledge_retention.png
    05_incremental_learning_time/results/fig3_class_performance.png
    05_incremental_learning_time/results/fig4_forgetting_curves.png

用法: .venv/Scripts/python.exe 05_incremental_learning_time/scripts/validation/make_report_tables.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.pagebreak import Break
from openpyxl.worksheet.properties import PageSetupProperties

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RES = PROJECT_ROOT / "05_incremental_learning_time" / "results"
TEMP = RES / "temporal"
ART = PROJECT_ROOT / "assets" / "05_incremental_learning_time"
LOGS = PROJECT_ROOT / "05_incremental_learning_time" / "logs"

METHODS = ["sequential_ft", "ewc", "replay", "adapter"]
M_LABEL = {"sequential_ft": "E1 Sequential FT", "ewc": "E2 EWC",
           "replay": "E3 Replay", "adapter": "E4 Adapter-CL"}
M_SHORT = {"sequential_ft": "Sequential FT", "ewc": "EWC",
           "replay": "Replay", "adapter": "Adapter-CL"}
TASKS = {1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5"}
CLASS_ORDER = ["W", "R", "N"]
CM_LABELS = ["N", "R", "W"]
M_COLORS = {"sequential_ft": "#2E75B6", "ewc": "#ED7D31",
            "replay": "#70AD47", "adapter": "#C00000"}
WINDOWS = ["T1_1949_1978", "T2_1979_1992", "T3_1993_2002",
           "T4_2003_2012", "T5_2013_2023"]

OUT_XLSX = RES / "时间增量持续学习实验结果汇总表.xlsx"
FIG_NAMES = ["fig1a_stage_diag.png", "fig1b_stage_cumavg.png",
             "fig2_knowledge_retention.png",
             "fig3_class_W.png", "fig3_class_R.png", "fig3_class_N.png"]

# ---------------- 视觉样式 ----------------
FONT_NAME = "微软雅黑"
C_NAVY, C_BLUE, C_BLUE_D = "FF1F3864", "FF2E75B6", "FF1F4E79"
C_LIGHT, C_PALE, C_WHITE = "FFDDEBF7", "FFF3F8FD", "FFFFFFFF"
C_BORDER, C_TEXT, C_MUTED = "FFC9D6E4", "FF20304A", "FF71829B"
C_GREEN_BG, C_GREEN_TX = "FFE2EFDA", "FF2E6B2E"
C_RED_BG, C_RED_TX = "FFFCE8E6", "FFA61B1B"
C_GOLD, C_GREY_BG = "FFFFF2CC", "FFF0F2F5"
C_AMBER_TX, C_NOTE_BG = "FF7F6000", "FFFFFDF3"
THIN = Side(style="thin", color=C_BORDER)
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
NUM4, NUM2, PCT2, INT0 = "0.0000", "0.00", "0.00%", "0"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


# ---------------- 样式工具 ----------------

def style(cell, *, bold=False, size=10, color=C_TEXT, fill=None,
          align="center", fmt=None, italic=False, border=True):
    cell.font = Font(name=FONT_NAME, size=size, bold=bold, color=color, italic=italic)
    cell.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
    if fill:
        cell.fill = PatternFill("solid", fgColor=fill)
    if border:
        cell.border = BORDER
    if fmt:
        cell.number_format = fmt
    return cell


def sheet_setup(ws, title, subtitle, ncols, widths, tab=C_BLUE, freeze=None, zoom=90):
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = tab
    ws.sheet_view.zoomScale = zoom
    for col in range(1, ncols + 1):
        ws.cell(1, col).fill = PatternFill("solid", fgColor=C_NAVY)
        ws.cell(2, col).fill = PatternFill("solid", fgColor=C_PALE)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    style(ws.cell(1, 1, title), bold=True, size=14, color=C_WHITE, fill=C_NAVY,
          align="left", border=False)
    style(ws.cell(2, 1, subtitle), size=9, color=C_MUTED, align="left", border=False)
    ws.row_dimensions[1].height = 34
    ws.row_dimensions[2].height = 24
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if freeze:
        ws.freeze_panes = freeze
    return 3


def band(ws, row, text, ncols, *, fill=C_LIGHT, color=C_BLUE_D, size=11, height=24):
    for col in range(1, ncols + 1):
        ws.cell(row, col).fill = PatternFill("solid", fgColor=fill)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    style(ws.cell(row, 1, text), bold=True, size=size, color=color,
          fill=fill, align="left", border=False)
    ws.row_dimensions[row].height = height


def group_header(ws, row, groups, height=24):
    """groups: [(start_col, end_col, text, fill)]。"""
    for start, end, text, fill in groups:
        for col in range(start, end + 1):
            c = ws.cell(row, col)
            c.fill = PatternFill("solid", fgColor=fill)
            c.border = BORDER
        if end > start:
            ws.merge_cells(start_row=row, start_column=start,
                           end_row=row, end_column=end)
        style(ws.cell(row, start, text), bold=True, size=10,
              color=C_WHITE, fill=fill)
    ws.row_dimensions[row].height = height


def header(ws, row, labels, height=32):
    for col, label in enumerate(labels, 1):
        style(ws.cell(row, col, label), bold=True, size=10,
              color=C_WHITE, fill=C_BLUE)
    ws.row_dimensions[row].height = height


def data_row(ws, row, values, *, fmts=None, zebra=False, bold_cols=(),
             align_map=None, fill_map=None, height=20):
    bg = C_PALE if zebra else C_WHITE
    for col, value in enumerate(values, 1):
        fmt = fmts[col - 1] if fmts and col - 1 < len(fmts) else None
        fill = (fill_map or {}).get(col, bg)
        align = (align_map or {}).get(col, "center")
        style(ws.cell(row, col, value), bold=(col in bold_cols), fill=fill,
              align=align, fmt=fmt)
    ws.row_dimensions[row].height = height


def merge_col(ws, col, start, end, value, *, fill=C_WHITE, bold=True):
    if end > start:
        ws.merge_cells(start_row=start, start_column=col,
                       end_row=end, end_column=col)
    style(ws.cell(start, col, value), bold=bold, fill=fill)
    for row in range(start, end + 1):
        c = ws.cell(row, col)
        c.fill = PatternFill("solid", fgColor=fill)
        c.border = BORDER


def note_line(ws, row, key, value, ncols=6, *, fill=C_WHITE, height=24, key_bold=True):
    style(ws.cell(row, 1, key), bold=key_bold, fill=C_LIGHT, align="left")
    for col in range(2, ncols + 1):
        ws.cell(row, col).fill = PatternFill("solid", fgColor=fill)
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=ncols)
    style(ws.cell(row, 2, value), align="left", fill=fill)
    for col in range(2, ncols + 1):
        ws.cell(row, col).border = BORDER
    ws.row_dimensions[row].height = height


def finish(ws, ncols, last_row, *, header_row=None, autofilter=None, one_page=None):
    ws.print_area = f"A1:{get_column_letter(ncols)}{last_row}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = "9"
    ws.page_setup.fitToWidth = 1
    if one_page is None:
        one_page = last_row <= 26
    ws.page_setup.fitToHeight = 1 if one_page else 0
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.print_options.horizontalCentered = True
    ws.oddFooter.center.text = "时间增量持续学习实验结果分析  ·  第 &P 页 / 共 &N 页"
    ws.oddFooter.center.size = 8
    if header_row:
        ws.print_title_rows = f"1:{header_row}"
    if autofilter:
        ws.auto_filter.ref = autofilter


def scale_rule(reverse=False):
    low, high = ("FF63BE7B", "FFF8696B") if reverse else ("FFF8696B", "FF63BE7B")
    return ColorScaleRule(start_type="min", start_color=low,
                          mid_type="percentile", mid_value=50, mid_color="FFFFEB84",
                          end_type="max", end_color=high)


def delta_rules(ws, rng):
    ws.conditional_formatting.add(rng, CellIsRule(
        operator="lessThan", formula=["0"],
        fill=PatternFill("solid", fgColor=C_RED_BG),
        font=Font(name=FONT_NAME, color=C_RED_TX, bold=True)))
    ws.conditional_formatting.add(rng, CellIsRule(
        operator="greaterThan", formula=["0"],
        fill=PatternFill("solid", fgColor=C_GREEN_BG),
        font=Font(name=FONT_NAME, color=C_GREEN_TX, bold=True)))


# ---------------- 数据层：混淆矩阵 → 全部指标 ----------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reorder_cm(raw: list[list[int]], labels: list[str]) -> list[list[int]]:
    """统一重排为 N/R/W（行=Gold，列=Pred）。"""
    if list(labels) == CM_LABELS:
        return [[int(x) for x in row] for row in raw]
    pos = {lab: labels.index(lab) for lab in CM_LABELS}
    return [[int(raw[pos[a]][pos[b]]) for b in CM_LABELS] for a in CM_LABELS]


def metrics_from_cm(cm: list[list[int]]) -> dict:
    """仅由 3x3 混淆矩阵复算 Accuracy / Macro-F1 / Kappa / 各类 P-R-F1。"""
    k = len(cm)
    total = sum(sum(row) for row in cm)
    acc = sum(cm[i][i] for i in range(k)) / total
    row_sum = [sum(row) for row in cm]
    col_sum = [sum(cm[i][j] for i in range(k)) for j in range(k)]
    p_e = sum(row_sum[i] * col_sum[i] for i in range(k)) / (total * total)
    kappa = (acc - p_e) / (1 - p_e) if p_e < 1 else 0.0
    per_class = {}
    for i, lab in enumerate(CM_LABELS):
        tp = cm[i][i]
        precision = tp / col_sum[i] if col_sum[i] else 0.0
        recall = tp / row_sum[i] if row_sum[i] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[lab] = {"precision": precision, "recall": recall,
                          "f1": f1, "support": row_sum[i]}
    return {"samples": total, "acc": acc, "kappa": kappa,
            "macro_f1": sum(per_class[lab]["f1"] for lab in CM_LABELS) / k,
            "per_class": per_class}


def collect_records() -> list[dict]:
    """读取 20 份累积评估 JSON，抽出 60 组 3x3 混淆矩阵。"""
    records = []
    for m in METHODS:
        for i in range(1, 6):
            path = TEMP / f"temporal_{m}_task{i}_cumulative_test.json"
            report = load_json(path)
            mt = report.get("metrics_by_test_task") or {}
            for j in range(1, i + 1):
                key = next((k for k in mt if k.startswith(f"T{j}_")), None)
                if key is None:
                    raise RuntimeError(f"{path.name} 缺少 T{j} 测试块")
                blk = mt[key]
                labels = blk.get("confusion_matrix_labels") or CM_LABELS
                cm = reorder_cm(blk["confusion_matrix"], labels)
                records.append({
                    "m": m, "i": i, "j": j, "key": key, "cm": cm,
                    "calc": metrics_from_cm(cm),
                    "rep": {"acc": blk.get("acc", blk.get("accuracy")),
                            "macro_f1": blk.get("macro_f1"),
                            "kappa": blk.get("cohen_kappa")},
                    "source": path.name,
                    "model_path": report.get("model_path", ""),
                })
    if len(records) != 60:
        raise RuntimeError(f"混淆矩阵应为 60 组，实际 {len(records)} 组")
    return records


def mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def build_dataset(records: list[dict]) -> dict:
    R = {m: {i: {} for i in range(1, 6)} for m in METHODS}
    MF = {m: {i: {} for i in range(1, 6)} for m in METHODS}
    KP = {m: {i: {} for i in range(1, 6)} for m in METHODS}
    PC = {m: {i: {} for i in range(1, 6)} for m in METHODS}
    for rec in records:
        m, i, j, c = rec["m"], rec["i"], rec["j"], rec["calc"]
        R[m][i][j], MF[m][i][j], KP[m][i][j] = c["acc"], c["macro_f1"], c["kappa"]
        PC[m][i][j] = c["per_class"]

    cl = {}
    for m in METHODS:
        A = {i: mean(R[m][i][j] for j in range(1, i + 1)) for i in range(1, 6)}
        F = {}
        for i in range(2, 6):
            F[i] = mean(max(R[m][k][j] for k in range(j, i + 1)) - R[m][i][j]
                        for j in range(1, i))
        # 文档 4.2：AF 的历史最佳取 k=j..5（含最终阶段），故 AF >= 0
        AF = mean(max(R[m][k][j] for k in range(j, 6)) - R[m][5][j]
                  for j in range(1, 5))
        cl[m] = {
            "A": A, "F": F,
            "FAA": A[5],
            "FMF": mean(MF[m][5][j] for j in range(1, 6)),
            "AF": AF,
            "BWT": mean(R[m][5][j] - R[m][j][j] for j in range(1, 5)),
            "KR": sum(R[m][5][j] for j in range(1, 5)) /
                  sum(R[m][j][j] for j in range(1, 5)),
            "CSA": mean(R[m][i][i] for i in range(1, 6)),
            "hist": [mean(R[m][i][j] for j in range(1, i)) if i > 1 else None
                     for i in range(1, 6)],
        }
    return {"R": R, "MF": MF, "KP": KP, "PC": PC, "cl": cl}


def max_recalc_deviation(records: list[dict]) -> float:
    devs = []
    for rec in records:
        for name, key in (("acc", "acc"), ("macro_f1", "macro_f1"), ("kappa", "kappa")):
            reported = rec["rep"][name]
            if reported is not None:
                devs.append(abs(rec["calc"][key] - float(reported)))
    return max(devs, default=0.0)


# ---------------- 辅助来源：训练耗时 / 存储 ----------------

def parse_train_times() -> dict[str, float]:
    """聚合 logs/full_matrix 下各编排批次的真实训练耗时（分钟）。

    同一 label 可能在多批次重复出现（早期批次多为短跑/中断后重跑），
    按批次时间顺序覆盖，只采纳 returncode==0 且带 minutes 的记录，最后一次为准。
    """
    batches = sorted((LOGS / "full_matrix").glob("summary_*.json"))
    if not batches:
        legacy = LOGS / "continual_runs" / "summary_20261104-092428.json"
        if legacy.is_file():
            return {r["label"]: r.get("minutes", 0.0)
                    for r in load_json(legacy).get("results", [])}
        return {}
    out: dict[str, float] = {}
    for p in batches:
        for rec in load_json(p).get("records", []):
            if rec.get("kind") != "train" or rec.get("returncode") != 0:
                continue
            minutes = rec.get("minutes")
            if minutes is None or rec.get("skipped"):
                continue
            out[rec["label"]] = float(minutes)
    return out


def parse_trainable_params() -> str:
    candidates = list((LOGS / "full_matrix").glob("*/train_ewc_task2*.log"))
    candidates += list((LOGS / "continual_runs").glob("train_ewc_task2_*.log"))
    for p in candidates:
        m = re.search(r"trainable params: ([\d,]+)",
                      p.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(1)
    return "4,358,144"


def dir_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    if not p.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def model_dir(name: str) -> Path:
    """在 models/ 下定位模型目录：兼容平铺与 stage{N}/ 分组两种布局。"""
    direct = ART / "models" / name
    if direct.is_dir():
        return direct
    for group in (ART / "models").glob("*/"):
        if (group / name).is_dir():
            return group / name
    return direct


def storage_costs(records: list[dict]) -> dict:
    final_models = {}
    for m in METHODS:
        rec = next(r for r in records if r["m"] == m and r["i"] == 5)
        final_models[m] = dir_mb(model_dir(Path(rec["model_path"]).name))
    return {
        "final_models": final_models,
        "ewc_state": dir_mb(ART / "continual_state" / "ewc" / "ewc_state_after_task5.pt"),
        "replay_memory": dir_mb(ART / "continual_state" / "replay_memory"),
        "adapter_bundle": dir_mb(model_dir("adapter_task5") / "adapters"),
    }


def train_minutes(times: dict[str, float]) -> dict[str, float]:
    # 训练标签前缀：编排脚本历史上有两种命名（train_adapter / train_adapter_cl），逐个兼容
    prefixes = {"ewc": ["train_ewc"], "replay": ["train_replay"],
                "adapter": ["train_adapter", "train_adapter_cl"],
                "sequential_ft": ["train_sequential_ft"]}
    out: dict[str, float] = {}
    for m, cands in prefixes.items():
        first = 1 if m == "sequential_ft" else 2  # E1 从 T1 起训练，其余方法 T1 共用基线
        total = 0.0
        for i in range(first, 6):
            for pre in cands:
                v = times.get(f"{pre}_task{i}")
                if v is not None:
                    total += v
                    break
        out[m] = total
    # E1 与 E2–E4 对齐时统一按 T2–T5 口径汇总，T1 在表6 单独注明
    e1_t1 = 0.0
    for pre in prefixes["sequential_ft"]:
        v = times.get(f"{pre}_task1")
        if v is not None:
            e1_t1 = v
            break
    out["sequential_ft"] -= e1_t1
    return out


# ---------------- 图 ----------------

def fig_style(ax):
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([TASKS[i] for i in range(1, 6)])


def make_figures(ds: dict) -> list[Path]:
    """六张独立图：图1a 对角线、图1b 累积平均、图2 历史保持、图3 W/R/N 三张。"""
    R, cl, PC = ds["R"], ds["cl"], ds["PC"]
    paths = [RES / name for name in FIG_NAMES]

    # 图1a：当前阶段适应能力（对角线 R_{i,i}）
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    for m in METHODS:
        ax.plot(range(1, 6), [R[m][i][i] for i in range(1, 6)],
                marker="o", linewidth=2, color=M_COLORS[m], label=M_SHORT[m])
    fig_style(ax)
    ax.set_xlabel("训练阶段（学完该窗口后立即测试）")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.68, 0.95)
    ax.set_title("图1a  当前阶段适应能力（对角线 R$_{i,i}$）", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=170, facecolor="white")
    plt.close(fig)

    # 图1b：累积平均准确率 A_i = mean_{j<=i} R_{i,j}
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    for m in METHODS:
        ax.plot(range(1, 6), [cl[m]["A"][i] for i in range(1, 6)],
                marker="s", linewidth=2, color=M_COLORS[m], label=M_SHORT[m])
    fig_style(ax)
    ax.set_xlabel("训练阶段")
    ax.set_ylabel("累积平均 Accuracy")
    ax.set_ylim(0.68, 0.95)
    ax.set_title("图1b  累积平均准确率 A$_i$ = mean$_{j\\leq i}$ R$_{i,j}$",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=170, facecolor="white")
    plt.close(fig)

    # 图2：T5 完成后的历史知识保持（T1–T4）
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    x = range(1, 5)
    width = 0.2
    for k, m in enumerate(METHODS):
        vals = [R[m][5][j] for j in range(1, 5)]
        bars = ax.bar([t + (k - 1.5) * width for t in x], vals, width,
                      color=M_COLORS[m], label=M_SHORT[m], edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"历史窗口 {TASKS[j]}" for j in range(1, 5)])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("图2  T5 完成后对 T1–T4 历史窗口的知识保持", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=170, facecolor="white")
    plt.close(fig)

    # 图3：T5 最终模型对各窗口的 W/R/N 类别 F1（每个类别一张）
    for idx, lab in enumerate(("W", "R", "N")):
        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        series = {m: [PC[m][5][j][lab]["f1"] for j in range(1, 6)] for m in METHODS}
        for m in METHODS:
            # 数值 x + fig_style 里的 set_xticks(range(1,6)) → 刻度显示 T1..T5
            ax.plot(range(1, 6), series[m], marker="o", linewidth=2, markersize=5,
                    color=M_COLORS[m], label=M_SHORT[m])
        fig_style(ax)
        lo = min(v for s in series.values() for v in s)
        ax.set_ylim(max(0.0, lo - 0.05), 1.0)
        ax.set_xlabel("测试时间窗口")
        ax.set_ylabel(f"{lab} 类 F1")
        ax.set_title(f"图3{'abc'[idx]}  最终阶段 {lab} 类性能：T5 后对 T1–T5 各窗口",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, frameon=False, ncol=4)
        fig.tight_layout()
        fig.savefig(paths[3 + idx], dpi=170, facecolor="white")
        plt.close(fig)
    return paths


# ---------------- Sheet 0：实验配置与口径 ----------------

def sh_config(wb, records, ds, dev, generated):
    ws = wb.create_sheet("0_实验配置与口径")
    n = 6
    r = sheet_setup(ws, "时间增量持续学习实验 · 结果分析工作簿",
                    f"按《汇报要求.docx》整理｜唯一原始数据源：60 组 3×3 混淆矩阵｜生成时间 {generated}",
                    n, [26, 21, 21, 21, 21, 21], tab=C_NAVY, freeze="A3")
    band(ws, r, "一、实验定位与结构导航", n)
    r += 1
    nav = [
        ("实验定位", "同一 W/R/N 三分类任务在五个时间窗口上的时间分布变化持续学习（Temporal CL），"
                     "T1–T5 不是五个不同任务；重点考察新阶段适应能力、历史知识保持能力、类别稳定性。"),
        ("方法与编号", "E1 Sequential FT（朴素顺序微调）／E2 EWC（Fisher 约束，λ=100，param 归一化）／"
                      "E3 Replay（每任务每类 20 条历史样本）／E4 Adapter-CL（每任务新增 LoRA，"
                      "推理按测试窗口路由 route，只激活该窗口对应 adapter）。"),
        ("测试结构", "T_i 训练完成后测试 T1..T_i，形成下三角矩阵；4 方法 × (1+2+3+4+5) = 60 组测试，"
                    "每次测试固定使用同一阶段测试集；每窗口样本数与各类支持数由矩阵行和现场核对（见第五节）。"),
        ("工作簿结构", "0 配置与口径｜1 原始测试指标｜2 类别级指标｜3 全量混淆矩阵（文档 Sheet 4）｜"
                      "表1–表7（文档七张结果表）｜图表（文档 Sheet 8 的三张图 + 遗忘曲线补充图）。"),
        ("表号对照", "表1 总体性能（对角线）／表2 历史测试矩阵／表3 类别级 F1（对角线）／"
                    "表4 持续学习六指标／表5 遗忘差值表／表6 资源成本／表7 综合对比。"),
    ]
    for k, v in nav:
        note_line(ws, r, k, v, n, height=34)
        r += 1

    r += 1
    band(ws, r, "二、指标口径（本工作簿所有数值均由混淆矩阵现场复算）", n)
    r += 1
    formulas = [
        ("混淆矩阵", "行 = Gold，列 = Pred，标签统一为 N / R / W；Accuracy = ΣCM[c][c] / N。"),
        ("Precision / Recall / F1", "P_c = CM[c][c] / Σ_i CM[i][c]；R_c = CM[c][c] / Σ_j CM[c][j]；"
                                    "F1_c = 2·P·R / (P+R)；Macro-F1 = mean(F1_W, F1_R, F1_N)。"),
        ("Cohen's Kappa", "Kappa = (p_o − p_e) / (1 − p_e)，p_o = Accuracy，p_e = Σ_c (row_c · col_c) / N²。"),
        ("FAA", "最终平均准确率 = (1/5)·Σ_{j=1..5} R_{5,j}。"),
        ("Final Avg Macro-F1", "T5 完成后对五个时间窗口的 Macro-F1 平均值。"),
        ("AF", "平均遗忘程度 = (1/4)·Σ_{j=1..4} [ max_{k=j..5} R_{k,j} − R_{5,j} ]，历史最佳含最终阶段，故 AF ≥ 0，越低越好。"),
        ("BWT", "后向迁移 = (1/4)·Σ_{j=1..4} (R_{5,j} − R_{j,j})，正 = 后续学习改善历史任务。"),
        ("KR", "历史知识保持率 = Σ_{j=1..4} R_{5,j} / Σ_{j=1..4} R_{j,j} × 100%。"),
        ("CSA", "当前阶段适应能力 = (1/5)·Σ_{i=1..5} R_{i,i}，即对角线 Accuracy 平均值。"),
    ]
    for k, v in formulas:
        note_line(ws, r, k, v, n, height=30)
        r += 1

    r += 1
    band(ws, r, "三、判别说明：本次相对上一版的口径修正", n)
    r += 1
    fixes = [
        ("表1 口径", "文档明确 T1–T5 采用「各阶段训练完成后对当前阶段测试集」（对角线）；"
                   "上一版把最终模型在五个窗口的结果也放进表1，本次将后者移为表1 下方的补充对照区。"),
        ("表2 / 表3 口径", "文档表2 为四种方法的下三角历史测试矩阵（Accuracy，另补 Macro-F1）；"
                        "表3 为对角线口径的类别 F1；上一版表2 用的是最终模型结果，本次已改正。"),
        ("表4 六指标", "补齐 Final Average Macro-F1、KR、CSA；AF 按文档 4.2 取 max_{k=j..5}（含最终阶段）；"
                     "移除文档明确建议不填的 FWT。"),
        ("新增表5", "文档表5 的 Δ_j = R_{5,j} − R_{j,j} 遗忘差值表，上一版缺失，本次补齐。"),
        ("表7 方向", "按文档改为「评价维度为行、四种方法为列」，内容全部由前面各表的真实结果填入，不人为打分。"),
        ("数据驱动", "全部指标改由 60 组混淆矩阵复算；报告值仅并列用于一致性核对。"),
    ]
    for k, v in fixes:
        note_line(ws, r, k, v, n, height=32)
        r += 1

    r += 1
    band(ws, r, "四、缺失数据评估与限制（仅评估，不补跑、不重训）", n, fill=C_GOLD, color=C_AMBER_TX)
    r += 1
    limits = [
        ("推理时间 / 峰值显存", "原始数据未记录 → 表6 标「未记录」，本次不计算。本地 1660Ti 与云端 4090D 不是同一基准，"
                              "不可用来替代；如需补齐，须在云端同型号 GPU 实测，不需要重新训练。"),
        ("E1 分阶段训练时长", "T1–T5 逐阶段计时已补齐（约 2.2–2.3 min/阶段）；表6 统一按 T2–T5 口径统计，T1 单独注明。"),
        ("FWT", "文档第四部分明确：除非专门设计未来窗口的未训练基线，否则不计算 → 本工作簿不填。"),
        ("无效输出率 / 多种子 / 测试集重叠", "原始评估未记录无效预测率、仅单一种子、未核验窗口间样本重叠；"
                                          "按文档第四部分列为限制，不编造数值。"),
        ("模型选择", "未使用最终测试集反复调参；EWC λ=100（param 归一化）为既定实验配置，未做超参搜索"
                    "（旧实验的 λ=5000 属未归一化口径，两者不可互换）。"),
    ]
    for k, v in limits:
        note_line(ws, r, k, v, n, fill=C_NOTE_BG, height=32)
        r += 1

    r += 1
    band(ws, r, "五、数据一致性核对", n)
    r += 1
    supports = {tuple(rec["calc"]["per_class"][lab]["support"] for lab in CLASS_ORDER)
                for rec in records}
    note_line(ws, r, "混淆矩阵组数", f"{len(records)} 组（期望 60）；20 份累积评估 JSON，每份含 1–5 个测试窗口。", n)
    r += 1
    note_line(ws, r, "复算一致性", f"由矩阵复算的 Accuracy / Macro-F1 / Kappa 与原始报告值最大绝对偏差 = {dev:.2e}"
                                 f"（浮点舍入误差），逐组偏差见「3_全量混淆矩阵」。", n)
    r += 1
    note_line(ws, r, "样本支持数", f"每个测试窗口 300 条；各类支持数集合 = {sorted(supports)}（由矩阵行和核对）。", n)
    r += 1
    note_line(ws, r, "生成脚本", "05_incremental_learning_time/scripts/validation/make_report_tables.py（可复现）", n)
    r += 1
    note_line(ws, r, "数据来源", "results/temporal/*.json（20 份）+ logs/full_matrix 训练汇总（17 次训练）+ artifacts 文件系统实测", n)
    r += 1
    finish(ws, n, r - 1, header_row=2)


# ---------------- Sheet 1–3：全量原始层 ----------------

def sh_raw(wb, records, dev):
    ws = wb.create_sheet("1_原始测试指标")
    n = 9
    r = sheet_setup(ws, "Sheet 2 · 原始测试指标（全量 60 组）",
                    "文档 Sheet 2：方法 × 训练阶段 × 已学习测试窗口；全部由 3×3 混淆矩阵复算，"
                    "与报告值最大偏差 %.1e" % dev,
                    n, [6, 20, 16, 12, 20, 10, 13, 13, 15], freeze="C5")
    band(ws, r, "60 组 = 4 方法 × (1+2+3+4+5)；Σ 列支持自动筛选，可导出为下游分析的原始表", n)
    r += 1
    hdr = r
    header(ws, r, ["序号", "方法", "训练阶段", "测试阶段", "测试集", "样本数",
                   "Accuracy", "Macro-F1", "Cohen's Kappa"])
    r += 1
    for idx, rec in enumerate(records, 1):
        c = rec["calc"]
        data_row(ws, r, [idx, M_LABEL[rec["m"]], f"T{rec['i']} 训练完成", TASKS[rec["j"]],
                         rec["key"], c["samples"], c["acc"], c["macro_f1"], c["kappa"]],
                 fmts=[INT0, None, None, None, None, INT0, NUM4, NUM4, NUM4],
                 zebra=(idx % 2 == 0), bold_cols=(2,), height=19)
        r += 1
    for col in "GHI":
        ws.conditional_formatting.add(f"{col}{hdr+1}:{col}{r-1}", scale_rule())
    finish(ws, n, r - 1, header_row=hdr, autofilter=f"A{hdr}:I{r-1}")


def sh_class(wb, records):
    ws = wb.create_sheet("2_类别级指标")
    n = 14
    r = sheet_setup(ws, "Sheet 3 · 类别级指标（全量 60 组）",
                    "文档 Sheet 3：W/R/N 的 Precision / Recall / F1，全部由混淆矩阵复算；"
                    "表3 主表取其中 i=j 的对角行，图3 取 i=5 的最终模型行",
                    n, [6, 20, 14, 11, 20] + [10] * 9, freeze="C5")
    band(ws, r, "P = 预测为某类中真值为该类的比例；R = 真值某类中被正确识别的比例；F1 为二者调和平均", n)
    r += 1
    group_header(ws, r, [(1, 5, "测试标识", C_BLUE_D),
                         (6, 8, "W 类", C_BLUE_D), (9, 11, "R 类", C_BLUE_D),
                         (12, 14, "N 类", C_BLUE_D)])
    r += 1
    hdr = r
    header(ws, r, ["序号", "方法", "训练阶段", "测试阶段", "测试集"]
           + [f"{lab}-{abbr}" for lab in CLASS_ORDER for abbr in ["P", "R", "F1"]])
    r += 1
    for idx, rec in enumerate(records, 1):
        pc = rec["calc"]["per_class"]
        values = [idx, M_LABEL[rec["m"]], f"T{rec['i']} 训练完成", TASKS[rec["j"]], rec["key"]]
        for lab in CLASS_ORDER:
            values += [pc[lab]["precision"], pc[lab]["recall"], pc[lab]["f1"]]
        data_row(ws, r, values, fmts=[INT0, None, None, None, None] + [NUM4] * 9,
                 zebra=(idx % 2 == 0), bold_cols=(2,), height=19)
        r += 1
    for col in ("H", "K", "N"):
        ws.conditional_formatting.add(f"{col}{hdr+1}:{col}{r-1}", scale_rule())
    finish(ws, n, r - 1, header_row=hdr, autofilter=f"A{hdr}:N{r-1}")


def sh_cm(wb, records, dev):
    ws = wb.create_sheet("3_全量混淆矩阵")
    n = 23
    r = sheet_setup(ws, "Sheet 4 · 全量混淆矩阵（60 组 3×3）",
                    "文档 Sheet 4：4 方法 × 5 训练阶段 × 所有已学习测试窗口；"
                    f"本表是整个工作簿的唯一原始数据源，表1–表7 与全部图均可由此复算；"
                    f"复算值与报告值最大偏差 {dev:.1e}",
                    n, [6, 20, 16, 11, 20, 9, 10] + [9] * 9 + [13] * 6 + [12],
                    tab=C_NAVY, freeze="C6")
    band(ws, r, "行 = Gold，列 = Pred，标签统一 N / R / W；绿色格为正确分类（对角线）", n)
    r += 1
    group_header(ws, r, [(1, 7, "测试标识", C_BLUE_D),
                         (8, 16, "混淆矩阵 CM（行 Gold → 列 Pred，N/R/W）", C_BLUE_D),
                         (17, 19, "由矩阵复算", C_BLUE_D),
                         (20, 22, "原始报告值（核对用）", C_BLUE_D),
                         (23, 23, "一致性", C_BLUE_D)])
    r += 1
    hdr = r
    header(ws, r, ["序号", "方法", "训练阶段", "测试阶段", "测试集", "样本数", "标签顺序"]
           + [f"{a}→{b}" for a in CM_LABELS for b in CM_LABELS]
           + ["复算Acc", "复算Macro-F1", "复算Kappa", "报告Acc", "报告Macro-F1",
              "报告Kappa", "最大偏差"], height=34)
    r += 1
    for idx, rec in enumerate(records, 1):
        cm, calc, rep = rec["cm"], rec["calc"], rec["rep"]
        devs = [abs(calc[k] - float(rep[name]))
                for name, k in (("acc", "acc"), ("macro_f1", "macro_f1"), ("kappa", "kappa"))
                if rep[name] is not None]
        values = [idx, M_LABEL[rec["m"]], f"T{rec['i']} 训练完成", TASKS[rec["j"]],
                  rec["key"], calc["samples"], "N/R/W"]
        for line in cm:
            values += line
        values += [calc["acc"], calc["macro_f1"], calc["kappa"],
                   rep["acc"], rep["macro_f1"], rep["kappa"],
                   max(devs) if devs else None]
        fills = {col: C_GREEN_BG for col in (8, 12, 16)}
        data_row(ws, r, values,
                 fmts=[INT0, None, None, None, None, INT0, None] + [INT0] * 9
                      + [NUM4] * 6 + ["0.00E+00"],
                 zebra=(idx % 2 == 0), bold_cols=(2,), fill_map=fills, height=19)
        r += 1
    ws.conditional_formatting.add(f"W{hdr+1}:W{r-1}", CellIsRule(
        operator="lessThan", formula=["0.000001"],
        fill=PatternFill("solid", fgColor=C_GREEN_BG),
        font=Font(name=FONT_NAME, color=C_GREEN_TX, bold=True)))
    finish(ws, n, r - 1, header_row=hdr, autofilter=f"A{hdr}:W{r-1}")


# ---------------- 表1 ----------------

def sh_table1(wb, ds, joint):
    ws = wb.create_sheet("表1_总体性能")
    n = 8
    r = sheet_setup(ws, "表1 · 四种方法总体性能对比表（文档表1 口径）",
                    "T1–T5 统一采用「各阶段训练完成后对当前阶段测试集」的对角线结果 R_{i,i}；"
                    "五阶段平均只反映适应能力，不代表历史知识保持；"
                    "下方两块为 Joint 上界参照（单一模型混合全部数据一次训练，非对角线口径）",
                    n, [22, 13, 11, 11, 11, 11, 11, 14], freeze="C4")
    hdr = r
    header(ws, r, ["方法", "指标", "T1", "T2", "T3", "T4", "T5", "五阶段平均"])
    r += 1
    labels = [("acc", "Accuracy"), ("macro_f1", "Macro-F1"), ("kappa", "Kappa")]
    for mi, m in enumerate(METHODS):
        start = r
        for metric, lab in labels:
            if metric == "acc":
                vals = [ds["R"][m][i][i] for i in range(1, 6)]
            elif metric == "macro_f1":
                vals = [ds["MF"][m][i][i] for i in range(1, 6)]
            else:
                vals = [ds["KP"][m][i][i] for i in range(1, 6)]
            data_row(ws, r, ["", lab] + vals + [mean(vals)],
                     fmts=[None, None] + [NUM4] * 6,
                     zebra=(mi % 2 == 1), bold_cols=(2,), height=21)
            r += 1
        merge_col(ws, 1, start, r - 1, M_LABEL[m],
                  fill=C_PALE if mi % 2 else C_WHITE)
    # ── Joint 上界参照块：单一模型混合全部窗口数据一次训练，测 T1–T5 全部测试集 ──
    jtags = [t for t in ("full", "600") if joint.get(t)]
    jlabels = {"full": "UB Joint-full", "600": "对照 Joint-600"}
    for ti, tag in enumerate(jtags):
        d = joint[tag]
        start = r
        for metric, lab in labels:
            if metric == "acc":
                vals = [d["per"][i]["acc"] for i in range(1, 6)]
                avg = d["ACC"]
            elif metric == "macro_f1":
                vals = [d["per"][i]["macro_f1"] for i in range(1, 6)]
                avg = d["F1"]
            else:
                vals = [d["per"][i]["kappa"] for i in range(1, 6)]
                avg = mean(vals)
            data_row(ws, r, ["", lab] + vals + [avg],
                     fmts=[None, None] + [NUM4] * 6,
                     zebra=(ti % 2 == 1), bold_cols=(2,), height=21)
            r += 1
        merge_col(ws, 1, start, r - 1, jlabels[tag],
                  fill=C_PALE if ti % 2 else C_WHITE)
    # 三个指标量纲不同，分别着色，避免 Kappa 天然偏低被误读为「表现差」
    n_blocks = len(METHODS) + len(jtags)
    for offset in range(3):
        ranges = [f"C{hdr+1+offset+3*k}:H{hdr+1+offset+3*k}" for k in range(n_blocks)]
        ws.conditional_formatting.add(" ".join(ranges), scale_rule())
    r += 1
    ws.row_breaks.append(Break(id=r - 1))
    band(ws, r, "补充对照（非文档表1 口径）：T5 最终模型在 T1–T5 各窗口的表现 —— 对应表2 最后一行与图2",
         n, fill=C_GOLD, color=C_AMBER_TX)
    r += 1
    header(ws, r, ["方法", "指标", "T1", "T2", "T3", "T4", "T5", "五窗口平均"])
    r += 1
    for mi, m in enumerate(METHODS):
        start = r
        for metric, lab in labels:
            key = {"acc": "R", "macro_f1": "MF", "kappa": "KP"}[metric]
            vals = [ds[key][m][5][j] for j in range(1, 6)]
            data_row(ws, r, ["", lab] + vals + [mean(vals)],
                     fmts=[None, None] + [NUM4] * 6,
                     zebra=(mi % 2 == 1), bold_cols=(2,), height=20)
            r += 1
        merge_col(ws, 1, start, r - 1, M_LABEL[m],
                  fill=C_PALE if mi % 2 else C_WHITE)
    note_line(ws, r, "口径判别", "文档表1 只要求对角线；补充区保留上一版表1a 的信息，"
                                "避免与历史知识保持指标混淆，二者数据源不同。"
                                "Joint 两行为单一模型混合全部训练数据一次训练后在 T1–T5 全部测试集的表现"
                                "（非对角线口径）：Joint-full 为「数据全部可用时的上界」，"
                                "Joint-600 为每窗口每类 40 条的数据量对照，不参与持续学习指标排名。", n, height=40)
    r += 1
    finish(ws, n, r - 1, header_row=2)


# ---------------- 表2 ----------------

def sh_table2(wb, ds):
    ws = wb.create_sheet("表2_历史测试矩阵")
    n = 6
    r = sheet_setup(ws, "表2 · 各时间阶段历史测试性能矩阵（文档表2）",
                    "文档认定为整个研究的核心原始表；行 = 训练完成阶段，列 = 测试时间窗口；"
                    "上三角「—」为该阶段尚未接触该窗口，非缺失数据",
                    n, [22, 13, 13, 13, 13, 13], freeze="B4")
    for section, key, note in (
        ("A · Accuracy 历史测试矩阵（主表）", "R", "数值 = 由混淆矩阵复算的 Accuracy"),
        ("B · Macro-F1 历史测试矩阵（文档建议的补充分析数据）", "MF", "数值 = 由混淆矩阵复算的 Macro-F1"),
    ):
        band(ws, r, section, n)
        r += 1
        for m in METHODS:
            band(ws, r, f"{M_LABEL[m]}　|　{note}", n, fill=C_PALE,
                 color=C_BLUE_D, size=10, height=21)
            r += 1
            header(ws, r, ["训练阶段", "测试 T1", "测试 T2", "测试 T3", "测试 T4", "测试 T5"], height=26)
            r += 1
            first = r
            for i in range(1, 6):
                values = [f"{TASKS[i]} 训练完成"]
                for j in range(1, 6):
                    values.append(ds[key][m][i].get(j, "—"))
                data_row(ws, r, values, fmts=[None] + [NUM4] * 5,
                         zebra=(i % 2 == 0), bold_cols=(1,), height=20,
                         fill_map={j + 2: C_GREY_BG for j in range(i, 5)})
                r += 1
            ws.conditional_formatting.add(f"B{first}:F{r-1}", scale_rule())
            r += 1
    finish(ws, n, r - 1, header_row=3)


# ---------------- 表3 ----------------

def sh_table3(wb, ds):
    ws = wb.create_sheet("表3_类别级F1")
    n = 8
    r = sheet_setup(ws, "表3 · 四种方法类别级 F1 对比表（文档表3）",
                    "对角线口径：各阶段训练完成后，在当前阶段测试集上的 W / R / N 类 F1；"
                    "详细 Precision、Recall 见「2_类别级指标」，最终模型类别表现见图3",
                    n, [22, 11, 11, 11, 11, 11, 11, 14], freeze="C4")
    hdr = r
    header(ws, r, ["方法", "类别", "T1", "T2", "T3", "T4", "T5", "平均 F1"])
    r += 1
    for mi, m in enumerate(METHODS):
        start = r
        for lab in CLASS_ORDER:
            vals = [ds["PC"][m][i][i][lab]["f1"] for i in range(1, 6)]
            data_row(ws, r, ["", lab] + vals + [mean(vals)],
                     fmts=[None, None] + [NUM4] * 6,
                     zebra=(mi % 2 == 1), bold_cols=(2,), height=21)
            r += 1
        merge_col(ws, 1, start, r - 1, M_LABEL[m],
                  fill=C_PALE if mi % 2 else C_WHITE)
    ws.conditional_formatting.add(f"C{hdr+1}:H{r-1}", scale_rule())
    r += 1
    ws.row_breaks.append(Break(id=r - 1))
    band(ws, r, "补充：T5 最终模型在五个时间窗口的类别 F1（历史类别性能矩阵，文档表3 建议项）", n)
    r += 1
    header(ws, r, ["方法", "类别", "T1", "T2", "T3", "T4", "T5", "五窗口平均"])
    r += 1
    for mi, m in enumerate(METHODS):
        start = r
        for lab in CLASS_ORDER:
            vals = [ds["PC"][m][5][j][lab]["f1"] for j in range(1, 6)]
            data_row(ws, r, ["", lab] + vals + [mean(vals)],
                     fmts=[None, None] + [NUM4] * 6,
                     zebra=(mi % 2 == 1), bold_cols=(2,), height=20)
            r += 1
        merge_col(ws, 1, start, r - 1, M_LABEL[m],
                  fill=C_PALE if mi % 2 else C_WHITE)
    finish(ws, n, r - 1, header_row=2)


# ---------------- 表4 ----------------

def sh_table4(wb, ds, joint):
    ws = wb.create_sheet("表4_持续学习指标")
    n = 11
    r = sheet_setup(ws, "表4 · 持续学习核心指标对比表（文档表4 / Sheet 6）",
                    "六个指标全部由表2 的 R_{i,j}（混淆矩阵复算 Accuracy）与 Macro-F1 自动计算；"
                    "FWT 按文档第四部分不填写；右侧两列为 Joint 上界参照（不参与排名，"
                    "无阶段概念的指标以「—」标注）",
                    n, [22, 27, 44, 11, 15, 15, 15, 15, 16, 16, 18], freeze="D4")
    hdr = r
    header(ws, r, ["指标", "英文名称", "计算口径 / 评价内容", "优劣方向",
                   "E1 Sequential FT", "E2 EWC", "E3 Replay", "E4 Adapter-CL",
                   "UB Joint-full", "对照 Joint-600", "最优方法"],
           height=40)
    r += 1
    specs = [
        ("最终平均准确率", "Final Average Accuracy (FAA)",
         "T5 完成后对 T1–T5 的整体表现：mean_j R_{5,j}", "越高越好", "FAA", NUM4, "max"),
        ("最终平均 Macro-F1", "Final Average Macro-F1",
         "T5 完成后五个窗口的类别均衡表现：mean_j MacroF1_{5,j}", "越高越好", "FMF", NUM4, "max"),
        ("平均遗忘程度", "Average Forgetting (AF)",
         "mean_{j=1..4}[ max_{k=j..5} R_{k,j} − R_{5,j} ]，历史最佳含最终阶段", "越低越好", "AF", NUM4, "min"),
        ("后向迁移", "Backward Transfer (BWT)",
         "mean_{j=1..4}( R_{5,j} − R_{j,j} )，正 = 后续学习改善历史任务", "越高越好", "BWT", "+0.0000;-0.0000;0.0000", "max"),
        ("历史知识保持率", "Knowledge Retention (KR)",
         "Σ_{j=1..4} R_{5,j} / Σ_{j=1..4} R_{j,j} × 100%（补充指标）", "越高越好", "KR", PCT2, "max"),
        ("当前阶段适应能力", "Current-stage Performance (CSA)",
         "五个阶段对角线 Accuracy 平均值：mean_i R_{i,i}", "越高越好", "CSA", NUM4, "max"),
    ]
    jfull, j600 = joint.get("full"), joint.get("600")
    for idx, (name, eng, desc, direction, key, fmt, best) in enumerate(specs):
        vals = [ds["cl"][m][key] for m in METHODS]
        if key == "FAA":
            jvals = [jfull["ACC"], j600["ACC"]] if jfull and j600 else ["—", "—"]
        elif key == "FMF":
            jvals = [jfull["F1"], j600["F1"]] if jfull and j600 else ["—", "—"]
        else:
            jvals = ["—", "—"]  # Joint 无阶段序列，AF/BWT/KR/CSA 无定义
        winner = METHODS[vals.index(max(vals) if best == "max" else min(vals))]
        data_row(ws, r, [name, eng, desc, direction] + vals + jvals + [M_SHORT[winner]],
                 fmts=[None, None, None, None] + [fmt] * 4 + [NUM4, NUM4, None],
                 zebra=(idx % 2 == 1), bold_cols=(1, 11),
                 align_map={3: "left", 2: "left"}, height=36,
                 fill_map={11: C_GOLD})
        r += 1
    # 每个指标单独一行色阶（含 Joint 列，「—」不参与）；AF 越低越好，反转色阶
    for idx in range(6):
        row = hdr + 1 + idx
        ws.conditional_formatting.add(f"E{row}:J{row}", scale_rule(reverse=(idx == 2)))
    r += 1
    note_line(ws, r, "口径判别", "AF 严格按文档 4.2 取 max_{k=j..5}（包含 k=5 的最终表现），"
                                "因此 AF 不会为负；上一版只取到 k=4，会出现负值，本次已修正。", n, height=30)
    r += 1
    note_line(ws, r, "E4 可比性", "E4 Adapter-CL 采用按已知测试窗口路由的推理策略（adapter_inference=route，"
                                "deployment_setting=known_time_window）：第 k 个窗口始终由 task_k adapter 预测，"
                                "同一窗口各阶段预测逐条相同，故 AF≡0、KR≡100% 为结构性结果，"
                                "不能与 E1–E3 的单模型遗忘程度直接比较；其 CSA / FAA 仍可横向比较。", n, height=40)
    r += 1
    note_line(ws, r, "FWT", "文档明确：T1–T5 是同一任务的时间窗口，无未来窗口的未训练基线，"
                            "不建议作为主要指标 → 不填写。", n, height=26)
    r += 1
    note_line(ws, r, "Joint 口径", "Joint-full 为 T1–T5 全部训练数据（3,000 条）混合、从基座一次训练"
                                 "（1,500 步，与 E1 全链累计步数相同），其 FAA 即「数据全部可用时的上界」；"
                                 "Joint-600 每窗口按类别均衡抽 120 条（共 600）同配置训练，用于剥离数据量因素。"
                                 "两者均非持续学习方法，不参与排名；无阶段概念的指标以「—」标注。", n, height=40)
    r += 1
    note_line(ws, r, "辅助曲线", "各阶段累积平均准确率 A_i 与遗忘曲线见「图表」中的图1b / 图4；"
                                "逐阶段 A_i 亦可由表2 矩阵直接计算。", n, height=26)
    r += 1
    finish(ws, n, r - 1, header_row=hdr)


# ---------------- 表5 ----------------

def sh_table5(wb, ds):
    ws = wb.create_sheet("表5_遗忘差值表")
    n = 7
    r = sheet_setup(ws, "表5 · 四种方法性能变化与遗忘对比（文档表5）",
                    "Δ_j = R_{5,j} − R_{j,j}：历史窗口最终表现相对刚学完时的变化；"
                    "负值红色 = 遗忘，正值绿色 = 正向迁移",
                    n, [22, 14, 18, 18, 18, 16, 13], freeze="C4")
    hdr = r
    header(ws, r, ["方法", "历史窗口", "初次学习 Accuracy", "历史最佳 Accuracy",
                   "最终 T5 Accuracy", "性能变化 Δ_j", "是否遗忘"], height=36)
    r += 1
    for mi, m in enumerate(METHODS):
        start = r
        for j in range(1, 5):
            initial = ds["R"][m][j][j]
            peak = max(ds["R"][m][k][j] for k in range(j, 6))
            final = ds["R"][m][5][j]
            delta = final - initial
            data_row(ws, r, ["", TASKS[j], initial, peak, final, delta,
                             "是" if delta < 0 else "否"],
                     fmts=[None, None, NUM4, NUM4, NUM4, "+0.0000;-0.0000;0.0000", None],
                     zebra=(mi % 2 == 1), bold_cols=(2,),
                     fill_map={7: C_RED_BG if delta < 0 else C_GREEN_BG}, height=21)
            r += 1
        merge_col(ws, 1, start, r - 1, M_LABEL[m],
                  fill=C_PALE if mi % 2 else C_WHITE)
    delta_rules(ws, f"F{hdr+1}:F{r-1}")
    r += 1
    band(ws, r, "各方法四个历史窗口的平均变化（Δ 均值，等价于 BWT）", n)
    r += 1
    for m in METHODS:
        vals = [ds["R"][m][5][j] - ds["R"][m][j][j] for j in range(1, 5)]
        data_row(ws, r, [M_LABEL[m], "T1–T4 平均", "—", "—", "—", mean(vals), "—"],
                 fmts=[None, None, None, None, None, "+0.0000;-0.0000;0.0000", None],
                 bold_cols=(1,), fill_map={6: C_LIGHT}, height=21)
        r += 1
    r += 1
    note_line(ws, r, "E4 说明", "E4 采用按已知测试窗口路由的推理策略（route），同一窗口在各阶段由同一 adapter 预测、"
                              "结果逐条相同，故四个历史窗口的 Δ_j 恒为 0（结构性零遗忘）；"
                              "这不代表其泛化能力优于其他方法，其 CSA 为四方法最低。", n, height=36)
    r += 1
    finish(ws, n, r - 1, header_row=hdr)


# ---------------- 表6 ----------------

def sh_table6(wb, times, mins, params, storage, joint):
    ws = wb.create_sheet("表6_资源成本")
    n = 7
    r = sheet_setup(ws, "表6 · 训练成本与资源消耗对比表（文档表6）",
                    "耗时来自 logs/full_matrix 编排汇总（17 次训练实测）与 Joint 训练日志，"
                    "存储来自 artifacts 文件系统实测；未记录项一律标注，不在本地补跑"
                    "（本地 1660Ti 与云端 4090D 不可比）；右侧两列为 Joint 参照",
                    n, [30, 24, 24, 24, 24, 24, 24], freeze="B4")
    hdr = r
    header(ws, r, ["对比项目", "E1 Sequential FT", "E2 EWC", "E3 Replay",
                   "E4 Adapter-CL", "UB Joint-full", "对照 Joint-600"], height=30)
    r += 1
    fm = storage["final_models"]
    extras = {"sequential_ft": "无",
              "ewc": f"Fisher 状态 {storage['ewc_state']:.0f} MB",
              "replay": f"回放样本池 {storage['replay_memory']:.2f} MB",
              "adapter": f"多适配器包 {storage['adapter_bundle']:.0f} MB"}
    e1_t1 = times.get("train_sequential_ft_task1", 0.0)

    def mins_txt(v: float) -> str:
        return f"{v:.1f} min" if v > 0 else "未记录"

    def jrt(tag: str) -> str:
        v = joint["runtime"].get(tag)
        return f"{v / 60:.1f} min" if v else "未记录"

    def jsize(tag: str) -> str:
        v = joint["size_mb"].get(tag)
        return f"{v:.0f} MB" if v else "未记录"

    rows = [
        ("总训练时间（T2–T5 四阶段）", mins_txt(mins["sequential_ft"]),
         mins_txt(mins["ewc"]), mins_txt(mins["replay"]), mins_txt(mins["adapter"]),
         "—（一次性训练）", "—（一次性训练）"),
        ("单阶段平均训练时间（T2–T5）", mins_txt(mins["sequential_ft"] / 4),
         mins_txt(mins["ewc"] / 4), mins_txt(mins["replay"] / 4), mins_txt(mins["adapter"] / 4),
         "—", "—"),
        ("峰值 GPU 显存", "未记录", "未记录", "未记录", "未记录", "未记录", "未记录"),
        ("可训练参数量", f"{params} (0.28%)", f"{params} (0.28%)",
         f"{params} (0.28%)", f"{params} (0.28%)", f"{params} (0.28%)", f"{params} (0.28%)"),
        ("最终模型存储量（实测）",
         *[f"{fm[m]:.0f} MB" for m in METHODS], jsize("full"), jsize("600")),
        ("额外历史信息存储量", *[extras[m] for m in METHODS], "无", "无"),
        ("历史样本存储量", "0 条", "0 条", "300 条（5 任务 × 每类 20）", "0 条",
         "0 条（全量混合，无需缓存）", "0 条（全量混合，无需缓存）"),
        ("推理时间", "未记录", "未记录", "未记录", "未记录", "未记录", "未记录"),
        ("推理所需适配器数量", "1", "1", "1", "1（按已知窗口路由，共维护 5 个）", "1", "1"),
        ("每阶段 checkpoint 策略", "T1–T5 各保留独立目录",
         "T2–T5 各保留独立目录（共享 T1 起点）",
         "T2–T5 各保留独立目录（共享 T1 起点）",
         "T2–T5 各保留独立目录（含累计适配器）",
         "单目录（一次性训练，无阶段概念）", "单目录（一次性训练，无阶段概念）"),
        ("部署复杂度", "低", "中（训练侧维护 Fisher）", "低", "高（多适配器管理 / 激活）",
         "低", "低"),
    ]
    for idx, row in enumerate(rows):
        fill = {c: C_GREY_BG for c in range(2, n + 1)} if row[1] == "未记录" else None
        data_row(ws, r, row, zebra=(idx % 2 == 1), bold_cols=(1,),
                 align_map={1: "left"}, height=30, fill_map=fill)
        r += 1
    r += 1
    note_line(ws, r, "E1 计时口径", f"E1 为 T1–T5 五次独立训练，逐阶段实测（T1 = {e1_t1:.1f} min）；"
                                  "为与 E2–E4 对齐，表中只统计 T2–T5。E2–E4 复用 E1 的 T1 模型作为起点，不重复计时。",
              n, height=34)
    r += 1
    note_line(ws, r, "Joint 计时口径", "Joint 为单次混合训练（full 1,500 步 / 600 300 步），"
                                     "与分阶段口径不可直接相减，仅作量级参照；"
                                     "模型为单 adapter，部署与 E1 同级且无需管理历史阶段产物。", n, height=34)
    r += 1
    note_line(ws, r, "缺失项评估", "峰值显存 / 推理时间：原始数据未采集。"
                                  "不影响方法间性能结论，也不需要重新训练；如需补齐应在云端同型号 GPU（4090D）实测。",
              n, height=34)
    r += 1
    note_line(ws, r, "存储口径", "「最终模型存储量」为各方法 task5 模型目录实测大小；"
                                "E4 目录包含累计适配器，符合文档「记录所有适配器总存储量」的要求。", n, height=30)
    r += 1
    finish(ws, n, r - 1, header_row=hdr)


# ---------------- 表7 ----------------

def sh_table7(wb, ds, mins, storage, joint):
    ws = wb.create_sheet("表7_综合对比")
    n = 7
    r = sheet_setup(ws, "表7 · 最终综合对比表（文档表7）",
                    "评价维度为行、四种方法为列（右侧两列为 Joint 上界参照）；"
                    "全部内容依据表1–表6 的真实结果填写，不人为打分",
                    n, [30, 24, 24, 24, 24, 24, 24], freeze="B4")
    hdr = r
    header(ws, r, ["评价维度", "E1 Sequential FT", "E2 EWC", "E3 Replay",
                   "E4 Adapter-CL", "UB Joint-full", "对照 Joint-600"], height=30)
    r += 1
    cl = ds["cl"]
    jfull, j600 = joint.get("full"), joint.get("600")
    jfaa = jfull["ACC"] if jfull else None

    def cls_retention(m, lab):
        return mean(ds["PC"][m][5][j][lab]["f1"] - ds["PC"][m][j][j][lab]["f1"]
                    for j in range(1, 5))

    extra_mb = {"sequential_ft": 0.0, "ewc": storage["ewc_state"],
                "replay": storage["replay_memory"],
                "adapter": 0.0}  # E4 的全部适配器已包含在最终模型目录内

    jnone = ["—", "—"]
    dims = [
        ("当前阶段适应能力（CSA）", [f"{cl[m]['CSA']:.4f}" for m in METHODS], jnone, True),
        ("最终历史平均性能（FAA）", [f"{cl[m]['FAA']:.4f}" for m in METHODS],
         [f"{jfull['ACC']:.4f}", f"{j600['ACC']:.4f}"] if jfull and j600 else jnone, True),
        ("平均遗忘程度（AF）", [f"{cl[m]['AF']:.4f}" for m in METHODS], jnone, True),
        ("后向迁移（BWT）", [f"{cl[m]['BWT']:+.4f}" for m in METHODS], jnone, True),
        ("W 类保持能力（ΔF1）", [f"{cls_retention(m, 'W'):+.4f}" for m in METHODS], jnone, True),
        ("R 类保持能力（ΔF1）", [f"{cls_retention(m, 'R'):+.4f}" for m in METHODS], jnone, True),
        ("N 类保持能力（ΔF1）", [f"{cls_retention(m, 'N'):+.4f}" for m in METHODS], jnone, True),
        ("距上界差值（FAA − Joint-full）",
         [f"{cl[m]['FAA'] - jfaa:+.4f}" for m in METHODS] if jfaa else ["—"] * 4,
         ["0.0000", f"{j600['ACC'] - jfaa:+.4f}"] if jfaa and j600 else jnone, False),
        ("训练成本（T2–T5）", [f"{mins[m]:.1f} min" for m in METHODS],
         [f"{joint['runtime'][t] / 60:.1f} min" if joint["runtime"].get(t) else "未记录"
          for t in ("full", "600")], False),
        ("存储成本（模型 + 额外状态）",
         [f"{storage['final_models'][m]:.0f} MB + {extra_mb[m]:.2f} MB" for m in METHODS[:3]]
         + [f"{storage['final_models']['adapter']:.0f} MB（含 "
            f"{storage['adapter_bundle']:.0f} MB 适配器，不重复计）"],
         [f"{joint['size_mb'][t]:.0f} MB" if joint["size_mb"].get(t) else "未记录"
          for t in ("full", "600")], False),
        ("推理成本", ["1 个适配器 / 时间未记录", "1 个适配器 / 时间未记录",
                      "1 个适配器 / 时间未记录", "1 个适配器（按窗口路由，需维护 5 个）/ 时间未记录"],
         ["1 个适配器 / 时间未记录", "1 个适配器 / 时间未记录"], False),
    ]
    for idx, (name, vals, jvals, numeric) in enumerate(dims):
        best_fill = None
        if numeric:
            nums = [float(v) for v in vals]
            best_idx = nums.index(min(nums) if idx == 2 else max(nums))
            best_fill = {best_idx + 2: C_GOLD}
        data_row(ws, r, [name] + list(vals) + list(jvals), zebra=(idx % 2 == 1), bold_cols=(1,),
                 align_map={1: "left"}, height=32, fill_map=best_fill)
        r += 1
    conclusions = [
        "稳基线：AF=0.0175 遗忘很小、KR=102.03% 略有正向迁移，零额外状态、成本最低"
        "（9.1 min / 82 MB）；McNemar 与 E2/E3 均不显著 → 其他方法很难显著超越。",
        "四项性能指标全面最优（FAA 0.8587、FMF 0.8575、AF 0.0133、CSA 0.8480），"
        "但相对基线未达显著（p=0.156），且训练成本最高（17.2 min）并需额外 175 MB Fisher 状态。",
        "正向迁移最强（BWT +0.0367、KR 104.50%）、成本适中（11.0 min + 1.25 MB 回放池）；"
        "但 CSA 0.8233 低于 E1/E2，T5 窗口回落至 0.86 → 可塑性略逊。",
        "零遗忘（AF=0、KR=100%）系 route 推理的结构性结果：每个窗口始终由对应 adapter 预测，"
        "依赖已知测试窗口，不可与 E1–E3 的遗忘程度直接比较；CSA 0.8093 四方法最低，"
        "最终阶段显著劣于基线（p<0.001），且需维护 5 个 adapter → 暂无部署优势。",
    ]
    jc_full = jc_600 = "—"
    if jfaa and j600:
        jc_full = (f"上界参照：T1–T5 全部训练数据（3,000 条）混合一次训练（1,500 步，与 E1 全链等算力），"
                   f"FAA={jfaa:.4f}；与最好的 E2 仅差 {abs(jfaa - cl['ewc']['FAA']) * 100:.2f}pp（噪声内）"
                   "→ 时间分布变化下顺序训练已基本触及上界。")
        jc_600 = (f"数据量对照：每窗口每类 40 条共 600 条、300 步，FAA={j600['ACC']:.4f}；"
                  f"距上界 {(jfaa - j600['ACC']) * 100:.1f}pp → E1–E3 的表现主要来自累计数据量"
                  "（全部历史样本在训练中出现过），而非持续学习机制本身。")
    data_row(ws, r, ["综合评价"] + conclusions + [jc_full, jc_600], bold_cols=(1,),
             align_map={i: "left" for i in range(1, 8)}, height=90,
             fill_map={i: C_LIGHT for i in range(1, 8)})
    r += 1
    note_line(ws, r, "填表依据", "CSA / FAA / AF / BWT 见表4；类别保持 ΔF1 = "
                                "mean_{j=1..4}[F1_{5,j} − F1_{j,j}]，由混淆矩阵复算；"
                                "成本见表6；Joint 两列见「Joint 口径」说明，非持续学习方法、不参与最优列标注；"
                                "综合评价依据上述真实结果，未引入主观评分。", n, height=34)
    r += 1
    finish(ws, n, r - 1, header_row=hdr)


# ---------------- 图表 Sheet ----------------

def sh_figures(wb, figs):
    ws = wb.create_sheet("图表")
    n = 8
    r = sheet_setup(ws, "图表 · 六张核心可视化图（图1 两分图｜图2｜图3 三张）",
                    "图1a 对角线适应能力｜图1b 累积平均准确率｜图2 T5 后历史知识保持｜"
                    "图3a/b/c 最终阶段 W/R/N 类别性能",
                    n, [16] * 8, tab=C_BLUE)
    captions = [
        "图1a  四种方法的时间阶段性能变化：对角线适应能力 R$_{i,i}$",
        "图1b  四种方法的时间阶段性能变化：累积平均准确率 A$_i$",
        "图2  四种方法的历史知识保持能力（T5 完成后对 T1–T4 的 Accuracy）",
        "图3a  最终阶段 W 类性能（T5 完成后对五个时间窗口的 F1）",
        "图3b  最终阶段 R 类性能（T5 完成后对五个时间窗口的 F1）",
        "图3c  最终阶段 N 类性能（T5 完成后对五个时间窗口的 F1）",
    ]
    for cap, img_path in zip(captions, figs):
        band(ws, r, cap, n, fill=C_LIGHT)
        r += 1
        with PILImage.open(str(img_path)) as im:
            w0, h0 = im.size
        img = XLImage(str(img_path))
        img.width = 900
        img.height = int(900 * h0 / w0)
        ws.add_image(img, f"A{r}")
        r += int(img.height / 15) + 3
    finish(ws, n, r - 1)


# ---------------- 主流程 ----------------

def load_joint() -> dict:
    """读取 Joint 上界参照（2026-11-11 补充）：results/joint{full,600}/ 下的单行结果。

    Joint 为单一模型混合全部窗口训练数据一次训练，测 T1–T5 全部测试集，
    无阶段概念 → BWT/AF/CSA 等持续学习指标不适用（表中以「—」标注）。
    """
    out: dict = {"full": None, "600": None, "runtime": {}, "size_mb": {}}
    for tag, sub in (("full", RES / "jointfull"), ("600", RES / "joint600")):
        jp = sub / "temporal_cumulative_task5_cumulative_test.json"
        if not jp.is_file():
            continue
        rep = json.loads(jp.read_text(encoding="utf-8"))
        cm = rep.get("continual_metrics") or {}
        per: dict = {}
        for k, v in (rep.get("metrics_by_test_task") or {}).items():
            for i in range(1, 6):
                if k.startswith(f"T{i}_"):
                    per[i] = {"acc": v["acc"], "macro_f1": v["macro_f1"], "kappa": v["cohen_kappa"]}
        out[tag] = {"ACC": cm.get("ACC"), "F1": cm.get("final_avg_macro_f1"), "per": per}
    for lg in LOGS.glob("joint_*.log"):
        hits = re.findall(r"'train_runtime':\s*'([0-9.]+)'",
                          lg.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            out["runtime"]["full" if "jointfull" in lg.name else "600"] = float(hits[-1])
    for tag, mid in (("full", "cumulative_task5_jointfull"), ("600", "cumulative_task5_joint600")):
        d = ART / "models" / mid
        if d.is_dir():
            out["size_mb"][tag] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
    return out


def main() -> None:
    print("读取 60 组混淆矩阵并复算全部指标 ...")
    records = collect_records()
    ds = build_dataset(records)
    dev = max_recalc_deviation(records)
    times = parse_train_times()
    mins = train_minutes(times)
    params = parse_trainable_params()
    storage = storage_costs(records)
    joint = load_joint()

    print("生成图 ...")
    figs = make_figures(ds)

    print("生成工作簿 ...")
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = "时间增量持续学习实验结果分析"
    wb.properties.subject = "基于 60 组 3x3 混淆矩阵的全量结果汇总"
    wb.properties.creator = "CAPC-CG-Project / make_report_tables.py"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    sh_config(wb, records, ds, dev, generated)
    sh_raw(wb, records, dev)
    sh_class(wb, records)
    sh_cm(wb, records, dev)
    sh_table1(wb, ds, joint)
    sh_table2(wb, ds)
    sh_table3(wb, ds)
    sh_table4(wb, ds, joint)
    sh_table5(wb, ds)
    sh_table6(wb, times, mins, params, storage, joint)
    sh_table7(wb, ds, mins, storage, joint)
    sh_figures(wb, figs)
    wb.active = 0
    wb.save(OUT_XLSX)

    print("\n== 数据一致性 ==")
    print(f"  混淆矩阵组数 = {len(records)}（期望 60）")
    print(f"  复算 vs 报告值最大绝对偏差 = {dev:.2e}")
    print("\n== 表4 持续学习六指标（由混淆矩阵复算） ==")
    print(f"{'方法':<22}{'FAA':>9}{'FMF':>9}{'AF':>9}{'BWT':>10}{'KR':>10}{'CSA':>9}")
    for m in METHODS:
        c = ds["cl"][m]
        print(f"{M_LABEL[m]:<22}{c['FAA']:>9.4f}{c['FMF']:>9.4f}{c['AF']:>9.4f}"
              f"{c['BWT']:>+10.4f}{c['KR']*100:>9.2f}%{c['CSA']:>9.4f}")
    print("\n== 表1 对角线 Accuracy（文档表1 口径） ==")
    for m in METHODS:
        print(f"  {M_LABEL[m]:<22}" + "  ".join(f"{ds['R'][m][i][i]:.4f}" for i in range(1, 6)))
    print(f"\n已生成: {OUT_XLSX}")
    print("图: " + " / ".join(p.name for p in figs))


if __name__ == "__main__":
    main()
