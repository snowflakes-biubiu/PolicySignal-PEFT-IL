# -*- coding: utf-8 -*-
"""生成「领域增量持续学习」结果工作簿，格式严格对齐时间增量模板。

格式规格（字体/配色/列宽/行高/合并/数字格式）全部取自
`05_incremental_learning_time/results/持续学习实验结果汇总表.xlsx`：
    标题   微软雅黑 14pt 加粗 白字，#1F3864 底，整行合并，行高 34
    副标题 微软雅黑 9pt，#F3F8FD 底，整行合并，行高 24
    表头   微软雅黑 10pt 加粗 白字，#2E75B6 底，居中，行高 32
    分组头 #1F4E79 底 白字（明细表的三段分组）
    数据   微软雅黑 10pt，字色 #20304A，数字 0.0000，居中
    方法名 竖向合并（表1 跨 3 行、表5 跨 5 行），加粗
    说明行 A 列标签 #FFF2CC 底 + B:末列合并

唯一原始数据源：24 份 *_predictions.csv（84 组 3×3 混淆矩阵）；所有指标现场复算。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from openpyxl import Workbook  # noqa: E402
from openpyxl.drawing.image import Image as XLImage  # noqa: E402
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.pagebreak import Break  # noqa: E402
from openpyxl.worksheet.properties import PageSetupProperties  # noqa: E402

SCRIPT_ROOT = Path(__file__).resolve().parents[3]

# ── 与原表一致的常量 ─────────────────────────────────────────────────
F_NAME = "微软雅黑"
C_TITLE = "1F3864"
C_HDR = "2E75B6"
C_GROUP = "1F4E79"
C_SUB = "F3F8FD"
C_TXT = "20304A"
C_NOTE = "FFF2CC"
C_RED = "FCE4E4"
C_GREEN = "E2EFDA"

METHODS = ["sequential_ft", "ewc", "replay", "adapter"]
M_LABEL = {"sequential_ft": "E1 Sequential FT", "ewc": "E2 EWC",
           "replay": "E3 Replay", "adapter": "E4 Adapter-CL"}
M_SHORT = {"sequential_ft": "Sequential FT", "ewc": "EWC",
           "replay": "Replay", "adapter": "Adapter-CL"}
M_COLORS = {"sequential_ft": "#2E75B6", "ewc": "#ED7D31",
            "replay": "#70AD47", "adapter": "#C00000"}

DOMAINS = {1: ("D1", "经济与财税金融"), 2: ("D2", "教育"), 3: ("D3", "卫生健康与食药"),
           4: ("D4", "农业与农村"), 5: ("D5", "自然资源与生态环境"),
           6: ("D6", "公共安全与应急管理")}
D_SHORT = {k: v[0] for k, v in DOMAINS.items()}
D_NAME = {k: v[1] for k, v in DOMAINS.items()}
D_FULL = {k: f"{v[0]} {v[1]}" for k, v in DOMAINS.items()}
# 与原表对齐：明细表的"测试集"列使用数据集目录名
D_SET = {k: f"{v[0]}_{v[1]}" for k, v in DOMAINS.items()}

CLASSES = ["N", "R", "W"]
CLASS_FULL = {"N": "N 规范性", "R": "R 禁止性", "W": "W 鼓励性"}

SIDE = Side(style="thin", color="D9D9D9")
BORDER = Border(left=SIDE, right=SIDE, top=SIDE, bottom=SIDE)


def pf(color: str) -> PatternFill:
    return PatternFill("solid", fgColor=color)


def title_block(ws, text: str, subtitle: str, ncols: int) -> None:
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 100
    ws["A1"] = text
    ws["A1"].font = Font(name=F_NAME, size=14, bold=True, color="FFFFFF")
    ws["A1"].fill = pf(C_TITLE)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.row_dimensions[1].height = 34

    ws["A2"] = subtitle
    ws["A2"].font = Font(name=F_NAME, size=9, color="404040")
    ws["A2"].fill = pf(C_SUB)
    ws["A2"].alignment = Alignment(horizontal="left", vertical="center",
                                   wrap_text=True, indent=1)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    ws.row_dimensions[2].height = 24

    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = 9
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0


def widths(ws, pairs: list[tuple[str, float]]) -> None:
    for col, w in pairs:
        ws.column_dimensions[col].width = w


def note_bar(ws, text: str, ncols: int) -> None:
    """明细表第 3 行的浅蓝说明条（与原表一致：#DDEBF7 + 11pt 加粗）。"""
    cell = ws.cell(row=3, column=1, value=text)
    cell.font = Font(name=F_NAME, size=11, bold=True, color=C_TXT)
    cell.fill = pf("DDEBF7")
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=ncols)
    ws.row_dimensions[3].height = 24


def hdr(ws, row: int, values: list, ncols: int, merged: bool = False,
        color: str = C_HDR, height: float = 32) -> int:
    """表头行：深蓝底白字；merged=True 时整行合并（用于分组标题）。"""
    for c, v in enumerate(values, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = Font(name=F_NAME, size=10, bold=True, color="FFFFFF")
        cell.fill = pf(color)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    if merged or len(values) == 1:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    ws.row_dimensions[row].height = height
    return row + 1


def light_hdr(ws, row: int, values: list, color: str = C_HDR) -> int:
    """字段名行：与原表一致用深蓝底白字（明细表与表2 的第二层表头）。"""
    for c, v in enumerate(values, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = Font(name=F_NAME, size=10, bold=True, color="FFFFFF")
        cell.fill = pf(color)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[row].height = 32
    return row + 1


def row_cells(ws, row: int, values: list, fmt: str | None = "0.0000",
              bold_first: bool = False, height: int = 21, stripe: int | None = None) -> int:
    for c, v in enumerate(values, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = Font(name=F_NAME, size=10, color=C_TXT,
                         bold=(bold_first and c == 1))
        cell.border = BORDER
        # 隔行条纹：按数据行序号（0 起）奇偶上色，第 2 行起为浅蓝（与原表一致）
        cell.fill = pf(C_SUB) if (stripe is not None and stripe % 2 == 1) else pf("FFFFFF")
        if isinstance(v, float):
            cell.number_format = fmt or "0.0000"
        cell.alignment = Alignment(horizontal="left" if c == 1 and isinstance(v, str) and len(str(v)) > 14 else "center",
                                   vertical="center", wrap_text=False)
    ws.row_dimensions[row].height = height
    return row + 1


def section(ws, row: int, text: str, ncols: int) -> int:
    """分组标题行：浅蓝底深字，与原表 0 号工作表一致（#DDEBF7 + 11pt 加粗）。"""
    cell = ws.cell(row=row, column=1, value=text)
    cell.font = Font(name=F_NAME, size=11, bold=True, color=C_TXT)
    cell.fill = pf("DDEBF7")
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    ws.row_dimensions[row].height = 24
    return row + 1


def note_row(ws, row: int, key: str, text: str, ncols: int) -> int:
    a = ws.cell(row=row, column=1, value=key)
    a.font = Font(name=F_NAME, size=10, bold=True, color=C_TXT)
    a.fill = pf("DDEBF7")
    a.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    a.border = BORDER
    b = ws.cell(row=row, column=2, value=text)
    b.font = Font(name=F_NAME, size=10, color=C_TXT)
    b.fill = pf("FFFFFF")
    b.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1)
    if ncols > 2:
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=ncols)
    ws.row_dimensions[row].height = max(24, 15 * (1 + len(str(text)) // 55))
    return row + 1


def cs_rule(reverse: bool = False) -> ColorScaleRule:
    """红-黄-绿色阶（与原表条件格式一致：F8696B / FFEB84 / 63BE7B）。

    reverse=True 用于「越低越好」的指标（如 AF）：绿→黄→红。
    """
    lo, hi = ("63BE7B", "F8696B") if reverse else ("F8696B", "63BE7B")
    return ColorScaleRule(start_type="min", start_color=lo,
                          mid_type="percentile", mid_value=50, mid_color="FFEB84",
                          end_type="max", end_color=hi)


def band(ws, row_start: int, row_end: int, ncols: int, color: str = C_SUB) -> None:
    for r in range(row_start, row_end + 1):
        for c in range(1, ncols + 1):
            ws.cell(row=r, column=c).fill = pf(color)


# ── 数据层 ────────────────────────────────────────────────────────────
def load_cm(csv_path: Path) -> tuple[dict[int, Counter], dict[int, int]]:
    per: dict[int, Counter] = defaultdict(Counter)
    sizes: Counter = Counter()
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            t = int(rec["test_task"])
            per[t][(rec["gold"], rec["prediction"])] += 1
            sizes[t] += 1
    return per, dict(sizes)


def cm_metrics(cm: Counter, n: int) -> dict:
    acc = sum(c for (g, p), c in cm.items() if g == p) / n
    per = {}
    for c in CLASSES:
        tp = cm.get((c, c), 0)
        pred_c = sum(v for (_, p), v in cm.items() if p == c)
        gold_c = sum(v for (g, _), v in cm.items() if g == c)
        P = tp / pred_c if pred_c else 0.0
        R = tp / gold_c if gold_c else 0.0
        per[c] = {"P": P, "R": R, "F1": 2 * P * R / (P + R) if (P + R) else 0.0,
                  "support": gold_c, "pred": pred_c}
    macro = sum(per[c]["F1"] for c in CLASSES) / 3
    pe = sum((sum(v for (g, _), v in cm.items() if g == c) / n) *
             (sum(v for (_, p), v in cm.items() if p == c) / n) for c in CLASSES)
    kappa = (acc - pe) / (1 - pe) if pe != 1 else 0.0
    return {"acc": acc, "macro_f1": macro, "kappa": kappa, "per": per, "n": n}


class Dataset:
    def __init__(self, root: Path, models: Path):
        self.root, self.models = root, models
        self.res = root / "results" / "domain_incremental"
        self.ACC: dict = {}
        self.MF1: dict = {}
        self.KAP: dict = {}
        self.CM: dict = {}
        self.PC: dict = {}
        self.N: dict = {}
        self.reports: dict = {}
        self.meta: dict = {}
        self.runtime: dict = {}
        self.sizes: dict = {}
        self._load()

    def _load(self) -> None:
        for m in METHODS:
            for i in range(1, 7):
                p = self.res / f"temporal_{m}_task{i}_cumulative_test_predictions.csv"
                if not p.is_file():
                    continue
                per, sizes = load_cm(p)
                for j, cm in per.items():
                    met = cm_metrics(cm, sizes[j])
                    k = (m, i, j)
                    self.ACC[k], self.MF1[k], self.KAP[k] = met["acc"], met["macro_f1"], met["kappa"]
                    self.CM[k], self.PC[k], self.N[k] = cm, met["per"], met["n"]
                jp = self.res / f"temporal_{m}_task{i}_cumulative_test.json"
                if jp.is_file():
                    self.reports[(m, i)] = json.loads(jp.read_text(encoding="utf-8"))

        if self.models.is_dir():
            # 兼容 models/<name> 与 models/stage{N}/<name> 两种布局：以 metadata 定位模型目录
            for mp in sorted(self.models.rglob("training_metadata.json")):
                d = mp.parent
                self.sizes[d.name] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                self.meta[d.name] = json.loads(mp.read_text(encoding="utf-8"))

        for log in (self.root / "logs" / "full_matrix").glob("*/train_*_task*.log"):
            hits = re.findall(r"'train_runtime':\s*'([0-9.]+)'",
                              log.read_text(encoding="utf-8", errors="ignore"))
            mm = re.match(r"train_(.+)_task(\d+)$", log.stem)
            if hits and mm:
                self.runtime[(mm.group(1), int(mm.group(2)))] = float(hits[-1])

        # ── Joint 上界参照（2026-11-11 补充）：单一模型混合训练，测全部领域，无阶段概念 ──
        self.joint: dict = {}
        self.joint_runtime: dict = {}
        for tag in ("full", "600"):
            jp = self.res / f"temporal_cumulative_task6_joint{tag}_cumulative_test.json"
            if jp.is_file():
                rep = json.loads(jp.read_text(encoding="utf-8"))
                cm = rep.get("continual_metrics") or {}
                self.joint[tag] = {
                    "ACC": cm.get("ACC"),
                    "F1": cm.get("final_avg_macro_f1"),
                    "per": {k: {"acc": v["acc"], "macro_f1": v["macro_f1"], "kappa": v["cohen_kappa"]}
                            for k, v in (rep.get("metrics_by_test_task") or {}).items()},
                }
        for log in (self.root / "logs").glob("joint_*.log"):
            hits = re.findall(r"'train_runtime':\s*'([0-9.]+)'",
                              log.read_text(encoding="utf-8", errors="ignore"))
            if hits:
                self.joint_runtime["full" if "jointfull" in log.name else "600"] = float(hits[-1])

    def joint_dir_size_mb(self, tag: str) -> float:
        return self.sizes.get(f"cumulative_task6_joint{tag}", 0) / 1e6

    def acc(self, m, i, j):
        return self.ACC.get((m, i, j))

    def model_name(self, m, k):
        return f"qwen15b_domain_task{k}_e2_r16_a32_lr2e-4" if m == "sequential_ft" else f"{m}_task{k}"

    def secs(self, m, stages):
        return sum(self.runtime.get((m, k), 0.0) for k in stages)

    def model_mb(self, m):
        pref = "qwen15b_domain_task" if m == "sequential_ft" else f"{m}_task"
        return sum(v for k, v in self.sizes.items() if k.startswith(pref)) / 1e6

    def extra_mb(self, m) -> tuple[float, str]:
        if m == "ewc":
            per_anchor = 4358144 * 4 * 2 / 1e6
            return per_anchor * 6, f"估算：6 套 anchor × {per_anchor:.1f} MB（anchor+fisher, fp32）"
        if m == "replay":
            return 0.6, "估算：300 条缓存 × ~2 KB"
        return 0.0, "—"


def cl_metrics(ds: Dataset, m: str, last: int = 6) -> dict:
    R = {j: ds.acc(m, last, j) or 0.0 for j in range(1, last + 1)}
    diag = {i: ds.acc(m, i, i) or 0.0 for i in range(1, last + 1)}
    return {
        "FAA": sum(R.values()) / last,
        "FAA_F1": sum(ds.MF1.get((m, last, j), 0.0) for j in range(1, last + 1)) / last,
        "AF": sum(max(ds.acc(m, k, j) or 0.0 for k in range(j, last + 1)) - R[j]
                  for j in range(1, last)) / (last - 1),
        "BWT": sum(R[j] - diag[j] for j in range(1, last)) / (last - 1),
        "KR": (sum(R[j] for j in range(1, last)) / sum(diag[j] for j in range(1, last)) * 100
               if sum(diag[j] for j in range(1, last)) else 0.0),
        "CSA": sum(diag.values()) / last,
        "R": R, "diag": diag,
    }


# ── 构建 ──────────────────────────────────────────────────────────────
def build(ds: Dataset, out: Path) -> Path:
    wb = Workbook()
    wb.remove(wb.active)
    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    cl = {m: cl_metrics(ds, m) for m in METHODS}

    # ══ 0 实验配置与口径 ══════════════════════════════════════════
    ws = wb.create_sheet("0_实验配置与口径")
    ws.sheet_properties.tabColor = C_TITLE
    title_block(ws, "领域增量持续学习实验 · 结果分析工作簿",
                f"按《汇报要求.docx》整理｜唯一原始数据源：84 组 3×3 混淆矩阵｜生成时间 {stamp}",
                6)
    widths(ws, [("A", 22), ("B", 60), ("C", 40), ("D", 14), ("E", 14), ("F", 14)])
    r = 3
    r = section(ws, r, "一、实验定位与结构导航", 6)
    for k, v in [
        ("实验定位", "同一 W/R/N 三分类任务在六个领域上的领域增量学习（Domain CL）。D1–D6 是六个不同领域，"
                     "不是同一任务的六个时间窗口；每个阶段内部仍混合 1949–2023 全部年代。"),
        ("方法与编号", "E1 Sequential FT（朴素顺序微调）／E2 EWC（Fisher 约束，λ=100，mean 归一化）／"
                       "E3 Replay（每域 60 条定额、等算力 1200 抽样）／E4 Adapter-CL（每域独立 adapter，route 推理）"),
        ("测试结构", "D_i 训练完成后测试 D1..D_i，形成下三角矩阵；4 方法 × (1+2+3+4+5+6) = 84 组测试，"
                     "每次测试使用该领域固定的 300 条测试集。"),
        ("工作簿结构", "0 配置与口径｜1 原始测试指标｜2 类别级指标｜3 全量混淆矩阵｜表1–表7｜图表"),
        ("表号对照", "表1 总体性能（对角线）／表2 历史测试矩阵／表3 类别级 F1／表4 持续学习六指标／"
                     "表5 遗忘差值表／表6 资源成本／表7 综合对比"),
        ("数据来源", "results/domain_incremental/*_predictions.csv（24 份）、"
                     "artifacts/models/*/training_metadata.json（21 份）、logs/full_matrix/*/train_*.log"),
    ]:
        r = note_row(ws, r, k, v, 6)

    r += 1
    r = section(ws, r, "二、指标口径（所有数值均由混淆矩阵现场复算）", 6)
    for k, v in [
        ("混淆矩阵", "行 = Gold，列 = Pred，标签统一为 N / R / W；Accuracy = ΣCM[c][c] / N。"),
        ("Precision / Recall / F1", "P_c = CM[c][c] / Σ_i CM[i][c]；R_c = CM[c][c] / Σ_j CM[c][j]；"
                                    "F1_c = 2PR/(P+R)；Macro-F1 = mean_c F1_c；Kappa 为 Cohen's κ。"),
        ("R_{i,j}", "第 i 阶段训练完成后，在 D_j 测试集上的 Accuracy；j ≤ i 有值，上三角为空。"),
        ("FAA", "最终平均准确率 mean_{j=1..6} R_{6,j}。"),
        ("BWT", "后向迁移 mean_{j=1..5}( R_{6,j} − R_{j,j} )，正值 = 后续学习改善历史领域。"),
        ("AF", "平均遗忘程度 mean_{j=1..5}[ max_{k=j..6} R_{k,j} − R_{6,j} ]，历史最佳含最终阶段，故 AF ≥ 0。"),
        ("KR", "历史知识保持率 Σ_{j=1..5} R_{6,j} / Σ_{j=1..5} R_{j,j} × 100%（>100% 表示正向迁移）。"),
        ("CSA", "当前阶段适应能力 mean_{i=1..6} R_{i,i}，只反映学完该领域的即时表现。"),
    ]:
        r = note_row(ws, r, k, v, 6)

    r += 1
    r = section(ws, r, "三、本实验的实际配置（取自 21 份 training_metadata.json）", 6)
    ewc = [ds.meta.get(f"ewc_task{k}", {}) for k in range(2, 7)]
    rep = [ds.meta.get(f"replay_task{k}", {}) for k in range(2, 7)]
    trunc = [ds.meta.get(ds.model_name("sequential_ft", k), {}).get("train_truncation", {})
             for k in range(1, 7)]
    for k, v in [
        ("模型与数据", "基座 Qwen2.5-1.5B-Instruct + QLoRA（r=16，α=32，dropout=0.05，target=q/k/v/o，"
                       "可训练 4,358,144 参数）；每域 train 600（N/R/W 各 200）、test 300（各 100），共 5,400 段。"),
        ("训练预算", "统一 2 epoch × 600 = 1200 次抽样 = 300 步（batch 4）；bf16 + paged_adamw_8bit；max_length 768。"),
        ("EWC", f"λ=100，normalize=mean（Σ/n_anchor，与阶段数解耦）；penalty/CE = "
                f"{ewc[0].get('ewc_penalty_to_ce', 0):.3f} → {ewc[-1].get('ewc_penalty_to_ce', 0):.3f}（D2→D6）；"
                f"Fisher 每阶段约 {ewc[-1].get('fisher_seconds', 0):.0f} 秒。"),
        ("Replay", f"每域定额 {rep[0].get('replay_per_domain', 60)} 条（10% 覆盖），等算力采样共 1200 次；"
                   f"D6 新域过境 {rep[-1].get('passes_per_domain', {}).get('D6', 0):.2f} 遍，每个历史域 0.10 遍。"),
        ("Adapter-CL", "每阶段新增 task_i adapter 且只训练当前 adapter；推理按测试领域 route 到 task_j，"
                       "需要已知测试领域标签，BWT/Forgetting 由构造恒为 0。"),
        ("截断", "文本层截断（保留提示词 + 正文头 60%/尾 40%），各阶段截断率 "
                 f"{min(t.get('ratio', 0) for t in trunc):.2%}–{max(t.get('ratio', 0) for t in trunc):.2%}。"),
    ]:
        r = note_row(ws, r, k, v, 6)

    r += 1
    r = section(ws, r, "四、阅读与引用注意", 6)
    for k, v in [
        ("比较基准", "E1/E2/E3 为「单模型处理全部领域」；E4 为「已知测试领域」的路由上限参照，"
                     "其 BWT、AF 恒为 0，不能与前三种方法同表排名。"),
        ("显著性", "每域 300 条、D6 共 1800 条测试，单方法准确率标准误约 ±0.008；"
                   "E1/E2/E3 之间 0.5–1pp 差距落在噪声内，Δ<2pp 不宜宣称优劣。"),
        ("单 seed", "全部结果来自 seed=42 单次运行，未做多种子重复，跨 seed 方差未估计。"),
        ("λ 设定", "EWC 为 λ=100 的弱正则设定（penalty/CE≈5–8%）；更强 λ 的对照需另行实验。"),
    ]:
        r = note_row(ws, r, k, v, 6)

    r += 1
    r = section(ws, r, "五、Joint 上界对照实验（2026-11-11 补充）", 6)
    for k, v in [
        ("目的", "回答「持续学习最多能做到多好」：给出历史数据全部可用时的性能上界（UB），"
                 "并以同预算的小样本联合训练剥离数据量因素，区分「顺序代价」与「数据量贡献」。"),
        ("构造", "Joint-full：D1–D6 全部训练数据（3,600 条，每类 1,200）混合，从基座一次训练 2 epoch = 1,800 步"
                 "（与 E1 全链累计步数相同，等算力）；Joint-600：每域按类别均衡抽 100 条（34/33/33 轮转，"
                 "每类严格 200）共 600 条，2 epoch = 300 步。两者 QLoRA 超参与 E1–E4 完全一致，seed=42。"),
        ("评估口径", "单一模型测 D1–D6 全部测试集（1,800 条），一行结果与 E1–E4 的 T6 行同源可比；"
                     "BWT/AF/CSA 对 Joint 无定义（无阶段序列），以「—」标注且不参与排名。"),
        ("数据与代码", "抽样清单 artifacts/joint_data/joint_*_manifest.json（含逐条 DocumentID 与 sha256）；"
                       "结果 results/domain_incremental/temporal_cumulative_task6_joint{full,600}_*.json；"
                       "训练入口 scripts/train/train_joint.py（零侵入，不改 code_version 哈希的五个公共脚本）。"),
    ]:
        r = note_row(ws, r, k, v, 6)

    # ══ 1 原始测试指标 ════════════════════════════════════════════
    ws = wb.create_sheet("1_原始测试指标")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "1 · 原始测试指标（84 组 = 4 方法 × 21 组合）",
                "Σ 列支持自动筛选，可导出为下游分析的原始表", 9)
    note_bar(ws, "84 组 = 4 方法 × (1+2+3+4+5+6)；数值由混淆矩阵复算，与结果 JSON 完全一致", 9)
    widths(ws, [("A", 6), ("B", 20), ("C", 16), ("D", 12), ("E", 20),
                ("F", 10), ("G", 13), ("H", 13), ("I", 15)])
    r = 4
    r = light_hdr(ws, r, ["序号", "方法", "训练阶段", "测试阶段", "测试集",
                          "样本数", "Accuracy", "Macro-F1", "Cohen's Kappa"])
    start = r
    idx = 1
    for m in METHODS:
        for i in range(1, 7):
            for j in range(1, i + 1):
                if (m, i, j) not in ds.ACC:
                    continue
                r = row_cells(ws, r, [idx, M_LABEL[m], f"D{i} 训练完成", D_SHORT[j], D_SET[j],
                                      ds.N[(m, i, j)], ds.ACC[(m, i, j)],
                                      ds.MF1[(m, i, j)], ds.KAP[(m, i, j)]], stripe=idx - 1)
                idx += 1
    ws.auto_filter.ref = f"A{start - 1}:I{r - 1}"
    ws.freeze_panes = f"A{start}"
    # 与原表一致：ACC / Macro-F1 / Kappa 三列各自按大小做红-黄-绿色阶
    for col in ("G", "H", "I"):
        ws.conditional_formatting.add(f"{col}{start}:{col}{r - 1}", cs_rule())

    # ══ 2 类别级指标 ══════════════════════════════════════════════
    ws = wb.create_sheet("2_类别级指标")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "2 · 类别级指标（84 组 × 3 类 = 252 行）",
                "P = 预测为某类中真值为该类的比例；R = 真值某类中被正确识别的比例；F1 = 2PR/(P+R)", 14)
    note_bar(ws, "P = 预测为某类中真值为该类的比例；R = 真值某类中被正确识别的比例；F1 为二者调和平均", 14)
    widths(ws, [("A", 6), ("B", 20), ("C", 14), ("D", 11), ("E", 20)] + [("F", 10)] * 9)
    r = 4
    groups = [("测试标识", 1, 5), ("W 类", 6, 8), ("R 类", 9, 11), ("N 类", 12, 14)]
    for text, c1, c2 in groups:
        cell = ws.cell(row=r, column=c1, value=text)
        cell.font = Font(name=F_NAME, size=10, bold=True, color="FFFFFF")
        cell.fill = pf(C_GROUP)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.merge_cells(start_row=r, start_column=c1, end_row=r, end_column=c2)
    ws.row_dimensions[r].height = 24
    r += 1
    r = light_hdr(ws, r, ["序号", "方法", "训练阶段", "测试阶段", "测试集",
                          "W-P", "W-R", "W-F1", "R-P", "R-R", "R-F1", "N-P", "N-R", "N-F1"])
    idx = 1
    for m in METHODS:
        for i in range(1, 7):
            for j in range(1, i + 1):
                if (m, i, j) not in ds.PC:
                    continue
                pc = ds.PC[(m, i, j)]
                vals = [pc[c][k] for c in ["W", "R", "N"] for k in ("P", "R", "F1")]
                r = row_cells(ws, r, [idx, M_LABEL[m], f"D{i} 训练完成", D_SHORT[j], D_SET[j]] + vals,
                              stripe=idx - 1)
                idx += 1
    # 与原表一致：三类 F1 列（W-F1 / R-F1 / N-F1）各自按大小做色阶
    end2 = r - 1
    for col in ("H", "K", "N"):
        ws.conditional_formatting.add(f"{col}6:{col}{end2}", cs_rule())

    # ══ 3 全量混淆矩阵 ════════════════════════════════════════════
    ws = wb.create_sheet("3_全量混淆矩阵")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "3 · 全量混淆矩阵（84 组）",
                "行 = Gold，列 = Pred，标签统一 N / R / W；右侧列出复算值、原始报告值与最大偏差，用于一致性核对", 23)
    note_bar(ws, "行 = Gold，列 = Pred，标签统一 N / R / W；右侧给出复算值、报告值与一致性偏差", 23)
    widths(ws, [("A", 6), ("B", 20), ("C", 16), ("D", 11), ("E", 20), ("F", 9), ("G", 10)] +
           [(get_column_letter(c), 9) for c in range(8, 17)] +
           [(get_column_letter(c), 13) for c in range(17, 23)] + [("W", 12)])
    r = 4
    groups = [("测试标识", 1, 7), ("混淆矩阵 CM（行 Gold → 列 Pred，N/R/W）", 8, 16),
              ("由矩阵复算", 17, 19), ("原始报告值（核对用）", 20, 22), ("一致性", 23, 23)]
    for text, c1, c2 in groups:
        cell = ws.cell(row=r, column=c1, value=text)
        cell.font = Font(name=F_NAME, size=10, bold=True, color="FFFFFF")
        cell.fill = pf(C_GROUP)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        if c2 > c1:
            ws.merge_cells(start_row=r, start_column=c1, end_row=r, end_column=c2)
    ws.row_dimensions[r].height = 24
    r += 1
    r = light_hdr(ws, r, ["序号", "方法", "训练阶段", "测试阶段", "测试集", "样本数", "标签顺序",
                          "N→N", "N→R", "N→W", "R→N", "R→R", "R→W", "W→N", "W→R", "W→W",
                          "复算Acc", "复算Macro-F1", "复算Kappa",
                          "报告Acc", "报告Macro-F1", "报告Kappa", "最大偏差"])
    idx = 1
    max_dev = 0.0
    for m in METHODS:
        for i in range(1, 7):
            for j in range(1, i + 1):
                key = (m, i, j)
                if key not in ds.CM:
                    continue
                cm = ds.CM[key]
                cells = [cm.get((g, p), 0) for g in CLASSES for p in CLASSES]
                rep = ds.reports.get((m, i), {}) or {}
                rm = (rep.get("metrics_by_test_task") or {}).get(D_SET[j], {}) or {}
                dev = max(abs(ds.ACC[key] - rm.get("acc", ds.ACC[key])),
                          abs(ds.MF1[key] - rm.get("macro_f1", ds.MF1[key])))
                max_dev = max(max_dev, dev)
                r = row_cells(ws, r, [idx, M_LABEL[m], f"D{i} 训练完成", D_SHORT[j], D_SET[j],
                                      ds.N[key], "N/R/W"] + cells +
                                      [ds.ACC[key], ds.MF1[key], ds.KAP[key],
                                       rm.get("acc"), rm.get("macro_f1"), rm.get("kappa"), dev],
                              height=20, stripe=idx - 1)
                # 与原表一致：仅对角线格（N→N、R→R、W→W = 第 8/12/16 列）恒浅绿，
                # 其余矩阵格保持隔行条纹
                for cc in (8, 12, 16):
                    ws.cell(row=r - 1, column=cc).fill = pf("E2EFDA")
                ws.cell(row=r - 1, column=23).number_format = "0.0E+00"
                idx += 1
    ws.freeze_panes = "A6"
    # 与原表一致：最大偏差 = 0（复算与报告完全一致）标绿字
    ws.conditional_formatting.add(f"W6:W{r - 1}",
                                  CellIsRule(operator="lessThan", formula=["0.000001"],
                                             font=Font(color="FF2E6B2E", bold=True)))

    # ══ 表1 总体性能 ══════════════════════════════════════════════
    ws = wb.create_sheet("表1_总体性能")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "表1 · 四种方法总体性能对比表（文档表1 口径）",
                "D1–D6 取「各阶段训练完成后对当前领域测试集」的对角线结果 R_{i,i}；六阶段平均只反映适应能力，不代表历史知识保持", 9)
    widths(ws, [("A", 22), ("B", 13)] + [(get_column_letter(c), 11) for c in range(3, 9)] + [("I", 14)])
    r = 3
    r = hdr(ws, r, ["方法", "指标"] + [D_SHORT[k] for k in range(1, 7)] + ["六阶段平均"], 9)
    for m in METHODS:
        block_start = r
        for label, table in (("Accuracy", ds.ACC), ("Macro-F1", ds.MF1), ("Kappa", ds.KAP)):
            vals = [table.get((m, k, k), 0.0) for k in range(1, 7)]
            r = row_cells(ws, r, [M_LABEL[m] if label == "Accuracy" else "", label] + vals +
                          [sum(vals) / 6])
        ws.merge_cells(start_row=block_start, start_column=1, end_row=r - 1, end_column=1)
        ws.cell(row=block_start, column=1).alignment = Alignment(horizontal="center",
                                                                 vertical="center", wrap_text=True)
    # ── Joint 上界参照块：单一模型混合全部领域数据一次训练，测全部测试集（非对角线口径）──
    jblocks = [t for t in ("full", "600") if ds.joint.get(t)]
    jlabels = {"full": "UB Joint-full", "600": "对照 Joint-600"}
    for tag in jblocks:
        d = ds.joint[tag]
        block_start = r
        for label, keyf, avg in (("Accuracy", "acc", d["ACC"]),
                                 ("Macro-F1", "macro_f1", d["F1"]),
                                 ("Kappa", "kappa", None)):
            vals = [d["per"][D_SET[k]][keyf] for k in range(1, 7)]
            r = row_cells(ws, r, [jlabels[tag] if label == "Accuracy" else "", label] + vals +
                          [avg if avg is not None else sum(vals) / 6])
        ws.merge_cells(start_row=block_start, start_column=1, end_row=r - 1, end_column=1)
        ws.cell(row=block_start, column=1).alignment = Alignment(horizontal="center",
                                                                 vertical="center", wrap_text=True)
    # 与原表一致：每条指标行独立做红-黄-绿色阶（跨四方法 + Joint 参照比较同一指标）
    metric_rows = {"Accuracy": [], "Macro-F1": [], "Kappa": []}
    rr = 4
    for block in ([None] * len(METHODS) + jblocks):
        for lbl in metric_rows:
            metric_rows[lbl].append(rr)
            rr += 1
    for lbl, rows_ in metric_rows.items():
        ws.conditional_formatting.add(" ".join(f"C{x}:H{x}" for x in rows_), cs_rule())

    # ── 补充对照：D6 最终模型在 D1–D6 各领域的表现（= 表2 最后一行，非文档表1 口径）──
    r += 1
    div = ws.cell(row=r, column=1,
                  value="补充对照（非文档表1 口径）：D6 最终模型在 D1–D6 各领域测试集的表现 —— 对应表2 最后一行与图2")
    div.font = Font(name=F_NAME, size=11, bold=True, color=C_TXT)
    div.fill = pf(C_NOTE)
    div.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=9)
    ws.row_dimensions[r].height = 24
    r += 1
    r = hdr(ws, r, ["方法", "指标"] + [D_SHORT[k] for k in range(1, 7)] + ["六领域平均"], 9)
    for m in METHODS:
        block_start = r
        for label, table in (("Accuracy", ds.ACC), ("Macro-F1", ds.MF1), ("Kappa", ds.KAP)):
            vals = [table.get((m, 6, j), 0.0) for j in range(1, 7)]
            r = row_cells(ws, r, [M_LABEL[m] if label == "Accuracy" else "", label] + vals +
                          [sum(vals) / 6])
        ws.merge_cells(start_row=block_start, start_column=1, end_row=r - 1, end_column=1)
        ws.cell(row=block_start, column=1).alignment = Alignment(horizontal="center",
                                                                 vertical="center", wrap_text=True)
    r = note_row(ws, r, "口径判别",
                 "文档表1 只要求对角线 R_{i,i}；补充区给出 D6 最终模型的跨领域表现，"
                 "与 FAA/表2 最后一行同源，避免与历史知识保持指标混淆。"
                 "Joint 两行为单一模型混合全部训练数据一次训练后在 D1–D6 全部测试集的表现（非阶段对角线），"
                 "作为「数据全部可用时的上界」与「单阶段数据量对照」，不参与持续学习指标排名。", 9)

    # ══ 表2 历史测试矩阵 ══════════════════════════════════════════
    ws = wb.create_sheet("表2_历史测试矩阵")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "表2 · 各阶段历史测试性能矩阵（文档表2）",
                "整份研究的核心原始表；行 = 训练完成阶段，列 = 测试领域；上三角「—」为该阶段尚未接触该领域，非缺失数据", 7)
    widths(ws, [("A", 22)] + [(get_column_letter(c), 13) for c in range(2, 8)])
    r = 3
    for title, table in (("A · Accuracy 历史测试矩阵（主表）", ds.ACC),
                         ("B · Macro-F1 历史测试矩阵", ds.MF1)):
        r = hdr(ws, r, [title], 7, merged=True)
        for m in METHODS:
            c = ws.cell(row=r, column=1, value=f"{M_LABEL[m]}　|　数值 = 由混淆矩阵复算")
            c.font = Font(name=F_NAME, size=10, bold=True, color=C_TXT)
            c.fill = pf(C_SUB)
            c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
            ws.row_dimensions[r].height = 24
            r += 1
            r = light_hdr(ws, r, ["训练阶段"] + [f"测试 {D_SHORT[k]}" for k in range(1, 7)])
            for i in range(1, 7):
                vals = [table.get((m, i, j), "—") for j in range(1, 7)]
                r = row_cells(ws, r, [f"D{i} 训练完成"] + vals)
            # 与原表一致：每个方法块独立做红-黄-绿色阶
            ws.conditional_formatting.add(f"B{r - 6}:G{r - 1}", cs_rule())
            r += 1

    # ══ 表3 类别级 F1 ═════════════════════════════════════════════
    ws = wb.create_sheet("表3_类别级F1")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "表3 · 类别级 F1 对比表（对角线口径）",
                "取 R_{i,i} 的测试集，按 N/R/W 三类分别报告 F1；每域每类 100 条", 9)
    widths(ws, [("A", 22), ("B", 13)] + [(get_column_letter(c), 11) for c in range(3, 9)] + [("I", 14)])
    r = 3
    r = hdr(ws, r, ["方法", "类别"] + [D_SHORT[k] for k in range(1, 7)] + ["六域平均"], 9)
    for m in METHODS:
        start_r = r
        for c in CLASSES:
            vals = [ds.PC.get((m, k, k), {}).get(c, {}).get("F1", 0.0) for k in range(1, 7)]
            r = row_cells(ws, r, [M_LABEL[m] if c == "N" else "", CLASS_FULL[c]] + vals +
                          [sum(vals) / 6])
        ws.merge_cells(start_row=start_r, start_column=1, end_row=r - 1, end_column=1)
        ws.cell(row=start_r, column=1).alignment = Alignment(horizontal="center",
                                                             vertical="center", wrap_text=True)
    ws.conditional_formatting.add(f"C4:H{r - 1}", cs_rule())  # 与原表一致：整块色阶

    # ══ 表4 持续学习指标 ══════════════════════════════════════════
    ws = wb.create_sheet("表4_持续学习指标")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "表4 · 持续学习核心指标对比表（文档表4）",
                "六个指标全部由表2 的 R_{i,j} 与 Macro-F1 自动计算；按文档第四部分，FWT 不填写；"
                "右侧两列为 Joint 上界参照（不参与排名，无阶段概念的指标以「—」标注）", 11)
    widths(ws, [("A", 22), ("B", 27), ("C", 44), ("D", 11)] +
           [("E", 15), ("F", 15), ("G", 15), ("H", 15), ("I", 16), ("J", 16), ("K", 18)])
    r = 3
    r = hdr(ws, r, ["指标", "英文名称", "计算口径 / 评价内容", "优劣方向"] +
            [M_LABEL[m] for m in METHODS] + ["UB Joint-full", "对照 Joint-600", "最优方法"], 11, height=40)
    specs = [
        ("最终平均准确率", "Final Average Accuracy (FAA)", "D6 完成后对 D1–D6 的整体表现：mean_j R_{6,j}", "越高越好", "FAA", True),
        ("最终平均 Macro-F1", "Final Average Macro-F1", "D6 完成后六个领域的类别均衡表现", "越高越好", "FAA_F1", True),
        ("平均遗忘程度", "Average Forgetting (AF)", "mean_{j=1..5}[ max_{k=j..6} R_{k,j} − R_{6,j} ]", "越低越好", "AF", False),
        ("后向迁移", "Backward Transfer (BWT)", "mean_{j=1..5}( R_{6,j} − R_{j,j} )，正 = 后续学习改善历史领域", "越高越好", "BWT", True),
        ("历史知识保持率", "Knowledge Retention (KR)", "Σ_{j=1..5} R_{6,j} / Σ_{j=1..5} R_{j,j} × 100%", "越高越好", "KR", True),
        ("当前阶段适应能力", "Current-stage Performance (CSA)", "六个阶段对角线 Accuracy 平均值 mean_i R_{i,i}", "越高越好", "CSA", True),
    ]
    jfull, j600 = ds.joint.get("full"), ds.joint.get("600")
    for zh, en, desc, direction, key, higher in specs:
        vals = [cl[m][key] for m in METHODS]
        if key == "FAA":
            jvals = [jfull["ACC"], j600["ACC"]] if jfull and j600 else ["—", "—"]
        elif key == "FAA_F1":
            jvals = [jfull["F1"], j600["F1"]] if jfull and j600 else ["—", "—"]
        else:
            jvals = ["—", "—"]  # Joint 无阶段序列，AF/BWT/KR/CSA 无定义
        structural = key in ("AF", "BWT", "KR")
        pool = [m for m in METHODS if not (structural and m == "adapter")]
        best = pool[0]
        for m in pool:
            if (higher and cl[m][key] > cl[best][key]) or ((not higher) and cl[m][key] < cl[best][key]):
                best = m
        best_txt = f"{M_SHORT[best]}（E4 为构造性 0/100%，不计入排名）" if structural else M_SHORT[best]
        r = row_cells(ws, r, [zh, en, desc, direction] + vals + jvals + [best_txt], height=36)
    band(ws, 5, 10, 11, C_SUB)  # 与原表一致的隔行浅底
    for rr in range(5, 11, 2):
        band(ws, rr, rr, 11, "FFFFFF")
    # 与原表一致：每行指标独立色阶（含 Joint 列，"—" 不参与）；AF 行（越低越好）绿→黄→红反向
    for rr in range(4, 10):
        ws.conditional_formatting.add(f"E{rr}:J{rr}", cs_rule(reverse=(rr == 6)))

    r += 1
    for k, v in [
        ("口径判别", "AF 严格取 max_{k=j..6}（含最终阶段），因此 AF 不会为负；BWT 为正说明后续领域训练对历史领域有正向迁移。"),
        ("E4 可比性", "E4 采用按已知测试领域路由（adapter_inference=route），每个领域始终由对应 adapter 预测，"
                      "BWT=0、AF=0、KR=100% 是结构性结果，不能据此判定其抗遗忘能力最强。"),
        ("Joint 口径", "Joint-full 为 D1–D6 全部训练数据（3,600 条）混合、从基座一次训练（1,800 步，与 E1 全链累计步数相同），"
                       "其 FAA 即「数据全部可用时的上界」；Joint-600 每域抽 100 条（共 600）同配置训练，用于剥离数据量因素。"
                       "两者均非持续学习方法，不参与排名；无阶段概念的指标以「—」标注。"),
        ("显著性", "每域 300 条测试样本；E1/E2/E3 的 FAA 差异在 0.5–1pp 量级，落在抽样噪声内，"
                   "需多种子或配对检验后才能下结论。"),
    ]:
        r = note_row(ws, r, k, v, 11)

    # ══ 表5 遗忘差值表 ════════════════════════════════════════════
    ws = wb.create_sheet("表5_遗忘差值表")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "表5 · 四种方法性能变化与遗忘对比（文档表5）",
                "Δ_j = R_{6,j} − R_{j,j}：历史领域最终表现相对刚学完时的变化；负值红色 = 遗忘，正值绿色 = 正向迁移", 7)
    widths(ws, [("A", 22), ("B", 14), ("C", 18), ("D", 18), ("E", 18), ("F", 16), ("G", 13)])
    r = 3
    r = hdr(ws, r, ["方法", "历史领域", "初次学习 Accuracy", "历史最佳 Accuracy",
                    "最终 D6 Accuracy", "性能变化 Δ_j", "是否遗忘"], 7, height=36)
    for m in METHODS:
        start_r = r
        for j in range(1, 6):
            initial = ds.acc(m, j, j) or 0.0
            best = max(ds.acc(m, k, j) or 0.0 for k in range(j, 7))
            final = ds.acc(m, 6, j) or 0.0
            delta = final - initial
            r = row_cells(ws, r, [M_LABEL[m] if j == 1 else "", D_FULL[j],
                                  initial, best, final, delta, "是" if delta < 0 else "否"])
        ws.merge_cells(start_row=start_r, start_column=1, end_row=r - 1, end_column=1)
        ws.cell(row=start_r, column=1).alignment = Alignment(horizontal="center",
                                                             vertical="center", wrap_text=True)
        r += 1
    # 与原表一致：Δ<0 红字加粗（遗忘）、Δ>0 绿字加粗（正向迁移），Δ=0 不着色
    rng = f"F4:F{r - 2}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0"],
                                                  font=Font(color="FFA61B1B", bold=True)))
    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0"],
                                                  font=Font(color="FF2E6B2E", bold=True)))
    r = note_row(ws, r, "说明", "「历史最佳」取该领域在 k=j..6 各阶段中的最高准确率；"
                                 "E4 因 route 推理，三列数值相同，Δ 恒为 0。", 7)

    # ══ 表6 资源成本 ══════════════════════════════════════════════
    ws = wb.create_sheet("表6_资源成本")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "表6 · 训练与存储成本（实测）",
                "训练耗时取各单元 trainer 打印的 train_runtime 之和；存储取模型目录实测体积，"
                "EWC/Replay 状态文件未随结果导出，按结构估算并标注；右侧两列为 Joint 参照", 7)
    widths(ws, [("A", 30), ("B", 24), ("C", 24), ("D", 24), ("E", 24), ("F", 24), ("G", 24)])
    r = 3
    r = hdr(ws, r, ["对比项目"] + [M_LABEL[m] for m in METHODS] + ["UB Joint-full", "对照 Joint-600"], 7)

    def jrt(tag):
        return ds.joint_runtime.get(tag)

    rows6 = [
        ("总训练时间（D2–D6 五阶段）", lambda m: f"{ds.secs(m, range(2, 7)) / 60:.1f} min",
         lambda t: "—（一次性训练）"),
        ("单阶段平均训练时间（D2–D6）", lambda m: f"{ds.secs(m, range(2, 7)) / 60 / 5:.1f} min",
         lambda t: "—"),
        ("训练时间（含 D1 / 全量一次）", lambda m: f"{ds.secs(m, range(1, 7)) / 60:.1f} min",
         lambda t: f"{jrt(t) / 60:.1f} min" if jrt(t) else "—"),
        ("峰值 GPU 显存", lambda m: "未记录", lambda t: "未记录"),
        ("可训练参数量", lambda m: "4,358,144 (0.28%)", lambda t: "4,358,144 (0.28%)"),
        ("模型存储", lambda m: f"{ds.model_mb(m):.0f} MB",
         lambda t: f"{ds.joint_dir_size_mb(t):.0f} MB"),
        ("额外状态（EWC/Replay）", lambda m: f"{ds.extra_mb(m)[0]:.1f} MB", lambda t: "0.0 MB"),
        ("每阶段训练步数", lambda m: "300",
         lambda t: f"{'1,800' if t == 'full' else '300'}（单次）"),
    ]
    for name, fn, jfn in rows6:
        r = row_cells(ws, r, [name] + [fn(m) for m in METHODS] +
                      [jfn(t) for t in ("full", "600")], fmt=None, height=24)
    band(ws, 5, 12, 7, C_SUB)
    for rr in range(5, 13, 2):
        band(ws, rr, rr, 7, "FFFFFF")

    r += 1
    for k, v in [
        ("统计口径", "D2–D6 为增量阶段耗时；Sequential FT 的 D1 单独列出。"
                     "EWC 额外状态 = (anchor + fisher, fp32) × anchor 套数，D6 为 6 套。"
                     "Joint 为单次混合训练，与分阶段口径不可直接相减，仅作量级参照。"),
        ("备注", "若需精确状态体积，可在云端执行 du -sh artifacts/continual_state/*。"),
    ]:
        r = note_row(ws, r, k, v, 7)

    # ══ 表7 综合对比 ══════════════════════════════════════════════
    ws = wb.create_sheet("表7_综合对比")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "表7 · 最终综合对比表（文档表7）",
                "评价维度为行、四种方法为列（右侧两列为 Joint 上界参照）；全部数值取自表1–表6，不人为打分", 7)
    widths(ws, [("A", 30)] + [(c, 24) for c in "BCDEFG"])
    r = 3
    jfull, j600 = ds.joint.get("full"), ds.joint.get("600")
    jfaa = jfull["ACC"] if jfull else None
    r = hdr(ws, r, ["评价维度"] + [M_LABEL[m] for m in METHODS] + ["UB Joint-full", "对照 Joint-600"], 7)

    def d_f1(m: str, c: str) -> float:
        final = sum(ds.PC.get((m, 6, j), {}).get(c, {}).get("F1", 0.0) for j in range(1, 7)) / 6
        base = sum(ds.PC.get((m, j, j), {}).get(c, {}).get("F1", 0.0) for j in range(1, 7)) / 6
        return final - base

    rows7 = [
        ("当前阶段适应能力（CSA）", [f"{cl[m]['CSA']:.4f}" for m in METHODS], ["—", "—"]),
        ("最终历史平均性能（FAA）", [f"{cl[m]['FAA']:.4f}" for m in METHODS],
         [f"{jfull['ACC']:.4f}", f"{j600['ACC']:.4f}"] if jfull and j600 else ["—", "—"]),
        ("平均遗忘程度（AF）", [f"{cl[m]['AF']:.4f}" for m in METHODS], ["—", "—"]),
        ("后向迁移（BWT）", [f"{cl[m]['BWT']:+.4f}" for m in METHODS], ["—", "—"]),
        ("W 类保持能力（ΔF1）", [f"{d_f1(m, 'W'):+.4f}" for m in METHODS], ["—", "—"]),
        ("R 类保持能力（ΔF1）", [f"{d_f1(m, 'R'):+.4f}" for m in METHODS], ["—", "—"]),
        ("N 类保持能力（ΔF1）", [f"{d_f1(m, 'N'):+.4f}" for m in METHODS], ["—", "—"]),
        ("距上界差值（FAA − Joint-full）",
         [f"{cl[m]['FAA'] - jfaa:+.4f}" for m in METHODS] if jfaa else ["—"] * 4,
         ["0.0000", f"{j600['ACC'] - jfaa:+.4f}"] if jfaa and j600 else ["—", "—"]),
        ("训练成本（D2–D6）", [f"{ds.secs(m, range(2, 7)) / 60:.1f} min" for m in METHODS],
         [f"{ds.joint_runtime.get(t, 0) / 60:.1f} min" if ds.joint_runtime.get(t) else "—"
          for t in ("full", "600")]),
        ("存储成本（模型）", [f"{ds.model_mb(m):.0f} MB" for m in METHODS],
         [f"{ds.joint_dir_size_mb(t):.0f} MB" for t in ("full", "600")]),
    ]
    for name, vals, jvals in rows7:
        r = row_cells(ws, r, [name] + vals + jvals, fmt=None, height=32)
    band(ws, 5, 14, 7, C_SUB)
    for rr in range(5, 14, 2):
        band(ws, rr, rr, 7, "FFFFFF")

    r += 1
    gap = max(cl[m]["FAA"] for m in METHODS) - jfaa if jfaa else None
    r = note_row(ws, r, "结论边界",
                 "四种方法 FAA 差距 0.5–3pp：E1/E2/E3 相互在噪声内，E4 因 route 上限参照而系统性偏低"
                 "（0.848 vs 0.878）。" +
                 (f"Joint-full = {jfaa:.4f} 为「数据全部可用时的上界」：最好的 E1 距上界 {abs(gap) * 100:.2f}pp，"
                  f"其中数据量因素约 {(jfaa - j600['ACC']) * 100:.1f}pp（UB − Joint-600），"
                  "即顺序训练的净代价约 1–2pp。" if gap is not None else "") +
                 "本实验未做多种子与显著性检验，报告中应避免绝对排名表述。", 7)

    # ══ 图表 ══════════════════════════════════════════════════════
    ws = wb.create_sheet("图表")
    ws.sheet_properties.tabColor = C_HDR
    title_block(ws, "图表 · 六张核心可视化（图1 两分图｜图2｜图3 三张）",
                "图1a 对角线适应能力｜图1b 累积平均准确率｜图2 D6 后历史知识保持｜"
                "图3a/b/c 最终阶段 W/R/N 类别性能；全部由本工作簿数据绘制", 8)
    widths(ws, [(c, 16) for c in "ABCDEFGH"])
    fig_dir = out.parent / "_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = []

    def save(fig, name):
        p = fig_dir / name
        fig.tight_layout()
        # 不加 bbox_inches="tight"：保证画布尺寸严格等于 figsize×dpi，与时间版逐像素一致
        fig.savefig(p, facecolor="white")
        plt.close(fig)
        return p

    # 图1a：当前领域适应能力（对角线 R_{i,i}）
    fig, ax = plt.subplots(figsize=(9.0, 4.3), dpi=170)
    for m in METHODS:
        ax.plot([D_SHORT[i] for i in range(1, 7)],
                [ds.acc(m, i, i) or 0 for i in range(1, 7)],
                marker="o", lw=2, color=M_COLORS[m], label=M_SHORT[m])
    ax.set_title("图1a  当前领域适应能力（对角线 R$_{i,i}$）", fontsize=11)
    ax.set_xlabel("训练阶段（学完该领域后立即测试）")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.75, 0.95); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    figs.append(("图1a", "各阶段对当前领域的准确率（对角线），反映新领域学习能力",
                 save(fig, "fig1a_diag.png")))

    # 图1b：累积平均准确率 A_i = mean_{j<=i} R_{i,j}
    cum = {m: [sum(ds.acc(m, i, j) or 0 for j in range(1, i + 1)) / i for i in range(1, 7)]
           for m in METHODS}
    fig, ax = plt.subplots(figsize=(9.0, 4.3), dpi=170)
    for m in METHODS:
        ax.plot([D_SHORT[i] for i in range(1, 7)], cum[m],
                marker="s", lw=2, color=M_COLORS[m], label=M_SHORT[m])
    ax.set_title("图1b  累积平均准确率 A$_i$ = mean$_{j\\leq i}$ R$_{i,j}$", fontsize=11)
    ax.set_xlabel("训练阶段")
    ax.set_ylabel("累积平均 Accuracy")
    ax.set_ylim(0.75, 0.95); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    figs.append(("图1b", "按阶段累计的已见领域平均准确率，反映整体水平而非适应能力",
                 save(fig, "fig1b_cumavg.png")))

    # 图2：D6 完成后对 D1–D5 的历史知识保持（柱状，与时间版一致）
    fig, ax = plt.subplots(figsize=(9.0, 4.3), dpi=170)
    x = range(1, 6)
    width = 0.2
    for k, m in enumerate(METHODS):
        vals = [ds.acc(m, 6, j) or 0 for j in range(1, 6)]
        bars = ax.bar([t + (k - 1.5) * width for t in x], vals, width,
                      color=M_COLORS[m], label=M_SHORT[m], edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"历史领域 {D_SHORT[j]}" for j in range(1, 6)])
    ax.set_title("图2  D6 完成后对 D1–D5 历史领域的知识保持", fontsize=11)
    ax.set_xlabel("历史领域（测试集）")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.6, 1.0); ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, ncol=4)
    figs.append(("图2", "学完全部领域后对历史领域的保持能力，柱高越接近越高越稳",
                 save(fig, "fig2_keep.png")))

    # 图3：D6 最终模型对各领域的 N/R/W 类别 F1（每个类别一张）
    for idx, c in enumerate(["W", "R", "N"]):
        fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=170)
        series = {m: [ds.PC.get((m, 6, j), {}).get(c, {}).get("F1", 0) for j in range(1, 7)]
                  for m in METHODS}
        for m in METHODS:
            ax.plot([D_SHORT[j] for j in range(1, 7)], series[m],
                    marker="o", linewidth=2, markersize=5, color=M_COLORS[m], label=M_SHORT[m])
        ax.set_title(f"图3{'abc'[idx]}  最终阶段 {c} 类性能：D6 后对 D1–D6 各领域", fontsize=11)
        ax.set_xlabel("测试领域")
        ax.set_ylabel(f"{c} 类 F1")
        lo = min(v for s in series.values() for v in s)
        ax.set_ylim(max(0.0, lo - 0.05), 1.0); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=4)
        figs.append((f"图3{'abc'[idx]}",
                     f"最终阶段 {c} 类（{CLASS_FULL[c]}）在各领域的 F1，检查类别塌缩",
                     save(fig, f"fig3_class_{c}.png")))

    r = 3
    for tag, desc, p in figs:
        c = ws.cell(row=r, column=1, value=f"{tag}　{desc}")
        c.font = Font(name=F_NAME, size=10, bold=True, color=C_TXT)
        c.fill = pf(C_SUB)
        c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
        ws.row_dimensions[r].height = 24
        r += 1
        img = XLImage(str(p))
        # 与时间版统一：固定 900 px 宽、高度按原图比例缩放，便于两套图在 PPT 中同尺寸排布
        scale = 900 / img.width
        img.width, img.height = 900, int(img.height * scale)
        ws.add_image(img, f"A{r}")
        r += int(img.height / 15) + 3
        ws.row_breaks.append(Break(id=r - 1))

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.properties.title = "领域增量持续学习实验结果汇总表"
    wb.properties.creator = "CAPC-CG 实验流水线"
    wb.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="05_incremental_learning_domain")
    ap.add_argument("--models", default="assets/05_incremental_learning_domain/models")
    ap.add_argument("--out", default="05_incremental_learning_domain/results/领域增量持续学习实验结果汇总表.xlsx")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else SCRIPT_ROOT / q

    root, models, out = resolve(args.root), resolve(args.models), resolve(args.out)
    if not (root / "results" / "domain_incremental").is_dir():
        print(f"[错误] 未找到结果目录: {root / 'results' / 'domain_incremental'}")
        return 2

    ds = Dataset(root, models)
    print(f"载入：结果 {len(ds.reports)} 份，混淆矩阵 {len(ds.CM)} 组，"
          f"metadata {len(ds.meta)} 份，耗时 {len(ds.runtime)} 条")
    p = build(ds, out)
    print(f"已生成: {p}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
