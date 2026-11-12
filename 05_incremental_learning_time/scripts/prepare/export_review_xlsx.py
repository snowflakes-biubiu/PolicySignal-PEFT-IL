"""导出最终 5,400 段的人工核对表（xlsx + csv）。

Sheet 全部5400：逐段明细 + 复核记录 + 盲审结论 + 空白人工判定列
Sheet 重点核对：300 段盲审样本，疑似问题排在最前
Sheet 统计：公式统计，Excel 打开自动计算
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DATA = Path(__file__).resolve().parents[3] / "assets" / "05_incremental_learning_time" / "data" / "domain_v14" / "composite"
FINAL = DATA / "final"
XLSX = DATA / "final_5400_review.xlsx"
CSV = DATA / "final_5400_review.csv"

HEAD = ["序号", "领域", "划分", "WRN", "年代", "年份", "标题", "段落正文", "字数",
        "来源文件", "标签模型", "标签置信度", "复核轮次", "复核模型", "复核置信度",
        "复核理由", "盲审抽中", "盲审判定领域", "盲审段落有效", "盲审问题",
        "人工判定", "人工备注"]


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    rows = []
    for p in sorted(FINAL.glob("D*_*.jsonl")):
        split = "train" if "train" in p.name else "test"
        for r in load_jsonl(p):
            rows.append((r, split, p.name))
    rows.sort(key=lambda x: (x[0]["domain"], x[1] != "train", x[0]["wrn"], x[0]["rank"]))

    best = {}
    for v in load_jsonl(DATA / "verdicts.jsonl"):
        if v.get("parsed") and (v["pid"] not in best or v["round"] >= best[v["pid"]]["round"]):
            best[v["pid"]] = v
    blind = {r["pid"]: r for r in load_jsonl(DATA / "blind_audit_samples.jsonl")}

    data = []
    for i, (r, split, fname) in enumerate(rows, 1):
        v = best.get(r["pid"], {})
        b = blind.get(r["pid"])
        bad = bool(b and b.get("parsed") and
                   (not b.get("valid") or b.get("blind_domain") != r["domain"]))
        data.append([
            i, r["domain"], split, r["wrn"], r["era"], r["year"], r["Title"], r["text"],
            len(r["text"]), fname, r.get("label_model"), r.get("label_conf"),
            f"r{v.get('round', '')}", v.get("model"), v.get("confidence"), v.get("reason", ""),
            "是" if b else "", b.get("blind_domain", "") if b else "",
            ("有效" if b.get("valid") else "无效") if b else "", "是" if bad else "",
            "", "",
        ])

    wb = Workbook()
    head_font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF", size=10)
    head_fill = PatternFill("solid", start_color="1F4E78")
    body_font = Font(name="Microsoft YaHei", size=10)
    input_fill = PatternFill("solid", start_color="FFF2CC")
    bad_fill = PatternFill("solid", start_color="FCE4E4")

    def style(ws, widths, n_rows, wrap_cols=(), bad_col=None, input_cols=()):
        for c, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w
        for cell in ws[1]:
            cell.font = head_font
            cell.fill = head_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows(min_row=2, max_row=n_rows + 1):
            flag_bad = bad_col and row[bad_col - 1].value == "是"
            for j, cell in enumerate(row, 1):
                cell.font = body_font
                cell.alignment = Alignment(vertical="top", wrap_text=j in wrap_cols)
                if flag_bad:
                    cell.fill = bad_fill
                elif j in input_cols:
                    cell.fill = input_fill
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(widths))}{n_rows + 1}"

    ws = wb.active
    ws.title = "全部5400"
    ws.append(HEAD)
    for row in data:
        ws.append(row)
    widths = [7, 22, 8, 7, 18, 8, 48, 80, 8, 16, 22, 12, 10, 22, 12, 22,
              10, 22, 12, 10, 14, 30]
    style(ws, widths, len(data), wrap_cols=(7, 8, 16), bad_col=20, input_cols=(21, 22))

    focus = [r for r in data if r[16] == "是"]
    focus.sort(key=lambda r: (r[19] != "是", r[1], r[3]))
    ws2 = wb.create_sheet("重点核对")
    ws2.append(["序号", "疑似问题", "领域", "划分", "WRN", "年代", "标题", "段落正文",
                "盲审判定领域", "盲审段落有效", "盲审理由", "人工判定", "人工备注"])
    pid_by_seq = {i: r[0]["pid"] for i, r in enumerate(rows, 1)}
    blind_reason = {r["pid"]: r.get("reason", "") for r in load_jsonl(DATA / "blind_audit_samples.jsonl")}
    for r in focus:
        ws2.append([r[0], r[19], r[1], r[2], r[3], r[4], r[6], r[7],
                    r[17], r[18], blind_reason.get(pid_by_seq.get(r[0]), ""), "", ""])
    style(ws2, [7, 10, 22, 8, 7, 18, 48, 80, 22, 12, 24, 14, 30],
          len(focus), wrap_cols=(7, 8, 11), bad_col=2, input_cols=(12, 13))

    ws3 = wb.create_sheet("统计")
    ws3.append(["维度", "类别", "数量", "说明"])
    dom_split = Counter((r[1], r[2]) for r in data)
    wrn = Counter(r[3] for r in data)
    era = Counter(r[4] for r in data)
    ws3.append(["总计", "全部", len(data), "导出时快照"])
    for d in sorted({r[1] for r in data}):
        for s in ("train", "test"):
            ws3.append(["域×划分", f"{d} / {s}", dom_split[(d, s)], "目标 600 / 300"])
    for w in ("W", "R", "N"):
        ws3.append(["WRN", w, wrn[w], "目标 1,800"])
    for e in ("T1_1949_1978", "T2_1979_1992", "T3_1993_2002",
              "T4_2003_2012", "T5_2013_2023"):
        ws3.append(["年代", e, era[e], "T1 不强制每类 5 段"])
    ws3.append(["盲审", "抽中", sum(1 for r in data if r[16] == "是"), "300 段分层抽样"])
    ws3.append(["盲审", "疑似问题", sum(1 for r in data if r[19] == "是"), "建议优先核对"])
    style(ws3, [16, 34, 14, 28], ws3.max_row - 1)
    for row in ws3.iter_rows(min_row=2, min_col=3, max_col=3):
        for cell in row:
            cell.number_format = "#,##0"

    xlsx_path, csv_path = XLSX, CSV
    try:
        wb.save(XLSX)
    except PermissionError:
        alt = time.strftime("_%Y%m%d_%H%M%S")
        xlsx_path = XLSX.with_name(XLSX.stem + alt + XLSX.suffix)
        csv_path = CSV.with_name(CSV.stem + alt + CSV.suffix)
        wb.save(xlsx_path)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        w.writerows(data)

    print(f"导出 {len(data):,} 段；盲审样本 {len(focus)} 段；疑似问题 "
          f"{sum(1 for r in data if r[19] == '是')} 段")
    print(f"xlsx: {xlsx_path}")
    print(f"csv : {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
