"""持续学习三方法（EWC / Replay / Adapter-CL）对比分析。

输入:  05_incremental_learning_time/results/temporal/temporal_{method}_task{N}_cumulative_test.json(.csv)
输出:  stdout 各表 + 05_incremental_learning_time/results/continual_summary.json

分析内容：
  1. 累积准确率矩阵（after_task N × 测试任务 T1..T5），四方法并排
  2. 逐测试任务的遗忘曲线（峰值、终值、Forgetting）
  3. 最终阶段逐类 F1（N/R/W 谁在退化）
  4. 混淆结构聚合（主导错误方向）
  5. McNemar 配对检验（sequential_ft 基线 vs 各持续学习方法，最终阶段）
  6. 最终阶段按年代切片的准确率

用法：
    python 05_incremental_learning_time/scripts/validation/analyze_continual_results.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "05_incremental_learning_time" / "results" / "temporal"
OUT_JSON = PROJECT_ROOT / "05_incremental_learning_time" / "results" / "continual_summary.json"

METHODS = ["sequential_ft", "ewc", "replay", "adapter"]
METHOD_CN = {"sequential_ft": "朴素顺序微调(基线)", "ewc": "EWC(λ=100,param)",
             "replay": "Replay(20条/类/任务)", "adapter": "Adapter-CL(多适配器)"}
TASK_NAMES = {1: "T1_1949_1978", 2: "T2_1979_1992", 3: "T3_1993_2002",
              4: "T4_2003_2012", 5: "T5_2013_2023"}


def load(method: str, stage: int) -> dict | None:
    p = RESULT_DIR / f"temporal_{method}_task{stage}_cumulative_test.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def load_predictions(method: str, stage: int) -> list[dict]:
    import csv
    p = RESULT_DIR / f"temporal_{method}_task{stage}_cumulative_test_predictions.csv"
    if not p.is_file():
        return []
    with p.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mcnemar_exact(a_rows: list[dict], b_rows: list[dict]) -> tuple[int, int, float]:
    """配对 McNemar 精确检验（按 ParagraphID 对齐）。返回 (b, c, p)。"""
    a = {r["ParagraphID"]: r for r in a_rows}
    b = {r["ParagraphID"]: r for r in b_rows}
    common = set(a) & set(b)
    b_only = sum(1 for k in common if a[k]["correct"] == "True" and b[k]["correct"] != "True")
    c_only = sum(1 for k in common if a[k]["correct"] != "True" and b[k]["correct"] == "True")
    n = b_only + c_only
    if n == 0:
        return 0, 0, 1.0
    k = min(b_only, c_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return b_only, c_only, min(p, 1.0)


def main() -> None:
    summary: dict = {"matrix": {}, "final_stage": {}, "forgetting_curves": {},
                     "per_class_final": {}, "confusion_final": {}, "mcnemar": {},
                     "era_slice_final": {}}

    # ---------- 1. 累积准确率矩阵 ----------
    print("=" * 100)
    print("【1】累积准确率矩阵  (行=训练到第N任务后 / 列=在Tk测试集上的ACC)")
    print("=" * 100)
    matrix: dict[str, dict[int, dict[str, float]]] = {}
    for m in METHODS:
        matrix[m] = {}
        # 优先从 task5 报告取完整矩阵（含历史行）；缺失的行再从各阶段文件补齐
        full_am: dict = {}
        d5 = load(m, 5)
        if d5:
            full_am = (d5.get("continual_metrics") or {}).get("accuracy_matrix", {})
        for n in range(1, 6):
            row = full_am.get(f"after_task{n}")
            if row is None:
                d = load(m, n)
                am = (d.get("continual_metrics") or {}).get("accuracy_matrix", {}) if d else {}
                row = am.get(f"after_task{n}", {})
            matrix[m][n] = dict(row)
    summary["matrix"] = {m: {str(n): v for n, v in rows.items()} for m, rows in matrix.items()}

    for m in METHODS:
        print(f"\n[{METHOD_CN[m]}]  ({m})")
        print(f"{'':>14}" + "".join(f"{'T'+str(k):>10}" for k in range(1, 6)))
        for n in range(1, 6):
            row = matrix[m].get(n, {})
            cells = ""
            for k in range(1, 6):
                key = TASK_NAMES[k]
                v = row.get(key)
                cells += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
            print(f"{'after_task'+str(n):>14}{cells}")

    # ---------- 2. 遗忘曲线 ----------
    print("\n" + "=" * 100)
    print("【2】逐测试任务遗忘曲线（每次学到新任务后，老任务的ACC变化）")
    print("=" * 100)
    for m in METHODS:
        print(f"\n[{METHOD_CN[m]}]")
        print(f"{'测试任务':<16}" + "".join(f"{'after_T'+str(k):>11}" for k in range(1, 6))
              + f"{'峰值':>9}{'终值':>9}{'遗忘':>9}")
        curves = {}
        for k in range(1, 6):
            key = TASK_NAMES[k]
            traj = [matrix[m].get(n, {}).get(key) for n in range(k, 6)]
            traj = [t for t in traj if t is not None]
            if not traj:
                continue
            peak, final = max(traj), traj[-1]
            curves[key] = {"trajectory": traj, "peak": peak, "final": final,
                           "forgetting": round(peak - final, 4)}
            cells = "".join(f"{t:>11.4f}" for t in traj) + f"{'':>{11*(5-len(traj))}}"
            print(f"{key:<16}{cells}{peak:>9.4f}{final:>9.4f}{peak-final:>+9.4f}")
        summary["forgetting_curves"][m] = curves

    # ---------- 3. 最终阶段汇总指标 ----------
    print("\n" + "=" * 100)
    print("【3】最终阶段（after_task5）汇总")
    print("=" * 100)
    print(f"{'方法':<26}{'ACC':>8}{'BWT':>9}{'Forgetting':>12}{'T1终值':>9}{'T1遗忘':>9}")
    for m in METHODS:
        d = load(m, 5)
        if not d or not d.get("continual_metrics"):
            continue
        cm = d["continual_metrics"]
        t1 = summary["forgetting_curves"].get(m, {}).get(TASK_NAMES[1], {})
        summary["final_stage"][m] = {
            "ACC": cm.get("ACC"), "BWT": cm.get("BWT"), "Forgetting": cm.get("Forgetting"),
            "T1_final": t1.get("final"), "T1_forgetting": t1.get("forgetting")}
        print(f"{METHOD_CN[m]:<26}{cm['ACC']:>8.4f}{cm['BWT']:>+9.4f}{cm['Forgetting']:>12.4f}"
              f"{t1.get('final', float('nan')):>9.4f}{t1.get('forgetting', float('nan')):>+9.4f}")

    # ---------- 4. 最终阶段逐类 F1 ----------
    print("\n" + "=" * 100)
    print("【4】最终阶段逐类 F1（按测试任务展开 + 各类均值）")
    print("=" * 100)
    for m in METHODS:
        d = load(m, 5)
        if not d:
            continue
        print(f"\n[{METHOD_CN[m]}]")
        print(f"{'测试任务':<16}{'N-F1':>9}{'R-F1':>9}{'W-F1':>9}{'Macro-F1':>10}{'Kappa':>9}")
        agg = defaultdict(list)
        for t in range(1, 6):
            info = d["metrics_by_test_task"].get(TASK_NAMES[t])
            if not info:
                continue
            pc = info["per_class"]
            for lab in ("N", "R", "W"):
                agg[lab].append(pc[lab]["f1"])
            print(f"{TASK_NAMES[t]:<16}{pc['N']['f1']:>9.4f}{pc['R']['f1']:>9.4f}"
                  f"{pc['W']['f1']:>9.4f}{info['macro_f1']:>10.4f}{info['cohen_kappa']:>9.4f}")
        means = {lab: round(sum(v)/len(v), 4) for lab, v in agg.items()}
        summary["per_class_final"][m] = means
        print(f"{'各类均值':<14}{means['N']:>9.4f}{means['R']:>9.4f}{means['W']:>9.4f}")

    # ---------- 5. 混淆结构聚合 ----------
    print("\n" + "=" * 100)
    print("【5】最终阶段混淆结构（T1..T5 合计 1500 条；行=gold 列=pred）")
    print("=" * 100)
    for m in METHODS:
        d = load(m, 5)
        if not d:
            continue
        total = [[0]*3 for _ in range(3)]
        for t in range(1, 6):
            info = d["metrics_by_test_task"].get(TASK_NAMES[t])
            if not info:
                continue
            for i in range(3):
                for j in range(3):
                    total[i][j] += info["confusion_matrix"][i][j]
        print(f"\n[{METHOD_CN[m]}]   (gold\\pred)  N      R      W")
        for i, lab in enumerate(("N", "R", "W")):
            print(f"  {lab:<6}{total[i][0]:>7}{total[i][1]:>7}{total[i][2]:>7}")
        errs = []
        for i, g in enumerate(("N", "R", "W")):
            for j, p in enumerate(("N", "R", "W")):
                if i != j:
                    errs.append((f"{g}→{p}", total[i][j]))
        errs.sort(key=lambda x: -x[1])
        summary["confusion_final"][m] = {"matrix": total,
                                         "top_errors": errs[:3]}
        print("  主导错误:", "  ".join(f"{k}={v}" for k, v in errs[:3]))

    # ---------- 6. McNemar 配对检验 ----------
    print("\n" + "=" * 100)
    print("【6】McNemar 配对检验（最终阶段 1500 条；基线=sequential_ft）")
    print("=" * 100)
    base_rows = load_predictions("sequential_ft", 5)
    for m in METHODS:
        if m == "sequential_ft":
            continue
        rows = load_predictions(m, 5)
        if not rows or not base_rows:
            print(f"  {m}: 缺预测文件，跳过")
            continue
        b, c, p = mcnemar_exact(base_rows, rows)
        verdict = "差异不显著" if p >= 0.05 else "差异显著"
        summary["mcnemar"][m] = {"baseline_only_correct": b, "method_only_correct": c, "p": p}
        print(f"  {METHOD_CN[m]:<26} 基线独对={b}  方法独对={c}  p={p:.4f}  → {verdict}")

    # ---------- 7. 最终阶段年代切片 ----------
    print("\n" + "=" * 100)
    print("【7】最终阶段按年代切片（1500 条按 era 分组 ACC）")
    print("=" * 100)
    eras = [TASK_NAMES[k] for k in range(1, 6)]
    print(f"{'方法':<26}" + "".join(f"{e[:2]+e[3:7]:>12}" for e in eras))
    for m in METHODS:
        rows = load_predictions(m, 5)
        if not rows:
            continue
        ranges = {"T1_1949_1978": (1949, 1978), "T2_1979_1992": (1979, 1992),
                  "T3_1993_2002": (1993, 2002), "T4_2003_2012": (2003, 2012),
                  "T5_2013_2023": (2013, 2023)}

        def era_of(year: float) -> str:
            for name, (lo, hi) in ranges.items():
                if lo <= year <= hi:
                    return name
            return "?"

        by_era = defaultdict(lambda: [0, 0])
        for r in rows:
            slot = by_era[era_of(int(float(r["year"])))]
            slot[1] += 1
            if r["correct"] == "True":
                slot[0] += 1
        cells = "".join(f"{(by_era[e][0]/by_era[e][1] if by_era[e][1] else 0):>12.4f}"
                        for e in eras)
        summary["era_slice_final"][m] = {e: (round(by_era[e][0]/by_era[e][1], 4)
                                             if by_era[e][1] else None) for e in eras}
        print(f"{METHOD_CN[m]:<26}{cells}")

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写入: {OUT_JSON}")


if __name__ == "__main__":
    main()
