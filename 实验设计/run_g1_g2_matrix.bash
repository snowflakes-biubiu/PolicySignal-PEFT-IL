#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录执行：bash run_g1_g2_matrix.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${SCRIPT:-run_qwen3_peft_benchmark.py}"
OUT_ROOT="${OUT_ROOT:-results_qwen3_benchmark}"

run_exp() {
  local exp_id="$1"
  local model="$2"
  local task="$3"
  local precision="$4"
  local r="$5"
  local alpha="$6"
  local scope="$7"
  local lr="$8"
  local bs="$9"
  local ga="${10}"
  local extra="${11:-}"

  echo "============================================================"
  echo "Running: ${exp_id}"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT}" \
    --exp-id "${exp_id}" \
    --model "${model}" \
    --task "${task}" \
    --precision "${precision}" \
    --lora-r "${r}" \
    --lora-alpha "${alpha}" \
    --target-scope "${scope}" \
    --lr "${lr}" \
    --batch-size "${bs}" \
    --grad-accum "${ga}" \
    --output-root "${OUT_ROOT}" \
    --skip-existing \
    ${extra}
}

# ------------------------------------------------------------
# G1：代际锚点 (Qwen2.5-1.5B vs Qwen3-1.7B) 与 bf16 LoRA 配比重标定
# ------------------------------------------------------------
for TASK in task1_level1 task2_level2; do
  run_exp "G1-1_qwen25_1.5b_qlora_r16_${TASK}" "Qwen/Qwen2.5-1.5B-Instruct" "${TASK}" "4bit" 16 32 "attn"     "2e-4" 4 4
  run_exp "G1-2_qwen3_1.7b_qlora_r16_${TASK}"  "Qwen/Qwen3-1.7B"            "${TASK}" "4bit" 16 32 "attn"     "2e-4" 4 4
  run_exp "G1-3_qwen3_1.7b_bf16_r8_${TASK}"    "Qwen/Qwen3-1.7B"            "${TASK}" "bf16" 8  16 "attn_mlp" "2e-4" 4 4
  run_exp "G1-4_qwen3_1.7b_bf16_r16_${TASK}"   "Qwen/Qwen3-1.7B"            "${TASK}" "bf16" 16 32 "attn_mlp" "2e-4" 4 4
  run_exp "G1-5_qwen3_1.7b_bf16_r16a16_${TASK}" "Qwen/Qwen3-1.7B"           "${TASK}" "bf16" 16 16 "attn_mlp" "2e-4" 4 4
done

# ------------------------------------------------------------
# G2：参数量阶梯对比 (1.7B / 4B / 8B) — 默认采用 r16/a32/attn_mlp
# 若 G1 结果显示 r8 更优，可把下面的 16 32 改为 8 16
# ------------------------------------------------------------
for TASK in task1_level1 task2_level2; do
  # G2-2 (Qwen3-1.7B) 与 G1-4 同配置，--skip-existing 会自动复用
  run_exp "G1-4_qwen3_1.7b_bf16_r16_${TASK}" "Qwen/Qwen3-1.7B" "${TASK}" "bf16" 16 32 "attn_mlp" "2e-4" 4 4
  run_exp "G2-3_qwen3_4b_bf16_r16_${TASK}"   "Qwen/Qwen3-4B"   "${TASK}" "bf16" 16 32 "attn_mlp" "2e-4" 4 4
  run_exp "G2-4_qwen3_8b_bf16_r16_${TASK}"   "Qwen/Qwen3-8B"   "${TASK}" "bf16" 16 32 "attn_mlp" "1e-4" 2 8 "--gradient-checkpointing"
done

# ------------------------------------------------------------
# 自动汇总所有 summary.json 到单张对比 CSV
# ------------------------------------------------------------
"${PYTHON_BIN}" - <<PY
import csv, json
from pathlib import Path

root = Path("${OUT_ROOT}")
rows = []
for p in sorted(root.glob("*/summary.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "exp_id": d["exp_id"],
        "task": d["task"],
        "model": d["model"],
        "precision": d["precision"],
        "r": d["lora_r"],
        "alpha": d["lora_alpha"],
        "scope": d["target_scope"],
        "lr": d["lr"],
        "trainable_pct": d["trainable_params_pct"],
        "accuracy": d["accuracy"],
        "macro_f1": d["macro_f1"],
        "cohen_kappa": d["cohen_kappa"],
        "train_sec": d["train_time_sec"],
        "train_vram_gb": d["peak_train_vram_gb"],
        "infer_sps": d["infer_samples_per_sec"],
        "eval_vram_gb": d["peak_eval_vram_gb"],
    })

out_csv = root / "g1_g2_comparison_summary.csv"
if rows:
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"已汇总 {len(rows)} 条实验结果至: {out_csv}")
PY