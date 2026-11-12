# ============================================================
# 实验标题：Qwen Level-1 Few-shot k=6 Checkpoint 修复
# ============================================================

import json
import shutil

INPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_fewshot6_300_checkpoint.json"
BACKUP_FILE = "02_llm_inference/results/qwen/qwen_level1_fewshot6_300_checkpoint_backup.json"


# ============================================================
# 读取 checkpoint
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [])

print("=" * 70)
print("实验标题：Qwen Level-1 Few-shot k=6 Checkpoint 修复")
print("=" * 70)

print("原 checkpoint 记录数：", len(results))


# ============================================================
# 统计成功 / 失败
# ============================================================

success_results = [
    item
    for item in results
    if item.get("prediction") in {"N", "R", "W"}
]

failed_results = [
    item
    for item in results
    if item.get("prediction") not in {"N", "R", "W"}
]

print("成功记录：", len(success_results))
print("失败记录：", len(failed_results))


# ============================================================
# 备份原 checkpoint
# ============================================================

shutil.copy2(
    INPUT_FILE,
    BACKUP_FILE
)

print()
print("原 checkpoint 已备份：")
print(BACKUP_FILE)


# ============================================================
# 只保留成功结果
# ============================================================

data["results"] = success_results

data["valid_predictions"] = len(success_results)

data["failed_predictions"] = 0


# ============================================================
# 保存修复后的 checkpoint
# ============================================================

with open(
    INPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 输出结果
# ============================================================

print()
print("=" * 70)
print("Checkpoint 修复完成")
print("=" * 70)

print("保留成功记录：", len(success_results))
print("删除失败记录：", len(failed_results))

print()
print("现在可以重新运行：")
print("python -u 02_llm_inference/scripts/runners/test_qwen_level1_fewshot6_300.py")

print("=" * 70)