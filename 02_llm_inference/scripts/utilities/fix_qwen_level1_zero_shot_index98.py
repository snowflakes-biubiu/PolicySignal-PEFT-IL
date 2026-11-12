# ============================================================
# 实验标题：Qwen Level-1 Zero-shot index=98 结果回填
# ============================================================

import json
import shutil

SOURCE_FILE = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300.json"
OUTPUT_FILE = "02_llm_inference/results/qwen/qwen_level1_zero_shot_300_fixed.json"

# 先复制原始结果文件
shutil.copy2(SOURCE_FILE, OUTPUT_FILE)

with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

found = False

for item in data["results"]:

    if item.get("index") == 98:

        print("找到 index=98")

        print("原 prediction：", item.get("prediction"))

        # 这是之前实际成功获得的 Qwen 模型输出
        item["prediction"] = "R"
        item["raw_output"] = "R"

        # 标记这是对已有成功重试结果的回填
        item["error"] = None
        item["error_detail"] = None
        item["attempt"] = "retry_success"

        found = True

        print("回填 prediction：R")

        break


if not found:
    raise RuntimeError("没有找到 index=98")


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("=" * 70)
print("index=98 回填完成")
print("=" * 70)
print("新文件：", OUTPUT_FILE)