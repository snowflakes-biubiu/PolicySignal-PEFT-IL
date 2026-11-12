import json
from pathlib import Path

files = [
    Path("shared/datasets/processed_readable/task1_level1__train.jsonl"),
    Path("shared/datasets/processed_readable/task1_level1__validation.jsonl"),
]

for path in files:
    with path.open("r", encoding="utf-8") as f:
        row = json.loads(next(f))

    print("\n文件：", path)
    print("顶层字段：", list(row.keys()))
    print("首条记录：")
    print(json.dumps(row, ensure_ascii=False, indent=2)[:3000])
