import json
from pathlib import Path

path = Path("06_generation_scale_benchmark_G3/data/g3/g3_manifest.json")
data = json.loads(path.read_text(encoding="utf-8"))
for split in data["splits"]:
    split["file"] = split["file"].replace("\\", "/")
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("fixed", len(data["splits"]), data["splits"][0]["file"])
