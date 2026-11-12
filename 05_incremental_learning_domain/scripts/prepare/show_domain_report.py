# -*- coding: utf-8 -*-
"""查看 domain build_report.json 的 WRN × 年代分布。"""
import json
import sys
from pathlib import Path

rep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for dom, t in rep["tasks"].items():
    print(f"\n=== {dom} ===")
    for split in ("train", "test"):
        grid = t[split]["by_class_era"]
        eras = sorted({e for grid2 in grid.values() for e in grid2})
        print(f"  [{split}] n={t[split]['n']}  类别={t[split]['per_class']}")
        header = "    " + "".join(f"{e:>8}" for e in eras) + "   合计"
        print(header)
        for lab in ("N", "R", "W"):
            row = grid.get(lab, {})
            vals = [row.get(e, 0) for e in eras]
            print("    " + "".join(f"{v:>8}" for v in vals) + f"{sum(vals):>8}   [{lab}]")
