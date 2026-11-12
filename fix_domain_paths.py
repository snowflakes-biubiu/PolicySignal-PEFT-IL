"""让拆分后的 domain 侧代码路径自洽（仅改目录名，不改训练逻辑）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "05_incremental_learning_domain" / "scripts"

EDITS = {
    "incremental_common.py": [
        ('INCREMENTAL_ROOT / "data" / "temporal"', 'INCREMENTAL_ROOT / "data" / "domain_incremental"'),
        ('INCREMENTAL_ROOT / "logs" / "temporal"', 'INCREMENTAL_ROOT / "logs" / "domain_incremental"'),
    ],
    "incremental_evaluation.py": [
        ('"results" / "temporal"', '"results" / "domain_incremental"'),
    ],
    "run_all_continual.py": [
        ('"results" / "temporal"', '"results" / "domain_incremental"'),
    ],
    "run_full_matrix.py": [
        ('"results" / "temporal"', '"results" / "domain_incremental"'),
    ],
    "prepare/validate_domain_tasks.py": [
        ('TEMPORAL = DATA / "temporal"',
         'TEMPORAL = (Path(__file__).resolve().parents[3] / "05_incremental_learning_time"\n'
         '                 / "data" / "temporal")'),
    ],
    "prepare/purge_temporal_overlap.py": [
        ('TEMPORAL = DATA / "temporal"',
         'TEMPORAL = (Path(__file__).resolve().parents[3] / "05_incremental_learning_time"\n'
         '                 / "data" / "temporal")'),
    ],
}

for rel, pairs in EDITS.items():
    p = D / rel
    t = p.read_text(encoding="utf-8")
    n = 0
    for old, new in pairs:
        if old in t:
            t = t.replace(old, new)
            n += 1
    if n:
        p.write_text(t, encoding="utf-8")
    print(f"{rel}: {n} 处")
