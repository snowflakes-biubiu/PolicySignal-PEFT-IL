"""把 05_incremental_learning 拆成 time / domain 两条独立实验线。

时间线：T1~T5 数据集、训练/评估框架、历史结果、模型产物、日志、文档
领域线：D1~D6 数据集、v14 标签、复合标注流水线；代码框架复制一份供后续改造

用法：
    python split_incremental_learning.py --dry-run
    python split_incremental_learning.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "05_incremental_learning"
TIME = ROOT / "05_incremental_learning_time"
DOMAIN = ROOT / "05_incremental_learning_domain"
MANIFEST = ROOT / "split_manifest.json"

TIME_TOP = ["artifacts", "results", "logs", "logs_20261104_bak",
            "continual_learning_plan.md", "实验现状与参数配置总览.md"]
TIME_DATA = ["temporal", "used_documents.json"]
DOMAIN_DATA = ["domain", "domain_v14", "domain_incremental"]
EMPTY_IN_DOMAIN = ["artifacts", "results", "logs"]

TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml",
            ".ps1", ".sh", ".cfg", ".ini"}
MAX_TEXT = 2_000_000


def log(msg: str) -> None:
    print(msg, flush=True)


def plan_check() -> list[str]:
    problems = []
    if not SRC.is_dir():
        problems.append(f"源目录不存在: {SRC}")
    if TIME.exists() or DOMAIN.exists():
        problems.append("目标目录已存在，请先确认是否已拆分过")
    for name in TIME_TOP + TIME_DATA + DOMAIN_DATA:
        p = SRC / name if name not in TIME_DATA + DOMAIN_DATA else SRC / "data" / name
        if not p.exists():
            problems.append(f"缺少待迁移项: {p}")
    return problems


def move(src: Path, dst: Path, events: list, dry: bool) -> None:
    events.append({"op": "move", "from": str(src.relative_to(ROOT)),
                   "to": str(dst.relative_to(ROOT))})
    log(f"  MOVE {src.relative_to(ROOT)}  →  {dst.relative_to(ROOT)}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def copytree(src: Path, dst: Path, events: list, dry: bool) -> None:
    events.append({"op": "copy", "from": str(src.relative_to(ROOT)),
                   "to": str(dst.relative_to(ROOT))})
    log(f"  COPY {src.relative_to(ROOT)}  →  {dst.relative_to(ROOT)}")
    if not dry:
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)


def rewrite_refs(base: Path, new_name: str, events: list, dry: bool) -> int:
    changed = 0
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        try:
            if p.stat().st_size > MAX_TEXT:
                continue
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "05_incremental_learning" not in text:
            continue
        if not dry:
            p.write_text(text.replace("05_incremental_learning", new_name), encoding="utf-8")
        changed += 1
    events.append({"op": "rewrite", "root": str(base.relative_to(ROOT)),
                   "new_name": new_name, "files": changed})
    log(f"  REWRITE {base.name}: {changed} 个文本文件 -> {new_name}")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run
    log(f"=== {'DRY-RUN' if dry else '执行拆分'} ===")
    problems = plan_check()
    if problems:
        for x in problems:
            log("  [阻断] " + x)
        return 2

    events = []
    log("\n[1] 时间线内容与产物")
    for name in TIME_TOP:
        move(SRC / name, TIME / name, events, dry)
    log("\n[2] 数据按主题分类")
    for name in TIME_DATA:
        move(SRC / "data" / name, TIME / "data" / name, events, dry)
    for name in DOMAIN_DATA:
        move(SRC / "data" / name, DOMAIN / "data" / name, events, dry)

    log("\n[3] 代码框架两边各一份")
    move(SRC / "scripts", TIME / "scripts", events, dry)
    copytree(TIME / "scripts", DOMAIN / "scripts", events, dry)

    log("\n[4] 领域线空产物目录")
    for name in EMPTY_IN_DOMAIN:
        if not dry:
            (DOMAIN / name).mkdir(parents=True, exist_ok=True)
        events.append({"op": "mkdir", "path": str((DOMAIN / name).relative_to(ROOT))})
        log(f"  MKDIR {DOMAIN.name}/{name}")

    log("\n[5] 更新路径引用")
    rewrite_refs(TIME, "05_incremental_learning_time", events, dry)
    rewrite_refs(DOMAIN, "05_incremental_learning_domain", events, dry)
    readme = ROOT / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        text = text.replace("05_incremental_learning", "05_incremental_learning_time")
        old = ("| `05_incremental_learning_time/` | Temporal/domain incremental-learning "
               "datasets, training, and evaluation |")
        new = ("| `05_incremental_learning_time/` | Temporal incremental learning: T1–T5 datasets, "
               "training, evaluation, artifacts |\n"
               "| `05_incremental_learning_domain/` | Domain incremental learning: D1–D6 datasets, "
               "v14 labels, composite annotation pipeline |")
        text = text.replace(old, new)
        if not dry:
            readme.write_text(text, encoding="utf-8")
        events.append({"op": "rewrite", "root": "README.md",
                       "new_name": "05_incremental_learning_time"})
        log("  REWRITE README.md")

    log("\n[6] 移除源目录空壳")
    if not dry:
        leftovers = [p for p in SRC.rglob("*") if p.is_file()
                     and not (p.suffix == ".pyc" or "__pycache__" in p.parts)]
        if leftovers:
            log(f"  [阻断] 源目录仍有 {len(leftovers)} 个未分发文件，终止删除：")
            for p in leftovers[:10]:
                log(f"    {p.relative_to(ROOT)}")
            return 3
    events.append({"op": "rmtree", "path": str(SRC.relative_to(ROOT)),
                   "note": "内容已全部分发；split_manifest.json 保留完整映射"})
    if not dry:
        shutil.rmtree(str(SRC))
    log(f"  RMTREE {SRC.name}（仅剩空壳/已复制内容）")

    if not dry:
        MANIFEST.write_text(json.dumps({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": str(SRC.relative_to(ROOT)),
            "targets": [TIME.name, DOMAIN.name],
            "events": events,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\n映射清单: {MANIFEST}")
    log("\n完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
