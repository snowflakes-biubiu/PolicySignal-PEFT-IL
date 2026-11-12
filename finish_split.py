"""续跑拆分：处理上次因文件占用中断的剩余项，并保证幂等。"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "05_incremental_learning"
TIME = ROOT / "05_incremental_learning_time"
DOMAIN = ROOT / "05_incremental_learning_domain"
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml",
            ".ps1", ".sh", ".cfg", ".ini"}
MAX_TEXT = 2_000_000
events = []


def log(m):
    print(m, flush=True)


def sig(p: Path):
    if p.is_file():
        return {p.name: p.stat().st_size}
    return {str(x.relative_to(p)): x.stat().st_size for x in p.rglob("*") if x.is_file()}


def safe_remove(p: Path) -> bool:
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return True
    except Exception as e:  # noqa: BLE001
        log(f"  [无法删除] {p.relative_to(ROOT)}: {type(e).__name__}: {str(e)[:90]}")
        return False


def place(src: Path, dst: Path) -> bool:
    if not src.exists():
        log(f"  SKIP 已迁移 {src.relative_to(ROOT)}")
        return True
    if dst.exists():
        a, b = sig(src), sig(dst)
        if a == b:
            log(f"  DEDUP 目标已完整，删除源副本 {src.relative_to(ROOT)}")
            ok = safe_remove(src)
            events.append({"op": "dedup_remove", "path": str(src.relative_to(ROOT)), "ok": ok})
            return ok
        log(f"  [阻断] 源与目标不一致 {src.relative_to(ROOT)}: "
            f"src={len(a)} dst={len(b)}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    events.append({"op": "move", "from": str(src.relative_to(ROOT)),
                   "to": str(dst.relative_to(ROOT))})
    log(f"  MOVE {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")
    return True


def rewrite(base: Path, name: str):
    n = 0
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        try:
            if p.stat().st_size > MAX_TEXT:
                continue
            t = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "05_incremental_learning" in t:
            p.write_text(t.replace("05_incremental_learning", name), encoding="utf-8")
            n += 1
    log(f"  REWRITE {base.name}: {n} 个文件 → {name}")
    events.append({"op": "rewrite", "root": base.name, "files": n, "name": name})


def main() -> int:
    log("=== 续跑拆分 ===")
    ok = True
    log("\n[1] 剩余数据")
    ok &= place(SRC / "data" / "domain_v14", DOMAIN / "data" / "domain_v14")
    ok &= place(SRC / "data" / "domain_incremental", DOMAIN / "data" / "domain_incremental")

    log("\n[2] 代码框架")
    ok &= place(SRC / "scripts", TIME / "scripts")
    if not (DOMAIN / "scripts").exists() and (TIME / "scripts").is_dir():
        shutil.copytree(TIME / "scripts", DOMAIN / "scripts", dirs_exist_ok=True)
        events.append({"op": "copy", "from": "time/scripts", "to": "domain/scripts"})
        log("  COPY time/scripts → domain/scripts")

    log("\n[3] 领域线产物目录")
    for n in ("artifacts", "results", "logs"):
        (DOMAIN / n).mkdir(parents=True, exist_ok=True)

    log("\n[4] 路径引用")
    rewrite(TIME, "05_incremental_learning_time")
    rewrite(DOMAIN, "05_incremental_learning_domain")
    readme = ROOT / "README.md"
    t = readme.read_text(encoding="utf-8")
    if "05_incremental_learning_time" not in t:
        t = t.replace("05_incremental_learning", "05_incremental_learning_time")
        old = ("| `05_incremental_learning_time/` | Temporal/domain incremental-learning "
               "datasets, training, and evaluation |")
        new = ("| `05_incremental_learning_time/` | Temporal incremental learning: T1–T5 datasets, "
               "training, evaluation, artifacts |\n"
               "| `05_incremental_learning_domain/` | Domain incremental learning: D1–D6 datasets, "
               "v14 labels, composite annotation pipeline |")
        t = t.replace(old, new)
        readme.write_text(t, encoding="utf-8")
        log("  REWRITE README.md")
        events.append({"op": "rewrite", "root": "README.md"})

    log("\n[5] 清理源目录")
    leftovers = [p for p in SRC.rglob("*") if p.is_file()] if SRC.exists() else []
    if not leftovers:
        shutil.rmtree(SRC, ignore_errors=True)
        log("  源目录已移除")
    else:
        log(f"  源目录仍有 {len(leftovers)} 个文件（多为被 Excel 打开的核对表），"
            f"暂保留为待清理副本")
        for p in leftovers[:5]:
            log(f"    {p.relative_to(ROOT)}")
    events.append({"op": "source_leftovers", "count": len(leftovers)})

    (ROOT / "split_manifest.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": SRC.name, "targets": [TIME.name, DOMAIN.name],
        "events": events,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n映射清单: split_manifest.json")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
