# -*- coding: utf-8 -*-
"""G3-SSL 静态配置与数据零泄漏/嵌套自检。"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import ssl_common as S  # noqa: E402

CHECKS: list[tuple[bool, str]] = []


def check(cond: bool, msg: str) -> None:
    CHECKS.append((bool(cond), msg))
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def main() -> int:
    cfg = S.read_json(S.SSL_CONFIG)
    check(cfg["task"] == "task2_level2", f"任务锁定为 {cfg['task']}")
    check(int(cfg["gold_per_class"]) == 300, "金标准边界 = 300/类")
    check(all(n > 300 for n in cfg["ssl_tiers"]), f"SSL 档位 {cfg['ssl_tiers']} 严格大于 300")
    check(0 < float(cfg["silver_weight"]) <= 1.0, f"伪标签 Loss 权重 = {cfg['silver_weight']}")
    check(cfg["pool"]["filter_column"] == "R_1" and cfg["pool"]["filter_value"] == "W", "仅用 R_1=='W' 筛选无标注池，不依赖 R_2")

    manifest_path = S.SSL_DATA_ROOT / "g3_ssl_manifest.json"
    if not manifest_path.is_file():
        print("[INFO] 尚未在云端构建 G3-SSL 数据，跳过数据集零泄漏与嵌套校验")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        val_norms = {norm(r["text"]) for r in S.preprocess.load_split(cfg["task"], "validation")}
        prev_texts: set[str] | None = None
        for item in sorted(manifest["splits"], key=lambda x: int(x["per_class"])):
            n = int(item["per_class"])
            p = S.PROJECT_ROOT / item["file"]
            check(p.is_file(), f"n={n}: 文件存在")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            check(digest == item["sha256"], f"n={n}: sha256 与 manifest 一致")
            recs = S.read_jsonl(p)
            pseudo_norms = {norm(r["text"]) for r in recs if r.get("source") != "gold"}
            check(not (pseudo_norms & val_norms), f"n={n}: 伪标签与验证集零重叠")
            all_norms = {norm(r["text"]) for r in recs}
            if prev_texts is not None:
                check(prev_texts <= all_norms, f"n={n}: 严格包含上一档子集")
            prev_texts = all_norms

    failed = [m for ok, m in CHECKS if not ok]
    print(f"\n=== G3-SSL 自检 {len(CHECKS) - len(failed)}/{len(CHECKS)} 通过 ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
