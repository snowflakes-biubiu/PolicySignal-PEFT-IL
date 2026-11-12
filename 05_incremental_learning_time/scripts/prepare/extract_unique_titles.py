"""Stream the full corpus CSV and export one record per normalized title."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from title_utils import normalize_title, title_key

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "assets" / "shared" / "datasets" / "processed_readable" / "full_corpus__train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "05_incremental_learning_time" / "data" / "domain" / "all_unique_titles.jsonl"


def extract(input_path: Path, output_path: Path, chunksize: int = 100_000) -> tuple[int, int]:
    title_records = {}
    rows_seen = 0
    for chunk in pd.read_csv(
            input_path, usecols=["Title"], dtype={"Title": "string"},
            keep_default_na=False, encoding="utf-8", chunksize=chunksize):
        rows_seen += len(chunk)
        titles = chunk["Title"].astype(str)
        normalized = titles.map(normalize_title)
        valid = normalized.ne("")
        grouped = pd.DataFrame({"normalized": normalized[valid], "title": titles[valid]})
        for value, group in grouped.groupby("normalized", sort=False):
            key = title_key(value)
            if key in title_records:
                title_records[key]["occurrence_count"] += len(group)
            else:
                title_records[key] = {
                    "title_key": key,
                    "Title": str(group["title"].iloc[0]).strip(),
                    "occurrence_count": len(group),
                }
        print(f"已扫描 {rows_seen:,} 行，唯一标题 {len(title_records):,}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as fout:
        for key in sorted(title_records):
            fout.write(json.dumps(title_records[key], ensure_ascii=False) + "\n")
    temp_path.replace(output_path)
    return rows_seen, len(title_records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be greater than zero")
    rows, titles = extract(args.input, args.output, args.chunksize)
    print(f"完成：扫描 {rows:,} 行，导出 {titles:,} 个唯一标题 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())