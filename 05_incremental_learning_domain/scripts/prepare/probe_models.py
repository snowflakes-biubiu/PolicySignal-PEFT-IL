"""逐个探测分类模型是否可用（鉴权/模型名/配额/输出格式/耗时/token）。

用法（项目根目录）：
    $env:DASHSCOPE_API_KEY = "sk-..."
    python 05_incremental_learning_domain/scripts/prepare/probe_models.py
    python 05_incremental_learning_domain/scripts/prepare/probe_models.py --titles 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))

from openai import OpenAI  # noqa: E402

from classify_domains import (  # noqa: E402
    BASE_URL,
    DEFAULT_MODELS,
    SYSTEM_PROMPT,
    parse_batch_answer,
)

CANDIDATES = (PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_domain" / "data" / "domain_v14" / "candidates.jsonl")


def pick_titles(n: int) -> list[str]:
    titles = []
    if CANDIDATES.is_file():
        with CANDIDATES.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    titles.append(json.loads(line)["Title"])
                if len(titles) >= n:
                    break
    return titles or ["关于加强安全生产工作的通知", "关于做好高校毕业生就业工作的通知"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--titles", type=int, default=2)
    ap.add_argument("--max-output-tokens", type=int, default=512)
    args = ap.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[卡点] 环境变量 DASHSCOPE_API_KEY 未设置")
        return 2

    titles = pick_titles(args.titles)
    payload = json.dumps([{"id": i, "title": t} for i, t in enumerate(titles)], ensure_ascii=False)
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=90)

    print(f"BASE_URL: {BASE_URL}")
    print(f"探测样本 {len(titles)} 条: {titles}\n")
    ok_models = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": payload}],
                temperature=0.0,
                max_tokens=args.max_output_tokens,
                timeout=90,
            )
            cost = time.time() - t0
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            total = getattr(usage, "total_tokens", None)
            parsed = "解析失败"
            try:
                pairs = parse_batch_answer(content, set(range(len(titles))))
                parsed = " | ".join(f"{titles[i][:12]}→{pairs[i][0]}" for i in sorted(pairs))
            except Exception as e:  # noqa: BLE001
                parsed = f"解析失败: {type(e).__name__}: {str(e)[:80]}"
            print(f"[OK] {model:<24} {cost:5.1f}s  tokens={total}  {parsed}")
            if not content.strip():
                print(f"     注意：content 为空（finish_reason={resp.choices[0].finish_reason}）")
            ok_models.append(model)
        except Exception as e:  # noqa: BLE001
            cost = time.time() - t0
            status = getattr(e, "status_code", None)
            code = getattr(e, "code", None)
            print(f"[失败] {model:<22} {cost:5.1f}s  status={status} code={code} "
                  f"{type(e).__name__}: {str(e)[:150]}")
    print(f"\n可用模型 {len(ok_models)}/{len(args.models.split(','))}: {ok_models}")
    return 0 if ok_models else 3


if __name__ == "__main__":
    sys.exit(main())
