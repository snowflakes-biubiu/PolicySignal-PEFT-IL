"""探测各模型是否支持关闭思考模式（可大幅降低 completion token）。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))

from openai import OpenAI  # noqa: E402

from classify_domains import BASE_URL, SYSTEM_PROMPT, parse_batch_answer  # noqa: E402

DATA = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_domain" / "data" / "domain_v14"
CANDIDATES = DATA / "candidates_priority.jsonl"

VARIANTS = [
    ("baseline", {}),
    ("enable_thinking=false", {"extra_body": {"enable_thinking": False}}),
    ("chat_template_kwargs", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
    ("thinking_budget=0", {"extra_body": {"thinking_budget": 0}}),
]


def titles(n: int) -> list[str]:
    out = []
    with CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)["Title"])
            if len(out) >= n:
                break
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "qwen3.8-omni-flash", "deepseek-v4.1-flash"]
    ts = titles(n)
    payload = json.dumps([{"id": i, "title": t} for i, t in enumerate(ts)], ensure_ascii=False)
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL, timeout=600)

    for model in models:
        print(f"\n=== {model}  n={n} ===")
        for name, kw in VARIANTS:
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": payload}],
                    temperature=0.0, max_tokens=8192, timeout=600, **kw)
                u = resp.usage
                content = resp.choices[0].message.content or ""
                ok = "解析失败"
                try:
                    ok = f"解析 {len(parse_batch_answer(content, set(range(n))))}/{n}"
                except Exception as e:  # noqa: BLE001
                    ok = f"解析失败({type(e).__name__})"
                print(f"  {name:<26} prompt={u.prompt_tokens} completion={u.completion_tokens} "
                      f"total={u.total_tokens} {time.time()-t0:5.1f}s  {ok}")
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<26} 失败 {type(e).__name__}: {str(e)[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
