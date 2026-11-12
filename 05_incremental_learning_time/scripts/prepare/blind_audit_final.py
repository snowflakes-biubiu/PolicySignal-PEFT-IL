"""最终 5,400 段的独立盲审：kimi-k3.1-pro 思考模式，不给原标签。

输出：领域错标率、段落质量不合格率、合并不合格率与 Wilson 95% 区间。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))
from openai import OpenAI  # noqa: E402

from classify_domains import BASE_URL  # noqa: E402
from domain_taxonomy_v14 import DOMAINS, DOMAIN_DEFS, PARSE_STEPS, RESOLUTION_RULES  # noqa: E402
from probe_context import load_api_key  # noqa: E402

OUT = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_time" / "data" / "domain_v14" / "composite"
FINAL = OUT / "final"
MODEL = "kimi-k3.1-pro"

SYSTEM = """你是政策文件段落的独立质检专家。给定「标题 + 段落正文」，独立判断：
1. valid：该段落能否作为标准训练样本。语义须自带可识别的政策事项；文种外壳、抬头落款、
   抄送主题词、纯编号日期、「现通知如下/请遵照执行/本办法自X日起施行」等无实质内容套话，
   以及正文截断无法判断的，一律 valid=false。
2. domain：以段落正文的核心事项判定其所属领域（14 类之一；确实无法判断填「无法归类」）。
   标题只作上下文，发文机关不是分类依据。不要猜测段落之外的信息。
只输出 JSON：{"items":[{"id":0,"valid":true,"domain":"<领域名>","reason":"<=20字"}]}
id 必须与输入一致，不得遗漏。
""" + "\n领域定义：\n" + DOMAIN_DEFS + "\n\n" + PARSE_STEPS + "\n\n" + RESOLUTION_RULES


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def parse_items(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json")
    raw = m.group(0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", raw).replace("'", '"'))
    return {int(it["id"]): it for it in obj.get("items", []) if "id" in it}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--thinking", action="store_true",
                    help="开思考模式（慢约 20 倍；默认关，用同一模型非思考做全样本盲审）")
    args = ap.parse_args()

    rows = []
    for p in sorted(FINAL.glob("D*_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_file"] = p.name
                    rows.append(r)
    rng = random.Random(args.seed)
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["domain"], r["wrn"])].append(r)
    picks = []
    per = max(1, args.sample // len(by_group))
    for key, lst in sorted(by_group.items()):
        rng.shuffle(lst)
        picks += lst[:per]
    rest = [r for r in rows if r["pid"] not in {x["pid"] for x in picks}]
    rng.shuffle(rest)
    picks += rest[:max(0, args.sample - len(picks))]
    print(f"最终集 {len(rows):,} 段；盲审分层抽 {len(picks)} 段", flush=True)

    client = OpenAI(api_key=load_api_key(), base_url=BASE_URL, timeout=300)
    results, used = [], 0
    for start in range(0, len(picks), args.batch):
        chunk = picks[start:start + args.batch]
        payload = json.dumps([{"id": i, "title": r["Title"],
                               "text": r["text"]} for i, r in enumerate(chunk)],
                             ensure_ascii=False)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": payload}],
                temperature=0.0, max_tokens=4096,
                extra_body={"enable_thinking": bool(args.thinking)})
            used += resp.usage.total_tokens or 0
            items = parse_items(resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            print(f"  批次 {start // args.batch + 1} 失败 {type(e).__name__}: {str(e)[:100]}",
                  flush=True)
            items = {}
        for i, r in enumerate(chunk):
            v = items.get(i, {})
            results.append({"pid": r["pid"], "domain": r["domain"], "era": r["era"],
                            "wrn": r["wrn"], "file": r["_file"], "Title": r["Title"],
                            "valid": bool(v.get("valid", False)),
                            "blind_domain": v.get("domain"), "reason": v.get("reason", ""),
                            "parsed": bool(v)})
        if (start // args.batch) % 5 == 0:
            print(f"  批次 {start // args.batch + 1}/"
                  f"{(len(picks) + args.batch - 1) // args.batch} tok={used:,}", flush=True)

    parsed = [r for r in results if r["parsed"]]
    bad_valid = [r for r in parsed if not r["valid"]]
    bad_domain = [r for r in parsed if r["blind_domain"] != r["domain"]]
    bad_any = [r for r in parsed if not r["valid"] or r["blind_domain"] != r["domain"]]
    lo, hi = wilson(len(bad_any), len(parsed))
    report = {
        "model": MODEL, "thinking": bool(args.thinking), "sample": len(picks),
        "parsed": len(parsed),
        "parse_failures": len(results) - len(parsed),
        "invalid_quality": len(bad_valid), "domain_mismatch": len(bad_domain),
        "combined_bad": len(bad_any),
        "observed_error_rate": round(len(bad_any) / max(len(parsed), 1), 4),
        "observed_accuracy": round(1 - len(bad_any) / max(len(parsed), 1), 4),
        "wilson_95_error_low": round(lo, 4), "wilson_95_error_high": round(hi, 4),
        "blind_tokens": used,
        "by_domain_mismatch": dict(Counter(r["domain"] for r in bad_domain)),
        "bad_samples": bad_any[:30],
    }
    (OUT / "blind_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "blind_audit_samples.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n盲审 {len(parsed)} 段：质量不合格 {len(bad_valid)}，领域错标 {len(bad_domain)}，"
          f"合并不合格 {len(bad_any)}", flush=True)
    print(f"实测准确率 {1 - len(bad_any)/max(len(parsed),1):.2%}；"
          f"错误率 95% CI {lo:.2%}~{hi:.2%}；消耗 {used:,} token", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
