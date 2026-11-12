"""用 kimi-k3.1-pro（思考模式）建立逐段合规标准 + 少量 gold 样本，实测拒绝率。

这一步只做「实验准备」：产出 rubric、pilot_gold、分层拒绝率估计，**不执行全量复核**。

用法：
    python 05_incremental_learning_time/scripts/prepare/build_rubric_pilot.py
    python ... --sample 120 --batch 10
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))

from openai import OpenAI  # noqa: E402

from classify_domains import BASE_URL  # noqa: E402
from domain_taxonomy_v14 import DOMAINS, DOMAIN_DEFS, PARSE_STEPS, RESOLUTION_RULES, SPECIAL_CASES  # noqa: E402
from probe_context import load_api_key  # noqa: E402

DATA = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_time" / "data" / "domain_v14"
OUT_DIR = DATA / "composite"
POOL = OUT_DIR / "pool.jsonl"
MODEL = "kimi-k3.1-pro"

RUBRIC = """你是政策文件段落质量与领域归属的复核专家。给定「标题 + 目标领域 + 段落正文」，判断该段落能否作为该领域的标准训练样本。

判定必须按以下顺序执行，逐条给结论：
1. 语义完整性：段落必须自带可识别的政策事项（有被管理/建设/保护的对象，或具体措施、规则、数据）。以下一律判不合格：
   - 文种外壳/抬头/落款/抄送/主题词/附件/签发/印发信息、纯编号、纯日期、纯文号、联系人；
   - 「现通知如下」「请遵照执行」「特此通知/批复」「本办法自X日起施行」等无实质内容的套话；
   - 纯数字表格、乱码、正文被截断到无法判断。
2. 领域一致性：以「段落正文所示核心事项」为准判定领域，标题只作上下文，不得仅凭标题继承。
   按下面的解析流程与冲突裁决原则判定；结果必须与给定的目标领域完全一致，否则不合格。
3. 单域纯净：段落若明确同时指向两个不同领域且按裁决原则仍无主导事项，判不合格。
4. 特殊情形：仅有编号无主题的人大建议/政协提案答复、省略号导致主题缺失、空标题纯文种，若正文同样无主题，判不合格。
5. 存疑处理：能判定但不太确定，给 keep=true 且 confidence<0.7，交给第二模型仲裁；不要为了省事一律判不合格。

注意：给定领域由上一阶段 LLM 按标题标注，可能出错；你的任务是独立复核，不要默认它正确。
也不要把「发文机关」当分类依据；不要因为段落里出现别的部门名就判跨域。

只输出 JSON：{"items":[{"id":0,"keep":true/false,"domain":"<14类之一或无法归类>","reason":"<=30字","confidence":0.0}]}
id 必须与输入一致，不得遗漏。
"""


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def parse_items(text: str) -> dict[int, dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"无 JSON: {text[:120]!r}")
    obj = json.loads(m.group(0))
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        raise ValueError("缺 items")
    return {int(it["id"]): it for it in items if "id" in it}


def stratified_sample(rows: list[dict], per_domain: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    picks = []
    for dom, items in sorted(by_dom.items()):
        # 优先覆盖 (era, wrn) 组合，再随机补足
        combos = defaultdict(list)
        for r in items:
            combos[(r["era"], r["wrn"])].append(r)
        keys = list(combos)
        rng.shuffle(keys)
        chosen = []
        for key in keys:
            if len(chosen) >= per_domain:
                break
            chosen.append(rng.choice(combos[key]))
        rest = [r for r in items if r["pid"] not in {c["pid"] for c in chosen}]
        rng.shuffle(rest)
        chosen += rest[:max(0, per_domain - len(chosen))]
        picks += chosen[:per_domain]
    return picks


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=90, help="pilot 段数（每域均分）")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="思考模式 reasoning 会占用输出预算，需留足")
    ap.add_argument("--resume", action="store_true", help="只补跑上次解析失败的段")
    args = ap.parse_args()

    rows = [json.loads(l) for l in POOL.open(encoding="utf-8") if l.strip()]
    per_domain = max(1, args.sample // 6)
    sample = stratified_sample(rows, per_domain, args.seed)

    gold_path = OUT_DIR / "pilot_gold.jsonl"
    prior = {}
    used_tokens = 0
    if args.resume and gold_path.is_file():
        for line in gold_path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                prior[r["pid"]] = r
        report_path = OUT_DIR / "pilot_report.json"
        if report_path.is_file():
            used_tokens = int(json.loads(report_path.read_text(encoding="utf-8"))
                              .get("pilot_tokens") or 0)
    todo = [r for r in sample
            if not (prior.get(r["pid"]) or {}).get("parsed")]
    print(f"池 {len(rows):,} 段，pilot 样本 {len(sample)} 段；"
          f"本次待跑 {len(todo)} 段（已完成 {len(sample) - len(todo)}）", flush=True)

    system = (RUBRIC + "\n\n领域定义：\n" + DOMAIN_DEFS + "\n\n" + PARSE_STEPS
              + "\n\n" + RESOLUTION_RULES + "\n\n" + SPECIAL_CASES)
    client = OpenAI(api_key=load_api_key(), base_url=BASE_URL, timeout=300)
    gold = []
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        payload = json.dumps([{"id": i, "title": r["Title"], "domain": r["domain"],
                               "wrn": r["wrn"], "era": r["era"], "text": r["snippet"]}
                              for i, r in enumerate(batch)], ensure_ascii=False)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": payload}],
                temperature=0.0, max_tokens=args.max_tokens,
                extra_body={"enable_thinking": True})
            used_tokens += resp.usage.total_tokens or 0
            choice = resp.choices[0]
            content = choice.message.content or ""
            if not content.strip():
                print(f"  批次 {start // args.batch + 1}: content 为空 "
                      f"fin={choice.finish_reason} comp={resp.usage.completion_tokens}", flush=True)
            items = parse_items(content)
        except Exception as e:  # noqa: BLE001
            print(f"  批次 {start // args.batch + 1} 失败: {type(e).__name__}: {str(e)[:120]}", flush=True)
            items = {}
        for i, r in enumerate(batch):
            v = items.get(i, {})
            gold.append({"pid": r["pid"], "domain": r["domain"], "era": r["era"],
                         "wrn": r["wrn"], "tier": r["tier"], "risk_flags": r["risk_flags"],
                         "Title": r["Title"], "snippet": r["snippet"],
                         "keep": bool(v.get("keep", False)), "verdict_domain": v.get("domain"),
                         "reason": v.get("reason", ""), "confidence": v.get("confidence"),
                         "model": MODEL, "parsed": bool(v)})
        print(f"  批次 {start // args.batch + 1}/{(len(todo) + args.batch - 1) // args.batch} "
              f"tokens={used_tokens:,}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    new_by_pid = {r["pid"]: r for r in gold}
    merged = []
    for r in sample:
        done = prior.get(r["pid"]) if (prior.get(r["pid"]) or {}).get("parsed") else None
        merged.append(done or new_by_pid.get(r["pid"]) or {
            "pid": r["pid"], "domain": r["domain"], "era": r["era"], "wrn": r["wrn"],
            "tier": r["tier"], "risk_flags": r["risk_flags"], "Title": r["Title"],
            "snippet": r["snippet"], "keep": False, "reason": "unparsed",
            "model": MODEL, "parsed": False})
    with gold_path.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gold = merged

    parsed = [r for r in gold if r["parsed"]]
    rejected = [r for r in parsed if not r["keep"]]
    k, n = len(rejected), len(parsed)
    lo, hi = wilson(k, n)

    def rate(sub):
        return (sum(1 for r in sub if not r["keep"]) / len(sub)) if sub else None

    by = {}
    for field in ("domain", "era", "wrn", "tier"):
        d = defaultdict(list)
        for r in parsed:
            d[r[field]].append(r)
        by[field] = {kk: {"n": len(v), "reject": round(rate(v), 3)}
                     for kk, v in sorted(d.items())}
    risky = [r for r in parsed if r["risk_flags"]]
    clean = [r for r in parsed if not r["risk_flags"]]
    by["risk"] = {"has_flags": {"n": len(risky), "reject": round(rate(risky) or 0, 3)},
                  "clean": {"n": len(clean), "reject": round(rate(clean) or 0, 3)}}

    report = {
        "model": MODEL, "thinking": True, "sample_size": len(sample),
        "parsed": n, "parse_failures": len(gold) - n,
        "rejected": k, "reject_rate": round(k / n, 4) if n else None,
        "wilson_95_low": round(lo, 4), "wilson_95_high": round(hi, 4),
        "recommended_oversample_p": round(hi, 4),
        "recommended_pool_size": int(5400 / max(1e-6, 1 - hi)) if hi < 1 else None,
        "current_pool_size": len(rows),
        "pilot_tokens": used_tokens,
        "by": by,
        "top_reasons": Counter(r["reason"] for r in rejected).most_common(15),
    }
    (OUT_DIR / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n拒绝率 {k}/{n} = {k / n:.1%}（95% CI {lo:.1%}~{hi:.1%}）", flush=True)
    print(f"建议按 p={hi:.1%} 放大 → 池需 ≈{report['recommended_pool_size']:,} 段；当前 {len(rows):,} 段", flush=True)
    print(f"分层: {json.dumps(by, ensure_ascii=False)[:900]}", flush=True)
    print(f"pilot 消耗 {used_tokens:,} token；已写 rubric.md / pilot_gold.jsonl / pilot_report.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
