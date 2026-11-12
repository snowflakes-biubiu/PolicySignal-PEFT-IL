"""逐段全查 + 复合定稿（默认是 dry-run；必须显式 --run 才会调用 API）。

复核策略（标准优先，其次省 token）：
  1. 主查模型逐段判 keep/domain/confidence，输入 = 标题 + 年代 + 目标域 + WRN + 段落正文片段；
  2. keep=false 直接淘汰（宁可错杀，池有冗余），不花仲裁 token；
  3. keep=true 且 domain 一致且 confidence>=0.80 直接接受；
  4. 中间地带（低置信 / 领域冲突 / 解析失败）交第二模型非思考复核，两模型一致才接受；
  5. 仍不一致的少量争议段交 kimi-k3.1-pro（思考模式）终裁；
  6. 复核完成后按 (域×年代×类) 配额从 keep 段中按 rank 取 5,400 段；不足的格子记入 gaps.json，
     由 build_composite_pool.py --augment-cell 从 28.5 万段未入池候选中补抽、只复核新增段。

用法（准备阶段）：
    python 05_incremental_learning_time/scripts/prepare/verify_composite.py --dry-run
正式执行（需用户确认后）：
    python ... --run
    python ... --compose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))

from openai import OpenAI  # noqa: E402

from classify_domains import BASE_URL  # noqa: E402
from domain_taxonomy_v14 import DOMAINS, RESOLUTION_RULES, TRAIN_DOMAINS  # noqa: E402
from probe_context import load_api_key  # noqa: E402

OUT = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_time" / "data" / "domain_v14" / "composite"
POOL = OUT / "pool.jsonl"
VERDICTS = OUT / "verdicts.jsonl"
BUDGET = OUT / "model_budget.json"

# 每个模型 1M token；主查按序接力，仲裁单独占一个模型额度
PRIMARY = ["deepseek-v4.1-flash", "qwen3.8-max-0902", "kimi-k3", "deepseek-v4-pro-0813"]
ARBITER = "kimi-k3.1-pro"
MODEL_BUDGET = 1_000_000
BATCH = 40
SNIPPET_CHARS = 160
NEED_PER_CLASS = 300

RUBRIC_LITE = """你是政策文件段落复核专家。判断段落能否作为「给定领域」的标准训练样本。
1. 语义完整：段落须自带可识别的政策事项；文种外壳、抬头落款、抄送主题词、纯编号日期、
   「现通知如下/请遵照执行/特此通知/本办法自X日起施行」等无实质内容套话一律不合格。
2. 领域一致：以段落正文的核心事项判定领域，标题只作上下文，不得仅凭标题继承；
   按下方冲突裁决原则判定，结果必须与给定领域一致。
3. 单域纯净：段落明确同时指向两个领域且无主导事项，不合格。
4. 仅有编号无主题的提案答复、截断致主题缺失，若正文同样无主题，不合格。
5. 给定领域可能标错，请独立复核；发文机关不是分类依据。
只输出 JSON：{"items":[{"id":0,"keep":true,"domain":"<领域名或无法归类>","reason":"<=20字","confidence":0.0}]}
""" + "\n冲突裁决原则：\n" + RESOLUTION_RULES + "\n14 个领域：" + "、".join(DOMAINS)


_SYSTEM_CACHE = None


def system_prompt() -> str:
    """把 kimi 思考模式产出的 gold 样本作为 few-shot 注入，统一判定尺度。"""
    global _SYSTEM_CACHE
    if _SYSTEM_CACHE is not None:
        return _SYSTEM_CACHE
    examples = []
    gold = OUT / "pilot_gold.jsonl"
    if gold.is_file():
        rows = [json.loads(l) for l in gold.open(encoding="utf-8") if l.strip()]
        pos = sorted([r for r in rows if r.get("parsed") and r.get("keep")],
                     key=lambda r: -(float(r.get("confidence") or 0)))[:3]
        neg = [r for r in rows if r.get("parsed") and not r.get("keep") and r.get("reason")][:3]
        for r in pos + neg:
            examples.append({"title": r["Title"][:60], "domain": r["domain"],
                             "text": r["snippet"][:120], "keep": r["keep"],
                             "reason": r.get("reason", "")[:20]})
    block = ""
    if examples:
        block = "\n\n判定示例（gold，来自 kimi-k3.1-pro 思考模式）：\n" + json.dumps(
            examples, ensure_ascii=False)
    _SYSTEM_CACHE = RUBRIC_LITE + block
    return _SYSTEM_CACHE


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_items(text: str) -> dict[int, dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json")
    raw = m.group(0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # 模型偶发尾逗号 / 单引号 / 截断，做一次有界修复
        fixed = re.sub(r",\s*([}\]])", r"\1", raw).replace("'", '"')
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            items = re.findall(
                r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"keep"\s*:\s*(true|false)\s*,'
                r'\s*"domain"\s*:\s*"([^"]*)"\s*,\s*"reason"\s*:\s*"[^"]*"\s*,'
                r'\s*"confidence"\s*:\s*([0-9.]+)\s*\}', fixed)
            if not items:
                raise
            return {int(i): {"id": int(i), "keep": k == "true", "domain": d,
                             "reason": "", "confidence": float(c)} for i, k, d, c in items}
    return {int(it["id"]): it for it in obj.get("items", []) if "id" in it}


def append_verdicts(vs: list[dict]) -> None:
    """每批判定立即落盘，进程被中断也不丢已付费 token 的结果。"""
    if not vs:
        return
    with VERDICTS.open("a", encoding="utf-8") as f:
        for v in vs:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
        f.flush()


def budget_state() -> dict:
    if BUDGET.is_file():
        return json.loads(BUDGET.read_text(encoding="utf-8"))
    return {"tokens": {m: 0 for m in PRIMARY + [ARBITER]}, "requests": {}}


def pick_primary(state: dict) -> str | None:
    for m in PRIMARY:
        if state["tokens"].get(m, 0) < MODEL_BUDGET:
            return m
    return None


def verdict_from(item: dict, rec: dict, model: str, rnd: int) -> dict:
    v = item or {}
    return {"pid": rec["pid"], "keep": bool(v.get("keep", False)),
            "verdict_domain": v.get("domain"), "reason": v.get("reason", ""),
            "confidence": v.get("confidence"), "model": model, "round": rnd,
            "parsed": bool(v), "domain": rec["domain"], "era": rec["era"],
            "wrn": rec["wrn"], "rank": rec["rank"]}


def run_round(client, state, records, model, rnd, thinking=False, batch=BATCH,
              max_tokens=4096, sink=None) -> tuple[list[dict], list[dict]]:
    """返回 (已判定, 未处理)。每批立刻 sink 落盘；致命错误时退役并交还剩余段。"""
    from classify_domains import _fatal_reason
    out = []
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        payload = json.dumps([{"id": i, "title": r["Title"], "domain": r["domain"],
                               "wrn": r["wrn"], "era": r["era"],
                               "text": r["snippet"][:SNIPPET_CHARS]}
                              for i, r in enumerate(chunk)], ensure_ascii=False)
        kw = {"extra_body": {"enable_thinking": True if thinking else False}}
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt()},
                          {"role": "user", "content": payload}],
                temperature=0.0, max_tokens=max_tokens, **kw)
            state["tokens"][model] = state["tokens"].get(model, 0) + (resp.usage.total_tokens or 0)
            items = parse_items(resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            fatal, why = _fatal_reason(e)
            if fatal:
                print(f"  [退役] {model}: {why[:120]}", flush=True)
                state["tokens"][model] = MODEL_BUDGET
                BUDGET.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                return out, records[start:]
            print(f"  [{model}] 批次失败 {type(e).__name__}: {str(e)[:100]}", flush=True)
            items = {}
        vs = [verdict_from(items.get(i), rec, model, rnd) for i, rec in enumerate(chunk)]
        out.extend(vs)
        if sink:
            sink(vs)
        tmp = BUDGET.with_suffix(".json.tmp")           # 原子写，防止中断损坏账本
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(BUDGET)
        if (start // batch) % 10 == 0:
            print(f"  [{model}] r{rnd} {start + len(chunk):,}/{len(records):,} "
                  f"tok={state['tokens'].get(model, 0):,}", flush=True)
    BUDGET.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, []


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compose", action="store_true")
    ap.add_argument("--arbitrate-only", action="store_true",
                    help="只跑两模型冲突且尚未终裁的段（断点续跑）")
    ap.add_argument("--arb-batch", type=int, default=20)
    ap.add_argument("--skip-arbitration", action="store_true",
                    help="补抽轮只做一次主查，跳过二次复核与终裁")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    pool = load_jsonl(POOL)
    state = budget_state()
    est_tokens = len(pool) * (SNIPPET_CHARS + 45) + (len(pool) // args.batch) * 900
    print(f"池 {len(pool):,} 段；主查预计 ≈{est_tokens / 1e6:.2f}M token（不含仲裁）")
    print(f"主查接力: {' → '.join(PRIMARY)}，各 {MODEL_BUDGET:,}；终裁: {ARBITER}（思考模式）")
    print(f"累计已用: " + "  ".join(f"{m}={v:,}" for m, v in state["tokens"].items()))
    if args.dry_run or not (args.run or args.compose or args.arbitrate_only):
        print("\n[dry-run] 未调用 API；正式执行需显式 --run / --arbitrate-only / --compose")
        return 0

    client = OpenAI(api_key=load_api_key(), base_url=BASE_URL, timeout=600)
    if args.arbitrate_only:
        by_round = defaultdict(dict)
        for v in load_jsonl(VERDICTS):
            by_round[v["pid"]][v["round"]] = v
        dispute = [r for r in pool
                   if (d := by_round.get(r["pid"])) and 1 in d and 2 in d and 3 not in d
                   and (d[1]["keep"] != d[2]["keep"]
                        or d[1]["verdict_domain"] != d[2]["verdict_domain"])]
        print(f"待终裁 {len(dispute):,} 段（已有 r3 {sum(1 for d in by_round.values() if 3 in d):,}）",
              flush=True)
        if dispute:
            run_round(client, state, dispute, ARBITER, 3, thinking=True,
                      batch=args.arb_batch, max_tokens=16384, sink=append_verdicts)
        return 0

    if args.run:
        done = {v["pid"]: v for v in load_jsonl(VERDICTS)}
        todo = [r for r in pool if r["pid"] not in done]
        print(f"待复核 {len(todo):,} 段（已完成 {len(done):,}）", flush=True)
        model = pick_primary(state)
        while todo and model:
            quota = max(1, int((MODEL_BUDGET - state["tokens"].get(model, 0)) /
                               (SNIPPET_CHARS + 45 + 900 / args.batch)))
            chunk, todo = todo[:quota], todo[quota:]
            vs, unfinished = run_round(client, state, chunk, model, 1,
                                       batch=args.batch, sink=append_verdicts)
            todo = unfinished + todo        # 模型退役时把剩余段交下一个模型
            print(f"  {model}: 本轮 {len(vs):,} 段，累计 {state['tokens'][model]:,} token",
                  flush=True)
            model = pick_primary(state)
        if todo:
            print(f"[警告] 主查模型额度不足，仍有 {len(todo):,} 段未复核", flush=True)
        if args.skip_arbitration:
            print("补抽轮：只做一次主查，跳过二次复核/终裁", flush=True)
            return 0

        # 中间地带交第二模型
        all_v = {v["pid"]: v for v in load_jsonl(VERDICTS)}
        mid = [r for r in pool if (v := all_v.get(r["pid"])) and (
            not v["parsed"] or (v["keep"] and (float(v["confidence"] or 0) < 0.80
                                                or v["verdict_domain"] != r["domain"])))]
        print(f"中间地带 {len(mid):,} 段交第二模型复核", flush=True)
        if mid:
            second = pick_primary(state) or PRIMARY[0]
            run_round(client, state, mid, second, 2, batch=args.batch,
                      sink=append_verdicts)

        # 第三轮：两模型仍冲突 → kimi-k3.1-pro 思考终裁
        by_round = defaultdict(dict)
        for v in load_jsonl(VERDICTS):
            by_round[v["pid"]][v["round"]] = v
        dispute = [r for r in pool
                   if (d := by_round.get(r["pid"])) and 1 in d and 2 in d
                   and (d[1]["keep"] != d[2]["keep"]
                        or d[1]["verdict_domain"] != d[2]["verdict_domain"])]
        print(f"两模型冲突 {len(dispute):,} 段交 {ARBITER} 思考终裁", flush=True)
        if dispute and state["tokens"].get(ARBITER, 0) < MODEL_BUDGET:
            run_round(client, state, dispute, ARBITER, 3, thinking=True,
                      batch=5, max_tokens=16384, sink=append_verdicts)
        return 0

    if args.compose:
        from build_composite_pool import text_key_of  # 段落严格互斥键

        verdicts = load_jsonl(VERDICTS)
        best = {}
        for v in verdicts:
            cur = best.get(v["pid"])
            # 后续轮次只有解析成功才覆盖；解析失败保留前一轮的有效判定
            if v.get("parsed") and (cur is None or v["round"] >= cur["round"]):
                best[v["pid"]] = v
            elif cur is None:
                best[v["pid"]] = v
        # 两模型打架且没有 kimi 终裁结论的争议段：保守淘汰（宁可错杀，池有冗余）
        rounds = defaultdict(dict)
        for v in verdicts:
            rounds[v["pid"]][v["round"]] = v

        def disputed(pid):
            d = rounds.get(pid, {})
            return (1 in d and 2 in d and 3 not in d
                    and (d[1]["keep"] != d[2]["keep"]
                         or d[1]["verdict_domain"] != d[2]["verdict_domain"]))

        accepted = [r for r in pool
                    if not disputed(r["pid"])
                    and (v := best.get(r["pid"])) and v["parsed"] and v["keep"]
                    and float(v["confidence"] or 0) >= 0.80
                    and v["verdict_domain"] == r["domain"]]
        print(f"通过复核 {len(accepted):,} / {len(pool):,} 段"
              f"（争议淘汰 {sum(1 for r in pool if disputed(r['pid'])):,}）", flush=True)

        # A3：段落内容严格互斥（跨文档、跨域、跨 split 全局唯一）；文档不互斥
        seen_tk, by_cell = set(), defaultdict(list)
        for r in sorted(accepted, key=lambda r: r["rank"]):
            tk = r.get("text_key") or text_key_of(r["text"])
            if tk in seen_tk:
                continue
            seen_tk.add(tk)
            r = dict(r)
            r["text_key"] = tk
            by_cell[(r["domain"], r["era"], r["wrn"])].append(r)

        manifest_cells = json.loads((OUT / "manifest.json").read_text(
            encoding="utf-8"))["cells"]
        gaps, final, leftover = [], [], defaultdict(list)
        for c in manifest_cells:                       # 遍历全部格子，零通过也要报缺口
            key = (c["domain"], c["era"], c["wrn"])
            recs = sorted(by_cell.get(key, []), key=lambda r: r["rank"])
            picked = recs[:c["need"]]
            final += picked
            for r in recs[c["need"]:]:                 # 富余段留作跨年代回填
                leftover[(key[0], key[2])].append((key[1], r))
            if len(picked) < c["need"]:
                gaps.append({"domain": key[0], "era": key[1], "wrn": key[2],
                             "have": len(picked), "need": c["need"],
                             "missing": c["need"] - len(picked)})

        # A1/A2：T1 不强制每类 5 段；某格子供给枯竭时，从同(域×WRN)的富余年代回填，
        # 优先补早期年代，保证每(域×WRN) 硬配额 300。段的年代属性不改动。
        era_pref = {e[0]: i for i, e in enumerate(
            [("T1_1949_1978",), ("T2_1979_1992",), ("T3_1993_2002",),
             ("T4_2003_2012",), ("T5_2013_2023",)])}
        used = {r["pid"] for r in final}
        by_dw_now = Counter((r["domain"], r["wrn"]) for r in final)
        for (dom, wrn), have in by_dw_now.items():
            short = NEED_PER_CLASS - have
            if short <= 0:
                continue
            extra = sorted(leftover[(dom, wrn)],
                           key=lambda x: (era_pref.get(x[0], 9), x[1]["rank"]))
            for _, r in extra:
                if short <= 0:
                    break
                if r["pid"] in used:
                    continue
                final.append(r)
                used.add(r["pid"])
                short -= 1
        (OUT / "gaps.json").write_text(json.dumps(gaps, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

        # A2 硬约束：每 (域×WRN) 精确 train 200 / test 100。
        # 先在每个年代内按 2:1 分层（保住 T1~T5 覆盖），再把误差校准回 200/100；
        # 挪动时只取「该年代在源集合里还有 >1 段」的段，避免清空某个年代。
        # A3：不做文档级互斥。
        split_rows = defaultdict(list)
        by_dw = defaultdict(list)
        for r in final:
            by_dw[(r["domain"], r["wrn"])].append(r)
        era_rank = {"T5_2013_2023": 0, "T4_2003_2012": 1, "T3_1993_2002": 2,
                    "T2_1979_1992": 3, "T1_1949_1978": 4}
        sk = lambda r: (era_rank.get(r["era"], 9), r["rank"])
        for (dom, wrn), recs in by_dw.items():
            by_era = defaultdict(list)
            for r in recs:
                by_era[r["era"]].append(r)
            tr, te = [], []
            for era, lst in by_era.items():
                lst.sort(key=lambda r: r["rank"])
                n = round(len(lst) * 2 / 3)
                if len(lst) >= 3:
                    n = min(max(n, 1), len(lst) - 1)
                tr += lst[:n]
                te += lst[n:]
            tr.sort(key=sk)
            te.sort(key=sk)
            while len(tr) > 200:
                cnt = Counter(r["era"] for r in tr)
                i = next((i for i, r in enumerate(tr) if cnt[r["era"]] > 1), 0)
                te.append(tr.pop(i))
            while len(tr) < 200 and te:
                cnt = Counter(r["era"] for r in te)
                i = next((i for i, r in enumerate(te) if cnt[r["era"]] > 1), 0)
                tr.append(te.pop(i))
            split_rows[(dom, "train")] += tr
            split_rows[(dom, "test")] += te

        final_dir = OUT / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        fields = ("pid", "text_key", "doc_id", "Title", "year", "era", "domain", "wrn",
                  "text", "label_model", "label_conf", "tier", "rank")
        for idx, dom in TRAIN_DOMAINS.items():
            for split in ("train", "test"):
                rows = split_rows[(dom, split)]
                with (final_dir / f"D{idx}_{split}.jsonl").open("w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")

        tkeys = [r["text_key"] for r in final]
        per_domain = {dom: {"train": len(split_rows[(dom, "train")]),
                            "test": len(split_rows[(dom, "test")])}
                      for _, dom in TRAIN_DOMAINS.items()}
        per_wrn = dict(Counter(r["wrn"] for r in final))
        dsw = Counter((dom, split, r["wrn"])
                      for (dom, split), rows in split_rows.items() for r in rows)
        split_wrn_ok = all(v == (200 if s == "train" else 100)
                           for (_, s, _), v in dsw.items())
        report = {
            "accepted": len(accepted), "final": len(final), "target": 6 * 3 * NEED_PER_CLASS,
            "by_domain_split_wrn_ok": split_wrn_ok,
            "unique_text_keys": len(set(tkeys)),
            "paragraph_strictly_exclusive": len(set(tkeys)) == len(tkeys),
            "by_domain_split": per_domain,
            "by_wrn": per_wrn,
            "by_era": dict(Counter(r["era"] for r in final)),
            "gaps": gaps,
            "by_domain_wrn": {f"{d}|{w}": c for (d, w), c in
                              sorted(Counter((r["domain"], r["wrn"]) for r in final).items())},
            "pass": (len(final) == 5400 and len(set(tkeys)) == len(tkeys)
                     and split_wrn_ok
                     and all(v["train"] == 600 and v["test"] == 300
                             for v in per_domain.values())
                     and all(c == NEED_PER_CLASS for c in
                             Counter((r["domain"], r["wrn"]) for r in final).values())
                     and len(Counter((r["domain"], r["wrn"]) for r in final)) == 18),
        }
        (OUT / "final_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "gaps"},
                         ensure_ascii=False)[:900], flush=True)
        print(f"PASS={report['pass']}  gaps={len(gaps)}", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
