"""逐模型探测「单请求最多能塞多少标题」，同时把模型结果直接入库（测试即生产）。

关键前提：请求带 extra_body={"enable_thinking": False}。实测开启思考时 40 条标题要
消耗 ~8,500 completion token，关闭后仅 ~700，速度也从 107s 降到 8s。

数据策略（不重复烧 token）：
  - 只取 candidates_priority.jsonl 中**尚未入库**的标题（按总库 title_key 去重），
    天然形成「下一波数据」游标，绝不重复处理同一批；
  - 解析成功的结果**立即追加**进总库 title_domain_labels_v14.jsonl，
    token/耗时统计另记 context_probe.jsonl；
  - 请求失败不推进游标，这批标题仍是待处理，下次运行自动重试。

每次请求结束立即落盘（flush），即使中途被杀也不会丢已入库的标签。

用法（项目根目录，一次探一个模型）：
    $env:DASHSCOPE_API_KEY = "sk-..."
    python 05_incremental_learning_time/scripts/prepare/probe_context.py --models deepseek-v4.1-flash --total 800
    python 05_incremental_learning_time/scripts/prepare/probe_context.py --models qwen3.8-omni-flash --start-n 400 --max-n 1600
    python 05_incremental_learning_time/scripts/prepare/probe_context.py --no-commit   # 只看容量不写库
    python 05_incremental_learning_time/scripts/prepare/probe_context.py --report      # 汇总成 md
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
    taxonomy_review,
)
from title_utils import title_key  # noqa: E402

DATA = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_time" / "data" / "domain_v14"
CANDIDATES = DATA / "candidates_priority.jsonl"
LABELS = DATA / "title_domain_labels_v14.jsonl"
OUT_DIR = DATA / "_calibration"
JSONL = OUT_DIR / "context_probe.jsonl"
PROJECT_ROOT = PREPARE_DIR.parents[2]
ENV_FILES = [PROJECT_ROOT / ".env", PROJECT_ROOT / "key.env"]


def load_api_key() -> str:
    """密钥优先取环境变量，其次取项目根目录 .env / key.env（每行 KEY=VALUE）。"""
    key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if key:
        return key
    for env_file in ENV_FILES:
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name.startswith("export "):
                name = name[len("export "):].strip()
            if name == "DASHSCOPE_API_KEY":
                return value.strip().strip('"').strip("'")
    return ""


def _done_keys() -> set[str]:
    """总库中已入库的 title_key（带 error 的记录视为未完成，可重试）。"""
    if not LABELS.is_file():
        return set()
    keys: set[str] = set()
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" not in rec:
                keys.add(rec.get("title_key") or title_key(rec.get("Title", "")))
    return keys


def load_pending_titles(limit: int) -> list[str]:
    """按优先级取尚未入库的标题（批内去重），最多 limit 条。"""
    done = _done_keys()
    titles: list[str] = []
    seen = set(done)
    with CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            title = json.loads(line)["Title"]
            key = title_key(title)
            if key in seen:
                continue
            seen.add(key)
            titles.append(title)
            if len(titles) >= limit:
                break
    return titles


def commit(model: str, titles: list[str],
           parsed: dict[int, tuple[str, float]]) -> int:
    """把一批解析结果直接写入总库，字段与 classify_domains 输出保持一致。"""
    rows = []
    for i, title in enumerate(titles):
        dom, conf = parsed[i]
        rows.append({"title_key": title_key(title), "Title": title,
                     "domain": dom, "confidence": round(conf, 3),
                     "model": model, **taxonomy_review(title, dom, conf)})
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    return len(rows)


def attempt(client, model: str, titles: list[str],
            max_tokens: int, cap: int) -> tuple[dict, dict[int, tuple[str, float]]]:
    """发一次批量请求；返回 (统计记录, 解析结果)。解析结果仅 ok 时可用。"""
    subset = titles[:cap]
    payload = json.dumps([{"id": i, "title": t} for i, t in enumerate(subset)],
                         ensure_ascii=False)
    t0 = time.time()
    parsed: dict[int, tuple[str, float]] = {}
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model,
           "n_titles": cap, "max_tokens": max_tokens, "prompt_chars": len(payload)}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": payload}],
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=600,
            extra_body={"enable_thinking": False},
        )
        rec["elapsed_s"] = round(time.time() - t0, 1)
        usage = getattr(resp, "usage", None)
        rec["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        rec["completion_tokens"] = getattr(usage, "completion_tokens", None)
        rec["total_tokens"] = getattr(usage, "total_tokens", None)
        choice = resp.choices[0]
        rec["finish_reason"] = choice.finish_reason
        content = choice.message.content or ""
        rec["content_chars"] = len(content)
        try:
            parsed = parse_batch_answer(content, set(range(cap)))
            rec["parsed"] = len(parsed)
            rec["ok"] = len(parsed) == cap
            rec["sample"] = [parsed[i][0] for i in range(min(3, cap))]
        except Exception as e:  # noqa: BLE001
            rec["parsed"] = 0
            rec["ok"] = False
            rec["parse_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    except Exception as e:  # noqa: BLE001
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {str(e)[:220]}"
        rec["status"] = getattr(e, "status_code", None)
    if not rec.get("ok"):
        parsed = {}
    return rec, parsed


def append(rec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def report() -> int:
    if not JSONL.is_file():
        print("还没有探测记录")
        return 2
    recs = [json.loads(l) for l in JSONL.open(encoding="utf-8") if l.strip()]
    best: dict[str, dict] = {}
    for rec in recs:
        if rec.get("ok"):
            cur = best.get(rec["model"])
            if cur is None or rec["n_titles"] > cur["n_titles"]:
                best[rec["model"]] = rec
    lines = ["# 单请求上下文容量探测（enable_thinking=false）", "",
             "| 模型 | 最大成功标题数 | prompt | completion | total | 耗时(s) | token/条 | 单模型 1M 可处理条数 |",
             "|---|---|---|---|---|---|---|---|"]
    for model in DEFAULT_MODELS:
        rec = best.get(model)
        if not rec:
            lines.append(f"| {model} | 0 | - | - | - | - | - | 0 |")
            continue
        per = rec["total_tokens"] / rec["n_titles"]
        lines.append(f"| {model} | {rec['n_titles']} | {rec['prompt_tokens']} | "
                     f"{rec['completion_tokens']} | {rec['total_tokens']} | "
                     f"{rec['elapsed_s']} | {per:.1f} | {int(1_000_000 / per):,} |")
    lines.append("")
    total_lab = 0
    dist: dict[str, int] = {}
    if LABELS.is_file():
        with LABELS.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "error" in row:
                        continue
                    total_lab += 1
                    dist[row.get("domain")] = dist.get(row.get("domain"), 0) + 1
    lines.append(f"总库已入库：{total_lab:,} 条（title_domain_labels_v14.jsonl）")
    for dom, cnt in sorted(dist.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"  - {dom}: {cnt:,}")
    lines.append("")
    fails = [(r["model"], r["n_titles"]) for r in recs if not r.get("ok")]
    if fails:
        lines.append("失败尝试（上限参考）：" + ", ".join(f"{m}@{n}" for m, n in fails))
    lines.append("")
    lines.append(f"逐次明细：`context_probe.jsonl`（{len(recs)} 次请求）")
    md = OUT_DIR / "context_probe.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写入 {md}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--start-n", type=int, default=400, help="单请求批量起点")
    ap.add_argument("--max-n", type=int, default=1600, help="单请求批量上限")
    ap.add_argument("--total", type=int, default=0,
                    help="本次最多处理多少条待入库标题（0 = 等于 max-n）")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--target-prompt-tokens", type=int, default=100_000)
    ap.add_argument("--no-commit", action="store_true", help="只测容量，不写总库")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()

    key = load_api_key()
    if not key:
        print(f"[卡点] 未找到 DASHSCOPE_API_KEY：环境变量没有，{ENV_FILE} 也没有")
        return 2
    models = [m.strip() for m in (args.models or ",".join(DEFAULT_MODELS)).split(",") if m.strip()]
    total = args.total or args.max_n
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=600)

    for model in models:
        print(f"\n=== {model} ===", flush=True)
        # 每个模型各自重算游标：已入库的标题绝不会被再次发送
        titles = load_pending_titles(args.max_n if args.no_commit else total)
        print(f"  下一波待入库标题 {len(titles):,} 条；start={args.start_n} max={args.max_n} "
              f"max_tokens={args.max_tokens} commit={'off' if args.no_commit else 'on'}",
              flush=True)
        if not titles:
            print("  没有待处理标题，跳过")
            continue

        n = args.start_n
        max_tokens = args.max_tokens
        offset = 0
        while offset < len(titles):
            cap = min(n, len(titles) - offset, args.max_n)
            if cap <= 0:
                break
            batch = titles[offset:offset + cap]
            rec, parsed = attempt(client, model, batch, max_tokens, cap)
            committed = 0
            if parsed and not args.no_commit:
                committed = commit(model, batch, parsed)
                rec["committed"] = committed
                rec["batch_start"] = batch[0][:60]
                rec["batch_end"] = batch[-1][:60]
            append(rec)
            flag = "OK" if rec.get("ok") else "FAIL"
            print(f"  n={cap:<5} {flag:<4} 入库={committed:<5} "
                  f"prompt={rec.get('prompt_tokens')} "
                  f"completion={rec.get('completion_tokens')} total={rec.get('total_tokens')} "
                  f"{rec.get('elapsed_s')}s fin={rec.get('finish_reason')} "
                  f"{(rec.get('error') or rec.get('parse_error') or '')[:80]}", flush=True)
            if rec.get("ok"):
                offset += cap
                if (rec.get("prompt_tokens") or 0) >= args.target_prompt_tokens:
                    break
                n = min(n * 2, args.max_n)
                continue
            msg = (rec.get("error") or "") + (rec.get("parse_error") or "")
            if "max_tokens" in msg.lower() and max_tokens > 8192:
                max_tokens //= 2
                print(f"    max_tokens 降到 {max_tokens} 重试（不推进游标）", flush=True)
                continue
            break
    return report()


if __name__ == "__main__":
    sys.exit(main())
