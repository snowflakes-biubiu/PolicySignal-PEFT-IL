"""全量标注 domain_v14 标题：多模型轮换 + token 预算 + 结果直接入库。

与 classify_domains.py 的差异（都是边跑边入库，区别在批量与 token 效率）：
  * 关闭 thinking（extra_body={"enable_thinking": False}）：实测 completion token 从
    ~8,500/40 条降到 ~700/40 条，是能跑满全量预算的前提；
  * 单请求批量大（默认 400 条，实测 deepseek-v4.1-flash 1600 条仍 ok=true）；
  * 每批成功后立即追加进总库；失败批次**不落库**，标题仍是待处理，下次自动重试；
  * 复用 classify_domains.ModelPool：每个模型 1M token 预算，用尽自动切下一个，
    全部用尽时安全退出（已入库结果不受影响）；
  * 断点续跑：按总库 title_key 跳过已完成标题，可跨进程结转预算；
  * 每批的 token 用量记入 _calibration/model_usage.jsonl，预算可审计。

用法（项目根目录）：
    python 05_incremental_learning_domain/scripts/prepare/annotate_domains.py --dry-run
    # 候选文档优先（47,625 条）
    python 05_incremental_learning_domain/scripts/prepare/annotate_domains.py
    # 扩到全语料唯一标题（33.7 万条），candidates 是它的子集，已入库的会自动跳过
    python 05_incremental_learning_domain/scripts/prepare/annotate_domains.py \
        --input 05_incremental_learning_domain/data/domain/all_unique_titles.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

PREPARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARE_DIR))

from openai import OpenAI  # noqa: E402

from classify_domains import (  # noqa: E402
    BASE_URL,
    DEFAULT_MODELS,
    ROTATION_STATE,
    SYSTEM_PROMPT,
    AllModelsExhausted,
    ModelPool,
    _fatal_reason,
    parse_batch_answer,
    taxonomy_review,
)
from probe_context import load_api_key  # noqa: E402
from title_utils import title_key  # noqa: E402

DATA = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_domain" / "data" / "domain_v14"
CANDIDATES = DATA / "candidates.jsonl"
ALL_TITLES = PREPARE_DIR.parents[2] / "assets" / "05_incremental_learning_domain" / "data" / "domain" / "all_unique_titles.jsonl"
LABELS = DATA / "title_domain_labels_v14.jsonl"
USAGE = DATA / "_calibration" / "model_usage.jsonl"

# 实测 token/条（用于没有 usage 日志时折算历史消耗）
RATE = {"qwen3.8-omni-flash": 58.5, "deepseek-v4.1-flash": 47.4}
DEFAULT_RATE = 60.0
RETRIES = 3

# 有些模型只接受 enable_thinking=True（传 False 会 400），这类模型后续不再传该参数
NO_THINK_FLAG: set[str] = set()
FLAG_LOCK = threading.Lock()


def done_keys(path: Path) -> set[str]:
    """总库中已成功入库的 title_key（带 error 的记录视为未完成）。"""
    if not path.is_file():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    keys.add(row.get("title_key") or title_key(row.get("Title", "")))
    return keys


def load_pending(input_paths: list[Path], label_path: Path, limit: int = 0) -> list[str]:
    """按输入顺序取尚未入库的标题（去重），candidates 在前的顺序会被保留。"""
    done = done_keys(label_path)
    titles: list[str] = []
    seen = set(done)
    for path in input_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                title = str(json.loads(line).get("Title", "")).strip()
                if not title:
                    continue
                key = title_key(title)
                if key in seen:
                    continue
                seen.add(key)
                titles.append(title)
    return titles[:limit] if limit else titles


def commit(label_path: Path, model: str, batch: list[str],
           parsed: dict[int, tuple[str, float]]) -> int:
    """把一批解析结果写进总库，字段与 classify_domains 输出一致。"""
    rows = []
    for i, title in enumerate(batch):
        dom, conf = parsed[i]
        rows.append({"title_key": title_key(title), "Title": title,
                     "domain": dom, "confidence": round(conf, 3),
                     "model": model, **taxonomy_review(title, dom, conf)})
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    return len(rows)


def log_usage(model: str, n: int, tokens: int, committed: int) -> None:
    USAGE.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model,
           "n": n, "tokens": tokens, "committed": committed}
    with USAGE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def prior_spend(label_path: Path) -> tuple[dict[str, float], str]:
    """各模型历史已消耗 token：usage 日志与「入库条数×实测单价」取较大值。

    单看 usage 日志会漏掉早期没记日志的进程；单看条数折算对末尾未提交批次不敏感，
    两者取大即可保证不漏记预算。
    """
    usage: dict[str, float] = {}
    if USAGE.is_file():
        with USAGE.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    usage[r["model"]] = usage.get(r["model"], 0.0) + float(r.get("tokens") or 0)
    counts: dict[str, int] = {}
    if label_path.is_file():
        with label_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                model = r.get("model", "?")
                counts[model] = counts.get(model, 0) + 1
    est = {model: cnt * RATE.get(model, DEFAULT_RATE) for model, cnt in counts.items()}
    models = set(usage) | set(est)
    spend = {m: max(usage.get(m, 0.0), est.get(m, 0.0)) for m in models}
    return spend, "usage 日志 / 入库条数折算取大"


def classify_batch(client, pool: ModelPool, batch: list[str],
                   max_tokens: int,
                   thinking_budget: int = 0) -> tuple[str | None, dict | None, int]:
    """返回 (model, parsed, total_tokens)；parsed 为 None 表示这批失败（不入库）。"""
    payload = json.dumps([{"id": i, "title": t} for i, t in enumerate(batch)],
                         ensure_ascii=False)
    while True:
        model = pool.current()          # 全部停用时抛 AllModelsExhausted
        switched = False
        reconfigure = False
        for attempt in range(RETRIES):
            try:
                kwargs = {"model": model,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": payload}],
                          "temperature": 0.0,
                          "max_tokens": max_tokens,
                          "timeout": 600}
                if thinking_budget > 0:
                    kwargs["extra_body"] = {"enable_thinking": True,
                                            "thinking_budget": thinking_budget}
                elif model not in NO_THINK_FLAG:
                    kwargs["extra_body"] = {"enable_thinking": False}
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    raise ValueError(
                        f"empty content; finish_reason={choice.finish_reason}; "
                        f"completion_tokens={getattr(resp.usage, 'completion_tokens', None)}")
                parsed = parse_batch_answer(content, set(range(len(batch))))
                total = getattr(resp.usage, "total_tokens", 0) or 0
                pool.add_tokens(model, total)
                return model, parsed, total
            except AllModelsExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "enable_thinking" in msg and model not in NO_THINK_FLAG:
                    # 个别模型只接受 True（如 qwen3.8-2.4t-a95b），改为一律不传该参数
                    with FLAG_LOCK:
                        NO_THINK_FLAG.add(model)
                    print(f"    [适配] {model} 不接受 enable_thinking=False，"
                          f"后续请求不再传该参数", flush=True)
                    reconfigure = True
                    break
                fatal, why = _fatal_reason(e)
                if fatal:
                    pool.fail(model, why)
                    switched = True
                    break
                if attempt + 1 >= RETRIES:
                    print(f"    [失败] {model} 批次 {len(batch)} 条："
                          f"{type(e).__name__}: {msg[:120]}", flush=True)
                    return model, None, 0
                time.sleep(min(2 ** attempt, 20))
        if reconfigure or switched:
            continue
        return model, None, 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", default=[CANDIDATES],
                    help="候选标题文件，可给多个，按顺序处理")
    ap.add_argument("--output", type=Path, default=LABELS)
    ap.add_argument("--state", type=Path, default=ROTATION_STATE, help="模型轮换/预算状态文件")
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-output-tokens", type=int, default=32768)
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="限制思考 token（个别模型只接受 enable_thinking=True，如 2.4t）")
    ap.add_argument("--budget", type=int, default=1_000_000, help="每个模型的 token 预算")
    ap.add_argument("--models", default="", help="逗号分隔，默认 5 个模型轮换")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条待入库标题")
    ap.add_argument("--no-seed", action="store_true", help="不结转历史 token 用量")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pending = load_pending(args.input, args.output, args.limit)
    total_done = len(done_keys(args.output))
    print(f"总库已入库 {total_done:,} 条；本次待处理 {len(pending):,} 条；"
          f"批量 {args.batch_size} 并发 {args.workers}", flush=True)
    if args.dry_run:
        n_models = len([m for m in args.models.split(",") if m.strip()]) or len(DEFAULT_MODELS)
        print(f"预计请求 {((len(pending) + args.batch_size - 1) // args.batch_size):,} 次，"
              f"预算 {args.budget:,}×{n_models} = {args.budget * n_models:,} token")
        return 0
    if not pending:
        print("没有待处理标题，收工。")
        return 0

    key = load_api_key()
    if not key:
        print("[卡点] 未找到 DASHSCOPE_API_KEY（环境变量、项目根目录 .env / key.env）")
        return 2

    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    pool = ModelPool(models or DEFAULT_MODELS, state_path=args.state, budget=args.budget)

    if not args.no_seed:
        spend, src = prior_spend(args.output)
        if spend:
            print(f"历史用量结转（{src}）：" +
                  "  ".join(f"{m}={v / 1000:.0f}k" for m, v in spend.items()), flush=True)
            for model, tokens in spend.items():
                if model in pool.state:
                    pool.apply_prior_spend(model, tokens)

    retired = [m for m in pool.models if pool.state[m]["exhausted"]]
    print(f"模型轮换: {' → '.join(pool.models)}  单模型预算 {pool.budget:,} token"
          + (f"（已停用 {retired}）" if retired else ""), flush=True)

    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=600)
    batches = [pending[i:i + args.batch_size]
               for i in range(0, len(pending), args.batch_size)]
    t0 = time.time()
    n_ok = n_lab = n_fail = 0
    stop_reason = ""

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        idx = 0
        pending_futs = {}
        for _ in range(min(args.workers, len(batches))):
            pending_futs[ex.submit(classify_batch, client, pool, batches[idx],
                                   args.max_output_tokens, args.thinking_budget)] = idx
            idx += 1

        while pending_futs:
            done, _ = wait(pending_futs, return_when=FIRST_COMPLETED)
            for fut in done:
                bidx = pending_futs.pop(fut)
                try:
                    model, parsed, tokens = fut.result()
                except AllModelsExhausted as e:
                    stop_reason = f"5 个模型 token 预算全部用尽：{e}"
                    for other in pending_futs:
                        other.cancel()
                    pending_futs.clear()
                    break
                committed = 0
                if parsed is None:
                    n_fail += 1
                else:
                    committed = commit(args.output, model, batches[bidx], parsed)
                    n_lab += committed
                    n_ok += 1
                log_usage(model, len(batches[bidx]), tokens, committed)
                pool.save()
                done_cnt = n_ok + n_fail
                if done_cnt % 5 == 0 or not pending_futs:
                    rate = n_lab / max(time.time() - t0, 1e-6)
                    print(f"  批次 {done_cnt}/{len(batches)}  模型={model}  "
                          f"本批 {len(batches[bidx])} 条  tokens={tokens:,}  "
                          f"入库={n_lab:,}  失败批={n_fail}  {rate:.1f} 条/秒", flush=True)
                if not stop_reason and idx < len(batches):
                    pending_futs[ex.submit(classify_batch, client, pool, batches[idx],
                                           args.max_output_tokens,
                                           args.thinking_budget)] = idx
                    idx += 1

    print(f"\n模型用量: {pool.summary()}", flush=True)
    if stop_reason:
        print(f"已安全停止：{stop_reason}", flush=True)
        print("已入库结果未受影响；重跑同一命令会跳过成功项、继续用剩余预算。", flush=True)

    dist: dict[str, int] = {}
    if args.output.is_file():
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "error" not in row:
                        dist[row.get("domain")] = dist.get(row.get("domain"), 0) + 1
    print("\n领域分布（累计）：")
    for dom, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {dom}: {cnt:,}")
    print(f"\n本轮 入库={n_lab:,} 失败批={n_fail} 用时={(time.time() - t0) / 60:.1f} 分钟")
    return 2 if stop_reason else 0


if __name__ == "__main__":
    sys.exit(main())
