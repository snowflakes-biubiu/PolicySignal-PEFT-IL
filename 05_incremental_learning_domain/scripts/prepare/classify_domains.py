"""用 LLM 按标题为候选文档打领域标签（domain_labels.jsonl）。

输入:  05_incremental_learning_domain/data/domain/domain_classification_input.jsonl
输出:  05_incremental_learning_domain/data/domain/domain_labels.jsonl

约定与 02_llm_inference 阶段一致：
    - OpenAI 兼容接口（DashScope compatible-mode）
    - 环境变量 DASHSCOPE_API_KEY
    - 默认模型 deepseek-v4-flash-0731，可由环境变量覆盖

特性：唯一标题去重、批量 API 请求、并发调用、指数退避重试、按 title_key 断点续跑。

用法（项目根目录执行）：
    python 05_incremental_learning_domain/scripts/prepare/classify_domains.py
    python 05_incremental_learning_domain/scripts/prepare/classify_domains.py --dry-run
    python 05_incremental_learning_domain/scripts/prepare/classify_domains.py --batch-size 16 --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from openai import OpenAI
from title_utils import title_key

# 14 类领域定义 / 提示词 / 版本号统一由 domain_taxonomy_v14 提供（单一真源，
# 对应 分类详情.md boundary-v4），本文件不再自带领域列表。
from domain_taxonomy_v14 import (  # noqa: E402
    DOMAINS,
    DOMAIN_UNSURE,
    SYSTEM_PROMPT,
    TAXONOMY_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOMAIN_DIR = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "data" / "domain"
INPUT_FILE = DOMAIN_DIR / "all_unique_titles.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "data" / "domain_v14" / "title_domain_labels_v14.jsonl"

MODEL = os.getenv("DOMAIN_CLASSIFICATION_MODEL", "deepseek-v4-flash-0731")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1")
API_KEY_ENV = "DASHSCOPE_API_KEY"
MAX_RETRIES = 4
MAX_CONSECUTIVE_ERROR_BATCHES = 3

# ------------------------------------------------------------------ 模型轮换
# 5 个模型轮流使用，每个预算 1M token；配额耗尽/不可用即自动切下一个。
DEFAULT_MODELS = [
    "qwen3.8-omni-flash",
    "deepseek-v4.1-flash",
    "qwen3.8-max-0902",
    "qwen3.8-27b",
    "qwen3.8-2.4t-a95b",
]
TOKEN_BUDGET = int(os.getenv("DOMAIN_MODEL_TOKEN_BUDGET", "1000000"))
ROTATION_STATE = (PROJECT_ROOT / "assets" / "05_incremental_learning_domain" / "data" / "domain_v14"
                  / "model_rotation_state.json")

FATAL_TERMS = (
    "invalid_api_key", "insufficient_quota", "quota exceeded", "quota_exceeded",
    "insufficient balance", "balance insufficient", "billing hard limit",
    "token balance exhausted", "exceeded your current quota",
    "model not found", "does not exist", "no available channel", "arrearage",
)

POOL = None  # 在 main() 中初始化


class AllModelsExhausted(RuntimeError):
    """所有模型都已停用（配额用尽/不可用）。已完成结果已落盘，可续跑。"""


class ModelPool:
    """按顺序轮换模型，把「停用状态 + token 消耗」落到 state 文件，续跑时跳过已停用模型。"""

    def __init__(self, models: list[str], state_path: Path = ROTATION_STATE,
                 budget: int = TOKEN_BUDGET):
        self.lock = threading.Lock()
        self.models = list(models)
        self.budget = budget
        self.state_path = state_path
        self.state = self._load()
        self.idx = 0
        self._skip_exhausted()

    def _load(self) -> dict:
        base = {m: {"exhausted": False, "tokens": 0, "requests": 0,
                    "ok": 0, "errors": 0, "reason": ""} for m in self.models}
        if self.state_path.is_file():
            try:
                old = json.loads(self.state_path.read_text(encoding="utf-8"))
                for model, rec in (old.get("models") or {}).items():
                    if model in base and isinstance(rec, dict):
                        base[model].update(rec)
            except Exception:  # noqa: BLE001
                pass
        return base

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": self.state, "active": self.current_or_none(),
                   "budget": self.budget}
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _skip_exhausted(self) -> None:
        while self.idx < len(self.models) and self.state[self.models[self.idx]]["exhausted"]:
            self.idx += 1

    def current_or_none(self):
        return self.models[self.idx] if self.idx < len(self.models) else None

    def current(self) -> str:
        with self.lock:
            model = self.current_or_none()
            if model is None:
                raise AllModelsExhausted(self.summary())
            return model

    def add_tokens(self, model: str, tokens: int) -> None:
        with self.lock:
            rec = self.state[model]
            rec["tokens"] += int(tokens or 0)
            rec["requests"] += 1
            rec["ok"] += 1
            if self.budget and rec["tokens"] >= self.budget:
                self._retire(model, f"token 预算用尽（{rec['tokens']:,} ≥ {self.budget:,}）")

    def save(self) -> None:
        """把当前 token 用量落盘（跨进程续跑时预算才不会重算）。"""
        with self.lock:
            self._save()

    def apply_prior_spend(self, model: str, tokens: int) -> None:
        """把历史进程已消耗的 token 计入预算；已达上限则直接退役。"""
        with self.lock:
            rec = self.state[model]
            rec["tokens"] = max(rec["tokens"], int(tokens or 0))
            if self.budget and rec["tokens"] >= self.budget:
                self._retire(model, f"历史用量已达预算（{rec['tokens']:,} ≥ {self.budget:,}）")
            else:
                self._save()

    def fail(self, model: str, reason: str) -> None:
        with self.lock:
            self.state[model]["errors"] += 1
            self._retire(model, reason)

    def _retire(self, model: str, reason: str) -> None:
        rec = self.state[model]
        if rec["exhausted"]:
            return
        rec["exhausted"] = True
        rec["reason"] = reason[:300]
        self.idx = max(self.idx, self.models.index(model) + 1)
        self._skip_exhausted()
        self._save()
        print(f"  [切换] {model} 停用：{reason[:110]} → 下一个 "
              f"{self.current_or_none() or '（无可用模型）'}", flush=True)

    def summary(self) -> str:
        parts = []
        for model in self.models:
            rec = self.state[model]
            flag = "[停用]" if rec["exhausted"] else ""
            parts.append(f"{model}: ok={rec['ok']} err={rec['errors']} "
                         f"tok={rec['tokens']:,}{flag}")
        return " | ".join(parts)


def _fatal_reason(exc: Exception) -> tuple[bool, str]:
    """判断该异常是否「此模型不能再用了」（配额/鉴权/模型不存在）。"""
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    message = str(exc).lower()
    hit = any(term in code or term in message for term in FATAL_TERMS)
    if status in {401, 402, 403, 404} or hit:
        return True, f"{type(exc).__name__}: status={status}, {message[:160]}"
    return False, ""

# 领域列表 / DOMAIN_UNSURE / TAXONOMY_VERSION / SYSTEM_PROMPT 均从
# domain_taxonomy_v14 导入（14 类，boundary-v4），下面保留的 v6 定义仅供历史对照。

_LEGACY_DOMAIN_DEFS_V6 = """- 经济与金融：财政、税收、金融、银行、货币、物价、投资、外贸、国企改革、经济规划、国有资产管理、财政补贴及涉企经济处罚
- 民生与教育：教育、学校、教师、职业培训、医疗、卫生、食品、劳动就业、社会保障、民政、人口、住房、文化、体育、旅游、文物保护、退役军人事务与安置
- 自然与生态环境：环境保护、污染防治、节能减排、气象、水利、防汛抗旱、林业、海洋、地质矿产、土地、核安全、动植物资源保护
- 科技与工业发展：科学技术、研究开发、创新、工业制造、信息化、电子通信、标准化、计量、质量、专利、能源电力、装备制造、农机与工业产品质量、科研平台与中心建设
- 法律与政治监管：法律法规、司法、公安、检察、监察纪检、机构编制、人事干部、宣传意识形态、民族宗教、国家安全、保密档案、信访、行政审批、人大政协议事程序与选举、官方标注的政治法律类文件
- 交通与基础设施：公路铁路民航港口航运等交通、运输、邮政、城市建设、城乡规划、市政基础设施、城镇化、建筑、燃气供热"""

_LEGACY_SYSTEM_PROMPT_V6 = (
    "你是中国中央政府政策文件的领域分类专家。只根据政策文件标题分类，不要推测标题之外的信息。"
    "从以下 6 个领域中选出最贴切的一个。文件形式不是分类依据：批复、公告、通知、名单、人大议案答复等，必须按标题所示的核心主题归类。\n"
    f"领域定义：\n{_LEGACY_DOMAIN_DEFS_V6}\n\n"
    "分类核心规则（必须严格遵守）：\n"
    "1. 【业务优先原则】若标题同时包含‘某业务领域’与‘人事任免/内部资质管理’（如：工程研究中心主任聘任、银行/证券高管任职资格审批），"
    "应优先归入该专业业务所属的专业领域（如科技、经济与金融），而非一律归入法律与政治监管的人事干部类。\n"
    "2. 【噪声与特征码忽略】标题末尾的数据库特征码、抓取标识（如形如 FBM-CLI... 的代码）不影响主题判断，应当忽略。\n"
    "3. 【省略号标题处理】若标题带有省略号，若前半部分已包含足够识别的政策主题，则按主题正常分类；若因截断导致核心业务主题完全缺失、无法辨别，则归为「无法归类」。\n"
    "4. 【提案答复特殊规则】人大代表/政协委员提案建议答复若标题仅有编号而无具体主题内容的，归为「无法归类」；若标题明确标注了官方业务大类（如‘政治法律类’），可作为分类参考归入对应领域。\n"
    "5. 专项细则：人大会议议事/表决制度归法律与政治监管；职业培训、教育培训归民生与教育；金融机构设立与资本事项归经济与金融。\n"
    "6. 仅当标题完全没有可识别的政策/业务主题，或主题明确不属于这六类时，才用「无法归类」。\n"
    "标题可能带有发文机关前缀，请按核心主题判断。每条输入都返回一个结果并保持 id 不变。\n"
    '只输出 JSON：{"items":[{"id":0,"domain":"<领域名或无法归类>","confidence":0.0}]}'
)


def build_user_prompt(rec: dict) -> str:
    return f"标题：{rec['Title']}"


def parse_answer(text: str):
    """从模型输出中解析 JSON，返回 (domain, confidence)。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"输出中无 JSON: {text[:120]!r}")
    obj = json.loads(m.group(0))
    dom = str(obj.get("domain", "")).strip()
    if dom not in DOMAINS and dom != DOMAIN_UNSURE:
        # 容错：模型可能输出相近表述
        for d in DOMAINS:
            if d in dom:
                dom = d
                break
        else:
            dom = DOMAIN_UNSURE
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return dom, min(max(conf, 0.0), 1.0)


def classify_one(client, rec: dict) -> dict:
    user = build_user_prompt(rec)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=64,
                timeout=60,
            )
            dom, conf = parse_answer(resp.choices[0].message.content or "")
            return {"title_key": title_key(rec["Title"]),
                    "Title": rec["Title"], "domain": dom,
                    "confidence": round(conf, 3), "model": MODEL}
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    return {"title_key": title_key(rec["Title"]),
            "Title": rec["Title"], "domain": DOMAIN_UNSURE,
            "confidence": 0.0, "model": MODEL,
            "error": f"{type(last_err).__name__}: {last_err}"}


def parse_batch_answer(text: str, expected_ids: set[int]) -> dict[int, tuple[str, float]]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"输出中无 JSON: {text[:120]!r}")
    payload = json.loads(match.group(0))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("JSON 缺少 items 数组")
    parsed = {}
    for item in items:
        try:
            item_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if item_id in expected_ids:
            parsed[item_id] = parse_domain_confidence(item)
    if set(parsed) != expected_ids:
        raise ValueError(f"返回条目不完整: expected={sorted(expected_ids)}, got={sorted(parsed)}")
    return parsed


def parse_domain_confidence(obj: dict) -> tuple[str, float]:
    dom = str(obj.get("domain", "")).strip()
    if dom not in DOMAINS and dom != DOMAIN_UNSURE:
        dom = next((domain for domain in DOMAINS if domain in dom), DOMAIN_UNSURE)
    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return dom, min(max(confidence, 0.0), 1.0)


def taxonomy_review(title: str, domain: str, confidence: float) -> dict:
    """标注需要人工复核的条目。

    v14 起不再做「标题关键词 → 改判领域」的硬编码规则（旧版那些规则绑定 v6 类名，
    换 taxonomy 后会产出非法标签），只按置信度与标题形态打复核标记。
    """
    note = ""
    manual_review = confidence < 0.75 or domain == DOMAIN_UNSURE

    if "..." in title or "…" in title:
        manual_review = True
        note = "来源标题含截断标记（S8）：核心主题可能缺失，需人工确认。"
    elif ("建议" in title or "提案" in title) and "答复" in title:
        manual_review = True
        note = "提案/建议答复类（S7）：若标题仅有编号无主题，应为「无法归类」。"
    elif confidence < 0.75 and domain != DOMAIN_UNSURE:
        note = "模型置信度偏低，建议人工核对领域边界。"

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "manual_review": manual_review,
        "suggested_domain": domain,
        "review_note": note,
    }


def classify_batch(client, batch: list[dict], max_output_tokens: int = 4096) -> list[dict]:
    """分类一批标题；遇模型配额/鉴权类错误自动换下一个模型重试同一批。"""
    numbered = [{"id": i, "title": rec["Title"]} for i, rec in enumerate(batch)]
    expected_ids = set(range(len(batch)))
    prompt = json.dumps(numbered, ensure_ascii=False)

    while True:
        model = POOL.current()          # 全部停用时抛 AllModelsExhausted
        switched = False
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_output_tokens,
                    timeout=90,
                )
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    usage = resp.usage
                    details = getattr(usage, "completion_tokens_details", None)
                    raise ValueError(
                        "empty message.content; "
                        f"finish_reason={choice.finish_reason}; "
                        f"completion_tokens={getattr(usage, 'completion_tokens', None)}; "
                        f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)}"
                    )
                results = parse_batch_answer(content, expected_ids)
                usage = getattr(resp, "usage", None)
                POOL.add_tokens(model, getattr(usage, "total_tokens", 0) or 0)
                return [
                    {"title_key": title_key(rec["Title"]), "Title": rec["Title"],
                     "domain": results[i][0], "confidence": round(results[i][1], 3),
                     "model": model,
                     **taxonomy_review(rec["Title"], results[i][0], results[i][1])}
                    for i, rec in enumerate(batch)
                ]
            except AllModelsExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                fatal, why = _fatal_reason(e)
                if fatal:
                    POOL.fail(model, why)
                    switched = True
                    break
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
        if switched:
            continue                    # 换模型重试同一批
        break                           # 非致命错误重试用尽 → 记 error 待续跑

    status = getattr(last_err, "status_code", None)
    code = str(getattr(last_err, "code", "") or "")
    error = f"{type(last_err).__name__}: status={status}, code={code or 'n/a'}"
    return [
        {"title_key": title_key(rec["Title"]), "Title": rec["Title"],
         "domain": DOMAIN_UNSURE, "confidence": 0.0, "model": model,
         **taxonomy_review(rec["Title"], DOMAIN_UNSURE, 0.0),
         "error": error, "error_status": status, "error_code": code or None}
        for rec in batch
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="并发批次数")
    ap.add_argument("--batch-size", type=int, default=12, help="每次 API 请求的标题数")
    ap.add_argument("--max-output-tokens", type=int, default=4096,
                    help="每次 API 请求的最大输出 token 数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部）")
    ap.add_argument("--dry-run", action="store_true", help="只统计批次数并展示样例，不调用 API")
    ap.add_argument("--input", type=Path, default=INPUT_FILE)
    ap.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = ap.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.max_output_tokens < 1:
        ap.error("--batch-size、--workers 和 --max-output-tokens 必须大于 0")

    source_recs = [json.loads(l) for l in
                   args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    unique_recs = {}
    for rec in source_recs:
        unique_recs.setdefault(title_key(str(rec.get("Title", ""))), rec)
    recs = list(unique_recs.values())
    print(f"唯一标题: {len(recs):,}/{len(source_recs):,}  并发: {args.workers}")
    if args.dry_run:
        count = min(len(recs), args.limit) if args.limit else len(recs)
        print(f"计划分类: {count:,} 条，批大小: {args.batch_size}，预计 API 请求: "
              f"{(count + args.batch_size - 1) // args.batch_size:,}")
        for rec in recs[:5]:
            print(f"  {rec.get('Title', '')}")
        return 0

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"环境变量 {API_KEY_ENV} 不存在。")

    done: set[str] = set()
    if args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    if "error" not in result:
                        done.add(result.get("title_key") or title_key(result.get("Title", "")))
        print(f"断点续跑: 已完成 {len(done):,} 条，跳过")
    todo = [r for r in recs if title_key(str(r.get("Title", ""))) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"待处理: {len(todo):,} 条")

    if not todo:
        print("没有待处理条目，退出。")
        return 0

    global POOL
    models = [m.strip() for m in os.getenv("DOMAIN_CLASSIFICATION_MODELS", "").split(",")
              if m.strip()]
    POOL = ModelPool(models or DEFAULT_MODELS)
    print(f"模型轮换: {' → '.join(POOL.models)}  单模型预算 {POOL.budget:,} token")
    retired = [m for m in POOL.models if POOL.state[m]["exhausted"]]
    if retired:
        print(f"  续跑：已停用 {retired}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60)
    n_ok = n_err = 0
    t0 = time.time()
    total_batches = math.ceil(len(todo) / args.batch_size)
    batch_iter = iter(todo[i:i + args.batch_size]
                      for i in range(0, len(todo), args.batch_size))
    max_pending = max(args.workers, args.workers * 2)
    pending = {}
    completed_batches = 0
    consecutive_error_batches = 0
    stop_reason = None

    with open(args.output, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            def submit_next():
                if stop_reason:
                    return False
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    return False
                future = ex.submit(classify_batch, client, batch, args.max_output_tokens)
                pending[future] = batch
                return True

            for _ in range(min(max_pending, total_batches)):
                submit_next()

            while pending:
                done_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    batch = pending.pop(future)
                    try:
                        outputs = future.result()
                    except AllModelsExhausted:
                        stop_reason = "5 个模型全部停用（配额/token 用尽），已完成结果已落盘。"
                        for pending_future in list(pending):
                            pending_future.cancel()
                        pending.clear()
                        break
                    for out in outputs:
                        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_ok += sum("error" not in out for out in outputs)
                    n_err += sum("error" in out for out in outputs)
                    completed_batches += 1
                    if all("error" in out for out in outputs):
                        consecutive_error_batches += 1
                        if consecutive_error_batches >= MAX_CONSECUTIVE_ERROR_BATCHES:
                            stop_reason = (
                                f"连续 {MAX_CONSECUTIVE_ERROR_BATCHES} 批全部失败，已停止提交新批次。"
                            )
                    else:
                        consecutive_error_batches = 0
                    if completed_batches % 25 == 0 or completed_batches == total_batches:
                        completed = n_ok + n_err
                        rate = completed / max(time.time() - t0, 1e-6)
                        eta = (len(todo) - completed) / max(rate, 1e-6) / 60
                        print(f"  批次 {completed_batches:,}/{total_batches:,}  "
                              f"ok={n_ok:,} err={n_err:,}  "
                              f"{rate:.1f} 条/秒  剩余约 {eta:.0f} 分钟", flush=True)
                    if not stop_reason:
                        submit_next()

    if POOL is not None:
        print(f"\n模型用量: {POOL.summary()}", flush=True)
    if stop_reason:
        print(f"\n任务已安全暂停：{stop_reason}", flush=True)
        print("已完成记录已落盘；重新运行同一命令会跳过成功项并重试 error 项。", flush=True)

    # 分布统计
    dist: dict[str, int] = {}
    if args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line).get("domain")
                    dist[d] = dist.get(d, 0) + 1
    print("\n领域标签分布（累计输出文件）：")
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")
    print(f"\n完成: ok={n_ok:,} err={n_err}  输出: {args.output}")
    return 2 if stop_reason else 0


if __name__ == "__main__":
    sys.exit(main())
