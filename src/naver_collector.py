# src/naver_collector.py
from __future__ import annotations

import os
import re
import time
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import requests
from dateutil import parser as dtparser
from pydantic import BaseModel, Field

KST = ZoneInfo("Asia/Seoul")
NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"

DEBUG = os.getenv("NAVER_DEBUG", "0") == "1"

# fetch control
FETCH_N_DEFAULT = int(os.getenv("NAVER_FETCH_N", "150"))
TIME_WINDOW_HOURS = int(os.getenv("NAVER_WINDOW_HOURS", "24"))

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))  # one retry at most by default
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
BATTERY_RELEVANCE_MIN = int(os.getenv("BATTERY_RELEVANCE_MIN", "60"))

# one-shot size control (try 1 call; if too large you can lower this)
LLM_ONESHOT_MAX = int(os.getenv("LLM_ONESHOT_MAX", "150"))  # try to keep 150 = one request
LLM_ALLOW_CHUNK_FALLBACK = os.getenv("LLM_ALLOW_CHUNK_FALLBACK", "1") == "1"
LLM_CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", "80"))  # only used if one-shot fails

def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    low = n.lower().strip()
    low = low.replace("google.", "")
    low = low.replace("light", "lite")
    low = low.replace("_", "-").replace(" ", "-")
    # common variants
    low = low.replace("gemini-2.5-flash-lite", "gemini-2.5-flash-lite")
    low = low.replace("gemini-2.5-flash-lie", "gemini-2.5-flash-lite")
    return low

def get_models() -> Dict[str, str]:
    """
    Centralized model selection inside naver_collector.
    Priority:
      GEMINI_MODEL (base)
      GEMINI_MODEL_DEDUPE
      GEMINI_MODEL_RANK
    """
    base = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    dedupe = _normalize_model_name(os.getenv("GEMINI_MODEL_DEDUPE", base))
    rank = _normalize_model_name(os.getenv("GEMINI_MODEL_RANK", base))
    return {"base": base, "dedupe": dedupe, "rank": rank}

def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    s = s.replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()

def _parse_pubdate_kst(pub: str) -> Optional[datetime]:
    try:
        dt = dtparser.parse(pub)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(KST)
    except Exception:
        return None

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""

def _clean_title_tail_publisher(title: str) -> Tuple[str, Optional[str]]:
    # "제목 - 언론사" 형태가 흔함
    if " - " not in title:
        return title.strip(), None
    base, tail = title.rsplit(" - ", 1)
    base = base.strip()
    tail = tail.strip()
    if base and 2 <= len(tail) <= 60:
        return base, tail
    return title.strip(), None

def _sleep_backoff(attempt: int) -> None:
    t = min(LLM_BACKOFF_MAX, (1.7 ** (attempt + 1)))
    t = t * (0.75 + random.random() * 0.5)
    time.sleep(t)

@dataclass
class NaverNewsItem:
    title: str
    description: str
    link: str
    source: str
    published_dt_kst: datetime
    rank: int  # API 응답 순서(작을수록 상단)

# -----------------------------
# 1) Collect last 24h with multi-queries until max_fetch
# -----------------------------
def collect_naver_last24h_multiquery(
    *,
    client_id: str,
    client_secret: str,
    queries: List[str],
    max_fetch: int = 150,
    sort: str = "date",
    timeout_sec: int = 20,
) -> List[NaverNewsItem]:
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    now = datetime.now(tz=KST)
    cutoff = now - timedelta(hours=TIME_WINDOW_HOURS)

    out: List[NaverNewsItem] = []
    seen_links: set[str] = set()
    rank_counter = 0

    for q in queries:
        if len(out) >= max_fetch:
            break

        start = 1
        while start <= 1000 and len(out) < max_fetch:
            display = min(100, max_fetch - len(out))
            params = {"query": q, "display": display, "start": start, "sort": sort}

            r = requests.get(NAVER_NEWS_ENDPOINT, headers=headers, params=params, timeout=timeout_sec)
            if r.status_code >= 400:
                raise RuntimeError(f"Naver API HTTP {r.status_code}: {r.text[:400]}")

            data = r.json()
            items = data.get("items") or []
            if not items:
                break

            reached_older = False
            for it in items:
                pub_dt = _parse_pubdate_kst(it.get("pubDate", ""))
                if not pub_dt:
                    continue
                if pub_dt < cutoff:
                    reached_older = True
                    continue

                title_raw = _strip_html(it.get("title", ""))
                desc = _strip_html(it.get("description", ""))
                origin = (it.get("originallink") or it.get("link") or "").strip()
                if not origin:
                    continue
                if origin in seen_links:
                    continue
                seen_links.add(origin)

                title, tail_pub = _clean_title_tail_publisher(title_raw)
                src = tail_pub or _domain(origin) or "NAVER"

                out.append(
                    NaverNewsItem(
                        title=title,
                        description=desc,
                        link=origin,
                        source=src,
                        published_dt_kst=pub_dt,
                        rank=rank_counter,
                    )
                )
                rank_counter += 1
                if len(out) >= max_fetch:
                    break

            if sort == "date" and reached_older:
                break
            start += 100

    return out

# -----------------------------
# 2) One-shot LLM scoring: event_key + battery relevance + monitoring importance
# -----------------------------
class OneShotScore(BaseModel):
    index: int
    event_key: str = Field(..., description="same event => same key, ASCII letters/digits/_")
    battery_relevance: int = Field(..., ge=0, le=100)
    monitoring_importance: int = Field(..., ge=0, le=100)

class OneShotResp(BaseModel):
    items: List[OneShotScore]

SYSTEM_ONESHOT = (
    "당신은 '배터리 산업 모니터링' 담당 편집자입니다.\n"
    "제목만 보고 다음 3가지를 판단합니다:\n"
    "1) event_key: 같은 사건/이슈면 같은 키(ASCII letters/digits/_). 단순 유사 주제는 묶지 말 것.\n"
    "2) battery_relevance(0~100): 배터리 산업과 직접 연관성.\n"
    "3) monitoring_importance(0~100): 산업 모니터링 관점 파급력.\n"
)

def _monitor_score(rel: int, imp: int) -> float:
    return 0.7 * float(rel) + 0.3 * float(imp)

def _call_llm_oneshot_scores(
    *,
    titles: List[str],
    model: str,
) -> OneShotResp:
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "아래는 뉴스 제목 목록입니다. 각 제목에 대해 event_key, battery_relevance, monitoring_importance를 산출하세요.\n"
        "반드시 JSON만 출력하고, 모든 index(0..N-1)을 1개씩 포함하세요.\n\n"
        "출력 스키마:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\"index\": 0, \"event_key\": \"...\", \"battery_relevance\": 0-100, \"monitoring_importance\": 0-100}\n"
        "  ]\n"
        "}\n\n"
        "[제목 목록]\n"
        + "\n".join([f"{i}: {t}" for i, t in enumerate(titles)])
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 0.1,
            "system_instruction": SYSTEM_ONESHOT,
            "response_mime_type": "application/json",
            "response_schema": OneShotResp,
        },
    )
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        parsed = OneShotResp.model_validate_json(resp.text)
    return parsed

def _scores_with_retry(titles: List[str], model: str) -> Optional[OneShotResp]:
    last_err: Optional[Exception] = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            return _call_llm_oneshot_scores(titles=titles, model=model)
        except Exception as e:
            last_err = e
            if attempt < LLM_RETRIES:
                _sleep_backoff(attempt)
            else:
                if DEBUG:
                    print(f"[WARN] oneshot scoring failed: {last_err}")
                return None
    return None

def _fallback_dedupe_by_string(items: List[NaverNewsItem]) -> Tuple[List[NaverNewsItem], int]:
    # very cheap fallback (doesn't need extra module)
    def norm(t: str) -> str:
        t = (t or "").lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^0-9a-z가-힣\s]", "", t)
        return t.strip()

    def ngrams(s: str, n=3):
        s = s.replace(" ", "")
        return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else {s}

    def sim(a: str, b: str) -> float:
        A = ngrams(norm(a))
        B = ngrams(norm(b))
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    th = float(os.getenv("NAVER_DEDUPE_THRESHOLD", "0.82"))
    kept: List[NaverNewsItem] = []
    kept_titles: List[str] = []
    for it in items:
        if any(sim(it.title, t) >= th for t in kept_titles):
            continue
        kept.append(it)
        kept_titles.append(it.title)
    return kept, (len(items) - len(kept))

def dedupe_and_rank_by_llm_one_shot(
    items: List[NaverNewsItem],
    *,
    top_k: int,
    models: Dict[str, str],
) -> Tuple[List[NaverNewsItem], List[int], Dict[str, Any]]:
    """
    ✅ 핵심: 1회 LLM 호출(one-shot)로 dedupe(event_key) + scoring(rel/imp)까지 수행
    - event_key별 1개 대표만 남기고
    - monitor_score 상위 top_k 반환
    """
    if not items:
        return [], [], {"raw_count": 0, "deduped_count": 0, "dropped": 0, "picked": 0, "mode": "empty"}

    # make titles list (limit to keep one-shot)
    limited_items = items[:LLM_ONESHOT_MAX]
    titles = [it.title for it in limited_items]

    if not GEMINI_API_KEY:
        deduped, dropped = _fallback_dedupe_by_string(items)
        picked = deduped[:top_k]
        scores = [50 for _ in picked]
        stats = {
            "raw_count": len(items),
            "deduped_count": len(deduped),
            "dropped": dropped,
            "picked": len(picked),
            "models": models,
            "mode": "fallback_no_llm",
        }
        return picked, scores, stats

    # 1) try one-shot
    parsed = _scores_with_retry(titles=titles, model=models["rank"])
    calls = 1 if parsed is not None else 0

    # 2) if failed and allowed, chunk fallback (still LLM-based but >1 calls)
    if parsed is None and LLM_ALLOW_CHUNK_FALLBACK:
        if DEBUG:
            print("[WARN] oneshot failed; trying chunk fallback ...")
        merged_items: List[OneShotScore] = []
        chunk_calls = 0
        for st in range(0, len(titles), LLM_CHUNK_SIZE):
            ch_titles = titles[st: st + LLM_CHUNK_SIZE]
            ch_parsed = _scores_with_retry(titles=ch_titles, model=models["rank"])
            chunk_calls += 1
            if ch_parsed is None:
                merged_items = []
                break
            # offset indices
            for x in ch_parsed.items:
                merged_items.append(
                    OneShotScore(
                        index=int(x.index) + st,
                        event_key=x.event_key,
                        battery_relevance=int(x.battery_relevance),
                        monitoring_importance=int(x.monitoring_importance),
                    )
                )
        if merged_items:
            parsed = OneShotResp(items=merged_items)
            calls = chunk_calls

    # 3) if still failed -> string fallback
    if parsed is None:
        deduped, dropped = _fallback_dedupe_by_string(items)
        picked = deduped[:top_k]
        scores = [50 for _ in picked]
        stats = {
            "raw_count": len(items),
            "deduped_count": len(deduped),
            "dropped": dropped,
            "picked": len(picked),
            "models": models,
            "mode": "fallback_string",
            "llm_calls": calls,
        }
        return picked, scores, stats

    # build maps
    ek_by_i: Dict[int, str] = {}
    rel_by_i: Dict[int, int] = {}
    imp_by_i: Dict[int, int] = {}

    for x in parsed.items:
        i = int(x.index)
        ek = (x.event_key or f"item_{i}").strip()
        ek = re.sub(r"[^0-9A-Za-z_]", "_", ek)[:80] or f"item_{i}"
        ek_by_i[i] = ek
        rel_by_i[i] = int(x.battery_relevance)
        imp_by_i[i] = int(x.monitoring_importance)

    # group by event_key -> choose representative
    by_event: Dict[str, List[int]] = {}
    for i in range(len(limited_items)):
        by_event.setdefault(ek_by_i.get(i, f"item_{i}"), []).append(i)

    reps: List[int] = []
    for ek, idxs in by_event.items():
        # 대표 선정: monitor_score 높고, 동점이면 더 상단노출(rank 낮은 것) 우선
        rep = max(
            idxs,
            key=lambda j: (
                _monitor_score(rel_by_i.get(j, 0), imp_by_i.get(j, 0)),
                -limited_items[j].rank,
            ),
        )
        reps.append(rep)

    # battery relevance filter
    reps_filtered = [i for i in reps if rel_by_i.get(i, 0) >= BATTERY_RELEVANCE_MIN]
    reps = reps_filtered or reps

    # sort reps by monitor_score desc, then relevance, importance, rank
    reps.sort(
        key=lambda j: (
            _monitor_score(rel_by_i.get(j, 0), imp_by_i.get(j, 0)),
            rel_by_i.get(j, 0),
            imp_by_i.get(j, 0),
            -limited_items[j].rank,
        ),
        reverse=True,
    )

    picked_idxs = reps[:top_k]
    picked = [limited_items[i] for i in picked_idxs]
    picked_scores = [int(round(_monitor_score(rel_by_i.get(i, 0), imp_by_i.get(i, 0)))) for i in picked_idxs]

    stats = {
        "raw_count": len(items),
        "deduped_count": len(reps),            # number of event reps before truncation
        "dropped": len(items) - len(reps),
        "picked": len(picked),
        "models": {"base": models["base"], "rank": models["rank"]},
        "mode": "llm_scored",
        "llm_calls": calls,
        "picked_battery_relevance": [rel_by_i.get(i, 0) for i in picked_idxs],
        "picked_monitoring_importance": [imp_by_i.get(i, 0) for i in picked_idxs],
    }
    return picked, picked_scores, stats


# -----------------------------
# Public API used by run.py
# -----------------------------
def collect_naver_top_last24h_deduped_and_ranked(
    *,
    client_id: str,
    client_secret: str,
    queries: List[str],
    fetch_n: int = 150,
    top_k: int = 15,
) -> Tuple[List[NaverNewsItem], List[int], Dict[str, Any]]:
    """
    Returns:
      picked_items: List[NaverNewsItem]
      picked_scores: List[int] (monitor_score 0-100-ish)
      stats: dict
    """
    models = get_models()

    raw = collect_naver_last24h_multiquery(
        client_id=client_id,
        client_secret=client_secret,
        queries=queries,
        max_fetch=fetch_n,
        sort="date",
    )

    # If nothing
    if not raw:
        return [], [], {"raw_count": 0, "deduped_count": 0, "dropped": 0, "picked": 0, "models": models, "mode": "empty"}

    # ✅ one-shot LLM dedupe+rank
    picked, picked_scores, stats = dedupe_and_rank_by_llm_one_shot(
        raw,
        top_k=top_k,
        models=models,
    )

    # attach models to stats
    stats = dict(stats or {})
    stats.setdefault("models", models)
    return picked, picked_scores, stats
