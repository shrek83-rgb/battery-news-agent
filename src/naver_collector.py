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
from google import genai

KST = ZoneInfo("Asia/Seoul")
NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"

# unified model names (single env)
def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    n = n.lower().replace("light", "lite").strip()
    return n

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
BASE_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
RANK_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL_RANK", BASE_MODEL))

DEBUG = os.getenv("NAVER_DEBUG", "0") == "1"

# fetch control
FETCH_N = int(os.getenv("NAVER_FETCH_N", "150"))
TIME_WINDOW_HOURS = int(os.getenv("NAVER_WINDOW_HOURS", "24"))

# LLM retries/backoff
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))

# relevance threshold
BATTERY_RELEVANCE_MIN = int(os.getenv("BATTERY_RELEVANCE_MIN", "60"))


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
    # common: "기사제목 - 언론사"
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
    rank: int  # API response order (smaller is higher)


# -----------------------------
# 1) Collect 24h by multi-query until max_fetch
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
# 2) One-shot LLM: event_key + relevance + importance (single call)
# -----------------------------
class TitleScore(BaseModel):
    index: int
    event_key: str = Field(..., description="Same event => same key (ASCII letters/digits/_)")
    battery_relevance: int = Field(..., ge=0, le=100)
    monitoring_importance: int = Field(..., ge=0, le=100)


class TitleScoreResp(BaseModel):
    items: List[TitleScore]


SYSTEM_SCORE = (
    "당신은 배터리 산업 뉴스 모니터링 담당자입니다.\n"
    "제목만 보고 아래 3가지를 산출합니다.\n"
    "1) event_key: '같은 사건/이슈'면 동일 키. 표현이 달라도 동일 사건이면 동일.\n"
    "   - 단, '주제가 비슷'한 정도는 동일 사건이 아닙니다.\n"
    "   - 키는 ASCII letters/digits/_만 사용.\n"
    "2) battery_relevance(0~100): 배터리 산업과의 직접 연관성.\n"
    "3) monitoring_importance(0~100): 산업 전반 모니터링 중요도(투자/증설/공급계약/정책·규제/관세/공급망/리콜·사고/핵심기술 등).\n"
    "규칙:\n"
    "- 모든 index는 정확히 1번씩 포함되어야 합니다.\n"
    "- JSON만 출력합니다."
)


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return text[s : e + 1]
    return text


def score_titles_one_shot(items: List[NaverNewsItem]) -> Optional[TitleScoreResp]:
    if not GEMINI_API_KEY or not items:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "[제목 목록]\n"
        + "\n".join([f"{i}: {it.title}" for i, it in enumerate(items)])
        + "\n\n"
        + "JSON 스키마:\n"
        + "{\"items\":[{\"index\":0,\"event_key\":\"...\",\"battery_relevance\":0-100,\"monitoring_importance\":0-100}]}\n"
    )

    last_err = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=RANK_MODEL,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_SCORE,
                    "temperature": 0.2,
                },
            )
            raw = _extract_json(resp.text)
            return TitleScoreResp.model_validate_json(raw)
        except Exception as e:
            last_err = e
            if attempt < LLM_RETRIES:
                _sleep_backoff(attempt)

    if DEBUG:
        print(f"[WARN] one-shot scoring failed: {last_err}")
    return None


def collect_naver_top_last24h_deduped_and_ranked(
    *,
    client_id: str,
    client_secret: str,
    queries: List[str],
    fetch_n: int = 150,
    top_k: int = 15,
) -> Tuple[List[NaverNewsItem], List[int], Dict[str, Any]]:
    raw = collect_naver_last24h_multiquery(
        client_id=client_id,
        client_secret=client_secret,
        queries=queries,
        max_fetch=fetch_n,
        sort="date",
    )

    stats: Dict[str, Any] = {"raw_count": len(raw)}

    scored = score_titles_one_shot(raw)
    if scored is None:
        # fallback: no-llm -> keep rank order, no dedupe (but at least return something)
        picked = raw[:top_k]
        scores = [10] * len(picked)
        stats.update(
            {
                "deduped_count": len(raw),
                "dropped": 0,
                "picked": len(picked),
                "mode": "fallback_no_llm",
                "models": {"rank": RANK_MODEL, "base": BASE_MODEL},
            }
        )
        return picked, scores, stats

    # build maps
    rel_by_i: Dict[int, int] = {}
    imp_by_i: Dict[int, int] = {}
    ek_by_i: Dict[int, str] = {}
    for x in scored.items:
        idx = int(x.index)
        ek = (x.event_key or f"item_{idx}").strip()
        # sanitize key
        ek = re.sub(r"[^A-Za-z0-9_]+", "_", ek)[:80] or f"item_{idx}"
        ek_by_i[idx] = ek
        rel_by_i[idx] = int(x.battery_relevance)
        imp_by_i[idx] = int(x.monitoring_importance)

    # group by event_key and pick representative per event
    by_event: Dict[str, List[int]] = {}
    for i in range(len(raw)):
        by_event.setdefault(ek_by_i.get(i, f"item_{i}"), []).append(i)

    reps: List[int] = []
    for ek, idxs in by_event.items():
        # representative: highest importance, then highest relevance, then smallest rank
        rep = sorted(
            idxs,
            key=lambda j: (-imp_by_i.get(j, 0), -rel_by_i.get(j, 0), raw[j].rank),
        )[0]
        reps.append(rep)

    # apply relevance filter (prefer battery-relevant events)
    reps_rel = [i for i in reps if rel_by_i.get(i, 0) >= BATTERY_RELEVANCE_MIN]
    reps_use = reps_rel if reps_rel else reps

    # final ranking: importance desc, relevance desc, rank asc
    reps_use.sort(key=lambda j: (-imp_by_i.get(j, 0), -rel_by_i.get(j, 0), raw[j].rank))

    picked_idxs = reps_use[:top_k]
    picked = [raw[i] for i in picked_idxs]
    picked_scores = [imp_by_i.get(i, 0) for i in picked_idxs]

    stats.update(
        {
            "deduped_count": len(reps_use),
            "dropped": len(raw) - len(reps_use),
            "picked": len(picked),
            "mode": "one_shot_eventkey",
            "models": {"rank": RANK_MODEL, "base": BASE_MODEL},
            "picked_battery_relevance": [rel_by_i.get(i, 0) for i in picked_idxs],
            "picked_monitoring_importance": [imp_by_i.get(i, 0) for i in picked_idxs],
        }
    )

    return picked, picked_scores, stats


# Backward-compatible function name (if other modules call old name)
def collect_naver_top15_last24h_deduped_and_ranked(
    *,
    client_id: str,
    client_secret: str,
    queries: List[str],
    fetch_n: int = 150,
    top_k: int = 15,
) -> Tuple[List[NaverNewsItem], List[int], Dict[str, Any]]:
    return collect_naver_top_last24h_deduped_and_ranked(
        client_id=client_id,
        client_secret=client_secret,
        queries=queries,
        fetch_n=fetch_n,
        top_k=top_k,
    )
