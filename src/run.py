# src/naver_collector.py
from __future__ import annotations

import os
import re
import json
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

# -----------------------------
# Model selection (single source of truth)
# -----------------------------
def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    if n.startswith("google.gemini-"):
        n = n.replace("google.", "", 1)

    low = n.lower().strip()
    low = low.replace("light", "lite")  # common typo

    # allow gemma models as-is
    if low.startswith("gemma-"):
        return low

    low2 = (
        low.replace("_", " ")
           .replace("-", " ")
           .replace("flashlite", "flash lite")
           .replace("  ", " ")
    )

    if "2.5" in low2 and "flash" in low2 and "lite" in low2:
        return "gemini-2.5-flash-lite"
    if "2.5" in low2 and "flash" in low2:
        return "gemini-2.5-flash"
    if "2.0" in low2 and "flash" in low2 and "lite" in low2:
        return "gemini-2.0-flash-lite"
    if "2.0" in low2 and "flash" in low2:
        return "gemini-2.0-flash"

    if low.startswith("gemini-"):
        return low
    return n


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
BASE_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
RANK_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL_RANK", BASE_MODEL))

DEBUG = os.getenv("NAVER_DEBUG", "0") == "1"

FETCH_N = int(os.getenv("NAVER_FETCH_N", "150"))
TIME_WINDOW_HOURS = int(os.getenv("NAVER_WINDOW_HOURS", "24"))

# request-count tight => default no retry
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "0"))
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "6"))

BATTERY_RELEVANCE_MIN = int(os.getenv("BATTERY_RELEVANCE_MIN", "55"))


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


@dataclass
class NaverNewsItem:
    title: str
    description: str
    link: str
    source: str
    published_dt_kst: datetime
    rank: int  # smaller => higher in API order


def collect_naver_last24h_multiquery(
    *,
    client_id: str,
    client_secret: str,
    queries: List[str],
    max_fetch: int = 150,
    sort: str = "date",
    timeout_sec: int = 20,
) -> List[NaverNewsItem]:
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

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
# LLM: score relevance + importance + event_key (title-only)
# -----------------------------
class TitleScore(BaseModel):
    index: int
    event_key: str = Field(..., description="A short stable key for the same event/issue (use ASCII letters/digits/_)")
    battery_relevance: int = Field(..., ge=0, le=100)
    monitoring_importance: int = Field(..., ge=0, le=100)


class TitleScoreResp(BaseModel):
    items: List[TitleScore]


def _llm_score_titles(titles: List[str], model: str) -> Optional[TitleScoreResp]:
    if not GEMINI_API_KEY or not titles:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "다음은 뉴스 제목 목록입니다. 각 제목에 대해 아래 정보를 JSON으로만 출력하세요.\n\n"
        "출력 스키마:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\"index\": 0, \"event_key\": \"...\", \"battery_relevance\": 0-100, \"monitoring_importance\": 0-100}\n"
        "  ]\n"
        "}\n\n"
        "규칙:\n"
        "- event_key: 같은 사건/이슈면 같은 event_key를 부여(표현이 달라도 동일 사건이면 동일 key).\n"
        "  예: POSCO_SK_lithium_supply_deal 처럼 ASCII/underscore로 짧게.\n"
        "- battery_relevance(0~100): 배터리 산업과의 직접 연관성 점수.\n"
        "  90~100: 소재(양극재/음극재/전해질/분리막), 셀/팩/ESS/EV배터리, 공급계약/증설/기술/안전/리사이클, 관세/정책 등 직접 영향\n"
        "  50~80: 배터리와 연관은 있으나 주변부(관련 산업/부품/인프라 등)\n"
        "  0~40: 배터리와 무관\n"
        "- monitoring_importance(0~100): 배터리 산업 모니터링 관점에서의 파급력.\n"
        "  대형 투자/증설/공급계약/M&A/정책·규제/관세/공급망 리스크/리콜·사고/핵심기술 진전은 높게.\n"
        "- 모든 index(0..N-1)에 대해 반드시 1개씩 출력.\n"
        "- 제목만 보고 판단, 과장/추측 금지.\n\n"
        "[제목 목록]\n"
        + "\n".join([f"{i}: {t}" for i, t in enumerate(titles)])
    )

    cfg: Dict[str, Any] = {"temperature": 0.2}
    # gemma 계열은 system_instruction이 막힐 수 있어 prompt 안에만 넣음
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    raw = _extract_json(resp.text)
    return TitleScoreResp.model_validate_json(raw)


def _heuristic_relevance(title: str) -> int:
    t = (title or "").lower()
    kws = [
        "battery", "2차전지", "이차전지", "전고체", "solid-state", "sodium", "나트륨",
        "cathode", "양극", "anode", "음극", "electrolyte", "전해질", "separator", "분리막",
        "lithium", "리튬", "nickel", "니켈", "lfp", "ncm", "recycling", "재활용", "black mass", "블랙매스",
        "ess", "ev", "전기차", "charging", "충전", "bms",
    ]
    score = 0
    for k in kws:
        if k in t:
            score += 12
    return min(100, max(0, score))


def _monitor_score(rel: int, imp: int) -> float:
    return 0.7 * rel + 0.3 * imp


def collect_naver_top15_last24h_deduped_and_ranked(
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

    titles = [it.title for it in raw]

    # single request (default) – retries optional
    scored: Optional[TitleScoreResp] = None
    last_err: Optional[Exception] = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            scored = _llm_score_titles(titles, model=RANK_MODEL)
            break
        except Exception as e:
            last_err = e
            scored = None
            if attempt < LLM_RETRIES:
                _sleep_backoff(attempt)

    # fallback: heuristic only (no LLM)
    if scored is None:
        if DEBUG:
            print(f"[WARN] NAVER LLM scoring failed -> fallback. err={last_err}")
        # treat each item as unique event_key; sort by heuristic relevance + recency(rank)
        scored_items = []
        for i, it in enumerate(raw):
            rel = _heuristic_relevance(it.title)
            imp = 10
            scored_items.append((i, f"item_{i}", rel, imp))
        mode = "fallback_no_llm"
    else:
        scored_items = [(x.index, x.event_key, x.battery_relevance, x.monitoring_importance) for x in scored.items]
        mode = "llm_scored"

    # build dicts for grouping
    by_event: Dict[str, List[int]] = {}
    rel_by_i: Dict[int, int] = {}
    imp_by_i: Dict[int, int] = {}
    for idx, ek, rel, imp in scored_items:
        idx = int(idx)
        ek = (ek or f"item_{idx}").strip()[:80] or f"item_{idx}"
        rel = int(max(0, min(100, rel)))
        imp = int(max(0, min(100, imp)))
        rel_by_i[idx] = rel
        imp_by_i[idx] = imp
        by_event.setdefault(ek, []).append(idx)

    # choose representative per event_key (highest monitor_score, tie by rank)
    reps: List[int] = []
    for ek, idxs in by_event.items():
        def key(i: int):
            return (_monitor_score(rel_by_i.get(i, 0), imp_by_i.get(i, 0)), -raw[i].rank * 0.0, -raw[i].rank)  # rank tie-break
        # we want max monitor_score, but smaller rank is better: use (score, -rank)
        rep = max(idxs, key=lambda i: (_monitor_score(rel_by_i.get(i, 0), imp_by_i.get(i, 0)), -raw[i].rank))
        reps.append(rep)

    # filter by relevance threshold first
    reps_sorted = sorted(
        reps,
        key=lambda i: (_monitor_score(rel_by_i.get(i, 0), imp_by_i.get(i, 0)), rel_by_i.get(i, 0), imp_by_i.get(i, 0), -raw[i].rank),
        reverse=True,
    )
    reps_filtered = [i for i in reps_sorted if rel_by_i.get(i, 0) >= BATTERY_RELEVANCE_MIN]
    picked_idxs = reps_filtered[:top_k]
    if len(picked_idxs) < top_k:
        # if not enough, fill from remaining reps (even if relevance low)
        extra = [i for i in reps_sorted if i not in picked_idxs]
        picked_idxs.extend(extra[: max(0, top_k - len(picked_idxs))])

    picked = [raw[i] for i in picked_idxs]
    picked_scores = [int(round(_monitor_score(rel_by_i.get(i, 0), imp_by_i.get(i, 0)))) for i in picked_idxs]
    picked_rels = [rel_by_i.get(i, 0) for i in picked_idxs]
    picked_imps = [imp_by_i.get(i, 0) for i in picked_idxs]

    stats = {
        "raw_count": len(raw),
        "deduped_count": len(reps),
        "dropped": max(0, len(raw) - len(reps)),
        "picked": len(picked),
        "models": {"base": BASE_MODEL, "rank": RANK_MODEL},
        "mode": mode,
        "picked_battery_relevance": picked_rels,
        "picked_monitoring_importance": picked_imps,
    }

    return picked, picked_scores, stats
