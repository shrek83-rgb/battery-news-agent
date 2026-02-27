# src/naver_collector.py
from __future__ import annotations

import os
import re
import time
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import requests
from dateutil import parser as dtparser
from pydantic import BaseModel, Field
from google import genai


# =============================
# Basic Config
# =============================

KST = ZoneInfo("Asia/Seoul")
NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"


def _normalize_model_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = n.replace("light", "lite")
    if n.startswith("google.gemini-"):
        n = n.replace("google.", "", 1)
    return n


GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
BASE_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
RANK_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL_RANK", BASE_MODEL))

FETCH_N = int(os.getenv("NAVER_FETCH_N", "150"))
TIME_WINDOW_HOURS = int(os.getenv("NAVER_WINDOW_HOURS", "24"))

BATTERY_RELEVANCE_MIN = int(os.getenv("BATTERY_RELEVANCE_MIN", "75"))

NEAR_DUP_SIM_TH = float(os.getenv("NAVER_NEAR_DUP_SIM_TH", "0.74"))
MAX_PER_ENTITY = int(os.getenv("NAVER_MAX_PER_ENTITY", "1"))
MAX_PER_TOPIC = int(os.getenv("NAVER_MAX_PER_TOPIC", "2"))

LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))
DEBUG = os.getenv("NAVER_DEBUG", "0") == "1"


# =============================
# Utilities
# =============================

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


def _sleep_backoff(attempt: int) -> None:
    t = min(8, (1.7 ** (attempt + 1)))
    t = t * (0.75 + random.random() * 0.5)
    time.sleep(t)


# =============================
# Data Model
# =============================

@dataclass
class NaverNewsItem:
    title: str
    description: str
    link: str
    source: str
    published_dt_kst: datetime
    rank: int


# =============================
# 1) Collect 24h
# =============================

def collect_naver_last24h_multiquery(
    *,
    client_id: str,
    client_secret: str,
    queries: List[str],
    max_fetch: int = 150,
    sort: str = "date",
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

            r = requests.get(NAVER_NEWS_ENDPOINT, headers=headers, params=params)
            if r.status_code >= 400:
                break

            items = r.json().get("items") or []
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

                title = _strip_html(it.get("title", ""))
                desc = _strip_html(it.get("description", ""))
                link = (it.get("originallink") or it.get("link") or "").strip()

                if not link or link in seen_links:
                    continue
                seen_links.add(link)

                source = _domain(link) or "NAVER"

                out.append(
                    NaverNewsItem(
                        title=title,
                        description=desc,
                        link=link,
                        source=source,
                        published_dt_kst=pub_dt,
                        rank=rank_counter,
                    )
                )
                rank_counter += 1

                if len(out) >= max_fetch:
                    break

            if reached_older:
                break

            start += 100

    return out


# =============================
# 2) One-shot LLM scoring
# =============================

class TitleScore(BaseModel):
    index: int
    event_key: str
    battery_relevance: int = Field(ge=0, le=100)
    monitoring_importance: int = Field(ge=0, le=100)


class TitleScoreResp(BaseModel):
    items: List[TitleScore]


SYSTEM_SCORE = (
    "당신은 배터리 산업 전문 모니터링 담당자입니다.\n"
    "제목만 보고 아래 3가지를 산출합니다.\n\n"
    "1) event_key: 동일 사건이면 동일 키(ASCII letters/digits/_).\n"
    "2) battery_relevance(0~100): 배터리 산업 직접 연관성.\n"
    "3) monitoring_importance(0~100): 배터리 산업 구조 변화 영향도.\n\n"
    "배터리 직접성이 낮으면 importance도 낮게 부여.\n"
    "펀드/거시경제 뉴스는 배터리 기업 명시 없으면 60점 이상 주지 말 것.\n"
    "JSON만 출력."
)


def score_titles_one_shot(items: List[NaverNewsItem]) -> Optional[TitleScoreResp]:
    if not GEMINI_API_KEY or not items:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "[제목 목록]\n"
        + "\n".join([f"{i}: {it.title}" for i, it in enumerate(items)])
        + "\n\n"
        + "{\"items\":[{\"index\":0,\"event_key\":\"...\",\"battery_relevance\":0,\"monitoring_importance\":0}]}"
    )

    last_err = None

    for attempt in range(LLM_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=RANK_MODEL,
                contents=prompt,
                config={"system_instruction": SYSTEM_SCORE, "temperature": 0.2},
            )
            return TitleScoreResp.model_validate_json(resp.text)
        except Exception as e:
            last_err = e
            if attempt < LLM_RETRIES:
                _sleep_backoff(attempt)

    if DEBUG:
        print(f"[WARN] LLM scoring failed: {last_err}")
    return None


# =============================
# 3) 강화 중복 제거 (제목 유사도 + 엔터티/토픽 제한)
# =============================

_TOPIC_KWS = {
    "전고체": ["전고체", "solid"],
    "리튬": ["리튬", "lithium"],
    "양극재": ["양극재", "전구체", "니켈"],
    "재활용": ["재활용", "리사이클"],
}


_KOR_ENTITY_SUFFIX = r"(?:그룹|홀딩스|에너지|화학|전지|배터리|머티리얼즈|소재|전자|솔루션|엔솔|이노베이션|온|SDI|대학교|대학)"
_RX_ENTITY = re.compile(rf"([가-힣A-Za-z0-9·&\.\-]{{2,24}}{_KOR_ENTITY_SUFFIX})")


def _norm_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^0-9a-z가-힣\s]", "", t)
    return t.strip()


def _ngrams(s: str, n=3):
    s = s.replace(" ", "")
    return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else {s}


def _title_sim(a: str, b: str) -> float:
    A = _ngrams(_norm_title(a))
    B = _ngrams(_norm_title(b))
    if not A or not B:
        return 0
    return len(A & B) / len(A | B)


def _extract_entities(title: str) -> List[str]:
    ents = [m.group(1) for m in _RX_ENTITY.finditer(title)]
    uniq = []
    for e in ents:
        if e not in uniq:
            uniq.append(e)
    return uniq[:2]


def _topic_key(title: str) -> Optional[str]:
    t = title.lower()
    for key, kws in _TOPIC_KWS.items():
        for kw in kws:
            if kw in t:
                return key
    return None


def diversify_selection(raw: List[NaverNewsItem], idxs: List[int], top_k: int):
    picked = []
    picked_titles = []
    entity_count = {}
    topic_count = {}

    for idx in idxs:
        it = raw[idx]
        title = it.title

        # near duplicate
        if any(_title_sim(title, pt) >= NEAR_DUP_SIM_TH for pt in picked_titles):
            continue

        # entity cap
        ents = _extract_entities(title)
        if any(entity_count.get(e, 0) >= MAX_PER_ENTITY for e in ents):
            continue

        # topic cap
        tk = _topic_key(title)
        if tk and topic_count.get(tk, 0) >= MAX_PER_TOPIC:
            continue

        picked.append(idx)
        picked_titles.append(title)

        for e in ents:
            entity_count[e] = entity_count.get(e, 0) + 1
        if tk:
            topic_count[tk] = topic_count.get(tk, 0) + 1

        if len(picked) >= top_k:
            break

    return picked


# =============================
# 4) Main Entry
# =============================

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
    )

    stats: Dict[str, Any] = {"raw_count": len(raw)}

    scored = score_titles_one_shot(raw)
    if not scored:
        return raw[:top_k], [0]*top_k, {"mode": "fallback_no_llm"}

    rel_by_i = {x.index: x.battery_relevance for x in scored.items}
    imp_by_i = {x.index: x.monitoring_importance for x in scored.items}
    ek_by_i = {x.index: x.event_key for x in scored.items}

    # group by event_key
    grouped = {}
    for i in range(len(raw)):
        grouped.setdefault(ek_by_i.get(i, f"item_{i}"), []).append(i)

    reps = []
    for g in grouped.values():
        rep = sorted(g, key=lambda j: (-imp_by_i.get(j, 0), raw[j].rank))[0]
        if rel_by_i.get(rep, 0) >= BATTERY_RELEVANCE_MIN:
            reps.append(rep)

    reps.sort(key=lambda j: (-rel_by_i.get(j, 0), -imp_by_i.get(j, 0), raw[j].rank))

    picked_idxs = diversify_selection(raw, reps, top_k)

    picked = [raw[i] for i in picked_idxs]
    scores = [imp_by_i.get(i, 0) for i in picked_idxs]

    stats.update({
        "deduped_count": len(reps),
        "picked": len(picked),
        "mode": "diversified",
    })

    return picked, scores, stats
