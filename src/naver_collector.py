# src/naver_collector.py
from __future__ import annotations

import os
import re
import time
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, List, Dict, Tuple, Optional
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import requests
from dateutil import parser as dtparser
from pydantic import BaseModel, Field
from google import genai


KST = ZoneInfo("Asia/Seoul")
NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL_DEDUPE", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# LLM dedupe tuning
LLM_TEMP = float(os.getenv("LLM_DEDUPE_TEMPERATURE", "0.0"))
LLM_RETRIES = int(os.getenv("LLM_DEDUPE_RETRIES", "2"))
LLM_BACKOFF_MAX = float(os.getenv("LLM_DEDUPE_BACKOFF_MAX", "6"))
LLM_CANDIDATES = int(os.getenv("LLM_DEDUPE_CANDIDATES", "10"))  # compare against top-N similar kept titles
FAST_SIM_GATE = float(os.getenv("LLM_DEDUPE_FAST_GATE", "0.72"))  # if no candidate above this, skip LLM call


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
    tail = tail.strip()
    base = base.strip()
    if 2 <= len(tail) <= 60 and base:
        return base, tail
    return title.strip(), None


def _normalize_title_for_similarity(title: str) -> str:
    t = _strip_html(title).lower()
    # remove common markers
    t = re.sub(r"\b(속보|단독|종합|인터뷰|분석|기획|칼럼|포토)\b", " ", t)
    t = re.sub(r"^\s*[\[\(\<].{0,12}?[\]\)\>]\s*", "", t)  # [단독], (종합), <속보>
    t = re.sub(r"\([^)]{0,24}\)", " ", t)
    t = re.sub(r"\[[^\]]{0,24}\]", " ", t)
    t = re.sub(r"[^0-9a-z가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_similarity(a: str, b: str) -> float:
    na = _normalize_title_for_similarity(a)
    nb = _normalize_title_for_similarity(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


@dataclass
class NaverNewsItem:
    title: str
    description: str
    link: str
    source: str
    published_dt_kst: datetime
    rank: int  # smaller = higher in API response


def collect_naver_last24h(
    *,
    client_id: str,
    client_secret: str,
    query: str,
    max_fetch: int = 150,
    sort: str = "date",
    timeout_sec: int = 20,
) -> List[NaverNewsItem]:
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    now = datetime.now(tz=KST)
    cutoff = now - timedelta(hours=24)

    out: List[NaverNewsItem] = []
    seen_links: set[str] = set()
    rank_counter = 0

    start = 1
    while start <= 1000 and len(out) < max_fetch:
        display = min(100, max_fetch - len(out))
        params = {"query": query, "display": display, "start": start, "sort": sort}

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

            # clean " - 언론사"
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
# LLM semantic dedupe (title-only)
# -----------------------------
class DedupDecision(BaseModel):
    is_duplicate: bool = Field(..., description="Whether the new title is the same news/event as a previous one")
    duplicate_of_rank: int | None = Field(None, description="rank of the kept item it duplicates (if any)")


SYSTEM_DEDUPE = (
    "당신은 뉴스 편집자입니다. "
    "두 뉴스 제목이 '같은 사건/이슈'인지 판단합니다. "
    "표현이 달라도 같은 사건이면 중복으로 간주합니다. "
    "서로 다른 사건이면 중복이 아닙니다. "
    "판단은 제목 텍스트만 기반으로 하며, 추측은 최소화합니다."
)


def _sleep_backoff(attempt: int) -> None:
    t = min(LLM_BACKOFF_MAX, 1.6 ** (attempt + 1))
    t = t * (0.75 + random.random() * 0.5)
    time.sleep(t)


def _llm_is_duplicate(
    client: genai.Client,
    new_title: str,
    candidates: List[Tuple[int, str]],
    model: str,
) -> DedupDecision:
    """
    candidates: list of (rank, title) for already-kept items
    returns: DedupDecision
    """
    # We require strict JSON output via response_schema.
    prompt = f"""
[새 제목]
{new_title}

[비교 대상(이미 선택된 제목들)]
""" + "\n".join([f"- ({rk}) {t}" for rk, t in candidates]) + """

[출력(JSON)]
- is_duplicate: true/false
- duplicate_of_rank: 중복이면 어느 (rank)와 같은지, 아니면 null

[판단 기준]
- 같은 회사/기관의 같은 발표/계약/투자/실적/사고/규제/제품 발표 등 같은 사건이면 중복(true)
- 같은 사건을 다른 표현으로 쓴 경우도 중복(true)
- 단순히 주제가 비슷한 정도(예: 전고체 전반, 배터리 시장 일반)는 중복(false)
""".strip()

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_DEDUPE,
            "temperature": LLM_TEMP,
            "response_mime_type": "application/json",
            "response_schema": DedupDecision,
        },
    )
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        parsed = DedupDecision.model_validate_json(resp.text)
    return parsed


def dedupe_by_llm_semantic(
    items: List[NaverNewsItem],
    *,
    fast_gate: float = FAST_SIM_GATE,
    candidates_n: int = LLM_CANDIDATES,
    model: str = DEFAULT_GEMINI_MODEL,
) -> List[NaverNewsItem]:
    """
    Maintain input order (rank order = "상위 노출" 우선).
    For each new item:
      - find top-N similar kept titles by string similarity
      - if none above fast_gate -> keep (no LLM call)
      - else ask LLM whether it's duplicate of one of candidates
    """
    if not GEMINI_API_KEY:
        # fallback to pure similarity (older behavior)
        return dedupe_by_title_similarity(items, threshold=0.90)

    client = genai.Client(api_key=GEMINI_API_KEY)

    kept: List[NaverNewsItem] = []
    kept_titles: List[Tuple[int, str]] = []  # (rank, title)

    for it in items:
        if not kept:
            kept.append(it)
            kept_titles.append((it.rank, it.title))
            continue

        # compute candidate similarities to already kept
        sims: List[Tuple[float, int, str]] = []
        for rk, t in kept_titles:
            sims.append((_title_similarity(it.title, t), rk, t))
        sims.sort(key=lambda x: x[0], reverse=True)

        # if no one is close enough, skip LLM and keep
        top_sim = sims[0][0] if sims else 0.0
        if top_sim < fast_gate:
            kept.append(it)
            kept_titles.append((it.rank, it.title))
            continue

        candidates = [(rk, t) for _, rk, t in sims[:candidates_n]]

        # LLM decision with retries
        decision: Optional[DedupDecision] = None
        last_err: Optional[Exception] = None
        for attempt in range(LLM_RETRIES + 1):
            try:
                decision = _llm_is_duplicate(client, it.title, candidates, model=model)
                break
            except Exception as e:
                last_err = e
                if attempt < LLM_RETRIES:
                    _sleep_backoff(attempt)
                else:
                    decision = None

        if decision is None:
            # If LLM completely fails, conservatively keep (or you can drop). We'll keep.
            # This prevents accidental over-filtering.
            kept.append(it)
            kept_titles.append((it.rank, it.title))
            continue

        if decision.is_duplicate and decision.duplicate_of_rank is not None:
            # drop duplicate
            continue

        kept.append(it)
        kept_titles.append((it.rank, it.title))

    return kept


def dedupe_by_title_similarity(items: List[NaverNewsItem], threshold: float = 0.88) -> List[NaverNewsItem]:
    kept: List[NaverNewsItem] = []
    for it in items:
        is_dup = False
        for k in kept:
            if _title_similarity(it.title, k.title) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(it)
    return kept


def select_top(items: List[NaverNewsItem], k: int = 15) -> List[NaverNewsItem]:
    return items[:k]


def collect_naver_top15_last24h_deduped(
    *,
    client_id: str,
    client_secret: str,
    query: str,
    fetch_n: int = 150,
    top_k: int = 15,
) -> List[NaverNewsItem]:
    raw = collect_naver_last24h(
        client_id=client_id,
        client_secret=client_secret,
        query=query,
        max_fetch=fetch_n,
        sort="date",
    )

    # ✅ LLM semantic dedupe (title-only)
    deduped = dedupe_by_llm_semantic(raw)

    return select_top(deduped, k=top_k)
