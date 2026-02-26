# src/naver_collector.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Iterable, List, Dict, Tuple, Optional
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import requests
from dateutil import parser as dtparser


KST = ZoneInfo("Asia/Seoul")
NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"  # :contentReference[oaicite:1]{index=1}


def _strip_html(s: str) -> str:
    # title/description contains <b> tags etc.
    s = re.sub(r"<[^>]+>", "", s or "")
    s = s.replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()


def _parse_pubdate_kst(pub: str) -> Optional[datetime]:
    # pubDate: RFC 822 like "Mon, 26 Feb 2026 07:12:00 +0900"
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
    """
    Naver/Google 형태 모두에서 종종 '제목 - 언론사'가 붙음.
    제목에서 제거하고, publisher로 반환.
    """
    if " - " not in title:
        return title.strip(), None
    base, tail = title.rsplit(" - ", 1)
    tail = tail.strip()
    base = base.strip()
    if 2 <= len(tail) <= 60 and base:
        return base, tail
    return title.strip(), None


def _normalize_title_for_similarity(title: str) -> str:
    """
    유사도 비교용 정규화:
    - HTML 제거
    - [단독], (종합), <속보> 등 흔한 접두/접미 제거
    - 특수문자 제거 / 공백 축약
    """
    t = _strip_html(title).lower()

    # 흔한 태그/머리말 제거
    t = re.sub(r"^\s*[\[\(\<].{0,10}?[\]\)\>]\s*", "", t)  # [단독] (종합) <속보> 등
    t = re.sub(r"\b(속보|단독|종합|인터뷰|분석|기획|칼럼)\b", " ", t)

    # 괄호 안 짧은 부가정보 제거
    t = re.sub(r"\([^)]{0,20}\)", " ", t)
    t = re.sub(r"\[[^\]]{0,20}\]", " ", t)

    # 기호 제거
    t = re.sub(r"[^0-9a-z가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_similarity(a: str, b: str) -> float:
    """
    제목 유사도: SequenceMatcher(문자열 기반)
    """
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
    # request order rank(작을수록 상단 노출)
    rank: int


def collect_naver_last24h(
    *,
    client_id: str,
    client_secret: str,
    query: str,
    max_fetch: int = 150,
    sort: str = "date",
    timeout_sec: int = 20,
) -> List[NaverNewsItem]:
    """
    네이버 뉴스 검색 API로 최근 24시간 이내 기사 최대 max_fetch건 수집.
    - API는 날짜필터가 없어서 sort=date로 가져온 뒤 pubDate 기준으로 24h 필터링
    - display=100 단위로 페이지네이션(start=1,101,...)
    """
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }  # :contentReference[oaicite:2]{index=2}

    now = datetime.now(tz=KST)
    cutoff = now - timedelta(hours=24)

    out: List[NaverNewsItem] = []
    seen_links: set[str] = set()
    rank_counter = 0

    start = 1
    while start <= 1000 and len(out) < max_fetch:
        display = min(100, max_fetch - len(out))
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": sort,
        }  # :contentReference[oaicite:3]{index=3}

        r = requests.get(NAVER_NEWS_ENDPOINT, headers=headers, params=params, timeout=timeout_sec)
        if r.status_code >= 400:
            raise RuntimeError(f"Naver API HTTP {r.status_code}: {r.text[:400]}")

        data = r.json()
        items = data.get("items") or []
        if not items:
            break

        # sort=date이면 최신 → 과거 순으로 내려온다고 가정하고,
        # cutoff보다 과거가 나타나면 다음 페이지도 과거일 가능성이 높으므로 종료(성능)
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

            # 제목 끝 "- 언론사" 처리
            title, tail_pub = _clean_title_tail_publisher(title_raw)

            # source 추정 (tail_pub 우선, 없으면 도메인)
            src = tail_pub or _domain(origin) or "NAVER"

            # 링크 중복 제거
            if origin in seen_links:
                continue
            seen_links.add(origin)

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
            # 더 내려가면 24h 밖이 많아질 가능성이 큼
            break

        start += 100

    return out


def dedupe_by_title_similarity(
    items: List[NaverNewsItem],
    threshold: float = 0.88,
) -> List[NaverNewsItem]:
    """
    제목 유사도 기반 중복 제거.
    - 입력 순서를 유지(=상위 노출 우선)
    - threshold 이상이면 같은 이슈로 간주하고 뒤의 것을 제거
    """
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


def select_top(
    items: List[NaverNewsItem],
    k: int = 15,
) -> List[NaverNewsItem]:
    # "상위 노출" = API 응답 상단(입력 순서 유지) 기준
    return items[:k]


def collect_naver_top15_last24h_deduped(
    *,
    client_id: str,
    client_secret: str,
    query: str,
    fetch_n: int = 150,
    dedupe_threshold: float = 0.88,
    top_k: int = 15,
) -> List[NaverNewsItem]:
    raw = collect_naver_last24h(
        client_id=client_id,
        client_secret=client_secret,
        query=query,
        max_fetch=fetch_n,
        sort="date",
    )
    deduped = dedupe_by_title_similarity(raw, threshold=dedupe_threshold)
    return select_top(deduped, k=top_k)
