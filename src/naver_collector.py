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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEDUP_MODEL = os.getenv("GEMINI_MODEL_DEDUPE", "gemini-2.0-flash")
RANK_MODEL = os.getenv("GEMINI_MODEL_RANK", "gemini-2.0-flash")

# Dedupe tuning
LLM_DEDUPE_TEMP = float(os.getenv("LLM_DEDUPE_TEMPERATURE", "0.0"))
LLM_DEDUPE_RETRIES = int(os.getenv("LLM_DEDUPE_RETRIES", "2"))
LLM_DEDUPE_BACKOFF_MAX = float(os.getenv("LLM_DEDUPE_BACKOFF_MAX", "6"))
LLM_DEDUPE_CANDIDATES = int(os.getenv("LLM_DEDUPE_CANDIDATES", "10"))
LLM_DEDUPE_FAST_GATE = float(os.getenv("LLM_DEDUPE_FAST_GATE", "0.70"))  # 낮출수록 LLM 더 자주 호출

# Importance tuning
LLM_IMPORTANCE_TEMP = float(os.getenv("LLM_IMPORTANCE_TEMPERATURE", "0.2"))
LLM_IMPORTANCE_RETRIES = int(os.getenv("LLM_IMPORTANCE_RETRIES", "2"))
LLM_IMPORTANCE_BACKOFF_MAX = float(os.getenv("LLM_IMPORTANCE_BACKOFF_MAX", "8"))
LLM_IMPORTANCE_CHUNK = int(os.getenv("LLM_IMPORTANCE_CHUNK", "30"))  # 한 번에 점수화할 제목 수


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
    """
    제목이 '... - 언론사' 형태면 언론사 분리
    """
    if " - " not in title:
        return title.strip(), None
    base, tail = title.rsplit(" - ", 1)
    base = base.strip()
    tail = tail.strip()
    if base and 2 <= len(tail) <= 60:
        return base, tail
    return title.strip(), None


def _normalize_title_for_similarity(title: str) -> str:
    t = _strip_html(title).lower()
    t = re.sub(r"\b(속보|단독|종합|인터뷰|분석|기획|칼럼|포토)\b", " ", t)
    t = re.sub(r"^\s*[\[\(\<].{0,12}?[\]\)\>]\s*", "", t)
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


def _sleep_backoff(attempt: int, max_sec: float) -> None:
    t = min(max_sec, (1.6 ** (attempt + 1)))
    t = t * (0.75 + random.random() * 0.5)
    time.sleep(t)


@dataclass
class NaverNewsItem:
    title: str
    description: str
    link: str
    source: str
    published_dt_kst: datetime
    rank: int  # API 응답 상단일수록 낮음(상위 노출)


# -----------------------------
# 1) Collect 24h / 150
# -----------------------------
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
    - API sort=date로 최신부터 가져온 후
    - pubDate를 KST로 파싱해서 24시간 이내만 유지
    - 최대 max_fetch개까지
    """
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

        # sort=date이면 더 내려갈수록 과거 비중이 커지므로 성능상 종료
        if sort == "date" and reached_older:
            break

        start += 100

    return out


# -----------------------------
# 2) LLM semantic dedupe (title-based)
# -----------------------------
class DedupDecision(BaseModel):
    is_duplicate: bool
    duplicate_of_rank: int | None = None


SYSTEM_DEDUPE = (
    "당신은 뉴스 편집자입니다. "
    "두 뉴스 제목이 '같은 사건/이슈'인지 판단합니다. "
    "표현이 달라도 같은 사건이면 중복(true)입니다. "
    "단순히 주제가 비슷한 정도(예: 배터리 시장 일반)는 중복(false)입니다. "
    "제목만 보고 판단하세요."
)


def _llm_is_duplicate(
    client: genai.Client,
    new_title: str,
    candidates: List[Tuple[int, str]],
    model: str,
) -> DedupDecision:
    prompt = (
        "[새 제목]\n"
        f"{new_title}\n\n"
        "[비교 대상(이미 선택된 제목들)]\n"
        + "\n".join([f"- ({rk}) {t}" for rk, t in candidates])
        + "\n\n"
        "[출력(JSON)]\n"
        "- is_duplicate: true/false\n"
        "- duplicate_of_rank: 중복이면 어느 (rank)와 같은지, 아니면 null\n\n"
        "[판단 기준]\n"
        "- 같은 회사/기관의 같은 발표/계약/투자/실적/사고/규제/제품 발표 등 같은 사건이면 중복(true)\n"
        "- 같은 사건을 다른 표현으로 쓴 경우도 중복(true)\n"
        "- 단순히 주제가 비슷한 정도는 중복(false)\n"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_DEDUPE,
            "temperature": LLM_DEDUPE_TEMP,
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
    fast_gate: float = LLM_DEDUPE_FAST_GATE,
    candidates_n: int = LLM_DEDUPE_CANDIDATES,
    model: str = DEDUP_MODEL,
) -> List[NaverNewsItem]:
    """
    입력 순서 유지(= 상위 노출 우선).
    - 문자열 유사도 상위 후보가 일정 수준(fast_gate) 이상일 때만 LLM 호출(속도/비용 절감)
    - LLM이 중복(true)로 판단하면 제외
    """
    if not GEMINI_API_KEY:
        # LLM 키 없으면 강한 문자열 유사도 기준으로만 제거
        return dedupe_by_title_similarity(items, threshold=0.92)

    client = genai.Client(api_key=GEMINI_API_KEY)

    kept: List[NaverNewsItem] = []
    kept_titles: List[Tuple[int, str]] = []

    for it in items:
        if not kept:
            kept.append(it)
            kept_titles.append((it.rank, it.title))
            continue

        sims: List[Tuple[float, int, str]] = []
        for rk, t in kept_titles:
            sims.append((_title_similarity(it.title, t), rk, t))
        sims.sort(key=lambda x: x[0], reverse=True)

        top_sim = sims[0][0] if sims else 0.0
        if top_sim < fast_gate:
            kept.append(it)
            kept_titles.append((it.rank, it.title))
            continue

        candidates = [(rk, t) for _, rk, t in sims[:candidates_n]]

        decision: Optional[DedupDecision] = None
        last_err: Optional[Exception] = None
        for attempt in range(LLM_DEDUPE_RETRIES + 1):
            try:
                decision = _llm_is_duplicate(client, it.title, candidates, model=model)
                break
            except Exception as e:
                last_err = e
                if attempt < LLM_DEDUPE_RETRIES:
                    _sleep_backoff(attempt, LLM_DEDUPE_BACKOFF_MAX)
                else:
                    decision = None

        if decision is None:
            # LLM 실패 시 과도하게 삭제하지 않도록 보수적으로 keep
            kept.append(it)
            kept_titles.append((it.rank, it.title))
            continue

        if decision.is_duplicate and decision.duplicate_of_rank is not None:
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


# -----------------------------
# 3) LLM importance scoring (title-only)
# -----------------------------
class ImportanceScore(BaseModel):
    index: int = Field(..., description="Index in the provided list")
    score: int = Field(..., ge=0, le=100, description="Importance score for industry-wide monitoring")


class ImportanceResp(BaseModel):
    scores: List[ImportanceScore]


SYSTEM_IMPORTANCE = (
    "당신은 배터리 산업(소재-셀-팩-재활용-정책-공급망) 모니터링 담당자입니다. "
    "아래 뉴스 제목들을 '산업 전반 변화 모니터링에 중요한지' 기준으로 0~100으로 점수화하세요.\n"
    "높은 점수 예:\n"
    "- 주요 기업의 대형 투자/증설/공장, 공급계약, M&A\n"
    "- 정책/규제/보조금/관세/IRA/EU 규정 등 산업 전반 영향\n"
    "- 소재 가격/수급, 공급망 리스크, 안전 리콜/사고\n"
    "- 전고체/나트륨 등 핵심 기술의 의미 있는 진전\n"
    "낮은 점수 예:\n"
    "- 단순 홍보/소규모 행사/지역 단신/반복 보도\n"
    "제목만 보고 판단하세요. 과장 금지."
)


def _llm_score_chunk(
    client: genai.Client,
    titles_with_idx: List[Tuple[int, str]],
    model: str,
) -> Dict[int, int]:
    prompt = (
        "[제목 목록]\n"
        + "\n".join([f"{i}. {t}" for i, t in titles_with_idx])
        + "\n\n"
        "[출력(JSON)]\n"
        "scores: [{index:int, score:int(0~100)}, ...]\n"
        "주의: 모든 index에 대해 반드시 score를 출력\n"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_IMPORTANCE,
            "temperature": LLM_IMPORTANCE_TEMP,
            "response_mime_type": "application/json",
            "response_schema": ImportanceResp,
        },
    )

    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        parsed = ImportanceResp.model_validate_json(resp.text)

    mapping: Dict[int, int] = {}
    for sc in parsed.scores:
        mapping[int(sc.index)] = int(sc.score)
    return mapping


def score_importance_by_llm(
    items: List[NaverNewsItem],
    model: str = RANK_MODEL,
    chunk_size: int = LLM_IMPORTANCE_CHUNK,
) -> List[int]:
    """
    Returns list of scores aligned to items index.
    """
    if not GEMINI_API_KEY:
        # fallback heuristic scoring if no LLM key
        return heuristic_importance_scores(items)

    client = genai.Client(api_key=GEMINI_API_KEY)
    scores = [0 for _ in items]

    # chunk scoring
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        titles_with_idx = [(start + i, it.title) for i, it in enumerate(chunk)]

        last_err: Optional[Exception] = None
        mapping: Optional[Dict[int, int]] = None
        for attempt in range(LLM_IMPORTANCE_RETRIES + 1):
            try:
                mapping = _llm_score_chunk(client, titles_with_idx, model=model)
                break
            except Exception as e:
                last_err = e
                if attempt < LLM_IMPORTANCE_RETRIES:
                    _sleep_backoff(attempt, LLM_IMPORTANCE_BACKOFF_MAX)
                else:
                    mapping = None

        if mapping is None:
            # fallback for this chunk only
            hs = heuristic_importance_scores(chunk)
            for i, s in enumerate(hs):
                scores[start + i] = s
            continue

        # fill scores (ensure all indices exist; if missing, fallback)
        for idx, _title in titles_with_idx:
            if idx in mapping:
                scores[idx] = mapping[idx]
            else:
                scores[idx] = heuristic_importance_scores([items[idx]])[0]

    return scores


def heuristic_importance_scores(items: List[NaverNewsItem]) -> List[int]:
    """
    아주 간단한 키워드 기반 fallback(LLM 없거나 실패 시).
    """
    high = [
        "투자", "조원", "억원", "수주", "공급", "계약", "협약", "mou", "deal", "agreement",
        "증설", "공장", "양산", "생산", "캐파", "capacity", "plant", "gigafactory",
        "규제", "관세", "보조금", "정책", "ira", "eu", "battery regulation",
        "리콜", "화재", "사고", "안전",
        "전고체", "나트륨", "solid-state", "sodium-ion", "breakthrough",
        "실적", "매출", "영업이익", "earnings", "profit",
        "m&a", "인수", "합병",
    ]
    mid = ["출시", "개발", "연구", "협력", "파트너", "파트너십", "pilot", "prototype"]

    out = []
    for it in items:
        t = it.title.lower()
        s = 10
        for k in high:
            if k.lower() in t:
                s += 10
        for k in mid:
            if k.lower() in t:
                s += 4
        # clamp
        out.append(min(100, s))
    return out


# -----------------------------
# 4) Combined pipeline: fetch 150 -> LLM dedupe -> LLM importance -> top 15
# -----------------------------
def collect_naver_top15_last24h_deduped_and_ranked(
    *,
    client_id: str,
    client_secret: str,
    query: str,
    fetch_n: int = 150,
    top_k: int = 15,
) -> Tuple[List[NaverNewsItem], List[int]]:
    """
    Returns (picked_items, picked_scores) aligned.
    """
    raw = collect_naver_last24h(
        client_id=client_id,
        client_secret=client_secret,
        query=query,
        max_fetch=fetch_n,
        sort="date",
    )

    deduped = dedupe_by_llm_semantic(raw)

    # importance scoring (title-only)
    scores = score_importance_by_llm(deduped)

    # select top_k by score desc, tie-break by rank asc (상위 노출 우선)
    idxs = list(range(len(deduped)))
    idxs.sort(key=lambda i: (-scores[i], deduped[i].rank))

    picked_idxs = idxs[:top_k]
    picked_items = [deduped[i] for i in picked_idxs]
    picked_scores = [scores[i] for i in picked_idxs]

    return picked_items, picked_scores
