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
NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news/news.md"

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
    low2 = (
        low.replace("_", " ")
           .replace("-", " ")
           .replace("flashlite", "flash lite")
           .replace("flash-lite", "flash lite")
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
BASE_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
# optional overrides
DEDUP_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL_DEDUPE", BASE_MODEL))
RANK_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL_RANK", BASE_MODEL))

DEBUG = os.getenv("NAVER_DEBUG", "0") == "1"

FETCH_N = int(os.getenv("NAVER_FETCH_N", "150"))
TIME_WINDOW_HOURS = int(os.getenv("NAVER_WINDOW_HOURS", "24"))

# Request count is tight -> keep retries low
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "6"))


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


@dataclass
class NaverNewsItem:
    title: str
    description: str
    link: str
    source: str
    published_dt_kst: datetime
    rank: int


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

            r = requests.get("https://openapi.naver.com/v1/search/news.json", headers=headers, params=params, timeout=timeout_sec)
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

                out.append(NaverNewsItem(title=title, description=desc, link=origin, source=src, published_dt_kst=pub_dt, rank=rank_counter))
                rank_counter += 1
                if len(out) >= max_fetch:
                    break

            if sort == "date" and reached_older:
                break
            start += 100

    return out


# -----------------------------
# ONE-SHOT LLM selection: dedupe + importance + topK
# -----------------------------
class OneShotSelection(BaseModel):
    # groups: each index appears exactly once (partition)
    groups: List[List[int]]
    # representative index per group (same length as groups)
    rep_indices: List[int]
    # importance score for each representative (0..100)
    rep_scores: List[int]
    # final chosen representative indices (length = top_k)
    selected_indices: List[int]


SYSTEM_ONESHOT = (
    "당신은 배터리 산업 모니터링 담당자이자 뉴스 편집자입니다.\n"
    "주어진 뉴스 제목 목록에 대해:\n"
    "1) 같은 사건/이슈(표현이 달라도 동일 사건)이면 같은 그룹으로 묶어 중복 제거하세요.\n"
    "   단순히 주제가 비슷한 정도는 같은 그룹이 아닙니다.\n"
    "2) 각 그룹에서 대표 1개를 선택하세요(가능하면 더 일반적/핵심을 잘 담은 제목).\n"
    "3) 대표 뉴스의 '산업 전반 변화 모니터링 중요도'를 0~100으로 점수화하세요.\n"
    "   높은 점수: 대형 투자/증설/공급계약/M&A/정책·규제/관세/공급망 리스크/리콜·사고/핵심기술 진전\n"
    "   낮은 점수: 단순 홍보/소규모 행사/반복 단신\n"
    "4) 최종적으로 중요도 상위 TOP_K개의 대표 뉴스 index를 selected_indices로 반환하세요.\n"
    "제목만 보고 판단하고, 과장/추측은 금지합니다.\n"
)


def dedupe_and_select_topk_one_shot(titles: List[str], top_k: int) -> Optional[OneShotSelection]:
    if not GEMINI_API_KEY or not titles:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)
    n = len(titles)

    prompt = (
        f"TOP_K={top_k}\n\n"
        "[제목 목록]\n"
        + "\n".join([f"{i}: {t}" for i, t in enumerate(titles)])
        + "\n\n"
        "반드시 JSON만 출력. 스키마 준수:\n"
        "{\n"
        "  \"groups\": [[...], ...],\n"
        "  \"rep_indices\": [...],\n"
        "  \"rep_scores\": [...],\n"
        "  \"selected_indices\": [...]\n"
        "}\n"
        "조건:\n"
        "- 0..N-1 모든 index가 groups에 정확히 한 번씩 등장\n"
        "- rep_indices 길이 == groups 길이\n"
        "- rep_scores 길이 == groups 길이 (0..100)\n"
        "- selected_indices 길이 == TOP_K, 모두 rep_indices 중 하나여야 함, 중복 없음\n"
    )

    last_err = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=DEDUP_MODEL,  # one-shot uses DEDUP_MODEL
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_ONESHOT,
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                    "response_schema": OneShotSelection,
                },
            )
            parsed = getattr(resp, "parsed", None)
            if parsed is None:
                parsed = OneShotSelection.model_validate_json(resp.text)
            return parsed
        except Exception as e:
            last_err = e
            if attempt < LLM_RETRIES:
                _sleep_backoff(attempt)
            else:
                if DEBUG:
                    print(f"[WARN] one-shot selection failed: {last_err}")
                return None


def collect_naver_top15_last24h_deduped_and_ranked(
    *,
    client_id: str,
    client_secret: str,
    queries: List[str],
    fetch_n: int = 150,
    top_k: int = 15,
) -> Tuple[List[NaverNewsItem], List[int], Dict[str, int]]:
    raw = collect_naver_last24h_multiquery(
        client_id=client_id,
        client_secret=client_secret,
        queries=queries,
        max_fetch=fetch_n,
        sort="date",
    )

    titles = [it.title for it in raw]
    selection = dedupe_and_select_topk_one_shot(titles, top_k=top_k)

    if selection is None:
        # fallback: no LLM (keep first top_k)
        picked = raw[:top_k]
        scores = [10] * len(picked)
        stats = {
            "raw_count": len(raw),
            "deduped_count": len(raw),
            "dropped": 0,
            "picked": len(picked),
            "models": {"base": BASE_MODEL, "dedupe": DEDUP_MODEL, "rank": RANK_MODEL},
            "mode": "fallback_no_llm",
        }
        return picked, scores, stats

    # Build deduped representatives list (for stats)
    rep_set = set(selection.rep_indices)
    deduped_count = len(rep_set)
    dropped = max(0, len(raw) - deduped_count)

    # selected_indices are representative indices
    selected = []
    selected_scores = []

    # Map rep score by rep index
    rep_score_by_rep: Dict[int, int] = {}
    for rep_idx, score in zip(selection.rep_indices, selection.rep_scores):
        rep_score_by_rep[int(rep_idx)] = int(score)

    for idx in selection.selected_indices:
        i = int(idx)
        if 0 <= i < len(raw):
            selected.append(raw[i])
            selected_scores.append(rep_score_by_rep.get(i, 10))

    stats = {
        "raw_count": len(raw),
        "deduped_count": deduped_count,
        "dropped": dropped,
        "picked": len(selected),
        "models": {"base": BASE_MODEL, "dedupe": DEDUP_MODEL, "rank": RANK_MODEL},
        "mode": "one_shot_llm",
    }

    if DEBUG:
        print(f"[INFO] Gemini models (BASE/DEDUP/RANK): {BASE_MODEL} / {DEDUP_MODEL} / {RANK_MODEL}")
        print(f"[INFO] one-shot stats: {stats}")

    return selected, selected_scores, stats
