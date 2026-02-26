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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEDUP_MODEL = os.getenv("GEMINI_MODEL_DEDUPE", "gemini-2.5-flash")
RANK_MODEL = os.getenv("GEMINI_MODEL_RANK", "gemini-2.5-flash")

DEBUG = os.getenv("NAVER_DEBUG", "0") == "1"

# fetch control
FETCH_N = int(os.getenv("NAVER_FETCH_N", "150"))
TIME_WINDOW_HOURS = int(os.getenv("NAVER_WINDOW_HOURS", "24"))

# LLM retries/backoff
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))

# importance chunk
LLM_IMPORTANCE_CHUNK = int(os.getenv("LLM_IMPORTANCE_CHUNK", "30"))


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
    rank: int  # API 응답 순서(작을수록 상단)


# -----------------------------
# 1) Collect 24h by MULTI QUERIES until 150
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

                out.append(NaverNewsItem(title=title, description=desc, link=origin, source=src, published_dt_kst=pub_dt, rank=rank_counter))
                rank_counter += 1
                if len(out) >= max_fetch:
                    break

            if sort == "date" and reached_older:
                break
            start += 100

    return out


# -----------------------------
# 2) LLM clustering dedupe (title-only)
# -----------------------------
class ClusterResp(BaseModel):
    # groups: list of lists of indices (0..n-1). Each index must appear exactly once.
    groups: List[List[int]]


SYSTEM_CLUSTER = (
    "당신은 뉴스 편집자입니다. "
    "아래 제목들을 '같은 사건/이슈'끼리 그룹으로 묶으세요. "
    "표현이 달라도 같은 사건이면 같은 그룹입니다. "
    "단순히 주제가 비슷한 정도는 같은 그룹이 아닙니다. "
    "모든 인덱스는 정확히 한 번씩만 등장해야 합니다."
)


def dedupe_by_llm_clustering(items: List[NaverNewsItem]) -> Tuple[List[NaverNewsItem], int]:
    """
    Returns (deduped_items, dropped_count).
    Representative per cluster = lowest rank (상위 노출 우선)
    """
    if not GEMINI_API_KEY or len(items) <= 1:
        return items, 0

    client = genai.Client(api_key=GEMINI_API_KEY)

    titles = [it.title for it in items]
    n = len(titles)

    # 한번에 너무 길어질 수 있으니(150) chunking 후 2-pass 병합
    # pass1: chunk 내 클러스터
    chunk_size = int(os.getenv("LLM_CLUSTER_CHUNK", "60"))
    groups_all: List[List[int]] = []

    def call_cluster(chunk_titles_with_global_idx: List[Tuple[int, str]]) -> List[List[int]]:
        prompt = "[제목 목록]\n" + "\n".join([f"{i}: {t}" for i, t in chunk_titles_with_global_idx]) + "\n\n" + \
                 "JSON으로만 출력: {\"groups\": [[index,index,...], ...]}\n" + \
                 "조건: 모든 index는 정확히 한 번씩 포함되어야 함."
        resp = client.models.generate_content(
            model=DEDUP_MODEL,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_CLUSTER,
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": ClusterResp,
            },
        )
        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            parsed = ClusterResp.model_validate_json(resp.text)
        return parsed.groups

    # chunk clustering
    for start in range(0, n, chunk_size):
        chunk = [(i, titles[i]) for i in range(start, min(n, start + chunk_size))]
        last_err = None
        chunk_groups = None
        for attempt in range(LLM_RETRIES + 1):
            try:
                chunk_groups = call_cluster(chunk)
                break
            except Exception as e:
                last_err = e
                if attempt < LLM_RETRIES:
                    _sleep_backoff(attempt)
                else:
                    chunk_groups = None
        if chunk_groups is None:
            # fallback: no dedupe in this chunk
            if DEBUG:
                print(f"[WARN] cluster failed for chunk {start}-{start+len(chunk)-1}: {last_err}")
            chunk_groups = [[i] for i, _ in chunk]
        groups_all.extend(chunk_groups)

    # pass2: merge clusters across chunks by clustering representatives
    reps = []
    rep_to_group = []
    for gi, g in enumerate(groups_all):
        # representative = lowest rank (input order is already rank-ish, but use rank field)
        rep_idx = min(g, key=lambda idx: items[idx].rank)
        reps.append((gi, rep_idx, titles[rep_idx]))
        rep_to_group.append(gi)

    if len(reps) > 1:
        rep_chunk = [(ri, t) for (gi, rep_idx, t), ri in zip(reps, range(len(reps)))]
        # cluster reps (by rep index in rep list)
        def call_cluster_reps(rep_titles: List[Tuple[int, str]]) -> List[List[int]]:
            prompt = "[대표 제목 목록]\n" + "\n".join([f"{i}: {t}" for i, t in rep_titles]) + "\n\n" + \
                     "JSON으로만 출력: {\"groups\": [[index,index,...], ...]}\n" + \
                     "조건: 모든 index는 정확히 한 번씩 포함되어야 함."
            resp = client.models.generate_content(
                model=DEDUP_MODEL,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_CLUSTER,
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_schema": ClusterResp,
                },
            )
            parsed = getattr(resp, "parsed", None)
            if parsed is None:
                parsed = ClusterResp.model_validate_json(resp.text)
            return parsed.groups

        last_err = None
        rep_groups = None
        for attempt in range(LLM_RETRIES + 1):
            try:
                rep_groups = call_cluster_reps([(i, reps[i][2]) for i in range(len(reps))])
                break
            except Exception as e:
                last_err = e
                if attempt < LLM_RETRIES:
                    _sleep_backoff(attempt)
                else:
                    rep_groups = None

        if rep_groups is not None:
            # merge original groups according to rep cluster
            merged_groups: List[List[int]] = []
            for rg in rep_groups:
                merged: List[int] = []
                for rep_list_idx in rg:
                    orig_group_idx = reps[rep_list_idx][0]
                    merged.extend(groups_all[orig_group_idx])
                merged_groups.append(sorted(set(merged)))
            groups_all = merged_groups
        else:
            if DEBUG:
                print(f"[WARN] rep-cluster failed: {last_err}")

    # final dedupe: pick one per group
    deduped: List[NaverNewsItem] = []
    for g in groups_all:
        rep_idx = min(g, key=lambda idx: items[idx].rank)
        deduped.append(items[rep_idx])

    dropped = len(items) - len(deduped)
    return deduped, dropped


# -----------------------------
# 3) Importance scoring (title-only) and pick top 15
# -----------------------------
class ImportanceScore(BaseModel):
    index: int = Field(..., description="Index in the provided list")
    score: int = Field(..., ge=0, le=100)


class ImportanceResp(BaseModel):
    scores: List[ImportanceScore]


SYSTEM_IMPORTANCE = (
    "당신은 배터리 산업 모니터링 담당자입니다. "
    "아래 뉴스 제목들을 '산업 전반 변화 모니터링에 중요한지' 기준으로 0~100으로 점수화하세요.\n"
    "높은 점수: 대형 투자/증설/공급계약/M&A/정책·규제/관세/공급망 리스크/리콜·사고/핵심기술 진전\n"
    "낮은 점수: 단순 홍보/소규모 행사/반복 단신\n"
    "제목만 보고 판단."
)


def score_importance_by_llm(items: List[NaverNewsItem]) -> List[int]:
    if not GEMINI_API_KEY:
        return [10] * len(items)

    client = genai.Client(api_key=GEMINI_API_KEY)
    scores = [0 for _ in items]

    def call_scores(chunk_pairs: List[Tuple[int, str]]) -> Dict[int, int]:
        prompt = "[제목 목록]\n" + "\n".join([f"{i}. {t}" for i, t in chunk_pairs]) + "\n\n" + \
                 "JSON: {\"scores\": [{\"index\":i,\"score\":0-100}, ...]} (모든 index 포함)"
        resp = client.models.generate_content(
            model=RANK_MODEL,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_IMPORTANCE,
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": ImportanceResp,
            },
        )
        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            parsed = ImportanceResp.model_validate_json(resp.text)
        return {int(x.index): int(x.score) for x in parsed.scores}

    for start in range(0, len(items), LLM_IMPORTANCE_CHUNK):
        chunk = items[start:start + LLM_IMPORTANCE_CHUNK]
        pairs = [(start + i, it.title) for i, it in enumerate(chunk)]
        mapping = None
        last_err = None
        for attempt in range(LLM_RETRIES + 1):
            try:
                mapping = call_scores(pairs)
                break
            except Exception as e:
                last_err = e
                if attempt < LLM_RETRIES:
                    _sleep_backoff(attempt)
                else:
                    mapping = None
        if mapping is None:
            if DEBUG:
                print(f"[WARN] importance scoring failed chunk {start}: {last_err}")
            for i, _ in pairs:
                scores[i] = 10
        else:
            for i, _ in pairs:
                scores[i] = mapping.get(i, 10)

    return scores


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

    deduped, dropped = dedupe_by_llm_clustering(raw)
    scores = score_importance_by_llm(deduped)

    idxs = list(range(len(deduped)))
    idxs.sort(key=lambda i: (-scores[i], deduped[i].rank))
    picked_idxs = idxs[:top_k]
    picked = [deduped[i] for i in picked_idxs]
    picked_scores = [scores[i] for i in picked_idxs]

    stats = {
        "raw_count": len(raw),
        "deduped_count": len(deduped),
        "dropped": dropped,
        "picked": len(picked),
    }
    return picked, picked_scores, stats
