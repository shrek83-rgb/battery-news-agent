from __future__ import annotations

import re
from typing import Any
from dateutil import parser as dtparser
import requests

from .utils import KST, normalize_url


NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"


def _strip_html(s: str) -> str:
    # Naver title/description has <b> tags
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _to_kst_date_str(pubdate: str) -> str | None:
    try:
        dt = dtparser.parse(pubdate)
        return dt.astimezone(KST).date().isoformat()
    except Exception:
        return None


def _infer_source_from_title_or_link(title: str, originallink: str) -> str:
    # Often title is ".... - 매체"
    if " - " in title:
        base, tail = title.rsplit(" - ", 1)
        tail = tail.strip()
        if 2 <= len(tail) <= 60:
            return tail
    # fallback to domain
    try:
        from urllib.parse import urlparse
        return urlparse(originallink).netloc
    except Exception:
        return "NAVER"


def collect_naver_news_yesterday(
    target_date: str,
    client_id: str,
    client_secret: str,
    queries: list[str],
    need: int = 10,
) -> list[dict[str, Any]]:
    """
    Collect Naver Search API news (sort=date), then filter to target KST date.
    Since API doesn't support date range directly, we over-fetch and filter by pubDate.
    """
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    results: list[dict[str, Any]] = []
    seen_links: set[str] = set()

    for q in queries:
        start = 1
        while start <= 1000 and len(results) < need:
            params = {
                "query": q,
                "display": 100,
                "start": start,
                "sort": "date",
            }
            r = requests.get(NAVER_NEWS_ENDPOINT, headers=headers, params=params, timeout=20)
            if r.status_code >= 400:
                raise RuntimeError(f"Naver API HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            items = data.get("items") or []
            if not items:
                break

            for it in items:
                pub = _to_kst_date_str(it.get("pubDate", ""))
                if pub != target_date:
                    continue

                title_raw = _strip_html(it.get("title", ""))
                desc = _strip_html(it.get("description", ""))

                origin = it.get("originallink") or it.get("link") or ""
                origin = normalize_url(origin)
                if not origin or origin in seen_links:
                    continue

                # Remove "- 언론사" tail from title if present
                title = title_raw
                if " - " in title:
                    base, tail = title.rsplit(" - ", 1)
                    if 2 <= len(tail.strip()) <= 60:
                        title = base.strip()

                source = _infer_source_from_title_or_link(title_raw, origin)

                results.append(
                    {
                        "title": title,
                        "published_at": pub,
                        "source": source,
                        "link": origin,  # 원문 링크 우선
                        "description": desc,
                        "provider": "naver",
                        "popularity_signal": "unknown",
                        "related_links": [],
                    }
                )
                seen_links.add(origin)

                if len(results) >= need:
                    break

            start += 100

    return results
