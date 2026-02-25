from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import feedparser
import requests
from dateutil import parser as dtparser

from .utils import KST, normalize_url


DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36"
)


def _parse_date(entry: dict[str, Any]) -> datetime | None:
    for key in ("published", "updated"):
        val = entry.get(key)
        if val:
            try:
                return dtparser.parse(val)
            except Exception:
                pass
    return None


def _to_kst_date_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(KST).date().isoformat()


def google_news_rss_url(query: str, hl: str = "en", gl: str = "US", ceid: str = "US:en") -> str:
    from urllib.parse import quote_plus
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"


def _fetch_url(url: str, timeout: int = 20, retries: int = 2, backoff_sec: float = 1.5) -> bytes | None:
    headers = {"User-Agent": DEFAULT_UA, "Accept": "application/rss+xml,application/xml,text/xml,*/*;q=0.8"}
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            # Some sites return 403/406 etc; treat as failure but don't crash
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.content
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
            else:
                # final failure
                return None
    return None


def collect_from_rss(url: str, source_name: str, target_date: str) -> list[dict[str, str]]:
    """
    Robust RSS collection:
    - fetch via requests with UA/timeout/retries
    - parse bytes with feedparser
    - if a source fails, return empty list (do not crash pipeline)
    """
    content = _fetch_url(url, timeout=25, retries=2)
    if content is None:
        print(f"[WARN] RSS fetch failed, skipping: {source_name} | {url}")
        return []

    feed = feedparser.parse(content)
    out: list[dict[str, str]] = []

    for e in getattr(feed, "entries", []):
        title = (e.get("title") or "").strip()
        link = normalize_url((e.get("link") or "").strip())

        dt = _parse_date(e)
        if not dt:
            continue

        published_at = _to_kst_date_str(dt)
        if published_at != target_date:
            continue

        description = (e.get("summary") or e.get("description") or "").strip()
        description = re.sub(r"\s+", " ", description)

        out.append(
            {
                "title": title,
                "published_at": published_at,
                "source": source_name,
                "link": link,
                "description": description,
            }
        )

    return out
