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
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.content
        except Exception:
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
            else:
                return None
    return None


def _extract_publisher_from_google_news(entry: dict[str, Any], title: str) -> tuple[str, str | None]:
    """
    Returns (clean_title, publisher_or_none).
    Google News entries often look like: "Some title - Publisher".
    Also sometimes entry['source'] contains publisher title.
    """
    publisher = None

    src_obj = entry.get("source")
    if isinstance(src_obj, dict):
        publisher = (src_obj.get("title") or "").strip() or None
    elif isinstance(src_obj, str):
        publisher = src_obj.strip() or None

    # title ends with " - Publisher"
    if " - " in title:
        base, tail = title.rsplit(" - ", 1)
        tail = tail.strip()
        # heuristic: tail likely a publisher if reasonably short
        if 2 <= len(tail) <= 60:
            if publisher is None:
                publisher = tail
            # if tail equals publisher, remove it
            if publisher and tail == publisher and base.strip():
                title = base.strip()
            # even if publisher was None, we set publisher=tail and remove tail
            elif publisher == tail and base.strip():
                title = base.strip()

    return title, publisher


def collect_from_rss(url: str, source_name: str, target_date: str) -> list[dict[str, str]]:
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

        real_source = source_name
        if source_name.lower().startswith("google news"):
            title, publisher = _extract_publisher_from_google_news(e, title)
            if publisher:
                real_source = publisher

        out.append(
            {
                "title": title,
                "published_at": published_at,
                "source": real_source,
                "link": link,
                "description": description,
            }
        )

    return out
