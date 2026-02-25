from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import feedparser
from dateutil import parser as dtparser

from .utils import KST, normalize_url


def _parse_date(entry: dict[str, Any]) -> datetime | None:
    # feedparser often provides published_parsed/updated_parsed
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
        # assume UTC if missing (best effort)
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(KST).date().isoformat()


def google_news_rss_url(query: str, hl: str = "en", gl: str = "US", ceid: str = "US:en") -> str:
    # Google News RSS search endpoint
    # Example:
    # https://news.google.com/rss/search?q=battery%20cathode%20after:2026-02-24%20before:2026-02-25&hl=en&gl=US&ceid=US:en
    from urllib.parse import quote_plus
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"


def collect_from_rss(url: str, source_name: str, target_date: str) -> list[dict[str, str]]:
    """
    Collect RSS entries and filter to target KST date (YYYY-MM-DD).
    Returns normalized dict items: title, published_at, source, link, description.
    """
    feed = feedparser.parse(url)
    out: list[dict[str, str]] = []

    for e in feed.entries:
        title = (e.get("title") or "").strip()
        link = normalize_url((e.get("link") or "").strip())

        dt = _parse_date(e)
        if not dt:
            # if date missing, skip (or keep but mark unknown)
            continue

        published_at = _to_kst_date_str(dt)
        if published_at != target_date:
            continue

        # Some Google News entries have "source" in title like "Title - Source"
        # We'll keep source_name (feed-defined) and also try to infer.
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
