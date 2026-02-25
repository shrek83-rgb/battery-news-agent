from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


KST = ZoneInfo("Asia/Seoul")


def kst_today_date_str() -> str:
    return datetime.now(tz=KST).date().isoformat()


def kst_yesterday_date_str() -> str:
    return (datetime.now(tz=KST).date() - timedelta(days=1)).isoformat()


def normalize_url(url: str) -> str:
    """Remove common tracking parameters and normalize URL."""
    if not url:
        return url
    try:
        p = urlparse(url)
        q = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_")
             and k.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}]
        new_query = urlencode(q, doseq=True)
        return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))
    except Exception:
        return url


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def getenv_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class NewsItem:
    title: str
    published_at: str  # YYYY-MM-DD
    source: str
    link: str
    tier: int
    popularity_signal: str  # most_read/trending/top_ranked/multi_source/unknown
    category: str
    summary_3_sentences: list[str]
    related_links: list[str]  # dedupe: optional reference links
    score: float
