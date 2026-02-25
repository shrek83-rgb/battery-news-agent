from __future__ import annotations

import re
from typing import Any

from .utils import domain_of


IMPACT_KEYWORDS = [
    # high-impact signals
    "investment", "invest", "billion", "million", "funding", "raises",
    "contract", "supply", "agreement", "MoU", "MOU", "deal",
    "earnings", "profit", "revenue", "guidance",
    "regulation", "ban", "tariff", "subsidy", "policy",
    "factory", "gigafactory", "plant", "production", "capacity",
    "breakthrough", "patent", "prototype",
    # Korean
    "투자", "조원", "억원", "수주", "공급", "계약", "협약", "실적", "매출", "영업이익",
    "규제", "관세", "보조금", "정책", "공장", "증설", "양산", "생산", "캐파",
    "기술", "특허", "시제품",
]


def infer_tier(link: str, sources_config: dict) -> int:
    d = domain_of(link)
    tiers = sources_config.get("tiers", {})
    for tier_str, cfg in tiers.items():
        tier = int(tier_str)
        for pat in cfg.get("domains", []):
            p = str(pat).lower()
            if p in d:
                return tier
    return 3


def popularity_signal_from_source(source_name: str) -> str:
    # MVP: if the feed itself is a "Most read/Trending" feed, label it.
    s = source_name.lower()
    if "most read" in s or "mostread" in s:
        return "most_read"
    if "trending" in s or "popular" in s or "top" in s:
        return "trending"
    return "unknown"


def score_item(it: dict[str, Any], tier: int, multi_source_hits: int = 1) -> float:
    """
    Heuristic scoring:
    - tier weight (enforced later in sorting)
    - popularity signal
    - impact keywords
    - multi-source presence
    """
    title = (it.get("title") or "")
    desc = (it.get("description") or "")
    text = f"{title} {desc}"

    # impact score
    impact = 0
    t_low = text.lower()
    for k in IMPACT_KEYWORDS:
        if k.lower() in t_low:
            impact += 1

    # multi-source score
    ms = max(0, multi_source_hits - 1)

    # popularity
    pop = it.get("popularity_signal", "unknown")
    pop_score = {"most_read": 4, "trending": 3, "top_ranked": 3, "multi_source": 2, "unknown": 0}.get(pop, 0)

    # base
    return pop_score * 10 + impact * 2 + ms * 3


def sort_key(it: dict[str, Any]) -> tuple:
    """
    Sorting rule:
    1) tier asc (1 first)
    2) strong popularity signal first
    3) score desc
    4) published_at desc (string YYYY-MM-DD works for date)
    """
    tier = int(it.get("tier", 3))
    pop = it.get("popularity_signal", "unknown")
    pop_strength = {"most_read": 0, "trending": 1, "top_ranked": 1, "multi_source": 2, "unknown": 3}.get(pop, 3)
    score = float(it.get("score", 0))
    published_at = it.get("published_at", "0000-00-00")
    return (tier, pop_strength, -score, published_at[::-1])  # simple tie breaker
