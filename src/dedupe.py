from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from .utils import normalize_url


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def dedupe_items(raw_items: list[dict[str, Any]], sim_threshold: float = 0.88) -> list[dict[str, Any]]:
    """
    Merge near-duplicate titles.
    Keeps one representative item and stores up to 2 related links.
    """
    items = []
    for it in raw_items:
        it["link"] = normalize_url(it.get("link", ""))
        it["related_links"] = []
        items.append(it)

    kept: list[dict[str, Any]] = []

    for it in items:
        merged = False
        for k in kept:
            if title_similarity(it["title"], k["title"]) >= sim_threshold:
                # merge: keep representative (first) and add reference link if different
                if it["link"] and it["link"] != k["link"]:
                    if len(k["related_links"]) < 2 and it["link"] not in k["related_links"]:
                        k["related_links"].append(it["link"])
                merged = True
                break
        if not merged:
            kept.append(it)

    return kept
