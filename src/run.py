# src/run.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .utils import getenv_int, kst_yesterday_date_str
from .collector import collect_from_rss, google_news_rss_url
from .dedupe import dedupe_items
from .ranker import infer_tier, popularity_signal_from_source, score_item, sort_key
from .tagger import classify_category, extract_companies
from .llm_enrich_gemini import enrich_items
from .renderer import write_outputs
from .datastore import write_daily_csv, upsert_master_csv, upsert_master_json
from .sitegen import build_daily_page, build_root_index


def load_config() -> dict:
    p = Path("config/sources.yaml")
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def build_google_news_queries() -> list[tuple[str, str]]:
    core = "battery (cathode OR anode OR electrolyte OR separator OR solid-state OR sodium-ion OR recycling)"
    scale = "battery (gigafactory OR plant OR production OR capacity OR investment OR supply agreement)"
    policy = "battery (policy OR regulation OR subsidy OR tariff OR IRA OR CBAM)"
    queries = [("Google News - Core", core), ("Google News - Scale", scale), ("Google News - Policy", policy)]
    return [(name, google_news_rss_url(q, hl="en", gl="US", ceid="US:en")) for name, q in queries]


def _default_naver_queries() -> list[str]:
    return [
        "배터리 2차전지",
        "전고체 배터리",
        "나트륨이온 배터리",
        "배터리 재활용 블랙매스",
        "양극재 전구체 LFP NCM",
        "음극재 흑연 실리콘",
        "전해질 첨가제",
        "분리막",
        "ESS 배터리",
        "배터리 관세 IRA",
    ]


def _get_naver_queries() -> list[str]:
    s = (os.getenv("NAVER_QUERIES") or "").strip()
    if not s:
        return _default_naver_queries()
    return [x.strip() for x in s.split(",") if x.strip()]


def collect_naver_items(target_date: str, need: int) -> list[dict[str, Any]]:
    naver_id = (os.getenv("NAVER_CLIENT_ID") or "").strip()
    naver_secret = (os.getenv("NAVER_CLIENT_SECRET") or "").strip()
    if not naver_id or not naver_secret:
        print("[WARN] NAVER_CLIENT_ID/SECRET not set. Skipping NAVER.")
        return []

    import src.naver_collector as nc  # local module

    fetch_n = getenv_int("NAVER_FETCH_N", 150)
    queries = _get_naver_queries()

    picked, scores, stats = nc.collect_naver_top15_last24h_deduped_and_ranked(
        client_id=naver_id,
        client_secret=naver_secret,
        queries=queries,
        fetch_n=fetch_n,
        top_k=need,
    )
    print(f"[INFO] NAVER stats: {stats}")

    items: list[dict[str, Any]] = []
    for it in picked:
        published_at = target_date
        pub_dt = getattr(it, "published_dt_kst", None)
        if pub_dt:
            try:
                published_at = pub_dt.date().isoformat()
            except Exception:
                published_at = target_date

        items.append(
            {
                "title": getattr(it, "title", ""),
                "description": getattr(it, "description", ""),
                "link": getattr(it, "link", ""),
                "source": getattr(it, "source", "NAVER"),
                "published_at": published_at,
                "provider": "naver",
                "related_links": [],
                "popularity_signal": "unknown",
            }
        )
    return items


def collect_google_items(target_date: str, need: int, cfg: dict) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []

    for source_name, url in build_google_news_queries():
        got = collect_from_rss(url, source_name, target_date)
        for it in got:
            it["provider"] = "google"
            it.setdefault("related_links", [])
            it["popularity_signal"] = popularity_signal_from_source(it.get("source", ""))
        raw.extend(got)

    for src in cfg.get("rss_sources", {}).get("fixed", []):
        name = src.get("name", "RSS")
        url = src.get("url", "")
        if not url:
            continue
        got = collect_from_rss(url, name, target_date)
        for it in got:
            it["provider"] = "rss"
            it.setdefault("related_links", [])
            it["popularity_signal"] = popularity_signal_from_source(it.get("source", ""))
        raw.extend(got)

    if not raw:
        return []

    deduped = dedupe_items(raw, sim_threshold=0.88)
    picked = deduped[:need]
    print(f"[INFO] GOOGLE/RSS: raw={len(raw)} deduped={len(deduped)} picked={len(picked)}")
    return picked


def main():
    cfg = load_config()

    target_date = kst_yesterday_date_str()

    naver_count = getenv_int("NAVER_COUNT", 10)
    google_count = getenv_int("GOOGLE_COUNT", 10)

    max_items = getenv_int("MAX_ITEMS", naver_count + google_count)
    min_items = getenv_int("MIN_ITEMS", min(10, max_items))

    # 1) Collect
    naver_items = collect_naver_items(target_date, need=naver_count)
    google_items = collect_google_items(target_date, need=google_count, cfg=cfg)

    raw = naver_items + google_items

    # Fill shortage best-effort (if one side missing)
    if len(naver_items) < naver_count:
        raw.extend(google_items[google_count : google_count + (naver_count - len(naver_items))])
    if len(google_items) < google_count:
        raw.extend(naver_items[naver_count : naver_count + (google_count - len(google_items))])

    # final dedupe across sources
    raw = dedupe_items(raw, sim_threshold=0.88)

    # 2) Enrich with tier/category/companies(dictionary) + scoring
    for it in raw:
        it["tier"] = infer_tier(it.get("link", ""), cfg)
        it["category"] = classify_category(it.get("title", ""), it.get("description", ""))
        it["companies"] = extract_companies(it.get("title", ""), it.get("description", ""), max_n=3)

        it.setdefault("popularity_signal", popularity_signal_from_source(it.get("source", "")))
        if it.get("related_links") and it["popularity_signal"] == "unknown":
            it["popularity_signal"] = "multi_source"

        it["score"] = score_item(it, tier=int(it["tier"]), multi_source_hits=1 + len(it.get("related_links", [])))

    raw.sort(key=sort_key)
    items = raw[:max_items]
    if len(items) < min_items:
        print(f"[WARN] Only {len(items)} items found (min requested {min_items}).")

    # 3) Gemini enrichment (batch inside llm_enrich_gemini.py)
    items = enrich_items(items, max_items=len(items))

    # merge companies again with dict match
    for it in items:
        llm_comps = it.get("companies") or []
        dict_comps = extract_companies(it.get("title", ""), it.get("description", ""), max_n=3)
        merged = []
        for c in llm_comps + dict_comps:
            if c and c not in merged:
                merged.append(c)
        it["companies"] = merged[:3]

    # 4) Write outputs
    out_dir = Path("outputs") / target_date
    md_path, json_path = write_outputs(out_dir, target_date, items)
    csv_path = write_daily_csv(out_dir, target_date, items)
    print(f"[OK] Wrote outputs: {md_path}, {json_path}, {csv_path}")

    # 5) Publish Pages HTML
    docs_dir = Path("docs")
    build_daily_page(target_date, items, docs_dir)
    build_root_index(docs_dir)
    print(f"[OK] Published Pages HTML: docs/{target_date}/ and docs/index.html")

    # 6) Upsert master DB
    data_dir = Path("data")
    upsert_master_csv(data_dir, items)
    upsert_master_json(data_dir, items)
    print("[OK] Upserted master DB: data/news_master.csv, data/news_master.json")


if __name__ == "__main__":
    main()
