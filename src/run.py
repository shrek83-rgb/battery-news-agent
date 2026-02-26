# src/run.py
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .collector import collect_from_rss, google_news_rss_url
from .naver_collector import collect_naver_news_yesterday
from .dedupe import dedupe_items
from .ranker import infer_tier, popularity_signal_from_source, score_item, sort_key
from .tagger import classify_category, extract_companies
from .llm_enrich_gemini import enrich_items
from .renderer import write_outputs
from .datastore import write_daily_csv, upsert_master_csv, upsert_master_json
from .sitegen import build_daily_page, build_root_index
from .utils import getenv_int, kst_yesterday_date_str


def load_config() -> dict:
    return yaml.safe_load(Path("config/sources.yaml").read_text(encoding="utf-8"))


def load_keywords() -> list[str]:
    p = Path("config/keywords.txt")
    if not p.exists():
        return []
    kws = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            kws.append(line)
    return kws


def build_google_news_queries() -> list[tuple[str, str]]:
    core = "battery (cathode OR anode OR electrolyte OR separator OR solid-state OR sodium-ion OR recycling)"
    scale = "battery (gigafactory OR plant OR production OR capacity OR investment OR supply agreement)"
    policy = "battery (policy OR regulation OR subsidy OR tariff OR IRA OR CBAM)"
    queries = [("Google News - Core", core), ("Google News - Scale", scale), ("Google News - Policy", policy)]
    urls = [(name, google_news_rss_url(q, hl="en", gl="US", ceid="US:en")) for name, q in queries]
    return urls


def main():
    target_date = kst_yesterday_date_str()

    min_items = getenv_int("MIN_ITEMS", 10)
    max_items = getenv_int("MAX_ITEMS", 20)

    naver_need = int(os.getenv("NAVER_COUNT", "10"))
    google_need = int(os.getenv("GOOGLE_COUNT", "10"))

    cfg = load_config()
    _ = load_keywords()  # reserved for future use

    # -----------------------------
    # 1) Collect NAVER (API)
    # -----------------------------
    naver_raw = []
    naver_id = os.getenv("NAVER_CLIENT_ID", "")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET", "")
    if naver_id and naver_secret:
        naver_queries = [
            "배터리 2차전지",
            "전고체 배터리",
            "나트륨이온 배터리",
            "배터리 재활용",
            "양극재 음극재 전해질 분리막",
        ]
        try:
            naver_raw = collect_naver_news_yesterday(
                target_date=target_date,
                client_id=naver_id,
                client_secret=naver_secret,
                queries=naver_queries,
                need=naver_need,
            )
            print(f"[INFO] Collected {len(naver_raw)} from NAVER API")
        except Exception as e:
            print(f"[WARN] NAVER collection failed: {e}")
    else:
        print("[WARN] NAVER_CLIENT_ID/SECRET not set. Skipping Naver collection.")

    # -----------------------------
    # 2) Collect GOOGLE (RSS)
    # -----------------------------
    google_raw = []
    for source_name, rss_url in build_google_news_queries():
        got = collect_from_rss(rss_url, source_name, target_date)
        for it in got:
            it["provider"] = "google"
            it.setdefault("related_links", [])
            it["popularity_signal"] = popularity_signal_from_source(it.get("source", ""))
        google_raw.extend(got)
    print(f"[INFO] Collected {len(google_raw)} from GOOGLE RSS")

    # Optional fixed RSS sources (if configured)
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
        google_raw.extend(got)

    # -----------------------------
    # 3) Dedupe within each provider, then pick 10/10
    # -----------------------------
    naver_deduped = dedupe_items(naver_raw, sim_threshold=0.88)
    google_deduped = dedupe_items(google_raw, sim_threshold=0.88)

    naver_pick = naver_deduped[:naver_need]
    google_pick = google_deduped[:google_need]

    raw = naver_pick + google_pick

    # 부족분 보충(중복은 dedupe로 어느 정도 제거됨)
    if len(naver_pick) < naver_need:
        raw.extend(google_deduped[google_need : google_need + (naver_need - len(naver_pick))])
    if len(google_pick) < google_need:
        raw.extend(naver_deduped[naver_need : naver_need + (google_need - len(google_pick))])

    # 마지막 전체 dedupe 한 번 더
    raw = dedupe_items(raw, sim_threshold=0.88)

    # -----------------------------
    # 4) Tier + Category + Companies(dictionary) + Scoring
    # -----------------------------
    for it in raw:
        it["tier"] = infer_tier(it.get("link", ""), cfg)
        it["category"] = classify_category(it.get("title", ""), it.get("description", ""))
        # dictionary-based companies (will be merged with Gemini output later)
        it["companies"] = extract_companies(it.get("title", ""), it.get("description", ""), max_n=3)

        # popularity signal: keep if already set, else infer
        it.setdefault("popularity_signal", popularity_signal_from_source(it.get("source", "")))

        # multi_source signal if related links exist
        if it.get("related_links") and it["popularity_signal"] == "unknown":
            it["popularity_signal"] = "multi_source"

        it["score"] = score_item(it, tier=int(it["tier"]), multi_source_hits=1 + len(it.get("related_links", [])))

    # sort with enforced tier priority
    raw.sort(key=sort_key)

    # cut to max_items (ensure min_items best-effort)
    items = raw[:max_items]
    if len(items) < min_items:
        print(f"[WARN] Only {len(items)} items found (min requested {min_items}).")

    # -----------------------------
    # 5) Gemini enrichment: summary + companies
    # -----------------------------
    items = enrich_items(items, max_items=max_items)

    # Merge Gemini companies with dictionary companies again (dedupe)
    for it in items:
        llm_comps = it.get("companies") or []
        dict_comps = extract_companies(it.get("title", ""), it.get("description", ""), max_n=3)
        merged = []
        for c in llm_comps + dict_comps:
            if c and c not in merged:
                merged.append(c)
        it["companies"] = merged[:3]

    # -----------------------------
    # 6) Write outputs (daily)
    # -----------------------------
    out_dir = Path("outputs") / target_date
    md_path, json_path = write_outputs(out_dir, target_date, items)
    csv_path = write_daily_csv(out_dir, target_date, items)
    print(f"[OK] Wrote: {md_path}, {json_path}, {csv_path}")

    # -----------------------------
    # 7) Publish Pages HTML
    # -----------------------------
    docs_dir = Path("docs")
    build_daily_page(target_date, items, docs_dir)
    build_root_index(docs_dir)
    print(f"[OK] Published HTML pages under docs/{target_date}/")

    # -----------------------------
    # 8) Upsert master DB files (single continuously updated file)
    # -----------------------------
    data_dir = Path("data")
    upsert_master_csv(data_dir, items)
    upsert_master_json(data_dir, items)
    print("[OK] Upserted master DB files: data/news_master.csv, data/news_master.json")


if __name__ == "__main__":
    main()
