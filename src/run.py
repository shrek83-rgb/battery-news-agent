from __future__ import annotations

# from .cardnews import generate_cards
from .sitegen import build_daily_page, build_root_index
import os
from pathlib import Path

import yaml

from .collector import collect_from_rss, google_news_rss_url
from .dedupe import dedupe_items
from .ranker import infer_tier, popularity_signal_from_source, score_item, sort_key
from .tagger import classify_category, summarize_3_sentences
from .renderer import write_outputs
# from .drive_uploader import ensure_date_folder, upload_or_update_file
from .utils import getenv_int, kst_yesterday_date_str

def load_config() -> dict:
    cfg_path = Path("config/sources.yaml")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def load_keywords() -> list[str]:
    p = Path("config/keywords.txt")
    kws = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            kws.append(line)
    return kws


def build_google_news_queries(keywords: list[str]) -> list[tuple[str, str]]:
    """
    Return list of (source_name, rss_url).
    We'll use 2~3 focused queries instead of 1 huge query to avoid dilution.
    """
    # Core query: battery + major topics
    core = "battery (cathode OR anode OR electrolyte OR separator OR solid-state OR sodium-ion OR recycling)"
    # Optional: include manufacturing/scale signals
    scale = "battery (gigafactory OR plant OR production OR capacity OR investment OR supply agreement)"
    # Optional: policy/regulation
    policy = "battery (policy OR regulation OR subsidy OR tariff OR IRA OR CBAM)"

    queries = [("Google News - Core", core), ("Google News - Scale", scale), ("Google News - Policy", policy)]
    urls = [(name, google_news_rss_url(q, hl="en", gl="US", ceid="US:en")) for name, q in queries]
    return urls


def main():
    target_date = kst_yesterday_date_str()
    min_items = getenv_int("MIN_ITEMS", 10)
    max_items = getenv_int("MAX_ITEMS", 20)

    cfg = load_config()
    keywords = load_keywords()  # currently unused in query builder; kept for later fine-tuning

    raw: list[dict] = []

    # 1) Google News RSS (query based)
    for source_name, rss_url in build_google_news_queries(keywords):
        items = collect_from_rss(rss_url, source_name, target_date)
        print(f"[INFO] Collected {len(items)} from {source_name}")
        raw.extend(collect_from_rss(rss_url, source_name, target_date))

    # 2) Fixed RSS sources (from config)
    for it in cfg.get("rss_sources", {}).get("fixed", []):
        name = it.get("name", "RSS")
        url = it.get("url", "")
        if url:
            items = collect_from_rss(rss_url, source_name, target_date)
            print(f"[INFO] Collected {len(items)} from {source_name}")
            raw.extend(collect_from_rss(url, name, target_date))

    if not raw:
        print(f"[WARN] No items collected for {target_date}.")
        # still write empty outputs and upload (optional)
        items = []
    else:
        # normalize + enrich
        for it in raw:
            it["popularity_signal"] = popularity_signal_from_source(it.get("source", ""))

        # 3) dedupe
        deduped = dedupe_items(raw, sim_threshold=0.88)

        # 4) tier + tag + summary + scoring
        for it in deduped:
            it["tier"] = infer_tier(it.get("link", ""), cfg)
            it["category"] = classify_category(it.get("title", ""), it.get("description", ""))
            it["summary_3_sentences"] = summarize_3_sentences(it.get("title", ""), it.get("description", ""), it.get("source", ""))

            # If dedupe merged references, treat as multi_source signal lightly
            if it.get("related_links"):
                if it["popularity_signal"] == "unknown":
                    it["popularity_signal"] = "multi_source"

            it["score"] = score_item(it, tier=it["tier"], multi_source_hits=1 + len(it.get("related_links", [])))

        # 5) sort with enforced priority
        deduped.sort(key=sort_key)

        # 6) cut 10~20
        items = deduped[:max_items]
        if len(items) < min_items:
            # keep as many as possible; in production you might expand sources/queries
            print(f"[WARN] Only {len(items)} items found (min requested {min_items}).")

    # 7) write outputs to outputs/YYYY-MM-DD/
    out_dir = Path("outputs") / target_date
    md_path, json_path = write_outputs(out_dir, target_date, items)
    print(f"[OK] Wrote: {md_path} , {json_path}")

    # 7.5) Generate card images into outputs/YYYY-MM-DD/cards/
    # created_cards = generate_cards(target_date, items, out_dir)
    # print(f"[OK] Created {len(created_cards)} cards under {out_dir / 'cards'}")

    # 7.6) Publish to docs/ for GitHub Pages
    # docs_dir = Path("docs")
    # day_docs_dir = docs_dir / target_date
    # (day_docs_dir / "cards").mkdir(parents=True, exist_ok=True)

    # Copy cards to docs/YYYY-MM-DD/cards/
    for p in created_cards:
        (day_docs_dir / "cards" / p.name).write_bytes(p.read_bytes())

    # Build docs/YYYY-MM-DD/index.html (cards link to original news)
    build_daily_page(target_date, items, docs_dir)

    # Build docs/index.html (archive)
    build_root_index(docs_dir)

    print(f"[OK] Published pages under docs/{target_date}/ and docs/index.html")

    # 8) upload to Drive date folder
    # parent_folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
    # if not parent_folder_id:
    #     raise RuntimeError("Missing env: GDRIVE_FOLDER_ID")

    # date_folder_id = ensure_date_folder(parent_folder_id, target_date, supports_all_drives=True)

    # upload_or_update_file(md_path, date_folder_id, mime_type="text/markdown", supports_all_drives=True)
    # upload_or_update_file(json_path, date_folder_id, mime_type="application/json", supports_all_drives=True)
    # print(f"[OK] Uploaded to Drive folder: {target_date}/")


if __name__ == "__main__":
    main()
