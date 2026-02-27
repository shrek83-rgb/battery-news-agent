# src/run.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .utils import getenv_int, kst_yesterday_date_str
from .collector import collect_from_rss, google_news_rss_url
from .dedupe import dedupe_items
from .ranker import infer_tier, popularity_signal_from_source
from .tagger import classify_category, extract_companies
from .llm_enrich_gemini import enrich_items
from .renderer import write_outputs
from .datastore import write_daily_csv, upsert_master_csv, upsert_master_json
from .sitegen import build_daily_page, build_root_index


# -----------------------------
# Logging helpers
# -----------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _t() -> float:
    return time.monotonic()


# -----------------------------
# Config
# -----------------------------
def load_config() -> dict:
    p = Path("config/sources.yaml")
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    low = n.lower().strip()
    # normalize common typos
    low = low.replace("light", "lite")
    return low


def get_models() -> Dict[str, str]:
    """
    Single place to manage models across modules.
    Env priority:
      GEMINI_MODEL (base)
      GEMINI_MODEL_DEDUPE
      GEMINI_MODEL_RANK
      GEMINI_MODEL_SUMMARY (optional override for summary)
    """
    base = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    dedupe = _normalize_model_name(os.getenv("GEMINI_MODEL_DEDUPE", base))
    rank = _normalize_model_name(os.getenv("GEMINI_MODEL_RANK", base))
    summary = _normalize_model_name(os.getenv("GEMINI_MODEL_SUMMARY", base))
    return {"base": base, "dedupe": dedupe, "rank": rank, "summary": summary}


def build_google_news_queries() -> list[tuple[str, str]]:
    # Broad battery-related query set; downstream LLM will filter by relevance/importance.
    core = "battery (cathode OR anode OR electrolyte OR separator OR solid-state OR sodium-ion OR recycling OR LFP OR NCM OR precursor OR lithium)"
    scale = "battery (gigafactory OR plant OR production OR capacity OR investment OR supply agreement OR off-take OR MOU OR JV OR merger OR acquisition)"
    policy = "battery (policy OR regulation OR subsidy OR tariff OR IRA OR CBAM OR export control OR sanctions)"
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


# -----------------------------
# Google/RSS: collect candidates, then LLM select by battery relevance + monitoring importance
# -----------------------------
def _monitor_score(rel: int, imp: int) -> float:
    # You can tune weights later
    return 0.7 * float(rel) + 0.3 * float(imp)


def _select_google_by_llm_battery_relevance(
    candidates: List[Dict[str, Any]],
    top_k: int,
    models: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Uses naver_collector's LLM scoring utilities if available, otherwise fallback to simple ranking.
    Strategy:
      - limit candidates size
      - call one-shot LLM title scoring (relevance+importance+event_key)
      - pick representative per event_key
      - select top_k by monitor_score
    """
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or not candidates:
        picked = candidates[:top_k]
        for it in picked:
            it["battery_relevance"] = 60
            it["monitoring_importance"] = 10
            it["monitor_score"] = int(round(_monitor_score(60, 10)))
        return picked

    # Optional: import helper from naver_collector (keeps one place of schema/prompt if you implemented there)
    # But to avoid circular complexity, we do local minimal one-shot scoring here via google-genai.
    try:
        from google import genai
        from pydantic import BaseModel, Field
        from typing import List as _List

        class _TitleScore(BaseModel):
            index: int
            event_key: str
            battery_relevance: int = Field(..., ge=0, le=100)
            monitoring_importance: int = Field(..., ge=0, le=100)

        class _Resp(BaseModel):
            items: _List[_TitleScore]

        client = genai.Client(api_key=api_key)
        titles = [(i, (candidates[i].get("title") or "").strip()) for i in range(len(candidates))]

        prompt = (
            "다음은 뉴스 제목 목록입니다. 각 제목에 대해 배터리 산업 연관성과 모니터링 중요도를 JSON으로만 출력하세요.\n\n"
            "출력 스키마:\n"
            "{\n"
            "  \"items\": [\n"
            "    {\"index\": 0, \"event_key\": \"...\", \"battery_relevance\": 0-100, \"monitoring_importance\": 0-100}\n"
            "  ]\n"
            "}\n\n"
            "규칙:\n"
            "- event_key: 같은 사건/이슈면 같은 키(ASCII letters/digits/_). 표현이 달라도 동일 사건이면 동일 키.\n"
            "- battery_relevance(0~100): 배터리 산업과의 직접 연관성.\n"
            "- monitoring_importance(0~100): 산업 모니터링 관점 파급력.\n"
            "- 모든 index(0..N-1)에 대해 1개씩 출력.\n"
            "- 제목만 보고 판단.\n\n"
            "[제목 목록]\n"
            + "\n".join([f"{i}: {t}" for i, t in titles])
        )

        resp = client.models.generate_content(
            model=models["rank"],
            contents=prompt,
            config={"temperature": 0.2},
        )

        text = (resp.text or "").strip()
        # best-effort JSON extraction
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s : e + 1]

        parsed = _Resp.model_validate_json(text)

        rel_by_i: Dict[int, int] = {}
        imp_by_i: Dict[int, int] = {}
        ek_by_i: Dict[int, str] = {}
        for x in parsed.items:
            rel_by_i[int(x.index)] = int(x.battery_relevance)
            imp_by_i[int(x.index)] = int(x.monitoring_importance)
            ek = (x.event_key or f"item_{x.index}").strip()[:80] or f"item_{x.index}"
            ek_by_i[int(x.index)] = ek

        # representative per event
        by_event: Dict[str, List[int]] = {}
        for i in range(len(candidates)):
            by_event.setdefault(ek_by_i.get(i, f"item_{i}"), []).append(i)

        reps: List[int] = []
        for ek, idxs in by_event.items():
            rep = max(
                idxs,
                key=lambda j: (_monitor_score(rel_by_i.get(j, 0), imp_by_i.get(j, 0)), -j),
            )
            reps.append(rep)

        # apply relevance threshold
        relevance_min = int(os.getenv("BATTERY_RELEVANCE_MIN", "55"))
        reps = [i for i in reps if rel_by_i.get(i, 0) >= relevance_min] or reps

        reps.sort(
            key=lambda j: (
                _monitor_score(rel_by_i.get(j, 0), imp_by_i.get(j, 0)),
                rel_by_i.get(j, 0),
                imp_by_i.get(j, 0),
            ),
            reverse=True,
        )

        picked_idxs = reps[:top_k]
        picked: List[Dict[str, Any]] = []
        for i in picked_idxs:
            it = dict(candidates[i])
            rel = rel_by_i.get(i, 0)
            imp = imp_by_i.get(i, 0)
            it["battery_relevance"] = rel
            it["monitoring_importance"] = imp
            it["monitor_score"] = int(round(_monitor_score(rel, imp)))
            picked.append(it)

        return picked

    except Exception as e:
        _log(f"[WARN] GOOGLE LLM select failed -> fallback. err={e}")
        picked = candidates[:top_k]
        for it in picked:
            it["battery_relevance"] = 60
            it["monitoring_importance"] = 10
            it["monitor_score"] = int(round(_monitor_score(60, 10)))
        return picked


def collect_google_candidates(target_date: str, cfg: dict) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []

    # google news queries
    for source_name, url in build_google_news_queries():
        got = collect_from_rss(url, source_name, target_date)
        for it in got:
            it["provider"] = "google"
            it.setdefault("related_links", [])
            it["popularity_signal"] = popularity_signal_from_source(it.get("source", ""))
        raw.extend(got)

    # fixed RSS sources in config
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

    # basic dedupe by title similarity
    return dedupe_items(raw, sim_threshold=float(os.getenv("GOOGLE_DEDUPE_THRESHOLD", "0.88")))


def collect_google_items(target_date: str, need: int, cfg: dict, models: Dict[str, str]) -> List[Dict[str, Any]]:
    cand = collect_google_candidates(target_date, cfg)
    _log(f"[INFO] GOOGLE candidates: {len(cand)}")

    if not cand:
        return []

    cand_limit = int(os.getenv("GOOGLE_CANDIDATES", "90"))
    cand = cand[:cand_limit]

    picked = _select_google_by_llm_battery_relevance(cand, top_k=need, models=models)
    _log(f"[INFO] GOOGLE picked: {len(picked)}")
    return picked


# -----------------------------
# NAVER: use naver_collector module (your current version)
# -----------------------------
def collect_naver_items(target_date: str, need: int, models: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    naver_id = (os.getenv("NAVER_CLIENT_ID") or "").strip()
    naver_secret = (os.getenv("NAVER_CLIENT_SECRET") or "").strip()
    if not naver_id or not naver_secret:
        _log("[WARN] NAVER_CLIENT_ID/SECRET not set. Skipping NAVER.")
        return [], {"raw_count": 0, "deduped_count": 0, "dropped": 0, "picked": 0}

    import src.naver_collector as nc

    fetch_n = getenv_int("NAVER_FETCH_N", 150)
    queries = _get_naver_queries()

    # NOTE: Your naver_collector already supports dedupe+rank (title-only).
    picked, picked_scores, stats = nc.collect_naver_top15_last24h_deduped_and_ranked(
        client_id=naver_id,
        client_secret=naver_secret,
        queries=queries,
        fetch_n=fetch_n,
        top_k=need,
    )

    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(picked):
        pub_dt = getattr(it, "published_dt_kst", None)
        published_at = target_date
        if pub_dt:
            try:
                published_at = pub_dt.date().isoformat()
            except Exception:
                published_at = target_date

        d: Dict[str, Any] = {
            "title": getattr(it, "title", ""),
            "description": getattr(it, "description", ""),
            "link": getattr(it, "link", ""),
            "source": getattr(it, "source", "NAVER"),
            "published_at": published_at,
            "provider": "naver",
            "related_links": [],
            "popularity_signal": "unknown",
        }

        # If your naver_collector returns per-item scores in stats later, you can map them here.
        # picked_scores is importance score (0-100)
        if idx < len(picked_scores):
            d["monitor_score"] = int(picked_scores[idx])
            d["monitoring_importance"] = int(picked_scores[idx])

        out.append(d)

    stats = dict(stats or {})
    stats.setdefault("picked", len(out))
    stats.setdefault("models", {"base": models["base"], "dedupe": models["dedupe"], "rank": models["rank"]})
    return out, stats


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    _log("[BOOT] src.run started")
    t0 = _t()

    cfg = load_config()
    target_date = kst_yesterday_date_str()

    models = get_models()
    _log(f"[ENV] GEMINI_API_KEY set: {bool((os.getenv('GEMINI_API_KEY') or '').strip())}")
    _log(f"[INFO] Gemini models (BASE/DEDUP/RANK/SUMMARY): {models['base']} / {models['dedupe']} / {models['rank']} / {models['summary']}")
    _log(f"[CONFIG] target_date={target_date}")

    naver_count = getenv_int("NAVER_COUNT", 10)
    google_count = getenv_int("GOOGLE_COUNT", 10)
    max_items = getenv_int("MAX_ITEMS", naver_count + google_count)
    min_items = getenv_int("MIN_ITEMS", min(10, max_items))

    _log(f"[CONFIG] counts NAVER={naver_count} GOOGLE={google_count} MAX={max_items} MIN={min_items}")

    # 1) Collect NAVER
    _log("[STEP] Collect NAVER (24h via API -> LLM dedupe/rank) ...")
    t1 = _t()
    naver_items, naver_stats = collect_naver_items(target_date, need=naver_count, models=models)
    _log(f"[DONE] NAVER items={len(naver_items)} in {(_t()-t1):.1f}s | stats={naver_stats}")

    # 2) Collect GOOGLE/RSS candidates and pick by battery relevance + importance
    _log("[STEP] Collect GOOGLE/RSS candidates -> LLM pick by battery relevance ...")
    t2 = _t()
    google_items = collect_google_items(target_date, need=google_count, cfg=cfg, models=models)
    _log(f"[DONE] GOOGLE/RSS picked={len(google_items)} in {(_t()-t2):.1f}s")

    # 3) Merge and global dedupe
    raw = naver_items + google_items
    raw = dedupe_items(raw, sim_threshold=float(os.getenv("GLOBAL_DEDUPE_THRESHOLD", "0.88")))
    _log(f"[STEP] After merge dedupe: raw={len(raw)}")

    # 4) Add tier/category/companies and ensure basic score fields exist
    _log("[STEP] Add tier/category/companies ...")
    for it in raw:
        it["tier"] = infer_tier(it.get("link", ""), cfg)
        it["category"] = classify_category(it.get("title", ""), it.get("description", ""))
        it["companies"] = extract_companies(it.get("title", ""), it.get("description", ""), max_n=3)

        it.setdefault("popularity_signal", popularity_signal_from_source(it.get("source", "")))
        it.setdefault("monitor_score", it.get("monitor_score", 0))

    # 5) Sort and pick final MAX
    def _sort_key(x: Dict[str, Any]) -> Tuple[int, int, int]:
        # tier asc, monitor_score desc, (optional) relevance desc
        tier = int(x.get("tier", 3))
        mscore = int(x.get("monitor_score", 0))
        rel = int(x.get("battery_relevance", 0)) if x.get("battery_relevance") is not None else 0
        return (tier, -mscore, -rel)

    raw.sort(key=_sort_key)
    items = raw[:max_items]

    _log(f"[STATS] items_final={len(items)} "
         f"providers(naver={sum(1 for x in items if x.get('provider')=='naver')}, "
         f"google={sum(1 for x in items if x.get('provider')=='google')}, "
         f"rss={sum(1 for x in items if x.get('provider')=='rss')})")

    if len(items) < min_items:
        _log(f"[WARN] Only {len(items)} items found (min requested {min_items}). Will still write outputs.")

    # 6) Gemini batch enrichment for summaries + companies
    # NOTE: llm_enrich_gemini.enrich_items already does per-item fallback to content-based summary.
    _log("[STEP] Gemini enrich (summary_3_sentences + companies) ...")
    t3 = _t()
    items = enrich_items(items, max_items=len(items), model=models["summary"])
    _log(f"[DONE] Gemini enrich finished in {(_t()-t3):.1f}s")

    # Ensure all have 3-sentence summary list
    summaries_present = sum(1 for x in items if x.get("summary_3_sentences"))
    _log(f"[STATS] summaries_present={summaries_present}/{len(items)}")

    # 7) Write outputs (ALWAYS write, even if small)
    _log("[STEP] Write outputs + pages + db ...")
    _log(f"[WRITE] target_date={target_date}")
    _log(f"[WRITE] outputs_dir=outputs/{target_date} docs_dir=docs/{target_date}")

    out_dir = Path("outputs") / target_date
    md_path, json_path = write_outputs(out_dir, target_date, items)
    csv_path = write_daily_csv(out_dir, target_date, items)

    docs_dir = Path("docs")
    build_daily_page(target_date, items, docs_dir)
    build_root_index(docs_dir)

    data_dir = Path("data")
    upsert_master_csv(data_dir, items)
    upsert_master_json(data_dir, items)

    _log(f"[OK] Wrote outputs: {md_path}, {json_path}, {csv_path}")
    _log(f"[OK] Published Pages HTML: docs/{target_date}/ and docs/index.html")
    _log("[OK] Upserted master DB: data/news_master.csv, data/news_master.json")

    _log(f"[DONE] total {(_t()-t0):.1f}s")


if __name__ == "__main__":
    main()
