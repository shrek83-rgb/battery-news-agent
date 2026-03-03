# src/run.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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
    low = low.replace("light", "lite")
    return low


def get_models() -> Dict[str, str]:
    """
    Single place to manage models across modules.
    Env priority:
      GEMINI_MODEL (base)
      GEMINI_MODEL_DEDUPE
      GEMINI_MODEL_RANK
      GEMINI_MODEL_SUMMARY
    """
    base = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    dedupe = _normalize_model_name(os.getenv("GEMINI_MODEL_DEDUPE", base))
    rank = _normalize_model_name(os.getenv("GEMINI_MODEL_RANK", base))
    summary = _normalize_model_name(os.getenv("GEMINI_MODEL_SUMMARY", base))
    return {"base": base, "dedupe": dedupe, "rank": rank, "summary": summary}


def build_google_news_queries() -> list[tuple[str, str]]:
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
# Scoring helpers
# -----------------------------
def _monitor_score(rel: int, imp: int) -> float:
    return 0.7 * float(rel) + 0.3 * float(imp)


def _pop_strength(sig: str) -> int:
    # sort helper: stronger popularity first
    s = (sig or "").lower()
    if s in ("most_read", "trending", "top_ranked"):
        return 3
    if s in ("multi_source",):
        return 2
    if s in ("unknown", "", None):
        return 0
    return 1


# -----------------------------
# GOOGLE/RSS: collect candidates, then LLM select by battery relevance + monitoring importance
# -----------------------------
def _select_google_by_llm_battery_relevance(
    candidates: List[Dict[str, Any]],
    top_k: int,
    models: Dict[str, str],
) -> List[Dict[str, Any]]:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or not candidates:
        picked = candidates[:top_k]
        for it in picked:
            it["battery_relevance"] = 60
            it["monitoring_importance"] = 10
            it["monitor_score"] = int(round(_monitor_score(60, 10)))
        return picked

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

        # one representative per event_key
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

        relevance_min = int(os.getenv("BATTERY_RELEVANCE_MIN", "60"))
        reps2 = [i for i in reps if rel_by_i.get(i, 0) >= relevance_min]
        reps = reps2 or reps

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

    return dedupe_items(raw, sim_threshold=float(os.getenv("GOOGLE_DEDUPE_THRESHOLD", "0.88")))


def collect_google_items(target_date: str, need: int, cfg: dict, models: Dict[str, str]) -> List[Dict[str, Any]]:
    cand = collect_google_candidates(target_date, cfg)
    _log(f"[INFO] GOOGLE candidates: {len(cand)}")
    if not cand:
        return []

    cand_limit = int(os.getenv("GOOGLE_CANDIDATES", "120"))
    cand = cand[:cand_limit]

    picked = _select_google_by_llm_battery_relevance(cand, top_k=need, models=models)
    _log(f"[INFO] GOOGLE picked: {len(picked)}")
    return picked


# -----------------------------
# NAVER
# -----------------------------
def collect_naver_items(target_date: str, need: int, models: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    naver_id = (os.getenv("NAVER_CLIENT_ID") or "").strip()
    naver_secret = (os.getenv("NAVER_CLIENT_SECRET") or "").strip()
    if not naver_id or not naver_secret:
        _log("[WARN] NAVER_CLIENT_ID/SECRET not set. Skipping NAVER.")
        return [], {"raw_count": 0, "deduped_count": 0, "dropped": 0, "picked": 0}

    from . import naver_collector as nc

    fetch_n = getenv_int("NAVER_FETCH_N", 150)
    queries = _get_naver_queries()

    picked, picked_scores, stats = nc.collect_naver_top_last24h_deduped_and_ranked(
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

        if idx < len(picked_scores):
            d["monitor_score"] = int(picked_scores[idx])
            d["monitoring_importance"] = int(picked_scores[idx])

        out.append(d)

    stats = dict(stats or {})
    stats.setdefault("picked", len(out))
    stats.setdefault("models", {"base": models["base"], "dedupe": models["dedupe"], "rank": models["rank"]})
    return out, stats


# -----------------------------
# GLOBAL: Strong dedupe by LLM event_key (1 call) -> fallback to sim dedupe
# -----------------------------
def global_dedupe_items(items: List[Dict[str, Any]], models: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Stronger dedupe than pure string similarity:
      - If GEMINI_API_KEY present: 1 request -> assign event_key to each title and keep 1 representative per event.
      - Representative preference: lower tier (1 better), higher popularity, higher monitor_score, then 최신.
      - On any failure/quota: fallback to dedupe_items(sim_threshold=GLOBAL_DEDUPE_THRESHOLD)
    """
    if not items:
        return items

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return dedupe_items(items, sim_threshold=float(os.getenv("GLOBAL_DEDUPE_THRESHOLD", "0.88")))

    # allow disable
    if (os.getenv("GLOBAL_LLM_DEDUPE", "1").strip() != "1"):
        return dedupe_items(items, sim_threshold=float(os.getenv("GLOBAL_DEDUPE_THRESHOLD", "0.88")))

    try:
        from google import genai
        from pydantic import BaseModel, Field
        from typing import List as _List

        class _EK(BaseModel):
            index: int
            event_key: str = Field(..., description="same event => same key, ASCII letters/digits/_")

        class _Resp(BaseModel):
            items: _List[_EK]

        client = genai.Client(api_key=api_key)
        titles = [(i, (items[i].get("title") or "").strip()[:200]) for i in range(len(items))]

        prompt = (
            "아래 뉴스 제목들을 '같은 사건/이슈' 단위로 묶기 위한 event_key를 부여하세요.\n"
            "조건:\n"
            "- event_key는 ASCII letters/digits/_ 만 사용.\n"
            "- 표현이 달라도 같은 사건이면 같은 event_key.\n"
            "- 단순히 주제가 비슷한 정도는 같은 event_key로 묶지 말 것.\n"
            "- 모든 index(0..N-1)를 반드시 1개씩 출력.\n"
            "- 반드시 JSON만 출력.\n\n"
            "출력: {\"items\": [{\"index\":0,\"event_key\":\"...\"}, ...]}\n\n"
            "[제목 목록]\n"
            + "\n".join([f"{i}: {t}" for i, t in titles])
        )

        resp = client.models.generate_content(
            model=models["dedupe"],
            contents=prompt,
            config={"temperature": 0.0},
        )

        text = (resp.text or "").strip()
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s : e + 1]

        parsed = _Resp.model_validate_json(text)

        ek_by_i: Dict[int, str] = {}
        for x in parsed.items:
            ek = (x.event_key or f"item_{x.index}").strip()[:80] or f"item_{x.index}"
            ek_by_i[int(x.index)] = ek

        def _rep_score(it: Dict[str, Any]) -> Tuple[int, int, int, str]:
            tier = int(it.get("tier", 3))  # lower is better
            pop = _pop_strength(it.get("popularity_signal", "unknown"))
            ms = int(it.get("monitor_score", 0))
            pub = str(it.get("published_at", ""))  # ISO string compare ok
            # we want: tier asc, pop desc, ms desc, pub desc
            return (tier, -pop, -ms, pub)

        grouped: Dict[str, List[int]] = {}
        for i in range(len(items)):
            grouped.setdefault(ek_by_i.get(i, f"item_{i}"), []).append(i)

        kept: List[Dict[str, Any]] = []
        for ek, idxs in grouped.items():
            # choose best representative
            best_i = min(idxs, key=lambda j: _rep_score(items[j]))
            rep = dict(items[best_i])

            # attach up to 2 related links
            related = []
            for j in idxs:
                if j == best_i:
                    continue
                link = (items[j].get("link") or "").strip()
                src = (items[j].get("source") or "").strip()
                if link and all(link != x.get("link") for x in related):
                    related.append({"source": src, "link": link})
                if len(related) >= 2:
                    break

            rep.setdefault("related_links", [])
            # keep existing + new unique
            seen = set((x.get("link") for x in rep["related_links"] if isinstance(x, dict)))
            for r in related:
                if r["link"] not in seen:
                    rep["related_links"].append(r)
                    seen.add(r["link"])
                if len(rep["related_links"]) >= 2:
                    break

            kept.append(rep)

        return kept

    except Exception as e:
        _log(f"[WARN] GLOBAL LLM dedupe failed -> fallback sim dedupe. err={e}")
        return dedupe_items(items, sim_threshold=float(os.getenv("GLOBAL_DEDUPE_THRESHOLD", "0.88")))


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

    max_items = int(os.getenv("MAX_ITEMS", "20"))
    min_items = int(os.getenv("MIN_ITEMS", "10"))

    # hard policy: 10 + 10 (within MAX)
    naver_count = min(10, max_items)
    google_count = min(10, max_items - naver_count)

    _log(f"[CONFIG] counts NAVER={naver_count} GOOGLE={google_count} MAX={max_items} MIN={min_items}")

    # 1) Collect NAVER
    _log("[STEP] Collect NAVER (24h via API -> LLM dedupe/rank) ...")
    t1 = _t()
    naver_items, naver_stats = collect_naver_items(target_date, need=naver_count, models=models)
    _log(f"[DONE] NAVER items={len(naver_items)} in {(_t()-t1):.1f}s | stats={naver_stats}")

    # 2) Collect GOOGLE/RSS
    _log("[STEP] Collect GOOGLE/RSS candidates -> LLM pick by battery relevance ...")
    t2 = _t()
    google_items = collect_google_items(target_date, need=google_count, cfg=cfg, models=models)
    _log(f"[DONE] GOOGLE/RSS picked={len(google_items)} in {(_t()-t2):.1f}s")

    # 3) Merge
    raw = naver_items + google_items

    # 4) Add tier/category/companies early (helps dedupe representative selection)
    _log("[STEP] Add tier/category/companies ...")
    for it in raw:
        it["tier"] = infer_tier(it.get("link", ""), cfg)
        it["category"] = classify_category(it.get("title", ""), it.get("description", ""))
        it["companies"] = extract_companies(it.get("title", ""), it.get("description", ""), max_n=3)
        it.setdefault("popularity_signal", popularity_signal_from_source(it.get("source", "")))
        it.setdefault("monitor_score", int(it.get("monitor_score", 0)))

    # 5) Strong global dedupe (LLM event_key if possible)
    _log("[STEP] Global dedupe (strong) ...")
    before = len(raw)
    raw = global_dedupe_items(raw, models=models)
    _log(f"[STEP] After global dedupe: {before} -> {len(raw)}")

    # 6) Sort and pick final MAX
    def _sort_key(x: Dict[str, Any]) -> Tuple[int, int, int, str]:
        tier = int(x.get("tier", 3))             # 1 best
        pop = _pop_strength(x.get("popularity_signal", "unknown"))
        mscore = int(x.get("monitor_score", 0))
        pub = str(x.get("published_at", ""))
        # tier asc, pop desc, mscore desc, pub desc
        return (tier, -pop, -mscore, pub)

    raw.sort(key=_sort_key)
    items = raw[:max_items]

    _log(
        f"[STATS] items_final={len(items)} "
        f"providers(naver={sum(1 for x in items if x.get('provider')=='naver')}, "
        f"google={sum(1 for x in items if x.get('provider')=='google')}, "
        f"rss={sum(1 for x in items if x.get('provider')=='rss')})"
    )

    if len(items) < min_items:
        _log(f"[WARN] Only {len(items)} items found (min requested {min_items}). Will still write outputs.")

    # 7) Gemini batch enrichment for summaries + companies
    _log("[STEP] Gemini enrich (summary_3_sentences + companies) ...")
    t3 = _t()
    items = enrich_items(items, max_items=len(items), model=models["summary"])
    _log(f"[DONE] Gemini enrich finished in {(_t()-t3):.1f}s")

    summaries_present = sum(1 for x in items if x.get("summary_3_sentences"))
    _log(f"[STATS] summaries_present={summaries_present}/{len(items)}")

    # 8) Write outputs + pages + DB
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
