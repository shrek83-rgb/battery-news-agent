# src/preview_naver_enrich.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import src.naver_collector as nc
from src.tagger import classify_category, extract_companies
from src.llm_enrich_gemini import enrich_items


def _default_queries() -> List[str]:
    # 150개 채우기용(최근 24h)
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


def _collect_top_items(
    client_id: str,
    client_secret: str,
    top_k: int,
    fetch_n: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    naver_collector.py 버전이 여러 번 바뀌었을 수 있어서,
    가능한 함수 시그니처를 모두 지원하도록 안전하게 호출합니다.
    """
    # 1) 가장 최신(queries 리스트) 버전
    if hasattr(nc, "collect_naver_top15_last24h_deduped_and_ranked"):
        fn = nc.collect_naver_top15_last24h_deduped_and_ranked

        queries_env = os.getenv("NAVER_QUERIES", "").strip()
        if queries_env:
            queries = [q.strip() for q in queries_env.split(",") if q.strip()]
        else:
            queries = _default_queries()

        try:
            # (client_id, client_secret, queries=[...], fetch_n=150, top_k=15) 형태
            picked, scores, stats = fn(
                client_id=client_id,
                client_secret=client_secret,
                queries=queries,
                fetch_n=fetch_n,
                top_k=top_k,
            )
            return picked, {"scores": scores, "stats": stats, "mode": "queries"}
        except TypeError:
            # 2) 구버전(query 단일 문자열) 형태로 재시도
            query = os.getenv("NAVER_QUERY", "배터리 2차전지").strip()
            picked, scores = fn(
                client_id=client_id,
                client_secret=client_secret,
                query=query,
                fetch_n=fetch_n,
                top_k=top_k,
            )
            return picked, {"scores": scores, "stats": {}, "mode": "single_query"}

    # 3) 최후의 fallback: raw 수집만이라도(제목/설명/링크는 확보)
    query = os.getenv("NAVER_QUERY", "배터리 2차전지").strip()
    raw = nc.collect_naver_last24h(
        client_id=client_id,
        client_secret=client_secret,
        query=query,
        max_fetch=fetch_n,
        sort="date",
    )
    picked = raw[:top_k]
    return picked, {"scores": None, "stats": {"raw_count": len(raw)}, "mode": "raw_fallback"}


def main():
    cid = os.getenv("NAVER_CLIENT_ID", "").strip()
    csec = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요합니다.")

    top_k = int(os.getenv("NAVER_TOP_K", "15"))
    fetch_n = int(os.getenv("NAVER_FETCH_N", "150"))

    print(f"[ENV] GEMINI_API_KEY set: {bool(os.getenv('GEMINI_API_KEY','').strip())}")
    print(f"[NAVER] fetch_n={fetch_n} top_k={top_k}")

    picked_items, meta = _collect_top_items(cid, csec, top_k=top_k, fetch_n=fetch_n)
    print(f"[INFO] collection_mode={meta.get('mode')}")
    if meta.get("stats"):
        print(f"[INFO] stats={meta['stats']}")

    # NaverNewsItem -> dict (pipeline format)
    items: List[Dict[str, Any]] = []
    for it in picked_items:
        # dataclass 형태를 가정하지만, 혹시 dict일 수도 있어 방어
        title = getattr(it, "title", None) or it.get("title", "")
        desc = getattr(it, "description", None) or it.get("description", "")
        link = getattr(it, "link", None) or it.get("link", "")
        source = getattr(it, "source", None) or it.get("source", "")
        pub_dt = getattr(it, "published_dt_kst", None)

        published_at = ""
        if pub_dt:
            try:
                published_at = pub_dt.date().isoformat()
            except Exception:
                published_at = ""

        item = {
            "title": title,
            "description": desc,
            "link": link,
            "source": source,
            "published_at": published_at,
            "provider": "naver",
        }

        # 분야/기업(사전) 먼저 채움
        item["category"] = classify_category(title, desc)
        item["companies"] = extract_companies(title, desc, max_n=3)

        items.append(item)

    # Gemini로 summary + 기업(가능하면) enrich
    # max_items=top_k로 제한: 프리뷰는 15개만
    items = enrich_items(items, max_items=top_k)

    # 기업은 (Gemini + 사전) 합치기
    for it in items:
        llm_comps = it.get("companies") or []
        dict_comps = extract_companies(it.get("title",""), it.get("description",""), max_n=3)
        merged = []
        for c in llm_comps + dict_comps:
            if c and c not in merged:
                merged.append(c)
        it["companies"] = merged[:3]

        # summary 3개 보정
        s = it.get("summary_3_sentences") or ["", "", ""]
        while len(s) < 3:
            s.append("")
        it["summary_3_sentences"] = s[:3]

    # 출력(제목 + 분야 + 기업 + 3문장 요약)
    show_link = os.getenv("SHOW_LINK", "0") == "1"
    show_source = os.getenv("SHOW_SOURCE", "1") == "1"

    print("---- enriched preview ----")
    for i, it in enumerate(items, 1):
        title = it.get("title", "")
        category = it.get("category", "기타")
        comps = it.get("companies") or []
        s1, s2, s3 = (it.get("summary_3_sentences") or ["", "", ""])[:3]
        source = it.get("source", "")
        link = it.get("link", "")

        print(f"{i:02d}. {title}")
        if show_source:
            print(f"    - 언론사: {source}")
        print(f"    - 분야: {category}")
        print(f"    - 관련기업: {', '.join(comps) if comps else '-'}")
        print(f"    - 요약1: {s1}")
        print(f"    - 요약2: {s2}")
        print(f"    - 요약3: {s3}")
        if show_link:
            print(f"    - 링크: {link}")
        print("")

    print("[OK] Preview done.")


if __name__ == "__main__":
    main()
