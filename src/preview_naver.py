# src/preview_naver.py
from __future__ import annotations

import os

import src.naver_collector as nc


def main():
    cid = os.getenv("NAVER_CLIENT_ID", "").strip()
    csec = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요합니다.")

    query = os.getenv("NAVER_QUERY", "배터리 2차전지").strip()
    fetch_n = int(os.getenv("NAVER_FETCH_N", "150"))
    top_k = int(os.getenv("NAVER_TOP_K", "15"))

    print(f"[ENV] GEMINI_API_KEY set: {bool(os.getenv('GEMINI_API_KEY','').strip())}")
    print(f"[NAVER] query='{query}' fetch_n={fetch_n} top_k={top_k}")

    # 1) raw 150 (24h)
    raw = nc.collect_naver_last24h(
        client_id=cid,
        client_secret=csec,
        query=query,
        max_fetch=fetch_n,
        sort="date",
    )
    print(f"[STEP] raw_count={len(raw)}")

    # 2) LLM dedupe (or fallback)
    deduped = nc.dedupe_by_llm_semantic(raw) if hasattr(nc, "dedupe_by_llm_semantic") else raw
    print(f"[STEP] deduped_count={len(deduped)} (dropped={len(raw)-len(deduped)})")

    # 3) importance scoring (or fallback)
    if hasattr(nc, "score_importance_by_llm"):
        scores = nc.score_importance_by_llm(deduped)
        print(f"[STEP] scored_count={len(scores)}")
        idxs = list(range(len(deduped)))
        idxs.sort(key=lambda i: (-scores[i], getattr(deduped[i], "rank", i)))
        picked = [deduped[i] for i in idxs[:top_k]]
        picked_scores = [scores[i] for i in idxs[:top_k]]
    else:
        picked = deduped[:top_k]
        picked_scores = None

    # 제목만 출력(요청사항). 점수도 보고 싶으면 SHOW_SCORE=1
    show_score = os.getenv("SHOW_SCORE", "0") == "1"

    print("---- titles ----")
    for i, it in enumerate(picked, 1):
        if show_score and picked_scores is not None:
            print(f"{i:02d}. ({picked_scores[i-1]}) {it.title}")
        else:
            print(f"{i:02d}. {it.title}")


if __name__ == "__main__":
    main()
