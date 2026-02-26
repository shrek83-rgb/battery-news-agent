# src/preview_naver.py
from __future__ import annotations

import os
from .naver_collector import collect_naver_top15_last24h_deduped_and_ranked


def main():
    cid = os.getenv("NAVER_CLIENT_ID", "").strip()
    csec = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요합니다.")

    if not os.getenv("GEMINI_API_KEY", "").strip():
        print("[WARN] GEMINI_API_KEY not set. Dedupe/importance will fallback to heuristics and may be weaker.")

    query = os.getenv("NAVER_QUERY", "배터리 2차전지").strip()
    fetch_n = int(os.getenv("NAVER_FETCH_N", "150"))
    top_k = int(os.getenv("NAVER_TOP_K", "15"))

    picked, scores = collect_naver_top15_last24h_deduped_and_ranked(
        client_id=cid,
        client_secret=csec,
        query=query,
        fetch_n=fetch_n,
        top_k=top_k,
    )

    print(f"[NAVER] query='{query}' / fetched<= {fetch_n} / picked={top_k}")
    print("---- titles ----")
    # 제목만 출력(요청사항). 점수도 보고 싶으면 SHOW_SCORE=1 설정.
    show_score = os.getenv("SHOW_SCORE", "0") == "1"
    for i, it in enumerate(picked, 1):
        if show_score:
            print(f"{i:02d}. ({scores[i-1]}) {it.title}")
        else:
            print(f"{i:02d}. {it.title}")


if __name__ == "__main__":
    main()
