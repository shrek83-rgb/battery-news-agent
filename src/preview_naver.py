# src/preview_naver.py
from __future__ import annotations

import os
from .naver_collector import collect_naver_top15_last24h_deduped


def main():
    cid = os.getenv("NAVER_CLIENT_ID", "").strip()
    csec = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요합니다.")

    # LLM dedupe requires GEMINI_API_KEY
    if not os.getenv("GEMINI_API_KEY", "").strip():
        print("[WARN] GEMINI_API_KEY is not set. Dedupe will fallback to string similarity and may be weaker.")

    query = os.getenv("NAVER_QUERY", "배터리 2차전지").strip()
    fetch_n = int(os.getenv("NAVER_FETCH_N", "150"))
    top_k = int(os.getenv("NAVER_TOP_K", "15"))

    picked = collect_naver_top15_last24h_deduped(
        client_id=cid,
        client_secret=csec,
        query=query,
        fetch_n=fetch_n,
        top_k=top_k,
    )

    print(f"[NAVER] query='{query}' / fetched<= {fetch_n} / top={top_k}")
    print("---- titles ----")
    for i, it in enumerate(picked, 1):
        print(f"{i:02d}. {it.title}")


if __name__ == "__main__":
    main()
