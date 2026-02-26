# src/preview_naver.py
from __future__ import annotations

import os
from src.naver_collector import collect_naver_top15_last24h_deduped_and_ranked


def main():
    cid = os.getenv("NAVER_CLIENT_ID", "").strip()
    csec = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not cid or not csec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요합니다.")

    if not os.getenv("GEMINI_API_KEY", "").strip():
        print("[WARN] GEMINI_API_KEY not set. Clustering/importance will be weak.")

    # 여러 쿼리로 150개까지 채우기
    queries = [
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

    fetch_n = int(os.getenv("NAVER_FETCH_N", "150"))
    top_k = int(os.getenv("NAVER_TOP_K", "15"))

    picked, scores, stats = collect_naver_top15_last24h_deduped_and_ranked(
        client_id=cid,
        client_secret=csec,
        queries=queries,
        fetch_n=fetch_n,
        top_k=top_k,
    )

    print(f"[ENV] GEMINI_API_KEY set: {bool(os.getenv('GEMINI_API_KEY','').strip())}")
    print(f"[STEP] raw_count={stats['raw_count']} deduped_count={stats['deduped_count']} dropped={stats['dropped']} picked={stats['picked']}")

    print("---- titles ----")
    show_score = os.getenv("SHOW_SCORE", "0") == "1"
    for i, it in enumerate(picked, 1):
        if show_score:
            print(f"{i:02d}. ({scores[i-1]}) {it.title}")
        else:
            print(f"{i:02d}. {it.title}")


if __name__ == "__main__":
    main()
