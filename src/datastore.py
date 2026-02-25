# src/datastore.py 에 추가/교체

from __future__ import annotations
import csv, json, hashlib
from pathlib import Path
from typing import Any


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _uid_from_item(it: dict[str, Any]) -> str:
    link = (it.get("link") or "").strip()
    norm = link  # 이미 normalize_url 했다고 가정
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def upsert_master_csv(data_dir: Path, items: list[dict[str, Any]]) -> Path:
    """
    data/news_master.csv 하나를 계속 업데이트(중복 제거 + 최신값 반영)
    key = uid(sha1(link))
    """
    _ensure_dir(data_dir)
    p = data_dir / "news_master.csv"

    fields = ["uid", "date", "title", "source", "link", "category", "tier", "companies",
              "summary_1", "summary_2", "summary_3", "popularity_signal"]

    # 1) 기존 로드
    existing: dict[str, dict[str, str]] = {}
    if p.exists():
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                uid = row.get("uid")
                if uid:
                    existing[uid] = row

    # 2) 신규/업데이트 병합
    for it in items:
        uid = _uid_from_item(it)
        summ = it.get("summary_3_sentences") or ["", "", ""]
        while len(summ) < 3:
            summ.append("")
        comps = it.get("companies") or []

        existing[uid] = {
            "uid": uid,
            "date": it.get("published_at", ""),
            "title": it.get("title", ""),
            "source": it.get("source", ""),
            "link": it.get("link", ""),
            "category": it.get("category", ""),
            "tier": str(it.get("tier", "")),
            "companies": "; ".join(comps),
            "summary_1": summ[0],
            "summary_2": summ[1],
            "summary_3": summ[2],
            "popularity_signal": it.get("popularity_signal", "unknown"),
        }

    # 3) 날짜 최신순으로 정렬해 다시 저장
    rows = list(existing.values())
    rows.sort(key=lambda x: (x.get("date",""), x.get("source",""), x.get("title","")), reverse=True)

    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    return p


def upsert_master_json(data_dir: Path, items: list[dict[str, Any]]) -> Path:
    """
    data/news_master.json 하나를 계속 업데이트(중복 제거 + 최신값 반영)
    key = uid(sha1(link))
    """
    _ensure_dir(data_dir)
    p = data_dir / "news_master.json"

    existing: dict[str, dict[str, Any]] = {}
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for it in data:
                uid = it.get("uid")
                if uid:
                    existing[uid] = it

    for it in items:
        uid = _uid_from_item(it)
        merged = dict(it)
        merged["uid"] = uid
        existing[uid] = merged

    rows = list(existing.values())
    rows.sort(key=lambda x: (x.get("published_at",""), x.get("source",""), x.get("title","")), reverse=True)

    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
