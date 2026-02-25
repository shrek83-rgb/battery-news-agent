from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_daily_csv(out_dir: Path, date_str: str, items: list[dict[str, Any]]) -> Path:
    _ensure_dir(out_dir)
    p = out_dir / f"battery_news_{date_str}.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "title", "source", "link", "category", "tier", "companies", "summary_1", "summary_2", "summary_3", "popularity_signal"])
        for it in items:
            summ = it.get("summary_3_sentences") or ["", "", ""]
            while len(summ) < 3:
                summ.append("")
            comps = it.get("companies") or []
            w.writerow([
                date_str,
                it.get("title", ""),
                it.get("source", ""),
                it.get("link", ""),
                it.get("category", ""),
                it.get("tier", ""),
                "; ".join(comps),
                summ[0], summ[1], summ[2],
                it.get("popularity_signal", "unknown"),
            ])
    return p


def append_master_csv(data_dir: Path, items: list[dict[str, Any]]) -> Path:
    _ensure_dir(data_dir)
    p = data_dir / "news_master.csv"
    exists = p.exists()
    with p.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["date", "title", "source", "link", "category", "tier", "companies", "summary_1", "summary_2", "summary_3", "popularity_signal"])
        for it in items:
            date_str = it.get("published_at", "")
            summ = it.get("summary_3_sentences") or ["", "", ""]
            while len(summ) < 3:
                summ.append("")
            comps = it.get("companies") or []
            w.writerow([
                date_str,
                it.get("title", ""),
                it.get("source", ""),
                it.get("link", ""),
                it.get("category", ""),
                it.get("tier", ""),
                "; ".join(comps),
                summ[0], summ[1], summ[2],
                it.get("popularity_signal", "unknown"),
            ])
    return p


def append_master_jsonl(data_dir: Path, items: list[dict[str, Any]]) -> Path:
    _ensure_dir(data_dir)
    p = data_dir / "news_master.jsonl"
    with p.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return p
