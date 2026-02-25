from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import NewsItem


def render_md(date_str: str, items: list[dict[str, Any]]) -> str:
    lines = []
    lines.append(f"# 배터리 뉴스 데일리 브리핑 ({date_str} 전날 기준)")
    lines.append("")
    lines.append(f"- 총 {len(items)}건 (우선순위: 공시/보도자료(1) > 주요 언론(2) > 업계/기타(3))")
    lines.append("")
    lines.append("| # | 출처등급 | 분야 | 발행일 | 매체 | 제목 | 링크 | 인기신호 |")
    lines.append("|---:|:---:|:---|:---:|:---|:---|:---|:---|")

    for i, it in enumerate(items, 1):
        title = it["title"].replace("|", "\\|")
        link = it["link"]
        source = it["source"].replace("|", "\\|")
        lines.append(
            f"| {i} | {it['tier']} | {it['category']} | {it['published_at']} | {source} | {title} | {link} | {it['popularity_signal']} |"
        )

    lines.append("")
    lines.append("## 상세 요약")
    lines.append("")

    for i, it in enumerate(items, 1):
        lines.append(f"### {i}. {it['title']}")
        lines.append(f"- 발행일: {it['published_at']}")
        lines.append(f"- 매체: {it['source']} (출처등급 {it['tier']})")
        lines.append(f"- 분야: {it['category']}")
        lines.append(f"- 링크: {it['link']}")
        if it.get("related_links"):
            lines.append(f"- 참고 링크: " + ", ".join(it["related_links"]))
        lines.append(f"- 인기신호: {it['popularity_signal']}")
        lines.append("- 3문장 요약:")
        for s in it["summary_3_sentences"]:
            lines.append(f"  - {s}")
        lines.append("")

    return "\n".join(lines)


def write_outputs(base_dir: Path, date_str: str, items: list[dict[str, Any]]) -> tuple[Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    md_path = base_dir / f"battery_news_{date_str}.md"
    json_path = base_dir / f"battery_news_{date_str}.json"

    md_path.write_text(render_md(date_str, items), encoding="utf-8")
    payload = {"date_range": date_str, "items": items}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return md_path, json_path
