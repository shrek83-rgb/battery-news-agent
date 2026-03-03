from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _escape_md_cell(s: Any) -> str:
    """Escape markdown table cell content."""
    if s is None:
        return ""
    s = str(s)
    return s.replace("|", "\\|").strip()


def _related_links_to_urls(x: Any) -> list[str]:
    """
    related_links can be:
      - ["https://...", ...]
      - [{"link": "https://..."}, ...]
      - [{"url": "https://..."}, ...]
      - "https://..."
      - None
    normalize to list[str]
    """
    out: list[str] = []
    if not x:
        return out

    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []

    if isinstance(x, list):
        for v in x:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
            elif isinstance(v, dict):
                s = (v.get("link") or v.get("url") or "").strip()
                if s:
                    out.append(s)

    # de-dup preserve order
    uniq: list[str] = []
    for u in out:
        if u not in uniq:
            uniq.append(u)
    return uniq


def _summary_to_list(x: Any) -> list[str]:
    """
    summary_3_sentences can be:
      - ["...", "...", "..."]
      - "..."
      - None
    normalize to list[str] (prefer up to 3 lines, but allow longer if already present)
    """
    if not x:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return []


def render_md(date_str: str, items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"# 배터리 뉴스 데일리 브리핑 ({date_str} 전날 기준)")
    lines.append("")
    lines.append(f"- 총 {len(items)}건 (우선순위: 공시/보도자료(1) > 주요 언론(2) > 업계/기타(3))")
    lines.append("")
    lines.append("| # | 출처등급 | 분야 | 발행일 | 매체 | 제목 | 링크 | 인기신호 |")
    lines.append("|---:|:---:|:---|:---:|:---|:---|:---|:---|")

    for i, it in enumerate(items, 1):
        title = _escape_md_cell(it.get("title", ""))
        link = (it.get("link") or "").strip()
        source = _escape_md_cell(it.get("source", ""))
        tier = _escape_md_cell(it.get("tier", ""))
        category = _escape_md_cell(it.get("category", ""))
        published_at = _escape_md_cell(it.get("published_at", ""))
        pop = _escape_md_cell(it.get("popularity_signal", "unknown"))

        lines.append(
            f"| {i} | {tier} | {category} | {published_at} | {source} | {title} | {link} | {pop} |"
        )

    lines.append("")
    lines.append("## 상세 요약")
    lines.append("")

    for i, it in enumerate(items, 1):
        title_raw = (it.get("title") or "").strip()
        lines.append(f"### {i}. {title_raw}")

        lines.append(f"- 발행일: {it.get('published_at', '')}")
        lines.append(f"- 매체: {it.get('source', '')} (출처등급 {it.get('tier', '')})")
        lines.append(f"- 분야: {it.get('category', '')}")
        lines.append(f"- 링크: {it.get('link', '')}")

        related_urls = _related_links_to_urls(it.get("related_links"))
        if related_urls:
            # 요구사항: 참고 링크 최대 2개
            lines.append("- 참고 링크: " + ", ".join(related_urls[:2]))

        lines.append(f"- 인기신호: {it.get('popularity_signal', 'unknown')}")

        lines.append("- 3문장 요약:")
        summary = _summary_to_list(it.get("summary_3_sentences"))
        if summary:
            for s in summary:
                lines.append(f"  - {s}")
        else:
            lines.append("  - (요약 없음)")

        # optionally show companies if present
        companies = it.get("companies") or []
        if isinstance(companies, list):
            comps = [str(c).strip() for c in companies if str(c).strip()]
            if comps:
                lines.append(f"- 관련 기업: {', '.join(comps[:3])}")

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
