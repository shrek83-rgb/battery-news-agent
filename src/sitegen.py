from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _escape(s: str) -> str:
    return html.escape(s or "", quote=True)


def build_daily_page(date_str: str, items: list[dict[str, Any]], docs_dir: Path) -> Path:
    day_dir = docs_dir / date_str
    (day_dir / "cards").mkdir(parents=True, exist_ok=True)

    # Cards are expected at: docs/YYYY-MM-DD/cards/01.png ...
    # We'll just reference them.
    cards = []
    for i, it in enumerate(items, 1):
        title = _escape(it.get("title", ""))
        link = _escape(it.get("link", ""))
        source = _escape(it.get("source", ""))
        category = _escape(it.get("category", "기타"))
        tier = _escape(str(it.get("tier", 3)))
        pub = _escape(it.get("published_at", date_str))
        img_name = f"{i:02d}.png"

        cards.append(f"""
          <div class="card">
            <a class="imglink" href="{link}" target="_blank" rel="noopener noreferrer">
              <img src="cards/{img_name}" alt="{title}">
            </a>
            <div class="meta">
              <div class="tag">{category} · Tier {tier}</div>
              <a class="title" href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>
              <div class="sub">{source} · {pub}</div>
              <a class="btn" href="{link}" target="_blank" rel="noopener noreferrer">원문 보기 →</a>
            </div>
          </div>
        """)

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>배터리 카드뉴스 - {date_str}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #fff; color: #111; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 18px 60px; }}
    .top {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; }}
    h1 {{ margin: 0; font-size: 22px; }}
    .nav a {{ color:#111; text-decoration:none; border-bottom:1px solid #ddd; }}
    .grid {{ display:grid; grid-template-columns: 1fr; gap: 18px; margin-top: 20px; }}
    @media (min-width: 860px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
    .card {{ border:1px solid #eee; border-radius:16px; overflow:hidden; display:flex; flex-direction:column; }}
    .imglink img {{ width:100%; height:auto; display:block; }}
    .meta {{ padding: 14px 14px 16px; }}
    .tag {{ font-size: 12px; color:#555; margin-bottom: 6px; }}
    .title {{ display:block; font-size: 16px; line-height: 1.35; color:#111; text-decoration:none; margin-bottom: 6px; }}
    .title:hover {{ text-decoration: underline; }}
    .sub {{ font-size: 12px; color:#666; margin-bottom: 10px; }}
    .btn {{ display:inline-block; padding: 8px 10px; border:1px solid #ddd; border-radius:10px; text-decoration:none; color:#111; font-size: 13px; }}
    .btn:hover {{ background:#f7f7f7; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <h1>배터리 카드뉴스 ({date_str} 전날 기준)</h1>
      <div class="nav"><a href="../index.html">전체 아카이브</a></div>
    </div>
    <div class="grid">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""
    out_path = day_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def build_root_index(docs_dir: Path) -> Path:
    # list date folders that contain index.html
    dates = []
    for p in docs_dir.iterdir():
        if p.is_dir() and (p / "index.html").exists():
            dates.append(p.name)
    dates.sort(reverse=True)

    links = "\n".join(
        f'<li><a href="{_escape(d)}/index.html">{_escape(d)}</a></li>'
        for d in dates
    )

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>배터리 카드뉴스 아카이브</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#fff; color:#111; }}
    .wrap {{ max-width: 900px; margin: 0 auto; padding: 28px 18px 60px; }}
    h1 {{ margin: 0 0 10px; font-size: 22px; }}
    p {{ margin: 0 0 18px; color:#555; }}
    ul {{ margin: 0; padding-left: 18px; }}
    a {{ color:#111; text-decoration:none; border-bottom:1px solid #ddd; }}
    a:hover {{ border-bottom-color:#111; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>배터리 카드뉴스 아카이브</h1>
    <p>날짜를 클릭하면 해당 날짜 카드뉴스 페이지로 이동합니다.</p>
    <ul>
      {links}
    </ul>
  </div>
</body>
</html>
"""
    out_path = docs_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def load_items_from_output_json(json_path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    date_str = data.get("date_range") or json_path.stem.split("_")[-1]
    items = data.get("items") or []
    return date_str, items
