from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _e(s: str) -> str:
    return html.escape(s or "", quote=True)


def _cat_slug(cat: str) -> str:
    m = {
        "cathode": "cathode",
        "anode": "anode",
        "electrolyte": "electrolyte",
        "separator": "separator",
        "전고체": "solid_state",
        "나트륨": "sodium",
        "재활용": "recycling",
        "장비": "equipment",
        "정책": "policy",
        "기타": "etc",
    }
    return m.get(cat, "etc")


def build_daily_page(date_str: str, items: list[dict[str, Any]], docs_dir: Path) -> Path:
    day_dir = docs_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    cards_html = []
    for i, it in enumerate(items, 1):
        title = _e(it.get("title", ""))
        link = _e(it.get("link", ""))
        source = _e(it.get("source", ""))
        category_raw = it.get("category", "기타")
        category = _e(category_raw)
        tier = _e(str(it.get("tier", 3)))
        pub = _e(it.get("published_at", date_str))

        companies = it.get("companies", []) or []
        comp_html = "".join(f'<span class="chip">{_e(c)}</span>' for c in companies) if companies else '<span class="chip muted">-</span>'

        summ = it.get("summary_3_sentences") or ["", "", ""]
        while len(summ) < 3:
            summ.append("")
        summ_html = "".join(f"<li>{_e(s)}</li>" for s in summ[:3] if s)

        cat_class = f"cat-{_cat_slug(category_raw)}"

        cards_html.append(f"""
        <a class="card {cat_class}" href="{link}" target="_blank" rel="noopener noreferrer">
          <div class="cardTop">
            <div class="num">{i:02d}</div>
            <div class="badgeRow">
              <span class="badge">{category}</span>
              <span class="meta">Tier {tier}</span>
            </div>
          </div>

          <div class="title">{title}</div>

          <div class="rows">
            <div class="row">
              <div class="k">언론사</div>
              <div class="v">{source}</div>
            </div>
            <div class="row">
              <div class="k">관련기업</div>
              <div class="v chips">{comp_html}</div>
            </div>
          </div>

          <div class="summaryBox">
            <div class="summaryTitle">3문장 요약</div>
            <ul class="summary">{summ_html}</ul>
          </div>

          <div class="footer">
            <div>{pub}</div>
          </div>
        </a>
        """)

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>배터리 카드뉴스 - {date_str}</title>
  <style>
    :root {{
      --blue:#0B4EDB;
      --blue2:#0A43C2;
      --yellow:#FFD24A;
      --ink:#0A1B3D;
      --muted:#5B6B88;
      --line:rgba(10,27,61,.10);
      --card:#FFFFFF;
      --panel:#F5F7FF;

      /* category colors */
      --cathode:#FF5A5F;
      --anode:#00A870;
      --electrolyte:#7C3AED;
      --separator:#2563EB;
      --solid:#F59E0B;
      --sodium:#06B6D4;
      --recycling:#10B981;
      --equipment:#6B7280;
      --policy:#EF4444;
      --etc:#64748B;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: linear-gradient(180deg,var(--blue),var(--blue2));
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 16px 60px;
    }}
    .frame {{
      background: var(--card);
      border-radius: 22px;
      border: 6px solid rgba(255,255,255,.25);
      box-shadow: 0 14px 40px rgba(0,0,0,.18);
      overflow:hidden;
    }}
    .head {{
      padding: 26px 26px 18px;
      background: #fff;
    }}
    .pill {{
      display:inline-flex; align-items:center; gap:10px;
      background: var(--yellow);
      color: var(--ink);
      font-weight:800;
      border-radius: 14px;
      padding: 10px 14px;
      border: 3px solid rgba(11,78,219,.18);
    }}
    .pill .dot {{
      width:28px;height:28px;border-radius:10px;
      background: var(--blue);
      color:#fff; display:flex; align-items:center; justify-content:center;
      font-weight:900;
    }}
    h1 {{
      margin:16px 0 0;
      font-size: 34px;
      letter-spacing: -0.02em;
    }}
    .sub {{
      margin-top:10px; color: var(--muted); font-size: 14px; line-height:1.5;
    }}
    .sub a {{ color: var(--blue); text-decoration: none; border-bottom:1px solid rgba(11,78,219,.25); }}

    .grid {{
      padding: 18px 26px 26px;
      background: var(--panel);
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    @media (min-width: 900px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
    }}

    .card {{
      --cat: var(--etc);
      display:block;
      text-decoration:none;
      color:inherit;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 16px 14px;
      box-shadow: 0 10px 20px rgba(10,27,61,.06);
      transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
      border-left: 10px solid var(--cat);
    }}
    .card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 14px 26px rgba(10,27,61,.12);
      border-color: rgba(11,78,219,.25);
    }}

    /* category color binding */
    .cat-cathode {{ --cat: var(--cathode); }}
    .cat-anode {{ --cat: var(--anode); }}
    .cat-electrolyte {{ --cat: var(--electrolyte); }}
    .cat-separator {{ --cat: var(--separator); }}
    .cat-solid_state {{ --cat: var(--solid); }}
    .cat-sodium {{ --cat: var(--sodium); }}
    .cat-recycling {{ --cat: var(--recycling); }}
    .cat-equipment {{ --cat: var(--equipment); }}
    .cat-policy {{ --cat: var(--policy); }}
    .cat-etc {{ --cat: var(--etc); }}

    .cardTop {{
      display:flex; align-items:flex-start; justify-content:space-between; gap:10px;
      margin-bottom: 10px;
    }}
    .num {{
      width:44px;height:44px;border-radius:14px;
      background: var(--blue);
      color:#fff;
      display:flex; align-items:center; justify-content:center;
      font-weight: 900;
      letter-spacing:.02em;
      flex: 0 0 auto;
    }}
    .badgeRow {{
      flex:1;
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      padding-top: 2px;
    }}
    .badge {{
      display:inline-flex;
      padding: 6px 10px;
      background: color-mix(in srgb, var(--cat) 14%, white);
      color: var(--cat);
      border-radius: 999px;
      font-weight: 900;
      font-size: 12px;
      border: 1px solid color-mix(in srgb, var(--cat) 22%, white);
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      font-weight: 800;
    }}
    .title {{
      font-size: 16px;
      font-weight: 900;
      line-height: 1.35;
      letter-spacing: -0.01em;
      margin-bottom: 10px;
    }}

    .rows {{ margin-top: 8px; display: grid; gap: 8px; }}
    .row {{ display:flex; gap:10px; align-items:flex-start; }}
    .k {{ width:64px; flex:0 0 auto; font-size:12px; color: var(--muted); font-weight:900; }}
    .v {{ flex:1; font-size:13px; color:#223253; line-height:1.35; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
    .chip {{
      display:inline-flex; align-items:center;
      padding:6px 10px;
      border-radius:999px;
      border:1px solid rgba(11,78,219,.14);
      background: rgba(11,78,219,.06);
      color:#0B4EDB;
      font-weight:900;
      font-size:12px;
    }}
    .chip.muted {{
      background: #f2f4f8;
      color:#6b7280;
      border-color:#e5e7eb;
    }}

    .summaryBox {{
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .summaryTitle {{
      font-size: 12px;
      font-weight: 900;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .summary {{
      margin: 0;
      padding-left: 18px;
      color: #223253;
      font-size: 13px;
      line-height: 1.5;
    }}
    .footer {{
      margin-top: 12px;
      display:flex; justify-content:flex-end;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}

    .bottom {{
      padding: 14px 26px 20px;
      background: #fff;
      border-top: 1px solid var(--line);
      display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    .bottom a {{ color: var(--blue); text-decoration:none; border-bottom:1px solid rgba(11,78,219,.25); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="frame">
      <div class="head">
        <div class="pill"><span class="dot">N</span> 오늘의 배터리 카드뉴스</div>
        <h1>{date_str} 전날 뉴스 TOP</h1>
        <div class="sub">
          카드 클릭 시 원문이 새 탭에서 열립니다. · <a href="../">전체 아카이브</a>
        </div>
      </div>

      <div class="grid">
        {''.join(cards_html)}
      </div>

      <div class="bottom">
        <div>우선순위: 공시/보도자료(1) &gt; 주요 언론(2) &gt; 업계/기타(3)</div>
        <div><a href="../">아카이브로 이동</a></div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_path = day_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def build_root_index(docs_dir: Path) -> Path:
    dates = []
    for p in docs_dir.iterdir():
        if p.is_dir() and (p / "index.html").exists():
            dates.append(p.name)
    dates.sort(reverse=True)

    items = "\n".join(
        f'<li style="margin:10px 0;"><a style="color:#0B4EDB;text-decoration:none;border-bottom:1px solid rgba(11,78,219,.25);" href="{_e(d)}/">{_e(d)}</a></li>'
        for d in dates
    )

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>배터리 카드뉴스 아카이브</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0B4EDB; }}
    .wrap {{ max-width: 900px; margin: 0 auto; padding: 28px 16px 60px; }}
    .box {{ background:#fff; border-radius:22px; padding: 22px 22px 26px; box-shadow: 0 14px 40px rgba(0,0,0,.18); border: 6px solid rgba(255,255,255,.25); }}
    h1 {{ margin:0 0 8px; color:#0A1B3D; font-size: 22px; }}
    p {{ margin:0 0 16px; color:#5B6B88; }}
    ul {{ margin:0; padding-left: 18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="box">
      <h1>배터리 카드뉴스 아카이브</h1>
      <p>날짜를 클릭하면 해당 날짜 카드뉴스 페이지로 이동합니다.</p>
      <ul>{items}</ul>
    </div>
  </div>
</body>
</html>
"""
    out_path = docs_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
