from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _e(s: str) -> str:
    return html.escape(s or "", quote=True)


def _icon_svg(kind: str) -> str:
    # 간단한 라인 아이콘 (SVG) - 외부 리소스 없이 깔끔하게
    # kind: category 기반으로 매핑
    icons = {
        "cathode": "M12 2v20M2 12h20M6 6l12 12M18 6L6 18",  # X + cross
        "anode": "M4 18l8-14 8 14H4z",                    # triangle
        "electrolyte": "M8 2h8v6l3 4v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V12l3-4V2z",  # bottle-ish
        "separator": "M5 4h14v16H5z M8 4v16 M12 4v16 M16 4v16",  # grid
        "전고체": "M7 7h10v10H7z M4 12h3 M17 12h3 M12 4v3 M12 17v3",  # solid block
        "나트륨": "M6 7h12M6 12h12M6 17h12",                      # lines
        "재활용": "M8 7l4-4 4 4M16 17l-4 4-4-4M4 12a8 8 0 0 1 8-8M20 12a8 8 0 0 1-8 8",  # recycle-ish
        "장비": "M4 17h16M7 17V7h10v10M9 9h6",                   # machine-ish
        "정책": "M7 4h10v6H7z M6 10h12v10H6z",                   # doc
        "기타": "M12 2a10 10 0 1 0 0.001 0z M9 10a3 3 0 1 1 6 0c0 2-3 2-3 5 M12 18h.01",  # question
    }
    path = icons.get(kind, icons["기타"])
    return f"""
    <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
      <path d="{path}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """


def build_daily_page(date_str: str, items: list[dict[str, Any]], docs_dir: Path) -> Path:
    day_dir = docs_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    # 10~15개로 보이도록(기본값: items가 이미 MAX_ITEMS로 컷됨)
    cards_html = []
    for i, it in enumerate(items, 1):
        title = _e(it.get("title", ""))
        link = _e(it.get("link", ""))
        source = _e(it.get("source", ""))
        category = _e(it.get("category", "기타"))
        tier = _e(str(it.get("tier", 3)))
        pub = _e(it.get("published_at", date_str))
        pop = _e(it.get("popularity_signal", "unknown"))
        summ = it.get("summary_3_sentences") or []
        summ = [s.strip() for s in summ if s and s.strip()]
        while len(summ) < 3:
            summ.append("")
        bullets = "".join(f"<li>{_e(s)}</li>" for s in summ[:3] if s)
        
        # ✅ 여기: companies를 가져와서 HTML로 변환
        companies = it.get("companies", []) or []
        comp_html = "".join(f'<span class="chip">{_e(c)}</span>' for c in companies) if companies else '<span class="chip muted">-</span>'

        cards_html.append(f"""
        <a class="card" href="{link}" target="_blank" rel="noopener noreferrer">
          <div class="cardTop">
            <div class="num">{i:02d}</div>
            <div class="badgeRow">
              <span class="badge">{category}</span>
              <span class="meta">Tier {tier} · {pop}</span>
            </div>
          </div>
          <div class="title">{title}</div>
          <ul class="bullets">{bullets}</ul>
          <div class="footer">
            <div class="src">{source}</div>
            <div class="date">{pub}</div>
          </div>
          <div class="icon">{_icon_svg(it.get("category","기타"))}</div>
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
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
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
    .sub a:hover {{ border-bottom-color: var(--blue); }}

    .section {{
      padding: 18px 26px 10px;
      background: var(--panel);
      border-top: 1px solid var(--line);
    }}
    .sectionTitle {{
      display:inline-block;
      background: var(--yellow);
      border-radius: 14px;
      padding: 10px 14px;
      font-weight: 900;
      border: 3px solid rgba(11,78,219,.18);
    }}

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
      position: relative;
      display:block;
      text-decoration:none;
      color:inherit;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 16px 14px;
      box-shadow: 0 10px 20px rgba(10,27,61,.06);
      transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
      min-height: 210px;
    }}
    .card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 14px 26px rgba(10,27,61,.12);
      border-color: rgba(11,78,219,.25);
    }}
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
      background: rgba(11,78,219,.10);
      color: var(--blue);
      border-radius: 999px;
      font-weight: 800;
      font-size: 12px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .title {{
      font-size: 16px;
      font-weight: 900;
      line-height: 1.35;
      letter-spacing: -0.01em;
      margin-bottom: 10px;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow:hidden;
    }}
    .bullets {{
      margin:0;
      padding-left: 18px;
      color: #223253;
      font-size: 13px;
      line-height: 1.45;
      display: -webkit-box;
      -webkit-line-clamp: 4;
      -webkit-box-orient: vertical;
      overflow:hidden;
    }}
    .footer {{
      position:absolute;
      left:16px; right:16px; bottom:12px;
      display:flex; justify-content:space-between; gap:10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .src {{
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      max-width: 70%;
    }}
    .icon {{
      position:absolute;
      right:14px; top: 62px;
      width:46px; height:46px;
      border-radius: 16px;
      background: rgba(255,210,74,.55);
      display:flex; align-items:center; justify-content:center;
      color: var(--blue);
      border: 1px solid rgba(11,78,219,.12);
    }}
    .bottom {{
      padding: 14px 26px 20px;
      background: #fff;
      border-top: 1px solid var(--line);
      display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
      color: var(--muted);
      font-size: 12px;
    }}
    .bottom a {{ color: var(--blue); text-decoration:none; border-bottom:1px solid rgba(11,78,219,.25); }}
    .rows { margin-top: 10px; display: grid; gap: 8px; }
    .row { display:flex; gap:10px; align-items:flex-start; }
    .k { width:64px; flex:0 0 auto; font-size:12px; color: var(--muted); font-weight:800; }
    .v { flex:1; font-size:13px; color:#223253; line-height:1.35; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; }
    .chip {
      display:inline-flex; align-items:center;
      padding:6px 10px;
      border-radius:999px;
      border:1px solid rgba(11,78,219,.14);
      background: rgba(11,78,219,.06);
      color:#0B4EDB;
      font-weight:800;
      font-size:12px;
    }
    .chip.muted {
      background: #f2f4f8;
      color:#6b7280;
      border-color:#e5e7eb;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="frame">
      <div class="head">
        <div class="pill"><span class="dot">N</span> 오늘의 배터리 카드뉴스</div>
        <h1>{date_str} 전날 뉴스 TOP</h1>
        <div class="sub">
          카드/제목 클릭 시 원문이 새 탭에서 열립니다. ·
          <a href="../index.html">전체 아카이브</a>
        </div>
      </div>

      <div class="section">
        <div class="sectionTitle">뉴스 카드 목록 (2열)</div>
      </div>

      <div class="grid">
        {''.join(cards_html)}
      </div>

      <div class="bottom">
        <div>우선순위: 공시/보도자료(1) &gt; 주요 언론(2) &gt; 업계/기타(3)</div>
        <div><a href="../index.html">아카이브로 이동</a></div>
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
        f'<li style="margin:10px 0;"><a style="color:#0B4EDB;text-decoration:none;border-bottom:1px solid rgba(11,78,219,.25);" href="{_e(d)}/index.html">{_e(d)}</a></li>'
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
