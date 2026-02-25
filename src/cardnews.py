from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = (1080, 1080)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    # Prefer bundled Korean-capable font
    candidates = [
        Path("assets/fonts/Pretendard-Regular.ttf"),
        Path("assets/fonts/NotoSansKR-Regular.ttf"),
        # Ubuntu runner fallback (may not render Korean correctly)
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    # Last resort
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.replace("\n", " ").split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _ellipsize(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    s = text
    while s and draw.textlength(s + "…", font=font) > max_width:
        s = s[:-1]
    return (s + "…") if s else "…"


def generate_cards(date_str: str, items: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """
    Create 1080x1080 PNG cards.
    Returns list of created card paths.
    """
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    font_title = _load_font(54)
    font_body = _load_font(34)
    font_small = _load_font(28)

    created: list[Path] = []

    for idx, it in enumerate(items, 1):
        img = Image.new("RGB", CANVAS_SIZE, "white")
        draw = ImageDraw.Draw(img)

        margin = 70
        x = margin
        y = margin
        w = CANVAS_SIZE[0] - 2 * margin

        # Header (date + category)
        header = f"{date_str} | {it.get('category','기타')}"
        draw.text((x, y), header, font=font_small, fill="black")
        y += 55

        # Title (max 2 lines)
        title = (it.get("title") or "").strip()
        title_lines = _wrap_text(draw, title, font_title, w)
        if len(title_lines) > 2:
            title_lines = title_lines[:2]
            title_lines[-1] = _ellipsize(draw, title_lines[-1], font_title, w)

        for line in title_lines:
            draw.text((x, y), line, font=font_title, fill="black")
            y += 70

        y += 18
        # Divider
        draw.line((x, y, x + w, y), fill="black", width=2)
        y += 30

        # Summary (3 sentences)
        summ = it.get("summary_3_sentences") or []
        summ = [s.strip() for s in summ if s and s.strip()]
        # ensure 3 bullets
        while len(summ) < 3:
            summ.append("")

        bullet_indent = 28
        max_body_width = w - bullet_indent
        for s in summ[:3]:
            if not s:
                continue
            wrapped = _wrap_text(draw, s, font_body, max_body_width)
            # limit lines per bullet to avoid overflow
            wrapped = wrapped[:3]
            # bullet
            draw.text((x, y), "•", font=font_body, fill="black")
            yy = y
            for li, line in enumerate(wrapped):
                draw.text((x + bullet_indent, yy), line, font=font_body, fill="black")
                yy += 46
            y = yy + 14

        # Footer
        source = it.get("source", "")
        tier = it.get("tier", 3)
        pub = it.get("published_at", "")
        pop = it.get("popularity_signal", "unknown")
        footer = f"{source} | {pub} | Tier {tier} | {pop}"
        footer = _ellipsize(draw, footer, font_small, w)
        draw.text((x, CANVAS_SIZE[1] - margin - 40), footer, font=font_small, fill="black")

        card_path = cards_dir / f"{idx:02d}.png"
        img.save(card_path, format="PNG")
        created.append(card_path)

    return created
