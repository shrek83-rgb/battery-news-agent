from __future__ import annotations

import re


CATEGORIES = [
    ("cathode", ["cathode", "양극", "양극재", "ncm", "lfp", "lco", "nca", "nickel", "망간", "코발트", "철인산"]),
    ("anode", ["anode", "음극", "음극재", "silicon", "graphite", "흑연", "실리콘"]),
    ("electrolyte", ["electrolyte", "전해질", "liquid electrolyte", "salt", "염", "additive", "첨가제"]),
    ("separator", ["separator", "분리막", "membrane"]),
    ("전고체", ["solid-state", "all-solid-state", "전고체", "고체전해질", "sulfide", "oxide", "polymer"]),
    ("나트륨", ["sodium-ion", "나트륨", "na-ion", "na ion"]),
    ("재활용", ["recycling", "recycle", "재활용", "black mass", "블랙매스", "hydrometallurgy", "pyrometallurgy"]),
    ("장비", ["equipment", "장비", "coater", "캘린더", "dry electrode", "건식", "formation", "조립", "코팅"]),
    ("정책", ["policy", "regulation", "규제", "subsidy", "보조금", "ira", "crma", "cbam", "tariff", "관세"]),
]


def classify_category(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    for cat, keys in CATEGORIES:
        for k in keys:
            if k.lower() in text:
                return cat
    return "기타"


def summarize_3_sentences(title: str, description: str, source: str) -> list[str]:
    """
    비용 0원 룰 기반 3문장 요약(MVP).
    - description(리드문)이 있으면 문장 1~2개를 가져오고, 부족하면 템플릿으로 보완
    """
    desc = re.sub(r"\s+", " ", (description or "")).strip()

    # Split into sentences (very rough, works for EN/KR mixed)
    candidates = re.split(r"(?<=[\.\!\?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", desc)
    candidates = [c.strip() for c in candidates if c.strip()]

    s1 = f"{title}({source}) 관련 소식입니다."
    s2 = candidates[0] if len(candidates) >= 1 else "기사 본문에서 핵심 사실(투자/기술/정책/실적)을 확인해야 합니다."
    s3 = candidates[1] if len(candidates) >= 2 else "배터리 밸류체인(소재/셀/팩/리사이클링) 및 시장에 미칠 영향을 점검하세요."

    # Ensure 3 sentences and avoid overly long lines
    return [_trim(s1), _trim(s2), _trim(s3)]


def _trim(s: str, limit: int = 220) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"
