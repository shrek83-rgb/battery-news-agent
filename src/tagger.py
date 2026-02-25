from __future__ import annotations

import re
from pathlib import Path


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


def extract_companies(title: str, description: str, max_n: int = 3) -> list[str]:
    text = f"{title} {description}".lower()
    p = Path("config/companies.txt")
    if not p.exists():
        return []
    found: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name.lower() in text and name not in found:
            found.append(name)
        if len(found) >= max_n:
            break
    return found
