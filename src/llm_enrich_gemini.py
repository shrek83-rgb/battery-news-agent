from __future__ import annotations

import os
from typing import Any, Dict, List

from google import genai
from pydantic import BaseModel, Field


class NewsEnriched(BaseModel):
    summary_3_sentences: List[str] = Field(min_length=3, max_length=3)
    companies: List[str] = Field(default_factory=list, max_length=3)


SYSTEM_INSTRUCTION = (
    "당신은 배터리 산업 뉴스 편집자입니다. "
    "주어진 제목과 설명만 근거로 요약/추출하세요. "
    "추측 금지, 기사에 없는 내용은 만들지 마세요."
)


def enrich_item(title: str, description: str, source: str, link: str, model: str) -> Dict[str, Any]:
    client = genai.Client()  # GEMINI_API_KEY를 env에서 읽음 :contentReference[oaicite:4]{index=4}

    prompt = f"""
[기사]
- 제목: {title}
- 설명: {description}
- 언론사/출처: {source}
- 링크: {link}

[요구사항]
- summary_3_sentences: 정확히 3문장(각 원소는 1문장). "사실 → 의미/영향 → 추가 맥락" 순으로.
- companies: 기사와 직접 관련된 기업/기관명 0~3개(중복 제거). 없으면 빈 배열.
- 한국어로 작성하되, 고유명사/수치/날짜는 원문 표기를 최대한 유지.
- 링크/출처를 요약에 다시 쓰지 말 것.
- 반드시 JSON만 출력(스키마 준수).
""".strip()

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": NewsEnriched,
        },
    )

    # SDK가 JSON을 파싱해 resp.parsed로 주는 흐름(문서에 안내) :contentReference[oaicite:5]{index=5}
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        # 혹시 파싱이 안 되면 text에서 재시도(최후수단)
        parsed = NewsEnriched.model_validate_json(resp.text)

    data = parsed.model_dump()

    # 보정(안전)
    s = [x.strip() for x in data.get("summary_3_sentences", []) if isinstance(x, str)]
    while len(s) < 3:
        s.append("")
    data["summary_3_sentences"] = s[:3]

    comps = [c.strip() for c in data.get("companies", []) if isinstance(c, str) and c.strip()]
    uniq = []
    for c in comps:
        if c not in uniq:
            uniq.append(c)
    data["companies"] = uniq[:3]

    return data


import concurrent.futures

def enrich_items(items: List[Dict[str, Any]], max_items: int, model: str = "gemini-2.0-flash") -> List[Dict[str, Any]]:
    if not os.environ.get("GEMINI_API_KEY"):
        print("[WARN] GEMINI_API_KEY not set. Skipping Gemini enrichment.")
        for it in items:
            it.setdefault("companies", [])
            it.setdefault("summary_3_sentences", ["", "", ""])
        return items

    n = min(len(items), max_items)
    out = items[:]  # copy

    def work(i: int):
        it = out[i]
        return i, enrich_item(
            title=it.get("title", ""),
            description=it.get("description", ""),
            source=it.get("source", ""),
            link=it.get("link", ""),
            model=model,
        )

    max_workers = int(os.getenv("GEMINI_WORKERS", "4"))
    timeout_sec = int(os.getenv("GEMINI_TIMEOUT_SEC", "25"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work, i): i for i in range(n)}
        for fut in concurrent.futures.as_completed(futures, timeout=timeout_sec * n):
            i = futures[fut]
            try:
                _, enriched = fut.result(timeout=timeout_sec)
                out[i]["summary_3_sentences"] = enriched["summary_3_sentences"]
                out[i]["companies"] = enriched["companies"]
            except Exception as e:
                print(f"[WARN] Gemini enrich failed/timeout for item {i+1}: {e}")
                out[i].setdefault("companies", [])
                out[i].setdefault("summary_3_sentences", ["", "", ""])

    # items beyond n: ensure fields exist
    for j in range(n, len(out)):
        out[j].setdefault("companies", [])
        out[j].setdefault("summary_3_sentences", ["", "", ""])

    return out
