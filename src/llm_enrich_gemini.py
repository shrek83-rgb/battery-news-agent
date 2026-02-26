from __future__ import annotations

import os
import concurrent.futures
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


def enrich_items(items: List[Dict[str, Any]], max_items: int, model: str = "gemini-2.0-flash") -> List[Dict[str, Any]]:
    from .tagger import fallback_summary_3_sentences  # 추가한 fallback 사용

    if not os.environ.get("GEMINI_API_KEY"):
        print("[WARN] GEMINI_API_KEY not set. Skipping Gemini enrichment.")
        for it in items:
            it["companies"] = it.get("companies") or []
            it["summary_3_sentences"] = fallback_summary_3_sentences(
                it.get("title",""), it.get("description",""), it.get("category","기타"), it.get("source","")
            )
        return items

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    n = min(len(items), max_items)
    out = items[:]  # copy

    batch_size = int(os.getenv("GEMINI_BATCH_SIZE", "5"))
    retries = int(os.getenv("GEMINI_RETRIES", "2"))

    class BatchItem(BaseModel):
        index: int
        summary_3_sentences: List[str] = Field(min_length=3, max_length=3)
        companies: List[str] = Field(default_factory=list, max_length=3)

    class BatchResp(BaseModel):
        items: List[BatchItem]

    def call_batch(batch: list[tuple[int, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
        payload = []
        for idx, it in batch:
            payload.append({
                "index": idx,
                "title": it.get("title",""),
                "description": it.get("description",""),
                "source": it.get("source",""),
                "link": it.get("link",""),
            })

        prompt = f"""
아래 기사 목록에 대해, 각 기사별로 3문장 요약과 관련 기업을 추출하세요.

요구사항:
- summary_3_sentences: 정확히 3문장(각 원소는 1문장). "사실 → 의미/영향 → 추가 맥락" 순서.
- companies: 기사와 직접 관련된 기업/기관명 0~3개(중복 제거). 없으면 빈 배열.
- 한국어로 작성하되 고유명사/수치/날짜는 원문 표기 유지.
- 반드시 JSON만 출력(스키마 준수).
- 각 결과는 입력의 index를 그대로 포함.

입력:
{payload}
""".strip()

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": BatchResp,
            },
        )

        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            parsed = BatchResp.model_validate_json(resp.text)

        result: dict[int, dict[str, Any]] = {}
        for x in parsed.items:
            s = [t.strip() for t in x.summary_3_sentences if isinstance(t, str)]
            while len(s) < 3:
                s.append("")
            comps = [c.strip() for c in x.companies if isinstance(c, str) and c.strip()]
            uniq = []
            for c in comps:
                if c not in uniq:
                    uniq.append(c)
            result[int(x.index)] = {"summary_3_sentences": s[:3], "companies": uniq[:3]}
        return result

    # 배치 처리
    for start in range(0, n, batch_size):
        batch_indices = list(range(start, min(start + batch_size, n)))
        batch = [(i, out[i]) for i in batch_indices]

        ok = False
        last_err = None
        for _ in range(retries + 1):
            try:
                mapping = call_batch(batch)
                for i in batch_indices:
                    enriched = mapping.get(i)
                    if enriched:
                        out[i]["summary_3_sentences"] = enriched["summary_3_sentences"]
                        out[i]["companies"] = enriched["companies"]
                    else:
                        # 해당 index가 응답에서 빠졌으면 fallback
                        out[i]["companies"] = out[i].get("companies") or []
                        out[i]["summary_3_sentences"] = fallback_summary_3_sentences(
                            out[i].get("title",""), out[i].get("description",""), out[i].get("category","기타"), out[i].get("source","")
                        )
                ok = True
                break
            except Exception as e:
                last_err = e

        if not ok:
            print(f"[WARN] Gemini batch failed ({start}-{start+len(batch_indices)-1}): {last_err}")
            for i in batch_indices:
                out[i]["companies"] = out[i].get("companies") or []
                out[i]["summary_3_sentences"] = fallback_summary_3_sentences(
                    out[i].get("title",""), out[i].get("description",""), out[i].get("category","기타"), out[i].get("source","")
                )

    # max_items 밖(혹시 남아있으면)도 3문장 보장
    for j in range(n, len(out)):
        out[j]["companies"] = out[j].get("companies") or []
        out[j]["summary_3_sentences"] = out[j].get("summary_3_sentences") or fallback_summary_3_sentences(
            out[j].get("title",""), out[j].get("description",""), out[j].get("category","기타"), out[j].get("source","")
        )

    return out


def enrich_items(items: List[Dict[str, Any]], max_items: int, model: str = "gemini-2.0-flash") -> List[Dict[str, Any]]:
    if not os.environ.get("GEMINI_API_KEY"):
        print("[WARN] GEMINI_API_KEY not set. Skipping Gemini enrichment.")
        for it in items:
            it.setdefault("companies", [])
            it.setdefault("summary_3_sentences", ["", "", ""])
        return items

    n = min(len(items), max_items)
    out = items[:]

    max_workers = int(os.getenv("GEMINI_WORKERS", "4"))
    timeout_sec = int(os.getenv("GEMINI_TIMEOUT_SEC", "25"))

    def work(i: int):
        it = out[i]
        return i, enrich_item(
            title=it.get("title", ""),
            description=it.get("description", ""),
            source=it.get("source", ""),
            link=it.get("link", ""),
            model=model,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work, i): i for i in range(n)}
        for fut in concurrent.futures.as_completed(futures, timeout=timeout_sec * max(1, n)):
            i = futures[fut]
            try:
                _, enriched = fut.result(timeout=timeout_sec)
                out[i]["summary_3_sentences"] = enriched["summary_3_sentences"]
                out[i]["companies"] = enriched["companies"]
            except Exception as e:
                print(f"[WARN] Gemini enrich failed/timeout for item {i+1}: {e}")
                out[i].setdefault("companies", [])
                out[i].setdefault("summary_3_sentences", ["", "", ""])

    for j in range(n, len(out)):
        out[j].setdefault("companies", [])
        out[j].setdefault("summary_3_sentences", ["", "", ""])

    return out
