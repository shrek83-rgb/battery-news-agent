# src/llm_enrich_gemini.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple, Optional

from google import genai
from pydantic import BaseModel, Field


def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    if n.startswith("google.gemini-"):
        n = n.replace("google.", "", 1)

    low = n.lower().strip()
    # common typo: light -> lite
    low = low.replace("light", "lite")

    low2 = (
        low.replace("_", " ")
           .replace("-", " ")
           .replace("flashlite", "flash lite")
           .replace("  ", " ")
    )

    if "2.5" in low2 and "flash" in low2 and "lite" in low2:
        return "gemini-2.5-flash-lite"
    if "2.5" in low2 and "flash" in low2:
        return "gemini-2.5-flash"
    if "2.0" in low2 and "flash" in low2 and "lite" in low2:
        return "gemini-2.0-flash-lite"
    if "2.0" in low2 and "flash" in low2:
        return "gemini-2.0-flash"

    if low.startswith("gemini-"):
        return low
    return n


# -----------------------------
# Config (single source of truth)
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
BASE_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
SUMMARY_MODEL = _normalize_model_name(os.getenv("GEMINI_MODEL_SUMMARY", BASE_MODEL))

# ✅ default 0 retries to guarantee "one request"
BATCH_RETRIES = int(os.getenv("GEMINI_BATCH_RETRIES", "0"))

DEBUG_LOG = os.getenv("GEMINI_DEBUG", "0") == "1"


SYSTEM_INSTRUCTION = (
    "당신은 배터리 산업 뉴스 편집자입니다. "
    "입력으로 주어진 제목/설명/언론사 정보만 근거로 요약하세요. "
    "추측 금지, 기사에 없는 내용은 만들지 마세요. "
    "각 문장은 1문장으로 간결하게 작성하세요."
)


# -----------------------------
# Fallback summarization (content-based; no template)
# -----------------------------
def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _ensure_sentence_end(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.endswith((".", "!", "?", "다.", "요.")):
        return s
    return s + "."


def fallback_summary_3_sentences_from_description(title: str, description: str) -> List[str]:
    title = (title or "").strip()
    desc = re.sub(r"\s+", " ", (description or "")).strip()

    sents = _split_sentences(desc)
    if len(sents) >= 3:
        return [_ensure_sentence_end(sents[0]), _ensure_sentence_end(sents[1]), _ensure_sentence_end(sents[2])]

    base = desc if desc else title
    base = (base or "").strip()
    if not base:
        return ["", "", ""]

    if len(sents) == 2:
        leftover = base
        for s in sents[:2]:
            leftover = leftover.replace(s, " ")
        leftover = re.sub(r"\s+", " ", leftover).strip() or base
        n = len(leftover)
        cut = max(1, n // 2)
        s3 = leftover[cut:].strip() if len(leftover[cut:].strip()) > 10 else leftover[:cut].strip()
        return [_ensure_sentence_end(sents[0]), _ensure_sentence_end(sents[1]), _ensure_sentence_end(s3)]

    if len(sents) == 1:
        leftover = re.sub(r"\s+", " ", base.replace(sents[0], " ")).strip() or base
        n = len(leftover)
        cut1 = max(1, n // 2)
        s2 = leftover[:cut1].strip()
        s3 = leftover[cut1:].strip()
        if len(s2) < 8 or len(s3) < 8:
            n2 = len(base)
            c1 = max(1, n2 // 3)
            c2 = max(c1 + 1, 2 * n2 // 3)
            s2 = base[c1:c2].strip()
            s3 = base[c2:].strip()
        return [_ensure_sentence_end(sents[0]), _ensure_sentence_end(s2), _ensure_sentence_end(s3)]

    n = len(base)
    c1 = max(1, n // 3)
    c2 = max(c1 + 1, 2 * n // 3)
    s1 = base[:c1].strip()
    s2 = base[c1:c2].strip()
    s3 = base[c2:].strip()
    return [_ensure_sentence_end(s1), _ensure_sentence_end(s2), _ensure_sentence_end(s3)]


# -----------------------------
# Batch schema
# -----------------------------
class BatchItem(BaseModel):
    index: int
    summary_3_sentences: List[str] = Field(default_factory=list)
    companies: List[str] = Field(default_factory=list)


class BatchResp(BaseModel):
    items: List[BatchItem]


def _call_batch_once(client: genai.Client, payload: List[Dict[str, Any]], model: str) -> BatchResp:
    prompt = f"""
아래 기사 목록에 대해, 각 기사별로 3문장 요약과 관련 기업을 추출하세요.

요구사항:
- summary_3_sentences: 정확히 3문장(각 원소는 1문장). "사실 → 의미/영향 → 추가 맥락" 순서.
- companies: 기사와 직접 관련된 기업/기관명 0~3개(중복 제거). 없으면 빈 배열.
- 한국어로 작성하되, 고유명사/수치/날짜는 원문 표기를 최대한 유지.
- 반드시 JSON만 출력. 스키마 준수.

입력(배열):
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
    return parsed


def enrich_items(items: List[Dict[str, Any]], max_items: int, model: str | None = None) -> List[Dict[str, Any]]:
    """
    ✅ Gemini 요청 1회(기본)로 top-N 요약/기업 추출.
    - BATCH_RETRIES 기본 0: 진짜 1회 보장
    - 실패 시 추가 요청 없이 fallback으로 채움
    """
    out = items[:]
    n = min(len(out), max_items)
    use_model = _normalize_model_name(model) if model else SUMMARY_MODEL

    if DEBUG_LOG:
        print(f"[INFO] Gemini models (BASE/SUMMARY): {BASE_MODEL} / {use_model}")

    # No API key -> fallback only
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set. Using content-based fallback summaries.")
        for i in range(n):
            it = out[i]
            it["summary_3_sentences"] = fallback_summary_3_sentences_from_description(it.get("title", ""), it.get("description", ""))
            it.setdefault("companies", it.get("companies") or [])
        return out

    # Prepare payload (trim to control tokens)
    payload: List[Dict[str, Any]] = []
    for i in range(n):
        it = out[i]
        payload.append({
            "index": i,
            "title": (it.get("title") or "")[:300],
            "description": (it.get("description") or "")[:1200],
            "source": (it.get("source") or "")[:120],
            "link": (it.get("link") or "")[:300],
        })

    client = genai.Client(api_key=GEMINI_API_KEY)

    calls = 0
    parsed: Optional[BatchResp] = None
    last_err: Optional[Exception] = None

    # ✅ call at most 1 + retries (default 0 retries)
    for attempt in range(BATCH_RETRIES + 1):
        try:
            calls += 1
            parsed = _call_batch_once(client, payload, model=use_model)
            break
        except Exception as e:
            last_err = e
            parsed = None

    if DEBUG_LOG:
        print(f"[INFO] Gemini batch calls={calls} (retries={BATCH_RETRIES})")

    # Build mapping index -> (sents, comps)
    mapping: Dict[int, Tuple[List[str], List[str]]] = {}
    if parsed is not None:
        for x in parsed.items:
            idx = int(x.index)
            s = [t.strip() for t in (x.summary_3_sentences or []) if isinstance(t, str)]
            while len(s) < 3:
                s.append("")
            s = [_ensure_sentence_end(t) for t in s[:3]]

            comps = [c.strip() for c in (x.companies or []) if isinstance(c, str) and c.strip()]
            uniq: List[str] = []
            for c in comps:
                if c not in uniq:
                    uniq.append(c)

            mapping[idx] = (s, uniq[:3])

    else:
        print(f"[WARN] Gemini batch enrichment failed (no retry beyond {BATCH_RETRIES}): {last_err}")

    # Apply results per item (no extra requests)
    for i in range(n):
        it = out[i]
        if i in mapping:
            sents, comps = mapping[i]
            # if any sentence missing, fill with fallback (still no new request)
            if sum(1 for t in sents if t.strip()) < 3:
                sents = fallback_summary_3_sentences_from_description(it.get("title", ""), it.get("description", ""))
            it["summary_3_sentences"] = sents[:3]

            existing = it.get("companies") or []
            merged: List[str] = []
            for c in comps + existing:
                c = (c or "").strip()
                if c and c not in merged:
                    merged.append(c)
            it["companies"] = merged[:3]
        else:
            it["summary_3_sentences"] = fallback_summary_3_sentences_from_description(it.get("title", ""), it.get("description", ""))
            it.setdefault("companies", it.get("companies") or [])

    return out
