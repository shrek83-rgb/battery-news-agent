# src/llm_enrich_gemini.py
from __future__ import annotations

import os
import re
import time
import random
from typing import Any, Dict, List, Tuple, Optional

from google import genai


# -----------------------------
# Config (env overridable)
# -----------------------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Retries / backoff
RETRIES = int(os.getenv("GEMINI_RETRIES", "4"))            # total retries after first try
BACKOFF_BASE = float(os.getenv("GEMINI_BACKOFF_BASE", "1.6"))
BACKOFF_MAX = float(os.getenv("GEMINI_BACKOFF_MAX", "12"))

# Behavior
STRICT_REQUIRE_THREE_NONEMPTY = os.getenv("GEMINI_STRICT_3", "1") == "1"
DEBUG_LOG = os.getenv("GEMINI_DEBUG", "0") == "1"

SYSTEM_INSTRUCTION = (
    "당신은 배터리 산업 뉴스 편집자입니다. "
    "입력으로 주어진 제목/설명/언론사 정보만 근거로 요약하세요. "
    "추측 금지, 기사에 없는 내용은 만들지 마세요. "
    "각 문장은 1문장으로 간결하게 작성하세요."
)


# -----------------------------
# Helpers: fallback summarization (content-based)
# -----------------------------
def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def _ensure_sentence_end(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # Add period if no obvious ending punctuation
    if s.endswith((".", "!", "?", "다.", "요.")):
        return s
    return s + "."


def fallback_summary_3_sentences_from_description(title: str, description: str) -> List[str]:
    """
    템플릿 금지(동일 문장 반복 금지). description/제목에서 '내용 기반'으로 3문장을 반드시 만든다.
    - description에서 문장 3개가 나오면 그대로 사용
    - 부족하면 description(또는 title)을 길이로 3등분해 문장처럼 구성(내용 기반)
    """
    title = (title or "").strip()
    desc = re.sub(r"\s+", " ", (description or "")).strip()

    sents = _split_sentences(desc)
    if len(sents) >= 3:
        return [ _ensure_sentence_end(sents[0]), _ensure_sentence_end(sents[1]), _ensure_sentence_end(sents[2]) ]

    base = desc if desc else title
    base = (base or "").strip()
    if not base:
        return ["", "", ""]

    # if only 1~2 sentences exist, use them and fill remaining by chunking remaining text
    if len(sents) == 2:
        # third sentence from leftover/chunk
        leftover = base
        # remove the first two sentences from base (best-effort)
        for s in sents[:2]:
            leftover = leftover.replace(s, " ")
        leftover = re.sub(r"\s+", " ", leftover).strip()
        if not leftover:
            leftover = base
        # make chunk
        n = len(leftover)
        cut = max(1, n // 2)
        s3 = leftover[cut:].strip() if len(leftover[cut:].strip()) > 10 else leftover[:cut].strip()
        return [_ensure_sentence_end(sents[0]), _ensure_sentence_end(sents[1]), _ensure_sentence_end(s3)]

    if len(sents) == 1:
        # make 2 more from chunking remaining text
        leftover = base.replace(sents[0], " ")
        leftover = re.sub(r"\s+", " ", leftover).strip()
        if not leftover:
            leftover = base
        n = len(leftover)
        cut1 = max(1, n // 2)
        s2 = leftover[:cut1].strip()
        s3 = leftover[cut1:].strip()
        # if chunks too short, reuse different parts of base
        if len(s2) < 8 or len(s3) < 8:
            n2 = len(base)
            c1 = max(1, n2 // 3)
            c2 = max(c1 + 1, 2 * n2 // 3)
            s2 = base[c1:c2].strip()
            s3 = base[c2:].strip()
        return [_ensure_sentence_end(sents[0]), _ensure_sentence_end(s2), _ensure_sentence_end(s3)]

    # 0 sentence: chunk base into 3 parts
    n = len(base)
    c1 = max(1, n // 3)
    c2 = max(c1 + 1, 2 * n // 3)
    s1 = base[:c1].strip()
    s2 = base[c1:c2].strip()
    s3 = base[c2:].strip()
    return [_ensure_sentence_end(s1), _ensure_sentence_end(s2), _ensure_sentence_end(s3)]


# -----------------------------
# Gemini prompting + parsing
# -----------------------------
def _build_prompt(title: str, description: str, source: str, link: str) -> str:
    # We enforce a strict plain-text format for robust parsing.
    return f"""
[기사]
- 제목: {title}
- 설명: {description}
- 언론사: {source}
- 링크: {link}

[출력 형식(반드시 이 형식으로만 출력)]
S1: (첫 문장 요약)
S2: (둘째 문장 요약)
S3: (셋째 문장 요약)
C: (관련 기업/기관 0~3개를 ';'로 구분. 없으면 빈칸)

[작성 규칙]
- S1/S2/S3는 각각 정확히 1문장.
- "사실 → 의미/영향 → 추가 맥락" 순서를 최대한 지켜 작성.
- 기사에 없는 내용은 만들지 말 것(추측 금지).
- 한국어로 작성하되, 고유명사/수치/날짜는 원문 표기를 최대한 유지.
- 링크/출처를 요약 문장에 다시 쓰지 말 것.
""".strip()


_RX_S = re.compile(r"^\s*(S1|S2|S3)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_RX_C = re.compile(r"^\s*C\s*:\s*(.*?)\s*$", re.IGNORECASE)


def _parse_gemini_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Parse the strict format:
      S1: ...
      S2: ...
      S3: ...
      C: a; b; c
    """
    s_map: Dict[str, str] = {}
    companies_line = ""

    for line in (text or "").splitlines():
        m = _RX_S.match(line)
        if m:
            key = m.group(1).upper()
            s_map[key] = m.group(2).strip()
            continue
        m = _RX_C.match(line)
        if m:
            companies_line = (m.group(1) or "").strip()

    sents = [s_map.get("S1", "").strip(), s_map.get("S2", "").strip(), s_map.get("S3", "").strip()]
    sents = [_ensure_sentence_end(s) for s in sents]

    companies: List[str] = []
    if companies_line:
        parts = [p.strip() for p in companies_line.split(";") if p.strip()]
        uniq: List[str] = []
        for p in parts:
            if p not in uniq:
                uniq.append(p)
        companies = uniq[:3]

    return sents, companies


def _call_gemini(client: genai.Client, prompt: str, model: str) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0.2,
        },
    )
    return (resp.text or "").strip()


def _sleep_backoff(attempt: int) -> None:
    # exponential backoff with jitter
    t = min(BACKOFF_MAX, BACKOFF_BASE ** (attempt + 1))
    t = t * (0.75 + random.random() * 0.5)
    time.sleep(t)


def enrich_one(
    client: genai.Client,
    item: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    title = (item.get("title") or "").strip()
    description = (item.get("description") or "").strip()
    source = (item.get("source") or "").strip()
    link = (item.get("link") or "").strip()

    prompt = _build_prompt(title, description, source, link)

    last_err: Optional[Exception] = None
    for attempt in range(RETRIES + 1):
        try:
            text = _call_gemini(client, prompt, model=model)
            if DEBUG_LOG:
                print(f"[DEBUG] Gemini raw (item={title[:30]}...): {text[:200]}")

            if not text:
                raise RuntimeError("Empty Gemini response")

            sents, companies = _parse_gemini_text(text)

            if STRICT_REQUIRE_THREE_NONEMPTY:
                if not (sents[0] and sents[1] and sents[2]):
                    raise RuntimeError(f"Parse incomplete: {text[:160]}")

            # If still empty, fallback (content-based)
            if not (sents[0] and sents[1] and sents[2]):
                sents = fallback_summary_3_sentences_from_description(title, description)

            item["summary_3_sentences"] = sents[:3]

            # Merge companies with existing (dict match might already exist)
            existing = item.get("companies") or []
            merged: List[str] = []
            for c in (companies + existing):
                c = (c or "").strip()
                if c and c not in merged:
                    merged.append(c)
            item["companies"] = merged[:3]

            return item

        except Exception as e:
            last_err = e
            if attempt < RETRIES:
                _sleep_backoff(attempt)
            else:
                break

    # Final fallback: 반드시 3문장(내용 기반)
    print(f"[WARN] Gemini failed for '{title[:60]}': {last_err}")
    item["summary_3_sentences"] = fallback_summary_3_sentences_from_description(title, description)
    item.setdefault("companies", item.get("companies") or [])
    return item


def enrich_items(
    items: List[Dict[str, Any]],
    max_items: int,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Enrich top-N items with:
      - summary_3_sentences (always 3 sentences, content-based)
      - companies (0~3)
    If Gemini fails, fallback still generates 3 sentences based on description/title (no fixed template lines).
    """
    out = items[:]  # shallow copy
    n = min(len(out), max_items)

    # If no API key: fallback for all
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set. Using content-based fallback summaries.")
        for i in range(n):
            it = out[i]
            it["summary_3_sentences"] = fallback_summary_3_sentences_from_description(
                it.get("title", ""), it.get("description", "")
            )
            it.setdefault("companies", it.get("companies") or [])
        # ensure rest also has 3 sentences
        for j in range(n, len(out)):
            it = out[j]
            it["summary_3_sentences"] = it.get("summary_3_sentences") or fallback_summary_3_sentences_from_description(
                it.get("title", ""), it.get("description", "")
            )
            it.setdefault("companies", it.get("companies") or [])
        return out

    client = genai.Client(api_key=GEMINI_API_KEY)

    # sequential (slower but stable)
    for i in range(n):
        out[i] = enrich_one(client, out[i], model=model)

    # Ensure all items have 3 sentences even beyond n
    for j in range(n, len(out)):
        it = out[j]
        if not it.get("summary_3_sentences"):
            it["summary_3_sentences"] = fallback_summary_3_sentences_from_description(
                it.get("title", ""), it.get("description", "")
            )
        # Normalize length
        s = it.get("summary_3_sentences") or ["", "", ""]
        while len(s) < 3:
            s.append("")
        it["summary_3_sentences"] = s[:3]
        it.setdefault("companies", it.get("companies") or [])

    return out
