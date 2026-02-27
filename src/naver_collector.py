import re

# 강화 dedupe 파라미터 (env로 조절)
NEAR_DUP_SIM_TH = float(os.getenv("NAVER_NEAR_DUP_SIM_TH", "0.78"))  # 0~1, 높을수록 더 공격적으로 제거
MAX_PER_ENTITY = int(os.getenv("NAVER_MAX_PER_ENTITY", "2"))         # 같은 기업/기관 최대 몇 개까지 허용
MAX_PER_TOPIC = int(os.getenv("NAVER_MAX_PER_TOPIC", "3"))           # (옵션) 같은 테마 키워드 최대 몇 개

_TOPIC_KWS = {
    "전고체": ["전고체", "고체전해질", "solid-state", "solid state"],
    "리튬": ["리튬", "lithium"],
    "양극재": ["양극재", "cathode", "전구체", "니켈", "NCM", "LFP"],
    "음극재": ["음극재", "anode", "흑연", "실리콘"],
    "재활용": ["재활용", "리사이클", "recycling", "폐배터리"],
    "나트륨": ["나트륨", "sodium"],
    "ESS": ["ESS", "에너지저장", "energy storage"],
}

# 간단 엔터티(기업/기관) 추출: 한국 기업/기관명 패턴 + 대표 키워드
_KOR_ENTITY_SUFFIX = r"(?:그룹|홀딩스|에너지|화학|전지|배터리|머티리얼즈|머티리얼|소재|제철|산업|전자|솔루션|엔솔|이노베이션|모빌리티|테크|테크놀로지|리서치|캐피탈|온|SDI|대학|대학교)"
_RX_ENTITY = re.compile(rf"([가-힣A-Za-z0-9·&\.\-]{{2,24}}{_KOR_ENTITY_SUFFIX})")

def _norm_title(t: str) -> str:
    t = (t or "").strip().lower()
    # 기호/중복 공백 정리
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\[\]\(\)\"'“”‘’·•]", " ", t)
    t = re.sub(r"[^0-9a-z가-힣\s\.\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _char_ngrams(s: str, n: int = 3) -> set[str]:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def _title_sim(a: str, b: str) -> float:
    a2 = _norm_title(a)
    b2 = _norm_title(b)
    if not a2 or not b2:
        return 0.0
    A = _char_ngrams(a2, 3)
    B = _char_ngrams(b2, 3)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _extract_entities(title: str) -> List[str]:
    t = (title or "").strip()
    ents = [m.group(1).strip() for m in _RX_ENTITY.finditer(t)]
    # 자주 나오는 회사/기관 수동 보강(짧은 표기)
    for k in ["SK온", "LG엔솔", "LG에너지솔루션", "삼성SDI", "포스코", "이수스페셜티", "경북대", "경북대학교"]:
        if k in t and k not in ents:
            ents.append(k)
    # uniq
    out = []
    for e in ents:
        if e and e not in out:
            out.append(e)
    return out[:2]  # 대표 엔터티 1~2개만

def _topic_key(title: str) -> Optional[str]:
    t = (title or "").lower()
    for key, kws in _TOPIC_KWS.items():
        for kw in kws:
            if kw.lower() in t:
                return key
    return None

def diversify_and_dedupe_by_title(
    items: List[NaverNewsItem],
    scores_rel: Dict[int, int],
    scores_imp: Dict[int, int],
    top_k: int,
) -> List[int]:
    """
    입력: 후보 인덱스 리스트(items의 인덱스)
    출력: 선택된 인덱스 리스트 (len <= top_k)
    로직:
      - (relevance, importance, rank) 순으로 우선 정렬된 후보를 순회하면서
      - 이미 선택된 것들과 제목 유사도 >= threshold 면 스킵
      - 같은 엔터티(기업/기관) 노출 MAX_PER_ENTITY 초과면 스킵
      - 같은 토픽 노출 MAX_PER_TOPIC 초과면 스킵(옵션)
    """
    picked: List[int] = []
    picked_titles: List[str] = []
    entity_count: Dict[str, int] = {}
    topic_count: Dict[str, int] = {}

    for idx in items:
        it = items[idx]
        t = it.title

        # 1) near-duplicate title check
        too_similar = False
        for pt in picked_titles:
            if _title_sim(t, pt) >= NEAR_DUP_SIM_TH:
                too_similar = True
                break
        if too_similar:
            continue

        # 2) entity cap
        ents = _extract_entities(t)
        if ents:
            # 대표 엔터티 중 하나라도 cap 초과면 스킵
            if any(entity_count.get(e, 0) >= MAX_PER_ENTITY for e in ents):
                continue

        # 3) topic cap (optional but useful for "전고체만 3개" 같은 쏠림 방지)
        tk = _topic_key(t)
        if tk:
            if topic_count.get(tk, 0) >= MAX_PER_TOPIC:
                continue

        # accept
        picked.append(idx)
        picked_titles.append(t)

        for e in ents:
            entity_count[e] = entity_count.get(e, 0) + 1
        if tk:
            topic_count[tk] = topic_count.get(tk, 0) + 1

        if len(picked) >= top_k:
            break

    return picked
