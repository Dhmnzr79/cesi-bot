import re

SAFE_VERBATIM_TYPES = {"contacts","prices","safety","warranty","doctors"}

def should_verbatim(meta: dict) -> bool:
    return bool(meta.get("verbatim") is True and meta.get("doc_type") in SAFE_VERBATIM_TYPES)

def clamp_bullets(text: str, max_bullets: int = 6) -> str:
    lines = [l.strip(" •-\t") for l in text.splitlines() if l.strip()]
    return " — ".join(lines[:max_bullets])

def slice_for_prompt(text: str, limit_chars: int = 900) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t[:limit_chars]

def smart_trim(text: str, limit: int, allow_ellipsis: bool) -> str:
    """Умная обрезка по границам предложений или пунктов списка"""
    if len(text) <= limit: 
        return text
    
    cut = text[:limit]
    
    # шаг 1: попробуй найти последнюю границу предложения
    SENT_END = '.!?…;'
    idx = max(cut.rfind(ch) for ch in SENT_END)
    if idx >= 0:
        return cut[:idx+1].rstrip()
    
    # шаг 2: попробуй закончить на конец пункта списка/строки
    for marker in ['\n- ', '\n• ', '\n— ', '\n– ', '\n* ']:
        idx = cut.rfind(marker)
        if idx >= 0:
            return cut[:idx].rstrip()
    
    # шаг 3: fallback — до последнего пробела
    idx = cut.rfind(' ')
    trimmed = cut[:idx].rstrip() if idx > 0 else cut.rstrip()
    
    if not allow_ellipsis:
        return trimmed.rstrip('. ').rstrip() + '.'
    
    return (trimmed + '…').rstrip()

