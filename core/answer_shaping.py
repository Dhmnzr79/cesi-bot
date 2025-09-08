import re

SAFE_VERBATIM_TYPES = {"contacts", "prices_short"}

def should_verbatim(meta: dict) -> bool:
    return bool(meta.get("verbatim") is True and meta.get("doc_type") in SAFE_VERBATIM_TYPES)

def clamp_bullets(text: str, max_bullets: int = 6) -> str:
    lines = [l.strip(" •-\t") for l in text.splitlines() if l.strip()]
    return " — ".join(lines[:max_bullets])

def slice_for_prompt(text: str, limit_chars: int = 900) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t[:limit_chars]

