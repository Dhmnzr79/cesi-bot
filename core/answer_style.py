# core/answer_style.py
from dataclasses import dataclass

@dataclass
class Style:
    mode: str               # "list" | "compact"
    max_chars: int
    allow_ellipsis: bool
    max_bullets: int

SAFE_LIST_THEMES = {"safety","contraindications","consultation","pain","pain_faq"}

def decide_style(theme: str) -> Style:
    t = (theme or "").lower()
    if t in {"prices","prices_short","contacts","warranty"}:
        return Style(mode="compact", max_chars=420, allow_ellipsis=False, max_bullets=0)
    if t in SAFE_LIST_THEMES:
        # структурный ответ списком, мягкий длинный лимит
        return Style(mode="list", max_chars=860, allow_ellipsis=True, max_bullets=6)
    if t in {"doctors"}:
        return Style(mode="list", max_chars=420, allow_ellipsis=True, max_bullets=5)
    if t in {"consultation_steps"}:
        return Style(mode="list", max_chars=720, allow_ellipsis=True, max_bullets=6)
    # дефолт — компактный параграф
    return Style(mode="compact", max_chars=480, allow_ellipsis=True, max_bullets=0)
