# core/empathy.py
from __future__ import annotations
import re, time, random, yaml, os
from typing import Optional, Dict, Any

_CFG = None
_TRIGGERS = None
_DOC_TAG_MAP = None

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(base_dir="config"):
    global _CFG, _TRIGGERS, _DOC_TAG_MAP
    _CFG = _load_yaml(os.path.join(base_dir, "empathy.yaml"))
    _TRIGGERS = _load_yaml(os.path.join(base_dir, "empathy_triggers.yaml"))
    # опциональная карта "документ/слаг -> тег"
    _DOC_TAG_MAP = (_CFG or {}).get("doc_tag_map", {})
    return _CFG

def _now() -> float:
    return time.time()

def detect_tag_from_text(user_text: str) -> Optional[str]:
    if not _TRIGGERS:
        return None
    t = user_text.lower()
    for tag, stems in (_TRIGGERS.get("triggers") or {}).items():
        for s in stems:
            if s in t:
                return tag
    return None

def infer_tag_from_doc(topic_meta: Dict[str, Any]) -> Optional[str]:
    if not topic_meta:
        return None
    
    # 1) явный фронтматтер
    if topic_meta.get("empathy_tag"):
        return topic_meta["empathy_tag"]
    
    # 2) по пути/слагу из карты
    slug = (topic_meta.get("slug") or "").lower()
    path = (topic_meta.get("path") or "").lower().replace(".md","")
    
    if slug and slug in _DOC_TAG_MAP:
        return _DOC_TAG_MAP[slug]
    if path and path in _DOC_TAG_MAP:
        return _DOC_TAG_MAP[path]
    
    return None

def _has_prices(answer_text: str) -> bool:
    # цифры + ₽/руб/тыс — простая эвристика
    return bool(re.search(r"(\d[\d\s]{0,6})(₽|руб|тыс)", answer_text.lower()))

def maybe_opener_or_bridge(answer_text: str, user_text: str, topic_meta: Dict[str, Any], session: Dict[str, Any], intent: Optional[str]) -> Optional[str]:
    """
    Вернёт строку (эмпатия-опенер ИЛИ бридж), либо None.
    """
    if not os.getenv("ENABLE_EMPATHY", "true").lower() == "true":
        return None
    
    settings = (_CFG or {}).get("settings", {})
    cooldown = int(os.getenv("EMPATHY_COOLDOWN_SECONDS", settings.get("cooldown_seconds", 60)))
    blocklist = set((settings.get("blocklist_doc_tags") or []))
    
    doc_tag = topic_meta.get("doc_tag") or topic_meta.get("tag")
    if doc_tag in blocklist:
        return None
    
    last_ts = session.get("empathy_last_ts")
    if last_ts and (_now() - last_ts) < cooldown:
        return None
    
    # 1) определить финальный тег
    tag = intent or infer_tag_from_doc(topic_meta) or detect_tag_from_text(user_text) or "neutral"
    
    # 1.5) эмпатия на цене без цифр — отключаем по флагу
    if tag == "price" and os.getenv("EMPATHY_FOR_PRICE", "false") == "false" and not _has_prices(answer_text):
        return None
    
    # 2) цена/сроки: если в ответе конкретика — бридж вместо эмпатии
    if tag in ("price","duration") and _has_prices(answer_text) and os.getenv("ENABLE_PRICE_BRIDGE","true") == "true":
        bridges = (((_CFG or {}).get("phrases") or {}).get(tag) or {}).get("bridges") or []
        if bridges:
            phrase = _pick_non_repeating(tag, bridges, session)
            _mark_empathy_used(session, tag, phrase)
            return phrase
        # если бриджей нет — ничего не вставляем
        return None
    
    # 3) ограничение повторов по тегу
    max_consecutive = int(((_CFG or {}).get("settings", {})).get("max_consecutive_tag", 2))
    recent = session.get("empathy_recent", [])
    recent_tags = [t for (t, _p) in recent[-max_consecutive:]]
    if len(recent_tags) == max_consecutive and all(t == tag for t in recent_tags):
        return None
    
    # 4) обычный опенер
    openers = (((_CFG or {}).get("phrases") or {}).get(tag) or {}).get("openers") or []
    if not openers:
        return None
    
    phrase = _pick_non_repeating(tag, openers, session)
    _mark_empathy_used(session, tag, phrase)
    return phrase

def _pick_non_repeating(tag: str, pool: list[str], session: Dict[str, Any]) -> Optional[str]:
    used = session.get("empathy_recent", [])  # [(tag, phrase)]
    
    # не повторять последнюю фразу
    pool2 = [p for p in pool if not used or p != used[-1][1]]
    
    choice = random.choice(pool2 or pool) if pool else None
    return choice

def _mark_empathy_used(session: Dict[str, Any], tag: str, phrase: str):
    session["empathy_last_ts"] = _now()
    hist = session.setdefault("empathy_recent", [])
    hist.append((tag, phrase))
    if len(hist) > 5:
        hist[:-5] = []  # оставляем последние 5