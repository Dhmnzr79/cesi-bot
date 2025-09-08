# core/cta.py
from __future__ import annotations
import yaml, os, time
from typing import Optional, Dict, Any

_CFG = None

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(base_dir="config"):
    global _CFG
    _CFG = _load_yaml(os.path.join(base_dir, "cta.yaml"))
    return _CFG

def _now(): return time.time()

def _cta_key(obj: Dict[str, Any]) -> str:
    # ключ для антидубля (тип + url/params.topic/doctor)
    base = obj.get("type","")
    url = obj.get("url","")
    topic = (obj.get("params") or {}).get("topic","")
    doctor = (obj.get("params") or {}).get("doctor","")
    return f"{base}|{url}|{topic}|{doctor}"

def _cooldown_ok(session: Dict[str, Any], seconds: int) -> bool:
    last = session.get("last_cta_at")
    if not last: return True
    return (_now() - last) >= seconds

def _is_high_intent(intent: Optional[str]) -> bool:
    tags = set((_CFG or {}).get("high_intent_tags") or [])
    return bool(intent and intent in tags)

def _mark_shown(session: Dict[str, Any], obj: Dict[str, Any]):
    session["last_cta_at"] = _now()
    session["last_cta_key"] = _cta_key(obj)

def build_cta_from_topic(topic_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not topic_meta: return None
    action = topic_meta.get("cta_action")
    label  = topic_meta.get("cta_text")
    params = topic_meta.get("cta_params") or {}
    if not action or not label: return None

    # сопоставь action -> url/phone (минимум — url)
    url = topic_meta.get("cta_url") or _intent_to_url_fallback(action, params)
    obj = {"type": action, "label": label, "url": url, "params": params}
    obj["key"] = _cta_key(obj)
    return obj

def _kv(key: str, val: str) -> str:
    return f"{key}={val}" if val else ""

def _intent_to_url_fallback(action: str, params: Dict[str, Any]) -> str:
    # подстелим дефолтный маршрут
    topic = params.get("topic","general")
    if action == "book_doctor":
        return f"/book?topic=doctor&{_kv('doctor', params.get('doctor'))}"
    if action == "book_ct":
        return "/book?topic=ct"
    # consultation / call / whatsapp и прочее
    return f"/book?topic={topic}"

def build_cta_from_intent(intent: Optional[str]) -> Optional[Dict[str, Any]]:
    if not intent: return None
    imap = (_CFG or {}).get("intent_map") or {}
    obj = imap.get(intent)
    if not obj: return None
    obj = dict(obj)  # копия
    obj["key"] = _cta_key(obj)
    return obj

def build_default_cta() -> Dict[str, Any]:
    obj = dict(((_CFG or {}).get("default")) or {})
    obj["key"] = _cta_key(obj)
    return obj

def decide_cta(intent: Optional[str],
               topic_meta: Dict[str, Any],
               session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cooldown = int(os.getenv("CTA_COOLDOWN_SECONDS", (_CFG or {}).get("cooldown_seconds", 90)))
    override = os.getenv("CTA_HIGH_INTENT_OVERRIDE","true") == "true"

    # строим кандидат из темы/интента, а дефолт — только если флаг включён
    cand = build_cta_from_topic(topic_meta) or build_cta_from_intent(intent)
    if not cand and os.getenv("ENABLE_CTA_GLOBAL_FALLBACK", "true") == "true":
        cand = build_default_cta()

    if not cand: return None

    same = (session.get("last_cta_key") == cand["key"])
    if same:
        return None

    if not _cooldown_ok(session, cooldown):
        # пропускаем из-за кулдауна, кроме high-intent
        if not (override and _is_high_intent(intent)):
            return None

    # TTL подавления после клика
    post_click_suppress = int(os.getenv("CTA_POST_CLICK_SUPPRESS_SECONDS", "75"))
    if session.get("cta_clicked_at") and (_now() - session["cta_clicked_at"] < post_click_suppress):
        return None

    _mark_shown(session, cand)
    return cand

def register_cta_click(session: Dict[str, Any]):
    # зови это при событии клика
    session["cta_clicked_at"] = _now()
    # опционально: session.pop("cta_clicked_recently", None)
