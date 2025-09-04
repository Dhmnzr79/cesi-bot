import random
from typing import Tuple, Dict, Any

def triggers_match(user_query: str, triggers: Dict[str, list[str]]) -> Tuple[str | None, bool]:
    q = user_query.lower()
    for cat, kws in triggers.items():
        if any(kw in q for kw in kws):
            return cat, True
    return None, False

def llm_classify(user_query: str, snippet: str, llm_client, model: str) -> Tuple[str | None, float]:
    prompt = {"role": "system",
              "content": 'Ты классифицируешь эмоцию пациента. Верни ТОЛЬКО JSON: {"category":"...", "confidence":0..1}.'}
    user = {"role": "user",
            "content": f"Вопрос: <<{user_query}>>\nКонтекст: <<{snippet[:800]}>>\n"
                       f"Категории: pain, price, trust, consultation, osseointegration, safety, none"}
    resp = llm_client.chat.completions.create(
        model=model,
        messages=[prompt, user],
        response_format={"type": "json_object"}
    )
    try:
        data = resp.choices[0].message.parsed or {}
        cat = (data.get("category") or "none").lower()
        conf = float(data.get("confidence", 0))
        valid = {"pain","price","trust","consultation","osseointegration","safety"}
        return (cat if cat in valid else "none", conf)
    except Exception:
        return (None, 0.0)

def detect_emotion(user_query: str, retrieved_snippet: str, fm: Dict[str, Any],
                   empathy_cfg: Dict[str, Any], triggers_bank: Dict[str, list[str]],
                   llm_client=None, model="gpt-4o-mini") -> Tuple[str, str, float]:
    # 1) жёсткие стопы
    if fm.get("verbatim"):
        return ("none", "verbatim", 1.0)
    if fm.get("doc_type") in set(empathy_cfg["topics"].get("blocked_hard", [])):
        return ("none", "hard_block", 1.0)

    # 2) явная пометка в MD
    fm_emotion = (fm.get("emotion") or "none").lower()
    if fm_emotion != "none":
        return (fm_emotion, "frontmatter", 1.0)

    mode = empathy_cfg.get("mode", "hybrid")
    soft_blocked = fm.get("doc_type") in set(empathy_cfg["topics"].get("blocked_soft", []))

    # 3) триггеры / LLM
    trig_cat, trig_hit = triggers_match(user_query, triggers_bank)
    llm_cat, llm_conf = (None, 0.0)
    if mode in ("llm", "hybrid") and (not trig_hit or mode == "llm"):
        llm_cat, llm_conf = llm_classify(user_query, retrieved_snippet, llm_client, model)

    # 4) мягкая блокировка с «пробоем»
    if soft_blocked:
        if empathy_cfg["override"].get("allow_on_soft_block", True):
            if trig_hit and empathy_cfg["override"].get("triggers_allow_override", True):
                return (trig_cat, "triggers_override", 1.0)
            if llm_cat and llm_conf >= float(empathy_cfg["override"].get("llm_min_confidence", 0.75)):
                return (llm_cat, "llm_override", llm_conf)
        return ("none", "soft_block", max(llm_conf, 0.0))

    # 5) обычные темы
    if mode == "triggers":
        return (trig_cat if trig_hit else "none", "triggers", 1.0 if trig_hit else 0.0)
    if mode == "llm":
        return (llm_cat if llm_conf >= float(empathy_cfg.get("min_confidence", 0.55)) else "none", "llm", llm_conf)
    # hybrid
    if trig_hit:
        return (trig_cat, "triggers", 1.0)
    return (llm_cat if llm_conf >= float(empathy_cfg.get("min_confidence", 0.55)) else "none", "llm", llm_conf)

def build_answer(core_text: str, emotion: str, empathy_bank: Dict[str, dict],
                 cfg: Dict[str, Any], rng: random.Random | None = None) -> str:
    if emotion == "none":
        return core_text
    rng = rng or random.Random()
    openers = empathy_bank.get(emotion, {}).get("openers", [])
    closers = empathy_bank.get(emotion, {}).get("closers", [])
    parts = []
    if rng.random() >= float(cfg.get("skip_opener_probability", 0.2)) and openers:
        parts.append(rng.choice(openers))
    parts.append(core_text)
    if len(parts) < int(cfg.get("max_phrases", 2)) + 1 and closers:
        parts.append(rng.choice(closers))
    return "\n\n".join(parts)
