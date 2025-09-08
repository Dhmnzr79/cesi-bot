# core/answer_builder.py
"""
Модуль для построения единого JSON-API контракта ответов.
Централизованная сборка структурированных ответов бота.
"""

from typing import Dict, List, Optional, Any
import json
import re

# Импорты для эмпатии и CTA
from core import empathy, cta

def build_json(
    short: str = "",
    bullets: Optional[List[str]] = None,
    body_md: Optional[str] = None,
    cta_text: Optional[str] = None,
    cta_link: Optional[str] = None,
    followups: Optional[List[Dict]] = None,
    used_chunks: Optional[List[str]] = None,
    meta: Optional[Dict] = None,
    warnings: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Строит единый JSON-ответ согласно контракту API.
    
    Args:
        short: Короткий ответ по сути вопроса
        bullets: Список деталей/пунктов
        body_md: Markdown контент (альтернатива short+bullets)
        cta_text: Текст призыва к действию
        cta_link: Ссылка для CTA
        followups: Список follow-up вопросов
        used_chunks: Список использованных чанков
        meta: Метаданные ответа
        warnings: Предупреждения
    
    Returns:
        Dict с полной структурой ответа
    """
    response = {
        "version": "1.0",
        "timestamp": meta.get("timestamp") if meta else None,
        "response": {
            "text": short,
            "bullets": bullets or [],
            "body_md": body_md
        },
        "cta": {
            "text": cta_text,
            "link": cta_link
        } if cta_text else None,
        "followups": followups or [],
        "meta": {
            "used_chunks": used_chunks or [],
            "confidence": meta.get("confidence") if meta else None,
            "source": meta.get("source") if meta else None,
            "warnings": warnings or []
        }
    }
    
    # Убираем None значения
    response = {k: v for k, v in response.items() if v is not None}
    if response.get("response"):
        response["response"] = {k: v for k, v in response["response"].items() if v is not None}
    if response.get("meta"):
        response["meta"] = {k: v for k, v in response["meta"].items() if v is not None}
    
    return response

# --- Back-compat для Legacy кода ---
LOW_REL_JSON = {
    "text": "Сейчас не могу ответить. Напишите вопрос чуть короче или позвоните нам.",
    "meta": {"source": "guard", "relevance_score": None}
}

def to_legacy_text(payload):
    """Legacy UI ждёт plain text. Берём 'text' из payload."""
    try:
        return payload.get("text", "") if isinstance(payload, dict) else (payload or "")
    except Exception:
        return ""

def postprocess(answer_text: str, user_text: str, intent: str, topic_meta: dict, session: dict) -> dict:
    """
    1) вставляем эмпатию-опенер или бридж (одна короткая строка)
    2) решаем CTA (одна кнопка)
    """
    # 0. убедимся, что конфиги загружены (однократно)
    if not hasattr(empathy, '_CFG') or not empathy._CFG:
        empathy.load_config()
    if not hasattr(cta, '_CFG') or not cta._CFG:
        cta.load_config()
    
    # 1) эмпатия/бридж
    opener_or_bridge = None
    try:
        opener_or_bridge = empathy.maybe_opener_or_bridge(
            answer_text=answer_text,
            user_text=user_text,
            topic_meta=topic_meta or {},
            session=session,
            intent=intent
        )
    except Exception as e:
        print(f"Empathy error: {e}")
    
    if opener_or_bridge:
        answer_text = f"{answer_text}\n\n_{opener_or_bridge}_"
    
    # 2) CTA
    cta_obj = None
    try:
        cta_obj = cta.decide_cta(
            intent=intent,
            topic_meta=topic_meta or {},
            session=session
        )
    except Exception as e:
        print(f"CTA error: {e}")
    
    # 3) сборка payload'а
    payload = {"text": answer_text}
    
    if cta_obj:
        payload["cta"] = cta_obj
        
        # если фронт пока legacy через action_buttons
        if session.get("legacy_mode"):
            btn = {
                "text": cta_obj["label"],
                "action": cta_obj.get("url"),
                "variant": "primary"
            }
            payload["action_buttons"] = [btn]
    
    return payload

def extract_bullets(text: str) -> List[str]:
    """
    Извлекает bullet points из текста.
    Ищет строки, начинающиеся с -, *, • или цифр.
    """
    bullets = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Проверяем различные форматы bullet points
        if (line.startswith('- ') or 
            line.startswith('* ') or 
            line.startswith('• ') or
            re.match(r'^\d+\.\s+', line)):
            # Убираем маркер и добавляем в список
            bullet = re.sub(r'^[-*•]\s+', '', line)
            bullet = re.sub(r'^\d+\.\s+', '', bullet)
            if bullet:
                bullets.append(bullet)
    
    return bullets

def format_response(response_data: Dict[str, Any], format_type: str = "json") -> str:
    """
    Форматирует ответ в нужном формате.
    
    Args:
        response_data: Данные ответа
        format_type: Тип форматирования ("json", "text", "markdown")
    
    Returns:
        Отформатированная строка
    """
    if format_type == "json":
        return json.dumps(response_data, ensure_ascii=False, indent=2)
    
    elif format_type == "text":
        text_parts = []
        
        if response_data.get("response", {}).get("text"):
            text_parts.append(response_data["response"]["text"])
        
        if response_data.get("response", {}).get("bullets"):
            for bullet in response_data["response"]["bullets"]:
                text_parts.append(f"• {bullet}")
        
        if response_data.get("cta", {}).get("text"):
            text_parts.append(f"\n{response_data['cta']['text']}")
        
        return "\n".join(text_parts)
    
    elif format_type == "markdown":
        md_parts = []
        
        if response_data.get("response", {}).get("text"):
            md_parts.append(response_data["response"]["text"])
        
        if response_data.get("response", {}).get("bullets"):
            md_parts.append("")
            for bullet in response_data["response"]["bullets"]:
                md_parts.append(f"- {bullet}")
        
        if response_data.get("cta", {}).get("text"):
            md_parts.append(f"\n**{response_data['cta']['text']}**")
        
        return "\n".join(md_parts)
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")

# Legacy функции для обратной совместимости
def build_legacy_response(text: str, bullets: List[str] = None, cta_text: str = None, cta_link: str = None) -> Dict[str, Any]:
    """Строит ответ в старом формате для обратной совместимости."""
    response = {"text": text}
    
    if bullets:
        response["bullets"] = bullets
    
    if cta_text:
        response["cta"] = {"text": cta_text, "link": cta_link}
    
    return response

# --- Back-compat для Legacy кода ---
LOW_REL_JSON = {
    "text": "Сейчас не могу ответить. Напишите вопрос чуть короче или позвоните нам.",
    "meta": {"source": "guard", "relevance_score": None}
}

def to_legacy_text(payload):
    """Legacy UI ждёт plain text. Берём 'text' из payload."""
    try:
        return payload.get("text", "") if isinstance(payload, dict) else (payload or "")
    except Exception:
        return ""