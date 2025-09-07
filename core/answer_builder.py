# core/answer_builder.py
"""
Модуль для построения единого JSON-API контракта ответов.
Централизованная сборка структурированных ответов бота.
"""

from typing import Dict, List, Optional, Any
import json


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
        followups: Список связанных вопросов
        used_chunks: ID использованных чанков
        meta: Дополнительные метаданные
        warnings: Предупреждения/ошибки
    
    Returns:
        Dict с полной структурой ответа
    """
    
    # Собираем секцию answer
    answer = {
        "short": short.strip() if short else "",
        "bullets": bullets or [],
        "body_md": body_md
    }
    
    # Собираем CTA (только если есть и текст, и ссылка)
    cta = {}
    if cta_text and cta_link:
        cta = {
            "text": cta_text,
            "link": cta_link,
            "type": "primary"
        }
    
    # Собираем метаданные
    if meta is None:
        meta = {}
    
    final_meta = {
        "low_relevance": False,
        "has_ui_cta": bool(cta_text and cta_link),
        **meta
    }
    
    # Собираем финальный ответ
    response = {
        "answer": answer,
        "cta": cta,
        "followups": followups or [],
        "used_chunks": used_chunks or [],
        "meta": final_meta,
        "warnings": warnings or []
    }
    
    return response


def build_low_relevance_json(
    message: str = "К сожалению, в моей базе нет информации по этому вопросу.",
    cta_text: Optional[str] = None,
    cta_link: Optional[str] = None
) -> Dict[str, Any]:
    """
    Строит JSON для случаев низкой релевантности.
    
    Args:
        message: Сообщение о низкой релевантности
        cta_text: Текст CTA (опционально)
        cta_link: Ссылка CTA (опционально)
    
    Returns:
        Dict с структурой ответа низкой релевантности
    """
    
    answer = {
        "short": message,
        "bullets": [],
        "body_md": None
    }
    
    cta = {}
    if cta_text and cta_link:
        cta = {
            "text": cta_text,
            "link": cta_link,
            "type": "primary"
        }
    
    meta = {
        "low_relevance": True,
        "has_ui_cta": bool(cta_text and cta_link)
    }
    
    return {
        "answer": answer,
        "cta": cta,
        "followups": [],
        "used_chunks": [],
        "meta": meta,
        "warnings": []
    }


def to_legacy_text(resp_json: Dict[str, Any]) -> str:
    """
    Конвертирует новый JSON-ответ в старый текстовый формат для обратной совместимости.
    
    Args:
        resp_json: JSON-ответ в новом формате
    
    Returns:
        Строка в старом формате
    """
    answer = resp_json.get("answer", {})
    parts = []
    
    # Добавляем короткий ответ
    short = answer.get("short", "").strip()
    if short:
        parts.append(short)
    
    # Добавляем bullets
    bullets = answer.get("bullets", [])
    if bullets:
        bullet_text = "\n• " + "\n• ".join(bullets)
        parts.append(bullet_text)
    
    # Возвращаем объединенный текст
    return "\n".join([p for p in parts if p])


# Константы для feature flags
LOW_REL_JSON = build_low_relevance_json(
    "К сожалению, в моей базе нет информации по этому вопросу.",
    "Записаться на консультацию"
)

