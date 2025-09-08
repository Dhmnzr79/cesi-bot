# core/logger.py
"""
Модуль логирования ответов бота.
Структурированные логи для анализа качества и отладки.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from config.feature_flags import feature_flags


def log_bot_response(
    user_query: str,
    response_data: Dict[str, Any],
    theme_hint: Optional[Dict[str, Any]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    scores: Optional[Dict[str, float]] = None,
    relevance_score: Optional[float] = None,
    guard_threshold: Optional[float] = None,
    low_relevance: bool = False,
    shown_cta: bool = False,
    followups_count: int = 0,
    opener_used: bool = False,
    closer_used: bool = False
) -> None:
    """
    Логирует структурированный ответ бота.
    
    Args:
        user_query: Запрос пользователя
        response_data: Данные ответа (JSON или legacy)
        theme_hint: Информация о теме запроса
        candidates: Список кандидатов с деталями
        scores: Скорры релевантности
        relevance_score: Общий скорр релевантности
        guard_threshold: Порог guard системы
        low_relevance: Флаг низкой релевантности
        shown_cta: Показана ли CTA
        followups_count: Количество followups
        opener_used: Использован ли opener эмпатии
        closer_used: Использован ли closer эмпатии
    """
    
    # Создаем структуру лога
    meta = response_data.get("meta", {})
    followups = response_data.get("followups", [])
    
    log_entry = {
        "ts": datetime.now().isoformat() + "Z",
        "user_query": user_query,
        "theme_hint": theme_hint or {},
        "primary_doc_type": meta.get("doc_type", "unknown"),
        "meta": {
            "doc_type": meta.get("doc_type", "unknown"),
            "topic": meta.get("topic", ""),
            "low_relevance": meta.get("low_relevance", False),
            "has_ui_cta": meta.get("has_ui_cta", False)
        },
        "candidates": candidates or [],
        "evidence": _extract_evidence(response_data),
        "scores": scores or {},
        "relevance_score": relevance_score,
        "guard_threshold": guard_threshold,
        "low_relevance": low_relevance,
        "shown_cta": shown_cta,
        "opener_used": opener_used,
        "closer_used": closer_used,
        "followups_count": followups_count,
        "followups_labels": [f.get("label", "") for f in followups if isinstance(f, dict)],
        "response_mode": feature_flags.get("RESPONSE_MODE")
    }
    
    # Записываем в файл
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        log_file = logs_dir / "bot_responses.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        print(f"📝 Лог записан: {log_file}")
        
    except Exception as e:
        print(f"❌ Ошибка логирования: {e}")


def _extract_evidence(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает информацию об источнике ответа.
    
    Args:
        response_data: Данные ответа
    
    Returns:
        Словарь с информацией об источнике
    """
    evidence = {}
    
    # Для JSON формата
    if "answer" in response_data:
        used_chunks = response_data.get("used_chunks", [])
        if used_chunks:
            # Берем первый чанк как основной источник
            first_chunk = used_chunks[0]
            if isinstance(first_chunk, str) and "#" in first_chunk:
                file_part, section_part = first_chunk.split("#", 1)
                evidence = {
                    "file": file_part,
                    "h2": section_part
                }
    
    # Для legacy формата
    elif "used_chunks" in response_data:
        used_chunks = response_data["used_chunks"]
        if used_chunks:
            first_chunk = used_chunks[0]
            if isinstance(first_chunk, str) and "#" in first_chunk:
                file_part, section_part = first_chunk.split("#", 1)
                evidence = {
                    "file": file_part,
                    "h2": section_part
                }
    
    return evidence


def format_candidates_for_log(candidates: List[tuple]) -> List[Dict[str, Any]]:
    """
    Форматирует кандидатов для логирования.
    
    Args:
        candidates: Список (chunk, score) кортежей
    
    Returns:
        Список словарей с информацией о кандидатах
    """
    formatted = []
    
    for chunk, score in candidates:
        if hasattr(chunk, 'file_name') and hasattr(chunk, 'text'):
            # Извлекаем заголовок из текста
            h2_match = None
            if hasattr(chunk, 'text'):
                import re
                h2_match = re.search(r'(?m)^##\s+(.+?)\s*$', chunk.text)
            
            formatted.append({
                "file": chunk.file_name,
                "h2": h2_match.group(1) if h2_match else "unknown",
                "source": "chunk",
                "score": score
            })
    
    return formatted
