# core/logger.py
"""
Модуль логирования ответов бота.
Структурированные логи для анализа качества и отладки.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from config.feature_flags import feature_flags

# Именованные логгеры для разных типов логов
log_q = logging.getLogger("cesi.queries")
log_r = logging.getLogger("cesi.bot_responses")
log_m = logging.getLogger("cesi.minimal_logs")


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


def log_query(q: str, session_id: str = None) -> None:
    """Логирует запрос пользователя"""
    log_q.info(json.dumps({"q": q, "session": session_id or "unknown"}, ensure_ascii=False))


def log_response(q: str, candidates: List[Dict[str, Any]] = None, answer_len: int = 0) -> None:
    """Логирует ответ бота"""
    log_r.info(json.dumps({
        "q": q, 
        "candidates": candidates or [], 
        "answer_len": answer_len
    }, ensure_ascii=False))


def log_minimal(q: str, best_score: float, threshold: float, cta: bool, low_rel: bool = False) -> None:
    """Логирует минимальную информацию о запросе"""
    log_m.info(json.dumps({
        "q": q, 
        "best": best_score, 
        "thr": threshold, 
        "cta": cta, 
        "low_rel": low_rel
    }, ensure_ascii=False))


def format_candidates_for_log(candidates: list[Any], top: int | None = None):
    """
    Универсальная функция для форматирования кандидатов в логах.
    
    Args:
        candidates: Список кандидатов (кортежи, объекты, словари)
        top: Максимальное количество кандидатов для логирования
    
    Returns:
        Список словарей с информацией о кандидатах
    """
    out = []
    def to_item(cand, rank):
        # вариант 1: кортеж/список (chunk, score)
        if isinstance(cand, (tuple, list)) and len(cand) >= 2:
            chunk, score = cand[0], cand[1]
        else:
            chunk, score = cand, getattr(cand, "score", getattr(cand, "hybrid", getattr(cand, "rerank", None)))
        h2  = getattr(chunk, "h2", getattr(chunk, "h2_id", None))
        doc = getattr(chunk, "doc", getattr(chunk, "file", getattr(chunk, "doc_id", None)))
        try:
            s = float(score) if score is not None else None
        except Exception:
            s = None
        return {"rank": rank, "score": s, "h2": h2, "doc": doc}
    for i, c in enumerate(candidates, 1):
        out.append(to_item(c, i))
        if top and i >= top:
            break
    return out
