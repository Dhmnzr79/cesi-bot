# core/guard.py
"""
Модуль Guard системы для защиты от галлюцинаций.
Проверяет релевантность найденной информации перед ответом.
"""

from typing import Dict, Tuple, Any, Optional, List
from .answer_builder import LOW_REL_JSON


def _extract_score(item: Any) -> Optional[float]:
    """
    Извлекает скорр из элемента в любом формате.
    
    Args:
        item: Элемент с скорром (кортеж, словарь, объект)
    
    Returns:
        Скорр или None если не найден
    """
    # Старый формат: (chunk, score)
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return float(item[1])
    
    # Новый формат: {"chunk": ..., "score": ...}
    if isinstance(item, dict) and "score" in item:
        return float(item["score"])
    
    # Альтернативный формат: {"scores": {"final": ..., "bm25": ..., ...}}
    if isinstance(item, dict) and "scores" in item:
        scores_dict = item["scores"]
        for key in ["final", "bm25", "dense", "rerank", "top1", "relevance"]:
            if key in scores_dict:
                return float(scores_dict[key])
    
    # Объект с атрибутом score
    if hasattr(item, "score"):
        try:
            return float(getattr(item, "score"))
        except (ValueError, TypeError):
            return None
    
    return None


def extract_scores_from_candidates(candidates: List[Any], top_n: int = 3) -> Dict[str, float]:
    """
    Извлекает скорры из кандидатов для guard проверки.
    Поддерживает оба формата: кортежи и словари.
    
    Args:
        candidates: Список кандидатов (кортежи или словари)
        top_n: Количество топ кандидатов для анализа
    
    Returns:
        Словарь с максимальными скоррами
    """
    if not candidates:
        return {}
    
    # Берем топ кандидатов
    top_candidates = candidates[:top_n]
    
    # Извлекаем скорры с поддержкой разных форматов
    scores = []
    for item in top_candidates:
        score = _extract_score(item)
        if score is not None:
            scores.append(score)
    
    if not scores:
        return {}
    
    # Возвращаем максимальный скорр как общий показатель релевантности
    max_score = max(scores)
    
    return {
        "max_score": max_score,
        "avg_score": sum(scores) / len(scores),
        "top1_score": scores[0] if scores else 0.0
    }


def apply_guard(scores: Dict[str, float], threshold: float = 0.35) -> Tuple[bool, float]:
    """
    Применяет guard-проверку релевантности.
    
    Args:
        scores: Словарь со скоррами релевантности
        threshold: Порог релевантности (по умолчанию 0.35)
    
    Returns:
        Tuple[is_low_relevance, relevance_score]
    """
    # Извлекаем доступные скорры
    available_scores = []
    
    # Проверяем разные варианты названий скорров
    score_keys = [
        "max_score", "top1_score", "avg_score",  # Из extract_scores_from_candidates
        "bm25_top1", "dense_top1", "rerank_top1",  # Прямые скорры
        "bm25_score", "dense_score", "rerank_score",  # Альтернативные названия
        "final_score", "relevance_score"  # Общие названия
    ]
    
    for key in score_keys:
        if key in scores and scores[key] is not None:
            available_scores.append(scores[key])
    
    # Вычисляем максимальный скорр релевантности
    if available_scores:
        relevance_score = max(available_scores)
    else:
        relevance_score = 0.0
    
    # Проверяем порог
    is_low_relevance = relevance_score < threshold
    
    return is_low_relevance, relevance_score


def apply_guard_with_bypasses(
    scores: Dict[str, float], 
    threshold: float = 0.35,
    meta: Dict[str, Any] = None
) -> Tuple[bool, float]:
    """
    Применяет guard с обходами для высокоуверенных сигналов
    
    Args:
        scores: Словарь скорров релевантности
        threshold: Порог релевантности
        meta: Метаданные для проверки обходов
    
    Returns:
        Tuple[is_low_relevance, relevance_score]
    """
    # Сначала применяем обычный guard
    is_low_relevance, relevance_score = apply_guard(scores, threshold)
    
    # Если уже не low_relevance, возвращаем как есть
    if not is_low_relevance:
        return is_low_relevance, relevance_score
    
    # Проверяем высокоуверенные сигналы для обхода
    if meta:
        high_confidence = any([
            meta.get("source") == "alias",          # сработал alias_fallback
            meta.get("doctor_card_hit") is True,    # найден Н2/Н3 карточки врача
            meta.get("exact_h2_match") is True,     # точное совпадение заголовка
        ])
        
        if high_confidence:
            print(f"✅ Guard bypass: high_confidence signal detected")
            return False, relevance_score  # Принудительно не low_relevance
    
    return is_low_relevance, relevance_score


def should_use_guard_response(
    scores: Dict[str, float], 
    threshold: float = 0.35,
    meta: Dict[str, Any] = None
) -> Tuple[bool, Optional[Dict]]:
    """
    Определяет, нужно ли использовать guard-ответ.
    
    Args:
        scores: Словарь со скоррами
        threshold: Порог релевантности
        meta: Метаданные для проверки обходов
    
    Returns:
        Tuple[should_use_guard, guard_response_json]
    """
    is_low_relevance, relevance_score = apply_guard_with_bypasses(scores, threshold, meta)
    
    if is_low_relevance:
        # Логируем причины guard'а
        if meta:
            meta["guard_reason"] = {
                "relevance_score": relevance_score,
                "threshold": threshold,
                "alias": meta.get("source") == "alias",
                "doctor_card_hit": meta.get("doctor_card_hit", False),
            }
        
        # Возвращаем стандартный guard-ответ
        guard_response = LOW_REL_JSON.copy()
        guard_response["meta"]["relevance_score"] = relevance_score
        guard_response["meta"]["guard_threshold"] = threshold
        return True, guard_response
    
    return False, None


def guard_with_candidates(
    candidates: List[Tuple], 
    threshold: float = 0.35,
    meta: Dict[str, Any] = None
) -> Tuple[bool, Optional[Dict], Dict[str, float]]:
    """
    Применяет guard проверку к кандидатам.
    
    Args:
        candidates: Список (chunk, score) кортежей
        threshold: Порог релевантности
        meta: Метаданные для проверки обходов
    
    Returns:
        Tuple[should_use_guard, guard_response_json, extracted_scores]
    """
    # Извлекаем скорры из кандидатов
    extracted_scores = extract_scores_from_candidates(candidates)
    
    # Применяем guard проверку
    should_use_guard, guard_response = should_use_guard_response(extracted_scores, threshold, meta)
    
    return should_use_guard, guard_response, extracted_scores
