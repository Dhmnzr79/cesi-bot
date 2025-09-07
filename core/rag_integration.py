# core/rag_integration.py
"""
Модуль интеграции с существующей RAG системой.
Обеспечивает совместимость и улучшения качества поиска.
"""

from typing import List, Tuple, Dict, Any, Optional
from .normalize import normalize_query_for_search
from .router import theme_router, apply_theme_boost_to_candidates
from .guard import guard_with_candidates
from config.feature_flags import feature_flags


def enhance_rag_retrieval(
    query: str,
    original_retrieve_func,
    *args,
    **kwargs
) -> Tuple[List, Dict[str, Any]]:
    """
    Улучшает поиск RAG системы с помощью новых компонентов.
    
    Args:
        query: Запрос пользователя
        original_retrieve_func: Оригинальная функция поиска
        *args, **kwargs: Аргументы для оригинальной функции
    
    Returns:
        Tuple[enhanced_chunks, enhancement_meta]
    """
    enhancement_meta = {
        "original_query": query,
        "normalization_applied": False,
        "theme_boost_applied": False,
        "guard_applied": False,
        "detected_themes": []  # Список вместо set для JSON сериализации
    }
    
    # Нормализация запроса (если включена)
    if feature_flags.is_enabled("ENABLE_NORMALIZATION"):
        normalized_query = normalize_query_for_search(query)
        enhancement_meta["normalization_applied"] = True
        enhancement_meta["normalized_query"] = normalized_query
        # Используем нормализованный запрос для поиска
        search_query = normalized_query
    else:
        search_query = query
    
    # Тематический роутинг
    detected_themes = theme_router.detect_themes(search_query)
    enhancement_meta["detected_themes"] = list(detected_themes)  # Конвертируем set в list
    
    # Вызываем оригинальную функцию поиска
    try:
        # Получаем кандидатов от оригинальной системы
        original_chunks = original_retrieve_func(search_query, *args)
        
        # Если получили кандидатов со скоррами, применяем улучшения
        if isinstance(original_chunks, list) and original_chunks:
            # Проверяем, есть ли скорры в кандидатах
            first_item = original_chunks[0]
            if isinstance(first_item, tuple) and len(first_item) == 2:
                # Кандидаты со скоррами: [(chunk, score), ...]
                candidates_with_scores = original_chunks
                
                # Применяем тематический буст
                if detected_themes:
                    enhanced_candidates = apply_theme_boost_to_candidates(
                        candidates_with_scores, 
                        detected_themes
                    )
                    enhancement_meta["theme_boost_applied"] = True
                else:
                    enhanced_candidates = candidates_with_scores
                
                # Guard проверка (если включена)
                if feature_flags.is_enabled("ENABLE_GUARD"):
                    should_use_guard, guard_response, extracted_scores = guard_with_candidates(
                        enhanced_candidates,
                        feature_flags.get("GUARD_THRESHOLD", 0.35)
                    )
                    
                    if should_use_guard:
                        enhancement_meta["guard_applied"] = True
                        enhancement_meta["guard_response"] = guard_response
                        enhancement_meta["relevance_scores"] = extracted_scores
                        # Возвращаем пустой список чанков - guard сработал
                        return [], enhancement_meta
                
                # Возвращаем улучшенных кандидатов
                return enhanced_candidates, enhancement_meta
            else:
                # Обычные чанки без скорров
                return original_chunks, enhancement_meta
        else:
            # Нет результатов
            return original_chunks, enhancement_meta
            
    except Exception as e:
        print(f"❌ Ошибка в enhance_rag_retrieval: {e}")
        # Fallback к оригинальной функции
        return original_retrieve_func(query, *args), enhancement_meta


def extract_candidates_from_rag_meta(rag_meta: Dict[str, Any]) -> Optional[List[Tuple]]:
    """
    Извлекает кандидатов со скоррами из метаданных RAG.
    
    Args:
        rag_meta: Метаданные от RAG системы
    
    Returns:
        Список кандидатов со скоррами или None
    """
    # Пытаемся найти кандидатов в разных местах метаданных
    candidates = None
    
    # Проверяем разные возможные ключи
    possible_keys = [
        "candidates", "candidates_with_scores", "search_results",
        "retrieved_chunks", "chunks_with_scores"
    ]
    
    for key in possible_keys:
        if key in rag_meta and isinstance(rag_meta[key], list):
            candidates = rag_meta[key]
            break
    
    return candidates


def integrate_with_existing_rag(
    query: str,
    rag_response: str,
    rag_meta: Dict[str, Any],
    relevant_chunks: List = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Интегрирует новую систему с существующей RAG.
    
    Args:
        query: Запрос пользователя
        rag_response: Ответ от RAG
        rag_meta: Метаданные RAG
        relevant_chunks: Релевантные чанки
    
    Returns:
        Tuple[enhanced_response, enhanced_meta]
    """
    from .legacy_adapter import process_query_with_new_system
    
    # Извлекаем кандидатов со скоррами из метаданных
    candidates_with_scores = extract_candidates_from_rag_meta(rag_meta)
    
    # Обрабатываем с новой системой
    enhanced_response, enhanced_meta = process_query_with_new_system(
        user_query=query,
        relevant_chunks=relevant_chunks or [],
        rag_meta={**rag_meta, "response": rag_response},
        candidates_with_scores=candidates_with_scores
    )
    
    return enhanced_response, enhanced_meta


def apply_jump_boost(candidates: List[Dict[str, Any]], jump: Dict[str, Any] = None, boost: float = 0.25) -> List[Dict[str, Any]]:
    """
    Применяет буст к кандидатам на основе jump данных.
    
    Args:
        candidates: Список кандидатов
        jump: Данные jump от фронтенда
        boost: Размер буста
    
    Returns:
        Список кандидатов с примененным бустом
    """
    if not (jump and candidates):
        return candidates
    
    h2_target = jump.get("h2_id")
    topic = jump.get("topic")
    
    boosted_count = 0
    for c in candidates:
        if (topic and c.get("topic") == topic) or (h2_target and (c.get("h2_id") or c.get("h2")) == h2_target):
            old_score = c.get("score", 0.0)
            c["score"] = old_score + boost
            boosted_count += 1
            print(f"🚀 Jump boost: {old_score:.3f} → {c['score']:.3f} (+{boost}) for {c.get('h2_id', c.get('h2', 'unknown'))}")
    
    if boosted_count > 0:
        print(f"✅ Applied jump boost to {boosted_count} candidates")
    
    return candidates
