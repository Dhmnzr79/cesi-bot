# core/legacy_adapter.py
"""
Адаптер для обратной совместимости с существующим кодом.
Позволяет плавно переходить от старого формата к новому.
"""

from typing import Dict, Any, Tuple, List, Optional
from .answer_builder import build_json, to_legacy_text, LOW_REL_JSON
from .guard import should_use_guard_response
from .normalize import normalize_query_for_search
from .followups import followups_from_frontmatter
from .md_loader import parse_frontmatter, extract_cta_from_frontmatter, extract_meta_from_frontmatter


def safe_followups(fm: Dict[str, Any], md_body: str) -> List[Dict[str, Any]]:
    """
    Безопасное получение followups без падений.
    """
    try:
        base = followups_from_frontmatter(fm)
    except Exception:
        base = []
    
    try:
        from .followups_enhanced import get_smart_followups
        smart = get_smart_followups(md_body, fm)
    except Exception:
        smart = []
    
    return (base or smart)[:3]


def _json_safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Делает метаданные безопасными для JSON сериализации.
    """
    meta = dict(meta or {})
    for k, v in meta.items():
        if isinstance(v, set):
            meta[k] = list(v)
    return meta
from config.feature_flags import feature_flags


def process_query_with_new_system(
    user_query: str,
    relevant_chunks: list,
    rag_meta: Dict[str, Any],
    candidates_with_scores: List[tuple] = None,
    payload: Optional[dict | str] = None,  # + добавили
) -> Tuple[str, Dict[str, Any]]:
    """
    Обрабатывает запрос с использованием новой системы.
    
    Args:
        user_query: Запрос пользователя
        relevant_chunks: Найденные релевантные чанки
        rag_meta: Метаданные от RAG системы
        candidates_with_scores: Кандидаты со скоррами для guard проверки
    
    Returns:
        Tuple[response_text, response_metadata]
    """
    
    # Перед postprocessing - достаём/нормализуем payload
    if payload is None:
        payload = rag_meta.get("response")  # мы кладём его выше в adapt_rag_response
    if not isinstance(payload, dict):
        payload = {"text": (str(payload) if payload is not None else "")}
    
    # Нормализация запроса (если включена)
    if feature_flags.is_enabled("ENABLE_NORMALIZATION"):
        normalized_query = normalize_query_for_search(user_query)
    else:
        normalized_query = user_query
    
    # Guard проверка (если включена)
    if feature_flags.is_enabled("ENABLE_GUARD"):
        from .guard import guard_with_candidates
        
        # Логируем meta до guard для диагностики
        print(f"🔍 legacy_adapter: rag_meta.keys()={list(rag_meta.keys())}")
        print(f"🔍 legacy_adapter: len(candidates_with_scores)={len(candidates_with_scores)}")
        
        # Диагностический лог перед guard
        import logging
        logger = logging.getLogger("cesi.legacy_adapter")
        logger.info({
            "ev": "pre-guard",
            "cand_cnt": len(rag_meta.get("candidates_with_scores") or []),
            "rel": rag_meta.get("relevance_score") or (rag_meta.get("meta") or {}).get("relevance_score"),
        })
        
        # Используем кандидаты со скоррами если доступны
        if candidates_with_scores:
            should_use_guard, guard_response, extracted_scores = guard_with_candidates(
                candidates_with_scores,
                feature_flags.get("GUARD_THRESHOLD", 0.35),
                rag_meta
            )
        else:
            # Fallback к старым скоррам из метаданных
            scores = {
                "bm25_top1": rag_meta.get("bm25_score", 0.0),
                "dense_top1": rag_meta.get("dense_score", 0.0),
                "rerank_top1": rag_meta.get("rerank_score", 0.0)
            }
            
            # Байпас: если есть кандидаты - guard не нужен
            cands = rag_meta.get("candidates_with_scores") or []
            if len(cands) > 0:
                should_use_guard, guard_response = False, None
            else:
                should_use_guard, guard_response = should_use_guard_response(
                    scores, 
                    feature_flags.get("GUARD_THRESHOLD", 0.35),
                    rag_meta
                )
            
            # на случай, если нижний слой вернул старый формат
            if isinstance(guard_response, tuple) and len(guard_response) == 2:
                payload, meta = guard_response
                if isinstance(payload, dict):
                    payload.setdefault("meta", {})
                    payload["meta"].update(meta or {})
                guard_response = payload
            
            extracted_scores = scores
        
        if should_use_guard:
            # Возвращаем guard-ответ
            if feature_flags.is_json_mode():
                return None, guard_response
            else:
                legacy_text = to_legacy_text(guard_response)
                return legacy_text, {
                    "guard_used": True, 
                    "relevance_score": extracted_scores.get("max_score", 0.0),
                    "guard_threshold": feature_flags.get("GUARD_THRESHOLD", 0.35)
                }
    
    # Обычная обработка
    if not relevant_chunks:
        # Нет релевантных чанков
        if feature_flags.is_json_mode():
            return None, LOW_REL_JSON
        else:
            legacy_text = to_legacy_text(LOW_REL_JSON)
            return legacy_text, {"no_chunks": True}
    
    # Извлекаем данные из первого чанка
    primary_chunk = relevant_chunks[0]
    
    # Парсим frontmatter если есть
    frontmatter = {}
    if hasattr(primary_chunk, 'metadata') and primary_chunk.metadata:
        # Конвертируем metadata в dict
        frontmatter = {
            "doc_type": getattr(primary_chunk.metadata, 'doc_type', 'info'),
            "topic": getattr(primary_chunk.metadata, 'topic', ''),
            "title": getattr(primary_chunk.metadata, 'title', ''),
            "cta_text": getattr(primary_chunk.metadata, 'cta_text', ''),
            "cta_link": getattr(primary_chunk.metadata, 'cta_link', ''),
            "tone": getattr(primary_chunk.metadata, 'tone', 'friendly'),
            "emotion": getattr(primary_chunk.metadata, 'emotion', ''),
            "verbatim": getattr(primary_chunk.metadata, 'verbatim', False)
        }
    
    # Извлекаем CTA
    cta_text, cta_link = extract_cta_from_frontmatter(frontmatter)
    
    # Извлекаем followups (если включены)
    followups = []
    if feature_flags.is_enabled("ENABLE_FOLLOWUPS"):
        from .followups_enhanced import get_smart_followups
        from .followups import filter_followups
        
        # Получаем контент для генерации followups
        content = ""
        if hasattr(primary_chunk, 'text'):
            content = primary_chunk.text
        
        # Безопасное получение followups
        followups = safe_followups(frontmatter, content)
        
        # 3) Убрать ссылку на текущий раздел (если это ответ по доп-H2)
        current_h2_id = None
        if hasattr(primary_chunk, 'metadata') and primary_chunk.metadata:
            current_h2_id = getattr(primary_chunk.metadata, 'h2_id', None)
        
        followups = filter_followups(followups, current_h2_id, limit=3)
    
    # Post-processing перед сборкой JSON
    from .postprocessing import clean_response_text, strip_textual_cta, clamp_text
    
    # Очищаем short и bullets
    # было (по стек-трейсу так и падало): 
    # short = clean_response_text(rag_meta.get("response", ""))
    # стало:
    text = ( (payload.get("text") if isinstance(payload, dict) else None) 
             or rag_meta.get("best_text")  # мы кладём его в rag_engine
             or "" )
    short = clean_response_text(text)
    short = clamp_text(short)
    bullets = [clean_response_text(b) for b in rag_meta.get("bullets", [])]
    
    # Удаляем текстовые CTA если есть UI CTA
    if cta_text and cta_link:
        short = strip_textual_cta(short, cta_text)
    
    # Строим JSON ответ
    meta = _json_safe_meta(extract_meta_from_frontmatter(frontmatter))
    meta["followups_position"] = "above_cta"  # подсказка фронту про порядок
    
    # финальный текст для виджета
    final_text = short or text
    payload["text"] = final_text
    
    json_response = build_json(
        short=short,
        bullets=bullets,
        cta_text=cta_text,
        cta_link=cta_link,
        followups=followups,
        used_chunks=rag_meta.get("used_chunks", []),
        meta=meta,
        warnings=rag_meta.get("warnings", [])
    )
    
    # Добавляем final_text для виджета
    json_response["response_mode"] = "json"  # ← ИСПРАВЛЕНО
    json_response["final_text"] = final_text
    json_response["text"] = final_text  # ← ВАЖНО: некоторые фронты читают text
    json_response["low_relevance"] = False  # + на бэку совместимость со старым фронтом
    json_response["has_ui_cta"] = bool(payload.get("cta"))  # + чтобы фронт не думал, что это фолбэк
    json_response["cta"] = payload.get("cta") or {}
    
    # Контрольный лог
    import logging
    logger = logging.getLogger("cesi.legacy_adapter")
    logger.info({"ev":"finalize", "final_text_len": len(final_text), "has_cta": bool(payload.get("cta"))})
    
    # Валидация структуры (если включена)
    if feature_flags.is_enabled("POSTPROCESS_STRIP_TEXTUAL_CTA"):
        from .postprocessing import validate_response_structure
        json_response = validate_response_structure(json_response)
    
    # Возвращаем в нужном формате
    if feature_flags.is_json_mode():
        return None, json_response
    else:
        # В legacy режиме возвращаем только short + bullets, без CTA/followups
        legacy_text = to_legacy_text(json_response)
        return legacy_text, {
            "json_converted": True,
            "legacy_mode": True,
            "cta_separated": bool(cta_text and cta_link),
            "followups_separated": len(followups) > 0
        }


def adapt_rag_response(
    user_query: str,
    rag_response: str,
    rag_meta: Dict[str, Any],
    relevant_chunks: list = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Адаптирует ответ от RAG системы к новому формату.
    
    Args:
        user_query: Запрос пользователя
        rag_response: Ответ от RAG системы
        rag_meta: Метаданные от RAG
        relevant_chunks: Релевантные чанки
    
    Returns:
        Tuple[response_text, response_metadata]
    """
    
    # Если новая система отключена, возвращаем как есть
    if feature_flags.is_legacy_mode() and not any([
        feature_flags.is_enabled("ENABLE_FOLLOWUPS"),
        feature_flags.is_enabled("ENABLE_EMPATHY"),
        feature_flags.is_enabled("ENABLE_GUARD")
    ]):
        return rag_response, rag_meta
    
    # Используем новую систему
    return process_query_with_new_system(
        user_query, 
        relevant_chunks or [], 
        {**rag_meta, "response": rag_response},
        rag_meta.get("candidates_with_scores", [])
    )
