# core/empathy_enhanced.py
"""
Улучшенный модуль эмпатии.
Более точная детекция эмоций и контекстная эмпатия.
"""

import random
from typing import Tuple, Dict, Any, Optional, List
from .empathy import detect_emotion, build_answer


def detect_emotion_enhanced(
    user_query: str,
    retrieved_snippet: str,
    frontmatter: Dict[str, Any],
    empathy_cfg: Dict[str, Any],
    triggers_bank: Dict[str, list[str]],
    llm_client=None,
    model: str = "gpt-4o-mini"
) -> Tuple[str, str, float]:
    """
    Улучшенная детекция эмоций с учетом контекста.
    
    Args:
        user_query: Запрос пользователя
        retrieved_snippet: Найденный фрагмент
        frontmatter: Метаданные frontmatter
        empathy_cfg: Конфигурация эмпатии
        triggers_bank: Банк триггеров
        llm_client: LLM клиент
        model: Модель LLM
    
    Returns:
        Tuple[emotion, detector, confidence]
    """
    
    # Используем базовую детекцию
    emotion, detector, confidence = detect_emotion(
        user_query, retrieved_snippet, frontmatter, 
        empathy_cfg, triggers_bank, llm_client, model
    )
    
    # Дополнительные улучшения
    if emotion != "none":
        # Повышаем уверенность для определенных типов запросов
        confidence = enhance_confidence_by_context(
            user_query, emotion, confidence, frontmatter
        )
        
        # Проверяем контекстную релевантность
        if not is_emotion_contextually_relevant(emotion, frontmatter, retrieved_snippet):
            return ("none", "context_filter", 0.0)
    
    return emotion, detector, confidence


def enhance_confidence_by_context(
    user_query: str,
    emotion: str,
    base_confidence: float,
    frontmatter: Dict[str, Any]
) -> float:
    """
    Повышает уверенность в эмоции на основе контекста.
    
    Args:
        user_query: Запрос пользователя
        emotion: Определенная эмоция
        base_confidence: Базовая уверенность
        frontmatter: Метаданные frontmatter
    
    Returns:
        Улучшенная уверенность
    """
    confidence = base_confidence
    
    # Проверяем соответствие эмоции и типа документа
    doc_type = frontmatter.get("doc_type", "")
    topic = frontmatter.get("topic", "")
    
    # Маппинг эмоций на типы документов
    emotion_doc_mapping = {
        "pain": ["implants", "safety"],
        "price": ["prices", "implants"],
        "trust": ["clinic", "doctors", "warranty"],
        "consultation": ["consultation", "clinic"],
        "safety": ["safety", "implants"],
        "reassurance": ["safety", "implants", "prices"]
    }
    
    # Повышаем уверенность если эмоция соответствует типу документа
    if emotion in emotion_doc_mapping:
        if doc_type in emotion_doc_mapping[emotion] or topic in emotion_doc_mapping[emotion]:
            confidence = min(1.0, confidence + 0.1)
    
    # Проверяем ключевые слова в запросе
    query_lower = user_query.lower()
    
    # Маппинг эмоций на ключевые слова
    emotion_keywords = {
        "pain": ["больно", "боль", "дискомфорт", "страшно", "боюсь"],
        "price": ["дорого", "стоимость", "цена", "дешево", "бюджет"],
        "trust": ["доверять", "надежно", "опыт", "квалификация", "гарантии"],
        "safety": ["безопасно", "риск", "осложнения", "побочные"],
        "consultation": ["консультация", "записаться", "прием", "визит"]
    }
    
    if emotion in emotion_keywords:
        keyword_matches = sum(1 for keyword in emotion_keywords[emotion] if keyword in query_lower)
        if keyword_matches > 0:
            confidence = min(1.0, confidence + 0.05 * keyword_matches)
    
    return confidence


def is_emotion_contextually_relevant(
    emotion: str,
    frontmatter: Dict[str, Any],
    retrieved_snippet: str
) -> bool:
    """
    Проверяет, релевантна ли эмоция контексту.
    
    Args:
        emotion: Определенная эмоция
        frontmatter: Метаданные frontmatter
        retrieved_snippet: Найденный фрагмент
    
    Returns:
        True если эмоция релевантна контексту
    """
    # Проверяем verbatim документы
    if frontmatter.get("verbatim", False):
        return False
    
    # Проверяем служебные типы документов
    doc_type = frontmatter.get("doc_type", "")
    blocked_types = ["contacts", "clinic", "warranty", "doctors"]
    
    if doc_type in blocked_types:
        return False
    
    # Проверяем соответствие эмоции и контента
    snippet_lower = retrieved_snippet.lower()
    
    # Маппинг эмоций на ключевые слова в контенте
    emotion_content_keywords = {
        "pain": ["боль", "дискомфорт", "анестезия", "обезболивание"],
        "price": ["стоимость", "цена", "рублей", "оплата"],
        "trust": ["опыт", "гарантия", "качество", "надежно"],
        "safety": ["безопасно", "риск", "осложнения", "стерильность"],
        "consultation": ["консультация", "запись", "прием", "визит"]
    }
    
    if emotion in emotion_content_keywords:
        keywords = emotion_content_keywords[emotion]
        return any(keyword in snippet_lower for keyword in keywords)
    
    return True


def build_answer_enhanced(
    core_text: str,
    emotion: str,
    empathy_bank: Dict[str, dict],
    cfg: Dict[str, Any],
    frontmatter: Dict[str, Any],
    user_query: str,
    rng: random.Random = None
) -> str:
    """
    Улучшенное построение ответа с эмпатией.
    
    Args:
        core_text: Основной текст ответа
        emotion: Определенная эмоция
        empathy_bank: Банк эмпатии
        cfg: Конфигурация
        frontmatter: Метаданные frontmatter
        user_query: Запрос пользователя
        rng: Генератор случайных чисел
    
    Returns:
        Ответ с эмпатией
    """
    if emotion == "none":
        return core_text
    
    rng = rng or random.Random()
    
    # Получаем эмпатию для эмоции
    empathy_data = empathy_bank.get(emotion, {})
    openers = empathy_data.get("openers", [])
    closers = empathy_data.get("closers", [])
    
    # Фильтруем эмпатию по контексту
    openers = filter_empathy_by_context(openers, frontmatter, user_query)
    closers = filter_empathy_by_context(closers, frontmatter, user_query)
    
    parts = []
    
    # Добавляем opener
    if (rng.random() >= float(cfg.get("skip_opener_probability", 0.2)) and 
        openers and len(parts) < int(cfg.get("max_phrases", 2))):
        parts.append(rng.choice(openers))
    
    # Добавляем основной текст
    parts.append(core_text)
    
    # Добавляем closer
    if (len(parts) < int(cfg.get("max_phrases", 2)) + 1 and 
        closers and not has_ui_cta(frontmatter)):
        parts.append(rng.choice(closers))
    
    return "\n\n".join(parts)


def filter_empathy_by_context(
    empathy_phrases: List[str],
    frontmatter: Dict[str, Any],
    user_query: str
) -> List[str]:
    """
    Фильтрует фразы эмпатии по контексту.
    
    Args:
        empathy_phrases: Список фраз эмпатии
        frontmatter: Метаданные frontmatter
        user_query: Запрос пользователя
    
    Returns:
        Отфильтрованный список фраз
    """
    if not empathy_phrases:
        return empathy_phrases
    
    filtered = []
    doc_type = frontmatter.get("doc_type", "")
    topic = frontmatter.get("topic", "")
    
    for phrase in empathy_phrases:
        # Пропускаем фразы, не подходящие по типу документа
        if doc_type in ["contacts", "clinic", "warranty", "doctors"]:
            if any(word in phrase.lower() for word in ["запишитесь", "консультация", "прием"]):
                continue
        
        # Пропускаем фразы, не подходящие по теме
        if topic == "prices" and "дорого" in phrase.lower():
            continue
            
        filtered.append(phrase)
    
    return filtered if filtered else empathy_phrases


def has_ui_cta(frontmatter: Dict[str, Any]) -> bool:
    """
    Проверяет, есть ли UI CTA в метаданных.
    
    Args:
        frontmatter: Метаданные frontmatter
    
    Returns:
        True если есть UI CTA
    """
    cta_text = frontmatter.get("cta_text", "")
    cta_link = frontmatter.get("cta_link", "")
    return bool(cta_text and cta_link)


def pick_empathy_enhanced(
    user_query: str,
    doc_type: str,
    has_ui_cta: bool,
    frontmatter: Dict[str, Any],
    empathy_bank: Dict[str, dict],
    empathy_cfg: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Улучшенный выбор эмпатии с учетом контекста.
    
    Args:
        user_query: Запрос пользователя
        doc_type: Тип документа
        has_ui_cta: Есть ли UI CTA
        frontmatter: Метаданные frontmatter
        empathy_bank: Банк эмпатии
        empathy_cfg: Конфигурация эмпатии
    
    Returns:
        Tuple[opener, closer]
    """
    
    # Проверяем блокировки
    if doc_type in {"contacts", "clinic", "warranty", "doctors"}:
        return None, None
    
    # Если есть UI CTA, не добавляем closer
    if has_ui_cta:
        return None, None
    
    # Определяем эмоцию
    emotion = frontmatter.get("emotion", "none")
    if emotion == "none":
        return None, None
    
    # Получаем эмпатию
    empathy_data = empathy_bank.get(emotion, {})
    openers = empathy_data.get("openers", [])
    closers = empathy_data.get("closers", [])
    
    # Фильтруем по контексту
    openers = filter_empathy_by_context(openers, frontmatter, user_query)
    closers = filter_empathy_by_context(closers, frontmatter, user_query)
    
    # Выбираем случайно
    opener = random.choice(openers) if openers else None
    closer = random.choice(closers) if closers else None
    
    return opener, closer
