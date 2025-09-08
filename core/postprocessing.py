# core/postprocessing.py
"""
Модуль post-processing для очистки ответов.
Удаление дублирующихся CTA и улучшение текста.
"""

import re
from typing import Dict, Any, Optional


def strip_textual_cta(text: str, cta_text: Optional[str] = None) -> str:
    """
    Удаляет текстовые CTA из ответа если есть UI CTA.
    
    Args:
        text: Текст ответа
        cta_text: Текст CTA кнопки
    
    Returns:
        Очищенный текст
    """
    if not text or not cta_text:
        return text
    
    # Паттерны для поиска CTA в тексте
    cta_patterns = [
        r"запишитесь\s+на\s+консультацию[.!]?",
        r"записаться\s+на\s+консультацию[.!]?",
        r"запишитесь\s+к\s+врачу[.!]?",
        r"записаться\s+к\s+врачу[.!]?",
        r"обратитесь\s+к\s+специалисту[.!]?",
        r"свяжитесь\s+с\s+нами[.!]?",
        r"позвоните\s+нам[.!]?",
        r"приходите\s+на\s+консультацию[.!]?",
        r"приходите\s+к\s+нам[.!]?",
        r"запишитесь\s+на\s+прием[.!]?",
        r"записаться\s+на\s+прием[.!]?",
        r"запишитесь\s+на\s+визит[.!]?",
        r"записаться\s+на\s+визит[.!]?",
    ]
    
    # Добавляем паттерн на основе конкретного CTA текста
    if cta_text:
        cta_escaped = re.escape(cta_text.lower())
        cta_patterns.append(cta_escaped)
    
    cleaned_text = text
    
    # Удаляем найденные CTA
    for pattern in cta_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    
    # Очищаем лишние пробелы и переносы
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Множественные переносы
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Множественные пробелы
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def clean_response_text(s):
    """
    Очищает текст ответа от служебных элементов.
    
    Args:
        s: Исходный текст (может быть None, dict, str)
    
    Returns:
        Очищенный текст
    """
    # защита от None/не-строк
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    
    if not s:
        return s
    
    # 1) убрать HTML-комментарии (в т.ч. aliases)
    s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL)
    
    # 2) убрать Markdown-заголовки в тексте
    s = re.sub(r'^\s*#{1,6}\s*', '', s, flags=re.MULTILINE)
    
    # 3) убрать «---» и одиночные маркеры
    s = re.sub(r'^\s*---+\s*$', '', s, flags=re.MULTILINE)
    s = re.sub(r'^\s*[-.]\s*$', '', s, flags=re.MULTILINE)
    
    # 4) склеить множественные пробелы/переносы
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    
    # 5) убрать повторы слов подряд (очень частый артефакт)
    s = re.sub(r'(?i)\b([а-яёa-z0-9]{2,})(?:\s+\1){1,}\b', r'\1', s)
    
    # 6) косметика тире
    s = re.sub(r'\s*-\s*-\s*', '-', s)
    
    return s.strip()

def clamp_text(s, max_chars=900, max_lines=14):
    """Обрезает текст до разумных пределов"""
    lines = [ln.strip() for ln in (s or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    
    lines = lines[:max_lines]
    s = '\n'.join(lines)
    return (s[:max_chars] + '...') if len(s) > max_chars else s


def enhance_bullets_formatting(bullets: list) -> list:
    """
    Улучшает форматирование списков.
    
    Args:
        bullets: Список элементов
    
    Returns:
        Улучшенный список
    """
    if not bullets:
        return bullets
    
    enhanced = []
    
    for bullet in bullets:
        if not bullet or not isinstance(bullet, str):
            continue
        
        # Очищаем от служебных символов
        clean_bullet = clean_response_text(bullet)
        
        # Проверяем минимальную длину
        if len(clean_bullet) > 5:
            enhanced.append(clean_bullet)
    
    return enhanced


def postprocess_answer(
    answer_data: Dict[str, Any],
    cta_text: Optional[str] = None,
    cta_link: Optional[str] = None
) -> Dict[str, Any]:
    """
    Выполняет post-processing ответа.
    
    Args:
        answer_data: Данные ответа
        cta_text: Текст CTA
        cta_link: Ссылка CTA
    
    Returns:
        Обработанные данные ответа
    """
    processed = answer_data.copy()
    
    # Обрабатываем short
    if "short" in processed:
        processed["short"] = clean_response_text(processed["short"])
        
        # Удаляем CTA из текста если есть UI CTA
        if cta_text and cta_link:
            processed["short"] = strip_textual_cta(processed["short"], cta_text)
    
    # Обрабатываем bullets
    if "bullets" in processed:
        processed["bullets"] = enhance_bullets_formatting(processed["bullets"])
    
    # Обрабатываем body_md
    if "body_md" in processed and processed["body_md"]:
        processed["body_md"] = clean_response_text(processed["body_md"])
        
        # Удаляем CTA из body_md если есть UI CTA
        if cta_text and cta_link:
            processed["body_md"] = strip_textual_cta(processed["body_md"], cta_text)
    
    return processed


def remove_duplicate_cta_from_text(text: str, cta_text: Optional[str] = None) -> str:
    """
    Удаляет дублирующиеся CTA из текста.
    
    Args:
        text: Текст ответа
        cta_text: Текст CTA кнопки
    
    Returns:
        Текст без дублирующихся CTA
    """
    if not text or not cta_text:
        return text
    
    # Ищем повторяющиеся CTA в тексте
    cta_lower = cta_text.lower()
    text_lower = text.lower()
    
    # Считаем количество вхождений CTA
    cta_count = text_lower.count(cta_lower)
    
    # Если CTA встречается больше одного раза, удаляем лишние
    if cta_count > 1:
        # Оставляем только первое вхождение
        parts = text.split(cta_text, 1)
        if len(parts) == 2:
            # Удаляем CTA из второй части
            second_part = strip_textual_cta(parts[1], cta_text)
            text = parts[0] + cta_text + second_part
    
    return text


def validate_response_structure(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидирует и исправляет структуру ответа.
    
    Args:
        response_data: Данные ответа
    
    Returns:
        Валидированные данные ответа
    """
    validated = response_data.copy()
    
    # Проверяем обязательные поля
    if "answer" not in validated:
        validated["answer"] = {}
    
    answer = validated["answer"]
    
    # Устанавливаем значения по умолчанию
    if "short" not in answer:
        answer["short"] = ""
    
    if "bullets" not in answer:
        answer["bullets"] = []
    
    if "body_md" not in answer:
        answer["body_md"] = None
    
    # Проверяем типы данных
    if not isinstance(answer["bullets"], list):
        answer["bullets"] = []
    
    # Проверяем CTA
    if "cta" not in validated:
        validated["cta"] = {}
    
    # Проверяем followups
    if "followups" not in validated:
        validated["followups"] = []
    
    if not isinstance(validated["followups"], list):
        validated["followups"] = []
    
    # Проверяем метаданные
    if "meta" not in validated:
        validated["meta"] = {}
    
    meta = validated["meta"]
    
    # Устанавливаем значения по умолчанию для метаданных
    if "low_relevance" not in meta:
        meta["low_relevance"] = False
    
    if "has_ui_cta" not in meta:
        meta["has_ui_cta"] = bool(validated.get("cta", {}).get("text"))
    
    # Проверяем warnings
    if "warnings" not in validated:
        validated["warnings"] = []
    
    if not isinstance(validated["warnings"], list):
        validated["warnings"] = []
    
    return validated
