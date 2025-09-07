# core/normalize.py
"""
Модуль для нормализации русских текстов.
Обработка особенностей русского языка для улучшения поиска.
"""

import re
# str - встроенный тип, не нужно импортировать


def normalize_ru(text: str) -> str:
    """
    Нормализует русский текст для поиска.
    
    Args:
        text: Исходный текст
    
    Returns:
        Нормализованный текст
    """
    if not text:
        return ""
    
    # Приводим к нижнему регистру
    normalized = text.lower()
    
    # Заменяем ё на е
    normalized = normalized.replace("ё", "е")
    
    # Убираем лишние пробелы
    normalized = normalized.strip()
    
    return normalized


def normalize_query_for_search(query: str) -> str:
    """
    Нормализует запрос пользователя для поиска.
    
    Args:
        query: Запрос пользователя
    
    Returns:
        Нормализованный запрос
    """
    if not query:
        return ""
    
    # Базовая нормализация
    normalized = normalize_ru(query)
    
    # Убираем знаки препинания в конце
    normalized = re.sub(r'[.!?]+$', '', normalized)
    
    # Нормализуем множественные пробелы
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()
