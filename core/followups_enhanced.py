# core/followups_enhanced.py
"""
Улучшенный модуль для работы с followups.
Автоматическое генерирование связанных вопросов на основе контента.
"""

import re
from typing import Dict, List, Any, Optional
from .md_loader import parse_frontmatter, extract_followups_from_frontmatter


def generate_followups_from_content(
    content: str, 
    frontmatter: Dict[str, Any],
    max_followups: int = 3
) -> List[Dict[str, Any]]:
    """
    Генерирует followups на основе контента MD файла.
    
    Args:
        content: Содержимое MD файла
        frontmatter: Метаданные frontmatter
        max_followups: Максимальное количество followups
    
    Returns:
        Список followups в формате API
    """
    followups = []
    
    # Сначала проверяем, есть ли followups в frontmatter
    frontmatter_followups = extract_followups_from_frontmatter(frontmatter)
    if frontmatter_followups:
        return frontmatter_followups[:max_followups]
    
    # Генерируем followups на основе заголовков
    h2_sections = extract_h2_sections(content)
    
    for section in h2_sections[:max_followups]:
        followup = create_followup_from_section(section, frontmatter)
        if followup:
            followups.append(followup)
    
    return followups


def extract_h2_sections(content: str) -> List[Dict[str, str]]:
    """
    Извлекает секции H2 из контента.
    
    Args:
        content: Содержимое MD файла
    
    Returns:
        Список секций с заголовками и содержимым
    """
    sections = []
    
    # Разбиваем по H2 заголовкам
    h2_blocks = re.split(r'(?m)^##\s+', content)
    
    for block in h2_blocks[1:]:  # Пропускаем первый блок (до первого H2)
        lines = block.split('\n')
        if not lines:
            continue
            
        title = lines[0].strip()
        body = '\n'.join(lines[1:]).strip()
        
        # Пропускаем слишком короткие секции
        if len(body) < 50:
            continue
            
        sections.append({
            "title": title,
            "body": body,
            "h2_id": slugify_title(title)
        })
    
    return sections


def create_followup_from_section(
    section: Dict[str, str], 
    frontmatter: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Создает followup на основе секции.
    
    Args:
        section: Секция с заголовком и содержимым
        frontmatter: Метаданные frontmatter
    
    Returns:
        Followup в формате API или None
    """
    title = section["title"]
    h2_id = section["h2_id"]
    
    # Создаем вопрос на основе заголовка
    query = create_question_from_title(title, frontmatter)
    
    if not query:
        return None
    
    return {
        "label": title,
        "query": query,
        "jump": {
            "topic": frontmatter.get("topic", ""),
            "h2_id": h2_id
        }
    }


def create_question_from_title(title: str, frontmatter: Dict[str, Any]) -> Optional[str]:
    """
    Создает вопрос на основе заголовка секции.
    
    Args:
        title: Заголовок секции
        frontmatter: Метаданные frontmatter
    
    Returns:
        Сгенерированный вопрос или None
    """
    # Маппинг заголовков на вопросы
    title_to_question = {
        "что такое": "что такое",
        "суть": "что это такое",
        "коротко": "расскажите подробнее",
        "подробно": "расскажите подробнее",
        "кому подходит": "кому это подходит",
        "когда не подходит": "когда это не подходит",
        "что важно знать": "что важно знать",
        "ограничения": "какие ограничения",
        "нюансы": "какие нюансы",
        "преимущества": "какие преимущества",
        "недостатки": "какие недостатки",
        "стоимость": "сколько это стоит",
        "цены": "сколько это стоит",
        "сроки": "сколько времени занимает",
        "длительность": "сколько времени занимает",
        "безопасность": "насколько это безопасно",
        "риски": "какие риски",
        "гарантии": "какие гарантии",
        "результат": "какой результат",
        "эффективность": "насколько эффективно"
    }
    
    title_lower = title.lower()
    
    # Ищем подходящий вопрос
    for key, question in title_to_question.items():
        if key in title_lower:
            # Добавляем контекст из темы
            topic = frontmatter.get("topic", "")
            if topic:
                return f"{question} {topic}"
            return question
    
    # Если не нашли подходящий маппинг, создаем общий вопрос
    if len(title) > 3:
        return f"расскажите про {title.lower()}"
    
    return None


def slugify_title(title: str) -> str:
    """
    Создает slug из заголовка.
    
    Args:
        title: Заголовок
    
    Returns:
        Slug для h2_id
    """
    # Убираем знаки препинания и приводим к нижнему регистру
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    # Заменяем пробелы на дефисы
    slug = re.sub(r'[-\s]+', '-', slug)
    # Убираем дефисы в начале и конце
    return slug.strip('-')


def enhance_followups_with_context(
    followups: List[Dict[str, Any]], 
    user_query: str,
    frontmatter: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Улучшает followups с учетом контекста запроса.
    
    Args:
        followups: Список followups
        user_query: Запрос пользователя
        frontmatter: Метаданные frontmatter
    
    Returns:
        Улучшенный список followups
    """
    if not followups:
        return followups
    
    # Фильтруем followups по релевантности к запросу
    relevant_followups = []
    query_lower = user_query.lower()
    
    for followup in followups:
        label = followup.get("label", "").lower()
        query_text = followup.get("query", "").lower()
        
        # Проверяем релевантность
        if (any(word in label for word in query_lower.split()) or 
            any(word in query_text for word in query_lower.split())):
            relevant_followups.append(followup)
    
    # Если нет релевантных, возвращаем все
    if not relevant_followups:
        return followups[:3]
    
    return relevant_followups[:3]


def get_smart_followups(
    content: str,
    frontmatter: Dict[str, Any],
    user_query: str,
    max_followups: int = 3
) -> List[Dict[str, Any]]:
    """
    Получает умные followups с учетом контекста.
    
    Args:
        content: Содержимое MD файла
        frontmatter: Метаданные frontmatter
        user_query: Запрос пользователя
        max_followups: Максимальное количество followups
    
    Returns:
        Список умных followups
    """
    # Генерируем базовые followups
    base_followups = generate_followups_from_content(content, frontmatter, max_followups)
    
    # Улучшаем с учетом контекста
    enhanced_followups = enhance_followups_with_context(base_followups, user_query, frontmatter)
    
    return enhanced_followups

