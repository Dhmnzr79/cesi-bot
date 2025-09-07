# core/followups.py
"""
Модуль для работы с followups (связанными вопросами).
Извлечение и обработка followups из frontmatter.
"""

from typing import Dict, List, Any
from .md_loader import extract_followups_from_frontmatter


def followups_from_frontmatter(frontmatter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Извлекает followups из frontmatter в формате для API.
    
    Args:
        frontmatter: Словарь с метаданными frontmatter
    
    Returns:
        Список followups в формате API
    """
    return extract_followups_from_frontmatter(frontmatter)


def format_followups_for_api(followups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Форматирует followups для API ответа.
    
    Args:
        followups: Список followups
    
    Returns:
        Отформатированный список followups
    """
    formatted = []
    
    for followup in followups:
        if isinstance(followup, dict):
            formatted_followup = {
                "label": followup.get("label", ""),
                "query": followup.get("query", ""),
                "jump": followup.get("jump", {})
            }
            formatted.append(formatted_followup)
    
    return formatted


def validate_followup(followup: Dict[str, Any]) -> bool:
    """
    Проверяет валидность followup.
    
    Args:
        followup: Followup для проверки
    
    Returns:
        True если followup валиден
    """
    if not isinstance(followup, dict):
        return False
    
    # Проверяем обязательные поля
    required_fields = ["label", "query"]
    for field in required_fields:
        if not followup.get(field):
            return False
    
    return True


def filter_valid_followups(followups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Фильтрует только валидные followups.
    
    Args:
        followups: Список followups для фильтрации
    
    Returns:
        Список только валидных followups
    """
    return [f for f in followups if validate_followup(f)]


def filter_followups(followups: List[Dict[str, Any]], current_h2_id: str = None, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Фильтрует followups, убирая дубли и ссылки на текущий раздел.
    
    Args:
        followups: Список followups
        current_h2_id: ID текущего H2 раздела
        limit: Максимальное количество followups
    
    Returns:
        Отфильтрованный список followups
    """
    if not followups:
        return []
    
    seen = set()
    out = []
    
    for f in followups:
        h2 = (f.get("jump") or {}).get("h2_id")
        
        # не показываем ссылку на текущий раздел и убираем дубли
        key = (f.get("label"), h2 or f.get("query"))
        
        if h2 and current_h2_id and h2 == current_h2_id:
            continue
            
        if key in seen:
            continue
            
        seen.add(key)
        out.append(f)
        
        if len(out) >= limit:
            break
    
    return out
