# core/md_loader.py
"""
Модуль для работы с Markdown файлами и frontmatter.
Парсинг и извлечение метаданных из MD файлов.
"""

import re
import yaml
from typing import Dict, Tuple, List, Any, Optional
from core.normalize import normalize_ru


# Регулярное выражение для поиска frontmatter
FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.S)


def parse_frontmatter(md_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Парсит YAML frontmatter из Markdown файла.
    
    Args:
        md_text: Содержимое Markdown файла
    
    Returns:
        Tuple[frontmatter_dict, content_without_frontmatter]
    """
    match = FRONTMATTER_RE.match(md_text)
    
    if match:
        yaml_content = match.group(1)
        try:
            frontmatter = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            frontmatter = {}
        
        # Возвращаем frontmatter и контент без него
        content = md_text[match.end():]
        return frontmatter, content
    
    # Если frontmatter не найден
    return {}, md_text


def extract_followups_from_frontmatter(frontmatter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Извлекает followups из frontmatter.
    
    Args:
        frontmatter: Словарь с метаданными frontmatter
    
    Returns:
        Список followups в формате для API
    """
    followups = []
    
    # Получаем followups из frontmatter (максимум 3)
    frontmatter_followups = frontmatter.get("followups", [])[:3]
    
    for item in frontmatter_followups:
        if isinstance(item, dict):
            followup = {
                "label": item.get("label", ""),
                "query": item.get("query", ""),
                "jump": {
                    "topic": frontmatter.get("topic", ""),
                    "h2_id": item.get("h2_id", "")
                }
            }
            followups.append(followup)
    
    return followups


def extract_cta_from_frontmatter(frontmatter: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекает CTA информацию из frontmatter.
    
    Args:
        frontmatter: Словарь с метаданными frontmatter
    
    Returns:
        Tuple[cta_text, cta_link]
    """
    cta_text = frontmatter.get("cta_text")
    cta_link = frontmatter.get("cta_link")
    
    return cta_text, cta_link


def extract_meta_from_frontmatter(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает основные метаданные из frontmatter.
    
    Args:
        frontmatter: Словарь с метаданными frontmatter
    
    Returns:
        Словарь с метаданными для API
    """
    meta = {
        "doc_type": frontmatter.get("doc_type", "info"),
        "topic": frontmatter.get("topic", ""),
        "title": frontmatter.get("title", ""),
        "updated": frontmatter.get("updated", ""),
        "tone": frontmatter.get("tone", "friendly"),
        "emotion": frontmatter.get("emotion", ""),
        "criticality": frontmatter.get("criticality", "medium"),
        "verbatim": frontmatter.get("verbatim", False)
    }
    
    return meta


def build_index_text(frontmatter: dict, body_md: str, h2_texts: list[str]) -> str:
    """
    Собирает текст для индексации, включая заголовок и алиасы.
    
    Args:
        frontmatter: Метаданные из YAML
        body_md: Тело Markdown файла
        h2_texts: Список H2 заголовков
    
    Returns:
        Нормализованный текст для индексации
    """
    title = (frontmatter.get("title") or "").strip()
    aliases = frontmatter.get("aliases") or []
    parts = [title] + aliases + h2_texts + [body_md]
    raw = "\n".join([p for p in parts if p])
    return normalize_ru(raw)  # нормализуем ДО индексации


# Глобальная карта алиасов для fallback
ALIAS_MAP = {}


def register_aliases(frontmatter: dict, file_path: str):
    """
    Регистрирует алиасы в глобальной карте для fallback поиска.
    
    Args:
        frontmatter: Метаданные из YAML
        file_path: Путь к файлу
    """
    primary_h2_id = frontmatter.get("primary_h2_id")
    aliases = frontmatter.get("aliases") or []
    
    for alias in aliases:
        normalized_alias = normalize_ru(alias)
        ALIAS_MAP[normalized_alias] = {
            "file_path": file_path,
            "primary_h2_id": primary_h2_id
        }


def alias_fallback(query: str):
    """
    Fallback поиск по алиасам, если обычный поиск не дал результатов.
    
    Args:
        query: Поисковый запрос
    
    Returns:
        Список кандидатов или пустой список
    """
    normalized_query = normalize_ru(query)
    hit = ALIAS_MAP.get(normalized_query)
    
    if not hit:
        return []
    
    return [{
        "file": hit["file_path"],
        "h2_id": hit["primary_h2_id"],  # может быть None, тогда возьмём первый H2
        "source": "alias",
        "scores": {"bm25": 1.0, "dense": 1.0, "rerank": 1.0},
        "score": 1.0
    }]


def pick_best_chunk(candidates: list[dict], frontmatter: dict, jump: dict | None):
    """
    Выбирает лучший чанк для ответа на основе jump данных и frontmatter.
    
    Args:
        candidates: Список кандидатов
        frontmatter: Метаданные документа
        jump: Данные о переходе (может быть None)
    
    Returns:
        Лучший кандидат или None
    """
    # клик по followup (click on followup)
    if jump and jump.get("h2_id"):
        tid = jump["h2_id"]
        for c in candidates:
            if (c.get("h2_id") or c.get("h2")) == tid:
                return c

    # основной раздел из шапки (main section from frontmatter)
    primary = (frontmatter or {}).get("primary_h2_id")
    if primary:
        for c in candidates:
            if (c.get("h2_id") or c.get("h2")) == primary:
                return c

    # запасной — первый кандидат (fallback — first candidate)
    return candidates[0] if candidates else None
