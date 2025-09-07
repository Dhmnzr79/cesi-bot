# core/router.py
"""
Модуль тематического роутинга.
Определяет темы запросов и применяет весовые коэффициенты.
"""

import json
import re
from pathlib import Path
from typing import Set, Dict, Any, List
from .normalize import normalize_ru


class ThemeRouter:
    """Класс для тематического роутинга запросов."""
    
    def __init__(self, themes_path: str = "config/themes.json"):
        self.themes_path = Path(themes_path)
        self.theme_map = {}
        self._load_themes()
    
    def _load_themes(self):
        """Загружает темы из конфигурационного файла."""
        try:
            if self.themes_path.exists():
                with open(self.themes_path, "r", encoding="utf-8") as f:
                    self.theme_map = json.load(f)
                
                # Компилируем regex для каждой темы
                for theme, config in self.theme_map.items():
                    if "query_regex" in config:
                        try:
                            config["query_regex_compiled"] = re.compile(
                                config["query_regex"], 
                                re.IGNORECASE
                            )
                        except re.error:
                            print(f"⚠️ Ошибка компиляции regex для темы {theme}")
                            config["query_regex_compiled"] = None
                    
                    # Нормализуем tag_aliases
                    if "tag_aliases" in config:
                        config["tag_aliases"] = [
                            normalize_ru(str(tag)) for tag in config["tag_aliases"]
                        ]
                    
                    # Устанавливаем вес по умолчанию
                    config["weight"] = float(config.get("weight", 0.2))
                
                print(f"✅ Загружено {len(self.theme_map)} тем из {self.themes_path}")
            else:
                print(f"⚠️ Файл тем не найден: {self.themes_path}")
                self.theme_map = {}
                
        except Exception as e:
            print(f"❌ Ошибка загрузки тем: {e}")
            self.theme_map = {}
    
    def detect_themes(self, query: str) -> Set[str]:
        """
        Определяет темы запроса.
        
        Args:
            query: Запрос пользователя
        
        Returns:
            Множество найденных тем
        """
        if not query or not self.theme_map:
            return set()
        
        detected_themes = set()
        normalized_query = normalize_ru(query)
        
        for theme, config in self.theme_map.items():
            # Проверяем regex
            if "query_regex_compiled" in config and config["query_regex_compiled"]:
                if config["query_regex_compiled"].search(query):
                    detected_themes.add(theme)
                    continue
            
            # Проверяем tag_aliases
            if "tag_aliases" in config:
                for alias in config["tag_aliases"]:
                    if alias in normalized_query:
                        detected_themes.add(theme)
                        break
        
        return detected_themes
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Классифицирует запрос и возвращает детальную информацию о теме.
        
        Args:
            query: Запрос пользователя (уже нормализованный)
        
        Returns:
            Словарь с информацией о теме
        """
        detected_themes = self.detect_themes(query)
        
        if not detected_themes:
            return {"theme": None, "weight": 0.0, "confidence": 0.0}
        
        # Берем первую найденную тему
        primary_theme = list(detected_themes)[0]
        weight = self.get_theme_weight(primary_theme)
        
        return {
            "theme": primary_theme,
            "weight": weight,
            "confidence": 1.0 if len(detected_themes) == 1 else 0.8,
            "all_themes": list(detected_themes)
        }
    
    def get_theme_weight(self, theme: str) -> float:
        """
        Получает вес темы.
        
        Args:
            theme: Название темы
        
        Returns:
            Вес темы (0.0-1.0)
        """
        if theme in self.theme_map:
            return self.theme_map[theme].get("weight", 0.2)
        return 0.0
    
    def apply_theme_boost(self, score: float, theme: str) -> float:
        """
        Применяет буст темы к скорру.
        
        Args:
            score: Исходный скорр
            theme: Тема для буста
        
        Returns:
            Скорр с примененным бустом
        """
        weight = self.get_theme_weight(theme)
        return score + weight
    
    def get_theme_config(self, theme: str) -> Dict[str, Any]:
        """
        Получает конфигурацию темы.
        
        Args:
            theme: Название темы
        
        Returns:
            Конфигурация темы
        """
        return self.theme_map.get(theme, {})


# Глобальный экземпляр роутера
theme_router = ThemeRouter()


def route_topics(query: str) -> Set[str]:
    """
    Определяет темы запроса (совместимость с существующим кодом).
    
    Args:
        query: Запрос пользователя
    
    Returns:
        Множество найденных тем
    """
    return theme_router.detect_themes(query)


def apply_theme_boost_to_candidates(
    candidates: List[tuple], 
    detected_themes: Set[str]
) -> List[tuple]:
    """
    Применяет тематический буст к кандидатам.
    
    Args:
        candidates: Список (chunk, score) кортежей
        detected_themes: Найденные темы
    
    Returns:
        Список кандидатов с примененным бустом
    """
    if not detected_themes or not candidates:
        return candidates
    
    boosted_candidates = []
    
    for chunk, score in candidates:
        final_score = score
        
        # Применяем буст если тема чанка совпадает с найденной
        if hasattr(chunk, 'metadata') and chunk.metadata:
            chunk_topic = getattr(chunk.metadata, 'topic', '')
            if chunk_topic in detected_themes:
                final_score = theme_router.apply_theme_boost(score, chunk_topic)
        
        boosted_candidates.append((chunk, final_score))
    
    # Сортируем по новому скорру
    boosted_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return boosted_candidates


# ШАГ 5: Тематический роутер «doctors» (чтобы theme_hint ≠ null)
def build_doctor_name_regex():
    """Сгенерируй один общий regex из всех фамилий (aliases) и общих слов"""
    names = []
    
    # Собираем все алиасы из документов типа "doctors"
    try:
        from .md_loader import ALIAS_MAP
        for alias, data in ALIAS_MAP.items():
            # Проверяем, что это алиас врача (можно добавить проверку по doc_type)
            if len(alias) > 2:  # фильтруем короткие алиасы
                names.append(alias)
    except:
        pass
    
    # Убираем дубликаты и фильтруем
    names = list(set(names))
    names = [n for n in names if len(n) > 2]
    
    # Строим паттерн: общие слова + алиасы
    common_words = r"(?:врач|доктор|имплантолог)\b"
    aliases_part = "|".join(re.escape(name) for name in names)
    
    if aliases_part:
        pattern = f"{common_words}|(?:{aliases_part})"
    else:
        pattern = common_words
    
    return pattern


# Компилируем regex для поиска врачей
DOCTORS_QUERY_REGEX = re.compile(build_doctor_name_regex(), re.IGNORECASE)


def route_theme(user_q: str) -> str | None:
    """
    Определяет тему запроса для theme_hint.
    
    Args:
        user_q: Запрос пользователя
    
    Returns:
        Название темы или None
    """
    nq = normalize_ru(user_q)
    
    # Проверяем тему "doctors"
    if DOCTORS_QUERY_REGEX.search(nq):
        return "doctors"
    
    # ...другие темы...
    
    return None
