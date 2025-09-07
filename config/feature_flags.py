# config/feature_flags.py
"""
Конфигурация feature flags для новой системы ответов.
Позволяет включать/выключать функции без изменения кода.
"""

import os
from typing import Dict, Any


class FeatureFlags:
    """Класс для управления feature flags."""
    
    def __init__(self):
        self.flags = self._load_flags()
    
    def _load_flags(self) -> Dict[str, Any]:
        """Загружает флаги из переменных окружения."""
        return {
            # Режим ответа: json или legacy
            "RESPONSE_MODE": os.getenv("RESPONSE_MODE", "legacy"),
            
            # Включение новых функций
            "ENABLE_FOLLOWUPS": os.getenv("ENABLE_FOLLOWUPS", "false").lower() == "true",
            "ENABLE_EMPATHY": os.getenv("ENABLE_EMPATHY", "false").lower() == "true",
            "ENABLE_GUARD": os.getenv("ENABLE_GUARD", "false").lower() == "true",
            
            # Post-processing
            "POSTPROCESS_STRIP_TEXTUAL_CTA": os.getenv("POSTPROCESS_STRIP_TEXTUAL_CTA", "false").lower() == "true",
            
            # Пороги и настройки
            "GUARD_THRESHOLD": float(os.getenv("GUARD_THRESHOLD", "0.2")),
            
            # Нормализация
            "ENABLE_NORMALIZATION": os.getenv("ENABLE_NORMALIZATION", "false").lower() == "true",
            
            # UX улучшения
            "ENABLE_SMART_FOLLOWUPS": os.getenv("ENABLE_SMART_FOLLOWUPS", "false").lower() == "true",
            "ENABLE_ENHANCED_EMPATHY": os.getenv("ENABLE_ENHANCED_EMPATHY", "false").lower() == "true",
            "SUPPRESS_EMPATHY_CLOSER_WHEN_CTA": os.getenv("SUPPRESS_EMPATHY_CLOSER_WHEN_CTA", "true").lower() == "true",
        }
    
    def get(self, flag_name: str, default: Any = None) -> Any:
        """Получает значение флага."""
        return self.flags.get(flag_name, default)
    
    def is_enabled(self, flag_name: str) -> bool:
        """Проверяет, включен ли флаг."""
        return self.flags.get(flag_name, False)
    
    def is_json_mode(self) -> bool:
        """Проверяет, включен ли JSON режим."""
        return self.flags.get("RESPONSE_MODE") == "json"
    
    def is_legacy_mode(self) -> bool:
        """Проверяет, включен ли legacy режим."""
        return self.flags.get("RESPONSE_MODE") == "legacy"


# Глобальный экземпляр
feature_flags = FeatureFlags()
