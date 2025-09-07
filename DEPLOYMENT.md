# 🚀 Инструкция по развертыванию

## 📋 Что нужно сделать на хостинге:

### 1. **Скопировать файлы**
```bash
# Все новые модули уже в проекте:
core/
├── answer_builder.py      # JSON-API контракт
├── normalize.py           # Нормализация запросов
├── guard.py              # Guard система
├── router.py             # Тематический роутинг
├── rag_integration.py    # Интеграция с RAG
├── md_loader.py          # Парсинг MD файлов
├── followups.py          # Followups из frontmatter
├── followups_enhanced.py # Умные followups
├── empathy_enhanced.py   # Улучшенная эмпатия
├── postprocessing.py     # Post-processing
├── legacy_adapter.py     # Адаптер совместимости
└── logger.py            # Структурированное логирование

config/
├── feature_flags.py      # Feature flags система
└── themes.json          # Тематический роутинг
```

### 2. **Создать .env файл**
```bash
# Скопировать config/production.env в .env
cp config/production.env .env
```

### 3. **Настройки .env для разных сценариев**

#### **Для нового фронтенда (JSON API):**
```ini
RESPONSE_MODE=json
ENABLE_GUARD=true
ENABLE_NORMALIZATION=true
ENABLE_FOLLOWUPS=true
ENABLE_EMPATHY=true
POSTPROCESS_STRIP_TEXTUAL_CTA=true
ENABLE_SMART_FOLLOWUPS=true
ENABLE_ENHANCED_EMPATHY=true
SUPPRESS_EMPATHY_CLOSER_WHEN_CTA=true
GUARD_THRESHOLD=0.35
```

#### **Для старого фронтенда (Legacy):**
```ini
RESPONSE_MODE=legacy
ENABLE_GUARD=true
ENABLE_NORMALIZATION=true
ENABLE_FOLLOWUPS=true
ENABLE_EMPATHY=true
POSTPROCESS_STRIP_TEXTUAL_CTA=true
ENABLE_SMART_FOLLOWUPS=true
ENABLE_ENHANCED_EMPATHY=true
SUPPRESS_EMPATHY_CLOSER_WHEN_CTA=true
GUARD_THRESHOLD=0.35
```

#### **Для постепенного внедрения (только качество):**
```ini
RESPONSE_MODE=legacy
ENABLE_GUARD=true
ENABLE_NORMALIZATION=true
ENABLE_FOLLOWUPS=false
ENABLE_EMPATHY=false
POSTPROCESS_STRIP_TEXTUAL_CTA=false
ENABLE_SMART_FOLLOWUPS=false
ENABLE_ENHANCED_EMPATHY=false
SUPPRESS_EMPATHY_CLOSER_WHEN_CTA=true
GUARD_THRESHOLD=0.35
```

### 4. **Перезапустить приложение**
```bash
# Для systemd
sudo systemctl restart your-app-name

# Для PM2
pm2 restart your-app-name

# Для Docker
docker-compose restart
```

## 📊 Мониторинг после развертывания:

### **Проверить логи:**
```bash
# Структурированные логи
tail -f logs/bot_responses.jsonl

# Старые логи
tail -f logs/queries.jsonl
```

### **Ключевые метрики:**
- `low_relevance` - должно быть 5-15%
- `followups_count` - должно быть 2-3 для большинства ответов
- `opener_used` / `closer_used` - эмпатия работает
- `shown_cta` - CTA показываются

### **Настройка порога Guard:**
- Если `low_relevance` < 5% - понизить `GUARD_THRESHOLD` до 0.30
- Если `low_relevance` > 15% - повысить `GUARD_THRESHOLD` до 0.40

## 🔧 Troubleshooting:

### **Проблема: "Module not found"**
```bash
# Проверить, что все файлы скопированы
ls -la core/
ls -la config/
```

### **Проблема: "Feature flag not working"**
```bash
# Проверить .env файл
cat .env | grep ENABLE_
```

### **Проблема: "Followups не показываются"**
```bash
# Проверить, что ENABLE_FOLLOWUPS=true
# Проверить, что в MD файлах есть followups в frontmatter
```

## 📈 Постепенное внедрение:

1. **Неделя 1:** Только Guard система (`ENABLE_GUARD=true`)
2. **Неделя 2:** + Нормализация (`ENABLE_NORMALIZATION=true`)
3. **Неделя 3:** + Followups (`ENABLE_FOLLOWUPS=true`)
4. **Неделя 4:** + Эмпатия (`ENABLE_EMPATHY=true`)
5. **Неделя 5:** Переход на JSON API (`RESPONSE_MODE=json`)

## ✅ Чек-лист развертывания:

- [ ] Все файлы скопированы
- [ ] .env файл создан
- [ ] Приложение перезапущено
- [ ] Логи проверены
- [ ] Метрики в норме
- [ ] Функции работают

