# 🤖 Подключение LLM для классификации эмоций

Этот документ описывает, как включить «умный режим» анализа эмоций в вопросах пациента. Задача — автоматически определять категорию (`pain`, `price`, `trust` и т.д.) не только по словарям, но и через LLM.

---

## 1. Правила приоритета
1. Если `verbatim:true` → эмпатия **OFF**.  
2. Если `emotion` в md явно указана (например, `"pain"`) → это **override**.  
3. Если `emotion:none` → эмпатия OFF.  
4. Если ничего не указано → используем `mode` из `/config/empathy_config.yaml`.

---

## 2. Конфиг
Файл: `/config/empathy_config.yaml`

```yaml
mode: "hybrid"               # "llm" | "triggers" | "hybrid"
min_confidence: 0.55         # порог уверенности LLM
max_phrases: 2               # opener+closer не более 2
skip_opener_probability: 0.2 # 20% ответов без opener
block_doc_types: ["prices","warranty","contacts","contraindications"]
```

---

## 3. Промпт для классификатора
LLM вызывается до генерации ответа. Она возвращает JSON с категорией и уверенностью.

```text
SYSTEM:
Ты коротко классифицируешь эмоцию/сомнение в вопросе пациента о стоматологии.
Верни ТОЛЬКО JSON: {"category":"...", "confidence":0..1}.

USER:
Вопрос: <<{user_query}>>
Контекст: <<{retrieved_snippet}>>

Категории:
- pain: страх боли, анестезия
- price: деньги/стоимость/дорого
- trust: доверие врачу/клинике
- consultation: первый визит
- osseointegration: «не приживётся», отторжение
- safety: стерильность, инфекции
- none: эмоции нет
```

### Пример ответа
```json
{"category":"osseointegration","confidence":0.78}
```

---

## 4. Алгоритм `detect_emotion`
```python
def detect_emotion(user_query, retrieved_snippet, fm_emotion, fm_verbatim, doc_type, cfg):
    if fm_verbatim or doc_type in cfg["block_doc_types"]:
        return ("none", "blocked", 1.0)

    if fm_emotion and fm_emotion != "none":
        return (fm_emotion, "frontmatter", 1.0)

    mode = cfg["mode"]

    trig_cat, trig_hit = triggers_match(user_query)

    if mode == "triggers":
        return (trig_cat if trig_hit else "none", "triggers", 1.0 if trig_hit else 0.0)

    if mode == "llm":
        llm_cat, llm_conf = llm_classify(user_query, retrieved_snippet)
        return (llm_cat if llm_conf >= cfg["min_confidence"] else "none", "llm", llm_conf)

    if mode == "hybrid":
        if trig_hit:
            return (trig_cat, "triggers", 1.0)
        llm_cat, llm_conf = llm_classify(user_query, retrieved_snippet)
        if llm_conf >= cfg["min_confidence"]:
            return (llm_cat, "llm", llm_conf)
        return ("none", "none", llm_conf)
```

---

## 5. Сбор ответа
```python
def build_answer(core_text, emotion, empathy_bank, cfg, rng):
    if emotion == "none":
        return core_text

    openers = empathy_bank[emotion].get("openers", [])
    closers = empathy_bank[emotion].get("closers", [])

    parts = []
    if rng.random() >= cfg["skip_opener_probability"] and openers:
        parts.append(rng.choice(openers))

    parts.append(core_text)

    if len(parts) < cfg["max_phrases"] + 1 and closers:
        parts.append(rng.choice(closers))

    return "\n\n".join(parts)
```
> `empathy_bank` — это распарсенный `/config/empathy.yaml`.

---

## 6. Интеграция в пайплайн
1. Ретрив → `retrieved_snippet`.  
2. Определяем `emotion` через `detect_emotion`.  
3. Составляем `core_text`.  
4. Применяем `build_answer`.  
5. Добавляем CTA.  

---

## 7. Журналирование
Логируем результат классификации:

```json
{
  "q": "Будет ли больно при имплантации?",
  "emotion": "pain",
  "detector": "llm",
  "confidence": 0.83,
  "opener_used": true,
  "closer_used": false
}
```

---

## 8. Кэширование
- LRU-кэш 500–1000 ключей.  
- Ключ: нормализованный текст вопроса.  
- TTL: 24 часа.  

---

## 9. Тесты
- Минимум 5–10 примеров на каждую категорию.  
- 20 примеров на `none`.  
- Проверить, что `block_doc_types` и `verbatim:true` отключают эмпатию.  

---

## 10. Где разместить
- `/config/empathy.yaml` — фразы  
- `/config/empathy_triggers.yaml` — словари  
- `/config/empathy_config.yaml` — настройки  
- `/core/empathy.py` — код функций (`detect_emotion`, `build_answer`, `llm_classify`)  
