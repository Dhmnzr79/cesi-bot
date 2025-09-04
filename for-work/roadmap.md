# 🗺️ Дорожная карта доработок бота (RAG + эмпатия + CTA)

> Файл для разработчика в Cursor. Цель — довести MVP до стабильного продакшена с управляемой эмпатией, аккуратными CTA и точным ретривом.

## 0) Текущий статус (сент. 2025)
- ✅ RAG-пайплайн есть: BM25 + эмбеддинги, базовый реранк (эвристика), разбиение по `##/###`.
- ✅ Логирование запросов в JSONL.
- ✅ Базовый CTA (тайм-троттлинг) подключён.
- ✅ Валидация телефона, базовая анти-спам логика.
- ⛳ Эмпатия: **нет** загрузки YAML (`/config/empathy.yaml`), **нет** триггеров и LLM-классификатора.
- ⛳ CTA: **нет** конфиг-файла с действиями/триггерами и правилом `min_questions_gap`.
- ⛳ Реранк LLM: отсутствует (только эвристика).
- ⛳ Multi-query rewrite: отсутствует.
- ⛳ Answer compression: отсутствует.
- ⛳ Кэш эмбеддингов: отсутствует.
- ⛳ Виджет-конфиг/метрика: отсутствуют.

---

## 1) План по этапам

### Неделя 1 — Стабильность и управление ответом
- [ ] **Эмпатия из YAML**  
  - Подключить `/config/empathy.yaml`, `/config/empathy_triggers.yaml`, `/config/empathy_config.yaml`.
  - Реализовать `detect_emotion()` с режимами: `"triggers" | "llm" | "hybrid"`.
  - Реализовать `build_answer()` (opener/closer + core_text).  
  - **Блок-лист:** выключить эмпатию на doc_type: `prices, warranty, contacts, contraindications` и при `verbatim:true`.

- [ ] **CTA из конфига**  
  - Завести `/config/cta.yaml`: `actions`, `triggers`, `min_questions_gap`, `time_throttle_sec`.
  - Реализовать `pick_cta()` с приоритетами и троттлингом.

- [ ] **Логи расширить**  
  - Добавить в запись: `emotion`, `detector`, `confidence`, `opener_used`, `closer_used`, `cta_action`, `cta_reason`.

**Definition of Done (Неделя 1):**
- Эмпатия работает по категориям `pain|price|trust|consultation|osseointegration|safety`.
- При `verbatim:true` и на блок-лист тем эмпатия не вставляется.
- CTA управляется из `/config/cta.yaml`, не дёргается чаще правил.
- Логи содержат все поля, удобно анализировать.

---

### Неделя 2 — Точность и вариативность
- [ ] **LLM-реранкер (минимум)**  
  - Для top-6 кандидатов → mini-prompt «релевантность 0..1» → выбрать top-1/2.
- [ ] **Multi-query rewrite**  
  - Генерировать 2–3 перефраза вопроса; объединять кандидатов и реранкуем.
- [ ] **Answer compression**  
  - Если ответ > N символов → мини-промпт на сжатие до 2–3 предложений **без потери чисел**.
- [ ] **Эмпатия: вариативность**  
  - В YAML по 5–7 фраз на тему; `skip_opener_probability` = 0.2 по конфигу.

**Definition of Done (Неделя 2):**
- NDCG@3 по ручному тест-листу вырос ≥10% против Недели 1.
- Доля «слишком длинных ответов» < 10%.
- Повторяемость одинаковых opener/closer в 5 подряд ответах < 15%.

---

### Неделя 3 — Производительность и маркетинг
- [ ] **Кэш эмбеддингов** (SQLite/JSON): хэш текста → вектор; при старте не пересчитывать.
- [ ] **Антиспам/лиды**: лимиты по IP/UA; бэкап лидов в Google Sheets/локальный JSON.
- [ ] **Виджет-конфиг и метрика**: вынести тексты и CTA в `widget-config.json`; добавить Web/Я.Метрику на клики CTA.
- [ ] **Ручной eval**: 50 вопросов, недельный регресс-отчёт (accuracy/clarity/empathy/CTA CTR).

**Definition of Done (Неделя 3):**
- Среднее время ответа −15% за счёт кэша.
- Лиды не теряются (есть бэкап), спам снижен.
- Метрика кликов CTA собирается.

---

## 2) Файлы/папки, которые нужно добавить

```
/config/
  empathy.yaml                 # библиотека фраз (opener/closer) — готово в docs
  empathy_triggers.yaml        # словари-триггеры — готово в docs
  empathy_config.yaml          # режимы/пороги/skip_opener_probability
  cta.yaml                     # действия/триггеры/троттлинг

/core/
  empathy.py                   # detect_emotion, build_answer, llm_classify, triggers_match
  cta.py                       # pick_cta

/docs/
  empathy.md                   # инструкция по эмпатии
  empathy_llm.md               # инструкция по LLM-классификатору
```

**Шаблон `/config/empathy_config.yaml`:**
```yaml
mode: "hybrid"               # "llm" | "triggers" | "hybrid"
min_confidence: 0.55
max_phrases: 2
skip_opener_probability: 0.2
block_doc_types: ["prices","warranty","contacts","contraindications"]
```

**Шаблон `/config/cta.yaml`:**
```yaml
time_throttle_sec: 90
min_questions_gap: 1

actions:
  consultation:
    text: "Записаться на консультацию"

  consultation_doctor:
    text: "Познакомиться с врачом на консультации"

triggers:
  # по словам/эмоциям/топикам
  - when_emotion: ["pain","osseointegration","trust","consultation","price"]
    action: "consultation"
```

---

## 3) Точки в коде (куда вставлять)

### `rag_engine.py`
- [ ] После ретрива: собрать `retrieved_snippet` для LLM-реранка и классификатора эмоций.
- [ ] Вставить `detect_emotion(...)` (читает frontmatter, конфиги, триггеры/LLM).
- [ ] Синтезировать `core_text` (перефраз, либо экстракт при `verbatim:true`).
- [ ] Применить `build_answer(core_text, emotion, empathy_bank, cfg, rng)`.
- [ ] Компрессор ответа (если включён).
- [ ] Вернуть структуру `{text, emotion, detector, confidence, used_chunks}`.

### `app.py`
- [ ] Получить `emotion`/`detector`/`confidence` → добавить в лог.
- [ ] Вызвать `pick_cta(session_state, last_cta_ts, emotion, doc_type, question_count)`.
- [ ] Применить троттлинг CTA (time + questions gap).
- [ ] Отдать финальный ответ + CTA.

---

## 4) Логи (JSONL)
Каждая запись:
```json
{
  "ts": "2025-09-04T12:34:56Z",
  "q": "...",
  "answer_len": 523,
  "doc_ids": ["implants-pain","consultation-overview"],
  "doc_type": "faq",
  "verbatim": false,
  "emotion": "pain",
  "detector": "llm",
  "confidence": 0.81,
  "opener_used": true,
  "closer_used": true,
  "cta_action": "consultation",
  "cta_reason": "emotion=pain;gap_ok;throttle_ok"
}
```

---

## 5) Тест-план (минимум)
- Набор из 60 вопросов: по 8 на каждую эмоцию + 12 на `none`.
- Метрики:
  - Точность классификации эмоции ≥ 0.75 (manual labels).
  - Релевантность ответа (оценка 1–5) среднее ≥ 4.0.
  - CTR CTA ≥ 12% на вопросах с эмоцией `price|consultation|trust`.
  - Жалобы на «слишком длинно» < 10%.

---

## 6) Риски/заметки
- Слишком агрессивная эмпатия → ощущение «скрипта». Смягчаем `skip_opener_probability` и лимит `max_phrases`.
- Неверные цены/условия → всегда `verbatim:true` и без эмпатии; числа — только из MD.
- Реранк LLM → ограничить токены (короткий сниппет 400–600 симв.).
- Флоу CTA → не показывать при `contraindications`, `warranty`, `contacts`.

---

## 7) Definition of Ready (к запуску этапов)
- Проставлены `emotion`/`verbatim` в ключевых MD (цены/гарантии/контакты — `none + verbatim:true`).
- В репозитории лежат: `empathy.yaml`, `empathy_triggers.yaml`, `empathy_config.yaml`, `cta.yaml`.
- Созданы заглушки модулей `/core/empathy.py`, `/core/cta.py`.
- Логи включены и проверены локально.

---

## 8) Быстрый чек-лист перед релизом
- [ ] Конфиги загружаются без ошибок, YAML валиден.
- [ ] Эмпатия не активируется в `prices/warranty/contacts/contraindications`.
- [ ] CTA не «дребезжит» (таймаут и gap работают).
- [ ] Ответы длиннее 1200 симв. корректно сжимаются.
- [ ] Логи содержат `emotion/detector/confidence/cta_reason`.
- [ ] Ручной тест-лист проходит с целевыми метриками.
