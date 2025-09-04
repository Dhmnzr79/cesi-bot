# rag_engine.py
import os
import numpy as np
from openai import OpenAI
import faiss
import yaml
import re
import json
import difflib
import random
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Set, Tuple
from textwrap import dedent
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from core.empathy import detect_emotion, build_answer

# ==== КОНСТАНТЫ ====
CONFIG_DIR = Path("config")
MD_DIR = Path("md")
THEMES_PATH = CONFIG_DIR / "themes.json"

# ==== АНТИ-ВОДА РЕГЕКСЫ ====
ANTI_FLUFF = [
    r"^мы\s+рады\s+сообщить.*",
    r"^с\s+радостью\s+сообщаем.*",
    r"^мы\s+гордимся.*",
    r"^в\s+нашей\s+клинике\s+мы.*",
    r"^это\s+замечательн[оы].*",
    r"постараемся\s+максимально\s+понять.*",
    r"^современн[а-я]+\s+технологи.*",
    r"^открывает\s+возможности.*",
]

# ==== ГЛОБАЛЬНЫЕ СТРУКТУРЫ ДЛЯ КАТАЛОГА СУЩНОСТЕЙ ====
ENTITY_INDEX = {}  # alias_norm -> {"topic": str, "entity": str, "doc_id": str, "section": str}
ENTITY_CHUNKS = {}  # (topic, entity) -> RetrievedChunk

# ==== BM25 ИНДЕКС ====
bm25_index = None
bm25_corpus = []

def _normalize(text: str) -> str:
    # NBSP -> обычный пробел; CRLF -> LF
    return text.replace('\u00A0', ' ').replace('\r\n', '\n').replace('\r', '\n')

def _norm(s: str) -> str:
    """Нормализация строки для индексации"""
    return re.sub(r'\s+', ' ', s.lower().replace('\u00a0', ' ')).strip()

def _slugify_implant_kind(name: str) -> str:
    """Создает slug для вида имплантации"""
    n = _norm(name)
    
    # Специфичные маппинги
    if re.search(r'all\s*[- ]?on\s*[- ]?4', n):
        return "all-on-4"
    if re.search(r'all\s*[- ]?on\s*[- ]?6', n):
        return "all-on-6"
    if n.startswith("одно"):
        return "single-stage"
    if n.startswith("класс"):
        return "classic"
    
    # Общий slug
    return re.sub(r'[^a-z0-9\-]+', '-', n)  # unidecode(n) - опционально

load_dotenv()

# Загрузка конфигов эмпатии
with open(os.path.join("config", "empathy_config.yaml"), "r", encoding="utf-8") as f:
    EMPATHY_CFG = yaml.safe_load(f)
with open(os.path.join("config","empathy.yaml"), "r", encoding="utf-8") as f:
    EMPATHY_BANK = yaml.safe_load(f)
with open(os.path.join("config", "empathy_triggers.yaml"), "r", encoding="utf-8") as f:
    EMPATHY_TRIGGERS = yaml.safe_load(f)
_RNG = random.Random()

# Инициализация OpenAI клиента v1
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
    print("✅ OpenAI клиент v1 успешно инициализирован")
except Exception as e:
    print(f"❌ Ошибка инициализации OpenAI клиента: {e}")
    openai_client = None

# Типы данных
class Frontmatter:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get('id', '')
        self.slug = data.get('slug', '')
        self.title = data.get('title', '')
        self.description = data.get('description', '')
        self.doc_type = data.get('doc_type', 'info')
        self.topic = data.get('topic', '')
        self.tags = data.get('tags', [])
        self.audience = data.get('audience', '')
        self.updated = data.get('updated', '')
        self.locale = data.get('locale', 'ru-RU')
        self.tone = data.get('tone', 'friendly')
        self.emotion = data.get('emotion', '')
        self.criticality = data.get('criticality', 'medium')
        self.verbatim = data.get('verbatim', False)
        self.policy_ref = data.get('policy_ref', [])
        self.source = data.get('source', [])
        self.preferred_format = data.get('preferred_format', ['short', 'bullets', 'cta'])
        self.canonical_url = data.get('canonical_url', '')
        self.noindex = data.get('noindex', False)
        self.cta_action = data.get('cta_action', '')
        self.cta_text = data.get('cta_text', '')
        self.cta_link = data.get('cta_link', '')

class RetrievedChunk:
    def __init__(self, id: str, text: str, metadata: Frontmatter, file_name: str):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.file_name = file_name
        # Безопасное извлечение атрибутов с fallback значениями
        self.updated = metadata.updated if metadata else ""
        self.criticality = metadata.criticality if metadata else "medium"
        self.doc_type = metadata.doc_type if metadata else "info"
        self.tags = metadata.tags if metadata else []

class SynthJSON:
    def __init__(self, short: str, bullets: List[str], cta: str, used_chunks: List[str], tone: str, warnings: Optional[List[str]] = None):
        self.short = short
        self.bullets = bullets
        self.cta = cta
        self.used_chunks = used_chunks
        self.tone = tone
        self.warnings = warnings or []

def parse_yaml_front_matter(text: str):
    """Парсит YAML front matter из markdown файла"""
    pattern = r'^---\s*\n(.*?)\n---\s*\n'
    m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    
    if m:
        yaml_content = m.group(1)
        try:
            metadata = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            metadata = {}
        # КЛЮЧЕВАЯ ПРАВКА: вырезаем только первый front-matter
        content = text[m.end():]
        return Frontmatter(metadata), content.strip()
    else:
        return Frontmatter({}), text

def chunk_text_by_sections(content: str, file_name: str) -> List[RetrievedChunk]:
    """Разбивает текст на чанки по секциям (## и ###)"""
    chunks = []
    
    # Разрезаем Н2 (## ...)
    h2_blocks = re.split(r'(?m)^\s*##\s+', content)
    
    for block in h2_blocks:
        block = block.strip()
        if not block:
            continue
        
        lines = block.splitlines()
        h2_title = lines[0]
        h2_body = '\n'.join(lines[1:])
        
        # Если внутри есть Н3 режем по ним (### ...)
        h3_blocks = re.split(r'(?m)^\s*###\s+', block)
        if len(h3_blocks) > 1:
            # первый элемент до первого ### это сам h2 заголовок, его пропустим
            for h3 in h3_blocks[1:]:
                h3 = h3.strip()
                if not h3:
                    continue
                
                lines = h3.splitlines()
                h3_title = lines[0]
                h3_body = '\n'.join(lines[1:])
                
                text = f"## {h2_title}\n### {h3_title}\n{h3_body}"
                chunk_id = f"{file_name}#{h2_title}_{h3_title}"
                temp_metadata = Frontmatter({})
                chunks.append(RetrievedChunk(chunk_id, text.strip(), temp_metadata, file_name))
        else:
            # Н3 нет - сохраняем Н2 как единый чанк
            text = f"## {h2_title}\n{h2_body}"
            chunk_id = f"{file_name}#{h2_title}"
            temp_metadata = Frontmatter({})
            chunks.append(RetrievedChunk(chunk_id, text.strip(), temp_metadata, file_name))
    
    return chunks

def extract_aliases_from_chunk(chunk_text: str) -> List[str]:
    """Извлекает алиасы из HTML-комментариев в чанке"""
    aliases = []
    
    # Ищем <!-- aliases: [...] -->
    alias_match = re.search(r'<!--\s*aliases:\s*\[(.*?)\]\s*-->', chunk_text)
    if alias_match:
        alias_text = alias_match.group(1)
        aliases = [alias.strip() for alias in alias_text.split(',') if alias.strip()]
    
    return aliases

def update_entity_index(chunk: RetrievedChunk, topic: str, entity_key: str):
    """Обновляет ENTITY_INDEX с алиасами из чанка"""
    # Извлекаем заголовок ##
    header_match = re.search(r'(?m)^##\s+(.+?)\s*$', chunk.text)
    if not header_match:
        return
    
    title = header_match.group(1).strip()
    
    # Извлекаем алиасы
    aliases = extract_aliases_from_chunk(chunk.text)
    aliases.append(title)  # Добавляем сам заголовок
    
    # Индексируем все алиасы
    for alias in aliases:
        alias_norm = _norm(alias)
        if alias_norm:
            ENTITY_INDEX[alias_norm] = {
                "topic": topic,
                "entity": entity_key,
                "doc_id": chunk.file_name,
                "section": title
            }

# Загружаем и разбиваем все markdown-файлы
folder_path = Path("md/")
all_chunks: List[RetrievedChunk] = []

# Глобальные переменные для динамического поиска врачей
DOCTOR_NAME_TOKENS: set[str] = set()
# Прямая адресация врачей: имя -> чанк
DOCTOR_NAME_TO_CHUNK: dict[str, RetrievedChunk] = {}
DOCTOR_REGEX = None
DOCTOR_NAME_REGEX = None  # только имена
DOCTOR_QUERY_REGEX = None  # имена + слова "врач/доктор/..."

# ==== THEMES (усиление по темам) ====
try:
    from pathlib import Path
    import json, re
    
    BASE_DIR = Path(__file__).resolve().parent
    THEMES_PATH = BASE_DIR / "config" / "themes.json"
    
    with open(THEMES_PATH, "r", encoding="utf-8") as f:
        THEME_MAP = json.load(f)
    
    # компиляция регексов и нормализация
    for k, cfg in THEME_MAP.items():
        cfg["query_regex_compiled"] = re.compile(cfg["query_regex"], re.IGNORECASE)
        cfg["tag_aliases"] = [str(t).lower() for t in cfg.get("tag_aliases", [])]
        cfg["weight"] = float(cfg.get("weight", 0.2))
    print(f"OK: themes.json loaded: {len(THEME_MAP)} themes from {THEMES_PATH}")
    for theme, cfg in THEME_MAP.items():
        print(f"  Theme: {theme}: weight={cfg.get('weight', 'N/A')}, regex={cfg.get('query_regex', 'N/A')[:30]}...")
except Exception as e:
    print(f"WARNING: Не удалось загрузить themes.json: {e}")
    THEME_MAP = {}

def _extract_doctor_names_from_text(text: str) -> list[str]:
    """Извлекает имена врачей из текста"""
    names = set()
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # срезать префикс ### / ##
        if line.startswith('### '):
            line = line[4:]
        elif line.startswith('## '):
            line = line[3:]
            
        # Имя Фамилия [Отчество]
        match = re.match(r'^[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}$', line)
        if match:
            full = match.group(0)
            parts = full.split()
            if len(parts) >= 2:
                names.add(full)  # полное ФИО
                names.add(parts[0])  # фамилия
                names.add(f"{parts[0]} {parts[1]}")  # Фамилия Имя
                names.add(f"{parts[1]} {parts[0]}")  # Имя Фамилия
                print(f"    🏥 Извлечено имя: '{full}' → {parts[0]}, {parts[1]} {parts[0]}")
    
    return sorted(list(names))

def _rebuild_doctor_regex():
    """Пересобирает regex для поиска врачей"""
    global DOCTOR_REGEX, DOCTOR_NAME_REGEX, DOCTOR_QUERY_REGEX
    
    base = ['врач', 'доктор', 'хирург', 'имплантолог', 'специалист']
    name_tokens = sorted([re.escape(t) for t in DOCTOR_NAME_TOKENS], key=len, reverse=True)
    base_tokens = [re.escape(t) for t in base]

    if name_tokens:
        DOCTOR_NAME_REGEX = re.compile(r'(' + '|'.join(name_tokens) + r')', re.IGNORECASE)
        DOCTOR_QUERY_REGEX = re.compile(r'(' + '|'.join(name_tokens + base_tokens) + r')', re.IGNORECASE)
    else:
        DOCTOR_NAME_REGEX = None
        DOCTOR_QUERY_REGEX = re.compile(r'(' + '|'.join(base_tokens) + r')', re.IGNORECASE)
    
    # Оставляем старый DOCTOR_REGEX для совместимости
    DOCTOR_REGEX = DOCTOR_QUERY_REGEX

def build_empathy_prompt(tone: str = "friendly", emotion: str = "empathy", allow_emoji: bool = True, cta_text: str | None = None, cta_link: str | None = None) -> str:
    """Собирает короткий промпт для «оживления» ответа без искажения фактов."""
    
    emoji_rule = "Иногда (не чаще одного-двух на весь ответ) добавляй уместные эмодзи 🦷😌📅💬. Если неуместно — не добавляй." if allow_emoji else "Не используй эмодзи."
    
    cta_rule = ""
    if cta_text:
        cta_rule = f"Заверши мягкой фразой-приглашением: «{cta_text}»"
        if cta_link:
            cta_rule += f" (ссылка: {cta_link})."
    
    prompt = dedent(f"""
    Ты - внимательный ассистент стоматологической клиники.
    Тон: {tone}, эмоция: {emotion}. Пиши просто и по-человечески.
    
    Задача:
    1. Возьми текст и переформулируй теплее, но СОХРАНИ ВСЕ ЧИСЛА И ПРОЦЕНТЫ БЕЗ ИЗМЕНЕНИЙ (например: 99,8%, 0,2%, 97-98%).
    2. Ничего не выдумывай — только факты из текста.
    3. Структура: короткое вступление -> полезные детали (абзацы или список) -> мягкое завершение.
    4. Без служебных пометок и заголовков ##, без HTML. Разделяй абзацы пустой строкой; списки — дефис
    5. {emoji_rule}
    6. {cta_rule}
    
    Отвечай только текстом.
    """).strip()
    
    return prompt

def postprocess_answer_with_empathy(base_text: str, tone: str = "friendly", emotion: str = "empathy", cta_text: str | None = None, cta_link: str | None = None) -> str:
    """Делает финальную «оживляющую» переформулировку ответа. Если LLM недоступна/ошибка — возвращает исходный base_text."""
    
    try:
        allow_emoji = True  # вместо random.random() < 0.33
        
        system_prompt = build_empathy_prompt(tone, emotion, allow_emoji, cta_text, cta_link)
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,  # еще меньше креативности
            top_p=0.8,       # ограничить выбор токенов
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_text}
            ]
        )
        
        text = completion.choices[0].message.content.strip()
        
        # Постобработка
        text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)  # убираем заголовки
        text = re.sub(r'\n{3,}', '\n\n', text)  # нормализуем переносы
        
        return text
        
    except Exception as e:
        print(f"Постпроцессор не сработал: {e}")
        return base_text

def theme_boost(score: float, theme_key: str, cfg: dict, chunk) -> float:
    # бустим, если в тегах или тексте есть тематические алиасы
    text_l = chunk.text.lower()
    tags_l = getattr(chunk.metadata, 'tags_lower', [])
    
    # Проверяем теги
    tag_match = any(alias in tags_l for alias in cfg["tag_aliases"])
    # Проверяем текст
    text_match = any(alias in text_l for alias in cfg["tag_aliases"])
    
    if tag_match or text_match:
        boosted_score = score + cfg["weight"]
        print(f"      🎯 Буст {theme_key}: теги={tags_l}, текст содержит={[alias for alias in cfg['tag_aliases'] if alias in text_l]}")
        return boosted_score
    return score

# Утилиты для поиска цифр и процентов
PCT_RE = re.compile(r'\b\d{1,3}(?:[.,]\d{1,2})?\s?%')
NUM_RE = re.compile(r'\b\d{1,5}\b')

def has_numeric_facts(text: str) -> bool:
    """Проверяет наличие цифр или процентов в тексте"""
    text_l = text.lower()
    return bool(PCT_RE.search(text_l) or NUM_RE.search(text_l))

def _find_doctor_direct_or_fuzzy(query: str):
    q = (query or "").lower().replace("\u00a0", "")
    # 1) прямое подстрочное совпадение по ключам
    for k, ch in DOCTOR_NAME_TO_CHUNK.items():
        if k in q:
            return ch
    # 2) морфологически «мягко» по фамилии: Моисеев/Моисеева/Моисееву...
    for k, ch in DOCTOR_NAME_TO_CHUNK.items():
        if " " not in k: # только однословные ключи = фамилии
            if re.search(rf"\b{k}\w*\b", q, flags=re.IGNORECASE):
                return ch
    # 3) опечатки
    tokens = re.findall(r"[а-яё]{4,}", q)
    keys = list(DOCTOR_NAME_TO_CHUNK.keys())
    for t in tokens:
        best = difflib.get_close_matches(t, keys, n=1, cutoff=0.84)
        if best:
            return DOCTOR_NAME_TO_CHUNK[best[0]]
    return None

def fallback_theme_chunks(theme_key: str, limit: int = 3):
    """Если семантика промахнулась - вернуть несколько явных тематических чанков."""
    if not theme_key:
        return []
    
    cfg = THEME_MAP[theme_key]
    out = []
    
    print(f"🔍 Fallback для темы '{theme_key}': ищем теги {cfg['tag_aliases']}")
    
    for ch in all_chunks:
        tags_l = getattr(ch.metadata, 'tags_lower', [])
        text_l = ch.text.lower()
        
        # Проверяем теги
        tag_match = any(alias in (tags_l or []) for alias in cfg["tag_aliases"])
        # Проверяем текст
        text_match = any(alias in text_l for alias in cfg["tag_aliases"])
        
        if tag_match or text_match:
            out.append(ch)
            print(f"  ✅ Найден чанк: {ch.file_name} (теги: {tags_l}, текст содержит: {[alias for alias in cfg['tag_aliases'] if alias in text_l]})")
            if len(out) >= limit:
                break
    
    print(f"🔍 Fallback вернул {len(out)} чанков")
    return out

def route_topics(query: str) -> Set[str]:
    """Роутер по темам - определяет темы запроса"""
    if not hasattr(route_topics, 'theme_map'):
        try:
            with open(THEMES_PATH, "r", encoding="utf-8") as f:
                route_topics.theme_map = json.load(f)
        except Exception as e:
            print(f"⚠️ Не удалось загрузить themes.json: {e}")
            route_topics.theme_map = {}
    
    detected_topics = set()
    query_lower = query.lower()
    
    for topic, config in route_topics.theme_map.items():
        # Проверяем regex
        if "query_regex" in config:
            try:
                pattern = re.compile(config["query_regex"], re.IGNORECASE)
                if pattern.search(query):
                    detected_topics.add(topic)
                    continue
            except Exception:
                pass
        
        # Проверяем tag_aliases
        if "tag_aliases" in config:
            for alias in config["tag_aliases"]:
                if alias.lower() in query_lower:
                    detected_topics.add(topic)
                    break
    
    return detected_topics

def strip_fluff_start(text: str) -> str:
    """Удаляет вступительную 'воду-абзац' из текста"""
    lines = text.split('\n')
    start_removed = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Проверяем все анти-флуфф паттерны
        for pattern in ANTI_FLUFF:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                # Удаляем эту строку и все пустые строки после неё
                lines = lines[i+1:]
                start_removed = True
                break
        
        if start_removed:
            break
    
    # Убираем пустые строки в начале
    while lines and not lines[0].strip():
        lines = lines[1:]
    
    return '\n'.join(lines)

try:
    # Проверяем наличие папки md
    if not folder_path.exists():
        print(f"ERROR: Папка {folder_path} НЕ найдена!")
        raise Exception(f"Папка {folder_path} не существует")
    
    # Проверяем наличие doctors.md
    doctors_file = Path("md/doctors.md")
    if doctors_file.exists():
        print(f"OK: Файл doctors.md найден")
    else:
        print(f"ERROR: Файл doctors.md НЕ найден!")
    
    # Проверяем наличие themes.json
    themes_file = Path("config/themes.json")
    if themes_file.exists():
        print(f"OK: Файл config/themes.json найден")
    else:
        print(f"ERROR: Файл config/themes.json НЕ найден!")
    
    for file in folder_path.rglob("*.md"):
        try:
            print(f"📄 Обрабатываю файл: {file}")
            text = file.read_text(encoding="utf-8")
            print(f"  📖 Прочитан файл: {file.name} ({len(text)} символов)")
            
            # Парсим YAML front matter
            metadata, content = parse_yaml_front_matter(text)
            content = _normalize(content) # нормализуем пробелы
            print(f"  ✅ YAML парсинг: {metadata.id if metadata.id else 'без ID'}")
            
            # если это файл с врачами - собрать имена
            if getattr(metadata, 'doc_type', '') == 'doctor' or file.name == "doctors.md":
                found = _extract_doctor_names_from_text(content)
                if found:
                    DOCTOR_NAME_TOKENS.update(found)
                    print(f"  🏥 Найдены врачи в {file.name}: {found}")
            
            # Разбиваем на чанки по секциям
            file_chunks = chunk_text_by_sections(content, file.name)
            print(f"  📝 Создано чанков: {len(file_chunks)}")
            
            # ==== ИНДЕКСАЦИЯ КАТАЛОГА ИМПЛАНТОВ ====
            if metadata.doc_type == "catalog" and metadata.topic == "implants":
                print(f"  🏷️ Индексируем каталог имплантов из {file.name}")
                for ch in file_chunks:
                    # Ищем секции по ### Заголовок
                    m = re.search(r'(?m)^###\s+(.+?)\s*$', ch.text)
                    if not m:
                        continue
                    
                    name = m.group(1).strip()
                    
                    # Парсим алиасы из HTML-комментариев
                    alias_m = re.search(r'<!--\s*aliases:\s*\[(.*?)\]\s*-->', ch.text)
                    aliases = [name]
                    if alias_m:
                        aliases.extend([a.strip() for a in alias_m.group(1).split(',')])
                    
                    # Создаем entity_key
                    entity_key = _slugify_implant_kind(name)
                    print(f"    📋 Секция: '{name}' → '{entity_key}' (алиасы: {aliases})")
                    
                    # Сохраняем чанк
                    ENTITY_CHUNKS[("implants", entity_key)] = ch
                    
                    # Индексируем алиасы
                    for a in aliases:
                        ENTITY_INDEX[_norm(a)] = {
                            "topic": "implants", 
                            "entity": entity_key, 
                            "doc_id": metadata.id or file.name, 
                            "section": name
                        }
            
            # Привязываем метаданные к каждому чанку
            for chunk in file_chunks:
                chunk.metadata = metadata
                
                # Обновляем ENTITY_INDEX для всех чанков
                if metadata.topic:
                    update_entity_index(chunk, metadata.topic, chunk.id)
                
                # перед append(chunk) - нормализуй теги в метаданных
                try:
                    if hasattr(chunk.metadata, "tags") and isinstance(chunk.metadata.tags, list):
                        chunk.metadata.tags_lower = [str(t).strip().lower() for t in chunk.metadata.tags]
                    else:
                        chunk.metadata.tags_lower = []
                except Exception:
                    chunk.metadata.tags_lower = []
                
                all_chunks.append(chunk)
                
                # Если это карточка врача (подзаголовок ### Имя Фамилия [Отчество]) — запоминаем прямую ссылку
                hdr = re.search(r'(?m)^###\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})\s*$', chunk.text)
                if hdr:
                    full = hdr.group(1).strip()
                    parts = full.split()
                    last = parts[0] # Фамилия
                    first = parts[1] if len(parts) > 1 else ""
                    patr = parts[2] if len(parts) > 2 else ""
                    
                    # Ключи, по которым реально спрашивают
                    keys = {
                        full,                               # "Моисеев Кирилл Николаевич"
                        last,                               # "Моисеев"
                        (f"{last} {first}").strip(),        # "Моисеев Кирилл"
                        (f"{first} {last}").strip(),        # "Кирилл Моисеев"
                    }
                    
                    for k in keys:
                        DOCTOR_NAME_TO_CHUNK[k.lower()] = chunk
                    
                    # Для регэкспа/подсветки - без отчеств, чтобы не засорять
                    DOCTOR_NAME_TOKENS.update({full, last, f"{first} {last}".strip(), f"{last} {first}".strip()})
                    print(f"🔗 Карточка врача: {full} → {getattr(chunk, 'section', chunk.file_name)}; ключи: {sorted(keys)}")
            
            print(f"  ✅ Файл обработан успешно")
        except Exception as e:
            print(f"❌ Ошибка при обработке файла {file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\u23f3 Найдено {len(all_chunks)} чанков")
    
    # Создаем BM25 индекс
    if all_chunks:
        print(f"🔍 Создаем BM25 индекс для {len(all_chunks)} чанков...")
        bm25_corpus = []
        for chunk in all_chunks:
            # Токенизируем текст для BM25
            tokens = re.findall(r'\w+', chunk.text.lower())
            bm25_corpus.append(tokens)
        
        bm25_index = BM25Okapi(bm25_corpus)
        print(f"✅ BM25 индекс создан")
    
    # Отладочная информация о чанках
    for chunk in all_chunks[:5]:  # Показываем первые 5 чанков
        print(f"  📄 {chunk.file_name}: {chunk.text[:80]}...")
    
    # Пересобираем regex для врачей
    _rebuild_doctor_regex()
    print(f"Врачи: Собраны имена врачей: {len(DOCTOR_NAME_TOKENS)} -> {sorted(list(DOCTOR_NAME_TOKENS))[:6]} ...")
    
    # Проверяем, что есть чанки для обработки
    if len(all_chunks) == 0:
        print("⚠️ Предупреждение: Не найдено ни одного чанка для обработки")
        # Создаем пустой индекс
        dimension = 1536  # размерность text-embedding-3-small
        index = faiss.IndexFlatL2(dimension)
    else:
        # Получаем эмбеддинги для всех фрагментов
        print(f"\u23f3 Генерация эмбеддингов для {len(all_chunks)} чанков...")
        
        def get_embedding(text: str) -> List[float]:
            resp = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return resp.data[0].embedding
        
        # Делаем функцию глобально доступной
        globals()['get_embedding'] = get_embedding
        
        # Создаем эмбеддинги только для текста чанков
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = [get_embedding(text) for text in chunk_texts]
        
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))
        
        print(f"✅ Индекс создан с {len(all_chunks)} чанками")

except Exception as e:
    print(f"❌ Критическая ошибка при инициализации: {e}")
    import traceback
    traceback.print_exc()
    # Создаем пустой индекс для fallback
    dimension = 1536
    index = faiss.IndexFlatL2(dimension)
    all_chunks = []

# Fallback функция get_embedding
def get_embedding(text: str) -> List[float]:
    """Fallback функция для создания эмбеддингов"""
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"Ошибка при создании эмбеддинга: {e}")
        # Возвращаем нулевой вектор
        return [0.0] * 1536

def generate_query_variants(query: str) -> List[str]:
    """Генерирует 2-3 варианта перефразировки запроса для лучшего поиска"""
    if not openai_client:
        return [query]  # Fallback к исходному запросу
    
    prompt = f"""Перефразируй вопрос пациента о стоматологии в 2-3 разных варианта для поиска информации.

Исходный вопрос: "{query}"

Создай варианты:
1. Более формальный/медицинский
2. Более простой/бытовой  
3. С синонимами и альтернативными формулировками

Верни ТОЛЬКО JSON:
{{"variants": ["вариант 1", "вариант 2", "вариант 3"]}}

Примеры:
- "больно ли ставить имплант" → ["болезненность имплантации", "дискомфорт при установке импланта", "ощущения во время имплантации"]
- "сколько стоит имплантация" → ["стоимость имплантации зубов", "цена на импланты", "расценки на имплантацию"]"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        variants = result.get("variants", [])
        
        # Добавляем исходный запрос в начало
        all_variants = [query] + variants
        # Убираем дубликаты, сохраняя порядок
        unique_variants = []
        seen = set()
        for variant in all_variants:
            if variant.lower() not in seen:
                unique_variants.append(variant)
                seen.add(variant.lower())
        
        print(f"🔍 Multi-query: сгенерировано {len(unique_variants)} вариантов запроса")
        return unique_variants[:3]  # Максимум 3 варианта
        
    except Exception as e:
        print(f"❌ Ошибка генерации вариантов запроса: {e}")
        return [query]  # Fallback к исходному запросу

def hybrid_retriever(query: str, top_n: int = 20) -> List[Tuple[RetrievedChunk, float]]:
    """Гибридный ретривер: объединяет BM25 и эмбеддинги"""
    if not all_chunks or len(all_chunks) == 0:
        return []
    
    candidates = []
    
    # ==== BM25 поиск ====
    if bm25_index:
        query_tokens = re.findall(r'\w+', query.lower())
        bm25_scores = bm25_index.get_scores(query_tokens)
        bm25_candidates = []
        
        for i, score in enumerate(bm25_scores):
            if score > 0:
                bm25_candidates.append((all_chunks[i], score))
        
        # Нормализуем BM25 scores
        if bm25_candidates:
            max_score = max(score for _, score in bm25_candidates)
            if max_score > 0:
                bm25_candidates = [(chunk, score / max_score * 0.6) for chunk, score in bm25_candidates]
                candidates.extend(bm25_candidates[:top_n])
    
    # ==== Эмбеддинги поиск ====
    try:
        query_embedding = get_embedding(query)
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=min(top_n, len(all_chunks)))
        
        # Нормализуем embedding scores
        max_dist = max(D[0]) if len(D[0]) > 0 else 1.0
        if max_dist > 0:
            embedding_candidates = []
            for i, dist in zip(I[0], D[0]):
                score = (1.0 - dist / max_dist) * 0.4  # Нормализуем и применяем вес
                embedding_candidates.append((all_chunks[i], score))
            candidates.extend(embedding_candidates)
    except Exception as e:
        print(f"Ошибка в embedding поиске: {e}")
    
    # ==== Объединяем и убираем дубли ====
    seen_chunks = set()
    unique_candidates = []
    
    for chunk, score in candidates:
        if chunk.id not in seen_chunks:
            seen_chunks.add(chunk.id)
            unique_candidates.append((chunk, score))
    
    # Сортируем по score и возвращаем top_n
    unique_candidates.sort(key=lambda x: x[1], reverse=True)
    return unique_candidates[:top_n]

def llm_rerank(candidates: List[Tuple[RetrievedChunk, float]], query: str) -> List[Tuple[RetrievedChunk, float]]:
    """LLM-реранкинг для точной оценки релевантности чанков"""
    if not candidates or not openai_client:
        return candidates
    
    # Берем только top-6 кандидатов для реранкинга
    top_candidates = candidates[:6]
    
    # Формируем промпт для LLM
    chunks_text = ""
    for i, (chunk, score) in enumerate(top_candidates):
        # Берем только заголовок и первые 200 символов для экономии токенов
        header_match = re.search(r'(?m)^##\s+(.+?)\s*$', chunk.text)
        header = header_match.group(1) if header_match else "Без заголовка"
        preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        chunks_text += f"{i+1}. Заголовок: {header}\nТекст: {preview}\n\n"
    
    prompt = f"""Оцени релевантность каждого фрагмента для ответа на вопрос пользователя.

Вопрос: "{query}"

Фрагменты:
{chunks_text}

Верни ТОЛЬКО JSON с оценками от 0.0 до 1.0:
{{"scores": [0.8, 0.3, 0.9, 0.1, 0.7, 0.2]}}

Где 1.0 = максимально релевантно, 0.0 = не релевантно."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        llm_scores = result.get("scores", [])
        
        # Применяем LLM-оценки к кандидатам
        reranked = []
        for i, (chunk, base_score) in enumerate(top_candidates):
            if i < len(llm_scores):
                # Комбинируем базовый score с LLM-оценкой (70% LLM, 30% базовый)
                final_score = llm_scores[i] * 0.7 + base_score * 0.3
                reranked.append((chunk, final_score))
            else:
                reranked.append((chunk, base_score))
        
        # Сортируем по финальному score
        reranked.sort(key=lambda x: x[1], reverse=True)
        print(f"🔍 LLM-реранкинг: {len(reranked)} кандидатов переоценены")
        
        return reranked
        
    except Exception as e:
        print(f"❌ Ошибка LLM-реранкинга: {e}")
        return candidates

def reranker(candidates: List[Tuple[RetrievedChunk, float]], query: str, detected_topics: Set[str]) -> List[RetrievedChunk]:
    """Реранкер с LLM-оценкой релевантности"""
    if not candidates:
        return []
    
    # Сначала применяем LLM-реранкинг
    llm_reranked = llm_rerank(candidates, query)
    
    # Затем применяем эвристические бонусы
    scored_candidates = []
    query_lower = query.lower()
    
    for chunk, base_score in llm_reranked:
        final_score = base_score
        
        # +0.2 если тема от router совпала с темой чанка
        if detected_topics and chunk.metadata.topic:
            if chunk.metadata.topic in detected_topics:
                final_score += 0.2
        
        # +0.1 если найден alias через ENTITY_INDEX
        for alias, meta in ENTITY_INDEX.items():
            if alias in query_lower:
                if meta["doc_id"] == chunk.file_name:
                    final_score += 0.1
                    break
        
        scored_candidates.append((chunk, final_score))
    
    # Сортируем по финальному score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем 2-3 лучших чанка
    return [chunk for chunk, _ in scored_candidates[:3]]



def retrieve_relevant_chunks(query: str, top_k: int = 8) -> List[RetrievedChunk]:
    """Извлекает релевантные чанки с multi-query rewrite и улучшенным ранжированием"""
    if len(all_chunks) == 0:
        print("⚠️ Нет чанков для поиска")
        return []
    
    # ==== РОУТЕР ПО ТЕМАМ ====
    detected_topics = route_topics(query)
    print(f"🎯 Роутер определил темы: {detected_topics}")
    
    # ==== БЫСТРЫЙ ПУТЬ: ДЕТЕКТ СУЩНОСТИ ====
    q = _norm(query or "")
    hit = None
    
    # Прямое вхождение алиаса
    for alias, meta in ENTITY_INDEX.items():
        if alias in q:
            hit = ENTITY_CHUNKS.get((meta["topic"], meta["entity"]))
            break
    
    # Мягкий матч (разделители -/-/- и пробелы)
    if not hit:
        q2 = re.sub(r'[---]', '', q)
        for alias, meta in ENTITY_INDEX.items():
            a2 = re.sub(r'[---]', '', alias)
            if a2 in q2:
                hit = ENTITY_CHUNKS.get((meta["topic"], meta["entity"]))
                break
    
    if hit:
        print(f"✅ Найдена сущность каталога: '{query}' → {hit.id}")
        return [hit]  # ровно нужная секция каталога
    
    # ==== СПЕЦИАЛЬНЫЕ ЗАПРОСЫ: ЛИСТИНГ И СРАВНЕНИЕ ====
    # Листинг ("какие виды")
    if re.search(r'(какие|какой|что за).* (вид|вариант).* (имплантац|имплант)', q):
        kinds_order = ["single-stage", "classic", "all-on-4", "all-on-6"]
        out = [ENTITY_CHUNKS[("implants", k)] for k in kinds_order if ("implants", k) in ENTITY_CHUNKS]
        print(f"📋 Листинг видов имплантации: найдено {len(out)} типов")
        return out[:4]
    
    # Сравнение ("чем отличается / что лучше")
    if re.search(r'(чем\s+отлич|различ|разница|сравн|что\s+лучше|vs)', q):
        kinds_order = ["single-stage", "classic", "all-on-4", "all-on-6"]
        out = [ENTITY_CHUNKS[("implants", k)] for k in kinds_order if ("implants", k) in ENTITY_CHUNKS]
        print(f"📊 Сравнение видов имплантации: найдено {len(out)} типов")
        return out[:4]
    
    # 0) если нашли врача напрямую — сразу отдаём карточку, минуя FAISS
    hit = _find_doctor_direct_or_fuzzy(query)
    if hit:
        print(f"✅ Карточка врача по запросу: {query} → {getattr(hit, 'section', hit.file_name)}")
        return [hit] # без дедупа
    
    try:
        print(f"🔍 Поиск: '{query}' в {len(all_chunks)} чанках")
        
        # ==== MULTI-QUERY REWRITE ====
        query_variants = generate_query_variants(query)
        print(f"🔍 Multi-query: используем {len(query_variants)} вариантов запроса")
        
        # ==== ГИБРИДНЫЙ РЕТРИВЕР ДЛЯ КАЖДОГО ВАРИАНТА ====
        all_candidates = []
        for variant in query_variants:
            candidates = hybrid_retriever(variant, top_n=15)  # Меньше кандидатов на вариант
            all_candidates.extend(candidates)
            print(f"🔍 Вариант '{variant[:30]}...': найдено {len(candidates)} кандидатов")
        
        # ==== ОБЪЕДИНЕНИЕ И ДЕДУПЛИКАЦИЯ ====
        seen_chunks = set()
        unique_candidates = []
        
        for chunk, score in all_candidates:
            if chunk.id not in seen_chunks:
                seen_chunks.add(chunk.id)
                unique_candidates.append((chunk, score))
        
        # Сортируем по score и берем top
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = unique_candidates[:25]  # Больше кандидатов для реранкинга
        
        print(f"🔍 Multi-query: объединено {len(unique_candidates)} уникальных кандидатов")
        
        # ==== РЕРАНКЕР ====
        final_chunks = reranker(top_candidates, query, detected_topics)
        print(f"🔍 Реранкер отобрал {len(final_chunks)} финальных чанков")
        
        # если ничего внятного не попало и тема известна — жёсткий fallback
        if not final_chunks and detected_topics:
            theme_key = list(detected_topics)[0]
            final_chunks = fallback_theme_chunks(theme_key, limit=top_k)
        
        # если ищем врача - не режем результаты по file_name
        if DOCTOR_NAME_TO_CHUNK and _find_doctor_direct_or_fuzzy(query):
            return final_chunks[:top_k] # без дедупа
        
        # иначе оставляем защитный дедуп по файлам
        seen = set()
        out = []
        for chunk in final_chunks:
            if chunk.file_name not in seen:
                seen.add(chunk.file_name)
                out.append(chunk)
        
        print(f"✅ Возвращаем {len(out)} уникальных чанков")
        return out[:top_k]
        
    except Exception as e:
        print(f"❌ Ошибка при поиске чанков: {e}")
        return []

def synthesize_answer(chunks: List[RetrievedChunk], user_query: str) -> SynthJSON:
    """Синтезирует структурированный JSON ответ"""
    
    # Формируем контекст из чанков
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"ID: {chunk.id}\nОбновлено: {chunk.updated}\nКритичность: {chunk.criticality}\nКонтент:\n{chunk.text}\n")
    
    context = "\n---\n".join(context_parts)
    print(f"📝 Контекст для синтеза: {len(context)} символов")
    print(f"📝 Первые 200 символов контекста: {context[:200]}...")
    
    # Определяем метаданные для ответа (берем из первого чанка)
    primary_chunk = chunks[0] if chunks else None
    tone = primary_chunk.metadata.tone if primary_chunk else "friendly"
    preferred_format = primary_chunk.metadata.preferred_format if primary_chunk else ["short", "bullets", "cta"]
    verbatim = primary_chunk.metadata.verbatim if primary_chunk else False
    
    # Системный промпт с учетом verbatim
    if verbatim:
        system_prompt = f"""
Ты — дружелюбный и внимательный ассистент стоматологической клиники ЦЭСИ на Камчатке.
Твоя задача — отвечать на вопросы пациентов строго по фактам из базы знаний.

Правила:
0. ТОЛЬКО факты из контекста, без выдумок
1. Коротко и по делу
2. Пункты/списки
3. Если в контексте есть числа/сроки — НЕ изменять
4. Без эмпатии/общих фраз — только факты
5. Без служебных символов Markdown в ответе пользователю
6. Разбивай текст на абзацы и смысловые блоки
7. Если информации нет — честно скажи об этом

## Контекст для ответа:
{context}

## Вопрос пользователя:
{user_query}

## Метаданные:
- Тон: {tone}
- Предпочтительный формат: {preferred_format}
- CTA текст: {primary_chunk.metadata.cta_text if primary_chunk else 'Записаться на консультацию'}

Отвечай строго в формате JSON:
{{
    "short": "Короткий ответ по сути вопроса",
    "bullets": ["Деталь 1", "Деталь 2", "Деталь 3"],
    "cta": "Мягкий призыв к действию",
    "used_chunks": ["ID1", "ID2"],
    "tone": "{tone}",
    "warnings": []
}}
"""
    else:
        system_prompt = f"""
Ты — дружелюбный и внимательный ассистент стоматологической клиники ЦЭСИ на Камчатке.
Твоя задача — отвечать на вопросы пациентов просто, понятно и с заботой, но строго по делу и на основе базы знаний.
Тон общения — тёплый, экспертный и уважительный.

Правила:
0. ВАЖНО: Если вопрос о боли, страхе, дискомфорте — ВСЕГДА сначала дай подробный информативный ответ на основе базы знаний, а потом короткую успокаивающую фразу. НИКОГДА не предлагай сразу консультацию без ответа!
1. Всегда пиши в формате обычного диалога, без служебных символов Markdown в ответе пользователю.
2. Разбивай текст на абзацы и смысловые блоки.
3. Используй человеческие формулировки, избегай пустых фраз.
4. Эмодзи используй только при необходимости и умеренно.
5. Если вопрос тревожный — сначала факты из базы знаний, потом мягкая успокаивающая фраза.
6. Формулировки и цифры из базы знаний сохраняй максимально точно, адаптируя под живой диалог.
7. Если информации нет — честно скажи об этом и предложи альтернативу: консультацию или другой способ узнать ответ.
8. В конце при возможности добавь мягкий CTA с предложением консультации.

Как работать с базой:
1. YAML-фронтматтер игнорируй при формулировании ответа, но используй его значения для стиля и структуры.
2. При построении ответа ориентируйся на поля:
   - 'preferred_format' — структура ответа.
   - 'cta_text' и 'cta_link' — для финальной фразы.
   - 'tone', 'emotion' — для стиля.
3. Используй контент из разделов:
   - **Коротко** — для первого абзаца (прямой ответ).
   - **Детали** — для второго абзаца (факты и уточнения).
   - **Следующий шаг** — для заключения и CTA.

## Контекст для ответа:
{context}

## Вопрос пользователя:
{user_query}

## Метаданные:
- Тон: {tone}
- Предпочтительный формат: {preferred_format}
- CTA текст: {primary_chunk.metadata.cta_text if primary_chunk else 'Записаться на консультацию'}

ВАЖНО: Если вопрос о враче или сотруднике клиники, отвечай только по базе. Если данных нет, скажи, что информации нет, без предположений.

Отвечай строго в формате JSON:
{{
    "short": "Короткий ответ по сути вопроса",
    "bullets": ["Деталь 1", "Деталь 2", "Деталь 3"],
    "cta": "Мягкий призыв к действию",
    "used_chunks": ["ID1", "ID2"],
    "tone": "{tone}",
    "warnings": []
}}
"""

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,  # низкая температура для стабильности
        max_tokens=250,   # ограничиваем длину
        top_p=0.8,       # ограничиваем выбор токенов
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Ты отвечаешь строго JSON по схеме {short, bullets[], cta, used_chunks, tone, warnings}. Никогда не используй Markdown заголовки в ответе."},
            {"role": "user", "content": system_prompt}
        ]
    )
    
    try:
        json_response = json.loads(completion.choices[0].message.content)
        return SynthJSON(
            short=json_response.get("short", ""),
            bullets=json_response.get("bullets", []),
            cta=json_response.get("cta", ""),
            used_chunks=json_response.get("used_chunks", []),
            tone=json_response.get("tone", tone),
            warnings=json_response.get("warnings", [])
        )
    except json.JSONDecodeError:
        # Fallback если JSON некорректный
        return SynthJSON(
            short="Извините, произошла ошибка при обработке ответа.",
            bullets=["Попробуйте переформулировать вопрос"],
            cta="Обратитесь к администратору",
            used_chunks=[],
            tone="friendly",
            warnings=["JSON parsing error"]
        )

def compress_answer(text: str, max_length: int = 800) -> str:
    """Сжимает длинный ответ до указанной длины без потери ключевой информации"""
    if len(text) <= max_length or not openai_client:
        return text
    
    # Считаем количество предложений
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) <= 3:
        return text  # Если мало предложений, не сжимаем
    
    prompt = f"""Сожми этот ответ о стоматологии до {max_length} символов, сохранив ВСЕ важные факты и цифры.

ВАЖНО:
- Сохрани ВСЕ числа, проценты, сроки, цены
- Сохрани ключевую медицинскую информацию
- Убери только "воду" и повторения
- Оставь структуру: короткий ответ + детали + призыв к действию

Исходный ответ:
{text}

Сжатый ответ:"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        
        compressed = response.choices[0].message.content.strip()
        
        # Проверяем, что сжатие не слишком агрессивное
        if len(compressed) < max_length * 0.5:
            print(f"⚠️ Сжатие слишком агрессивное, возвращаем исходный текст")
            return text
        
        print(f"🔍 Answer compression: {len(text)} → {len(compressed)} символов")
        return compressed
        
    except Exception as e:
        print(f"❌ Ошибка сжатия ответа: {e}")
        return text

def render_markdown(synth: SynthJSON) -> str:
    """Рендерит JSON в красивый Markdown без служебных заголовков"""
    
    def clean_text(text: str) -> str:
        """Очищает текст от служебных символов и инструкций"""
        # Убираем служебные символы и инструкции
        cleaned = re.sub(r'^(?:\*|Ответ:|Инструкция:|JSON:|Промпт:)\s*', '', text, flags=re.IGNORECASE)
        # Убираем Markdown заголовки
        cleaned = re.sub(r'^\s*#{1,6}\s*', '', cleaned)  # <- срезаем '## ...'
        # Убираем лишние пробелы
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Убираем кавычки в начале и конце
        cleaned = re.sub(r'^["\']+|["\']+$', '', cleaned)
        return cleaned.strip()
    
    # Очищаем и форматируем части
    short = clean_text(synth.short)
    cta = clean_text(synth.cta)
    
    # Форматируем bullets
    bullets = []
    for bullet in synth.bullets:
        clean_bullet = clean_text(bullet)
        if clean_bullet and len(clean_bullet) > 5:  # Минимальная длина
            bullets.append(f"- {clean_bullet}")
    
    # Собираем ответ БЕЗ служебных заголовков
    blocks = []
    
    # Короткий ответ
    if short:
        blocks.append(short)
    
    # Детали
    if bullets:
        if blocks:  # Если уже есть контент, добавляем разделение
            blocks.append("")
        blocks.append("\n".join(bullets))
    
    # Следующий шаг (CTA) - без лишних отступов
    if cta:
        if blocks:  # Если уже есть контент, добавляем разделение
            blocks.append("")
        blocks.append(cta)
    
    # Собираем результат
    markdown = "\n".join(blocks)
    
    # Нормализуем пустые строки (максимум 1 подряд)
    markdown = re.sub(r'\n{2,}', '\n', markdown)
    
    # Применяем анти-воду
    markdown = strip_fluff_start(markdown)
    
    return markdown

# Основная функция
def get_rag_answer(user_message: str, history: List[Dict] = []) -> tuple[str, dict]:
    """Основная функция для получения ответа и метаданных"""
    # Увеличиваем top_k для поиска врачей
    if DOCTOR_REGEX and DOCTOR_REGEX.search(user_message):
        top_k = 12
        print(f"🔍 Поиск врачей: используем top_k={top_k}")
    else:
        top_k = 8  # немного шире по умолчанию
    
    # Извлекаем релевантные чанки
    relevant_chunks = retrieve_relevant_chunks(user_message, top_k=top_k)
    
    if not relevant_chunks:
        return """К сожалению, в моей базе нет информации по этому вопросу.

Запишитесь на бесплатную консультацию — наш специалист ответит на все ваши вопросы.""", {}
    
    # Синтезируем структурированный ответ
    synth_response = synthesize_answer(relevant_chunks, user_message)
    
    # Рендерим в Markdown
    markdown_response = render_markdown(synth_response)
    
    # ==== ЭМПАТИЯ ====
    # Определяем эмоцию на основе запроса и контекста
    retrieved_snippet = "\n".join([ch.text[:200] for ch in relevant_chunks[:3]])
    fm = {}
    if relevant_chunks:
        try:
            meta = relevant_chunks[0].metadata
            fm = {
                "emotion": getattr(meta, "emotion", None),
                "verbatim": getattr(meta, "verbatim", False),
                "doc_type": getattr(meta, "doc_type", None)
            }
        except Exception:
            pass
    
    emotion, detector, confidence = detect_emotion(
        user_query=user_message,
        retrieved_snippet=retrieved_snippet,
        fm=fm,
        empathy_cfg=EMPATHY_CFG,
        triggers_bank=EMPATHY_TRIGGERS,
        llm_client=openai_client,
        model="gpt-4o-mini"
    )
    
    # Собираем финальный текст с эмпатией
    core_text = markdown_response
    
    # Сначала добавляем openers/closers
    text_with_empathy = build_answer(core_text, emotion, EMPATHY_BANK, EMPATHY_CFG, _RNG)
    
    # Затем применяем LLM-переформулировку для более естественного звучания
    if emotion != "none" and openai_client:
        try:
            final_text = postprocess_answer_with_empathy(
                base_text=text_with_empathy,
                tone="friendly",
                emotion=emotion,
                cta_text=cta_text,
                cta_link=cta_link
            )
        except Exception as e:
            print(f"⚠️ LLM-переформулировка не сработала: {e}")
            final_text = text_with_empathy
    else:
        final_text = text_with_empathy
    
    # Постпроцессор: не даём «пустых» ответов
    if not relevant_chunks or len(markdown_response.strip()) < 40:
        print(f"⚠️ Пустой ответ, пробуем тематический fallback...")
        # Попробуем достать тематический факт для боли/цены/гарантии
        detected_topics = route_topics(user_message)
        print(f"🔍 Обнаружены темы: {detected_topics}")
        if detected_topics:
            theme_key = list(detected_topics)[0]
            fb = fallback_theme_chunks(theme_key, limit=2)
            print(f"🔍 Fallback чанков: {len(fb)}")
            if fb:
                markdown_response = "\n\n".join(ch.text.strip() for ch in fb if ch and ch.text)[:1200]
                print(f"✅ Fallback применен, новый размер: {len(markdown_response)} символов")
            else:
                print(f"❌ Fallback не сработал для темы: {theme_key}")
    
    # Собираем метаданные из первого чанка для CTA
    rag_meta = {}
    cta_text = None
    cta_link = None
    
    for ch in relevant_chunks:
        try:
            meta = ch.metadata
            if getattr(meta, "cta_text", None): cta_text = meta.cta_text
            if getattr(meta, "cta_link", None): cta_link = meta.cta_link
            
            # Собираем метаданные для CTA
            rag_meta = {
                "topic": getattr(meta, "topic", None),
                "doc_type": getattr(meta, "doc_type", None),
                "section": getattr(meta, "section", None),
                "cta_action": getattr(meta, "cta_action", None),
                "cta_text": cta_text,
                "cta_link": cta_link
            }
            break  # берем только из первого чанка
        except Exception:
            pass
    
    # Применяем анти-флуфф фильтр к финальному тексту
    final_text = strip_fluff_start(final_text)
    
    # ==== ANSWER COMPRESSION ====
    if len(final_text) > 800:  # Сжимаем только длинные ответы
        final_text = compress_answer(final_text, max_length=800)
    
    # Отладочная информация
    print(f"🔍 Вопрос: '{user_message}'")
    print(f"📄 Найдено чанков: {len(relevant_chunks)}")
    for i, chunk in enumerate(relevant_chunks[:3]):  # Показываем первые 3
        print(f"  {i+1}. {chunk.file_name}: {chunk.text[:100]}...")
    print(f"📝 Ответ: {final_text[:200]}...")
    
    # Добавляем used_chunks в метаданные
    used_ids = [ch.id for ch in relevant_chunks]
    rag_meta["used_chunks"] = used_ids
    
    # Добавляем поля эмпатии в метаданные
    rag_meta.update({
        "emotion": emotion,
        "detector": detector,
        "confidence": round(float(confidence), 2),
        "opener_used": (emotion != "none" and any(final_text.startswith(x) for x in EMPATHY_BANK.get(emotion, {}).get("openers", []))),
        "closer_used": (emotion != "none" and any(final_text.endswith(x) for x in EMPATHY_BANK.get(emotion, {}).get("closers", [])))
    })
    
    print(f"📋 Метаданные: {rag_meta}")
    
    return final_text, rag_meta

def log_query_response(user_message: str, response: str, metadata: dict, chunks_used: List[str] = None):
    """Логирует вопрос/ответ в файл для анализа"""
    import json
    from datetime import datetime
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": user_message,
        "answer": response,
        "metadata": metadata,
        "chunks_used": chunks_used or [],
        "answer_length": len(response),
        "has_cta": bool(metadata.get("cta_action")),
        "topic": metadata.get("topic", "unknown"),
        "doc_type": metadata.get("doc_type", "unknown"),
        # Поля эмпатии для отладки
        "emotion": metadata.get("emotion", "none"),
        "emotion_source": metadata.get("detector", "unknown"),
        "emotion_confidence": metadata.get("confidence", 0.0),
        "opener_used": metadata.get("opener_used", False),
        "closer_used": metadata.get("closer_used", False)
    }
    
    try:
        # Создаем папку logs если её нет
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Записываем в JSONL файл
        log_file = logs_dir / "queries.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        print(f"📝 Лог записан: {log_file}")
        
    except Exception as e:
        print(f"❌ Ошибка логирования: {e}")


