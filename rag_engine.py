# rag_engine.py
import os
import numpy as np
from openai import OpenAI
from core.faiss_compat import IndexFlatIP, normalize_L2_inplace, HAS_FAISS
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
# from core.empathy import detect_emotion, build_answer  # Функции не используются в новом коде

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
ALIAS_MAP = {}  # normalize(alias) -> {"file": path, "primary_h2_id": ...}

# ==== НОВАЯ АРХИТЕКТУРА ИНДЕКСОВ ====
# === 1) Нормализация тем ===
CANON = {"doctors","consultation","prices","warranty","contacts","implants","safety","clinic"}

def norm_topic(x: str) -> str:
    return (x or "").strip().lower()

def norm_text(x: str) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", (x or "").lower().strip())
    x = re.sub(r"\s+", " ", x)
    return x

# === 2) Контейнеры индексов ===
ALIAS_MAP_GLOBAL = {}   # norm(alias) -> {"topic":..., "file":...}
H2_INDEX = {}           # norm(h2_text/h2_id/local_alias) -> {"topic":..., "file":..., "h2_id":...}
FILE_META = {}          # file -> {"topic":..., "aliases": [...], "mini_links":[...]}
ALL_CHUNKS = []         # ваши чанк-объекты (как и раньше)

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

# Загрузка конфигов эмпатии (старые конфиги - не используются в новой архитектуре)
# with open(os.path.join("config", "empathy_config.yaml"), "r", encoding="utf-8") as f:
#     EMPATHY_CFG = yaml.safe_load(f)
# with open(os.path.join("config","empathy.yaml"), "r", encoding="utf-8") as f:
#     EMPATHY_BANK = yaml.safe_load(f)
# with open(os.path.join("config", "empathy_triggers.yaml"), "r", encoding="utf-8") as f:
#     EMPATHY_TRIGGERS = yaml.safe_load(f)
# _RNG = random.Random()

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
        self.aliases = data.get('aliases', [])  # ✅ Добавляем aliases
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

# === 3) Парс MD ===
RX_H2 = re.compile(r"(?m)^##\s+([^\n{]+?)(?:\s*\{#([^\}]+)\})?\s*$")
RX_LOC = re.compile(r"<!--\s*aliases:\s*\[(.*?)\]\s*-->", re.S|re.I)

def parse_frontmatter(text: str):
    if not text.startswith("---"): return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not m: return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)
    return fm, body

def parse_h2_sections(body: str):
    sections = []
    for m in RX_H2.finditer(body):
        title = m.group(1).strip()
        h2_id = m.group(2) or slugify(title)
        tail = body[m.end(): m.end()+400]
        loc = RX_LOC.search(tail)
        local_aliases = []
        if loc:
            raw = loc.group(1)
            local_aliases = [a.strip().strip("'\"") for a in re.split(r",\s*", raw) if a.strip()]
        sections.append({"title": title, "h2_id": h2_id, "local_aliases": local_aliases})
    return sections

def slugify(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower().strip())
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-а-яё]", "", s)
    return s

# === 4) Регистрация индексов при загрузке файла ===
def register_file(file_name: str, md_text: str):
    fm, body = parse_frontmatter(md_text)
    doc_type = norm_topic(fm.get("doc_type"))
    topic = norm_topic(fm.get("topic") or doc_type)
    if topic not in CANON:
        topic = "clinic"  # fallback на clinic

    aliases = fm.get("aliases") or []
    if isinstance(aliases, str): aliases = [aliases]
    mini_links = fm.get("mini_links") or []

    FILE_META[file_name] = {"topic": topic, "aliases": aliases, "mini_links": mini_links}

    # глобальные алиасы → файл/тема
    for a in aliases:
        ALIAS_MAP_GLOBAL[norm_text(a)] = {"topic": topic, "file": file_name}

    # H2 и локальные алиасы → точный индекс
    for s in parse_h2_sections(body):
        # индексируем заголовок и h2_id
        for key in [s["title"], s["h2_id"], *s["local_aliases"]]:
            H2_INDEX[norm_text(key)] = {"topic": topic, "file": file_name, "h2_id": s["h2_id"]}

# === 5) DEFAULT_H2 (страховка) ===
DEFAULT_H2 = {
  "consultation": ("consultation-free.md",         "обзор"),
  "prices":       ("prices-clinic.md",             None),
  "warranty":     ("warranty.md",                  "обзор"),
  "contacts":     ("clinic-contacts.md",           None),
  "implants":     ("implants-overview.md",         "обзор"),
  "safety":       ("implants-contraindications.md","обзор"),
  "doctors":      ("doctors.md",                   "обзор"),
  "clinic":       ("advantages-general.md",        "обзор")
}

def _find_chunk(file_name: str, h2_id: str|None):
    for ch in ALL_CHUNKS:
        if ch.file_name == file_name:
            if h2_id is None or getattr(ch.metadata, "h2_id", None) == h2_id:
                return ch
    return None

def get_default_chunk_for_topic(topic: str):
    file_name, h2_id = DEFAULT_H2.get(topic, (None, None))
    if file_name:
        ch = _find_chunk(file_name, h2_id)
        if ch: return ch, {"source":"default","topic":topic,"exact_h2_match":bool(h2_id)}
    # запасной путь — первый чанк этой темы
    for ch in ALL_CHUNKS:
        if getattr(ch.metadata, "topic", None) == topic:
            return ch, {"source":"default-any","topic":topic,"exact_h2_match":False}
    return None, {}

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
        
        # Парсим H2-алиасы из HTML-комментариев
        h2_aliases = []
        h2_id = None
        
        # Ищем {#id} в заголовке
        id_match = re.search(r'\{#([^}]+)\}', h2_title)
        if id_match:
            h2_id = id_match.group(1)
            h2_title = re.sub(r'\s*\{#[^}]+\}\s*', '', h2_title).strip()
        
        # Ищем алиасы в первой строке после заголовка
        if lines and len(lines) > 1:
            first_line = lines[1].strip()
            alias_match = re.search(r'<!--\s*aliases:\s*\[(.*?)\]\s*-->', first_line, re.IGNORECASE)
            if alias_match:
                aliases_str = alias_match.group(1)
                # Парсим алиасы из строки
                h2_aliases = [a.strip().strip('"\'') for a in re.split(r',\s*', aliases_str) if a.strip()]
        
        # Если внутри есть Н3 режем по ним (### ...)
        h3_blocks = re.split(r'(?m)^\s*###\s+', block)
        if len(h3_blocks) > 1:
            # parts[0] — содержимое до первого ### <- ЭТО НУЖНО СОХРАНИТЬ!
            if h3_blocks[0].strip():
                preamble_text = f"## {h2_title}\n{h3_blocks[0].strip()}"
                chunk_id = f"{file_name}#{h2_title}_preamble"
                
                # Создаем метаданные с H2-алиасами
                temp_metadata = Frontmatter({
                    "h2_id": h2_id,
                    "h2_title": h2_title,
                    "h2_aliases": h2_aliases
                })
                
                # Алиасы только для поиска, не в тексте ответа
                index_text = preamble_text
                chunks.append(RetrievedChunk(chunk_id, index_text.strip(), temp_metadata, file_name))
            
            # дальше создать чанки для каждого ### как сейчас
            for h3 in h3_blocks[1:]:
                h3 = h3.strip()
                if not h3:
                    continue
                
                lines = h3.splitlines()
                h3_title = lines[0]
                h3_body = '\n'.join(lines[1:])
                
                text = f"## {h2_title}\n### {h3_title}\n{h3_body}"
                chunk_id = f"{file_name}#{h2_title}_{h3_title}"
                
                # Создаем метаданные с H2-алиасами
                temp_metadata = Frontmatter({
                    "h2_id": h2_id,
                    "h2_title": h2_title,
                    "h2_aliases": h2_aliases
                })
                
                # Алиасы только для поиска, не в тексте ответа
                index_text = text
                chunks.append(RetrievedChunk(chunk_id, index_text.strip(), temp_metadata, file_name))
        else:
            # Н3 нет - сохраняем Н2 как единый чанк
            text = f"## {h2_title}\n{h2_body}"
            chunk_id = f"{file_name}#{h2_title}"
            
            # Создаем метаданные с H2-алиасами
            temp_metadata = Frontmatter({
                "h2_id": h2_id,
                "h2_title": h2_title,
                "h2_aliases": h2_aliases
            })
            
            # Алиасы только для поиска, не в тексте ответа
            index_text = text
            chunks.append(RetrievedChunk(chunk_id, index_text.strip(), temp_metadata, file_name))
    
    return chunks

def extract_aliases_from_chunk(chunk_text: str) -> List[str]:
    """Извлекает алиасы из HTML-комментариев в чанке"""
    m = re.search(r'<!--\s*aliases:\s*\[(.*?)\]\s*-->', chunk_text, re.S)
    if not m:
        return []
    raw = m.group(1)
    pairs = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
    aliases = [a or b for a, b in pairs]
    return [a.strip() for a in aliases if a and a.strip()]

def update_entity_index(chunk: RetrievedChunk, topic: str, entity_key: str):
    """Обновляет ENTITY_INDEX с алиасами из чанка"""
    # Извлекаем заголовок ##
    header_match = re.search(r'(?m)^##\s+(.+?)\s*$', chunk.text)
    title = header_match.group(1).strip() if header_match else ""
    
    # Алиасы: HTML-коммент в тексте + фронтматтер
    aliases = extract_aliases_from_chunk(chunk.text)
    if hasattr(chunk, "metadata") and getattr(chunk.metadata, "aliases", None):
        aliases.extend(chunk.metadata.aliases)
    if title:
        aliases.append(title)
    
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

# Back-compat функция для старого кода
from core.answer_builder import postprocess, LOW_REL_JSON

def postprocess_answer_with_empathy(answer_text, user_text, intent, topic_meta, session):
    """Back-compat: используем новый постпроцессор."""
    return postprocess(answer_text, user_text, intent, topic_meta, session)

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
    
    for ch in ALL_CHUNKS:
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
    
    # Подключаем фильтр для служебных MD файлов
    from core.md_filter import is_index_like
    import json, logging
    log_m = logging.getLogger("cesi.minimal_logs")
    
    all_md_files = list(folder_path.rglob("*.md"))
    skipped = 0
    docs = []
    
    for file in all_md_files:
        try:
            print(f"📄 Обрабатываю файл: {file}")
            text = file.read_text(encoding="utf-8")
            
            # Фильтруем служебные файлы
            if is_index_like(file, text):
                skipped += 1
                print(f"  ⏭️ Пропускаем служебный файл: {file.name}")
                continue
            
            print(f"  📖 Прочитан файл: {file.name} ({len(text)} символов)")
            
            # Регистрируем файл в новых индексах
            register_file(file.name, text)
            
            # Парсим YAML front matter
            metadata, content = parse_yaml_front_matter(text)
            content = _normalize(content) # нормализуем пробелы
            print(f"  ✅ YAML парсинг: {metadata.id if metadata.id else 'без ID'}")
            
            # Регистрируем алиасы для fallback поиска
            try:
                from core.md_loader import register_aliases
                # Конвертируем metadata в dict для register_aliases
                frontmatter_dict = {
                    "aliases": getattr(metadata, 'aliases', []),
                    "primary_h2_id": getattr(metadata, 'primary_h2_id', None)
                }
                register_aliases(frontmatter_dict, str(file))
            except Exception as e:
                print(f"  ⚠️ Ошибка регистрации алиасов: {e}")
            
            # Локальная регистрация алиасов
            for a in (getattr(metadata, 'aliases', ()) or []):
                ALIAS_MAP[_norm(a)] = {"file": str(file), "primary_h2_id": getattr(metadata, 'primary_h2_id', None)}
            
            # если это файл с врачами - собрать имена
            if getattr(metadata, 'doc_type', '') in ('doctor', 'doctors') or file.name == "doctors.md":
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
                
                # Регистрируем все чанки в ENTITY_CHUNKS (не только импланты)
                ENTITY_CHUNKS[(chunk.metadata.topic or "general", chunk.id)] = chunk
                
                # Если это карточка врача (подзаголовок ## или ### Имя Фамилия [Отчество]) — запоминаем прямую ссылку
                hdr = re.search(r'(?m)^#{2,3}\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})\s*$', chunk.text)
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
    
    # Логируем статистику фильтрации
    log_m.info(json.dumps({"ev":"filter_index_like","skipped":skipped,"total":len(all_md_files)}, ensure_ascii=False))
    
    print(f"\u23f3 Найдено {len(all_chunks)} чанков")
    
    # Инициализируем ALL_CHUNKS для новых индексов
    ALL_CHUNKS.extend(all_chunks)
    print(f"✅ ALL_CHUNKS инициализирован: {len(ALL_CHUNKS)} чанков")
    print(f"✅ ALIAS_MAP_GLOBAL: {len(ALIAS_MAP_GLOBAL)} алиасов")
    print(f"✅ H2_INDEX: {len(H2_INDEX)} заголовков")
    print(f"✅ FILE_META: {len(FILE_META)} файлов")
    
    # Логируем статистику индексов
    print(f"📊 INDEX STATS: docs={len(FILE_META)}, chunks={len(ALL_CHUNKS)}, aliases={len(ALIAS_MAP_GLOBAL)}, h2s={len(H2_INDEX)}")
    
    # Создаем BM25 индекс
    if ALL_CHUNKS:
        print(f"🔍 Создаем BM25 индекс для {len(ALL_CHUNKS)} чанков...")
        bm25_corpus = []
        for chunk in ALL_CHUNKS:
            # Токенизируем текст + алиасы для BM25
            alias_boost = " ".join(getattr(chunk.metadata, 'h2_aliases', ()) or [])
            tokens = re.findall(r'\w+', (chunk.text + " " + alias_boost).lower())
            bm25_corpus.append(tokens)
        
        bm25_index = BM25Okapi(bm25_corpus)
        print(f"✅ BM25 индекс создан")
    
    # Отладочная информация о чанках
    for chunk in ALL_CHUNKS[:5]:  # Показываем первые 5 чанков
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
        
        # Создаем эмбеддинги: текст + алиасы для поиска
        chunk_texts = []
        for chunk in ALL_CHUNKS:
            alias_boost = " ".join(getattr(chunk.metadata, 'h2_aliases', ()) or [])
            chunk_texts.append((chunk.text + " " + alias_boost).strip())
        embeddings = [get_embedding(text) for text in chunk_texts]
        
        dimension = len(embeddings[0])
        xb = np.asarray(embeddings, dtype="float32")
        normalize_L2_inplace(xb)
        index = IndexFlatIP(dimension)
        index.add(xb)
        
        print(f"✅ Индекс создан с {len(ALL_CHUNKS)} чанками")
        
        # Логируем backend при старте
        from core.logger import log_m
        log_m.info(json.dumps({"event": "faiss_backend", "value": "faiss" if HAS_FAISS else "numpy"}, ensure_ascii=False))

except Exception as e:
    print(f"❌ Критическая ошибка при инициализации: {e}")
    import traceback
    traceback.print_exc()
    print("⚠️ Embeddings недоступны, работаем только на BM25 и правилах")
    index = None
    # нет FAISS, но чанки оставляем!

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
        q = np.asarray([query_embedding], dtype="float32")
        normalize_L2_inplace(q)
        D, I = index.search(q, min(top_n, len(all_chunks)))
        
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

def select_chunk_by_alias(chunks: List[RetrievedChunk], query: str) -> RetrievedChunk | None:
    """Форс-матч по алиасам перед финальным выбором чанка"""
    ql = query.lower()
    for chunk in chunks:
        h2_aliases = getattr(chunk.metadata, 'h2_aliases', []) or []
        for alias in h2_aliases:
            al = alias.lower()
            if al == ql or al in ql:
                return chunk
    return None

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



# === 6) Ранний детектор H2 ===
def detect_section_early(user_q: str):
    q = norm_text(user_q)
    hit = H2_INDEX.get(q)
    if not hit:
        # мягкое вхождение (минимум 3 символа для избежания ложных срабатываний)
        for k, v in H2_INDEX.items():
            if k and len(k) > 2 and k in q:
                hit = v; break
    if not hit: return None, {}
    ch = _find_chunk(hit["file"], hit["h2_id"])
    if not ch: return None, {}
    return ch, {"source":"alias","exact_h2_match":True,"topic":hit["topic"]}

# === 7) Главная функция ретрива ===
def retrieve_relevant_chunks_new(user_q: str, theme_hint: str|None, candidates_func):
    """Новая функция ретрива: подсказки (H2/тема) добавляются к кандидатам, а не прерывают поиск."""
    print(f"🔍 NEW ENGINE: query='{user_q}', theme_hint='{theme_hint}'")

    # Если явно разрешён старый fastpath — оставим прежнее поведение
    if os.getenv("DISABLE_ALIAS_FASTPATH", "true").lower() != "true":
        ch, flags = detect_section_early(user_q)
        if ch:
            print(f"✅ H2 match found (fastpath): {ch.id}")
            return [ch], flags
        if theme_hint in CANON:
            ch, flags = get_default_chunk_for_topic(theme_hint)
            if ch:
                print(f"✅ Theme match found (fastpath): {theme_hint} -> {ch.id}")
                return [ch], flags

    mixed: list[RetrievedChunk] = []

    # 1) точный H2 — добавляем к кандидатам
    ch, _flags = detect_section_early(user_q)
    if ch:
        print(f"✅ H2 hint: {ch.id}")
        mixed.append(ch)

    # 2) дефолт по теме — тоже добавляем
    if theme_hint in CANON:
        ch2, _flags2 = get_default_chunk_for_topic(theme_hint)
        if ch2:
            print(f"✅ Theme hint: {theme_hint} -> {ch2.id}")
            mixed.append(ch2)

    # 3) обычный поиск
    search_cands = candidates_func(user_q) or []
    print(f"🔍 Search candidates: {len(search_cands)} found")
    mixed.extend(search_cands)

    # 4) дедуп по chunk.id или file_name
    seen = set()
    out = []
    for c in mixed:
        cid = getattr(c, "id", None) or getattr(getattr(c, "meta", {}), "get", lambda k=None: None)("id")
        fid = getattr(c, "file_name", None)
        key = cid or fid
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    # 5) Реранкер: дать базовый score и реально пересортировать
    def _ensure_scores(cands):
        """Убеждаемся, что у каждого кандидата есть базовый score"""
        for i, c in enumerate(cands):
            score = getattr(c, "score", None) or getattr(c, "cosine", None) or getattr(c, "bm25", None)
            if score is None:
                score = 0.5  # базовый score
            c.score = float(score) - i * 1e-6  # стабильность порядка
    
    def _bonus_for_query(c, q: str, theme: str) -> float:
        """Бонус за релевантность к запросу и теме"""
        t = (getattr(c, "text", "") or "").lower()
        q = (q or "").lower()
        b = 0.0
        
        # прижив/оссео
        if "прижив" in q or "оссео" in q:
            if any(x in t for x in ["прижив", "оссеоинтегр"]): b += 0.10
        
        # боязнь/боль
        if any(x in q for x in ["боюсь", "боль", "анестез", "обезбол"]):
            if any(x in t for x in ["без боли", "анестез", "обезбол"]): b += 0.10
        
        # контакты
        if theme == "contacts":
            if any(x in t for x in ["адрес", "телефон", "график", "как добраться"]): b += 0.10
        
        # гарантия
        if theme == "warranty":
            if "гаранти" in t: b += 0.08
        
        # цены
        if theme == "prices":
            if any(x in t for x in ["цена", "стоимость", "сколько стоит", "рассрочка"]): b += 0.08
        
        return b
    
    _ensure_scores(out)
    for c in out:
        c.score += _bonus_for_query(c, user_q, theme_hint or "")
    out.sort(key=lambda x: x.score, reverse=True)

    return out, {"source": "mix", "exact_h2_match": False}

def retrieve_relevant_chunks(query: str, top_k: int = None) -> List[RetrievedChunk]:
    if top_k is None:
        top_k = int(os.getenv("RAG_TOP_K", 5))  # было 8, теперь 5 по умолчанию
    """Извлекает релевантные чанки с multi-query rewrite и улучшенным ранжированием"""
    if len(ALL_CHUNKS) == 0:
        print("⚠️ Нет чанков для поиска")
        return []
    
    # ==== РОУТЕР ПО ТЕМАМ ====
    detected_topics = route_topics(query)
    print(f"🎯 Роутер определил темы: {detected_topics}")
    
    # Прямой alias-fallback по карте frontmatter (ОТКЛЮЧЕН)
    # q = _norm(query or "")
    # hit_map = ALIAS_MAP.get(q)
    # if not hit_map:
    #     # допускаем "alias ⊆ query" (например, запрос длиннее)
    #     for a, meta in ALIAS_MAP.items():
    #         if a and a in q:
    #             hit_map = meta
    #             break
    #
    # if hit_map:
    #     # находим первый чанк соответствующего файла и возвращаем его
    #     target = Path(hit_map["file"]).name
    #     for ch in ALL_CHUNKS:
    #         if ch.file_name == target:
    #             print(f"✅ Alias-fallback: {q} -> {hit_map['file']}")
    #             return [ch]
    
    # ==== БЫСТРЫЙ ПУТЬ: ДЕТЕКТ СУЩНОСТИ ====
    q = _norm(query or "")
    hit = None
    
    # Прямое вхождение алиаса
    for alias, meta in ENTITY_INDEX.items():
        if alias in q:
            hit = ENTITY_CHUNKS.get((meta["topic"], meta["entity"]))
            if not hit:
                hit = next((ch for ch in all_chunks if ch.id == meta["entity"]), None) \
                    or next((ch for ch in all_chunks if ch.file_name == meta["doc_id"]), None)
            if hit:
                break
    
    # Мягкий матч (разделители -/-/- и пробелы)
    if not hit:
        q2 = re.sub(r'[\-–—]', '', q)   # дефис/тире
        for alias, meta in ENTITY_INDEX.items():
            a2 = re.sub(r'[\-–—]', '', alias)   # дефис/тире
            if a2 in q2:
                hit = ENTITY_CHUNKS.get((meta["topic"], meta["entity"]))
                if not hit:
                    hit = next((ch for ch in all_chunks if ch.id == meta["entity"]), None) \
                        or next((ch for ch in all_chunks if ch.file_name == meta["doc_id"]), None)
                if hit:
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
        
        # ==== ФОРС-МАТЧ ПО АЛИАСАМ ====
        forced_chunk = select_chunk_by_alias([chunk for chunk, _ in top_candidates], query)
        if forced_chunk:
            print(f"🎯 Форс-матч по алиасу: {forced_chunk.id}")
            final_chunks = [forced_chunk]
        else:
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

def _strip_md(s: str) -> str:
    """Очистка от Markdown разметки"""
    if not s:
        return ""
    s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL)
    s = re.sub(r'(?im)^\s*aliases\s*:\s*\[.*?\]\s*$', '', s)
    s = re.sub(r'(?m)^\s*#{1,6}\s*', '', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'__(.+?)__', r'\1', s)
    s = re.sub(r'(?<!\w)_(.*?)_(?!\w)', r'\1', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def synthesize_answer(chunks: List[RetrievedChunk], user_query: str, verbatim=False) -> dict:
    """Мини-синтез без LLM: склеиваем 2-3 самых релевантных фрагмента, очищаем Markdown."""
    
    # Подготовка контекста с очисткой и shaping
    from core.text_clean import clean_section_for_prompt
    from core.answer_shaping import should_verbatim, clamp_bullets, slice_for_prompt
    import json, logging
    log_m = logging.getLogger("cesi.minimal_logs")
    
    prepared_chunks, prepared_len, verbatim_used = [], 0, False
    for sec in chunks[:3]:  # берем только первые 3 чанка
        meta = getattr(sec, "meta", {}) or {}
        clean = clean_section_for_prompt(sec.text)
        if should_verbatim(meta):
            prepared = clamp_bullets(clean, max_bullets=6)
            verbatim_used = True
        else:
            prepared = slice_for_prompt(clean, limit_chars=900)
        prepared_chunks.append(prepared)
        prepared_len += len(prepared)
    
    log_m.info(json.dumps({"ev":"prepared_context","count":len(prepared_chunks),"prepared_len":prepared_len,"verbatim":verbatim_used}, ensure_ascii=False))
    
    # Используем подготовленные чанки вместо сырых текстов
    if verbatim or not prepared_chunks:
        return {"text": prepared_chunks[0] if prepared_chunks else ""}
    
    return {"text": "\n\n".join(prepared_chunks)}

def synthesize_answer_old(chunks: List[RetrievedChunk], user_query: str, allow_cta: bool) -> SynthJSON:
    """Синтезирует структурированный JSON ответ"""
    
    # Определяем метаданные для ответа (берем из первого чанка)
    primary_chunk = chunks[0] if chunks else None
    tone = primary_chunk.metadata.tone if primary_chunk else "friendly"
    preferred_format = primary_chunk.metadata.preferred_format if primary_chunk else ["short", "bullets", "cta"]
    verbatim = primary_chunk.metadata.verbatim if primary_chunk else False
    
    # Если verbatim: true - отдаем текст "как есть" без LLM
    if verbatim:
        print(f"📝 Verbatim режим: отдаем текст без LLM пересказа")
        
        # Извлекаем bullets из текста
        bullets = []
        for chunk in chunks:
            lines = chunk.text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    bullet = line[2:].strip()
                    if bullet:
                        bullets.append(bullet)
    
    # CTA для промпта (только если разрешен)
    cta_for_prompt = (primary_chunk.metadata.cta_text if (primary_chunk and allow_cta) else "")
    
    return {
            "short": "Информация по вашему запросу:",
            "bullets": bullets,
            "cta": cta_for_prompt if cta_for_prompt else None,
            "used_chunks": [chunk.id for chunk in chunks],
            "tone": tone,
            "warnings": []
        }
    
    # Формируем контекст из чанков для LLM
    from core.text_clean import clean_section_for_prompt
    from core.answer_shaping import should_verbatim, clamp_bullets, slice_for_prompt
    import json, logging
    log_m = logging.getLogger("cesi.minimal_logs")
    
    context_parts = []
    total_before, total_after = 0, 0
    cleaned = []
    prepared_chunks = []
    
    for chunk in chunks:
        t = chunk.text
        total_before += len(t)
        c = clean_section_for_prompt(t)
        total_after += len(c)
        cleaned.append(c)
        
        # Применяем shaping для ответов
        meta = getattr(chunk, "meta", {}) or {}
        if should_verbatim(meta):
            prepared = clamp_bullets(c, max_bullets=6)
        else:
            prepared = slice_for_prompt(c, limit_chars=900)
        prepared_chunks.append(prepared)
        
        context_parts.append(f"ID: {chunk.id}\nОбновлено: {chunk.updated}\nКритичность: {chunk.criticality}\nКонтент:\n{prepared}\n")
    
    context = "\n---\n".join(context_parts)
    
    # Логируем статистику очистки
    log_m.info(json.dumps({"ev":"clean_section","orig_len":total_before,"clean_len":total_after,"reduced":total_before-total_after}, ensure_ascii=False))
    print(f"📝 Контекст для синтеза: {len(context)} символов")
    print(f"📝 Первые 200 символов контекста: {context[:200]}...")
    
    # CTA для промпта (только если разрешен)
    cta_for_prompt = (primary_chunk.metadata.cta_text if (primary_chunk and allow_cta) else "")
    
    # Адаптивная логика для verbatim: false файлов
    format_instruction = ""
    if "detailed" in preferred_format:
        format_instruction = "\n\nВАЖНО: Дай развернутый, подробный ответ с максимальным количеством деталей и объяснений. Пользователь нуждается в полной информации."
    elif "short" in preferred_format:
        format_instruction = "\n\nВАЖНО: Дай краткий, лаконичный ответ по сути вопроса. Без лишних деталей."
    
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
8. В конце при возможности добавь мягкий CTA с предложением консультации.{format_instruction}

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
{f"- CTA текст: {cta_for_prompt}" if cta_for_prompt else ""}

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

    # Жёсткий JSON-контракт с правилами краткости
    SYSTEM_RULES = """
Ты ассистент клиники. Отвечай спокойно и по делу.
Только ЧИСТЫЙ JSON (без Markdown-фенсов).

Запрещено:
- оглавления, перечисление тем, H2/H3;
- поля фронтматтера (title, aliases, mini_links, verbatim) и HTML-комментарии;
- длинные копипасты из базы;
- символы '#', '*', '•' и заголовки вида "### ..." в тексте;
- повторение заголовка документа в тексте.

Если STYLE=compact: не используй #, -, •, списки и заголовки; 3–6 коротких фраз; заверши точкой.
Если STYLE=list: до N пунктов, каждый 1 предложением; без заголовков; маркер не выводи — мы форматируем сами.

Схема:
{
 "answer": "string, ≤700 символов, без оглавлений и списков тем",
 "empathy": "string или ''",
 "cta": {"show": true|false, "variant": "consult"},
 "followups": [{"label":"string","query":"string"}]
}
Если уверен на <65%, задай 1 уточняющий вопрос в поле answer и не показывай CTA.
"""

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,  # низкая температура для стабильности
        max_tokens=220,   # ограничиваем длину
        top_p=0.8,       # ограничиваем выбор токенов
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": system_prompt}
        ]
    )
    
    try:
        # Логируем финиш модели
        finish = completion.choices[0].finish_reason  # "stop" | "length" | ...
        c_tokens = getattr(completion.usage, "completion_tokens", None)
        log_m.info(json.dumps({"ev":"llm_finish","finish":finish,"completion_tokens":c_tokens,"max_tokens":220}, ensure_ascii=False))
        
        json_response = json.loads(completion.choices[0].message.content)
        
        # Валидация и обрезка ответа
        answer = json_response.get("answer", "Извините, давайте уточню…")
        if len(answer) > 700:
            answer = answer[:700] + "..."
        
        empathy = json_response.get("empathy", "")
        cta = json_response.get("cta", {"show": False, "variant": "consult"})
        if not allow_cta:
            cta = {"show": False, "variant": "consult"}
        
        followups = json_response.get("followups", [])
        
        return SynthJSON(
            short=answer,
            bullets=[],  # Убираем bullets в пользу краткого answer
            cta=cta.get("show", False),
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
        # Проверяем, что text - это строка
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
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
    short = clean_text(synth.get("short", ""))
    cta = clean_text(synth.get("cta", ""))
    
    # Форматируем bullets
    bullets = []
    for bullet in synth.get("bullets", []):
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
    from logging import getLogger
    import json
    logger = getLogger("cesi.rag")
    logger.info("➡️ Новый вопрос: %s", user_message)
    
    # Логируем запрос в RAG
    from core.logger import log_query
    log_query(user_message, "rag_engine")
    
    # Всегда инициализируем — чтобы не ловить UnboundLocalError
    detected_topics = []
    theme_hint = None
    rag_meta = {"user_query": user_message}
    
    try:
        # --- 1) Явный оверрайд темы по ключевым словам (если используешь root_aliases.yaml)
        try:
            import yaml
            ROOT = yaml.safe_load(open("config/root_aliases.yaml", "r", encoding="utf-8")) or {}
            ql = user_message.lower()
            for doc_type, keys in (ROOT.get("root_aliases") or {}).items():
                if any(k in ql for k in keys):
                    theme_hint = doc_type
                    break
        except Exception:
            pass
    
        # --- 2) Авто-детект темы (если оверрайд не дал тему)
        if theme_hint is None:
            try:
                detected_topics = list(route_topics(user_message))
                print(f"🎯 Роутер определил темы: {detected_topics}")
            except Exception:
                detected_topics = []
            
            if detected_topics:
                theme_hint = detected_topics[0]

        # --- 2.5) Быстрый путь по алиасам (ВРЕМЕННО ОТКЛЮЧЕН)
        # q_norm = norm_text(user_message)
        # if q_norm in ALIAS_MAP_GLOBAL:
        #     key = ALIAS_MAP_GLOBAL[q_norm]
        #     # найдём чанк по файлу и/или якорю Н2
        #     ch = _find_chunk(key["file"], key.get("primary_h2_id"))
        #     if ch:
        #         topic_meta = getattr(ch.metadata, '__dict__', {}) or {}
        #         payload = postprocess(
        #             answer_text=ch.text,
        #             user_text=user_message,
        #             intent=theme_hint or key.get("topic"),
        #             topic_meta=topic_meta,
        #             session={},
        #         )
        #         return payload, {"user_query": user_message, "theme_hint": theme_hint, "fast_path": "alias"}

        # --- 3) Ретривал (2 прохода: с темой → без темы)
        # Увеличиваем top_k для поиска врачей
        if DOCTOR_REGEX and DOCTOR_REGEX.search(user_message):
            top_k = 12
            print(f"🔍 Поиск врачей: используем top_k={top_k}")
        else:
            top_k = int(os.getenv("RAG_TOP_K", 5))  # было 8, теперь 5 по умолчанию
        
        # Извлекаем релевантные чанки (используем новую логику)
        logger.info("🔎 theme_hint=%s detected_topics=%s", theme_hint, detected_topics)
        relevant_chunks, meta_flags = retrieve_relevant_chunks_new(
            user_message, 
            theme_hint=theme_hint,
            candidates_func=lambda q: retrieve_relevant_chunks(q, top_k=top_k)
        )
        
        # Мини-фильтр по теме (минимум логики, максимум эффекта)
        def filter_candidates(theme: str, q: str, cands: list):
            t = (q or "").lower()
            
            def h2_of(c):
                return (getattr(c, "h2", None) or getattr(c, "meta", {}).get("h2", "") or "").lower()
            
            def text_of(c):
                return (getattr(c, "text", "") or "").lower()
            
            # «приживаемость» — оставляем только куски, где явно есть прижив/оссео
            if any(k in t for k in ["прижив", "приживаем", "оссео"]):
                return [c for c in cands if any(x in h2_of(c) + " " + text_of(c) for x in ["прижив", "оссеоинтегр"])]
            
            # страх/боль — выкидываем «противопоказания»
            if any(k in t for k in ["боюсь", "боль", "страшно", "анестез", "обезбол"]):
                bad = ["противопоказан", "противопоказания"]
                return [c for c in cands if not any(x in h2_of(c) for x in bad)]
            
            return cands
        
        relevant_chunks = filter_candidates(theme_hint, user_message, relevant_chunks)
        
        # Лёгкий переранж (чтобы «нужное» всплывало первым)
        def bonus_for_query(c, q):
            t = (getattr(c, "text", "") or "").lower()
            b = 0.0
            
            # прижив/оссео
            if "прижив" in q or "оссео" in q:
                if any(x in t for x in ["прижив", "оссеоинтегр"]): b += 0.1
            
            # боязнь/боль
            if any(x in q for x in ["боюсь", "боль", "анестез", "обезбол"]):
                if any(x in t for x in ["без боли", "анестез", "обезбол"]): b += 0.1
            
            return b
        
        for c in relevant_chunks:
            base = getattr(c, "score", 0.0)  # твоя косинус/БМ25
            c.score = float(base) + bonus_for_query(c, user_message.lower())
        
        relevant_chunks = sorted(relevant_chunks, key=lambda x: x.score, reverse=True)
        
        # Быстрая диагностика (чтобы понять причину)
        for i, c in enumerate(relevant_chunks[:5], 1):
            logger.info({
                "ev": "rerank_top",
                "rank": i,
                "score": round(getattr(c, "score", 0.0), 3),
                "h2": getattr(c, "h2", None) or getattr(c, "meta", {}).get("h2"),
                "doc": getattr(c, "meta", {}).get("doc_id"),
            })
        
        # Если нет кандидатов, пробуем без темы
        if not relevant_chunks:
            logger.info("⚠️ Первый проход не дал результатов, пробуем без темы...")
            relevant_chunks, meta_flags = retrieve_relevant_chunks_new(
                user_message, 
                theme_hint=None,
                candidates_func=lambda q: retrieve_relevant_chunks(q, top_k=top_k)
            )
        
        # Логируем результат
        logger.info("📦 candidates=%d", len(relevant_chunks))
        logger.info("🔍 rag_engine: relevant_chunks[0] type=%s", type(relevant_chunks[0]) if relevant_chunks else "None")
        
        if not relevant_chunks:
            # честный низкий релеванс
            return LOW_REL_JSON.copy(), rag_meta

        # --- 4) Синтез ответа из нескольких чанков
        synth = synthesize_answer(relevant_chunks, user_message)
        answer_text = synth.get("text", "")

        # --- 4.5) Guard-fallback для «боль/страх»
        cand_cnt = len(relevant_chunks)
        if cand_cnt == 0:
            return LOW_REL_JSON.copy(), {"meta": {"relevance_score": 0.0, "cand_cnt": 0}}

        best = relevant_chunks[0]
        best_score = getattr(best, "score", 0.0)
        
        # Если после фильтра/переранжировки лучший скор слабый — отдай готовый «эмпатичный» ответ
        if theme_hint in ("safety", "pain", "fear_pain") and best_score < 0.42:
            payload = {
                "response": {"text": "Понимаю, что страшит именно боль. Процедура делается под местной анестезией – во время операции вы ничего не почувствуете.\n\nПосле – обычно как после удаления зуба: даём обезболивающее и сопровождаем.\n\nЕсли хотите, врач коротко расскажет, как всё проходит именно в вашем случае."},
                "cta": {"label": "Записаться на консультацию", "target": "whatsapp"},
                "followups": ["Сколько длится приём?", "Какая анестезия используется?"]
            }
            return payload, {**rag_meta, "guard_used": True, "relevance_score": best_score}
        
        score = None
        
        # Пытаемся извлечь score из чанка
        for key in ["score", "rank_score", "bm25_score", "similarity"]:
            if isinstance(best, dict) and key in best:
                score = best[key]
                break
            elif hasattr(best, key):
                score = getattr(best, key)
                break
        
        # Если score нет - ставим разумный дефолт
        if score is None:
            score = 0.7  # если нет score - даем безопасный дефолт
        
        # Если это форс-совпадение по алиасу - повышаем score
        if isinstance(best, dict) and (best.get("forced_by_alias") or best.get("h2_alias_hit")):
            score = max(score, 0.9)  # если это был форс-алиас (если у вас есть такой флаг)

        # --- 5) Постпроцесс: эмпатия/бридж + CTA
        best_chunk = relevant_chunks[0] if relevant_chunks else None
        topic_meta = getattr(best_chunk.metadata, '__dict__', {}) or {} if best_chunk else {}
        intent = theme_hint  # или твой интент-детектор
        payload = postprocess(
            answer_text=answer_text,
            user_text=user_message,
            intent=intent,
            topic_meta=topic_meta,
            session={},  # session пока пустой
        )
        
        # Подстраховка для payload
        if not isinstance(payload, dict):
            payload = {"text": str(payload) if payload is not None else best_text}
        elif not payload.get("text"):
            payload["text"] = best_text

        # Обновляем метаданные с relevance_score
        def _mget(meta, key, default=None):
            """Безопасный доступ к метаданным: dict/obj"""
            if meta is None:
                return default
            if isinstance(meta, dict):
                return meta.get(key, default)
            return getattr(meta, key, default)
        
        def _score_of(c):
            """Пытаемся получить скор из разных полей; если нет - None"""
            for k in ("score", "rank_score", "bm25_score", "similarity"):
                v = c[k] if isinstance(c, dict) and k in c else getattr(c, k, None)
                if v is not None:
                    return v
            return None
        
        cand_cnt = len(relevant_chunks)
        best = relevant_chunks[0]
        meta = getattr(best, "metadata", None) or getattr(best, "meta", {}) or {}
        best_text = getattr(best, "section_text", None) or getattr(best, "text", "") or ""
        # минимальная очистка прямо тут (та же, что в clean_response_text, но короткая)
        import re
        best_text = re.sub(r'<!--.*?>', '', best_text, flags=re.DOTALL)
        best_text = re.sub(r'^\s*#{1,6}\s*', '', best_text, flags=re.MULTILINE)
        best_text = best_text.strip()
        rag_meta["best_text"] = best_text
        score = _score_of(best)
        if score is None:
            score = 0.7  # дефолт, чтобы guard не резал нормальный ответ
        best_chunk_id = f"{_mget(meta, 'file_basename','')}" + (f"#{_mget(meta, 'h2_id','')}" if _mget(meta, 'h2_id') else "")
        
        # ВАЖНО: список словарей {"chunk": ..., "score": ... } – именно так ждёт адаптер/арр.ру
        candidates_with_scores = [
            {"chunk": c, "score": _score_of(c)}
            for c in relevant_chunks[:5]
        ]
        
        # rag_meta плоские поля (+ вложенный meta для guard - на всякий случай)
        rag_meta = {
            "relevance_score": float(score),
            "cand_cnt": cand_cnt,
            "theme_hint": theme_hint,
            "detected_topics": detected_topics,
            "best_chunk_id": best_chunk_id,
            "candidates_with_scores": candidates_with_scores,
            "best_text": best_text,  # пригодится адаптеру как fallback
            "meta": {
                "relevance_score": float(score),
                "cand_cnt": cand_cnt,
                "doc_type": _mget(meta, "doc_type"),
                "topic": _mget(meta, "topic"),
                "best_chunk_id": best_chunk_id,
            }
        }

        # Логируем ответ в RAG
        from core.logger import format_candidates_for_log
        from logging import getLogger
        log_m = getLogger("cesi.minimal_logs")
        try:
            log_m.info(json.dumps({
                "ev":"search_candidates",
                "count": len(relevant_chunks),
                "cands": format_candidates_for_log(relevant_chunks, top=3)
            }, ensure_ascii=False))
        except Exception as e:
            logging.getLogger("cesi").warning(f"log_format_error: {e}")
        
        return payload, rag_meta
        
    except Exception:
        # Никогда не падаем наружу — лог и безопасный ответ
        logger.exception("💥 get_rag_answer failed")
        return LOW_REL_JSON.copy(), rag_meta

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
        "has_cta": bool(metadata.get("cta_action")),
        "topic": metadata.get("topic", "unknown"),
        "doc_type": metadata.get("doc_type", "unknown"),
        # Поля эмпатии для отладки (старая логика - заменена на новую)
        # "emotion": metadata.get("emotion", "none"),
        # "emotion_source": metadata.get("detector", "unknown"),
        # "emotion_confidence": metadata.get("confidence", 0.0),
        # "opener_used": metadata.get("opener_used", False),
        # "closer_used": metadata.get("closer_used", False)
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


