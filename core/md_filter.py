from pathlib import Path
import re, yaml

ALIASES = {"alias.md","aliases.md","_aliases.md","index.md","_index.md"}

CONTACT_SIGNS = re.compile(r'(тел|whatsapp|адрес|ул\.|просп\.|пн|вт|ср|чт|пт|сб|вс|\+?\d[\d\-\s\(\)]{6,})', re.I)
FACT_SIGNS    = re.compile(r'(\d+\s?(?:₽|%|год|лет|мес|дн|мин))', re.I)

def _frontmatter(text: str):
    m = re.match(r'^---\n(.*?)\n---\n', text or "", flags=re.S)
    return yaml.safe_load(m.group(1)) if m else {}

def looks_like_fact_sheet(text: str) -> bool:
    t = (text or "").strip()
    if CONTACT_SIGNS.search(t): return True
    if FACT_SIGNS.search(t):    return True
    return False

def is_index_like(path: Path, text: str) -> bool:
    """
    Возвращает True, если файл следует считать «мусорным индексом» и НЕ индексировать.
    Логика:
      - если doc_type один из факт-типов — индексируем всегда;
      - иначе отбрасываем только ОЧЕНЬ короткие тексты (<80) без сигналов фактов;
      - короткие 80..120 оставляем, если видим факты (цифры/₽/телефон/адрес).
    """
    name = path.name.lower()
    if name in ALIASES:
        return True
    if text.count("<!-- aliases:") >= 2:
        return True
    
    t = (text or "")
    fm = _frontmatter(t) or {}
    doc_type = str(fm.get("doc_type", "")).lower()
    keep_types = {"contacts","prices","warranty","safety","doctors","consultation"}
    if doc_type in keep_types:
        return False  # индексируем, даже если короткие

    stripped = t.strip()
    n = len(stripped)
    ABS_MIN = 80
    SHORT   = 120

    if n < ABS_MIN:
        # Совсем короткие считаем индексом, кроме явных факт-листков
        return not looks_like_fact_sheet(stripped)
    if n < SHORT:
        # Пограничные: оставляем, если видим факты
        return not looks_like_fact_sheet(stripped)
    # Нормальные тексты — индексируем
    return False

