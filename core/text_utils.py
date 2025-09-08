import re

SENT_ENDS = ".!?…"
SENT_END_RE = re.compile(r'([.!?…])\s+')
HDR_RE  = re.compile(r'^\s{0,3}#{1,6}\s*', re.M)
BUL_RE  = re.compile(r'^\s*[-•*]\s*', re.M)
CONF_RE = re.compile(r'(?m)^[A-Z_]{2,}\s*=\s*\S+\s*$')  # DISABLE_...=true

# Аббревиатуры с точкой (не считаем их концом предложения)
ABBR_WITH_DOT = ("г.", "ул.", "д.", "стр.", "к.", "корп.", "кв.", "оф.", "см.", "т.к.", "т.е.")

def prettify_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    
    # убрать любые ### заголовки и одиночные «голые» заголовки
    t = re.sub(r'(?m)^\s*#{1,6}\s*.*$', '', t)  # ### Заголовок
    t = re.sub(r'(?m)^(?:[>\s\-•\*]+)?[А-ЯA-Z][^.!?\n]{2,80}$\n?', '', t)  # Голые заголовки
    
    # убрать дублирующийся заголовок в начале: "Фраза … Фраза …"
    t = re.sub(r'^(?P<x>[А-ЯA-Z][^.!?\n]{3,80})\s+(?P=x)\b', r'\g<x>', t)

    # маркдауны в начало строки превращаем в « — »
    t = re.sub(r'(?m)^\s*[-•\*]\s+', ' — ', t)

    # убрать двойные тире, --- и прочий мусор
    t = re.sub(r'[—-]{2,}', '—', t)  # двойные тире
    t = re.sub(r'---+', '', t)  # горизонтальные линии

    # переносы строк -> пробел
    t = re.sub(r'[ \t]*\n[ \t]*', ' ', t)

    # косметика: " — ." -> ".", " — …" -> "…"
    t = re.sub(r'\s*[—-]\s*[.]+', '.', t)
    t = re.sub(r'\s*[—-]\s*…', '…', t)

    # двойные пробелы и финальная очистка
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t

def _last_safe_break(txt: str) -> int:
    borders = ".!?…;—"
    i = len(txt) - 1
    while i >= 0:
        ch = txt[i]
        if ch in borders:
            tail = txt[max(0, i - 6):i+1].lower()
            if any(tail.endswith(abbr) for abbr in ABBR_WITH_DOT):  # "ул.", "г." и т.п.
                i -= 1; continue
            return i + 1
        i -= 1
    return -1

def sanitize_answer(s: str) -> str:
    s = HDR_RE.sub('', s)
    s = BUL_RE.sub('', s)
    s = CONF_RE.sub('', s)
    s = re.sub(r'#{2,6}\s*', '', s)  # на всякий случай, не только в начале строки
    # Убираем markdown форматирование
    s = re.sub(r'\*\*(.*?)\*\*', r'\1', s)  # **жирный** -> жирный
    s = re.sub(r'__(.*?)__', r'\1', s)      # __курсив__ -> курсив
    s = re.sub(r'\*(.*?)\*', r'\1', s)      # *курсив* -> курсив
    s = re.sub(r'_(.*?)_', r'\1', s)        # _курсив_ -> курсив
    # Превращаем списки в обычные фразы
    s = re.sub(r'\n[-•*]\s+', '; ', s)      # \n- пункт -> ; пункт
    s = re.sub(r'\s{2,}', ' ', s)
    s = re.sub(r'\s*—\s*—\s*', ' — ', s)
    return s.strip(' \n\t-•*')

def smart_truncate(s: str, limit: int) -> tuple[str, bool]:
    s = s.strip()
    if len(s) <= limit:
        return s, False
    cut = s[:limit]
    # 1) пробуем оборвать по последней точке/воскл./вопросу
    pos = max(cut.rfind(ch) for ch in SENT_ENDS)
    if pos >= 0:
        out = cut[:pos+1].rstrip()
    else:
        # 2) иначе по последнему пробелу (чтобы не резать слово)
        sp = cut.rfind(' ')
        out = (cut[:sp] if sp > 0 else cut).rstrip()
        if not out or out[-1] not in SENT_ENDS:
            out += '…'
    return out, True

def finalize_answer(text: str, limit: int = 700, allow_ellipsis: bool = True,
                    allow_bullets: bool = False, max_bullets: int = 0) -> tuple[str,bool]:
    """
    Возвращает (текст, truncated). Если allow_bullets=True — сохраняем до max_bullets пунктов,
    мягко обрезая по границам пунктов/предложений.
    """
    t = prettify_answer(text)  # твоя текущая предчистка (сохрани!)
    if not t: return "", False

    # 1) Если разрешены списки — нормализуем маркдауны в пункты
    if allow_bullets:
        # превращаем маркдаун-строки в пункты, режем по max_bullets
        # не сноси дефисы внутри слов и аббревиатуры
        items = []
        for raw in t.splitlines():
            s = raw.strip()
            if not s: continue
            # поймаем маркер пункта
            if s.startswith(("- ","• ","* ")):
                s = s[2:].strip()
            # одна короткая фраза, точка в конце
            if not s.endswith((".", "!", "?", "…")):
                s += "."
            items.append(s)
        if items:
            items = items[:max_bullets]
            out = "\n".join(f"— {x}" for x in items)
            truncated = len("\n".join(items)) < len(t)
            # мягкая усадка по символам
            if len(out) > limit:
                # обрежем последний пункт по предложению/пробелу и добавим «…» только если это не CTA-тема
                cut = out[:limit]
                last_nl = cut.rfind("\n")
                if last_nl > 0:
                    out = cut[:last_nl].rstrip()
                else:
                    sp = cut.rfind(" ")
                    out = (cut[:sp] if sp>0 else cut).rstrip()
                if allow_ellipsis:
                    out += "…"
                else:
                    if out and out[-1] not in ".!?": out += "."
                truncated = True
            return out, truncated
        # если явных пунктов нет — падаем вниз к параграфной логике

    # 2) Параграфная логика (как у тебя сейчас, с _last_safe_break/ABBR)
    if len(t) <= limit:
        end = _last_safe_break(t)
        out = t if end == -1 else t[:end].rstrip()
        return out, False

    cut = t[:limit]
    end = _last_safe_break(cut)
    if end == -1:
        sp = cut.rfind(" ", max(0, len(cut) - 30))
        end = sp if sp != -1 else len(cut)
    out = cut[:end].rstrip()

    # если обрезали на тире/дефисе — убираем его
    out = re.sub(r'[—-]\s*$', '', out).rstrip()

    truncated = len(out) < len(t)
    if not truncated:
        return out, False

    # закрытие хвоста:
    if out and out[-1] not in ".!?…":
        if allow_ellipsis: 
            out += "…"
        else: 
            out += "."
    return out, truncated

def finalize_empathy(s: str, limit: int = 220) -> tuple[str, bool]:
    s = sanitize_answer(s)
    return smart_truncate(s, limit)
