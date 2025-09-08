import re

SENT_ENDS = ".!?…"
SENT_END_RE = re.compile(r'([.!?…])\s+')
HDR_RE  = re.compile(r'^\s{0,3}#{1,6}\s*', re.M)
BUL_RE  = re.compile(r'^\s*[-•*]\s*', re.M)
CONF_RE = re.compile(r'(?m)^[A-Z_]{2,}\s*=\s*\S+\s*$')  # DISABLE_...=true

def sanitize_answer(s: str) -> str:
    s = HDR_RE.sub('', s)
    s = BUL_RE.sub('', s)
    s = CONF_RE.sub('', s)
    s = re.sub(r'#{2,6}\s*', '', s)  # на всякий случай, не только в начале строки
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

def finalize_answer(s: str, limit: int = 700) -> tuple[str, bool]:
    s = sanitize_answer(s)
    out, truncated = smart_truncate(s, limit)
    
    # Защита хвоста: если модель закончила без знака конца предложения — аккуратно закрыть
    END = ('.','!','?','…')
    if out and out[-1] not in END:
        # если в последних ~120 символах нет точки/воскл./вопроса — добавим многоточие
        tail = out[-120:]
        if not any(ch in tail for ch in END):
            out = out.rstrip() + '…'
            truncated = True
    
    return out, truncated

def finalize_empathy(s: str, limit: int = 220) -> tuple[str, bool]:
    s = sanitize_answer(s)
    return smart_truncate(s, limit)
