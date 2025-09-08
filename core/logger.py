from __future__ import annotations
import os, sys, json, logging
from pathlib import Path
from datetime import datetime

# База и папка логов — абсолютные пути (без относительных сюрпризов)
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.msg
        if isinstance(msg, (dict, list)):
            try:
                return json.dumps(msg, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ev":"log_format_error","err":str(e),"raw":str(msg)}, ensure_ascii=False)
        # строку не оборачиваем повторно
        return str(msg)

def _add_json_file_logger(name: str, filename: str, level=logging.INFO) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False
    # чистим дубли handler'ов
    lg.handlers = []
    fh = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(JsonLineFormatter())
    lg.addHandler(fh)
    return lg

def init_logging(console: bool = True, level: int = logging.INFO) -> None:
    root = logging.getLogger("cesi")
    if getattr(root, "_inited", False):
        return  # уже настроено

    root.setLevel(level)
    root.propagate = False
    # сбрасываем старые хендлеры (на случай «вольного» кода)
    root.handlers = []

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(ch)

    # ошибки — обычный текстовый лог
    err = logging.FileHandler(LOG_DIR / "errors.log", encoding="utf-8")
    err.setLevel(logging.ERROR)
    err.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(err)

    # JSONL-файлы
    _add_json_file_logger("cesi.minimal_logs", "minimal_logs.jsonl", level)
    _add_json_file_logger("cesi.bot_responses", "bot_responses.jsonl", level)

    root._inited = True
    root.info(f"[logging] initialized -> {LOG_DIR}")

def self_test() -> bool:
    # Пишем по одной строке во все каналы
    now = datetime.utcnow().isoformat() + "Z"
    for name in ("cesi", "cesi.minimal_logs", "cesi.bot_responses"):
        logging.getLogger(name).info({"ev":"log_self_test","logger":name,"ts":now})
    return True

# ---- Backward-compat exports ----
import logging
log_m = logging.getLogger("cesi.minimal_logs")     # как раньше
log_resp = logging.getLogger("cesi.bot_responses") # если где-то используется
log = logging.getLogger("cesi")                    # корневой

def log_query(q, meta=None):
    """Backward compatibility for log_query"""
    log_m.info(json.dumps({"ev": "query", "q": q, "meta": meta or {}}, ensure_ascii=False))

# ---- utils for RAG logging (backward-compat) ----
def _field(obj, *names):
    """Достаёт поле из объекта/словаря, поддерживает разные имена."""
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj.get(n)
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def format_candidates_for_log(candidates, top: int = 3):
    """
    Приводит кандидатов (dict или объект типа RetrievedChunk) к краткому виду для логов.
    Возвращает список словарей с ключами: ev, rank, score, h2, doc.
    """
    out = []
    if not candidates:
        return out
    # безопасно берём первые top
    seq = list(candidates)[:top]
    for i, c in enumerate(seq, 1):
        rank  = _field(c, "rank") or i
        score = _field(c, "score", "rank_score") or 0
        h2    = _field(c, "h2", "title")
        doc   = _field(c, "doc", "doc_name", "source")
        try:
            score = float(score)
        except Exception:
            pass
        out.append({"ev": "rerank_top", "rank": rank, "score": score, "h2": h2, "doc": doc})
    return out