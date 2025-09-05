#!/usr/bin/env python3
"""
Утилита для упаковки логов - извлекает ключевые поля для анализа
Вариант В: минимальные логи с ключевыми полями
"""

import json
import os

SRC = "logs/queries.jsonl"
DST = "logs/minimal_logs.jsonl"
MAX = 150

def trim(s, n=500):
    """Обрезает строку до n символов"""
    if isinstance(s, str) and len(s) <= n:
        return s
    return s[:n] + "..." if isinstance(s, str) else str(s)[:n] + "..."

rows = []

if os.path.exists(SRC):
    with open(SRC, "r", encoding="utf-8") as f:
        lines = f.readlines()[-MAX:]  # Берем последние MAX записей
    
    for ln in lines:
        try:
            d = json.loads(ln)
        except:
            continue
        
        m = d.get("metadata", {})
        
        # Извлекаем кандидатов до реранкинга
        candidates = m.get("candidates", {})
        before = []
        for k in ("bm25", "dense", "hybrid"):
            if candidates.get(k):
                before = candidates[k][:5]
                break
        
        # Извлекаем результаты реранкинга
        rerank = m.get("rerank", [])[:5]
        
        out = {
            "ts": d.get("timestamp"),
            "user_query": d.get("question"),
            "primary_chunk": {
                "file": m.get("primary_file"),
                "title": m.get("primary_title"),
                "doc_type": m.get("primary_doc_type"),
                "verbatim": m.get("primary_verbatim")
            },
            "candidates_before_rerank": [
                {
                    "file": x.get("file"),
                    "title": x.get("title"),
                    "doc_type": x.get("doc_type"),
                    "bm25": x.get("bm25"),
                    "dense": x.get("dense")
                }
                for x in (before or [])
            ],
            "rerank_scores": [
                {
                    "file": x.get("file"),
                    "title": x.get("title"),
                    "score_llm": x.get("score_llm"),
                    "score_final": x.get("score_final")
                }
                for x in (rerank or [])
            ],
            # Поля эмпатии
            "emotion": m.get("emotion"),
            "emotion_source": m.get("emotion_source") or m.get("detector"),
            "emotion_confidence": m.get("emotion_confidence") or m.get("confidence"),
            "opener_used": m.get("opener_used"),
            "closer_used": m.get("closer_used"),
            "features": {
                "empathy_enabled": m.get("empathy_enabled"),
                "postprocess_enabled": m.get("postprocess_enabled"),
                "cta_enabled": m.get("cta_enabled")
            },
            "cta_text": m.get("cta_text"),
            "cta_link": m.get("cta_link"),
            "final_len": len(d.get("answer") or ""),
            "final_text": trim(d.get("answer") or "")
        }
        
        rows.append(out)

# Записываем результат
with open(DST, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Сохранено {len(rows)} записей в {DST}")
