from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# грузим .env вне зависимости от текущей рабочей директории
env_loaded = load_dotenv(find_dotenv(), override=True)
if not env_loaded:  # fallback: .env рядом с app.py
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from core.logger import init_logging, self_test, LOG_DIR
init_logging(console=True)

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import uuid
import traceback
import re
import json
from openai import OpenAI

# приглушим болтливость werkzeug
import logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logger = logging.getLogger("cesi")
logger.info("✅ Логирование настроено (level=%s)", os.getenv("LOG_LEVEL", "INFO"))

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from collections import defaultdict


def wants_json(req):
    """Определяет, хочет ли фронтенд JSON ответ"""
    # фронт может прислать флаг или header Accept
    q = (req.args.get("format") or req.headers.get("X-Response-Mode", "")).lower()
    if q == "json":
        return True
    return "application/json" in req.headers.get("Accept", "").lower()

from rag_engine import get_rag_answer
from datetime import datetime, timezone, timedelta, time

# Глобальные константы
MAX_FINAL_CHARS = int(os.getenv("MAX_FINAL_CHARS", "800"))

# Отладка ENV переменных
print("DEBUG RESPONSE_MODE =", os.getenv("RESPONSE_MODE"))
print("DEBUG ENABLE_EMPATHY =", os.getenv("ENABLE_EMPATHY"))

# Инициализация OpenAI клиента v1
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
    print("✅ OpenAI клиент v1 успешно инициализирован в app.py")
except Exception as e:
    print(f"❌ Ошибка инициализации OpenAI клиента в app.py: {e}")
    openai_client = None

app = Flask(__name__)
CORS(app, origins=['https://dental41.ru', 'http://dental41.ru', 'https://dental-bot.ru', 'http://dental-bot.ru', 'https://dental-chat.ru', 'http://dental-chat.ru'])

session_messages = defaultdict(list)
session_states = defaultdict(dict)

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))


def check_rate_limit(session_id):
    """Проверяет rate-limit на заявки (не чаще 1/5 минут на session_id)"""
    now = datetime.now()
    last = session_states[session_id].get("last_lead_ts")
    
    if last and (now - last).seconds < 300:  # 5 минут = 300 секунд
        return False
    
    session_states[session_id]["last_lead_ts"] = now
    return True

def send_lead_email(name, phone):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = "\U0001F4E5 Новая заявка с чат-бота ЦЭСИ"

    body = f"Имя: {name}\nТелефон: {phone}"
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email успешно отправлен")
    except Exception as e:
        print("❌ Ошибка при отправке email:", e)


def is_clinic_open():
    """Проверяет, открыта ли клиника в данный момент"""
    # Камчатский край: GMT+12
    kamchatka_tz = timezone(timedelta(hours=12))
    now = datetime.now(kamchatka_tz)
    
    # День недели (0=понедельник, 6=воскресенье)
    weekday = now.weekday()
    current_time = now.time()
    
    # Воскресенье - выходной
    if weekday == 6:
        return False
    
    # Понедельник-Пятница: 8:00-20:00
    if 0 <= weekday <= 4:
        return time(8, 0) <= current_time <= time(20, 0)
    
    # Суббота: 8:00-14:00
    if weekday == 5:
        return time(8, 0) <= current_time <= time(14, 0)
    
    return False


# ==== КОНСТРУКТОР CTA ====
def build_cta(meta) -> dict | None:
    """Единый конструктор CTA кнопок"""
    if meta is None:
        meta = {}
    
    action = meta.get("cta_action")  # 'book', 'call', 'link'
    text = meta.get("cta_text") or "Записаться на консультацию"
    
    if action == "book":
        return {"type": "book", "text": text}
    elif action == "call" and meta.get("phone"):
        return {"type": "call", "text": text, "phone": meta["phone"]}
    elif action == "link" and meta.get("cta_link"):
        return {"type": "link", "text": text, "url": meta["cta_link"]}
    
    # fallback
    return {"type": "book", "text": "Записаться на консультацию"}





def detect_refusal(message):
    """Категоризированная детекция отказа"""
    # Нормализация
    msg = (message or "").strip().lower()
    
    # Явный отказ / пауза
    REFUSALS = {
        "hard": ["нет", "неинтересно", "отстаньте", "не надо", "удалите мои данные"],
        "soft": ["подумаю", "позже", "свяжусь сам", "не сейчас", "не сегодня", "пока не нужно", "сам свяжусь", "спасибо", "позже", "не сейчас"]
    }
    
    # Жесткий отказ
    if any(kw == msg or kw in msg for kw in REFUSALS["hard"]):
        return "hard"
    
    # Мягкий отказ
    if any(kw == msg or kw in msg for kw in REFUSALS["soft"]):
        return "soft"
    
    return None

def validate_phone(phone):
    """Простая валидация телефона (10+ цифр)"""
    digits = ''.join(filter(str.isdigit, phone))
    return len(digits) >= 10

def gpt_cancellation_check(message):
    """Детекция отказа от записи через GPT"""
    # Проверяем, что OpenAI клиент доступен
    if not openai_client:
        print("⚠️ OpenAI клиент недоступен, используем fallback")
        return False
    
    try:
        prompt = f"""
        Определи, хочет ли пользователь отказаться от записи на консультацию.
        
        Сообщение пользователя: "{message}"
        
        Ответь только "ДА" если это отказ, или "НЕТ" если это согласие или нейтральный ответ.
        
        Примеры отказов: "нет", "нет спасибо", "передумал", "не хочу", "отмена", "не буду"
        Примеры согласий: "да", "хочу", "запишите", "конечно", "имя Иван", "телефон 1234567890"
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        
        # Улучшенная проверка "ДА" с поддержкой пунктуации
        if answer.startswith("ДА"):
            return True
        return False
        
    except Exception as e:
        print(f"❌ Ошибка GPT-детекции отказов: {e}")
        return False

def is_name_valid(message):
    """
    Проверяет, что введено имя, а не вопрос или жалоба.
    """
    message = message.strip().lower()
    
    # Слишком длинное - не имя
    if len(message) > 50:
        return False
    
    # Содержит вопросительные слова - не имя
    question_words = ["что", "как", "где", "когда", "почему", "зачем", "сколько", "какой", "какая", "какие"]
    if any(word in message for word in question_words):
        return False
    
    # Содержит знаки препинания в середине - не имя
    if any(punct in message for punct in ["?", "!", ".", ",", ":", ";"]):
        return False
    
    # Содержит медицинские термины - не имя
    medical_terms = ["болит", "зуб", "имплант", "лечение", "врач", "клиника", "стоимость", "цена", "больно", "страшно"]
    if any(term in message for term in medical_terms):
        return False
    
    # Содержит только буквы, пробелы и дефисы - похоже на имя
    import re
    if re.match(r'^[а-яё\s\-]+$', message):
        return True
    
    return False

def is_cancellation(message):
    """Гибридная детекция отказа от записи"""
    # Сначала быстрая проверка очевидных случаев
    refusal_type = detect_refusal(message)
    if refusal_type:
        return refusal_type
    
    # Если не очевидно - используем GPT
    # LLM: True = мягкий отказ по умолчанию
    if gpt_cancellation_check(message):
        return "soft"
    
    return None


def format_contacts_answer(text: str) -> str:
    """
    Достаёт из сырого текста строки вида:
    'Адрес:', 'Время работы:', 'Телефон:', 'WhatsApp:', 'Парковка:'.
    Возвращает один компактный параграф без списков.
    """
    import re
    t = text or ""
    def grab(label):
        m = re.search(rf"{label}\s*:\s*(.+?)(?:[.;]|$)", t, flags=re.I)
        return (m.group(1).strip() if m else "")
    addr = grab("Адрес")
    time = grab("Время работы")
    tel  = grab("Телефон")
    wa   = grab("WhatsApp")
    park = grab("Парковка")
    parts = []
    if addr: parts.append(f"Адрес: {addr}")
    if time: parts.append(f"Время работы: {time}")
    if tel:  parts.append(f"Телефон: {tel}")
    if wa:   parts.append(f"WhatsApp: {wa}")
    if park: parts.append(f"Парковка: {park}")
    return "Адрес и контакты — " + "; ".join(parts) + "."


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        session_id = data.get('session_id') or str(uuid.uuid4())
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"response": "Пожалуйста, введите сообщение.", "session_id": session_id})

        session = session_states[session_id]

        # --- Этап сбора данных ---
        if session.get("state") == "ожидание_имени":
            # Проверяем отказ
            refusal_type = is_cancellation(message)
            if refusal_type == "hard":
                session.clear()
                return jsonify({
                    "response": "Хорошо, фиксирую. Если передумаете – я всегда на связи.",
                    "session_id": session_id
                })
            elif refusal_type == "soft":
                session.clear()
                return jsonify({
                    "response": "Понял, без спешки. Если появятся вопросы – подскажу.",
                    "session_id": session_id
                })
            
            # Проверяем, что введено имя, а не вопрос/жалоба
            if is_name_valid(message):
                session["имя"] = message
                session["state"] = "ожидание_телефона"
                return jsonify({
                    "response": f"Приятно познакомиться, {message}! А номер телефона подскажете?",
                    "session_id": session_id
                })
            else:
                # Если введен не вопрос, а что-то другое - переспрашиваем имя
                return jsonify({
                    "response": "Похоже, это не имя 🙂 Напишите, пожалуйста, как к вам обращаться (например: Иван Петров). Если пока неудобно — можем продолжить без записи.",
                    "session_id": session_id
                })

        if session.get("state") == "ожидание_телефона":
            # Проверяем отказ
            refusal_type = is_cancellation(message)
            if refusal_type == "hard":
                session.clear()
                return jsonify({
                    "response": "Хорошо, фиксирую. Если передумаете – я всегда на связи.",
                    "session_id": session_id
                })
            elif refusal_type == "soft":
                session.clear()
                return jsonify({
                    "response": "Понял, без спешки. Если появятся вопросы – подскажу.",
                    "session_id": session_id
                })
            
            name = session.get("имя", "Не указано")
            phone = message
            
            # Валидация телефона
            if not validate_phone(phone):
                return jsonify({
                    "response": "Пожалуйста, введите корректный номер телефона (минимум 10 цифр).",
                    "session_id": session_id
                })
            
            # Проверяем rate-limit на заявки
            if not check_rate_limit(session_id):
                return jsonify({
                    "response": "Заявка уже отправлена недавно. Мы скоро свяжемся.",
                    "session_id": session_id
                })
            
            # Проверяем время работы клиники
            if is_clinic_open():
                send_lead_email(name, phone)
                session.clear()
                return jsonify({
                    "response": "Спасибо! Мы передали ваши данные администратору \U0001F60A\n\nНаш специалист свяжется с вами в ближайшее время.",
                    "session_id": session_id
                })
            else:
                session.clear()
                return jsonify({
                    "response": "Спасибо за обращение! К сожалению, наша клиника сейчас закрыта. Мы свяжемся с вами в рабочее время.\n\nЧасы работы: Пн-Пт 8:00-20:00, Сб 8:00-14:00, Вс - выходной.",
                    "session_id": session_id
                })

        # --- Прямые команды на запись ---
        if message.lower() in ["записаться", "хочу записаться", "можно записаться", "запись", "консультация", "записаться на консультацию"]:
            session["state"] = "ожидание_имени"
            return jsonify({
                "response": "Отлично! Я помогу вам записаться на консультацию. Как вас зовут?",
                "session_id": session_id
            })

        # --- Обработка специальных действий ---
        if message.lower() in ["получить скидку на кт", "кт со скидкой", "скидка кт"]:
            return jsonify({
                "response": "Отлично! У нас есть специальное предложение для посетителей чата:\n\n📋 **КТ скидка 30%**\n\n• Промокод: «Чат»\n• Действует на все виды КТ\n• Скидка 30% от обычной цены\n\nЧтобы воспользоваться скидкой, просто назовите промокод «Чат» при записи на КТ.\n\nХотите записаться на КТ со скидкой?",
                "session_id": session_id,
                "cta": {
                    "text": "Записаться на КТ со скидкой",
                    "href": None,
                    "kind": "book",
                    "topic": "ct_discount",
                    "doctor": None
                }
            })

        # --- Обычная обработка через RAG ---
        # Импортируем новую систему
        # Импортируем оба адаптера
        from core import legacy_adapter
        from core.answer_builder import postprocess as json_adapter
        from config.feature_flags import feature_flags
        from core.rag_integration import enhance_rag_retrieval
        from core.router import theme_router
        from core.normalize import normalize_ru
        
        # 1. Нормализация запроса
        normalized_query = normalize_ru(message)
        
        # 2. Тематический роутинг
        from core.router import route_theme
        theme_hint = route_theme(message)  # Используем новый роутер
        
        # Логируем запрос
        from core.logger import log_query
        log_query(message, session_id)
        
        # 3. Получаем ответ от RAG
        rag_payload, rag_meta = get_rag_answer(normalized_query)
        
        # 4. Усиливаем кандидатов тематикой (если есть кандидаты)
        if rag_meta.get("candidates_with_scores"):
            from core.rag_integration import enhance_rag_retrieval, apply_jump_boost
            
            # Применяем jump boost если есть jump данные
            jump_data = request.json.get("jump") if request.is_json else None
            if jump_data:
                rag_meta["candidates_with_scores"] = apply_jump_boost(
                    rag_meta["candidates_with_scores"], 
                    jump_data
                )
            
            enhanced_chunks, enh_meta = enhance_rag_retrieval(
                normalized_query,
                lambda q: rag_meta.get("candidates_with_scores", []),
                theme_hint=theme_hint
            )
            rag_meta["candidates_with_scores"] = enhanced_chunks
            rag_meta.update(enh_meta or {})
        
        # 5. Адаптируем к новой системе (включает guard, followups, empathy)
        # ✅ Извлекаем настоящие чанки из candidates_with_scores (с предохранителем)
        raw = rag_meta.get("candidates_with_scores", [])
        norm = []
        for it in raw:
            if isinstance(it, dict) and "chunk" in it:
                norm.append((it["chunk"], it.get("score")))
            elif isinstance(it, (list, tuple)) and len(it) >= 1:
                norm.append((it[0], it[1] if len(it) > 1 else None))
            else:
                norm.append((it, None))
        
        chunks_only = [c for c, _ in norm]
        
        # Условная финализация ответа
        if feature_flags.is_json_mode():
            # Если rag_engine уже вернул готовый JSON — просто используем его
            if isinstance(rag_payload, dict) and any(k in rag_payload for k in ("answer", "text", "final_text", "followups")):
                adapted_meta = rag_payload
            else:
                # Иначе делаем постпроцесс один раз
                adapted_meta = json_adapter(
                    answer_text=(rag_payload.get("text") if isinstance(rag_payload, dict) else str(rag_payload or "")),
                    user_text=message,
                    intent=rag_meta.get("theme_hint"),
                    topic_meta=rag_meta.get("meta", {}),
                    session=session
                )
        else:
            # Legacy режим
            adapted_response, adapted_meta = legacy_adapter.adapt_rag_response(
                user_query=message,
                rag_response=rag_payload,
                rag_meta=rag_meta,
                relevant_chunks=chunks_only
            )
        
        # Определяем формат ответа
        # 1) если режим JSON -> ВСЕГДА отдаём JSON (независимо от Accept)
        if feature_flags.is_json_mode():
            # Нормализация payload (фикс для 'str'.get)
            import re
            
            def _clean_text(s: str) -> str:
                if not s: return ""
                # 1) убрать HTML-комменты и aliases: [...]
                s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL)
                s = re.sub(r'(?im)^\s*aliases\s*:\s*\[.*?\]\s*$', '', s)
                # 2) убрать ### и ## в начале строк
                s = re.sub(r'(?m)^\s*#{2,3}\s*', '', s)
                # 2.1) снять жир/курсив (**...** / __...__)
                s = re.sub(r'(\*\*|_)(.*?)\1', r'\2', s)
                s = re.sub(r'(?<!\w)_(.*?)_(?!\w)', r'\1', s)
                # 2.2) нормализовать маркеры списков (•, *) в '- '
                s = re.sub(r'(?m)^\s*[•*]\s+', '- ', s)
                # 3) схлопнуть лишние пустые строки
                s = re.sub(r'\n{3,}', '\n\n', s)
                # 4) убрать дубликаты первой строки
                lines = [ln.strip() for ln in s.splitlines()]
                if len(lines) >= 2 and lines[0] and lines[0] == lines[1]:
                    lines.pop(1)
                return "\n".join(lines).strip()
            
            def _as_payload(obj):
                """Приводит любой ответ к единому JSON-формату."""
                if isinstance(obj, dict): return obj
                if isinstance(obj, str):
                    t = _clean_text(obj)
                    return {"response": {"text": t}, "text": t}
                # на всякий случай – всё остальное в строку
                t = _clean_text(str(obj))
                return {"response": {"text": t}, "text": t}
            
            # Нормализуем то, что пришло из RAG
            adapted_meta = _as_payload(rag_payload)
            
            # Берём финальный текст из postprocess: response.text
            final_text = (
                (adapted_meta.get("response") or {}).get("text")
                or adapted_meta.get("final_text")
                or adapted_meta.get("text")
                or ""
            )
            final_text = _clean_text(final_text)
            
            # Ограничиваем длину ответа - убрано, управляется через finalize_answer()
            
            # Собираем чистый ответ
            resp = {
                "response": final_text,  # строка
                "text": final_text,
                "cta": adapted_meta.get("cta") or None,
                "followups": adapted_meta.get("followups") or [],
            }
            
            # Нормализуем ответ к строгому JSON-контракту
            def coerce_output(d: dict) -> dict:
                ans = d.get("answer") or d.get("text") or d.get("response") or ""
                emp = d.get("empathy") or d.get("opener") or d.get("opening") or ""
                cta = d.get("cta") or {}
                if isinstance(cta, bool): cta = {"show": bool(cta), "variant":"consult"}
                if "show" not in cta: cta = {"show": False, "variant":"consult"}
                fu = d.get("followups") or d.get("suggestions") or []
                return {
                    "answer": (ans or "").strip(),
                    "empathy": (emp or "").strip(),
                    "cta": cta,
                    "followups": fu
                }
            
            # Применяем нормализацию
            out = coerce_output(resp)
            
            # Умная обрезка и санитизация с лимитами по темам
            from core.text_utils import finalize_answer, finalize_empathy
            from core.answer_style import decide_style
            
            theme = (rag_meta.get("theme_hint") or "").lower()
            style = decide_style(theme)
            
            # Специальная обработка для контактов
            if theme == "contacts":
                raw_answer = out.get("answer","")
                formatted_answer = format_contacts_answer(raw_answer)
                ans, cut_a = finalize_answer(
                    formatted_answer,
                    limit=style.max_chars,
                    allow_ellipsis=style.allow_ellipsis,
                    allow_bullets=False,  # контакты всегда параграф
                    max_bullets=0,
                )
            else:
                ans, cut_a = finalize_answer(
                    out.get("answer",""),
                    limit=style.max_chars,
                    allow_ellipsis=style.allow_ellipsis,
                    allow_bullets=(style.mode=="list"),
                    max_bullets=style.max_bullets,
                )
            emp, cut_e = finalize_empathy(out.get("empathy",""), limit=200)
            out["answer"], out["empathy"] = ans, emp
            
            # если резали список — предложим раскрыть:
            if cut_a and style.mode=="list":
                fu = out.get("followups") or []
                if not any("полный" in (x.get("label","")+x.get("query","")).lower() for x in fu):
                    fu.append({"label":"Показать полный список","query":"покажите полный список по теме"})
                out["followups"] = fu
            
            # Проверяем что ответ не пустой
            if not out.get("answer"):
                # если вдруг где-то по дороге обнулили - честный каркас вместо фолбэка фронта
                from core.answer_builder import LOW_REL_JSON
                return jsonify(LOW_REL_JSON), 200
            
            # Логируем финальный ответ
            import logging, json
            log_m = logging.getLogger("cesi.minimal_logs")
            log_m.info(json.dumps({"ev":"final_answer","len": len(out.get("answer","")), "cta": bool(out.get("cta",{}).get("show")), "truncated": bool(cut_a)}, ensure_ascii=False))
            
            # Логируем ответ бота
            log_resp = logging.getLogger("cesi.bot_responses")
            log_resp.info(json.dumps({"ev":"bot_response","q": message, "answer": out.get("answer",""), "empathy": out.get("empathy",""), "cta": out.get("cta",{}), "followups": out.get("followups",[])}, ensure_ascii=False))
            
            # Возвращаем нормализованный ответ
            return jsonify(out), 200
        else:
            # Legacy формат
            # Строим CTA для совместимости
            cta = build_cta(rag_meta)
            
            # Rate-limit: не спамить (90-120 секунд)
            now = datetime.now()
            throttled = False
            if cta and session.get("last_cta_ts") and (now - session["last_cta_ts"]).seconds < 90:
                cta = None
                throttled = True
            if cta:
                session["last_cta_ts"] = now
            
            # Логируем запрос для анализа (после троттлинга CTA)
            # Старые логи удалены - используется только final_answer
            cta_final = cta if cta and not throttled else None
            
            session_messages[session_id].append({"role": "user", "content": message})
            session_messages[session_id].append({"role": "assistant", "content": adapted_response})
            
            # Конвертируем в legacy текст
            from core.answer_builder import to_legacy_text
            legacy_text = to_legacy_text(adapted_meta) if isinstance(adapted_meta, dict) else str(adapted_response)
            
            # ✅ Несгораемый фолбэк: не отдаём пустые ответы
            if not legacy_text or not legacy_text.strip():
                legacy_text = (
                    (adapted_meta.get("answer", {}) or {}).get("short")  # если есть короткий ответ
                    or (str(adapted_response) if adapted_response else "")  # или сырой ответ
                    or "Сейчас не могу ответить по базе. Сформулируйте вопрос чуть иначе, пожалуйста. "
                    "Могу также предложить записать вас на консультацию."
                )
            
            # ШАГ 1: ВСЕГДА отдаём JSON (даже в legacy режиме)
            payload = {"response": legacy_text}
            if cta_final:
                payload["cta"] = cta_final
            
            # Нормализуем Legacy ответ к строгому JSON-контракту
            out_legacy = coerce_output(payload)
            
            # Умная обрезка и санитизация с лимитами по темам
            from core.text_utils import finalize_answer, finalize_empathy
            from core.answer_style import decide_style
            
            theme_legacy = (rag_meta.get("theme_hint") or "").lower()
            style_legacy = decide_style(theme_legacy)
            
            # Специальная обработка для контактов
            if theme_legacy == "contacts":
                raw_answer_legacy = out_legacy.get("answer","")
                formatted_answer_legacy = format_contacts_answer(raw_answer_legacy)
                ans_legacy, cut_a_legacy = finalize_answer(
                    formatted_answer_legacy,
                    limit=style_legacy.max_chars,
                    allow_ellipsis=style_legacy.allow_ellipsis,
                    allow_bullets=False,  # контакты всегда параграф
                    max_bullets=0,
                )
            else:
                ans_legacy, cut_a_legacy = finalize_answer(
                    out_legacy.get("answer",""),
                    limit=style_legacy.max_chars,
                    allow_ellipsis=style_legacy.allow_ellipsis,
                    allow_bullets=(style_legacy.mode=="list"),
                    max_bullets=style_legacy.max_bullets,
                )
            emp_legacy, cut_e_legacy = finalize_empathy(out_legacy.get("empathy",""), limit=200)
            out_legacy["answer"], out_legacy["empathy"] = ans_legacy, emp_legacy
            
            # если резали список — предложим раскрыть:
            if cut_a_legacy and style_legacy.mode=="list":
                fu_legacy = out_legacy.get("followups") or []
                if not any("полный" in (x.get("label","")+x.get("query","")).lower() for x in fu_legacy):
                    fu_legacy.append({"label":"Показать полный список","query":"покажите полный список по теме"})
                out_legacy["followups"] = fu_legacy
            
            # Логируем финальный ответ
            import logging, json
            log_m = logging.getLogger("cesi.minimal_logs")
            log_m.info(json.dumps({"ev":"final_answer","len": len(out_legacy.get("answer","")), "cta": bool(out_legacy.get("cta",{}).get("show")), "truncated": bool(cut_a_legacy)}, ensure_ascii=False))
            
            # Логируем ответ бота
            log_resp = logging.getLogger("cesi.bot_responses")
            log_resp.info(json.dumps({"ev":"bot_response","q": message, "answer": out_legacy.get("answer",""), "empathy": out_legacy.get("empathy",""), "cta": out_legacy.get("cta",{}), "followups": out_legacy.get("followups",[])}, ensure_ascii=False))
            
            return jsonify(out_legacy), 200
            
    except Exception as e:
        print(f"Ошибка обработки сообщения: {e}")
        traceback.print_exc()
        
        # Логируем ошибку в файл
        import os
        os.makedirs("logs", exist_ok=True)
        with open("logs/errors.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}: {str(e)}\n{traceback.format_exc()}\n\n")
        
        # ШАГ 1: В except тоже отдаём JSON
        safe_text = "Сейчас не могу ответить. Напишите вопрос чуть короче или позвоните нам."
        return jsonify({"response": safe_text}), 200


@app.route('/health', methods=['GET'])
def health():
    return "ok", 200

@app.route('/admin/log-self-test', methods=['GET'])
def log_self_test():
    ok = self_test()
    return {"ok": ok, "log_dir": str(LOG_DIR)}


@app.route('/submit-lead', methods=['POST'])
def submit_lead():
    try:
        data = request.json
        name = data.get('name', '')
        phone = data.get('phone', '')
        source = data.get('source', 'widget')
        session_id = data.get('session_id', 'direct')
        
        # Проверяем отказ (если есть сообщение)
        message = data.get('message', '')
        if message:
            refusal_type = is_cancellation(message)
            if refusal_type:
                return jsonify({
                    "success": False, 
                    "message": "Понял, отменяю заявку." if refusal_type == "hard" else "Понял, без спешки."
                })
        
        if name and phone:
            # Проверяем rate-limit
            if not check_rate_limit(session_id):
                return jsonify({
                    "success": False, 
                    "message": "Заявка уже отправлена недавно. Мы скоро свяжемся."
                })
            
            try:
                send_lead_email(name, phone)
                return jsonify({"success": True, "message": "Заявка успешно отправлена!"})
            except Exception as e:
                print(f"Ошибка отправки заявки: {e}")
                return jsonify({"success": False, "message": "Ошибка отправки заявки"})
        else:
            return jsonify({"success": False, "message": "Не указаны имя или телефон"})
    
    except Exception as e:
        # Логируем ошибку
        import traceback
        error_msg = f"Ошибка в submit_lead: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Записываем в лог ошибок
        try:
            with open("logs/errors.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - submit_lead error: {error_msg}\n")
        except:
            pass
        
        # Возвращаем безопасный ответ
        return jsonify({"success": False, "message": "Произошла ошибка. Попробуйте позже."})


@app.route('/')
def root():
    return send_file('static/tester.html')


@app.route('/widget')
def widget():
    return jsonify(ok=True, app="cesi-bot", message="Widget endpoint")


@app.route('/widget-embed.js')
def widget_embed():
    return send_file('widget-embed.js')


@app.route('/demo')
def demo():
    return jsonify(ok=True, app="cesi-bot", message="Demo endpoint")


@app.route('/test-connection.html')
def test_connection():
    return jsonify(ok=True, app="cesi-bot", message="Test connection endpoint")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)