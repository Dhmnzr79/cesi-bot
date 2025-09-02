import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import uuid
import traceback
import re
from openai import OpenAI

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from collections import defaultdict

from rag_engine import get_rag_answer
from datetime import datetime, timezone, timedelta, time

load_dotenv()

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


@app.route('/chat', methods=['POST'])
def chat():
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
        
        session["имя"] = message
        session["state"] = "ожидание_телефона"
        return jsonify({
            "response": f"Приятно познакомиться, {message}! А номер телефона подскажете?",
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
    try:
        response, rag_meta = get_rag_answer(message)
        
        # Строим CTA
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
        try:
            from rag_engine import log_query_response
            # Берем used_chunks из метаданных
            used_chunks = rag_meta.get("used_chunks", [])
            # Логируем shown_cta (какая CTA реально показана после троттлинга)
            cta_final = cta if cta and not throttled else None
            log_query_response(message, response, {**rag_meta, "shown_cta": bool(cta_final)}, used_chunks)
        except Exception as e:
            print(f"⚠️ Ошибка логирования: {e}")
        
        session_messages[session_id].append({"role": "user", "content": message})
        session_messages[session_id].append({"role": "assistant", "content": response})
        
        return jsonify({
            "response": response,
            "cta": cta,
            "session_id": session_id
        })
    except Exception as e:
        print(f"Ошибка обработки сообщения: {e}")
        traceback.print_exc()
        return jsonify({
            "response": "Извините, произошла ошибка. Попробуйте еще раз или позвоните нам по телефону +7(4152) 44-24-24",
            "session_id": session_id
        })


@app.route('/submit-lead', methods=['POST'])
def submit_lead():
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


@app.route('/')
def root():
    return send_file('widget.html')


@app.route('/widget')
def widget():
    return send_file('widget.html')


@app.route('/widget-embed.js')
def widget_embed():
    return send_file('widget-embed.js')


@app.route('/demo')
def demo():
    return send_file('test-connection.html')


@app.route('/test-connection.html')
def test_connection():
    return send_file('test-connection.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)