import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import uuid
import traceback
import re
import openai

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from collections import defaultdict

from rag_engine import get_rag_answer
from datetime import datetime, timezone, timedelta, time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app, origins=['https://dental41.ru', 'http://dental41.ru', 'https://dental-bot.ru', 'http://dental-bot.ru', 'https://dental-chat.ru', 'http://dental-chat.ru'])

session_messages = defaultdict(list)
session_states = defaultdict(dict)

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))


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





def simple_cancellation_check(message):
    """Простая детекция очевидных отказов"""
    cancel_phrases = [
        "нет, спасибо", "нет спасибо", "передумал", "отмена", "не нужно", 
        "не хочу", "не буду", "откажусь", "не надо", "не буду записываться",
        "передумал записываться", "отменяю", "не хочу записываться", "спасибо нет"
    ]
    message_lower = message.lower()
    
    # Проверяем точные фразы
    if any(phrase in message_lower for phrase in cancel_phrases):
        return True
    
    # Проверяем комбинации ключевых слов
    words = message_lower.split()
    if "нет" in words and "спасибо" in words:
        return True
    if "не" in words and ("хочу" in words or "нужно" in words or "буду" in words):
        return True
    
    # Проверяем простое "нет" (только если это единственное слово)
    if message_lower.strip() == "нет":
        return True
    
    return False

def gpt_cancellation_check(message):
    """Детекция отказа от записи через GPT"""
    try:
        prompt = f"""
        Определи, хочет ли пользователь отказаться от записи на консультацию.
        
        Сообщение пользователя: "{message}"
        
        Ответь только "ДА" если это отказ, или "НЕТ" если это согласие или нейтральный ответ.
        
        Примеры отказов: "нет", "нет спасибо", "передумал", "не хочу", "отмена", "не буду", "абракадабра", "фывфыв", "12345", "qwerty"
        Примеры согласий: "да", "хочу", "запишите", "конечно", "имя Иван", "телефон 1234567890"
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer == "ДА"
        
    except Exception as e:
        print("Ошибка GPT-детекции отказов:", e)
        return False

def is_cancellation(message):
    """Гибридная детекция отказа от записи"""
    # Сначала быстрая проверка очевидных случаев
    if simple_cancellation_check(message):
        return True
    
    # Если не очевидно - используем GPT
    return gpt_cancellation_check(message)


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
        if is_cancellation(message):
            session.clear()
            return jsonify({
                "response": "Поняла, отменяю запись.\n\nМожет быть, у вас есть вопросы по лечению или имплантации? Также могу предложить вам скидку 30% на КТ по промокоду «Чат».\n\nВыберите тему или напишите свой вопрос:",
                "session_id": session_id,
                "action_buttons": [
                    {"text": "Узнать про имплантацию", "action": "Узнать про имплантацию"},
                    {"text": "КТ со скидкой", "action": "Получить скидку на КТ"}
                ]
            })
        
        session["имя"] = message
        session["state"] = "ожидание_телефона"
        return jsonify({
            "response": f"Приятно познакомиться, {message}! А номер телефона подскажете?",
            "session_id": session_id
        })

    if session.get("state") == "ожидание_телефона":
        # Проверяем отказ
        if is_cancellation(message):
            session.clear()
            return jsonify({
                "response": "Поняла, отменяю запись.\n\nМожет быть, у вас есть вопросы по лечению или имплантации? Также могу предложить вам скидку 30% на КТ по промокоду «Чат».\n\nВыберите тему или напишите свой вопрос:",
                "session_id": session_id,
                "action_buttons": [
                    {"text": "Узнать про имплантацию", "action": "Узнать про имплантацию"},
                    {"text": "КТ со скидкой", "action": "Получить скидку на КТ"}
                ]
            })
        
        name = session.get("имя", "Не указано")
        phone = message
        
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
        session_messages[session_id].append({"role": "user", "content": message})
        session_messages[session_id].append({"role": "assistant", "content": response})
        
        # Строим CTA
        cta = build_cta(rag_meta)
        
        # Rate-limit: не спамить (90-120 секунд)
        now = datetime.now()
        if cta and session.get("last_cta_ts") and (now - session["last_cta_ts"]).seconds < 90:
            cta = None
        if cta:
            session["last_cta_ts"] = now
        
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
    
    if name and phone:
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