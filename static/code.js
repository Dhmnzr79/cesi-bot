let sessionId = localStorage.getItem("cesi_session_id") || null;

const greeting = `Здравствуйте! 👋<br>Я ассистент клиники ЦЭСИ. Расскажу, как проходит лечение, имплантация, и отвечу на любые вопросы по стоматологии.<br>С чего начнём?`;
const topics = [
  { emoji: "💬", text: "Записаться на консультацию" },
  { emoji: "🦷", text: "Узнать про имплантацию" },
  { emoji: "📍", text: "Посмотреть цены" },
  { emoji: "😬", text: "Боюсь боли" }
];

window.addEventListener('DOMContentLoaded', () => {
  showGreetingWithTopics();
  document.getElementById("userInput").placeholder = "Введите свой вопрос...";
});

function showGreetingWithTopics() {
  const chat = document.getElementById("chat");
  chat.innerHTML = "";
  appendMessage("bot", greeting);
  const topicsBlock = document.createElement("div");
  topicsBlock.className = "topics-block";
  topics.forEach(topic => {
    const btn = document.createElement("button");
    btn.className = "topic-btn";
    btn.innerHTML = `${topic.emoji} ${topic.text}`;
    btn.onclick = () => {
      sendMessage(topic.text);
      topicsBlock.remove();
      document.getElementById("topics-hint").remove();
    };
    topicsBlock.appendChild(btn);
  });
  chat.appendChild(topicsBlock);
  // Пояснение под кнопками
  const hint = document.createElement("div");
  hint.className = "topics-hint";
  hint.id = "topics-hint";
  hint.textContent = "Или напишите свой вопрос в чате — я отвечу на любые темы!";
  chat.appendChild(hint);
}

async function sendMessage(msg = null) {
  const input = document.getElementById("userInput");
  const message = msg || input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  input.value = "";

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId })
    });

    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem("cesi_session_id", sessionId);

    appendMessage("bot", data.response);

    if (data.show_booking_button) {
      showLeadFormButton();
    }
  } catch (e) {
    appendMessage("bot", "Ошибка при отправке сообщения.");
  }
}

function appendMessage(role, text) {
  const chat = document.getElementById("chat");
  const msg = document.createElement("div");
  msg.className = role;
  msg.innerHTML = text;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function showLeadFormButton() {
  const chat = document.getElementById("chat");
  const existing = document.getElementById("leadButton");
  if (existing) return;
  const btn = document.createElement("button");
  btn.textContent = "Записаться";
  btn.className = "lead-button";
  btn.id = "leadButton";
  btn.onclick = () => {
    sendMessage("Хочу записаться");
    btn.remove();
  };
  chat.appendChild(btn);
}

document.getElementById("sendButton").addEventListener("click", () => sendMessage());
document.getElementById("userInput").addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});
