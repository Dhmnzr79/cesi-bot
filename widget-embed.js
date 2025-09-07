// Виджет-чат ЦЭСИ для встраивания на сайт
(function() {
    'use strict';
    
    // Автоматическое определение URL сервера
    function getServerUrl() {
        // Если мы на том же домене, что и сервер
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:5000';
        }
        
        // Для нового домена бота - используем его
        if (window.location.hostname === 'dental-bot.ru' || window.location.hostname === 'dental-chat.ru') {
            return window.location.protocol + '//' + window.location.hostname;
        }
        
        // Для dental41.ru - используем новый домен бота
        if (window.location.hostname === 'dental41.ru') {
            return 'https://dental-bot.ru'; // НОВЫЙ ДОМЕН
        }
        
        // Для других доменов - используем новый домен бота
        return 'https://dental-bot.ru'; // НОВЫЙ ДОМЕН
    }
    
         // Конфигурация виджета
     var config = {
         serverUrl: getServerUrl(),
         consultantName: 'Анна',
         consultantStatus: 'Консультант клиники ЦЭСИ',
         mobileButtonText: 'Задать вопрос онлайн',
         primaryColor: '#23BFCF',
         secondaryColor: '#ffffff',
         accentColor: '#23BFCF',
         headerColor: '#F7FAFD',
         messageColor: '#F2FAFD'
     };

    // Создание стилей
    function createStyles() {
        var style = document.createElement('style');
        style.textContent = `
            .cesi-widget-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 10000;
                font-family: 'Arial', sans-serif;
            }
            
                         .cesi-widget-button {
                 display: flex;
                 align-items: center;
                 background: #23BFCF;
                 color: #fff;
                 padding: 12px 16px;
                 border-radius: 25px;
                 cursor: pointer;
                 box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                 transition: all 0.3s ease;
                 min-width: 280px;
                 max-width: 320px;
                 animation: pulse 2s infinite;
             }
            
                         .cesi-widget-button:hover {
                 background: #12a0af;
                 transform: translateY(-2px);
                 box-shadow: 0 6px 20px rgba(0,0,0,0.2);
             }
            
            .cesi-consultant-photo {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                margin-right: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                background: ${config.primaryColor};
            }
            
            .cesi-consultant-photo img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                border-radius: 50%;
            }
            
            .cesi-consultant-info {
                flex: 1;
            }
            
            .cesi-consultant-name {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 2px;
            }
            
            .cesi-consultant-status {
                font-size: 12px;
                opacity: 0.9;
                line-height: 1.2;
            }
            
                         .cesi-mobile-button {
                 display: none;
                 position: fixed;
                 bottom: 20px;
                 right: 20px;
                 background: #23BFCF;
                 color: #fff;
                 padding: 12px 20px;
                 border-radius: 25px;
                 cursor: pointer;
                 box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                 font-size: 14px;
                 font-weight: bold;
                 z-index: 10000;
                 transition: all 0.3s ease;
             }
            
                         .cesi-mobile-button:hover {
                 background: #12a0af;
                 transform: translateY(-2px);
                 box-shadow: 0 6px 20px rgba(0,0,0,0.2);
             }
            
                         .cesi-chat-window {
                 position: fixed;
                 bottom: 80px;
                 right: 20px;
                 width: 380px;
                 height: 600px;
                 background: white;
                 border-radius: 10px;
                 box-shadow: 0 15px 40px rgba(0,0,0,0.25);
                 display: none;
                 flex-direction: column;
                 z-index: 10000;
                 overflow: hidden;
             }
            
            .cesi-chat-header {
                background: ${config.headerColor};
                color: #333;
                padding: 20px;
                display: flex;
                align-items: center;
            }
            
            .cesi-chat-header-photo {
                width: 45px;
                height: 45px;
                border-radius: 50%;
                margin-right: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                background: ${config.primaryColor};
            }
            
            .cesi-chat-header-photo img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                border-radius: 50%;
            }
            
            .cesi-chat-header-info h4 {
                margin: 0;
                font-size: 16px;
                font-weight: bold;
            }
            
            .cesi-chat-header-info p {
                margin: 0;
                font-size: 12px;
                opacity: 0.9;
            }
            
                         .cesi-chat-close {
                 margin-left: auto;
                 background: none;
                 border: none;
                 color: #333;
                 font-size: 24px;
                 cursor: pointer;
                 padding: 0;
                 width: 30px;
                 height: 30px;
                 border-radius: 50%;
                 transition: background 0.2s;
                 display: flex;
                 align-items: center;
                 justify-content: center;
                 font-weight: bold;
             }
             
             .cesi-chat-close:hover {
                 background: rgba(0,0,0,0.1);
             }
            
            .cesi-chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
                         .cesi-message {
                 margin-bottom: 15px;
                 display: flex;
                 animation: slideIn 0.3s ease-out;
             }
             
             @keyframes slideIn {
                 from {
                     opacity: 0;
                     transform: translateY(10px);
                 }
                 to {
                     opacity: 1;
                     transform: translateY(0);
                 }
             }
            
            .cesi-message.user {
                justify-content: flex-end;
            }
            
                         .cesi-message-content {
                 max-width: 85%;
                 padding: 12px 16px;
                 border-radius: 18px;
                 font-size: 14px;
                 line-height: 1.4;
                 position: relative;
                 white-space: pre-wrap;
                 font-style: normal;
                 text-align: left;
             }
            
            .cesi-message-content em, .cesi-message-content i {
                font-style: normal;
            }
            
            .cesi-message.bot .cesi-message-content {
                background: ${config.messageColor};
                color: #333;
                border-bottom-left-radius: 5px;
            }
            
            .cesi-message.user .cesi-message-content {
                background: ${config.primaryColor};
                color: white;
                border-bottom-right-radius: 5px;
            }
            
            .cesi-action-buttons {
                padding: 0 20px 15px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                transition: opacity 0.3s ease;
            }
            
            .cesi-action-buttons.hidden {
                opacity: 0;
                pointer-events: none;
                height: 0;
                padding: 0;
                margin: 0;
                overflow: hidden;
            }
            
                         .cesi-action-button {
                 background: #23BFCF !important;
                 color: #fff !important;
                 border: none !important;
                 padding: 12px 16px;
                 border-radius: 8px;
                 cursor: pointer;
                 font-size: 14px;
                 font-weight: 500;
                 display: flex;
                 align-items: center;
                 gap: 10px;
                 transition: all 0.2s;
             }
            
                         .cesi-action-button:hover {
                 background: #12a0af !important;
                 color: #fff !important;
                 transform: translateY(-1px);
             }
            
            .cesi-action-button-icon {
                width: 20px;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
            }
            
                         .cesi-booking-button {
                 background: #23BFCF !important;
                 color: #fff !important;
                 border: none !important;
                 padding: 12px 16px;
                 border-radius: 8px;
                 cursor: pointer;
                 font-size: 14px;
                 font-weight: 500;
                 transition: all 0.2s;
                 display: flex;
                 align-items: center;
                 gap: 10px;
             }
            
                         .cesi-booking-button:hover {
                 background: #12a0af !important;
                 color: #fff !important;
                 transform: translateY(-1px);
             }
            
            
            
            .cesi-chat-input {
                padding: 20px;
                border-top: 1px solid #eee;
                display: flex;
                align-items: center;
                gap: 0;
                background: white;
            }
            
            .cesi-chat-input input {
                flex: 1;
                border: 2px solid #e9ecef;
                border-radius: 8px 0 0 8px;
                padding: 12px 20px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.2s;
            }
            
                         .cesi-chat-input input:focus {
                 border-color: #23BFCF;
             }
            
                         .cesi-chat-input button {
                 background: #23BFCF;
                 color: #fff;
                 border: none;
                 border-radius: 0 8px 8px 0;
                 width: 45px;
                 height: 45px;
                 cursor: pointer;
                 display: flex;
                 align-items: center;
                 justify-content: center;
                 font-size: 18px;
                 transition: all 0.2s;
             }
            
                         .cesi-chat-input button:hover {
                 background: #12a0af;
                 transform: scale(1.05);
             }
            
            .cesi-lead-form {
                display: none;
                padding: 20px;
                background: #f8f9fa;
                border-top: 1px solid #eee;
            }
            
            .cesi-lead-form input {
                width: 100%;
                padding: 12px 16px;
                margin-bottom: 12px;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.2s;
            }
            
                         .cesi-lead-form input:focus {
                 border-color: #23BFCF;
                 outline: none;
             }
            
                         .cesi-lead-form button {
                 width: 100%;
                 background: #23BFCF;
                 color: #fff;
                 border: none;
                 padding: 12px 16px;
                 border-radius: 8px;
                 cursor: pointer;
                 font-size: 14px;
                 font-weight: bold;
                 transition: all 0.2s;
             }
            
                         .cesi-lead-form button:hover {
                 background: #12a0af;
                 transform: translateY(-1px);
                 box-shadow: 0 4px 12px rgba(0,0,0,0.15);
             }
            
            .cesi-typing-dots {
                display: flex;
                gap: 4px;
                align-items: center;
            }
            
            .cesi-typing-dots span {
                width: 8px;
                height: 8px;
                background: #ccc;
                border-radius: 50%;
                animation: cesi-typing 1.4s infinite ease-in-out;
            }
            
            .cesi-typing-dots span:nth-child(1) {
                animation-delay: -0.32s;
            }
            
            .cesi-typing-dots span:nth-child(2) {
                animation-delay: -0.16s;
            }
            
            @keyframes cesi-typing {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            @media (max-width: 768px) {
                .cesi-widget-button {
                    display: none;
                }
                
                .cesi-mobile-button {
                    display: block;
                }
                
                .cesi-chat-window {
                    width: calc(100vw - 40px);
                    height: calc(100vh - 120px);
                    bottom: 80px;
                    right: 20px;
                    left: 20px;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Создание виджета
    function createWidget() {
        var container = document.createElement('div');
        container.className = 'cesi-widget-container';
        
        // Десктопная кнопка
        var desktopButton = document.createElement('div');
        desktopButton.className = 'cesi-widget-button';
        desktopButton.innerHTML = `
            <div class="cesi-consultant-photo">
                <img src="https://dental41.ru/avatar.webp" alt="Анна">
            </div>
            <div class="cesi-consultant-info">
                <div class="cesi-consultant-name">${config.consultantName}</div>
                <div class="cesi-consultant-status">${config.consultantStatus}</div>
            </div>
        `;
        desktopButton.onclick = cesiToggleChat;
        
        // Мобильная кнопка
        var mobileButton = document.createElement('div');
        mobileButton.className = 'cesi-mobile-button';
        mobileButton.textContent = config.mobileButtonText;
        mobileButton.onclick = cesiToggleChat;
        
        // Окно чата
        var chatWindow = document.createElement('div');
        chatWindow.className = 'cesi-chat-window';
        chatWindow.innerHTML = `
            <div class="cesi-chat-header">
                <div class="cesi-chat-header-photo">
                    <img src="https://dental41.ru/avatar.webp" alt="Анна">
                </div>
                <div class="cesi-chat-header-info">
                    <h4>${config.consultantName}</h4>
                    <p>${config.consultantStatus}</p>
                </div>
                <button class="cesi-chat-close" onclick="cesiToggleChat()">×</button>
            </div>
            <div class="cesi-chat-messages"></div>
                                     <div class="cesi-action-buttons">
                <button class="cesi-action-button" onclick="cesiStartBookingFromAction()">
                    <span class="cesi-action-button-icon">📅</span>
                    Записаться на консультацию
                </button>
                <button class="cesi-action-button" onclick="cesiSendAction('Узнать про имплантацию')">
                    <span class="cesi-action-button-icon">🦷</span>
                    Узнать про имплантацию
                </button>
                                 <button class="cesi-action-button" onclick="cesiSendAction('Посмотреть цены')">
                     <span class="cesi-action-button-icon">📍</span>
                     Посмотреть цены
                 </button>
                 <div style="text-align: center; color: #666; font-size: 12px; margin-top: 10px;">
                     Или напишите свой вопрос в чате
                 </div>
            </div>
            <div class="cesi-chat-input">
                <input type="text" placeholder="Введите свой вопрос..." onkeypress="cesiHandleKeyPress(event)">
                <button onclick="cesiSendMessage()">✈️</button>
            </div>
                         <!-- Форма убрана - используем диалог для сбора данных -->
        `;
        
        container.appendChild(desktopButton);
        container.appendChild(mobileButton);
        container.appendChild(chatWindow);
        document.body.appendChild(container);
    }

    // Глобальные переменные
    window.cesiChatOpen = false;
    window.cesiMessageCount = 0;
    window.cesiUserMessageCount = 0; // Счетчик сообщений пользователя
    window.cesiSessionId = 'session_' + Date.now();
    window.cesiInactivityTimer = null;
    window.cesiLastActivity = Date.now();

         // Функции чата
     window.cesiToggleChat = function() {
         var chatWindow = document.querySelector('.cesi-chat-window');
         var mobileButton = document.querySelector('.cesi-mobile-button');
         
         if (window.cesiChatOpen) {
             chatWindow.style.display = 'none';
             // Показываем мобильную кнопку только на мобильных
             if (window.innerWidth <= 768) {
                 mobileButton.style.display = 'block';
             } else {
                 mobileButton.style.display = 'none';
             }
             window.cesiChatOpen = false;
             
             // Очищаем таймер при закрытии чата
             if (window.cesiInactivityTimer) {
                 clearTimeout(window.cesiInactivityTimer);
             }
         } else {
             chatWindow.style.display = 'flex';
             // Скрываем мобильную кнопку при открытии чата
             mobileButton.style.display = 'none';
             window.cesiChatOpen = true;
             
             // Автоматическое приветствие
             if (window.cesiMessageCount === 0) {
                 setTimeout(function() {
                     cesiAppendMessage('bot', 'Здравствуйте! 👋\n\nЯ ассистент клиники ЦЭСИ.\n\nРасскажу, как проходит лечение, имплантация, и отвечу на любые вопросы по стоматологии.\n\nС чего начнём?');
                     // Запускаем таймер после приветствия
                     cesiResetInactivityTimer();
                 }, 500);
             } else {
                 // Запускаем таймер при открытии чата
                 cesiResetInactivityTimer();
             }
         }
     };

    // Функция скрытия кнопок
    window.cesiHideActionButtons = function() {
        var actionButtons = document.querySelector('.cesi-action-buttons');
        if (actionButtons) {
            actionButtons.classList.add('hidden');
        }
    };

    window.cesiSendMessage = function() {
        var input = document.querySelector('.cesi-chat-input input');
        var message = input.value.trim();
        
        if (message) {
            cesiAppendMessage('user', message);
            input.value = '';
            
            // Сбрасываем таймер неактивности
            cesiResetInactivityTimer();
            
            // Скрываем кнопки после отправки сообщения
            cesiHideActionButtons();
            
            // Показываем индикатор печати
            cesiShowTypingIndicator();
            
            console.log('Отправка сообщения на:', config.serverUrl + '/chat');
            
            // Отправка на сервер
            fetch(config.serverUrl + '/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: window.cesiSessionId
                })
            })
            .then(resp => {
                console.log('Ответ сервера:', resp.status);
                if (!resp.ok) {
                    throw new Error('HTTP ' + resp.status);
                }
                const ct = (resp.headers.get('content-type') || '').toLowerCase();
                if (ct.includes('application/json')) return resp.json();
                return resp.text().then(t => ({ response: t })); // обернули plain text
            })
            .then(data => {
                console.log('[BOT] response data:', data); // оставь для отладки
                // Скрываем индикатор печати
                cesiHideTypingIndicator();
                const text = data.response ?? (data.answer && data.answer.short) ?? data.text ?? data.message ?? '';
                if (text) cesiAppendMessage('bot', String(text));
                
                // Сбрасываем таймер неактивности после ответа бота
                cesiResetInactivityTimer();
                
                // Рендерим CTA кнопку от сервера
                if (data.cta) {
                    renderCTA(data.cta);
                }
                
                // Рендерим кнопки действий от сервера
                if (data.action_buttons) {
                    renderActionButtons(data.action_buttons);
                }
            })
            .catch(error => {
                console.error('Ошибка отправки сообщения:', error);
                // Скрываем индикатор печати
                cesiHideTypingIndicator();
                cesiAppendMessage('bot', 'Извините, произошла ошибка. Попробуйте еще раз или позвоните нам по телефону +7(4152) 44-24-24');
            });
        }
    };

    window.cesiHandleKeyPress = function(event) {
        if (event.key === 'Enter') {
            cesiSendMessage();
        }
    };
    
    window.cesiSendAction = function(action) {
        cesiAppendMessage('user', action);
        
        // Сбрасываем таймер неактивности
        cesiResetInactivityTimer();
        
        // Скрываем кнопки после нажатия на кнопку
        cesiHideActionButtons();
        
        // Показываем индикатор печати
        cesiShowTypingIndicator();
        
        console.log('Отправка действия на:', config.serverUrl + '/chat');
        
        // Отправка на сервер
        fetch(config.serverUrl + '/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: action,
                session_id: window.cesiSessionId
            })
        })
        .then(resp => {
            console.log('Ответ сервера:', resp.status);
            if (!resp.ok) {
                throw new Error('HTTP ' + resp.status);
            }
            const ct = (resp.headers.get('content-type') || '').toLowerCase();
            if (ct.includes('application/json')) return resp.json();
            return resp.text().then(t => ({ response: t })); // обернули plain text
        })
        .then(data => {
            console.log('[BOT] response data:', data); // оставь для отладки
            // Скрываем индикатор печати
            cesiHideTypingIndicator();
            const text = data.response ?? (data.answer && data.answer.short) ?? data.text ?? data.message ?? '';
            if (text) cesiAppendMessage('bot', String(text));
            
            // Сбрасываем таймер неактивности после ответа бота
            cesiResetInactivityTimer();
            
            // Рендерим CTA кнопку от сервера
            if (data.cta) {
                renderCTA(data.cta);
            }
            
            // Рендерим кнопки действий от сервера
            if (data.action_buttons) {
                renderActionButtons(data.action_buttons);
            }
        })
        .catch(error => {
            console.error('Ошибка отправки действия:', error);
            // Скрываем индикатор печати
            cesiHideTypingIndicator();
            cesiAppendMessage('bot', 'Извините, произошла ошибка. Попробуйте еще раз или позвоните нам по телефону +7(4152) 44-24-24');
        });
    };

    // Функция рендера CTA кнопки
    window.renderCTA = function(cta) {
        if (!cta) return;
        
        // Находим последнее сообщение бота
        var messagesContainer = document.querySelector('.cesi-chat-messages');
        var lastBotMessage = messagesContainer.querySelector('.cesi-message.bot:last-child');
        
        if (lastBotMessage) {
            var messageContent = lastBotMessage.querySelector('.cesi-message-content');
            if (messageContent) {
                // Создаем кнопку
                var btn = document.createElement('a');
                btn.className = 'cesi-booking-button';
                btn.textContent = cta.text || 'Записаться на консультацию';
                
                // Настраиваем ссылку в зависимости от типа CTA
                if (cta.type === 'link' && cta.url) {
                    btn.href = cta.url;
                    if (cta.url.startsWith('http')) {
                        btn.target = '_blank';
                        btn.rel = 'noopener';
                    }
                } else if (cta.type === 'call' && cta.phone) {
                    btn.href = 'tel:' + cta.phone;
                } else {
                    // book или fallback - открываем диалог записи
                    btn.href = '#';
                    btn.addEventListener('click', function(e) {
                        e.preventDefault();
                        cesiStartBooking();
                    });
                }
                
                // Добавляем иконку в зависимости от типа
                var iconSpan = document.createElement('span');
                iconSpan.className = 'cesi-action-button-icon';
                if (cta.type === 'call') {
                    iconSpan.textContent = '📞';
                } else if (cta.type === 'link') {
                    iconSpan.textContent = '🔗';
                } else {
                    iconSpan.textContent = '📝';
                }
                btn.insertBefore(iconSpan, btn.firstChild);
                
                // Добавляем кнопку в сообщение с отступом
                var buttonDiv = document.createElement('div');
                buttonDiv.style.marginTop = '12px';
                buttonDiv.appendChild(btn);
                messageContent.appendChild(buttonDiv);
            }
        }
    };

    // Функция рендера кнопок действий
    window.renderActionButtons = function(actionButtons) {
        if (!actionButtons || !actionButtons.length) return;
        
        // Находим последнее сообщение бота
        var messagesContainer = document.querySelector('.cesi-chat-messages');
        var lastBotMessage = messagesContainer.querySelector('.cesi-message.bot:last-child');
        
        if (lastBotMessage) {
            var messageContent = lastBotMessage.querySelector('.cesi-message-content');
            if (messageContent) {
                // Создаем контейнер для кнопок
                var buttonsContainer = document.createElement('div');
                buttonsContainer.style.marginTop = '12px';
                buttonsContainer.style.display = 'flex';
                buttonsContainer.style.flexDirection = 'column';
                buttonsContainer.style.gap = '8px';
                
                // Создаем кнопки
                actionButtons.forEach(function(buttonData) {
                    var btn = document.createElement('button');
                    btn.className = 'cesi-action-button';
                    btn.textContent = buttonData.text;
                    
                    btn.addEventListener('click', function() {
                        cesiSendAction(buttonData.action);
                    });
                    
                    buttonsContainer.appendChild(btn);
                });
                
                messageContent.appendChild(buttonsContainer);
            }
        }
    };

    window.cesiAppendMessage = function(type, text) {
        var messagesContainer = document.querySelector('.cesi-chat-messages');
        var messageDiv = document.createElement('div');
        messageDiv.className = 'cesi-message ' + type;
        
        // Безопасная вставка текста без HTML
        var wrap = document.createElement('div');
        wrap.className = 'cesi-message-content';
        wrap.textContent = text; // безопасно, без HTML
        messageDiv.appendChild(wrap);
        
        messagesContainer.appendChild(messageDiv);
        
                 // Проверяем, запрашивает ли бот номер телефона
         if (type === 'bot' && (text.includes('номер телефона') || text.includes('телефон'))) {
                          // Создаем поле ввода с маской телефона
              var phoneInputContainer = document.createElement('div');
              phoneInputContainer.style.marginTop = '12px';
              phoneInputContainer.style.display = 'flex';
              phoneInputContainer.style.flexDirection = 'column';
              phoneInputContainer.style.gap = '8px';
              phoneInputContainer.style.width = '100%';
              phoneInputContainer.style.alignItems = 'stretch';
              phoneInputContainer.style.justifyContent = 'flex-start';
              
                             var phoneInput = document.createElement('input');
               phoneInput.type = 'text';
               phoneInput.placeholder = '+7(___) ___-__-__';
              phoneInput.style.width = '100%';
              phoneInput.style.padding = '12px 16px';
              phoneInput.style.border = '2px solid #e9ecef';
              phoneInput.style.borderRadius = '8px';
              phoneInput.style.fontSize = '14px';
              phoneInput.style.outline = 'none';
              phoneInput.style.transition = 'border-color 0.2s';
              phoneInput.style.boxSizing = 'border-box';
              phoneInput.style.fontFamily = 'inherit';
            
            // Стили для фокуса
                         phoneInput.addEventListener('focus', function() {
                 this.style.borderColor = '#23BFCF';
             });
            
            phoneInput.addEventListener('blur', function() {
                this.style.borderColor = '#ddd';
            });
            
                                                                                                       // Мини-маска российского номера: +7 (___) ___-__-__
               function attachPhoneMask(input, submitBtn) {
                   const digitsOnly = v => v.replace(/\D/g,'');
                   const format = v => {
                       // Форматируем как +7 (XXX) XXX-XX-XX
                       let d = digitsOnly(v);
                       // Нормализация «8...» -> «7...», «9...» -> «7 9...», всегда начинаем с 7
                       if (d.startsWith('8')) d = '7' + d.slice(1);
                       if (d.startsWith('9')) d = '7' + d;
                       if (!d.startsWith('7')) d = '7' + d;
                       d = d.slice(0, 11); // максимум 11 цифр (7 + 10)
                       const parts = ['+7'];
                       if (d.length > 1) parts.push(' (', d.slice(1,4));
                       if (d.length >= 4) parts.push(')', d.slice(4,7));
                       if (d.length >= 7) parts.push('-', d.slice(7,9));
                       if (d.length >= 9) parts.push('-', d.slice(9,11));
                       return parts.join('');
                   };
                   const isValid = v => digitsOnly(v).length === 11;
                   const toE164 = v => isValid(v) ? '+' + digitsOnly(v) : null; // вернёт, например, "+79991234567" или null
                   
                   const onInput = () => {
                       // простая маска: форматируем и блокируем отправку, если невалидно
                       input.value = format(input.value);
                       if (submitBtn) submitBtn.disabled = !isValid(input.value);
                   };
                   
                   input.addEventListener('input', onInput);
                   input.addEventListener('blur', onInput);
                   onInput(); // инициализация
                   
                   // Возвращаем удобный АРI
                   return {
                       isValid: () => isValid(input.value),
                       getE164: () => toE164(input.value)
                   };
               }
               
               // Подключаем маску
               var mask = attachPhoneMask(phoneInput, sendButton);
             
                                                       // Обработка клавиш - маска сама управляет вводом
            
                                                                                                                                             // Обработка Enter для отправки (только при валидном номере)
                 phoneInput.addEventListener('keypress', function(e) {
                     if (e.key === 'Enter') {
                         var phone = mask.getE164();
                         if (phone) {
                             cesiAppendMessage('user', phoneInput.value);
                             phoneInput.value = '';
                             phoneInputContainer.remove();
                             
                             // Отправляем на сервер
                             cesiSendPhoneToServer(phone);
                         }
                     }
                 });
            
                         var sendButton = document.createElement('button');
             sendButton.textContent = 'Отправить';
             sendButton.style.width = '100%';
             sendButton.style.padding = '12px 16px';
                           sendButton.style.background = '#23BFCF';
                           sendButton.style.color = '#fff';
             sendButton.style.border = 'none';
             sendButton.style.borderRadius = '8px';
             sendButton.style.cursor = 'pointer';
             sendButton.style.fontSize = '14px';
             sendButton.style.fontWeight = '500';
             sendButton.style.transition = 'all 0.2s';
             sendButton.style.fontFamily = 'inherit';
             sendButton.style.boxSizing = 'border-box';
             sendButton.style.flexShrink = '0';
            
            sendButton.addEventListener('mouseenter', function() {
                this.style.background = '#12a0af';
            });
            
                         sendButton.addEventListener('mouseleave', function() {
                 this.style.background = '#23BFCF';
             });
            
                                                                                                             // Функция проверки валидности номера (теперь управляется маской)
                function updateSendButton() {
                    if (mask.isValid()) {
                        sendButton.style.background = '#23BFCF';
                        sendButton.style.cursor = 'pointer';
                        sendButton.disabled = false;
                    } else {
                        sendButton.style.background = '#ccc';
                        sendButton.style.cursor = 'not-allowed';
                        sendButton.disabled = true;
                    }
                }
             
                                                                                     sendButton.addEventListener('click', function() {
                    var phone = mask.getE164();
                    if (phone) {
                        cesiAppendMessage('user', phoneInput.value);
                        phoneInput.value = '';
                        phoneInputContainer.remove();
                        
                        // Отправляем на сервер
                        cesiSendPhoneToServer(phone);
                    }
                });
            
            phoneInputContainer.appendChild(phoneInput);
            phoneInputContainer.appendChild(sendButton);
            wrap.appendChild(phoneInputContainer);
            
            // Фокус на поле ввода
            setTimeout(function() {
                phoneInput.focus();
            }, 100);
        }
        
        // Плавный скролл к началу нового сообщения (не к кнопке CTA)
        if (type === 'bot') {
            // Для сообщений бота скроллим к началу сообщения
            messageDiv.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        } else {
            // Для сообщений пользователя скроллим к концу
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        if (type === 'user') {
            window.cesiMessageCount++;
            window.cesiUserMessageCount++;
            
            // Удаляем старые кнопки записи при новом сообщении пользователя
            var oldButtons = document.querySelectorAll('.cesi-booking-button');
            oldButtons.forEach(function(button) {
                // Если кнопка внутри сообщения - удаляем только кнопку
                if (button.closest('.cesi-message-content')) {
                    button.closest('div').remove();
                } else {
                    // Если отдельное сообщение с кнопкой - удаляем все сообщение
                    button.closest('.cesi-message').remove();
                }
            });
            
            // Удаляем поля ввода телефона при новом сообщении пользователя
            var phoneInputs = document.querySelectorAll('input[placeholder*="телефон"], input[placeholder*="+7"]');
            phoneInputs.forEach(function(input) {
                if (input.closest('.cesi-message-content')) {
                    input.closest('div').remove();
                }
            });
        }
    };



    // Функция начала записи из кнопки в сообщении бота
    window.cesiStartBooking = function() {
        // Отправляем на сервер для начала диалога
        cesiSendAction('Записаться на консультацию');
    };

    // Функция начала записи из начальных кнопок действий
    window.cesiStartBookingFromAction = function() {
        // Скрываем кнопки действий
        cesiHideActionButtons();
        // Отправляем на сервер для начала диалога
        cesiSendAction('Записаться на консультацию');
    };

    // Функция начала записи из сообщения о неактивности
    window.cesiStartBookingFromInactivity = function() {
        cesiSendAction('Записаться на консультацию');
    };



    // Функции для работы с формой убраны - используем диалог
    
         // Функция отправки телефона на сервер
     window.cesiSendPhoneToServer = function(phoneNumber) {
         // Показываем индикатор печати
         cesiShowTypingIndicator();
         
         // Отправляем на сервер
         fetch(config.serverUrl + '/chat', {
             method: 'POST',
             headers: {
                 'Content-Type': 'application/json',
             },
             body: JSON.stringify({
                 message: phoneNumber,
                 session_id: window.cesiSessionId
             })
         })
         .then(resp => {
             if (!resp.ok) {
                 throw new Error('HTTP ' + resp.status);
             }
             const ct = (resp.headers.get('content-type') || '').toLowerCase();
             if (ct.includes('application/json')) return resp.json();
             return resp.text().then(t => ({ response: t })); // обернули plain text
         })
         .then(data => {
             console.log('[BOT] response data:', data); // оставь для отладки
             // Скрываем индикатор печати
             cesiHideTypingIndicator();
             const text = data.response ?? (data.answer && data.answer.short) ?? data.text ?? data.message ?? '';
             if (text) cesiAppendMessage('bot', String(text));
             
             // Сбрасываем таймер неактивности после ответа бота
             cesiResetInactivityTimer();
             
             // Рендерим CTA кнопку от сервера
             if (data.cta) {
                 renderCTA(data.cta);
             }
             
             // Рендерим кнопки действий от сервера
             if (data.action_buttons) {
                 renderActionButtons(data.action_buttons);
             }
         })
         .catch(error => {
             console.error('Ошибка отправки телефона:', error);
             // Скрываем индикатор печати
             cesiHideTypingIndicator();
             cesiAppendMessage('bot', 'Извините, произошла ошибка. Попробуйте еще раз или позвоните нам по телефону +7(4152) 44-24-24');
         });
     };
     
     
    
    // Функции для управления таймером неактивности
    window.cesiResetInactivityTimer = function() {
        window.cesiLastActivity = Date.now();
        
        // Очищаем предыдущий таймер
        if (window.cesiInactivityTimer) {
            clearTimeout(window.cesiInactivityTimer);
        }
        
        // Устанавливаем новый таймер на 30 секунд
        window.cesiInactivityTimer = setTimeout(function() {
            cesiShowInactivityMessage();
        }, 30000); // 30 секунд
    };
    
         window.cesiShowInactivityMessage = function() {
         // Проверяем, что чат открыт и нет активного диалога записи
         if (!window.cesiChatOpen) return;
         
         var messagesContainer = document.querySelector('.cesi-chat-messages');
         var inactivityDiv = document.createElement('div');
         inactivityDiv.className = 'cesi-message bot';
                   inactivityDiv.innerHTML = `<div class="cesi-message-content">Есть вопросы? Запишитесь на консультацию и получите скидку 30% на КТ по промокоду «Чат»<div style="margin-top: 12px;"><button onclick="cesiStartBookingFromInactivity()" class="cesi-booking-button"><span class="cesi-action-button-icon">📝</span>Записаться</button></div></div>`;
         messagesContainer.appendChild(inactivityDiv);
         
         // Плавный скролл к началу сообщения о неактивности
         inactivityDiv.scrollIntoView({ 
             behavior: 'smooth', 
             block: 'start' 
         });
     };
    
    window.cesiStartBookingFromInactivity = function() {
        cesiSendAction('Записаться на консультацию');
    };
    
    // Функции для индикатора печати
    window.cesiShowTypingIndicator = function() {
        var messagesContainer = document.querySelector('.cesi-chat-messages');
        var typingDiv = document.createElement('div');
        typingDiv.className = 'cesi-message bot cesi-typing-indicator';
        typingDiv.innerHTML = `
            <div class="cesi-message-content">
                <div class="cesi-typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        
        // Плавный скролл к индикатору печати
        typingDiv.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    };
    
    window.cesiHideTypingIndicator = function() {
        var typingIndicator = document.querySelector('.cesi-typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    };

    // Инициализация
    function init() {
        createStyles();
        createWidget();
        
        // Принудительно скрываем мобильную кнопку на десктопе
        if (window.innerWidth > 768) {
            var mobileButton = document.querySelector('.cesi-mobile-button');
            if (mobileButton) {
                mobileButton.style.display = 'none';
            }
        }
        
        // Дополнительная проверка через 1 секунду
        setTimeout(function() {
            if (window.innerWidth > 768) {
                var mobileButton = document.querySelector('.cesi-mobile-button');
                if (mobileButton) {
                    mobileButton.style.display = 'none';
                }
            }
        }, 1000);
    }

    // Запуск после загрузки страницы
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(); 