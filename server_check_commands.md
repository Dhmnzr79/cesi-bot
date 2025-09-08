# Команды для проверки логов на сервере

## 1. Быстрая проверка логирования
```bash
python -c "
from core.logging_setup import setup_logging
from core.logger import log_query, log_response, log_minimal
setup_logging(None)
log_query('Тест с сервера', 'server_test')
log_response('Тест с сервера', [{'file': 'test.md', 'h2': 'Тест', 'emb': 0.9, 'bm25': 0.8, 'hybrid': 0.85}], 100)
log_minimal('Тест с сервера', 0.85, 0.65, True, False)
print('✅ Тестовые логи записаны')
"
```

## 2. Проверка файлов логов
```bash
# Проверяем директорию
ls -la /root/cesi-bot/logs/

# Смотрим последние записи
echo "=== queries.jsonl ==="
tail -3 /root/cesi-bot/logs/queries.jsonl

echo "=== bot_responses.jsonl ==="
tail -3 /root/cesi-bot/logs/bot_responses.jsonl

echo "=== minimal_logs.jsonl ==="
tail -3 /root/cesi-bot/logs/minimal_logs.jsonl
```

## 3. Проверка через реальный запрос
```bash
# Запускаем бота (если не запущен)
cd /root/cesi-bot
python app.py &

# Ждем запуска
sleep 3

# Отправляем тестовый запрос
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Сколько стоит имплант?", "session_id": "server_test_123"}'

# Проверяем логи
echo "=== После запроса ==="
tail -1 /root/cesi-bot/logs/queries.jsonl
tail -1 /root/cesi-bot/logs/bot_responses.jsonl
tail -1 /root/cesi-bot/logs/minimal_logs.jsonl
```

## 4. Проверка формата JSON
```bash
# Проверяем валидность JSON
python -c "
import json
with open('/root/cesi-bot/logs/queries.jsonl', 'r') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            print('✅ queries.jsonl:', data)
            break
        except:
            print('❌ Ошибка JSON в queries.jsonl')
            break
"
```

## 5. Мониторинг в реальном времени
```bash
# Следим за логами в реальном времени
tail -f /root/cesi-bot/logs/queries.jsonl &
tail -f /root/cesi-bot/logs/bot_responses.jsonl &
tail -f /root/cesi-bot/logs/minimal_logs.jsonl &

# Отправляем несколько запросов
for i in {1..3}; do
  curl -X POST http://localhost:5000/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"Тест $i\", \"session_id\": \"test_$i\"}"
  sleep 1
done
```

## 6. Проверка размеров файлов
```bash
# Проверяем размеры файлов
du -h /root/cesi-bot/logs/*.jsonl

# Считаем количество записей
wc -l /root/cesi-bot/logs/*.jsonl
```

## 7. Проверка прав доступа
```bash
# Проверяем права
ls -la /root/cesi-bot/logs/

# Если нужно исправить права
chmod 755 /root/cesi-bot/logs/
chmod 644 /root/cesi-bot/logs/*.jsonl
```

