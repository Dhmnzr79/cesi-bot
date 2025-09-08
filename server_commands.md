# Команды для диагностики на сервере

## 1. Быстрая проверка логирования
```bash
python -c "import logging, json; import os; from core.logging_setup import setup_logging; import time; setup_logging(None); lg=logging.getLogger('cesi.queries'); lg.info(json.dumps({'ping':time.time()}))"
```

## 2. Проверка файла queries.jsonl
```bash
cat /root/cesi-bot/logs/queries.jsonl
# или
tail -5 /root/cesi-bot/logs/queries.jsonl
```

## 3. Проверка процессов
```bash
# Найти процессы gunicorn/uvicorn
pgrep -af 'gunicorn|uvicorn'

# Взять PID и проверить открытые файлы
sudo lsof -p <PID> | grep cesi-bot/logs
```

## 4. Поиск "убежавших" файлов
```bash
sudo find / -name "queries.jsonl" 2>/dev/null
```

## 5. Проверка прав
```bash
whoami
ls -ld /root/cesi-bot/logs

# Если сервис не root - выдать права
chown -R <user>:<group> /root/cesi-bot/logs
```

## 6. Проверка содержимого директории
```bash
ls -la /root/cesi-bot/logs/
```

## 7. Проверка работы приложения
```bash
# Запустить приложение и проверить логи
python app.py &
sleep 5
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"message": "тест"}'
tail -5 /root/cesi-bot/logs/queries.jsonl
```

