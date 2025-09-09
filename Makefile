# Makefile для CESI-bot

.PHONY: help dryrun dryrun-precise dryrun-hybrid test clean

help:
	@echo "Доступные команды:"
	@echo "  dryrun          - Запустить dryrun тесты в текущем режиме"
	@echo "  dryrun-precise  - Запустить dryrun в режиме PRECISE_SIMPLE"
	@echo "  dryrun-hybrid   - Запустить dryrun в режиме HYBRID_TIGHT"
	@echo "  test            - Запустить все тесты"
	@echo "  clean           - Очистить логи"

dryrun:
	@echo "🧪 Запуск dryrun тестов..."
	python tools/eval.py

dryrun-precise:
	@echo "🧪 Запуск dryrun тестов в режиме PRECISE_SIMPLE..."
	python tools/eval.py --mode PRECISE_SIMPLE

dryrun-hybrid:
	@echo "🧪 Запуск dryrun тестов в режиме HYBRID_TIGHT..."
	python tools/eval.py --mode HYBRID_TIGHT

test: dryrun-precise dryrun-hybrid
	@echo "✅ Все тесты завершены"

clean:
	@echo "🧹 Очистка логов..."
	rm -f logs/eval/*.json
	rm -f logs/rag_trace.jsonl
	@echo "✅ Логи очищены"

