#!/usr/bin/env python3
"""
CLI инструмент для тестирования RAG-пайплайна
Использование: python tools/eval.py [--trace] [--mode PRECISE_SIMPLE|HYBRID_TIGHT]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_test_queries() -> List[Dict[str, Any]]:
    """Загружает эталонные вопросы для тестирования"""
    return [
        {
            "query": "сколько стоит имплантация",
            "expected_topic": "prices",
            "expected_file": "prices-implant.md"
        },
        {
            "query": "адрес клиники",
            "expected_topic": "contacts", 
            "expected_file": "clinic-contacts.md"
        },
        {
            "query": "больно ли ставить имплант",
            "expected_topic": "safety",
            "expected_file": "faq-implants-pain.md"
        },
        {
            "query": "какие врачи работают",
            "expected_topic": "doctors",
            "expected_file": "doctors.md"
        },
        {
            "query": "гарантия на импланты",
            "expected_topic": "warranty",
            "expected_file": "warranty.md"
        },
        {
            "query": "приживаемость имплантов",
            "expected_topic": "safety",
            "expected_file": "faq-implants-osseointegration.md"
        },
        {
            "query": "консультация бесплатная",
            "expected_topic": "consultation",
            "expected_file": "consultation-free.md"
        },
        {
            "query": "противопоказания к имплантации",
            "expected_topic": "safety",
            "expected_file": "implants-contraindications.md"
        },
        {
            "query": "как добраться до клиники",
            "expected_topic": "contacts",
            "expected_file": "clinic-contacts.md"
        },
        {
            "query": "сколько длится процедура имплантации",
            "expected_topic": "implants",
            "expected_file": "faq-implants-duration.md"
        }
    ]

def run_single_test(query_data: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """Запускает один тест"""
    try:
        # Импортируем RAG engine
        from rag_engine import get_rag_answer
        
        query = query_data["query"]
        print(f"🔍 Тестируем: '{query}'")
        
        # Вызываем RAG
        start_time = datetime.now()
        response, metadata = get_rag_answer(query)
        end_time = datetime.now()
        
        # Анализируем результат
        result = {
            "query": query,
            "expected_topic": query_data["expected_topic"],
            "expected_file": query_data["expected_file"],
            "response_type": type(response).__name__,
            "has_response": bool(response),
            "response_length": len(str(response)) if response else 0,
            "metadata_keys": list(metadata.keys()) if metadata else [],
            "execution_time_ms": (end_time - start_time).total_seconds() * 1000,
            "success": True,
            "error": None
        }
        
        # Проверяем метаданные
        if metadata:
            result["relevance_score"] = metadata.get("relevance_score", 0.0)
            result["cand_cnt"] = metadata.get("cand_cnt", 0)
            result["theme_hint"] = metadata.get("theme_hint")
            result["best_chunk_id"] = metadata.get("best_chunk_id")
            
            # Проверяем соответствие ожиданиям
            if result["theme_hint"] == query_data["expected_topic"]:
                result["topic_match"] = True
            else:
                result["topic_match"] = False
                
            if result["best_chunk_id"] and query_data["expected_file"] in result["best_chunk_id"]:
                result["file_match"] = True
            else:
                result["file_match"] = False
        else:
            result["topic_match"] = False
            result["file_match"] = False
        
        if trace:
            result["full_response"] = response
            result["full_metadata"] = metadata
            
        return result
        
    except Exception as e:
        return {
            "query": query_data["query"],
            "expected_topic": query_data["expected_topic"],
            "expected_file": query_data["expected_file"],
            "success": False,
            "error": str(e),
            "execution_time_ms": 0
        }

def run_dryrun(mode: str = None, trace: bool = False) -> Dict[str, Any]:
    """Запускает dryrun тесты"""
    print(f"🧪 Запуск dryrun тестов...")
    if mode:
        print(f"📋 Режим: {mode}")
        os.environ["RAG_MODE"] = mode
    
    # Загружаем тестовые запросы
    test_queries = load_test_queries()
    print(f"📝 Загружено {len(test_queries)} тестовых запросов")
    
    results = []
    total_time = 0
    successful_tests = 0
    topic_matches = 0
    file_matches = 0
    
    for i, query_data in enumerate(test_queries, 1):
        print(f"\n--- Тест {i}/{len(test_queries)} ---")
        result = run_single_test(query_data, trace)
        results.append(result)
        
        total_time += result["execution_time_ms"]
        
        if result["success"]:
            successful_tests += 1
            if result.get("topic_match"):
                topic_matches += 1
            if result.get("file_match"):
                file_matches += 1
                
            print(f"✅ Успех: {result['execution_time_ms']:.1f}ms")
            if result.get("topic_match"):
                print(f"   🎯 Тема совпала: {result['theme_hint']}")
            if result.get("file_match"):
                print(f"   📄 Файл совпал: {result['best_chunk_id']}")
        else:
            print(f"❌ Ошибка: {result['error']}")
    
    # Создаем сводку
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": os.getenv("RAG_MODE", "PRECISE_SIMPLE"),
        "total_tests": len(test_queries),
        "successful_tests": successful_tests,
        "failed_tests": len(test_queries) - successful_tests,
        "topic_matches": topic_matches,
        "file_matches": file_matches,
        "avg_execution_time_ms": total_time / len(test_queries) if test_queries else 0,
        "success_rate": successful_tests / len(test_queries) if test_queries else 0,
        "topic_match_rate": topic_matches / successful_tests if successful_tests else 0,
        "file_match_rate": file_matches / successful_tests if successful_tests else 0,
        "results": results
    }
    
    # Сохраняем результаты
    logs_dir = Path("logs/eval")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = logs_dir / f"dryrun_{timestamp}.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 СВОДКА:")
    print(f"   Всего тестов: {summary['total_tests']}")
    print(f"   Успешных: {summary['successful_tests']}")
    print(f"   Неудачных: {summary['failed_tests']}")
    print(f"   Совпадений тем: {summary['topic_matches']}")
    print(f"   Совпадений файлов: {summary['file_matches']}")
    print(f"   Среднее время: {summary['avg_execution_time_ms']:.1f}ms")
    print(f"   Успешность: {summary['success_rate']:.1%}")
    print(f"   Точность тем: {summary['topic_match_rate']:.1%}")
    print(f"   Точность файлов: {summary['file_match_rate']:.1%}")
    print(f"\n💾 Результаты сохранены: {result_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="CLI для тестирования RAG-пайплайна")
    parser.add_argument("--trace", action="store_true", help="Включить детальное логирование")
    parser.add_argument("--mode", choices=["PRECISE_SIMPLE", "HYBRID_TIGHT"], 
                       help="Режим RAG для тестирования")
    
    args = parser.parse_args()
    
    try:
        summary = run_dryrun(args.mode, args.trace)
        
        # Проверяем, что нет критических ошибок
        if summary["failed_tests"] > 0:
            print(f"\n⚠️  Обнаружены ошибки в {summary['failed_tests']} тестах")
            sys.exit(1)
        
        if summary["success_rate"] < 0.8:
            print(f"\n⚠️  Низкая успешность: {summary['success_rate']:.1%}")
            sys.exit(1)
            
        print(f"\n✅ Все тесты прошли успешно!")
        
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

