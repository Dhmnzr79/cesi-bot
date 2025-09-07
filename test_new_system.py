#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rag_engine

def test_new_architecture():
    print("🔍 Тестируем новую архитектуру...")
    
    # Проверяем индексы
    print(f"✅ CANON: {rag_engine.CANON}")
    print(f"✅ ALIAS_MAP_GLOBAL: {len(rag_engine.ALIAS_MAP_GLOBAL)} алиасов")
    print(f"✅ H2_INDEX: {len(rag_engine.H2_INDEX)} заголовков")
    print(f"✅ FILE_META: {len(rag_engine.FILE_META)} файлов")
    print(f"✅ ALL_CHUNKS: {len(rag_engine.ALL_CHUNKS)} чанков")
    
    # Тестируем поиск
    print("\n🧪 Тестируем поиск...")
    
    # Тест 1: Поиск по алиасу
    test_queries = [
        "технологии в клинике",
        "консультация", 
        "цены",
        "врачи"
    ]
    
    for query in test_queries:
        print(f"\n📝 Запрос: '{query}'")
        
        # Используем новую функцию поиска
        try:
            chunks, flags = rag_engine.retrieve_relevant_chunks_new(
                query, 
                None, 
                lambda q: rag_engine.hybrid_retriever(q, 5)
            )
            
            print(f"  🎯 Найдено: {len(chunks)} чанков")
            print(f"  📊 Флаги: {flags}")
            
            if chunks:
                chunk = chunks[0]
                print(f"  📄 Файл: {chunk.file_name}")
                print(f"  📝 Текст: {chunk.text[:100]}...")
        
    except Exception as e:
            print(f"  ❌ Ошибка: {e}")
    
    print("\n✅ Тест завершен!")

if __name__ == "__main__":
    test_new_architecture()