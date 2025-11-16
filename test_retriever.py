# test_retriever.py
# -*- coding: utf-8 -*-
"""
Test script for retriever
"""

from retriever_custom import retrieve



if __name__ == "__main__":
    # Test
    test_q = "Kết cấu hạ tầng đường bộ bao gồm những gì?"
    print(f"\n🔍 Test query: {test_q}")
    print(f"{'='*60}")
    ctx, sources = retrieve(test_q)
    
    print(f"\n📄 Context:")
    print(ctx)
    
    print(f"\n📚 Sources:")
    for i, src in enumerate(sources, 1):
        print(f"  [{i}] {src}")