# test_retriever.py
# -*- coding: utf-8 -*-
"""
Test script for retriever
"""

from retriever import retrieve, client

if __name__ == "__main__":
    try:
        q = "Người điều khiển xe ô tô không thắt dây đai an toàn khi xe đang chạy bị phạt bao nhiêu tiền?"
        ctx, srcs = retrieve(q, k=5, base_alpha=0.55)
        
        print("\n📘 Full Context (all chunks):\n")
        print(ctx)
        
        print("\n📚 Sources:")
        for s in srcs:
            print(" -", s)
    finally:
        client.close()
        print("\n✓ Weaviate connection closed")
