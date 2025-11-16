# -*- coding: utf-8 -*-
"""
Index enriched law chunks into Weaviate
- Collection: LawChunks
- Embedding: Alibaba-NLP/gte-multilingual-base (from enriched_text)
- BM25: article_no, article_title, clause_no, point, clause_head, text
- Reranker: BAAI/bge-reranker-v2-m3 (uses enriched_text)
- Retrieval: Hybrid (dynamic alpha) + Reranker
"""

import os, json, weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.data import DataObject
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

# ========================= CONFIG =========================
DATA_DIR = Path("data/processed")
COLLECTION_NAME = "LawChunks"

# Embedding model (từ evaluation: gte-multilingual-base tốt nhất)
EMB_MODEL = "Alibaba-NLP/gte-multilingual-base"

# ========================= INIT =========================
print("🔌 Connecting to Weaviate...")
client = weaviate.connect_to_local()

try:
    # Xóa collection cũ (nếu có)
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        print(f"🧹 Deleted existing collection: {COLLECTION_NAME}")

    print("📐 Creating schema...")

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),  # tự nhúng vector
        properties=[
            # Metadata (không dùng cho BM25, chỉ display)
            Property(name="law", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="law_code", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="chapter", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="section", data_type=DataType.TEXT, skip_vectorization=True),
            
            # BM25 optimized fields (CHỈ 6 fields này được BM25 search)
            Property(name="article_no", data_type=DataType.TEXT),       # "15", "22" - match "Điều 15"
            Property(name="article_title", data_type=DataType.TEXT),    # "Phạm vi điều chỉnh"
            Property(name="clause_no", data_type=DataType.TEXT),        # "1", "2" - match "Khoản 2"
            Property(name="point", data_type=DataType.TEXT),            # "a", "b" - match "Điểm a"
            Property(name="clause_head", data_type=DataType.TEXT),      # Header của khoản
            Property(name="text", data_type=DataType.TEXT),             # Nội dung chính
            
            # Display fields (không dùng cho BM25)
            Property(name="bullet_idx", data_type=DataType.NUMBER),
            Property(name="granularity", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="header", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="display_citation", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="path_text", data_type=DataType.TEXT, skip_vectorization=True),
            
            # For reranker (full context với tags)
            Property(name="enriched_text", data_type=DataType.TEXT, skip_vectorization=True),
            
            Property(name="source_file", data_type=DataType.TEXT, skip_vectorization=True),
        ],
        # enable hybrid search với HNSW vector index
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=128,
            max_connections=64,
        ),
        # BM25 auto bật trên các TEXT field ở trên
    )

    collection = client.collections.get(COLLECTION_NAME)
    print(f"✅ Collection created: {COLLECTION_NAME}")

    # ========================= EMBEDDING MODEL =========================
    print("🧠 Loading embedding model:", EMB_MODEL)
    embedder = SentenceTransformer(EMB_MODEL, device="cpu",trust_remote_code=True)

    # ========================= INDEXING =========================
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print("⚠️ No processed files found. Run chunker first.")
        raise SystemExit

    for file_path in json_files:
        print(f"\n📄 Indexing: {file_path.name}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        print(f"  → {len(data)} chunks to embed")

        batch: list[DataObject] = []
        for rec in tqdm(data, desc="Embedding & inserting", ncols=80):
            enriched = rec.get("enriched_text", "") or ""
            vec = embedder.encode(enriched, normalize_embeddings=True).astype("float32").tolist()

            batch.append(
                DataObject(
                    properties={
                        "law": rec.get("law", ""),
                        "law_code": rec.get("law_code", ""),
                        "chapter": rec.get("chapter", ""),
                        "section": rec.get("section", ""),
                        "article_no": rec.get("article_no", ""),
                        "article_title": rec.get("article_title", ""),
                        "clause_no": rec.get("clause_no"),
                        "point": rec.get("point", ""),
                        "bullet_idx": rec.get("bullet_idx"),
                        "granularity": rec.get("granularity", ""),
                        "header": rec.get("header", ""),
                        "display_citation": rec.get("display_citation", ""),
                        "path_text": rec.get("path_text", ""),
                        "clause_head": rec.get("clause_head", ""),
                        "text": rec.get("text", ""),
                        "enriched_text": enriched,  # Full context cho reranker
                        "source_file": rec.get("source_file", ""),
                    },
                    vector=vec,
                )
            )

            if len(batch) >= 64:
                collection.data.insert_many(batch)
                batch = []

        if batch:
            collection.data.insert_many(batch)

        print(f"✅ Done {file_path.name}")

    print("🎉 All files indexed successfully.")
finally:
    client.close()
