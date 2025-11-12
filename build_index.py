# -*- coding: utf-8 -*-
"""
Index enriched law chunks into Weaviate
- Collection: LawChunks
- Vector: from enriched_text (dense embedding)
- BM25: from rerank_title + rerank_body + path_text
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

# chọn model embed (phải giống retriever)
EMB_MODEL = "BAAI/bge-m3"

# ========================= INIT =========================
print("🔌 Connecting to Weaviate...")
client = weaviate.connect_to_local()

try:
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        print(f"🧹 Deleted existing collection: {COLLECTION_NAME}")

    print("📐 Creating schema...")

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),  # ta tự nhúng vector
        properties=[
            Property(name="law", data_type=DataType.TEXT),
            Property(name="law_code", data_type=DataType.TEXT),
            Property(name="chapter", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="article_no", data_type=DataType.TEXT),
            Property(name="article_title", data_type=DataType.TEXT),
            Property(name="clause_no", data_type=DataType.NUMBER),
            Property(name="point", data_type=DataType.TEXT),
            Property(name="bullet_idx", data_type=DataType.NUMBER),
            Property(name="granularity", data_type=DataType.TEXT),
            Property(name="header", data_type=DataType.TEXT),
            Property(name="display_citation", data_type=DataType.TEXT),
            Property(name="path_text", data_type=DataType.TEXT),
            Property(name="clause_head", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="enriched_text", data_type=DataType.TEXT),   # để lưu lại
            Property(name="rerank_title", data_type=DataType.TEXT),
            Property(name="rerank_body", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
        ],
        # enable hybrid search with HNSW vector index
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=128,
            max_connections=64,
        ),
        # BM25 is enabled by default in Weaviate v4
    )

    collection = client.collections.get(COLLECTION_NAME)
    print(f"✅ Collection created: {COLLECTION_NAME}")

    # ========================= EMBEDDING MODEL =========================
    print("🧠 Loading embedding model:", EMB_MODEL)
    embedder = SentenceTransformer(EMB_MODEL, device="cpu")

    # ========================= INDEXING =========================
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print("⚠️ No processed files found. Run chunker first.")
        exit()

    for file_path in json_files:
        print(f"\n📄 Indexing: {file_path.name}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        print(f"  → {len(data)} chunks to embed")
        
        batch = []
        for rec in tqdm(data, desc="Embedding & inserting", ncols=80):
            vec = embedder.encode(rec["enriched_text"], normalize_embeddings=True).astype("float32").tolist()
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
                        "enriched_text": rec.get("enriched_text", ""),
                        "rerank_title": rec.get("rerank_title", ""),
                        "rerank_body": rec.get("rerank_body", ""),
                        "source_file": rec.get("source_file", ""),
                    },
                    vector=vec
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