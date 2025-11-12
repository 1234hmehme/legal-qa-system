import json, pickle, os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

# Disable MPS to avoid segmentation fault on macOS
torch.backends.mps.is_available = lambda: False
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

# Connect to Weaviate
print("Connecting to Weaviate...")
try:
    client = weaviate.connect_to_local()
    print("✓ Connected to Weaviate")
except Exception as e:
    print(f"⚠ Could not connect to Weaviate: {e}")
    print("Falling back to FAISS only")
    client = None

FILES = [
    ("Hiến pháp 2013", "data/processed/Hienphap2013.json"),
    ("Nghị định 168/2024", "data/processed/168-2024-ND-CP.json"),
    ("Luật TT, ATGT 2024", "data/processed/luat_trat_tu_an_toan_giaothong_duongbo.json"),
    ("Luật Đường bộ 2024", "data/processed/luatduongbo.json"),
]

os.makedirs("index", exist_ok=True)
all_docs, all_meta = [], []

for law_name, path in FILES:
    items = json.load(open(path, encoding="utf-8"))
    for x in items:
        doc = f"{x['id']} {x.get('title','')}. {x['text']}"
        all_docs.append(doc)
        all_meta.append({
            "law": law_name,
            "article_id": x["id"],  # Đổi từ "id" thành "article_id"
            "title": x.get("title",""),
            "chapter": x.get("chapter",""),
            "source_file": x.get("source_file",""),
            "granularity": x.get("granularity",""),
            "slice_id": x.get("slice_id",""),
        })


emb = SentenceTransformer("BAAI/bge-m3", device='cpu')
print("Model BAAI/bge-m3 loaded on CPU")

# Process in batches to avoid memory issues
batch_size = 16  # Smaller batch size to be safer
all_vectors = []

print(f"Processing {len(all_docs)} documents in batches of {batch_size}...")
for i in range(0, len(all_docs), batch_size):
    batch = all_docs[i:i+batch_size]
    batch_num = i//batch_size + 1
    total_batches = (len(all_docs)-1)//batch_size + 1
    print(f"Processing batch {batch_num}/{total_batches}")
    
    try:
        batch_vecs = emb.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        batch_vecs = batch_vecs.astype("float32")
        all_vectors.append(batch_vecs)
        print(f"  ✓ Batch {batch_num} completed successfully")
    except Exception as e:
        print(f"  ✗ Error in batch {batch_num}: {e}")
        raise e

vecs = np.vstack(all_vectors)

# Upload to Weaviate
if not client:
    print("❌ Weaviate not available. Please start Weaviate with: docker-compose up -d")
    exit(1)

print("Uploading to Weaviate...")
try:
    # Create collection/class
    collection_name = "LawArticles"
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
    
    # Create new collection with custom vectors
    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=None,  # We'll provide vectors manually
        properties=[
            weaviate.classes.config.Property(
                name="law",
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
            weaviate.classes.config.Property(
                name="article_id",  # Đổi từ "id" thành "article_id"
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
            weaviate.classes.config.Property(
                name="title",
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
            weaviate.classes.config.Property(
                name="chapter",
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
            weaviate.classes.config.Property(
                name="text",
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
            weaviate.classes.config.Property(
                name="source_file",
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
            weaviate.classes.config.Property(
                name="granularity",
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
            weaviate.classes.config.Property(
                name="slice_id",
                data_type=weaviate.classes.config.DataType.TEXT,
            ),
        ]
    )
    
    # Batch upload
    print(f"Uploading {len(all_docs)} documents to Weaviate...")
    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        batch_end = min(i + batch_size, len(all_docs))
        with collection.batch.dynamic() as batch:
            for j in range(i, batch_end):
                batch.add_object(
                    properties={
                        "law": all_meta[j]["law"],
                        "article_id": all_meta[j]["article_id"],  # Đổi từ "id"
                        "title": all_meta[j]["title"],
                        "chapter": all_meta[j]["chapter"],
                        "text": all_docs[j],
                        "source_file": all_meta[j]["source_file"],
                        "granularity": all_meta[j]["granularity"],
                        "slice_id": all_meta[j]["slice_id"],
                    },
                    vector=vecs[j].tolist(),
                )
        print(f"  Uploaded {batch_end}/{len(all_docs)}")
    
    print("✓ Weaviate upload complete")
    client.close()
    print("✅ Built Weaviate index:", len(all_docs))
except Exception as e:
    print(f"❌ Error uploading to Weaviate: {e}")
    exit(1)
