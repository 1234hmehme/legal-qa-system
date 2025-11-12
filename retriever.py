# retriever.py
# -*- coding: utf-8 -*-
"""
Retriever for Vietnamese Law QA System (Weaviate v4)
- Hybrid retrieval (BM25 + vector) on LawChunks
- Cross-Encoder reranking (BAAI/bge-reranker-v2-m3)
- Embed model: BAAI/bge-m3 (same as indexing)
- Rerank trên: rerank_title + rerank_body (ít nhiễu, giàu ngữ cảnh)
"""

import os
import re
import torch
import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------- ENV SETUP ----------------
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.backends.mps.is_available = lambda: False  # type: ignore
torch.set_num_threads(1)

# ---------------- MODEL LOADING ----------------
print("🔹 Loading embedding model (BAAI/bge-m3)...")
emb_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("✓ Embedding model loaded on CPU")

print("🔹 Loading reranker model (BAAI/bge-reranker-v2-m3)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
print("✓ Reranker model loaded on CPU")

# ---------------- WEAVIATE CONNECTION ----------------
print("🌐 Connecting to Weaviate...")
client = weaviate.connect_to_local()
collection = client.collections.get("LawChunks")
print("✓ Connected to collection: LawChunks")

# ---------------- UTILS ----------------
LEGAL_HINT_RE = re.compile(r"\b(Chương|Mục|Điều|Khoản|Điểm)\b", re.IGNORECASE)
NUM_HINT_RE = re.compile(r"\b(\d{1,3})\b")

def tune_alpha_and_pool(query: str, base_alpha: float = 0.55, k: int = 5):
    """
    Heuristic:
      - Nếu query có chỉ mục pháp lý (Chương/Điều/Khoản/Điểm/số...), nghiêng BM25 hơn (alpha ↓).
      - Nếu query miêu tả hành vi bằng ngôn ngữ tự nhiên, nghiêng semantic hơn (alpha ↑ nhẹ).
    """
    alpha = base_alpha
    initial_k = max(10, k * 5)

    if LEGAL_HINT_RE.search(query) or NUM_HINT_RE.search(query):
        alpha = max(0.35, base_alpha - 0.2)   # thiên BM25 hơn
        initial_k = max(15, k * 6)            # pool nhiều hơn để rerank
    else:
        alpha = min(0.7, base_alpha + 0.1)    # thiên semantic hơn

    return alpha, initial_k

# ---------------- RETRIEVAL FUNCTION ----------------
def retrieve(question: str, k: int = 5, base_alpha: float = 0.55):
    """
    Hybrid retrieval + Cross-Encoder reranking
    Returns: (context, sources)
    """
    print(f"\n🔍 Retrieving for question: {question}")

    # Heuristic tune
    alpha, initial_k = tune_alpha_and_pool(question, base_alpha=base_alpha, k=k)
    print(f"   ▶ alpha={alpha:.2f}, initial_k={initial_k}")

    # 1) Encode question → dense vector (chuẩn với indexing)
    qv = emb_model.encode([question], normalize_embeddings=True).astype("float32")

    # 2) Hybrid search (Weaviate v4): BM25 + vector
    #    Lưu ý: v4 không nhận 'properties='; dùng 'return_properties' để lấy fields cần hiển thị/rerank.
    resp = collection.query.hybrid(
        query=question,
        vector=qv[0].tolist(),
        alpha=alpha,
        limit=initial_k,
        return_properties=[
            "law", "law_code", "header", "display_citation",
            "article_no", "clause_no", "point",
            "source_file", "path_text",
            "rerank_title", "rerank_body",
            "enriched_text",  # ✅ text đã có context khoản + mức phạt
            "text"  # backup/fallback
        ],
    )

    candidates = []
    for obj in resp.objects or []:
        p = obj.properties or {}
        # Ưu tiên rerank trên rerank_title + rerank_body
        rr_title = (p.get("rerank_title") or "").strip()
        rr_body  = (p.get("rerank_body") or "").strip()
        if not rr_body and not rr_title:
            # fallback rất hạn hữu: dùng text (leaf gốc)
            rr_body = (p.get("text") or "").strip()
        rerank_text = (rr_title + "\n" + rr_body).strip()

        if not rerank_text:
            continue

        candidates.append({
            "rerank_text": rerank_text,
            "props": p
        })

    if not candidates:
        print("⚠️ No candidates found.")
        return "", []

    # 3) Cross-Encoder rerank
    print(f"💡 Reranking {len(candidates)} candidates...")
    pairs = [[question, c["rerank_text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    topk = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k]

    # 4) Compose context + sources (hiển thị đầy đủ với enriched_text)
    contexts, sources = [], []
    for c in topk:
        p = c["props"]
        # Ưu tiên enriched_text (có context đầy đủ: chapter, article, clause_head)
        enriched = (p.get("enriched_text") or "").strip()
        if enriched:
            contexts.append(enriched)
        else:
            # Fallback: dùng text gốc
            header = p.get("header", "").strip()
            body = (p.get("text") or "").strip()
            if header and body:
                contexts.append(f"{header}: {body}")
            elif body:
                contexts.append(body)

        src = p.get("display_citation") or p.get("header", "") or ""
        law = p.get("law", "")
        sources.append(f"{law} – {src}" if law else src)

    context = "\n\n".join(contexts)
    print(f"✅ Retrieved {len(contexts)} top chunks")
    return context, sources

# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    try:
        q = "Theo luật mới, giấy phép lái xe hạng A1 cấp cho người lái xe mô tô hai bánh có dung tích xi-lanh đến bao nhiêu cm³?"
        ctx, srcs = retrieve(q, k=5, base_alpha=0.55)
        print("\n📘 Full Context (all chunks):\n")
        print(ctx)  # In đầy đủ không truncate
        print("\n📚 Sources:")
        for s in srcs:
            print(" -", s)
    finally:
        client.close()
        print("\n✓ Weaviate connection closed")
