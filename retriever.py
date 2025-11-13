# retriever.py
# -*- coding: utf-8 -*-
"""
Retriever for Vietnamese Law QA System (Weaviate v4)
- Hybrid retrieval (BM25 + vector) on LawChunks
- Cross-Encoder reranking (BAAI/bge-reranker-v2-m3)
- Embed model: BAAI/bge-m3 (same as indexing)
- Rerank trên: rerank_title + rerank_body
- Context cho LLM: law + chapter + section + Điều + Khoản + Điểm + nội dung
"""

import os
import re
import torch
import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------- ENV SETUP ----------------
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
LEGAL_HINT_RE = re.compile(r"\b(Chương|Mục|Điều|Khoản|Điểm)\s+[IVXLC\d]+", re.IGNORECASE)
# Chỉ match số khi đi kèm với từ khóa pháp lý: "Điều 15", "Khoản 2", v.v.

# Phát hiện thông tin số cụ thể trong query (giờ, km/h, triệu, số lần...)
NUMERIC_INFO_RE = re.compile(r"\b(\d+)\s*(giờ|km/h|triệu|nghìn|đồng|lần|ngày|tháng|năm|%|phần trăm|cm3|cc|tấn|km|m|kW)\b", re.IGNORECASE)


def tune_alpha_and_pool(query: str, base_alpha: float = 0.55, k: int = 5):
    """
    Heuristic:
      - Query có chỉ mục pháp lý cụ thể (Điều X, Khoản Y...) → thiên BM25 mạnh
      - Query có thông tin số cụ thể (22 giờ, 120 km/h...) → thiên BM25 vừa
      - Query mô tả hành vi thuần ngôn ngữ tự nhiên → thiên semantic
    """
    alpha = base_alpha
    initial_k = max(10, k * 5)

    if LEGAL_HINT_RE.search(query):
        # Có chỉ mục pháp lý cụ thể như "Điều 15", "Khoản 2"
        alpha = max(0.30, base_alpha - 0.25)   # thiên BM25 mạnh nhất
        initial_k = max(15, k * 6)
    elif NUMERIC_INFO_RE.search(query):
        # Có thông tin số cụ thể như "22 giờ", "120 km/h"
        alpha = max(0.40, base_alpha - 0.15)   # thiên BM25 vừa phải
        initial_k = max(12, k * 5)
    else:
        # Query ngôn ngữ tự nhiên thuần túy
        alpha = min(0.75, base_alpha + 0.20)   # thiên semantic mạnh
        initial_k = max(10, k * 4)

    return alpha, initial_k


# ---------------- RETRIEVAL FUNCTION ----------------
def retrieve(question: str, k: int = 5, base_alpha: float = 0.55):
    """
    Hybrid retrieval + Cross-Encoder reranking
    Returns: (context, sources)
    """
    print(f"\n🔍 Retrieving for question: {question}")

    # 1) Heuristic alpha & pool size
    alpha, initial_k = tune_alpha_and_pool(question, base_alpha=base_alpha, k=k)
    print(f"   ▶ alpha={alpha:.2f}, initial_k={initial_k}")

    # 2) Encode question → dense vector
    qv = emb_model.encode([question], normalize_embeddings=True).astype("float32")

    # 3) Hybrid search (Weaviate v4)
    resp = collection.query.hybrid(
        query=question,
        vector=qv[0].tolist(),
        alpha=alpha,
        limit=initial_k,
        return_properties=[
            "law", "law_code",
            "chapter", "section",
            "article_no", "article_title",
            "clause_no", "point", "bullet_idx",
            "granularity",
            "header", "display_citation",
            "path_text",
            "clause_head",
            "text",
            "rerank_title", "rerank_body",
            "source_file",
        ],
    )

    candidates = []
    for obj in resp.objects or []:
        p = obj.properties or {}
        rr_title = (p.get("rerank_title") or "").strip()
        rr_body = (p.get("rerank_body") or "").strip()

        # fallback nếu body rỗng
        if not rr_body:
            rr_body = (p.get("text") or "").strip()

        rerank_text = (rr_title + "\n" + rr_body).strip()
        if not rerank_text:
            continue

        candidates.append(
            {
                "rerank_text": rerank_text,
                "props": p,
            }
        )

    if not candidates:
        print("⚠️ No candidates found.")
        return "", []

    # 4) Cross-Encoder rerank
    print(f"💡 Reranking {len(candidates)} candidates...")
    pairs = [[question, c["rerank_text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    topk = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k]

    # 5) Compose context + sources cho LLM
    contexts, sources = [], []
    for c in topk:
        p = c["props"]

        law = (p.get("law") or "").strip()
        chapter = (p.get("chapter") or "").strip()
        section = (p.get("section") or "").strip()
        article_no = (p.get("article_no") or "").strip()
        article_title = (p.get("article_title") or "").strip()
        clause_no = (p.get("clause_no") or "")  # TEXT trong schema
        clause_head = (p.get("clause_head") or "").strip()
        point = (p.get("point") or "").strip()
        body_for_ctx = (p.get("text") or "").strip()  # dùng text gốc cho context

        lines = []

        # Luật / chương / mục / điều
        if law:
            lines.append(law)
        if chapter:
            lines.append(chapter)
        if section:
            lines.append(section)
        if article_no or article_title:
            art_line = f"Điều {article_no}".strip()
            if article_title:
                art_line += f". {article_title}"
            lines.append(art_line)

        # Phân biệt 2 trường hợp: có Điểm hay không
        if point:
            # 👉 LEAF = ĐIỂM: CẦN cả nội dung khoản cha + nội dung điểm
            if clause_no and clause_head:
                lines.append(f"Khoản {clause_no}. {clause_head}")
            elif clause_no:
                lines.append(f"Khoản {clause_no}")

            lines.append(f"Điểm {point})")

            if body_for_ctx:
                lines.append(body_for_ctx)

        else:
            # 👉 LEAF = KHOẢN (không có điểm): chỉ ghi label + nội dung khoản,
            # KHÔNG lặp lại clause_head nếu nó gần trùng text
            if clause_no:
                lines.append(f"Khoản {clause_no}")
            if body_for_ctx:
                lines.append(body_for_ctx)

        ctx_chunk = "\n".join(lines).strip()
        if ctx_chunk:
            contexts.append(ctx_chunk)

        src = p.get("display_citation") or p.get("header", "") or ""
        sources.append(f"{law} – {src}" if law and src else (src or law))

    context = "\n\n".join(contexts)
    print(f"✅ Retrieved {len(contexts)} top chunks")
    return context, sources


