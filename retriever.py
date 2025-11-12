import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import os

# Set environment variables to avoid segmentation fault
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable MPS to avoid segmentation fault
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(1)

# Load embedding model
emb = SentenceTransformer("BAAI/bge-m3", device="cpu")

# Load reranker model (cross-encoder)
print("Loading reranker model...")
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
print("✓ Reranker loaded")


# Connect to Weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("LawArticles")


def retrieve(question, k=5):
    """
    Hybrid search + Cross-Encoder Reranking
    """
    # 1. Mã hóa câu hỏi thành vector
    qv = emb.encode([question], normalize_embeddings=True).astype("float32")
    
    # 2. Hybrid search trong Weaviate (lấy nhiều candidates)
    initial_k = k * 5  # Lấy 5x để rerank
    response = collection.query.hybrid(
        query=question,
        vector=qv[0].tolist(),
        limit=initial_k,
        alpha=0.6,
    )
    
    # 3. Rerank bằng Cross-Encoder
    candidates = []
    for item in response.objects:
        props = item.properties
        text = props.get('text', '')
        candidates.append({
            'text': text,
            'props': props
        })
    
    # Score lại bằng cross-encoder (question, document)
    pairs = [[question, c['text']] for c in candidates]
    scores = reranker.predict(pairs)
    
    # Sắp xếp theo score và lấy top-k
    for i, candidate in enumerate(candidates):
        candidate['rerank_score'] = scores[i]
    
    candidates_sorted = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:k]
    
    # 4. Extract results
    ctxs, srcs = [], []
    for candidate in candidates_sorted:
        props = candidate['props']
        ctxs.append(props.get('text', ''))
        srcs.append(f"{props.get('law', '')} – {props.get('article_id', '')}")
    
    context = "\n\n".join(ctxs)
    return context, srcs
