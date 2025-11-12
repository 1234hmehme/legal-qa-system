# generator.py
# -*- coding: utf-8 -*-
"""
Legal Answer Generator (Gemini) for Vietnamese Law QA
- Lấy context từ retriever (Weaviate hybrid + reranker)
- Gọi Gemini 2.5 Flash để sinh câu trả lời ngắn gọn, chuẩn luật, có trích dẫn
"""

import os
import re
from typing import List, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# --- Lưu ý: cần có retriever.py trong cùng thư mục (đã code ở bước trước) ---
from retriever import retrieve  # retrieve(question: str, k: int = 5, base_alpha: float = 0.55)

# ===================== ENV & MODEL =====================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY không được thiết lập. Hãy export GEMINI_API_KEY=your_api_key")

genai.configure(api_key=api_key)

# Bạn có thể đổi qua "gemini-2.5-pro" nếu muốn chất lượng cao hơn:
MODEL_NAME = "gemini-2.5-flash"

# Cấu hình sinh
GEN_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "candidate_count": 1,
}

# ===================== PROMPT =====================
SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý pháp lý tiếng Việt. "
    "Trả lời chính xác, đầy đủ dựa trên các đoạn luật được cung cấp. "
    "Dùng đúng thuật ngữ pháp lý từ các đoạn luật. "
    "KHÔNG cần trích dẫn nguồn trong câu trả lời."
)

USER_PROMPT_TMPL = """\
Câu hỏi: {question}

Các đoạn luật:
{context}

Trả lời đầy đủ, chính xác dựa trên các đoạn luật trên. Dùng đúng thuật ngữ pháp lý. KHÔNG ghi nguồn hay căn cứ pháp lý.
"""

# ===================== UTILS =====================
def _dedupe_sources(sources: List[str]) -> List[str]:
    """Loại trùng citation, giữ nguyên thứ tự xuất hiện."""
    seen = set()
    out = []
    for s in sources:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if key and key not in seen:
            out.append(s)
            seen.add(key)
    return out

def _truncate_context(ctx: str, max_chars: int = 20000) -> str:
    if len(ctx) <= max_chars:
        return ctx
    return ctx[:max_chars] + "\n… (đã cắt ngắn context do quá dài)"

def _build_prompt(question: str, context: str) -> str:
    return USER_PROMPT_TMPL.format(question=question.strip(), context=context.strip())

# ===================== GENERATE =====================
def generate_answer(question: str, k: int = 6, base_alpha: float = 0.55) -> Tuple[str, List[str]]:
    """
    1) Gọi retriever để lấy context + sources
    2) Gọi Gemini để sinh câu trả lời
    3) Trả về (answer_text, citations_list)
    """
    import time
    
    # 1) Lấy context từ retriever
    t0 = time.time()
    context, sources = retrieve(question, k=k, base_alpha=base_alpha)
    print(f"⏱️  Retrieval time: {time.time()-t0:.2f}s")
    if not context.strip():
        return "Không đủ thông tin trong luật được cung cấp.", []

    sources = _dedupe_sources(sources)
    context = _truncate_context(context, max_chars=20000)

    # 2) Gọi Gemini
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=GEN_CONFIG,
    )
    prompt = _build_prompt(question, context)

    try:
        t1 = time.time()
        resp = model.generate_content(prompt)
        print(f"⏱️  Gemini API time: {time.time()-t1:.2f}s")
        text = (resp.text or "").strip()
        if not text:
            text = "Không đủ thông tin trong luật được cung cấp."
    except Exception as e:
        text = f"Lỗi khi gọi Gemini API: {e}"

    # Gemini tự trích dẫn trong "Căn cứ pháp lý" - không cần thêm sources nữa
    return text, sources

# ===================== QUICK TEST =====================
if __name__ == "__main__":
    q = "Sử dụng còi trong thời gian từ 22 giờ đến 05 giờ sáng trong khu dân cư thì bị phạt bao nhiêu tiền?"
    ans, cites = generate_answer(q, k=5, base_alpha=0.55)
    print("\n================= ANSWER =================\n")
    print(ans)
    print("\n================= SOURCES ================\n")
    for s in cites:
        print("-", s)
