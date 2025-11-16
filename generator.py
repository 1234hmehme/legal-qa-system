# generator.py
# -*- coding: utf-8 -*-
"""
Legal Answer Generator for Vietnamese Law QA (Gemini + Weaviate RAG)
- Lấy context từ retriever (Weaviate hybrid + reranker)
- Gọi Gemini 2.5 Flash để sinh câu trả lời
- Format OPTION B:
    1) Trả lời trực tiếp, ngắn gọn, rõ ràng
    2) Nếu có mức phạt → liệt kê theo từng loại đối tượng/phương tiện
    3) Cuối cùng luôn có dòng: "Căn cứ pháp lý: ..." (liệt kê Điều/Khoản/Điểm, không trích nguyên văn)
"""

import os
import re
from typing import List, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

# --- retriever: dùng BM25+pyvi custom retriever ---
from retriever_custom import retrieve  # retrieve(question: str, k: int = 5)


# ===================== ENV & MODEL CONFIG =====================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY không được thiết lập. Hãy export GEMINI_API_KEY=your_api_key")

genai.configure(api_key=api_key)

# Có thể đổi sang "gemini-2.5-pro" nếu cần chất lượng cao hơn
MODEL_NAME = "gemini-2.5-flash"

GEN_CONFIG = {
    "temperature": 0.1,   # trả lời ổn định, ít bịa
    "top_p": 0.9,
    "top_k": 40,
    "candidate_count": 1,
}


# ===================== PROMPTS (OPTION B) =====================
SYSTEM_INSTRUCTION = """
Bạn là trợ lý pháp lý tiếng Việt cho lĩnh vực giao thông đường bộ.
Bạn phải trả lời CHÍNH XÁC tuyệt đối dựa trên các đoạn luật được cung cấp trong context.

Quy tắc bắt buộc:
1) Trả lời trực tiếp, ngắn gọn, rõ ràng, dễ hiểu.
2) Nếu liên quan đến mức phạt hoặc nghĩa vụ:
   - Liệt kê rõ theo từng loại đối tượng/phương tiện (nếu có).
   - Ghi rõ khoảng tiền phạt, hình thức xử phạt (nếu context có).
3) Ở cuối câu trả lời, luôn ghi một dòng:
   "Căn cứ pháp lý: ..."
   và liệt kê các căn cứ dạng ngắn gọn:
   - điểm ... khoản ... Điều ... Nghị định ...
   - khoản ... Điều ... Luật ...
   Không trích nguyên văn nội dung luật trong phần "Căn cứ pháp lý".
4) Tuyệt đối KHÔNG bịa thêm điều luật, điều khoản hay mức phạt không xuất hiện trong context.
5) Không đưa lời khuyên pháp lý chủ quan, chỉ diễn giải và hệ thống hóa nội dung từ luật.
6) Nếu trong context KHÔNG có thông tin đủ rõ để trả lời câu hỏi, bạn PHẢI trả lời: 
   "Trong các trích dẫn luật được cung cấp, không có đủ thông tin để trả lời chính xác câu hỏi này."
   và KHÔNG được suy luận thêm ngoài nội dung context.
7) Nếu người dùng hỏi ngoài phạm vi giao thông đường bộ, hãy nói rõ rằng hệ thống chỉ được huấn luyện trên luật giao thông đường bộ và từ chối trả lời.
Mục tiêu: câu trả lời ngắn gọn – chính xác – dùng được ngay.
"""

USER_PROMPT_TMPL = """\
Câu hỏi của người dùng:
{question}

Các đoạn luật (trích từ văn bản pháp luật, bạn PHẢI bám sát):
{context}

Yêu cầu:
- Trước hết hãy trả lời trực tiếp câu hỏi, trình bày mạch lạc, dễ hiểu.
- Nếu câu hỏi liên quan đến mức phạt, trách nhiệm, nghĩa vụ thì hãy liệt kê rõ theo từng trường hợp liên quan trong context (ví dụ: theo loại xe, đối tượng vi phạm, hành vi...).
- Ở cuối câu trả lời, thêm một dòng riêng:
  "Căn cứ pháp lý: ..." 
  và liệt kê các căn cứ pháp lý dựa trên thông tin trong dòng [Căn cứ: ...] ở đầu mỗi đoạn luật.
  Ví dụ: nếu thấy "[Căn cứ: khoản 4 Điều 2 Luật số 35/2024/QH15]" thì ghi "khoản 4 Điều 2 Luật số 35/2024/QH15".
- Không được viện dẫn bất kỳ căn cứ nào không có trong các đoạn luật ở trên.
"""


# ===================== UTILS =====================
def _dedupe_sources(sources: List[str]) -> List[str]:
    """Loại bỏ trùng citation, giữ thứ tự xuất hiện đầu tiên."""
    seen = set()
    out: List[str] = []
    for s in sources:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if key and key not in seen:
            out.append(s)
            seen.add(key)
    return out


def _truncate_context(ctx: str, max_chars: int = 20000) -> str:
    """Giới hạn độ dài context để tránh quá token."""
    if len(ctx) <= max_chars:
        return ctx
    return ctx[:max_chars] + "\n… (đã cắt ngắn context do quá dài)"


def _build_prompt(question: str, context: str) -> str:
    return USER_PROMPT_TMPL.format(question=question.strip(), context=context.strip())


# ===================== CORE FUNCTION =====================
def generate_answer(question: str, k: int = 5, base_alpha: float = 0.55) -> Tuple[str, List[str]]:
    """
    Pipeline:
      1) Gọi retriever để lấy context + sources (alpha tự động tune)
      2) Gọi Gemini để sinh câu trả lời theo format OPTION B
      3) Trả về (answer_text, sources_list)
    - sources_list: danh sách căn cứ (đã dedupe), dùng để hiển thị thêm nếu muốn.
    """
    import time

    # 1) Lấy context từ retriever
    t0 = time.time()
    context, sources = retrieve(question, k=k)
    print(f"⏱️  Retrieval time: {time.time() - t0:.2f}s")

    if not context.strip() or len(context) < 300:
        return "Trong các trích dẫn luật được cung cấp, không có đủ thông tin để trả lời chính xác câu hỏi này.", []

    truncated_context = _truncate_context(context, max_chars=20000)
    sources = _dedupe_sources(sources)

    # 2) Gọi Gemini sinh câu trả lời
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=GEN_CONFIG,
    )
    prompt = _build_prompt(question, truncated_context)

    try:
        t1 = time.time()
        resp = model.generate_content(prompt)
        print(f"⏱️  Gemini API time: {time.time() - t1:.2f}s")
        text = (resp.text or "").strip()
        if not text:
            text = "Không tìm thấy đủ thông tin trong các văn bản luật đã được lập chỉ mục để trả lời câu hỏi này."
    except Exception as e:
        text = f"Lỗi khi gọi Gemini API: {e}"

    # 3) Trả về: câu trả lời + danh sách nguồn
    return text, sources


# ===================== QUICK TEST =====================
if __name__ == "__main__":
    q = "Một tài xế gây tai nạn và bỏ chạy khỏi hiện trường mà không dừng lại để cấp cứu người bị nạn. Hành vi này sẽ bị xử lý như thế nào theo luật giao thông đường bộ hiện hành?"
    ans, cites = generate_answer(q, k=5)

    print("\n================= ANSWER =================\n")
    print(ans)

    print("\n================= SOURCES ================\n")
    for i, src in enumerate(cites, 1):
        print(f" [{i}] {src}")