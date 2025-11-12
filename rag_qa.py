from retriever import retrieve
from generator import generate_answer

def ask_law(question, k=5):  # Reduce to 5 documents to keep context shorter
    context, sources = retrieve(question, k=k)
    answer = generate_answer(question, context)
    
    # Loại bỏ trùng lặp sources nhưng giữ thứ tự
    unique_sources = []
    seen = set()
    for src in sources:
        if src not in seen:
            unique_sources.append(src)
            seen.add(src)
    
    source_str = "; ".join(unique_sources)
    final = f"{answer.strip()}\n\nTheo: {source_str}"
    return final, sources

if __name__ == "__main__":
    q = "người đi xe dàn hàng ba bị xử phạt như thế nào?"
    ans, src = ask_law(q, k=5)  # Use k=5
    print("Câu hỏi:", q)
    print("Trả lời:", ans)
