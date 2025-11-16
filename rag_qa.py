from retriever_custom import retrieve
from generator import generate_answer

def ask_law(question, k=5):
    """
    Ask a legal question and get answer with sources
    
    Args:
        question: User question
        k: Number of chunks to retrieve (default: 5)
    
    Returns:
        (answer_text, sources_list)
    """
    import time
    
    # Step 1: Retrieve context
    print("🔍 Retrieving context...")
    t0 = time.time()
    context, sources = retrieve(question, k=k)
    print(f"⏱️  Retrieval time: {time.time() - t0:.2f}s")
    
    # Step 2: Generate answer
    print("🤖 Generating answer...")
    answer, sources = generate_answer(question, context, sources)
    
    return answer, sources

if __name__ == "__main__":

    q = """Tôi đi ô tô mà chạy sang làn đường dành cho xe máy thì bị xử phạt như thế nào?"""
    
    ans, src = ask_law(q, k=5)
    
    print("\n" + "="*60)
    print("Câu hỏi:", q)
    print("="*60)
    print("\n💡 Trả lời:")
    print(ans)
    
    print("\n📚 Nguồn:")
    for i, s in enumerate(src, 1):
        if s:
            print(f"  [{i}] {s}")
    print("="*60)
