from retriever import retrieve

# Test search
question = "người đi xe dàn hàng ba bị xử phạt như thế nào?"
print(f"Câu hỏi: {question}\n")

context, sources = retrieve(question, k=5)

print("=" * 80)
print("🔍 KẾT QUẢ TÌM KIẾM (Hybrid Search)")
print("=" * 80)

print(f"\n📚 Nguồn tham chiếu ({len(sources)} documents):")
for i, src in enumerate(sources, 1):
    print(f"  {i}. {src}")

print(f"\n📄 Context (preview):")
print(context[:500] + "...")
