import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cấu hình API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY không được thiết lập. Hãy export GEMINI_API_KEY=your_api_key")

genai.configure(api_key=api_key)

#kiểm tra các model hỗ trợ generateContent
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)


# Sử dụng alias model mới nhất
model = genai.GenerativeModel("gemini-2.5-flash")


PROMPT = """
Bạn là trợ lý pháp lý tiếng Việt chuyên nghiệp.
Dựa trên các đoạn luật sau, hãy trả lời câu hỏi một cách:
- Ngắn gọn, rõ ràng nhưng đầy đủ
- Đúng nội dung luật, trích dẫn cụ thể Điều/Khoản nếu có
- Sử dụng tiếng Việt tự nhiên, dễ hiểu

Nếu không có thông tin liên quan trong các đoạn luật được cung cấp, trả lời "Không đủ thông tin trong luật được cung cấp."

CÂU HỎI: {question}

CÁC ĐOẠN LUẬT LIÊN QUAN:
{context}

TRẢ LỜI:
"""

def generate_answer(question, context):
    try:
        # Giới hạn context để tránh vượt quá token limit (Gemini 2.5 Flash hỗ trợ ~1M tokens)
        max_context_length = 10000  
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... (đã cắt ngắn do context quá dài)"
        
        prompt = PROMPT.format(question=question, context=context)
        
        print("Đang gọi Gemini API...")
        response = model.generate_content(prompt)
        
        if response.text:
            return response.text.strip()
        else:
            return "Không thể tạo câu trả lời từ Gemini API."
            
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API: {e}")
        return f"Lỗi khi gọi Gemini API: {str(e)}"
