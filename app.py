#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Demo App for Vietnamese Law QA System - Chatbot Interface
"""

import streamlit as st
import time
from retriever_custom import retrieve
from generator import generate_answer

# Page config
st.set_page_config(
    page_title="Chatbot Luật Giao Thông",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f5f5f5;
    }
    
    /* Header */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chat-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    
    .chat-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .bot-message {
        background: white;
        color: #333;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        word-wrap: break-word;
    }
    
    .bot-icon {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        margin-right: 0.5rem;
        font-weight: bold;
    }
    
    .source-item {
        background: #fff8dc;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #ffc107;
        font-size: 0.85rem;
        color: #555;
    }
    
    .time-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Input container */
    .stChatInputContainer {
        border-top: 2px solid #e0e0e0;
        background: white;
        padding: 1rem 0;
    }
    
    /* Welcome message */
    .welcome-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 2rem 0;
    }
    
    .sample-question {
        background: #f8f9fa;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        border: 1px solid #dee2e6;
        transition: all 0.3s;
    }
    
    .sample-question:hover {
        background: #e9ecef;
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .metric-inline {
        display: inline-block;
        background: #f0f0f0;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        margin: 0.2rem;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True


# Header
st.markdown("""
<div class="chat-header">
    <div class="chat-title">⚖️ Chatbot Luật Giao Thông Đường Bộ</div>
    <div class="chat-subtitle">Hỏi đáp tức thì về Luật Giao Thông Việt Nam</div>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    k_value = st.slider(
        "Số lượng chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="Số đoạn văn bản được tìm kiếm"
    )
    
    st.markdown("---")
    
    st.markdown("""
    **🤖 Models:**
    - Embedding: `gte-multilingual-base`
    - Reranker: `bge-reranker-v2-m3`
    - LLM: `Gemini 2.5 Flash`
    
    **📚 Dữ liệu:**
    - Nghị định 168/2024/NĐ-CP
    - Luật 36/2024/QH15
    - Luật 35/2024/QH15
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_welcome = True
        st.rerun()

# Welcome screen
if st.session_state.show_welcome and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Chào mừng bạn!</h2>
        <p style='color: #666; margin: 1rem 0;'>
            Tôi là trợ lý AI chuyên về Luật Giao Thông Đường Bộ Việt Nam.<br>
            Hãy đặt câu hỏi hoặc chọn một trong các câu hỏi mẫu dưới đây:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample questions as buttons
    sample_questions = [
        "Kết cấu hạ tầng đường bộ bao gồm những gì?",
        "Người đi xe dàn hàng ba bị xử phạt như thế nào?",
        "Làn đường được định nghĩa là gì?",
        "Mức phạt cho người điều khiển xe không có GPLX?",
    ]
    
    col1, col2 = st.columns(2)
    for idx, question in enumerate(sample_questions):
        with col1 if idx % 2 == 0 else col2:
            if st.button(f"💡 {question}", key=f"sample_{idx}", use_container_width=True):
                st.session_state.show_welcome = False
                # Add to messages and process
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                st.rerun()

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
        <div style='text-align: right;'>
            <div class="user-message">
                <strong>🙋‍♂️ Bạn:</strong><br>
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='text-align: left;'>
            <div class="bot-message">
                <span class="bot-icon">⚖️</span>
                <strong>Trợ lý AI</strong><br><br>
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if 'sources' in message and message['sources']:
            with st.expander("📚 Nguồn tham khảo", expanded=False):
                for i, src in enumerate(message['sources'], 1):
                    if src:
                        st.markdown(f'<div class="source-item">[{i}] {src}</div>', unsafe_allow_html=True)
        
        # Show metrics if available
        if 'metrics' in message:
            m = message['metrics']
            st.markdown(f"""
            <div style='text-align: left; margin-top: 0.5rem;'>
                <span class="metric-inline">⏱️ {m['total']:.2f}s</span>
                <span class="metric-inline">🔍 {m['retrieval']:.2f}s</span>
                <span class="metric-inline">🤖 {m['generation']:.2f}s</span>
                <span class="metric-inline">📄 {m['chunks']} chunks</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Add spacing between messages
    st.markdown("<br>", unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("💬 Nhập câu hỏi của bạn...")

if user_input:
    # Hide welcome screen
    st.session_state.show_welcome = False
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Process and generate response
    with st.spinner("🤔 Đang suy nghĩ..."):
        start_time = time.time()
        
        try:
            # Retrieval
            t0 = time.time()
            context, sources = retrieve(user_input, k=k_value)
            retrieval_time = time.time() - t0
            
            # Generation
            t1 = time.time()
            answer, sources = generate_answer(user_input, context, sources)
            generation_time = time.time() - t1
            
            total_time = time.time() - start_time
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "metrics": {
                    "total": total_time,
                    "retrieval": retrieval_time,
                    "generation": generation_time,
                    "chunks": len(sources)
                }
            })
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Xin lỗi, đã xảy ra lỗi: {str(e)}"
            })
    
    st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;'>
    Built with ❤️ using Streamlit • Weaviate • Gemini
</div>
""", unsafe_allow_html=True)
