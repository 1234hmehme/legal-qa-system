# Vietnamese Law QA System 🇻🇳

Hệ thống hỏi đáp pháp luật giao thông đường bộ Việt Nam sử dụng RAG (Retrieval-Augmented Generation).

## 📋 Tổng quan

System này kết hợp:
- **BM25 + pyvi**: Keyword search với Vietnamese tokenization
- **Dense Retrieval**: Semantic search với Alibaba-NLP/gte-multilingual-base
- **Hybrid Search**: Dynamic alpha tuning dựa trên query pattern
- **Reranking**: BAAI/bge-reranker-v2-m3 cross-encoder
- **Generation**: Gemini 2.5 Flash cho câu trả lời tiếng Việt

## 🏗️ Kiến trúc

```
          ┌─────────────┐
          │   User      │
          │  Question   │
          └──────┬──────┘
                 │
                 v
┌──────────────────────────────────────┐
│          Hybrid Retrieval            │
│  ┌──────────┐       ┌──────────┐     │
│  │  BM25    │       │  Dense   │     │
│  │ (pyvi)   │       │ (gte-m)  │     │
│  └────┬─────┘       └────┬─────┘     │
│       └─────────┬────────┘           │
│                 v                    │
│         ┌───────────────┐            │
│         │ Alpha Tuning  │            │
│         └───────┬───────┘            │
│                 v                    │
│         ┌───────────────┐            │
│         │   Reranker    │            │
│         │  (bge-m3)     │            │
│         └───────┬───────┘            │
└─────────────────┼────────────────────┘
                  │
                  v
          ┌───────────────┐
          │    Context    │
          │   (Top 5)     │
          └───────┬───────┘
                  │
                  v
          ┌───────────────┐
          │    Gemini     │
          │  2.5 Flash    │
          └───────┬───────┘
                  │
                  v
          ┌───────────────┐
          │    Answer     │
          │  + Sources    │
          └───────────────┘
```

## 🚀 Setup

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Tạo file `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Khởi động Weaviate

```bash
docker-compose up -d
```

### 4. Chuẩn bị dữ liệu

Đặt các file văn bản luật vào `data/raw/`:
- `nghidinhso-168-2024-NĐ-CP.txt`
- `luatso-36-2024-QH15.txt`
- `luatso-35-2024-QH15.txt`

### 5. Xử lý dữ liệu & Build Index

```bash
cd backend

# Bước 1: Chunk văn bản luật
python clean_and_split.py

# Bước 2: Build Weaviate index
python build_index.py
```

### 6. Chạy ứng dụng

```bash
# Frontend (Recommended) - Streamlit Web UI
cd frontend
streamlit run app.py

# Hoặc CLI - Command line interface
cd backend
python rag_qa.py
```

## 💬 Sử dụng

### Chạy Frontend (Web UI - Streamlit) - Recommended ⭐

```bash
cd frontend
streamlit run app.py
```

**Features:**
- ✨ Welcome screen với giới thiệu và 3 ảnh minh họa
- 💬 Chat interface hiện đại với streaming text
- 📚 Source attribution (xem nguồn pháp luật tham khảo)
- ⏱️ Performance metrics (retrieval + generation time)
- 💡 Sample questions để bắt đầu nhanh
- 🎚️ Settings sidebar: điều chỉnh số chunks, xóa lịch sử
- 📊 Hiển thị metrics: thời gian xử lý, số chunks sử dụng

### Chạy CLI (Command Line)

```bash
cd backend
python rag_qa.py
```

Hoặc trong Python:

```python
from retriever_custom import retrieve
from generator import generate_answer

question = "Kết cấu hạ tầng đường bộ bao gồm những gì?"

# Step 1: Retrieval
context, sources = retrieve(question, k=5)

# Step 2: Generation
answer, sources = generate_answer(question, context, sources)

print(answer)
```

Hoặc đơn giản hơn:

```python
from rag_qa import ask_law

question = "Kết cấu hạ tầng đường bộ bao gồm những gì?"
answer, sources = ask_law(question, k=5)

print(answer)
```

### Test Retriever

```bash
cd backend
python test_retriever.py
```

## 📊 Pipeline Chi tiết

### 1. Data Processing (`clean_and_split.py`)

- **Input**: Raw text files (.txt)
- **Output**: Structured JSON chunks (`data/processed/`)
- **Process**:
  - Hierarchical chunking: Điều → Khoản → Điểm → Bullet
  - Full context enrichment với tags `[CHAPTER]`, `[ARTICLE]`, `[CLAUSE]`, `[POINT]`
  - Sliding window cho chunks dài (max 1500 tokens)

### 2. Indexing (`build_index.py`)

- **Vector Database**: Weaviate
- **Embedding Model**: Alibaba-NLP/gte-multilingual-base
- **Schema**:
  - BM25 fields: `article_no`, `article_title`, `clause_no`, `point`, `clause_head`, `text`
  - Vector field: từ `enriched_text`
  - Display fields: `display_citation`, `header`, `path_text`

### 3. Retrieval (`retriever_custom.py`)

#### BM25 Retrieval
- Tokenization: pyvi (Vietnamese word segmentation)
- Fields: article_no + article_title + clause_no + point + clause_head + text

#### Dense Retrieval
- Model: Alibaba-NLP/gte-multilingual-base
- Input: `enriched_text` (full context với tags)

#### Hybrid Search
- Formula: `score = α × dense_score + (1-α) × bm25_score`
- Dynamic alpha tuning:
  - Query có "Điều X", "Khoản Y" → α = 0.30 (favor BM25)
  - Query có số liệu (km/h, triệu đồng) → α = 0.40
  - Query semantic → α = 0.75 (favor dense)

#### Reranking
- Model: BAAI/bge-reranker-v2-m3
- Top 20 candidates → Top 5 final results

### 4. Generation (`generator.py`)

- **Model**: Gemini 2.5 Flash
- **Prompt Strategy**:
  - System instruction: Quy tắc trả lời chính xác, không bịa
  - Context: Top 5 chunks với citation `[Căn cứ: ...]`
  - Output format: Trả lời + "Căn cứ pháp lý: ..."

## 🎯 Features

 **Hierarchical Chunking**: Cấu trúc Điều → Khoản → Điểm

 **Context Enrichment**: Full context với tags cho embedding

 **Vietnamese Tokenization**: pyvi cho BM25

 **Dynamic Alpha Tuning**: Tự động điều chỉnh theo query pattern

 **Cross-Encoder Reranking**: Độ chính xác cao

 **Citation Tracking**: Ghi rõ căn cứ pháp lý

 **Gemini Integration**: Vietnamese legal answer generation

 **Modern Web UI**: Streamlit chat interface với streaming

 **Source Attribution**: Xem nguồn tham khảo trong Expander

 **Performance Metrics**: Hiển thị thời gian retrieval/generation

 **Sample Questions**: Gợi ý câu hỏi mẫu

 **Adjustable Settings**: Điều chỉnh số chunks, xóa lịch sử

## 📂 Cấu trúc Project

```
legal-qa-system/
├── frontend/                   # Streamlit Web UI
│   ├── app.py                 # Main chat interface
│   ├── assets/                # Hình ảnh minh họa
│   │   ├── law.png
│   │   ├── traffic.png
│   │   └── legal.png
│   └── bm25_index.pkl         # BM25 index cache
│
├── backend/                   # RAG Pipeline
│   ├── clean_and_split.py     # Data processing & chunking
│   ├── build_index.py         # Weaviate indexing
│   ├── retriever_custom.py    # Hybrid retrieval + reranking
│   ├── generator.py           # Gemini answer generation
│   ├── rag_qa.py              # Main QA pipeline
│   └── test_retriever.py      # Test retrieval
│
├── docker/                    # Docker setup
│   └── docker-compose.yml     # Weaviate container
│
├── data/
│   ├── raw/                   # Raw law text files
│   └── processed/             # Processed JSON chunks
│
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
└── README.md
```

## 🔧 Configuration

### Models

```python
# retriever_custom.py
EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# generator.py
LLM_MODEL = "gemini-2.5-flash"
```

### Retrieval Parameters

```python
INITIAL_K = 20    # Candidates before rerank
FINAL_K = 5       # Top results after rerank
BASE_ALPHA = 0.55 # Default hybrid alpha
```

### Chunking Parameters

```python
MAX_TOKENS_LEAF = 1500
WIN_TOK = 900
OVERLAP_TOK = 300
```

## 📝 Văn bản pháp luật

Hệ thống hỗ trợ:
- Nghị định số 168/2024/NĐ-CP (Xử phạt vi phạm giao thông)
- Luật số 36/2024/QH15 (Trật tự an toàn giao thông đường bộ)
- Luật số 35/2024/QH15 (Đường bộ)

## 🧪 Testing

```bash
# Test retrieval
python test_retriever.py

# Test full pipeline
python rag_qa.py
```

## 🐛 Troubleshooting

### Weaviate không kết nối được
```bash
docker-compose down
docker-compose up -d
docker ps  # Check container running
```

### BM25 index bị lỗi
```bash
rm bm25_index.pkl
python clean_and_split.py  # Rebuild
```

### Out of memory
- Giảm `INITIAL_K` trong retriever
- Giảm batch size trong build_index.py

## 📚 References

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Gemini API](https://ai.google.dev/gemini-api/docs)
- [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## 📄 License

MIT License

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
