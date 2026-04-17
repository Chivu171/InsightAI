# InsightAI 🤖 - Intelligent Document RAG Assistant

InsightAI is a full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, CSV, TXT) and have intelligent conversations with their content.

---

## 🛠️ Tech Stack & Engineering Choices

### **Frontend**
- **Framework**: **React 18** with **TypeScript**.
- **Build Tool**: **Vite** (for fast development and optimized production builds).
- **Styling**: **TailwindCSS** (for a modern, responsive UI).
- **Feature Set**: Chat interface, session-based message history, file upload management, and real-time loading states.

### **Backend**
- **Framework**: **FastAPI** (Python). Chosen for its high performance, asynchronous support, and automatic OpenAPI documentation.
- **File Parsing**: 
    - **PDF**: `PyMuPDF (fitz)` - Chosen for its speed and superior accuracy in extracting text compared to PyPDF2.
    - **CSV**: Native `csv` library with `io.StringIO` for row-based document transformation.
    - **TXT**: Standard UTF-8 decoding and cleaning.
- **RAG Engine**:
    - **Orchestration**: Custom implementation focusing on Hybrid Search.
    - **Vector DB**: **FAISS** (Facebook AI Similarity Search) for efficient local similarity search.
    - **Keyword Search**: **BM25 (rank_bm25)** for robust term-based retrieval.

### **LLM & Embeddings**
- **LLM**: **Google Gemini 1.5/2.0 Flash** (via API) or **Local LLMs** (via LM Studio/Ollama provider).
- **Embeddings**: `BAAI/bge-small-en-v1.5` - Provides a great balance between embedding speed and retrieval accuracy.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` - Used for fine-tuning the top-k results in the Fact-Checking pipeline.

---

## 🏗️ Architecture Decisions & Trade-offs

### 1. **Hybrid Retrieval (BM25 + Dense Search)**
- **Why**: Traditional vector search often misses specific keywords or technical jargon. By combining BM25 with Semantic Search via **Reciprocal Rank Fusion (RRF)**, the system ensures robustness and precision.
- **Trade-off**: Higher latency during retrieval, which we mitigated by using `faiss-cpu` for extremely fast vector operations.

### 2. **Dual-Path Routing: Fact vs. Summary**
- **Decision**: The system automatically classifies queries. 
    - **Fact queries** use high-granularity chunks for precision.
    - **Summary queries** use a dedicated **Block-based pipeline** (Large context chunks + TextRank + MMR).
- **Why**: Summarizing based on tiny 200-character chunks often leads to loss of information context. Blocks provide the "Full Story" to the LLM.

### 3. **TextRank + MMR Refinement**
- **Decision**: Instead of feeding 4000+ words to the LLM, we use TextRank to identify important sentences and **Maximal Marginal Relevance (MMR)** to ensure those sentences cover diverse topics without redundancy.
- **Trade-off**: Increases computation time on the backend but significantly reduces LLM input tokens and prevents "hallucinations" caused by information overload.

---

## 🚀 Setup Instructions

### **1. Prerequisites**
- Python 3.9+
- Node.js 18+

### **2. Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Create a `.env` file in `backend/`:
```env
GOOGLE_API_KEY=your_key_here
# Optional for local LLM:
LMSTUDIO_BASE_URL=http://localhost:1234
```
Start server:
```bash
uvicorn main:app --reload
```

### **3. Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🔮 What I would improve with more time

1. **Persistent Conversation History**: Currently, history is maintained in the frontend session. Moving this to a database (e.g., SQLite/PostgreSQL) would allow users to resume chats across devices.
2. **Streaming Responses**: implementing Server-Sent Events (SSE) to stream AI responses word-by-word for a better user experience.
3. **Dockerization**: Providing a `docker-compose.yml` to spin up the entire stack with a single command.
4. **Graph-RAG Integration**: For extremely complex documents, building a Knowledge Graph would allow the system to answer questions that require multi-hop reasoning better than standard vector search.
5. **Evaluation Pipeline**: Implementing a system like RAGAS to measure faithfulness and relevance quantitatively.

---
Built as a technical assignment for **ActiveFence**.
