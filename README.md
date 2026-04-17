# InsightAI 🤖 — AI Chatbot with File QA (RAG)

InsightAI is a full-stack chatbot that lets you upload a document (PDF/TXT/CSV) and ask questions about its content. It implements a simple session-based chat UX on the frontend and a FastAPI backend with local-first LLM integration and a hybrid RAG pipeline.

## Features (assignment mapping)
- Chat UI (React/TypeScript) with message history, loading states, and error handling.
- File upload (PDF/TXT/CSV) and background indexing with progress indicator.
- AI responses rendered as formatted Markdown (lists, code, math via KaTeX).
- Ask questions grounded in uploaded file content (citations/snippets returned by the API).
- Conversation history maintained per `session_id` (within the running backend process).

## Tech stack
- Frontend: React 18 + TypeScript + Vite + TailwindCSS.
- Backend: Python + FastAPI.
- Parsing: PyMuPDF (PDF), Python `csv` (CSV), UTF-8 text read (TXT). (`.docx` is also supported as an extra.)
- Retrieval: Hybrid search (BM25 + dense FAISS) with optional reranking.
- LLM:
  - Local: LM Studio OpenAI-compatible endpoint (`/v1/chat/completions`) with an open-source model you run locally.
  - Optional fallback: Google Gemini (only if you set `GOOGLE_API_KEY`).

## API
- `POST /upload`: upload + trigger (re)index in background
- `GET /progress`: indexing status/progress
- `POST /queryHybrid`: chat endpoint (expects `query` and optional `session_id`)
- `POST /reset`: clears the index
- `GET /status`: basic health/status

Backend runs on `http://localhost:8000`, frontend on `http://localhost:5173`.

## Setup (local dev)
Prereqs: Python 3.9+, Node.js 18+

### 1) Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Create `backend/.env` (local LLM recommended):
```env
# Local LLM via LM Studio (OpenAI-compatible)
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_MODEL=your-local-model-name

# Optional fallback (not required if local LLM is running)
# GOOGLE_API_KEY=...
```

### 2) Frontend
```bash
cd frontend
npm install
npm run dev
```

## Docker (optional)
This repo includes `docker-compose.yml` to run `frontend` + `backend`:
```bash
docker compose up --build
```

## Architecture decisions & trade-offs
- Hybrid retrieval (BM25 + dense FAISS): better keyword recall + semantic matching, at the cost of slightly higher retrieval compute.
- Background indexing: upload triggers a background job; UI polls `/progress` for a responsive UX.
- Session memory: conversation history is kept in-process (simple + fast), but is lost on backend restart (no DB yet).

## What I’d improve with more time
- True streaming responses (SSE/WebSocket) for token-by-token UI.
- Persist conversations + documents (SQLite/Postgres) and multi-user auth.
- More robust parsing (tables in PDF/CSV schema awareness) and better citation UX.
- Tests for endpoints + parsing and a small evaluation harness for retrieval quality.
