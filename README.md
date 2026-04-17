# InsightAI

InsightAI is a full-stack RAG chatbot for asking questions over uploaded documents. It includes a FastAPI backend, a React frontend, and a hybrid retrieval pipeline using BM25 + dense FAISS.

## What it supports

- Upload PDF, TXT, CSV, DOCX, and image files
- Background indexing with progress updates
- Question answering with citations
- Session-based chat history per backend process
- Local-first LLM support with optional Gemini fallback

## Run with Docker

### Prerequisites

- Docker Desktop or Docker Engine with Compose support
- A `backend/.env` file for LLM configuration

### 1) Configure backend env

Create `backend/.env`:

```env
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_MODEL=your-local-model-name

# Optional fallback
# GOOGLE_API_KEY=your_google_api_key
# GOOGLE_MODEL=gemini-2.5-flash
```

If you use Google Gemini, make sure `GOOGLE_API_KEY` is set. If you use a local model through LM Studio, make sure the LM Studio server is running before starting the stack.

### 2) Build and start

```bash
docker compose up --build
```

### 3) Open the app

- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:8000](http://localhost:8000)

## Useful endpoints

- `POST /upload` - upload a file and build the index
- `GET /progress` - indexing progress
- `POST /queryHybrid` - ask a question
- `POST /reset` - clear the index
- `GET /status` - backend status

## Docker layout

- `backend` runs `uvicorn main:app --host 0.0.0.0 --port 8000`
- `frontend` runs the Vite dev server on `0.0.0.0:5173`
- Both services use bind mounts for local development

## Notes

- The backend image is built from `backend/Dockerfile`
- The frontend image is built from `frontend/Dockerfile`
- `.dockerignore` files are included to keep secrets, caches, and build artifacts out of the images

## Troubleshooting

If the app starts but queries fail, check:

1. `backend/.env` is present and valid
2. LM Studio or Gemini is reachable
3. The backend container logs for indexing or model errors

If Docker build fails, try:

```bash
docker compose build --no-cache
docker compose logs -f backend
docker compose logs -f frontend
```
