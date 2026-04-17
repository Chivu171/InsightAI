# InsightAI Backend (FastAPI)

FastAPI service that powers InsightAI:
- Upload a file (PDF/TXT/CSV) and build an index in the background
- Chat endpoint for asking questions grounded in the uploaded content
- Session-based conversation context (`session_id`) stored in-memory

## Endpoints
- `POST /upload` (alias: `POST /uploadHybrid`): upload + start indexing
- `GET /progress`: returns `{ status, progress }` (and a `hybrid` key for FE compatibility)
- `POST /queryHybrid`: `{ query, session_id? }` → `{ answer, sources }`
- `POST /debug/queryHybrid`: same as query + debug stages
- `POST /reset`: clears index
- `GET /status`: health/status

## Setup
Prereqs: Python 3.9+

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## LLM configuration
Local-first via LM Studio (recommended). Create `backend/.env`:
```env
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_MODEL=your-local-model-name

# Optional fallback
# GOOGLE_API_KEY=...
```
