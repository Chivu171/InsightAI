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
OpenRouter-first, with optional LM Studio and Google fallbacks. Create `backend/.env`:
```env
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL_API=https://openrouter.ai/api/v1
OPENROUTER_MODEL_NAME=deepseek/deepseek-v4-flash:free
OPENROUTER_SITE_URL=http://localhost:5173
OPENROUTER_APP_NAME=InsightAI

# Optional local fallback
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_MODEL=your-local-model-name

# Optional Google fallback
# GOOGLE_API_KEY=...
```
