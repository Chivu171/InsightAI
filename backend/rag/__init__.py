"""RAG modules (indexing, retrieval, generation).

This package lives under `backend/rag/` while the FastAPI entrypoint lives in
`backend/`. When running with `uvicorn --reload`, the spawned subprocess may
not always have `backend/` on `sys.path`, so we ensure it here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
