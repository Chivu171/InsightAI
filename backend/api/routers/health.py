from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Lightweight liveness probe — does not touch RAG engine or models.

    Use this for plain process-up checks. Use /ready for model-warm checks.
    """
    return {"status": "ok", "version": "0.1.0"}


@router.get("/ready")
async def readiness(request: Request):
    """Readiness probe — 200 only when models are loaded.

    Returns 503 until the embedding model is initialized so Fly.io
    will not mark the machine healthy during the (slow) first-boot
    HuggingFace download. The first deploy can take 60-180s.
    """
    engine = getattr(request.app.state, "rag_engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "ready": False, "reason": "engine_not_initialized"},
        )
    if getattr(engine, "embeddings", None) is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading_models", "ready": False, "reason": "embeddings_not_loaded"},
        )
    if getattr(engine, "reranker", None) is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading_models", "ready": False, "reason": "reranker_not_loaded"},
        )
    return {
        "status": "ready",
        "ready": True,
        "vector_count": _vector_count(engine),
    }


def _vector_count(engine) -> int:
    try:
        index = getattr(getattr(engine, "vectorstore", None), "index", None)
        return int(getattr(index, "ntotal", 0) or 0)
    except Exception:
        return 0


@router.get("/")
async def root():
    build_info = "unknown"
    try:
        with open("/app/BUILD_INFO") as f:
            build_info = f.read().strip()
    except FileNotFoundError:
        pass
    return {
        "status": "ok",
        "message": "InsightAI API is running",
        "docs": "/docs",
        "build": build_info,
    }
