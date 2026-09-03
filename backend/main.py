from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging

from config import settings
from rag.engine import RAGEngine
from api.routers.chat import router as chat_router
from api.routers.documents import router as documents_router
from api.routers.health import router as health_router
from api.routers.metrics import router as metrics_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Surface the image build marker (set by Dockerfile via /app/BUILD_INFO)
# so fly logs immediately tells you which commit is running.
try:
    with open("/app/BUILD_INFO") as _f:
        _BUILD_INFO = _f.read().strip()
except FileNotFoundError:
    _BUILD_INFO = "build sha=unknown (no /app/BUILD_INFO)"
logger.info("InsightAI starting — %s | env=%s | debug=%s", _BUILD_INFO, settings.env, settings.debug)

app = FastAPI(title=settings.app_name)

if not settings.cors_origins:
    if settings.env == "production":
        # In production, an empty CORS_ORIGINS silently breaks every browser
        # request (opaque network errors). Surface an actionable message that
        # tells the operator exactly how to fix it before the app exits.
        if settings.cors_origins_default:
            logger.warning(
                "CORS_ORIGINS empty — falling back to cors_origins_default=%s "
                "(set CORS_ORIGINS explicitly for production).",
                settings.cors_origins_default,
            )
            cors_origins = settings.cors_origins_default
        else:
            raise RuntimeError(
                "CORS_ORIGINS is required in production but is empty.\n"
                "Fix it with:\n"
                "  fly secrets set -a <app> "
                'CORS_ORIGINS="https://<your-vercel-app>.vercel.app"\n'
                "(or set CORS_ORIGINS_DEFAULT in your environment as a safety net)."
            )
    else:
        logger.warning("CORS_ORIGINS is empty — allowing all origins in development")
        cors_origins = ["*"]
else:
    cors_origins = settings.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

app.state.rag_engine = RAGEngine()
app.state.rag_engine.load_index()

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(metrics_router)

