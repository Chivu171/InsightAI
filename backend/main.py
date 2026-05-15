from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from rag.engine import RAGEngine
from api.routers.chat import router as chat_router
from api.routers.documents import router as documents_router

app = FastAPI(title=settings.app_name)

if not settings.cors_origins:
    raise RuntimeError("CORS_ORIGINS is required in production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

app.state.rag_engine = RAGEngine()
app.state.rag_engine.load_index()

app.include_router(documents_router)
app.include_router(chat_router)

