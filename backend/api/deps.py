from __future__ import annotations

from fastapi import Request

from rag.engine import RAGEngine


def get_rag_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine

