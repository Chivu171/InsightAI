from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag.engine import RAGEngine

from ..deps import get_rag_engine
from ..schemas import QueryRequest

router = APIRouter(tags=["chat"])


@router.post("/queryHybrid")
async def query_hybrid(
    request: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    try:
        answer, sources = rag_engine.query(request.query, session_id=request.session_id)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

