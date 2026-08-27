from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from rag.engine import RAGEngine

from ..deps import get_rag_engine
from ..schemas import QueryRequest

router = APIRouter(tags=["chat"])

_DONE = object()  # sentinel for exhausted generator


# Debug / test endpoints — only available when NOT running in production.
# Keeps noise out of logs and removes an info-leak surface from prod.
if settings.env != "production":

    @router.get("/stream-test")
    async def stream_test():
        """Dummy SSE endpoint — test async streaming."""

        async def gen():
            for i in range(10):
                yield f"data: {json.dumps({'type': 'token', 'content': f'token-{i} '})}\n\n"
                await asyncio.sleep(0.3)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/stream-test-sync")
    async def stream_test_sync():
        """Test run_in_executor + sync generator — simulate blocking I/O."""
        import time

        def sync_gen():
            for i in range(10):
                time.sleep(0.3)  # simulate blocking token arrival
                yield f"data: {json.dumps({'type': 'token', 'content': f'sync-{i} '})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        async def gen():
            loop = asyncio.get_running_loop()
            sg = sync_gen()

            def get_next():
                try:
                    return next(sg)
                except StopIteration:
                    return _DONE

            while True:
                item = await loop.run_in_executor(None, get_next)
                if item is _DONE:
                    break
                yield item

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


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


@router.post("/queryHybrid/stream")
async def query_hybrid_stream(
    request: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    SSE streaming endpoint.

    Uses run_in_executor to call next() on the sync generator in a thread pool.
    Each token blocks the thread until it arrives from OpenRouter, while the
    event loop remains free to flush the HTTP chunk to the client before
    fetching the next token.
    """
    async def event_generator() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        sync_gen = rag_engine.stream_query(
            request.query, session_id=request.session_id
        )

        def get_next():
            try:
                return next(sync_gen)
            except StopIteration:
                return _DONE

        try:
            while True:
                item = await loop.run_in_executor(None, get_next)
                if item is _DONE:
                    break
                yield item
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

