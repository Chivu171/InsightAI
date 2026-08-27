from __future__ import annotations

import os
import shutil
import threading

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile

from config import settings
from rag.engine import RAGEngine

from ..deps import get_rag_engine

router = APIRouter(tags=["documents"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt"}

# Serializes index rebuilds and resets so concurrent /upload + /reset
# cannot corrupt the FAISS index on disk or in memory.
_index_lock = threading.Lock()


@router.get("/status")
async def get_status(rag_engine: RAGEngine = Depends(get_rag_engine)):
    vector_count = 0
    if rag_engine.vectorstore is not None:
        try:
            index = getattr(rag_engine.vectorstore, "index", None)
            vector_count = getattr(index, "ntotal", 0) if index is not None else 0
        except Exception:
            vector_count = 0
    return {
        "status": "ready" if rag_engine.vectorstore is not None else "no_index",
        "vector_count": vector_count,
    }


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., alias="file"),
    chunking_mode: str = Query("semantic"),
    chunk_size: int = Query(500),
    chunk_overlap: int = Query(100),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    try:
        # ── Security: validate file type & size ──────────────────────
        for file in files:
            filename = getattr(file, "filename", "") or ""
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
            # Size check (read into memory to measure; UploadFile is SpooledTemporaryFile)
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            if size > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large: {filename} ({size} bytes). Max is {MAX_FILE_SIZE} bytes (20MB).")
            if size == 0:
                raise HTTPException(status_code=400, detail=f"Empty file: {filename}")

        documents = []
        for file in files:
            documents.extend(rag_engine.extract_documents(file))

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        def process():
            # Serialize against /reset and other concurrent uploads; the
            # lock is acquired inside the background thread so the request
            # returns immediately to the caller.
            with _index_lock:
                rag_engine.status = "processing"
                rag_engine.progress = 10
                rag_engine.clear_index()
                rag_engine.build_index(
                    documents,
                    chunking_mode=chunking_mode,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                rag_engine.save_index()
                rag_engine.status = "done"
                rag_engine.progress = 100

        # Claim the slot immediately so a concurrent /reset sees "processing".
        if not _index_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="Indexing already in progress. Wait for the current upload to finish, or call /reset afterwards.",
            )
        _index_lock.release()

        background_tasks.add_task(process)
        return {"message": "Processing...", "status": "processing", "file_count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress")
async def get_progress(rag_engine: RAGEngine = Depends(get_rag_engine)):
    data = {
        "status": rag_engine.status,
        "progress": rag_engine.progress,
    }
    return {"hybrid": data, **data}


@router.post("/reset")
async def reset_index(rag_engine: RAGEngine = Depends(get_rag_engine)):
    if rag_engine.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Indexing in progress. Call /reset after the upload completes (poll /progress).",
        )
    # Acquire with try so we don't block the event loop while another
    # background task is mid-rebuild.
    if not _index_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Index rebuild in progress; cannot reset yet.",
        )
    try:
        rag_engine.clear_index()
        rag_engine.reset_session_store()
        shutil.rmtree(settings.vector_db_path, ignore_errors=True)
    finally:
        _index_lock.release()
    return {"status": "reset"}
