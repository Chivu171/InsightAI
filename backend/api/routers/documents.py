from __future__ import annotations

import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile

from config import settings
from rag.engine import RAGEngine

from ..deps import get_rag_engine

router = APIRouter(tags=["documents"])


@router.get("/status")
async def get_status(rag_engine: RAGEngine = Depends(get_rag_engine)):
    return {
        "status": "ready" if rag_engine.vectorstore is not None else "no_index",
        "vector_count": rag_engine.vectorstore.index.ntotal if rag_engine.vectorstore else 0,
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    chunking_mode: str = Query("semantic"),
    chunk_size: int = Query(500),
    chunk_overlap: int = Query(100),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    try:
        documents = rag_engine.extract_documents(file)
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        def process():
            rag_engine.clear_index()
            rag_engine.build_index(
                documents,
                chunking_mode=chunking_mode,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            rag_engine.save_index()

        background_tasks.add_task(process)
        return {"message": "Processing...", "status": "processing"}
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
    rag_engine.clear_index()
    shutil.rmtree(settings.vector_db_path, ignore_errors=True)
    return {"status": "reset"}

