import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import RAGEngine

app = FastAPI(title="InsightAI API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize separate engines for each chunking strategy
rag_hybrid = RAGEngine()
rag_hybrid.load_index()
rag_fixed = RAGEngine()

# Ensure temp directory for Docling
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    session_id: str = None


def get_engine(mode: str) -> RAGEngine:
    engine = {"hybrid": rag_hybrid, "fixed": rag_fixed}.get(mode)
    if engine is None:
        raise HTTPException(status_code=404, detail="Mode not found.")
    return engine

@app.get("/status")
async def get_status():
    return {
        "hybrid": {
            "status": "ready" if rag_hybrid.vectorstore is not None else "no_index",
            "vector_count": rag_hybrid.vectorstore.index.ntotal if rag_hybrid.vectorstore else 0,
            "chunking_mode": rag_hybrid.chunking_mode,
        },
        "fixed": {
            "status": "ready" if rag_fixed.vectorstore is not None else "no_index",
            "vector_count": rag_fixed.vectorstore.index.ntotal if rag_fixed.vectorstore else 0,
            "chunking_mode": rag_fixed.chunking_mode,
        },
    }

@app.post("/upload")
async def upload_hybrid(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        documents = rag_hybrid.extract_documents(file)

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        def process():
            rag_hybrid.clear_index()
            rag_hybrid.build_index(documents)
            rag_hybrid.save_index()

        background_tasks.add_task(process)

        return {
            "message": "File uploaded. Hybrid indexing in progress...",
            "status": "processing",
            "mode": "hybrid",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uploadHybrid")
async def upload_hybrid_alias(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    return await upload_hybrid(file=file, background_tasks=background_tasks)

@app.post("/uploadSimpleChunking")
async def upload_simple_chunking(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        documents = rag_fixed.extract_documents(file)

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        def process():
            rag_fixed.clear_index()
            rag_fixed.build_fixed_chunk_index(documents)

        background_tasks.add_task(process)

        return {
            "message": "File uploaded. Fixed chunk indexing in progress...",
            "status": "processing",
            "mode": "fixed",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/progress")
async def get_progress():
    return {
        "hybrid": {
            "status": rag_hybrid.status,
            "progress": rag_hybrid.progress,
        },
        "fixed": {
            "status": rag_fixed.status,
            "progress": rag_fixed.progress,
        },
    }

@app.get("/progress/{mode}")
async def get_progress_by_mode(mode: str):
    engine = get_engine(mode)

    return {
        "mode": mode,
        "status": engine.status,
        "progress": engine.progress,
    }


@app.post("/reset/{mode}")
async def reset_mode(mode: str):
    engine = get_engine(mode)
    engine.clear_index()

    if mode == "hybrid":
        shutil.rmtree("vector_db", ignore_errors=True)

    return {
        "mode": mode,
        "status": "reset",
    }

@app.post("/queryHybrid")
async def query_hybrid(request: QueryRequest):
    try:
        answer, sources = rag_hybrid.query(request.query, session_id=request.session_id)
        return {
            "answer": answer,
            "sources": sources,
            "mode": "hybrid",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/queryHybrid")
async def debug_query_hybrid(request: QueryRequest):
    try:
        result = rag_hybrid.debug_query(request.query, session_id=request.session_id)
        return {
            "mode": "hybrid",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/querySimpleChunking")
async def query_simple_chunking(request: QueryRequest):
    try:
        answer, sources = rag_fixed.query_fixed_chunking(request.query)
        return {
            "answer": answer,
            "sources": sources,
            "mode": "fixed",
        }
    except Exception as e:
        print("querySimpleChunking error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
