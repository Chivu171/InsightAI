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

# Initialize RAG Engine
rag_engine = RAGEngine()
rag_engine.load_index()

# Ensure upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    session_id: str = None


@app.get("/status")
async def get_status():
    return {
        "status": "ready" if rag_engine.vectorstore is not None else "no_index",
        "vector_count": rag_engine.vectorstore.index.ntotal if rag_engine.vectorstore else 0,
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        documents = rag_engine.extract_documents(file)
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        def process():
            rag_engine.clear_index()
            rag_engine.build_index(documents)
            rag_engine.save_index()

        background_tasks.add_task(process)
        return {"message": "Processing...", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uploadHybrid")
async def upload_hybrid_alias(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Alias for backward compatibility"""
    return await upload_file(file=file, background_tasks=background_tasks)


@app.get("/progress")
async def get_progress():
    # Return format flattened for simplicity, but keep hybrid key for FE compatibility if needed
    # Let's keep it compatible for now but add a flat version
    data = {
        "status": rag_engine.status,
        "progress": rag_engine.progress,
    }
    return {"hybrid": data, **data}


@app.post("/reset")
async def reset_index():
    rag_engine.clear_index()
    shutil.rmtree("vector_db", ignore_errors=True)
    return {"status": "reset"}

@app.post("/queryHybrid")
async def query_hybrid(request: QueryRequest):
    try:
        answer, sources = rag_engine.query(request.query, session_id=request.session_id)
        return {
            "answer": answer,
            "sources": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/queryHybrid")
async def debug_query_hybrid(request: QueryRequest):
    try:
        result = rag_engine.debug_query(request.query, session_id=request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
