import os
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
rag = RAGEngine()
rag.load_index()

class QueryRequest(BaseModel):
    query: str

@app.get("/status")
async def get_status():
    return {
        "status": "ready" if rag.vectorstore is not None else "no_index",
        "vector_count": rag.vectorstore.index.ntotal if rag.vectorstore else 0
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        documents = rag.extract_documents(file)

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text.")

        def process():
            rag.clear_index()
            rag.build_index(documents)
            rag.save_index()

        background_tasks.add_task(process)

        return {
            "message": "File uploaded. Indexing in progress...",
            "status": "processing"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress")
async def get_progress():
    return {
        "status": rag.status,
        "progress": rag.progress
    }

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        answer, sources = rag.query(request.query)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
