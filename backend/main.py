import os
from fastapi import FastAPI, UploadFile, File, HTTPException
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
        "status": "ready" if rag.index is not None else "no_index",
        "vector_count": rag.index.ntotal if rag.index else 0
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Step 1: Extract text
        text = rag.extract_text(file.file)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from file.")
        
        # Step 2: Build index
        rag.build_index(text)
        rag.save_index()
        
        return {
            "message": f"Successfully indexed {file.filename}",
            "chunks": len(rag.chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
