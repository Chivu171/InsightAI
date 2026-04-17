import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore.in_memory import InMemoryStore
from rank_bm25 import BM25Okapi

from .processor import DocumentProcessor
from .retrieval import Retriever
from .generation import Generator

class RAGEngine:
    def __init__(self, embedding_model="keepitreal/vietnamese-sbert", llm_model="gemini-1.5-flash"):
        self.embedding_model_name = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Components
        self.processor = DocumentProcessor(self.embeddings)
        self.retriever = Retriever()
        
        # LLM setup
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0)
        self.generator = Generator(self.llm)
        
        # State
        self.vectorstore = None
        self.block_vectorstore = None
        self.docstore = InMemoryStore()
        self.bm25_index = None
        self.bm25_corpus = []
        self.all_chunks = []
        self.conversations = {}
        
        self.status = "idle"
        self.progress = 0

    def build_index(self, docs):
        self.status = "processing"
        self.progress = 10
        
        # 1. Processing chunks
        self.all_chunks = self.processor.process_into_chunks(docs)
        self.progress = 40
        
        # 2. Build Dense Index
        print(f"[Engine] Building Dense Index for {len(self.all_chunks)} chunks...")
        self.vectorstore = FAISS.from_documents(self.all_chunks, self.embeddings)
        self.progress = 70
        
        # 3. Build BM25 Index
        print("[Engine] Building BM25 Index...")
        self.bm25_corpus = [doc.page_content for doc in self.all_chunks]
        tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # 4. Build Blocks for Summary queries
        blocks = self.processor.create_blocks(self.all_chunks)
        if blocks:
            self.block_vectorstore = FAISS.from_documents(blocks, self.embeddings)
        
        self.progress = 100
        self.status = "ready"
        print("[Engine] Index build complete.")

    def query(self, user_query, session_id=None, k=3):
        original_query = user_query
        history = self.conversations.get(session_id, [])
        
        # 1. Query Rewriting
        rewritten_query = self.generator.rewrite_query_with_history(user_query, history)
        
        # 2. Retrieval
        bm25_res = self.retriever.bm25_retrieve(rewritten_query, self.bm25_index, self.all_chunks, k=k*5)
        dense_res = self.retriever.dense_retrieve(rewritten_query, self.vectorstore, k=k*5)
        
        fused = self.retriever.rrf_fusion([bm25_res, dense_res], k=k*3)
        reranked = self.retriever.rerank(rewritten_query, fused, top_k=k)
        
        # 3. Generation
        answer, docs = self.generator.generate_answer(rewritten_query, reranked, original_query=original_query)
        citations = self.retriever.build_citations(docs)
        
        # 4. Update memory
        if session_id:
            if session_id not in self.conversations: self.conversations[session_id] = []
            self.conversations[session_id].append((original_query, answer))
            if len(self.conversations[session_id]) > 15: self.conversations[session_id] = self.conversations[session_id][-15:]
            
        return answer, citations

    def debug_query(self, user_query, session_id=None, k=3):
        original_query = user_query
        history = self.conversations.get(session_id, [])
        rewritten_query = self.generator.rewrite_query_with_history(user_query, history)
        
        bm25_res = self.retriever.bm25_retrieve(rewritten_query, self.bm25_index, self.all_chunks, k=k*5) if self.bm25_index else []
        dense_res = self.retriever.dense_retrieve(rewritten_query, self.vectorstore, k=k*5) if self.vectorstore else []
        fused = self.retriever.rrf_fusion([bm25_res, dense_res], k=k*3)
        reranked = self.retriever.rerank(rewritten_query, fused, top_k=k)
        
        answer, docs = self.generator.generate_answer(rewritten_query, reranked, original_query=original_query)
        citations = self.retriever.build_citations(docs)
        
        # Update memory
        if session_id:
            if session_id not in self.conversations: self.conversations[session_id] = []
            self.conversations[session_id].append((original_query, answer))
            
        return {
            "answer": answer,
            "sources": citations,
            "debug": {
                "rewritten_query": rewritten_query,
                "bm25_count": len(bm25_res),
                "dense_count": len(dense_res),
                "fused_count": len(fused),
                "reranked_count": len(reranked)
            }
        }

    def clear_index(self):
        self.vectorstore = None
        self.block_vectorstore = None
        self.bm25_index = None
        self.bm25_corpus = []
        self.all_chunks = []
        self.status = "idle"
        self.progress = 0

    def save_index(self, path="vector_db"):
        if not self.vectorstore: return
        os.makedirs(path, exist_ok=True)
        self.vectorstore.save_local(os.path.join(path, "faiss_index"))
        if self.block_vectorstore:
            self.block_vectorstore.save_local(os.path.join(path, "faiss_blocks"))
        
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "bm25_corpus": self.bm25_corpus,
                "all_chunks": self.all_chunks
            }, f)

    def load_index(self, path="vector_db"):
        if not os.path.exists(os.path.join(path, "faiss_index")): return
        self.vectorstore = FAISS.load_local(os.path.join(path, "faiss_index"), self.embeddings, allow_dangerous_deserialization=True)
        if os.path.exists(os.path.join(path, "faiss_blocks")):
            self.block_vectorstore = FAISS.load_local(os.path.join(path, "faiss_blocks"), self.embeddings, allow_dangerous_deserialization=True)
            
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)
            self.bm25_corpus = meta["bm25_corpus"]
            self.all_chunks = meta["all_chunks"]
            tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
        
        self.status = "ready"
        print(f"[Engine] Index loaded from {path}")

    def extract_documents(self, file):
        return self.processor.extract_documents(file)
