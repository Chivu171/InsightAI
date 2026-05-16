from __future__ import annotations

from google import genai
from sentence_transformers import CrossEncoder

from langchain_core.stores import InMemoryStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag import indexing
from rag.conversation_store import ConversationStore
from rag.query_router import QueryRouter


class RAGEngine:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        ollama_model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://127.0.0.1:11434",
    ):
        self.api_key = settings.google_api_key
        self.client = None
        self.google_model = settings.google_model
        self.gemini_model = self.google_model

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

        self.embedding_model_name = settings.embedding_model
        self.local_model = settings.lmstudio_model
        self.local_base_url = settings.lmstudio_base_url

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # Index state
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.retriever = None
        self.parent_splitter = SemanticChunker(self.embeddings)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        self.chunking_mode = None
        self.status = "idle"
        self.progress = 0

        # Retrieval state
        self.bm25_index = None
        self.bm25_corpus = []
        self.all_chunks = []
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.blocks = []
        self.block_vectorstore = None

        # Conversation memory lives behind a store so it can later move to Redis/DB.
        self.conversation_store = ConversationStore(
            max_turns=settings.max_turns,
            rewrite_history_turns=settings.rewrite_history_turns,
            ttl_seconds=settings.session_ttl_seconds,
            redis_url=settings.upstash_redis_rest_url,
            redis_token=settings.upstash_redis_rest_token,
        )
        self.router = QueryRouter(self)

    # Indexing
    def clear_index(self):
        self.conversation_store.clear()
        return indexing.clear_index(self)

    def extract_documents(self, file_obj):
        return indexing.extract_documents(self, file_obj)

    def build_index(
        self,
        text_or_docs,
        chunking_mode: str = "semantic",
        chunk_size: int = 600,
        chunk_overlap: int = 120,
    ):
        return indexing.build_index(
            self,
            text_or_docs,
            chunking_mode=chunking_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def save_index(self, folder_path="vector_db"):
        return indexing.save_index(self, folder_path=folder_path)

    def load_index(self, folder_path="vector_db"):
        return indexing.load_index(self, folder_path=folder_path)

    # Querying
    def query(self, user_query, k=3, session_id=None):
        return self.router._process_query(user_query, k=k, session_id=session_id)

    def debug_query(self, user_query, k=3, session_id=None):
        return self.router._process_debug_query(user_query, k=k, session_id=session_id)
