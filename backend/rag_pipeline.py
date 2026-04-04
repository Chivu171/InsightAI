import os
import pickle
import re
from datetime import datetime, timezone
import fitz
import numpy as np
from google import genai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", gemini_model="models/gemini-flash-latest"):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.gemini_model = gemini_model
        else:
            self.client = None
            self.gemini_model = None
        
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.retriever = None
        self.parent_splitter = SemanticChunker(self.embeddings)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        self.chunking_mode = None
        self.status = "idle"
        self.progress = 0

        # --- Hybrid retrieval components ---
        self.bm25_index = None          # BM25Okapi instance
        self.bm25_corpus = []           # Tokenized corpus for BM25
        self.all_chunks = []            # list[Document] – all parent chunks (shared by both retrievers)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def clear_index(self):
        """Reset the vectorstore, docstore, and BM25 index to empty states."""
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.retriever = None
        self.chunking_mode = None
        self.bm25_index = None
        self.bm25_corpus = []
        self.all_chunks = []
        self.status = "idle"
        self.progress = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r"\w+", text.lower())

    def extract_text(self, file_obj):
        """Extract text from FastAPI UploadFile or plain file object."""
        filename = getattr(file_obj, "filename", None) or getattr(file_obj, "name", "")

        # FastAPI UploadFile
        real_file = getattr(file_obj, "file", file_obj)

        if str(filename).lower().endswith(".pdf"):
            if hasattr(real_file, "seek"):
                real_file.seek(0)
            pdf_bytes = real_file.read()
            if hasattr(real_file, "seek"):
                real_file.seek(0)

            with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
                pages = []
                for page in pdf_document:
                    page_text = page.get_text("text").strip()
                    if page_text:
                        pages.append(page_text)
                return "\n".join(pages)
        else:
            content = real_file.read()
            return content.decode("utf-8") if isinstance(content, bytes) else content

    def extract_documents(self, file_obj) -> list[Document]:
        """Extract one or more LangChain documents with metadata preserved."""
        filename = getattr(file_obj, "filename", None) or getattr(file_obj, "name", "uploaded_file")
        real_file = getattr(file_obj, "file", file_obj)
        document_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", os.path.splitext(filename)[0]).strip("_") or "uploaded_file"
        uploaded_at = datetime.now(timezone.utc).isoformat()
        extension = os.path.splitext(filename)[1].lower()

        if hasattr(real_file, "seek"):
            real_file.seek(0)

        if extension == ".pdf":
            pdf_bytes = real_file.read()
            if hasattr(real_file, "seek"):
                real_file.seek(0)

            page_documents: list[Document] = []
            with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
                total_pages = len(pdf_document)

                for page_index, page in enumerate(pdf_document, start=1):
                    page_text = page.get_text("text").strip()
                    if not page_text:
                        continue

                    page_documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "document_id": document_id,
                                "document_name": filename,
                                "file_type": extension or "pdf",
                                "page": page_index,
                                "total_pages": total_pages,
                                "uploaded_at": uploaded_at,
                            },
                        )
                    )

            return page_documents

        content = real_file.read()
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        text = (text or "").strip()

        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "document_name": filename,
                    "file_type": extension or "text",
                    "page": None,
                    "total_pages": None,
                    "uploaded_at": uploaded_at,
                },
            )
        ]

    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    @staticmethod
    def _normalize_input_documents(text_or_docs):
        if isinstance(text_or_docs, str):
            return [
                Document(
                    page_content=text_or_docs,
                    metadata={
                        "document_id": "uploaded_file",
                        "document_name": "uploaded_file",
                        "file_type": "text",
                        "page": None,
                        "total_pages": None,
                        "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ]
        return text_or_docs

    @staticmethod
    def _attach_chunk_metadata(chunks: list[Document]) -> list[Document]:
        for chunk_index, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.metadata)
            document_id = metadata.get("document_id", "uploaded_file")
            page = metadata.get("page")
            page_label = f"p{page}" if page is not None else "p0"
            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = f"{document_id}_{page_label}_c{chunk_index}"
            chunk.metadata = metadata
        return chunks

    def _prepare_fixed_chunks(self, docs: list[Document], chunk_size=500, overlap=100) -> list[Document]:
        fixed_chunks: list[Document] = []

        for doc in docs:
            raw_chunks = self.chunk_text(doc.page_content, chunk_size=chunk_size, overlap=overlap)
            for local_index, chunk_text in enumerate(raw_chunks, start=1):
                cleaned_chunk = chunk_text.strip()
                if not cleaned_chunk:
                    continue

                metadata = dict(doc.metadata)
                metadata["fixed_chunk_local_index"] = local_index
                fixed_chunks.append(Document(page_content=cleaned_chunk, metadata=metadata))

        return self._attach_chunk_metadata(fixed_chunks)

    def _generate_answer_from_docs(self, user_query: str, docs: list[Document]):
        citations = self._build_citations(docs)
        relevant_chunks = [doc.page_content for doc in docs]

        if not self.client:
            return "LLM chưa được cấu hình (thiếu API Key).", citations

        context = "\n---\n".join(relevant_chunks)
        prompt = f"""
[Vai trò]
Bạn là chuyên gia đọc hiểu và phân tích bài báo khoa học (AI/ML).

[Bối cảnh]
Bạn đang trả lời câu hỏi dựa trên ngữ cảnh được truy xuất từ bài báo (RAG).
Ngữ cảnh có thể không đầy đủ, vì vậy bạn chỉ được phép sử dụng thông tin có trong ngữ cảnh.

[Nhiệm vụ]
Trả lời câu hỏi theo 3 phần:

1. **Answer (Câu trả lời chính)**:
   - Ngắn gọn, trực tiếp (1–2 câu)
   - Nếu hỏi "method chính" → chỉ chọn 1 method quan trọng nhất

2. **Explanation (Giải thích)**:
   - Giải thích vì sao câu trả lời đúng
   - Tổng hợp thông tin từ các đoạn liên quan trong ngữ cảnh
   - Có thể nhắc đến các thành phần, công thức, hoặc cơ chế liên quan

3. **Related Knowledge (Kiến thức liên quan)**:
   - Chỉ bổ sung nếu trong ngữ cảnh có đề cập
   - Ví dụ: phương pháp cải tiến, biến thể, hoặc kỹ thuật liên quan
   - Không được thêm kiến thức bên ngoài

[Ràng buộc]
- CHỈ sử dụng thông tin từ ngữ cảnh
- KHÔNG suy diễn ngoài
- Nếu không có thông tin → trả lời: "Không có trong tài liệu"
- Không lan man

[Định dạng]
**Answer:** ...
**Explanation:** ...
**Related Knowledge:** ... (có thể bỏ nếu không có)

---

Câu hỏi: {user_query}

Ngữ cảnh:
{context}

Trả lời:
"""
        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=prompt
        )
        return response.text, citations

    # ------------------------------------------------------------------ #
    #  Indexing
    # ------------------------------------------------------------------ #
    def build_index(self, text_or_docs):
        self.status = "processing"
        self.progress = 10

        docs = self._normalize_input_documents(text_or_docs)

        if not docs:
            raise ValueError("No documents available for indexing.")

        self.progress = 30

        if self.vectorstore is None:
            dummy_text = "initialization"
            self.vectorstore = FAISS.from_texts([dummy_text], self.embeddings)

        self.progress = 50

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=None,
        )

        self.progress = 60

        parent_docs = self.parent_splitter.split_documents(docs)
        parent_docs = self._attach_chunk_metadata(parent_docs)

        self.progress = 70

        self.retriever.add_documents(parent_docs)

        # --- Build BM25 index from the same parent chunks ---
        self.all_chunks.extend(parent_docs)
        tokenized = [self._tokenize(doc.page_content) for doc in self.all_chunks]
        self.bm25_corpus = tokenized
        self.bm25_index = BM25Okapi(tokenized)
        self.chunking_mode = "semantic"

        self.progress = 100
        self.status = "done"
        print(f"[Hybrid] Built BM25 index with {len(self.all_chunks)} chunks")
        return self.vectorstore, parent_docs

    def build_fixed_chunk_index(self, text_or_docs, chunk_size=800, overlap=250):
        self.status = "processing"
        self.progress = 10

        docs = self._normalize_input_documents(text_or_docs)
        if not docs:
            raise ValueError("No documents available for fixed chunk indexing.")

        self.progress = 35
        fixed_chunks = self._prepare_fixed_chunks(docs, chunk_size=chunk_size, overlap=overlap)
        if not fixed_chunks:
            raise ValueError("No fixed chunks were created from the provided documents.")

        self.progress = 60
        self.vectorstore = FAISS.from_documents(fixed_chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 12})

        self.progress = 80
        self.all_chunks = fixed_chunks
        self.bm25_corpus = [self._tokenize(doc.page_content) for doc in fixed_chunks]
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        self.chunking_mode = "fixed"

        self.progress = 100
        self.status = "done"
        print(f"[FixedChunking] Built index with {len(self.all_chunks)} fixed chunks")
        return self.vectorstore, fixed_chunks

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save_index(self, folder_path="vector_db"):
        if self.vectorstore is None:
            return False
        os.makedirs(folder_path, exist_ok=True)
        self.vectorstore.save_local(folder_path)
        with open(os.path.join(folder_path, "docstore.pkl"), "wb") as f:
            pickle.dump(self.docstore, f)
        # Persist BM25 data
        with open(os.path.join(folder_path, "bm25_data.pkl"), "wb") as f:
            pickle.dump({
                "all_chunks": self.all_chunks,
                "bm25_corpus": self.bm25_corpus,
            }, f)
        return True

    def load_index(self, folder_path="vector_db"):
        if os.path.exists(os.path.join(folder_path, "index.faiss")):
            self.vectorstore = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)
            docstore_path = os.path.join(folder_path, "docstore.pkl")
            if os.path.exists(docstore_path):
                with open(docstore_path, "rb") as f:
                    self.docstore = pickle.load(f)
            
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=None,
            )

            # Load BM25 data
            bm25_path = os.path.join(folder_path, "bm25_data.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                self.all_chunks = data["all_chunks"]
                self.bm25_corpus = data["bm25_corpus"]
                self.bm25_index = BM25Okapi(self.bm25_corpus)
                print(f"[Hybrid] Loaded BM25 index with {len(self.all_chunks)} chunks")

            self.chunking_mode = "semantic"
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Hybrid Retrieval
    # ------------------------------------------------------------------ #
    def _bm25_retrieve(self, query: str, k: int = 10) -> list[Document]:
        """Retrieve documents using BM25 keyword matching."""
        if self.bm25_index is None or len(self.all_chunks) == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include docs with non-zero BM25 score
                results.append(self.all_chunks[idx])
        return results

    def _dense_retrieve(self, query: str, k: int = 10) -> list[Document]:
        """Retrieve documents using dense embedding (FAISS) via ParentDocumentRetriever."""
        if self.retriever is None:
            return []
        try:
            results = self.retriever.invoke(query)
            return results[:k]
        except Exception as e:
            print(f"[Hybrid] Dense retrieval error: {e}")
            return []

    @staticmethod
    def _rrf_fusion(
        result_lists: list[list[Document]],
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion (RRF).

        For each document, the fused score is:
            score(d) = Σ  1 / (rank_i + rrf_k)
        where rank_i is the 1-based rank in result list i.
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                doc_id = doc.page_content  # use content as key (unique enough for chunks)
                doc_map[doc_id] = doc
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rank + rrf_k)

        # Sort by fused score descending
        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[did] for did in sorted_ids[:k]]

    def _rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        """Re-rank documents using a cross-encoder model."""
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_k]]

    @staticmethod
    def _build_citations(docs: list[Document]) -> list[dict]:
        citations = []
        for doc in docs:
            metadata = doc.metadata or {}
            snippet = " ".join(doc.page_content.split())
            citations.append(
                {
                    "document_id": metadata.get("document_id", "unknown_document"),
                    "document_name": metadata.get("document_name", "Tài liệu không rõ tên"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "file_type": metadata.get("file_type"),
                    "uploaded_at": metadata.get("uploaded_at"),
                    "snippet": snippet[:320],
                }
            )
        return citations

    # ------------------------------------------------------------------ #
    #  Query (Hybrid Pipeline)
    # ------------------------------------------------------------------ #
    def query(self, user_query, k=3):
        if self.retriever is None:
            # Try loading if exists
            if not self.load_index():
                return "Chưa có dữ liệu. Vui lòng tải file và index trước.", []

        # ---- Step 1: Parallel retrieval ----
        bm25_results = self._bm25_retrieve(user_query, k=k * 5)
        dense_results = self._dense_retrieve(user_query, k=k * 5)


        # ---- Step 2: RRF Fusion ----
        fused_results = self._rrf_fusion(
            [bm25_results, dense_results],
            k=k * 3,   # keep more candidates for reranking
        )

        # ---- Step 3: Rerank ----
        reranked_results = self._rerank(user_query, fused_results, top_k=k)

        if not reranked_results:
            return self._generate_answer_from_docs(user_query, [])

        return self._generate_answer_from_docs(user_query, reranked_results)

    def query_fixed_chunking(self, user_query, k=3):
        if self.retriever is None or self.chunking_mode != "fixed":
            return (
                "Chưa có fixed chunk index. Hãy chạy build_fixed_chunk_index(...) trước khi query.",
                [],
            )

        bm25_results = self._bm25_retrieve(user_query, k=k * 5)
        dense_results = self._dense_retrieve(user_query, k=k * 5)
        fused_results = self._rrf_fusion(
            [bm25_results, dense_results],
            k=k * 3,
        )
        reranked_results = self._rerank(user_query, fused_results, top_k=k)

        return self._generate_answer_from_docs(user_query, reranked_results)
