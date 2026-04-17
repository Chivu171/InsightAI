import os
import pickle
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional

import fitz
import numpy as np
import csv
import docx2txt
import requests
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import csv
import io
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

try:
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter




class RAGEngine:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        ollama_model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://127.0.0.1:11434",
    ):
        load_dotenv()

        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        self.gemini_model = self.google_model

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", model_name)
        self.local_model = os.getenv("LMSTUDIO_MODEL")
        lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL")
        self.local_base_url = lmstudio_base_url.rstrip("/") if lmstudio_base_url else ""

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.retriever = None
        self.parent_splitter = SemanticChunker(self.embeddings)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        self.chunking_mode = None
        self.status = "idle"
        self.progress = 0
        self.conversations = {}

        self.bm25_index = None
        self.bm25_corpus = []
        self.all_chunks = []
        self.blocks = []
        self.block_vectorstore = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def clear_index(self):
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.retriever = None
        self.chunking_mode = None
        self.bm25_corpus = []
        self.all_chunks = []
        self.blocks = []
        self.block_vectorstore = None
        self.status = "idle"
        self.progress = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def extract_text(self, file_path: str) -> str:
        """Helper to extract raw text from various file formats."""
        extension = os.path.splitext(file_path)[1].lower()
        
        # 1. Handle PDF
        if extension == ".pdf":
            try:
                doc = fitz.open(file_path)
                text_content = ""
                for page in doc:
                    text_content += page.get_text() + "\n"
                doc.close()
                return text_content
            except Exception as e:
                print(f"Error parsing PDF: {e}")
                return ""
                
        # 2. Handle CSV
        elif extension == ".csv":
            try:
                text_rows = []
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        # Join columns with a pipe to preserve row context
                        text_rows.append(" | ".join(row))
                return "\n".join(text_rows)
            except Exception as e:
                print(f"Error parsing CSV: {e}")
                return ""
                
        # 3. Handle Word (.docx)
        elif extension == ".docx":
            try:
                return docx2txt.process(file_path)
            except Exception as e:
                print(f"Error parsing Word file: {e}")
                return ""
                
        # 4. Default/Fallback (txt, md, etc.)
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                print(f"Error parsing text file: {e}")
                return ""

    def extract_documents(self, file_obj) -> list[Document]:
        """Entry point for FastAPI UploadFile. Saves temp, extracts text, and wraps in Documents."""
        filename = getattr(file_obj, "filename", "unknown_file")
        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", filename)
        
        # Save UploadFile to disk
        try:
            # We need to read from the start
            file_obj.file.seek(0)
            with open(temp_path, "wb") as f:
                f.write(file_obj.file.read())
        except Exception as e:
            print(f"Error saving file: {e}")
            return []
            
        full_text = self.extract_text(temp_path)
        if not full_text.strip():
            return []
            
        # Wrap as a single document with metadata
        # (Later chunking logic will split this)
        doc = Document(
            page_content=full_text,
            metadata={
                "document_name": filename,
                "file_type": os.path.splitext(filename)[1].lower(),
                "uploaded_at": datetime.now().isoformat()
            }
        )
        return [doc]

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

    def _create_blocks(self, block_size_chunks: int = 15, overlap_chunks: int = 3):
        """Groups chunks into larger blocks using a consistent chunk-based sliding window."""
        if not self.all_chunks:
            return
        
        print(f"[Blocks] Creating blocks from {len(self.all_chunks)} chunks (Sliding Window)...")
        blocks = []
        
        # Sliding window approach for perfect metadata consistency
        step = max(1, block_size_chunks - overlap_chunks)
        for i in range(0, len(self.all_chunks), step):
            window = self.all_chunks[i : i + block_size_chunks]
            if not window:
                break
                
            block_text = "\n\n".join([doc.page_content for doc in window])
            block_ids = [doc.metadata.get("chunk_id") for doc in window if doc.metadata.get("chunk_id")]
            
            block_doc = Document(
                page_content=block_text.strip(),
                metadata={
                    "block_id": f"block_{len(blocks)}",
                    "child_chunk_ids": block_ids,
                    "chunk_count": len(window)
                }
            )
            blocks.append(block_doc)
            
            # If we reached the end, stop
            if i + block_size_chunks >= len(self.all_chunks):
                break
                
        self.blocks = blocks
        self.block_vectorstore = FAISS.from_documents(blocks, self.embeddings)
        print(f"[Blocks] Successfully built {len(blocks)} blocks.")

    def _generate_answer_from_docs(self, user_query: str, docs: list[Document]):
        citations = self._build_citations(docs)
        relevant_chunks = [doc.page_content for doc in docs]

        if not relevant_chunks:
            return "Khong co trong tai lieu.", citations

        context = "\n---\n".join(relevant_chunks)
        prompt = f"""
[Vai trò]
Bạn là trợ lý AI hỗ trợ người dùng đọc hiểu, phân tích và tra cứu thông tin từ tài liệu.

[Bối cảnh]
Bạn đang trả lời dựa trên ngữ cảnh được truy xuất từ tài liệu bằng hệ thống RAG.
Tài liệu có thể là PDF, TXT, CSV hoặc dữ liệu văn bản/bảng biểu tương tự.
Ngữ cảnh được cung cấp có thể chưa đầy đủ, nên bạn phải ưu tiên thông tin có trong ngữ cảnh trước.

[Mục tiêu]
Giúp người dùng:
- hiểu đúng nội dung tài liệu
- xác định thông tin nào thực sự có trong tài liệu
- phân biệt giữa nội dung có căn cứ trong tài liệu và phần suy luận/mở rộng
- nhận được câu trả lời rõ ràng, hữu ích và đủ chiều sâu khi cần

[Nhiệm vụ]
Hãy trả lời câu hỏi của người dùng bằng tiếng Việt.
Tập trung vào việc trả lời đúng ý, rõ ràng, dễ hiểu và bám sát ngữ cảnh.
Không cần theo một cấu trúc trả lời cố định.
Có thể trả lời ngắn hoặc dài tùy theo câu hỏi.
Khi phù hợp, có thể mở rộng thêm kiến thức nền, cách hiểu hoặc bối cảnh liên quan để câu trả lời hữu ích hơn.

[Ràng buộc]
- Ưu tiên tuyệt đối thông tin có trong ngữ cảnh được cung cấp.
- Không được bịa hoặc khẳng định điều mà ngữ cảnh không hỗ trợ.
- Nếu ngữ cảnh không đủ để trả lời trọn vẹn, phải nói rõ phần nào có thể trả lời và phần nào chưa đủ dữ liệu.
- Nếu thông tin không có trong tài liệu, phải nói rõ: "Không có trong tài liệu".
- Nếu có phần mở rộng ngoài tài liệu, phải ghi rõ đó là phần giải thích thêm hoặc kiến thức nền, không phải nội dung được trích trực tiếp từ tài liệu.
- Nếu câu hỏi liên quan đến bảng, CSV hoặc số liệu, phải bám sát hàng, cột, giá trị và điều kiện có trong ngữ cảnh.
- Ưu tiên sự rõ ràng, chính xác và hữu ích hơn văn phong màu mè hoặc dài dòng.


Câu hỏi: {user_query}

Ngữ cảnh:
{context}

Trả lời:
"""
        try:
            answer = self._generate_with_lmstudio(prompt)
        except requests.RequestException as local_error:
            if not self.api_key:
                print(f"[Generate] Local LLM error and no Google API key available: {local_error}")
                return "Khong the ket noi LM Studio va chua cung cap Google API Key.", citations

            try:
                answer = self._generate_with_google(prompt)
            except requests.RequestException as google_error:
                print(f"[Generate] Local LLM error: {local_error}")
                print(f"[Generate] Google API error: {google_error}")
                return (
                    "Khong the ket noi Local LLM. Google API loi: "
                    f"{self._format_request_error(google_error)}",
                    citations,
                )

        return answer or "Khong co trong tai lieu.", citations

    def _generate_with_lmstudio(self, prompt: str) -> str:
        # Determine the endpoint based on the base_url
        # LM Studio typically exposes /v1/chat/completions compatible with OpenAI
        endpoint = f"{self.local_base_url}/chat/completions"
        if not self.local_base_url.endswith("/v1"):
            endpoint = f"{self.local_base_url}/v1/chat/completions"
            
        response = requests.post(
            endpoint,
            json={
                "model": self.local_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "stream": False
            },
            headers={"Content-Type": "application/json"},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def _generate_with_google(self, prompt: str) -> str:
        self.api_key = os.getenv("GOOGLE_API_KEY", self.api_key)
        response = requests.post(
            (
                "https://generativelanguage.googleapis.com/v1/models/"
                f"{self.google_model}:generateContent?key={self.api_key}"
            ),
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(texts).strip()

    @staticmethod
    def _format_request_error(error: requests.RequestException) -> str:
        response = getattr(error, "response", None)
        if response is not None:
            body = response.text.strip().replace("\n", " ")
            if len(body) > 300:
                body = body[:300] + "..."
            return f"HTTP {response.status_code} - {body or response.reason}"
        return str(error)

    def build_index(self, text_or_docs):
        self.status = "processing"
        self.progress = 10

        docs = self._normalize_input_documents(text_or_docs)
        if not docs:
            raise ValueError("No documents available for indexing.")

        self.progress = 30
        parent_docs = self.parent_splitter.split_documents(docs)
        parent_docs = self._attach_chunk_metadata(parent_docs)

        self.progress = 50
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([parent_docs[0]], self.embeddings)
            remaining_docs = parent_docs[1:]
        else:
            remaining_docs = parent_docs

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=None,
        )

        self.progress = 60
        if remaining_docs:
            self.retriever.add_documents(remaining_docs)

        self.all_chunks.extend(parent_docs)
        tokenized = [self._tokenize(doc.page_content) for doc in self.all_chunks]
        self.bm25_corpus = tokenized
        self.bm25_index = BM25Okapi(tokenized)
        self.chunking_mode = "semantic"

        self.progress = 100
        self.status = "done"
        print(f"[Hybrid] Built BM25 index with {len(self.all_chunks)} chunks")
        self._create_blocks()  # Build blocks for summary mode
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
        self._create_blocks()
        return self.vectorstore, fixed_chunks

    def save_index(self, folder_path="vector_db"):
        if self.vectorstore is None:
            return False

        os.makedirs(folder_path, exist_ok=True)
        self.vectorstore.save_local(folder_path)
        with open(os.path.join(folder_path, "docstore.pkl"), "wb") as f:
            pickle.dump(self.docstore, f)
        with open(os.path.join(folder_path, "bm25_data.pkl"), "wb") as f:
            pickle.dump(
                {
                    "all_chunks": self.all_chunks,
                    "bm25_corpus": self.bm25_corpus,
                },
                f,
            )
        
        # Save config
        with open(os.path.join(folder_path, "config.pkl"), "wb") as f:
            pickle.dump({"chunking_mode": self.chunking_mode}, f)
            
        return True

    def load_index(self, folder_path="vector_db"):
        if not os.path.exists(os.path.join(folder_path, "index.faiss")):
            return False

        try:
            self.vectorstore = FAISS.load_local(
                folder_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"[Load] FAISS load error: {e}")
            return False

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

        bm25_path = os.path.join(folder_path, "bm25_data.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            self.all_chunks = data["all_chunks"]
            self.bm25_corpus = data["bm25_corpus"]
            self.bm25_index = BM25Okapi(self.bm25_corpus)

        # Load config
        config_path = os.path.join(folder_path, "config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
                self.chunking_mode = config.get("chunking_mode", "semantic")
        else:
            self.chunking_mode = "semantic"
            
        print(f"[Load] Index loaded with mode: {self.chunking_mode}")
        if self.all_chunks:
            self._create_blocks() # Re-create blocks in memory
        return True

    def _bm25_retrieve(self, query: str, k: int = 10) -> list[Document]:
        if self.bm25_index is None or len(self.all_chunks) == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        return [self.all_chunks[idx] for idx in top_indices]

    def _dense_retrieve(self, query: str, k: int = 10) -> list[Document]:
        if self.retriever is None:
            return []
        try:
            results = self.retriever.invoke(query)
            return results[:k]
        except Exception as e:
            print(f"[Hybrid] Dense retrieval error: {e}")
            return []

    def _classify_query(self, query: str) -> str:
        """Classifies the query as 'fact' or 'summary' using keywords and LLM."""
        summary_keywords = [
            "tóm tắt", "ý chính", "nội dung chính", "bài nói về gì", "vấn đề chính",
            "main idea", "overview", "summary", "important points", "ngắn gọn", "bao quát"
        ]
        
        query_lower = query.lower()
        if any(kw in query_lower for kw in summary_keywords):
            return "summary"

        prompt = f"""Phân loại câu hỏi sau thành 1 trong 2 loại: 'fact' hoặc 'summary'.
- 'fact': Câu hỏi tra cứu thông tin cụ thể, con số, định nghĩa, sự kiện đơn lẻ.
- 'summary': Câu hỏi yêu cầu tóm tắt, lấy ý chính, cái nhìn tổng quan, so sánh giữa các phần.

Chỉ trả về duy nhất 1 từ 'fact' hoặc 'summary'.

Câu hỏi: {query}
Trả lời:"""
        # Try LM Studio first, then Google
        try:
            if self.local_base_url and self.local_model:
                res = self._generate_with_lmstudio(prompt).strip().lower()
                if "summary" in res: return "summary"
                if "fact" in res: return "fact"
        except:
            pass

        try:
            if self.api_key:
                res = self._generate_with_google(prompt).strip().lower()
                if "summary" in res: return "summary"
                return "fact"
        except:
            pass

        return "fact"

    def _mmr_select(self, query_embedding, docs, k=5, lambda_val=0.5):
        """Maximal Marginal Relevance selection for diversity enrichment."""
        if not docs:
            return []
        if len(docs) <= k:
            return docs
        
        doc_embeddings = self.embeddings.embed_documents([d.page_content for d in docs])
        query_embedding = np.array(query_embedding)
        doc_embeddings = np.array(doc_embeddings)
        
        # Normalize for cosine similarity
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)
        
        selected_indices = []
        candidate_indices = list(range(len(docs)))
        
        # Similarities to query
        similarities_to_query = np.dot(doc_embeddings, query_embedding)
        
        # Pick the first one (most relevant)
        best_idx = np.argmax(similarities_to_query)
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
        
        while len(selected_indices) < k and candidate_indices:
            best_mmr = -float('inf')
            best_idx = -1
            
            # Calculate similarity to selected set (diversity penalty)
            # similarities_to_selected[i] = max similarity of doc i to any doc already in selected
            for idx in candidate_indices:
                # Similarity to selected docs
                div_penalty = max([np.dot(doc_embeddings[idx], doc_embeddings[s]) for s in selected_indices])
                
                mmr_score = lambda_val * similarities_to_query[idx] - (1 - lambda_val) * div_penalty
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
            
        return [docs[i] for i in selected_indices]

    def _textrank_mmr_summarize(self, chunks: list[Document], user_query: str = None, k: int = 15, lambda_val: float = 0.6) -> str:
        """
        Refines retrieved context by selecting top sentences using TextRank and/or Semantic scoring + MMR.
        This follows 'Way 3': TextRank or cosine scoring (query/centroid) + MMR.
        """
        if not chunks:
            return ""
        
        # 1. Split into sentences
        raw_text = "\n".join([c.page_content for c in chunks])
        # Improved sentence splitter
        sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+', raw_text) if len(s.strip()) > 15]
        
        if len(sentences) <= k:
            return "\n".join(sentences)
            
        # 2. Embed sentences and normalize safely
        sent_embeddings = np.array(self.embeddings.embed_documents(sentences))
        # Add small epsilon to avoid zero vectors causing matmul errors
        norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
        sent_embeddings = sent_embeddings / (norms + 1e-10)
        
        # 3. Calculate Scores (Relevance)
        # 3a. TextRank score (local importance via graph)
        sim_matrix = sklearn_cosine_similarity(sent_embeddings)
        np.fill_diagonal(sim_matrix, 0.0) # Remove self-loops
        nx_graph = nx.from_numpy_array(sim_matrix)
        try:
            tr_scores_raw = nx.pagerank(nx_graph, max_iter=200)
        except:
            tr_scores_raw = {i: sum(sim_matrix[i]) for i in range(len(sentences))}
            
        # 3b. Query similarity
        use_query_scoring = False
        query_sims = np.zeros(len(sentences))
        if user_query and len(user_query.split()) >= 3:
            use_query_scoring = True
            query_embedding = np.array(self.embeddings.embed_query(user_query))
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            s_norms = sent_embeddings / (np.linalg.norm(sent_embeddings, axis=1, keepdims=True) + 1e-10)
            query_sims = np.dot(s_norms, q_norm)
        
        # 3c. Centroid similarity
        centroid = np.mean(sent_embeddings, axis=0)
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        s_norms = sent_embeddings / (np.linalg.norm(sent_embeddings, axis=1, keepdims=True) + 1e-10)
        centroid_sims = np.dot(s_norms, c_norm)
        
        # 4. Normalize and combine for 'Relevance'
        tr_max = max(tr_scores_raw.values()) if tr_scores_raw.values() else 1.0
        norm_tr = {i: s / tr_max for i, s in tr_scores_raw.items()}
        
        relevance_scores = []
        for i in range(len(sentences)):
            if use_query_scoring:
                # Specific query branch
                rel = 0.3 * norm_tr[i] + 0.7 * query_sims[i]
            else:
                # General/Centroid branch
                rel = 0.5 * norm_tr[i] + 0.5 * centroid_sims[i]
            relevance_scores.append(rel)
            
        # 5. MMR Selection (Diversity)
        selected_indices = []
        candidate_indices = list(range(len(sentences)))
        
        while len(selected_indices) < k and candidate_indices:
            best_mmr_score = -float('inf')
            best_idx = -1
            
            for idx in candidate_indices:
                relevance = relevance_scores[idx]
                
                if not selected_indices:
                    diversity_penalty = 0.0
                else:
                    # Max similarity to any already selected sentence
                    diversity_penalty = max([sim_matrix[idx][s] for s in selected_indices])
                
                mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity_penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx == -1: break
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
            
        return "\n".join([sentences[i] for i in selected_indices])

    @staticmethod
    def _doc_debug_label(doc: Document, max_len: int = 120) -> str:
        metadata = doc.metadata or {}
        label_bits = []
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            label_bits.append(str(chunk_id))
        section_path = metadata.get("section_path")
        if section_path:
            label_bits.append(str(section_path))
        page = metadata.get("page")
        if page is not None:
            label_bits.append(f"p{page}")
        prefix = " | ".join(label_bits)
        snippet = " ".join(doc.page_content.split())[:max_len]
        return f"{prefix}: {snippet}" if prefix else snippet

    def _log_retrieval_stage(self, stage: str, docs: list[Document], query: str, limit: int = 5) -> None:
        if not docs:
            print(f"[Retrieval:{stage}] no results for query={query!r}")
            return

        print(f"[Retrieval:{stage}] top {min(limit, len(docs))} for query={query!r}")
        for idx, doc in enumerate(docs[:limit], start=1):
            print(f"  {idx}. {self._doc_debug_label(doc)}")

    @staticmethod
    def _rrf_fusion(
        result_lists: list[list[Document]],
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[Document]:
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                doc_id = doc.page_content
                doc_map[doc_id] = doc
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rank + rrf_k)

        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[did] for did in sorted_ids[:k]]

    def _rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
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
                    "document_name": metadata.get("document_name", "Tai lieu khong ro ten"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "file_type": metadata.get("file_type"),
                    "uploaded_at": metadata.get("uploaded_at"),
                    "section_path": metadata.get("section_path"),
                    "section_title": metadata.get("section_title"),
                    "snippet": snippet[:320],
                }
            )
        return citations

    def _rewrite_query_with_history(self, user_query: str, session_id: str) -> str:
        """Uses LLM to rewrite a query based on the last few messages in the session."""
        history = self.conversations.get(session_id, [])
        if not history:
            return user_query
            
        # Take last 5 interactions
        recent_history = history[-5:]
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in recent_history])
        
        prompt = f"""Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi độc lập (standalone query) để có thể dùng tìm kiếm trong tài liệu. 
Lưu ý: Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm. Nếu câu hỏi đã rõ ràng, hãy giữ nguyên.

Lịch sử hội thoại:
{history_text}

Câu hỏi mới: {user_query}
Standalone Query:"""

        try:
            if self.local_base_url and self.local_model:
                rewritten = self._generate_with_lmstudio(prompt)
            else:
                rewritten = self._generate_with_google(prompt)
            print(f"[Memory] Rewritten Query: {rewritten.strip()}")
            return rewritten.strip()
        except:
            return user_query

    def query(self, user_query, k=3, session_id: str = None):
        if self.retriever is None and not self.load_index():
            return "Chua co du lieu. Vui long tai file va index truoc.", []

        # 0. Conversation Rewrite (if session exists)
        original_query = user_query
        if session_id:
            user_query = self._rewrite_query_with_history(user_query, session_id)

        # 1. Classification
        q_type = self._classify_query(user_query)
        print(f"[Query] Detected type: {q_type}")

        if q_type == "summary":
            # WAY B: Block-based summary retrieval (generalized)
            if self.block_vectorstore:
                print("[Query:Summary] Using Block-based retrieval...")
                # Tier 1: Block-level retrieval (large context)
                # We retrieve 3-5 largest blocks to cover a wide range
                relevant_blocks = self.block_vectorstore.similarity_search(user_query, k=4)
                self._log_retrieval_stage("Summary:Blocks-Retrieved", relevant_blocks, user_query)
                
                # Tier 2: Get child chunks from these blocks for diversity
                candidate_chunks = []
                seen_ids = set()
                for block in relevant_blocks:
                    child_ids = block.metadata.get("child_chunk_ids", [])
                    for cid in child_ids:
                        if cid not in seen_ids:
                            for chunk in self.all_chunks:
                                if chunk.metadata.get("chunk_id") == cid:
                                    candidate_chunks.append(chunk)
                                    seen_ids.add(cid)
                                    break
                
                # Log the raw candidate pool found inside blocks
                self._log_retrieval_stage("Summary:Candidates-From-Blocks", candidate_chunks, user_query, limit=10)
                
                # Apply MMR for diversity
                query_embedding = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(query_embedding, candidate_chunks, k=k+4, lambda_val=0.4)
                self._log_retrieval_stage("Summary:Final-Diverse-Chunks (MMR)", diverse_chunks, user_query)
                
                return self._generate_summary_from_chunks(user_query, diverse_chunks)
            else:
                # Fallback to standard MMR if blocks are not available
                print("[Query:Summary:Fallback] Using standard MMR + TextRank...")
                bm25_results = self._bm25_retrieve(user_query, k=20)
                dense_results = self._dense_retrieve(user_query, k=20)
                fused = self._rrf_fusion([bm25_results, dense_results], k=20)
                query_embedding = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(query_embedding, fused, k=k+6)
                self._log_retrieval_stage("Fallback-Diverse", diverse_chunks, user_query)
                
                return self._generate_summary_from_chunks(user_query, diverse_chunks)
        else:
            # Standard Pipeline for fact-checking
            bm25_results = self._bm25_retrieve(user_query, k=k * 5)
            self._log_retrieval_stage("bm25", bm25_results, user_query)
            
            dense_results = self._dense_retrieve(user_query, k=k * 5)
            self._log_retrieval_stage("dense", dense_results, user_query)
            
            fused_results = self._rrf_fusion([bm25_results, dense_results], k=k * 3)
            self._log_retrieval_stage("fusion", fused_results, user_query)
            
            final_results = self._rerank(user_query, fused_results, top_k=k)
            self._log_retrieval_stage("rerank", final_results, user_query)
            
            print(f"[Query:Fact] Selected {len(final_results)} reranked chunks")
            answer, sources = self._generate_answer_from_docs(user_query, final_results)

        # Update History
        if session_id:
            if session_id not in self.conversations:
                self.conversations[session_id] = []
            self.conversations[session_id].append((original_query, answer))
            if len(self.conversations[session_id]) > 15:
                self.conversations[session_id] = self.conversations[session_id][-15:]

        return answer, sources

    def _generate_summary_from_chunks(self, user_query: str, chunks: list[Document], k_sentences: int = 18):
        """Helper to run TextRank + MMR and generate final summary response."""
        print(f"[SummaryEngine] Refining {len(chunks)} chunks into {k_sentences} gold sentences...")
        refined_context = self._textrank_mmr_summarize(chunks, user_query=user_query, k=k_sentences)
        
        prompt = f"Dựa vào các ý chính được trích xuất dưới đây, hãy viết một bản tóm tắt đầy đủ, sâu sắc và mạch lạc cho câu hỏi: '{user_query}'\n\nÝ chính nội dung:\n{refined_context}"
        
        try:
            if self.local_base_url and self.local_model:
                answer = self._generate_with_lmstudio(prompt)
            else:
                answer = self._generate_with_google(prompt)
            return answer, self._build_citations(chunks)
        except:
            # Final fallback
            return self._generate_answer_from_docs(user_query, chunks)

    def debug_query(self, user_query, k=3, session_id: str = None):
        if self.retriever is None and not self.load_index():
            return {
                "answer": "Chua co du lieu. Vui long tai file va index truoc.",
                "sources": [],
                "debug": {},
            }

        # 0. Rewrite if session exists
        original_query = user_query
        if session_id:
            user_query = self._rewrite_query_with_history(user_query, session_id)

        q_type = self._classify_query(user_query)
        
        if q_type == "summary":
            # For debug, we run the logic but capture stages
            if self.block_vectorstore:
                relevant_blocks = self.block_vectorstore.similarity_search(user_query, k=4)
                self._log_retrieval_stage("Debug:Summary-Blocks", relevant_blocks, user_query)
                
                candidate_chunks = []
                seen_ids = set()
                for block in relevant_blocks:
                    for cid in block.metadata.get("child_chunk_ids", []):
                        if cid not in seen_ids:
                            for chunk in self.all_chunks:
                                if chunk.metadata.get("chunk_id") == cid:
                                    candidate_chunks.append(chunk)
                                    seen_ids.add(cid)
                                    break
                
                self._log_retrieval_stage("Debug:Summary-Candidates", candidate_chunks, user_query, limit=10)
                
                q_emb = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(q_emb, candidate_chunks, k=k+4, lambda_val=0.4)
                self._log_retrieval_stage("Debug:Summary-Final-Diverse", diverse_chunks, user_query)
                
                refined_context = self._textrank_mmr_summarize(diverse_chunks, user_query=user_query, k=15)
                answer, citations = self._generate_summary_from_chunks(user_query, diverse_chunks)
                
                # Detailed JSON response
                blocks_info = [{"id": b.metadata.get("block_id"), "content": b.page_content[:200]} for b in relevant_blocks]
                chunks_info = [{"id": d.metadata.get("chunk_id"), "content": d.page_content[:200]} for d in diverse_chunks]
                candidates_info = [{"id": c.metadata.get("chunk_id"), "content": c.page_content[:150]} for c in candidate_chunks[:10]]
                
                return {
                    "answer": answer,
                    "sources": citations,
                    "debug": {
                        "query_type": "summary",
                        "method": "block-based",
                        "blocks_retrieved": blocks_info,
                        "candidates_preview": candidates_info,
                        "diverse_chunks": chunks_info,
                        "refined_sentences_count": refined_context.count("\n") + 1,
                    }
                }
            else:
                # Fallback debug
                bm25 = self._bm25_retrieve(user_query, k=20)
                dense = self._dense_retrieve(user_query, k=20)
                fused = self._rrf_fusion([bm25, dense], k=20)
                q_emb = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(q_emb, fused, k=k+6)
                self._log_retrieval_stage("Debug:Fallback-Diverse", diverse_chunks, user_query)
                refined = self._textrank_mmr_summarize(diverse_chunks, user_query=user_query, k=15)
                answer, citations = self._generate_summary_from_chunks(user_query, diverse_chunks)
                
                return {
                    "answer": answer,
                    "sources": citations,
                    "debug": {
                        "query_type": "summary",
                        "method": "fallback-mmr",
                        "fusion_count": len(fused),
                        "diverse_chunks": [{"id": d.metadata.get("chunk_id"), "content": d.page_content[:200]} for d in diverse_chunks],
                        "refined_sentences_count": refined.count("\n") + 1
                    }
                }

        # Standard Fact debug
        bm25_results = self._bm25_retrieve(user_query, k=k * 5)
        dense_results = self._dense_retrieve(user_query, k=k * 5)
        fused_results = self._rrf_fusion([bm25_results, dense_results], k=k * 3)
        reranked_results = self._rerank(user_query, fused_results, top_k=k)

        self._log_retrieval_stage("bm25", bm25_results, user_query)
        self._log_retrieval_stage("dense", dense_results, user_query)
        self._log_retrieval_stage("fusion", fused_results, user_query)
        self._log_retrieval_stage("rerank", reranked_results, user_query)

        answer, sources = self._generate_answer_from_docs(user_query, reranked_results)
        
        # Update History (even in debug)
        if session_id:
            if session_id not in self.conversations:
                self.conversations[session_id] = []
            self.conversations[session_id].append((original_query, answer))
            if len(self.conversations[session_id]) > 15:
                self.conversations[session_id] = self.conversations[session_id][-15:]

        return {
            "answer": answer,
            "sources": sources,
            "debug": {
                "query_type": "fact",
                "bm25_results": [{"id": d.metadata.get("chunk_id"), "content": d.page_content[:200]} for d in bm25_results],
                "dense_results": [{"id": d.metadata.get("chunk_id"), "content": d.page_content[:200]} for d in dense_results],
                "fused_results": [{"id": d.metadata.get("chunk_id"), "content": d.page_content[:200]} for d in fused_results],
                "reranked_results": [{"id": d.metadata.get("chunk_id"), "content": d.page_content[:200]} for d in reranked_results],
            },
        }
