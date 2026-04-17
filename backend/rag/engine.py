import os

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

from langchain_core.stores import InMemoryStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from rag import indexing
from rag import retrieval
from rag import generator

import re
import numpy as np


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
        self.local_base_url = os.getenv("LMSTUDIO_BASE_URL", "").rstrip("/")

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

        # Conversation memory: session_id -> list[(original_user_query, assistant_answer)]
        self.conversations: dict[str, list[tuple[str, str]]] = {}

    def _rewrite_query_with_history(self, user_query: str, session_id: str) -> str:
        history_pairs = self.conversations.get(session_id, [])
        if not history_pairs:
            return user_query
        return generator.rewrite_query_with_history(self, user_query, history_pairs)

    # Indexing
    def clear_index(self):
        # Clearing the index also clears conversation history to avoid mixing contexts.
        self.conversations.clear()
        return indexing.clear_index(self)

    def extract_documents(self, file_obj):
        return indexing.extract_documents(self, file_obj)

    def build_index(self, text_or_docs):
        return indexing.build_index(self, text_or_docs)

    def save_index(self, folder_path="vector_db"):
        return indexing.save_index(self, folder_path=folder_path)

    def load_index(self, folder_path="vector_db"):
        return indexing.load_index(self, folder_path=folder_path)

    # Querying
    def _run_fact_pipeline(self, user_query: str, k: int):
        bm25_results = retrieval.bm25_retrieve(self, user_query, k=k * 5)
        dense_results = retrieval.dense_retrieve(self, user_query, k=k * 5)
        fused_results = retrieval.rrf_fusion([bm25_results, dense_results], k=k * 3)
        reranked_results = retrieval.rerank(self, user_query, fused_results, top_k=k)
        return bm25_results, dense_results, fused_results, reranked_results

    def _classify_query(self, query: str) -> str:
        summary_keywords = [
            "tóm tắt",
            "ý chính",
            "nội dung chính",
            "bài nói về gì",
            "vấn đề chính",
            "main idea",
            "overview",
            "summary",
            "important points",
            "ngắn gọn",
            "bao quát",
        ]
        query_lower = (query or "").lower()
        if any(kw in query_lower for kw in summary_keywords):
            return "summary"

        prompt = f"""Phân loại câu hỏi sau thành 1 trong 2 loại: 'fact' hoặc 'summary'.
- 'fact': Câu hỏi tra cứu thông tin cụ thể, con số, định nghĩa, sự kiện đơn lẻ.
- 'summary': Câu hỏi yêu cầu tóm tắt, lấy ý chính, cái nhìn tổng quan, so sánh giữa các phần.

Chỉ trả về duy nhất 1 từ 'fact' hoặc 'summary'.

Câu hỏi: {query}
Trả lời:"""

        try:
            if self.local_base_url and self.local_model:
                res = generator.generate_with_lmstudio(self, prompt).strip().lower()
                if "summary" in res:
                    return "summary"
                if "fact" in res:
                    return "fact"
        except Exception:
            pass

        try:
            if self.api_key:
                res = generator.generate_with_google(self, prompt).strip().lower()
                if "summary" in res:
                    return "summary"
                return "fact"
        except Exception:
            pass

        return "fact"

    def _mmr_select(self, query_embedding, docs, k=5, lambda_val=0.5):
        if not docs:
            return []
        if len(docs) <= k:
            return docs

        doc_embeddings = np.array(self.embeddings.embed_documents([d.page_content for d in docs]))
        query_embedding = np.array(query_embedding)

        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)

        selected_indices = []
        candidate_indices = list(range(len(docs)))
        similarities_to_query = np.dot(doc_embeddings, query_embedding)

        best_idx = int(np.argmax(similarities_to_query))
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

        while len(selected_indices) < k and candidate_indices:
            best_mmr = -float("inf")
            best_idx = -1

            for idx in candidate_indices:
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

    def _textrank_mmr_summarize(self, chunks, user_query: str = None, k: int = 15, lambda_val: float = 0.6) -> str:
        if not chunks:
            return ""

        raw_text = "\n".join([c.page_content for c in chunks])
        sentences = [
            s.strip()
            for s in re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+", raw_text)
            if len(s.strip()) > 15
        ]
        if len(sentences) <= k:
            return "\n".join(sentences)

        sent_embeddings = np.array(self.embeddings.embed_documents(sentences))
        norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
        sent_embeddings = sent_embeddings / (norms + 1e-10)

        # Cosine similarity matrix
        sim_matrix = np.dot(sent_embeddings, sent_embeddings.T)
        np.fill_diagonal(sim_matrix, 0.0)

        # PageRank (TextRank)
        n = sim_matrix.shape[0]
        damping = 0.85
        row_sums = sim_matrix.sum(axis=1, keepdims=True) + 1e-10
        M = sim_matrix / row_sums
        pr = np.full(n, 1.0 / n)
        for _ in range(200):
            pr_new = (1 - damping) / n + damping * M.T.dot(pr)
            if np.linalg.norm(pr_new - pr, 1) < 1e-6:
                pr = pr_new
                break
            pr = pr_new

        norm_tr = pr / (pr.max() + 1e-10)

        use_query_scoring = False
        query_sims = np.zeros(n)
        if user_query and len(user_query.split()) >= 3:
            use_query_scoring = True
            q = np.array(self.embeddings.embed_query(user_query))
            q = q / (np.linalg.norm(q) + 1e-10)
            query_sims = np.dot(sent_embeddings, q)

        centroid = np.mean(sent_embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroid_sims = np.dot(sent_embeddings, centroid)

        relevance_scores = []
        for i in range(n):
            if use_query_scoring:
                rel = 0.3 * norm_tr[i] + 0.7 * query_sims[i]
            else:
                rel = 0.5 * norm_tr[i] + 0.5 * centroid_sims[i]
            relevance_scores.append(float(rel))

        selected_indices = []
        candidate_indices = list(range(n))
        while len(selected_indices) < k and candidate_indices:
            best_score = -float("inf")
            best_idx = -1
            for idx in candidate_indices:
                relevance = relevance_scores[idx]
                if not selected_indices:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = float(max([sim_matrix[idx][s] for s in selected_indices]))
                mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)

        return "\n".join([sentences[i] for i in selected_indices])

    def _generate_summary_from_chunks(self, user_query: str, chunks, k_sentences: int = 18):
        refined_context = self._textrank_mmr_summarize(chunks, user_query=user_query, k=k_sentences)
        prompt = (
            f"Dựa vào các ý chính được trích xuất dưới đây, hãy viết một bản tóm tắt đầy đủ, sâu sắc và mạch lạc cho câu hỏi: '{user_query}'\n\n"
            f"Ý chính nội dung:\n{refined_context}"
        )
        try:
            if self.local_base_url and self.local_model:
                answer = generator.generate_with_lmstudio(self, prompt)
            else:
                answer = generator.generate_with_google(self, prompt)
            return answer, generator.build_citations(chunks)
        except Exception:
            return generator.generate_answer_from_docs(self, user_query, chunks)

    def query(self, user_query, k=3, session_id=None):
        if self.retriever is None and not self.load_index():
            return "Chua co du lieu. Vui long tai file va index truoc.", []

        original_query = user_query
        # Branching logic for session history
        if session_id:
            user_query = self._rewrite_query_with_history(user_query, session_id)

        q_type = self._classify_query(user_query)
        print("Query type:", q_type)

        if q_type == "summary":
            # Summary Pipeline: Block Retrieval -> MMR -> Summary Generation
            if self.block_vectorstore:
                relevant_blocks = self.block_vectorstore.similarity_search(user_query, k=4)
                retrieval.log_retrieval_stage("Summary:Blocks-Retrieved", relevant_blocks, user_query)
                
                # Performance Optimization: O(1) lookup map
                chunk_map = {c.metadata.get("chunk_id"): c for c in self.all_chunks}
                candidate_chunks = []
                seen_ids = set()
                for block in relevant_blocks:
                    for cid in block.metadata.get("child_chunk_ids", []):
                        if cid in chunk_map and cid not in seen_ids:
                            candidate_chunks.append(chunk_map[cid])
                            seen_ids.add(cid)

                retrieval.log_retrieval_stage("Summary:Candidates-From-Blocks", candidate_chunks, user_query, limit=10)
                query_embedding = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(query_embedding, candidate_chunks, k=k + 4, lambda_val=0.4)
                retrieval.log_retrieval_stage("Summary:Final-Diverse-Chunks (MMR)", diverse_chunks, user_query)
                answer, sources = self._generate_summary_from_chunks(user_query, diverse_chunks)
            else:
                # Fallback path if block store is missing
                bm25, dense, fused, reranked = self._run_fact_pipeline(user_query, k=20)
                query_embedding = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(query_embedding, reranked, k=k + 6)
                retrieval.log_retrieval_stage("Fallback-Diverse", diverse_chunks, user_query)
                answer, sources = self._generate_summary_from_chunks(user_query, diverse_chunks)
        else:
            # Fact Pipeline: Hybrid Search (BM25 + Dense) -> RRF -> Rerank
            bm25_results, dense_results, fused_results, reranked_results = self._run_fact_pipeline(
                user_query, k=k
            )
            retrieval.log_retrieval_stage("bm25", bm25_results, user_query)
            retrieval.log_retrieval_stage("dense", dense_results, user_query)
            retrieval.log_retrieval_stage("fusion", fused_results, user_query)
            retrieval.log_retrieval_stage("rerank", reranked_results, user_query)
            answer, sources = generator.generate_answer_from_docs(self, user_query, reranked_results)

        if session_id:
            self.conversations.setdefault(session_id, []).append((original_query, answer))
            if len(self.conversations[session_id]) > 15:
                self.conversations[session_id] = self.conversations[session_id][-15:]
        return answer, sources

    def debug_query(self, user_query, k=3, session_id=None):
        if self.retriever is None and not self.load_index():
            return {"answer": "Chua co du lieu. Vui long tai file va index truoc.", "sources": [], "debug": {}}

        original_query = user_query
        if session_id:
            user_query = self._rewrite_query_with_history(user_query, session_id)

        q_type = self._classify_query(user_query)
        if q_type == "summary":
            if self.block_vectorstore:
                relevant_blocks = self.block_vectorstore.similarity_search(user_query, k=4)
                retrieval.log_retrieval_stage("Summary:Blocks-Retrieved", relevant_blocks, user_query)

                candidate_chunks = []
                seen_ids = set()
                for block in relevant_blocks:
                    for cid in block.metadata.get("child_chunk_ids", []):
                        if cid in seen_ids:
                            continue
                        for chunk in self.all_chunks:
                            if chunk.metadata.get("chunk_id") == cid:
                                candidate_chunks.append(chunk)
                                seen_ids.add(cid)
                                break

                retrieval.log_retrieval_stage("Summary:Candidates-From-Blocks", candidate_chunks, user_query, limit=10)
                q_emb = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(q_emb, candidate_chunks, k=k + 4, lambda_val=0.4)
                retrieval.log_retrieval_stage("Summary:Final-Diverse-Chunks (MMR)", diverse_chunks, user_query)

                answer, sources = self._generate_summary_from_chunks(user_query, diverse_chunks)
                debug = {
                    "query_type": "summary",
                    "method": "block-based",
                    "blocks_retrieved": [
                        {"id": b.metadata.get("block_id"), "content": (b.page_content or "")[:200]}
                        for b in relevant_blocks
                    ],
                    "candidates_preview": retrieval.docs_to_debug_items(candidate_chunks[:10], content_chars=150),
                    "diverse_chunks": retrieval.docs_to_debug_items(diverse_chunks, content_chars=200),
                }
            else:
                bm25_results = retrieval.bm25_retrieve(self, user_query, k=20)
                dense_results = retrieval.dense_retrieve(self, user_query, k=20)
                fused_results = retrieval.rrf_fusion([bm25_results, dense_results], k=20)
                q_emb = self.embeddings.embed_query(user_query)
                diverse_chunks = self._mmr_select(q_emb, fused_results, k=k + 6)
                retrieval.log_retrieval_stage("Fallback-Diverse", diverse_chunks, user_query)

                answer, sources = self._generate_summary_from_chunks(user_query, diverse_chunks)
                debug = {
                    "query_type": "summary",
                    "method": "fallback-mmr",
                    "bm25_results": retrieval.docs_to_debug_items(bm25_results, content_chars=200),
                    "dense_results": retrieval.docs_to_debug_items(dense_results, content_chars=200),
                    "fused_results": retrieval.docs_to_debug_items(fused_results, content_chars=200),
                    "diverse_chunks": retrieval.docs_to_debug_items(diverse_chunks, content_chars=200),
                }
        else:
            bm25_results, dense_results, fused_results, reranked_results = self._run_fact_pipeline(
                user_query, k=k
            )

            retrieval.log_retrieval_stage("bm25", bm25_results, user_query)
            retrieval.log_retrieval_stage("dense", dense_results, user_query)
            retrieval.log_retrieval_stage("fusion", fused_results, user_query)
            retrieval.log_retrieval_stage("rerank", reranked_results, user_query)

            answer, sources = generator.generate_answer_from_docs(self, user_query, reranked_results)
            debug = {
                "query_type": "fact",
                "bm25_results": retrieval.docs_to_debug_items(bm25_results, content_chars=200),
                "dense_results": retrieval.docs_to_debug_items(dense_results, content_chars=200),
                "fused_results": retrieval.docs_to_debug_items(fused_results, content_chars=200),
                "reranked_results": retrieval.docs_to_debug_items(reranked_results, content_chars=200),
            }

        if session_id:
            self.conversations.setdefault(session_id, []).append((original_query, answer))
            if len(self.conversations[session_id]) > 15:
                self.conversations[session_id] = self.conversations[session_id][-15:]
        return {
            "answer": answer,
            "sources": sources,
            "debug": debug,
        }
