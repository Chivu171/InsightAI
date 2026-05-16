from __future__ import annotations

from rag import generator
from rag import retrieval
from rag.fact_service import FactService
from rag.summarization_service import SummarizationService


class QueryRouter:
    def __init__(self, engine):
        self.engine = engine
        self.fact_service = FactService(engine)
        self.summarization = SummarizationService(engine)

    def _rewrite_query_with_history(self, user_query: str, session_id: str) -> str:
        history_pairs = self.engine.conversation_store.get_rewrite_history(session_id)
        if not history_pairs:
            return user_query
        return generator.rewrite_query_with_history(self.engine, user_query, history_pairs)

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
            if self.engine.local_base_url and self.engine.local_model:
                res = generator.generate_with_lmstudio(self.engine, prompt).strip().lower()
                if "summary" in res:
                    return "summary"
                if "fact" in res:
                    return "fact"
        except Exception:
            pass

        try:
            if self.engine.api_key:
                res = generator.generate_with_google(self.engine, prompt).strip().lower()
                if "summary" in res:
                    return "summary"
                return "fact"
        except Exception:
            pass

        return "fact"

    def _process_query(self, user_query, k=3, session_id=None):
        if self.engine.retriever is None and not self.engine.load_index():
            return "Chua co du lieu. Vui long tai file va index truoc.", []

        original_query = user_query
        if session_id:
            user_query = self._rewrite_query_with_history(user_query, session_id)

        q_type = self._classify_query(user_query)
        print("Query type:", q_type)

        if q_type == "summary":
            if self.engine.block_vectorstore:
                relevant_blocks = self.engine.block_vectorstore.similarity_search(user_query, k=4)
                retrieval.log_retrieval_stage("Summary:Blocks-Retrieved", relevant_blocks, user_query)

                chunk_map = {c.metadata.get("chunk_id"): c for c in self.engine.all_chunks}
                candidate_chunks = []
                seen_ids = set()
                for block in relevant_blocks:
                    for cid in block.metadata.get("child_chunk_ids", []):
                        if cid in chunk_map and cid not in seen_ids:
                            candidate_chunks.append(chunk_map[cid])
                            seen_ids.add(cid)

                retrieval.log_retrieval_stage("Summary:Candidates-From-Blocks", candidate_chunks, user_query, limit=10)
                query_embedding = self.engine.embeddings.embed_query(user_query)
                diverse_chunks = self.summarization.mmr_select(query_embedding, candidate_chunks, k=k + 4, lambda_val=0.4)
                retrieval.log_retrieval_stage("Summary:Final-Diverse-Chunks (MMR)", diverse_chunks, user_query)
                answer, sources = self.summarization.generate_summary_from_chunks(user_query, diverse_chunks)
            else:
                bm25, dense, fused, reranked = self.fact_service.run_fact_pipeline(user_query, k=20)
                query_embedding = self.engine.embeddings.embed_query(user_query)
                diverse_chunks = self.summarization.mmr_select(query_embedding, reranked, k=k + 6)
                retrieval.log_retrieval_stage("Fallback-Diverse", diverse_chunks, user_query)
                answer, sources = self.summarization.generate_summary_from_chunks(user_query, diverse_chunks)
        else:
            answer, sources = self.fact_service.process_query(user_query, k=k)

        if session_id:
            print("[Memory] appending turn", session_id)
            self.engine.conversation_store.append_turn(session_id, original_query, answer)

        return answer, sources

    def _process_debug_query(self, user_query, k=3, session_id=None):
        if self.engine.retriever is None and not self.engine.load_index():
            return {"answer": "Chua co du lieu. Vui long tai file va index truoc.", "sources": [], "debug": {}}

        original_query = user_query
        if session_id:
            user_query = self._rewrite_query_with_history(user_query, session_id)

        q_type = self._classify_query(user_query)
        if q_type == "summary":
            if self.engine.block_vectorstore:
                relevant_blocks = self.engine.block_vectorstore.similarity_search(user_query, k=4)
                retrieval.log_retrieval_stage("Summary:Blocks-Retrieved", relevant_blocks, user_query)

                candidate_chunks = []
                seen_ids = set()
                for block in relevant_blocks:
                    for cid in block.metadata.get("child_chunk_ids", []):
                        if cid in seen_ids:
                            continue
                        for chunk in self.engine.all_chunks:
                            if chunk.metadata.get("chunk_id") == cid:
                                candidate_chunks.append(chunk)
                                seen_ids.add(cid)
                                break

                retrieval.log_retrieval_stage("Summary:Candidates-From-Blocks", candidate_chunks, user_query, limit=10)
                q_emb = self.engine.embeddings.embed_query(user_query)
                diverse_chunks = self.summarization.mmr_select(q_emb, candidate_chunks, k=k + 4, lambda_val=0.4)
                retrieval.log_retrieval_stage("Summary:Final-Diverse-Chunks (MMR)", diverse_chunks, user_query)

                answer, sources = self.summarization.generate_summary_from_chunks(user_query, diverse_chunks)
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
                bm25_results, dense_results, fused_results, _ = self.fact_service.run_fact_pipeline(
                    user_query, k=20
                )
                q_emb = self.engine.embeddings.embed_query(user_query)
                diverse_chunks = self.summarization.mmr_select(q_emb, fused_results, k=k + 6)
                retrieval.log_retrieval_stage("Fallback-Diverse", diverse_chunks, user_query)

                answer, sources = self.summarization.generate_summary_from_chunks(user_query, diverse_chunks)
                debug = {
                    "query_type": "summary",
                    "method": "fallback-mmr",
                    "bm25_results": retrieval.docs_to_debug_items(bm25_results, content_chars=200),
                    "dense_results": retrieval.docs_to_debug_items(dense_results, content_chars=200),
                    "fused_results": retrieval.docs_to_debug_items(fused_results, content_chars=200),
                    "diverse_chunks": retrieval.docs_to_debug_items(diverse_chunks, content_chars=200),
                }
        else:
            fact_debug = self.fact_service.process_debug_query(user_query, k=k)
            answer = fact_debug["answer"]
            sources = fact_debug["sources"]
            debug = fact_debug["debug"]

        if session_id:
            self.engine.conversation_store.append_turn(session_id, original_query, answer)

        return {
            "answer": answer,
            "sources": sources,
            "debug": debug,
        }
