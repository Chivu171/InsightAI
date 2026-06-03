from __future__ import annotations

from rag import generator
from rag import retrieval


class FactService:
    def __init__(self, engine):
        self.engine = engine

    def generate_fact_answer(self, user_query: str, docs):
        return generator.generate_answer_from_docs(self.engine, user_query, docs)

    def run_fact_pipeline(self, user_query: str, k: int):
        bm25_results = retrieval.bm25_retrieve(self.engine, user_query, k=k * 5)

        # HyDE: only for academic papers (adds ~1 LLM call but improves recall)
        use_hyde = getattr(self.engine, "doc_type", "general") == "academic_paper"
        if use_hyde:
            print("[FactService] Using HyDE for dense retrieval")
            dense_results = retrieval.dense_retrieve_with_hyde(self.engine, user_query, k=k * 5)
        else:
            dense_results = retrieval.dense_retrieve(self.engine, user_query, k=k * 5)

        fused_results = retrieval.rrf_fusion([bm25_results, dense_results], k=k * 3)
        reranked_results = retrieval.rerank(self.engine, user_query, fused_results, top_k=k)
        return bm25_results, dense_results, fused_results, reranked_results

    def process_query(self, user_query: str, k: int = 3):
        bm25_results, dense_results, fused_results, reranked_results = self.run_fact_pipeline(user_query, k=k)
        retrieval.log_retrieval_stage("bm25", bm25_results, user_query)
        retrieval.log_retrieval_stage("dense", dense_results, user_query)
        retrieval.log_retrieval_stage("fusion", fused_results, user_query)
        retrieval.log_retrieval_stage("rerank", reranked_results, user_query)
        answer, sources = self.generate_fact_answer(user_query, reranked_results)
        return answer, sources

    def process_debug_query(self, user_query: str, k: int = 3):
        bm25_results, dense_results, fused_results, reranked_results = self.run_fact_pipeline(user_query, k=k)

        retrieval.log_retrieval_stage("bm25", bm25_results, user_query)
        retrieval.log_retrieval_stage("dense", dense_results, user_query)
        retrieval.log_retrieval_stage("fusion", fused_results, user_query)
        retrieval.log_retrieval_stage("rerank", reranked_results, user_query)

        answer, sources = self.generate_fact_answer(user_query, reranked_results)
        debug = {
            "query_type": "fact",
            "bm25_results": retrieval.docs_to_debug_items(bm25_results, content_chars=200),
            "dense_results": retrieval.docs_to_debug_items(dense_results, content_chars=200),
            "fused_results": retrieval.docs_to_debug_items(fused_results, content_chars=200),
            "reranked_results": retrieval.docs_to_debug_items(reranked_results, content_chars=200),
        }
        return {
            "answer": answer,
            "sources": sources,
            "debug": debug,
        }
