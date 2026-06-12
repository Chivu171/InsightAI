from __future__ import annotations

import re

import numpy as np

from rag import generator, retrieval


class SummarizationService:
    def __init__(self, engine):
        self.engine = engine

    def run_summarize_pipeline(self, user_query: str, k: int = 3):
        """
        Full summary retrieval pipeline.
        Returns (blocks, candidate_chunks, diverse_chunks)
        """
        use_hyde = getattr(self.engine, "doc_type", "general") == "academic_paper"
        retrieval_query = user_query
        if use_hyde:
            hyde_prompt = (
                f"Write a concise technical paragraph (2-3 sentences) that would appear "
                f"in an academic paper and directly answers this question: {user_query}\n"
                f"Output only the paragraph, no preamble."
            )
            try:
                hypothetical = (generator.generate_text(self.engine, hyde_prompt) or "").strip()
                if hypothetical:
                    retrieval_query = hypothetical
            except Exception as e:
                print(f"[SummarizePipeline] HyDE failed: {e}")

        blocks = None
        candidate_chunks = []

        if self.engine.block_vectorstore:
            blocks = self.engine.block_vectorstore.similarity_search(retrieval_query, k=4)
            retrieval.log_retrieval_stage("Summary:Blocks-Retrieved", blocks, user_query)
            chunk_map = {c.metadata.get("chunk_id"): c for c in self.engine.all_chunks}
            seen_ids = set()
            for block in blocks:
                for cid in block.metadata.get("child_chunk_ids", []):
                    if cid in chunk_map and cid not in seen_ids:
                        candidate_chunks.append(chunk_map[cid])
                        seen_ids.add(cid)
            retrieval.log_retrieval_stage("Summary:Candidates-From-Blocks", candidate_chunks, user_query)
        else:
            from rag.fact_service import FactService
            fact_service = FactService(self.engine)
            _, _, _, candidate_chunks = fact_service.run_fact_pipeline(user_query, k=20)

        query_embedding = self.engine.embeddings.embed_query(retrieval_query)
        diverse_chunks = self.mmr_select(query_embedding, candidate_chunks, k=k + 4, lambda_val=0.4)
        retrieval.log_retrieval_stage("Summary:Final-Diverse-Chunks (MMR)", diverse_chunks, user_query)

        return blocks, candidate_chunks, diverse_chunks

    def mmr_select(self, query_embedding, docs, k=5, lambda_val=0.5):
        if not docs:
            return []
        if len(docs) <= k:
            return docs

        doc_embeddings = np.array(self.engine.embeddings.embed_documents([d.page_content for d in docs]))
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

    def textrank_mmr_summarize(self, chunks, user_query: str = None, k: int = 15, lambda_val: float = 0.6) -> str:
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

        sent_embeddings = np.array(self.engine.embeddings.embed_documents(sentences))
        norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
        sent_embeddings = sent_embeddings / (norms + 1e-10)

        sim_matrix = np.dot(sent_embeddings, sent_embeddings.T)
        np.fill_diagonal(sim_matrix, 0.0)

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
            q = np.array(self.engine.embeddings.embed_query(user_query))
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

    def generate_summary_from_chunks(self, user_query: str, chunks, k_sentences: int = 18):
        refined_context = self.textrank_mmr_summarize(chunks, user_query=user_query, k=k_sentences)
        prompt = (
            f"Dựa vào các ý chính được trích xuất dưới đây, hãy viết một bản tóm tắt đầy đủ, sâu sắc và mạch lạc cho câu hỏi: '{user_query}'\n\n"
            f"Ý chính nội dung:\n{refined_context}"
        )
        try:
            answer = generator.generate_text(self.engine, prompt)
            return answer, generator.build_citations(chunks)
        except Exception:
            return generator.generate_answer_from_docs(self.engine, user_query, chunks)
