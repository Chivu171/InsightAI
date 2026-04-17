import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class Retriever:
    def __init__(self):
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def bm25_retrieve(self, query, bm25_index: BM25Okapi, corpus, k=10):
        if not bm25_index:
            return []
        tokenized_query = query.lower().split()
        return bm25_index.get_top_n(tokenized_query, corpus, n=k)

    def dense_retrieve(self, query, vectorstore, k=10):
        if not vectorstore:
            return []
        return vectorstore.similarity_search(query, k=k)

    def rrf_fusion(self, results_list, k=10, c=60):
        """Reciprocal Rank Fusion (RRF) to merge multiple search results."""
        fused_scores = {}
        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.metadata.get("chunk_id") or doc.page_content
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = (0, doc)
                score, _ = fused_scores[doc_id]
                fused_scores[doc_id] = (score + 1.0 / (c + rank), doc)
        
        sorted_results = sorted(fused_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_results[:k]]

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return []
        passages = [doc.page_content for doc in docs]
        scores = self.reranker.predict([(query, p) for p in passages])
        for i, score in enumerate(scores):
            docs[i].metadata["rerank_score"] = float(score)
        
        return sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)[:top_k]

    def mmr_select(self, query_embedding, docs, k=3, lambda_param=0.5, embeddings_model=None):
        """Maximal Marginal Relevance to reduce redundancy."""
        if not docs or not embeddings_model:
            return docs[:k]
        
        doc_embeddings = np.array(embeddings_model.embed_documents([d.page_content for d in docs]))
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        selected_indices = [0]
        remaining_indices = list(range(1, len(docs)))
        
        while len(selected_indices) < k and remaining_indices:
            best_score = -np.inf
            best_idx = -1
            
            for i in remaining_indices:
                sim_to_query = np.dot(query_embedding, doc_embeddings[i]) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embeddings[i]))
                max_sim_to_selected = max([np.dot(doc_embeddings[i], doc_embeddings[j]) / (np.linalg.norm(doc_embeddings[i]) * np.linalg.norm(doc_embeddings[j])) for j in selected_indices])
                
                mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
        return [docs[i] for i in selected_indices]

    def build_citations(self, docs):
        """Build structured citations from retrieved documents."""
        citations = []
        for doc in docs:
            citations.append({
                "document_id": doc.metadata.get("document_id"),
                "document_name": doc.metadata.get("document_name"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "file_type": doc.metadata.get("file_type"),
                "uploaded_at": doc.metadata.get("uploaded_at"),
                "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            })
        return citations

    def log_retrieval_stage(self, stage_name, results, query):
        print(f"\n[Retrieval:{stage_name}] top {len(results)} for query='{query}'")
        for i, doc in enumerate(results[:2], 1):
            snippet = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i}. {doc.metadata.get('chunk_id', 'unknown')}: {snippet}...")
