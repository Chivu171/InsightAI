import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def bm25_retrieve(engine, query: str, k: int = 10):
    if engine.bm25_index is None or len(engine.all_chunks) == 0:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = engine.bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [engine.all_chunks[idx] for idx in top_indices]


def dense_retrieve(engine, query: str, k: int = 10):
    if engine.retriever is None:
        return []
    try:
        results = engine.retriever.invoke(query)
        return results[:k]
    except Exception as e:
        logger.warning("[Hybrid] Dense retrieval error: %s", e)
        return []


def dense_retrieve_with_hyde(engine, query: str, k: int = 10) -> list:
    """
    HyDE (Hypothetical Document Embeddings): generate a fake answer with LLM,
    embed that instead of the raw query for better semantic match with paper text.
    Falls back to standard dense retrieval on any failure.
    """
    if engine.vectorstore is None:
        return []

    hyde_prompt = (
        f"Write a concise technical paragraph (2-3 sentences) that would appear "
        f"in an academic paper and directly answers this question: {query}\n"
        f"Output only the paragraph, no preamble."
    )
    try:
        from rag.generator import generate_text  # avoid circular at module level
        hypothetical = (generate_text(engine, hyde_prompt) or "").strip()
        if not hypothetical:
            raise ValueError("Empty HyDE response")
        logger.info("[HyDE] Hypothetical: %s...", hypothetical[:120])
    except Exception as e:
        logger.warning("[HyDE] Falling back to standard dense retrieve: %s", e)
        return dense_retrieve(engine, query, k)

    try:
        embedding = engine.embeddings.embed_query(hypothetical)
        results = engine.vectorstore.similarity_search_by_vector(embedding, k=k)
        return results
    except Exception as e:
        logger.warning("[HyDE] Vector search error, fallback: %s", e)
        return dense_retrieve(engine, query, k)


def rrf_fusion(result_lists, k: int = 10, rrf_k: int = 60):
    doc_scores = {}
    doc_map = {}
    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rank + rrf_k)
    sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
    return [doc_map[did] for did in sorted_ids[:k]]


def rerank(engine, query: str, docs, top_k: int = 5):
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = engine.reranker.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


def doc_debug_label(doc, max_len: int = 120) -> str:
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


def log_retrieval_stage(stage: str, docs, query: str, limit: int = 5) -> None:
    if not docs:
        logger.info("[Retrieval:%s] no results for query=%r", stage, query)
        return
    logger.info("[Retrieval:%s] top %s for query=%r", stage, min(limit, len(docs)), query)
    for idx, doc in enumerate(docs[:limit], start=1):
        logger.debug("  %s. %s", idx, doc_debug_label(doc))


def docs_to_debug_items(docs, content_chars: int = 200):
    items = []
    for doc in docs or []:
        metadata = doc.metadata or {}
        items.append(
            {
                "id": metadata.get("chunk_id"),
                "content": (doc.page_content or "")[:content_chars],
            }
        )
    return items
