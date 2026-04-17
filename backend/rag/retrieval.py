import re
import numpy as np
from rank_bm25 import BM25Okapi


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
        print(f"[Hybrid] Dense retrieval error: {e}")
        return []


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
        print(f"[Retrieval:{stage}] no results for query={query!r}")
        return
    print(f"[Retrieval:{stage}] top {min(limit, len(docs))} for query={query!r}")
    for idx, doc in enumerate(docs[:limit], start=1):
        print(f"  {idx}. {doc_debug_label(doc)}")


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
