"""
RAG evaluation metrics.

Implements:
  * Retrieval quality — Precision@k, Recall@k, MRR, NDCG@k
  * Answer quality   — Faithfulness (LLM-as-judge, 1-5)
  * Latency          — Per-stage timing (retrieve / generate / total)

Three retrieval baselines are supported so the value of the hybrid
(BM25 + dense + RRF + cross-encoder rerank) pipeline can be measured
head-to-head against its parts:

  * ``bm25``           — keyword search only
  * ``dense``          — embedding search only (with HyDE for academic papers)
  * ``hybrid``         — BM25 + dense fused via RRF, then cross-encoder rerank
                         (this is the production pipeline)

A small test set lives in ``data/test_set.json`` with Vietnamese fact
questions and their ground-truth ``chunk_id`` values so retrieval
metrics can be computed automatically.  Run the harness with
``python -m scripts.eval_metrics`` or hit ``POST /metrics/evaluate``.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric primitives
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant.

    Empty retrieval returns 0.0 (NOT 1.0 — silence is not a hit).
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k retrieved.

    If the question has no relevant docs (shouldn't happen with a curated
    test set, but defensive) return 0.0 instead of NaN.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: Iterable[str]) -> float:
    """1/rank of the FIRST relevant doc in the ranked list, 0 if none."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr(retrieved_lists: list[list[str]], relevant_lists: list[Iterable[str]]) -> float:
    """Mean Reciprocal Rank across many queries."""
    if not retrieved_lists:
        return 0.0
    return sum(
        reciprocal_rank(r, rel) for r, rel in zip(retrieved_lists, relevant_lists)
    ) / len(retrieved_lists)


def dcg_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    """Discounted Cumulative Gain using binary relevance."""
    relevant = set(relevant_ids)
    score = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    """Normalized DCG — IDCG assumes all relevant docs are at the top.

    With binary relevance, IDCG = sum_{i=1..min(|R|,k)} 1/log2(i+1).
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant, k) / idcg


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage latency tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyReport:
    """Container for per-stage timings of a single query."""

    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    provider: str = "unknown"  # "openrouter" | "lmstudio" | "google" | "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieve_ms": round(self.retrieve_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "generate_ms": round(self.generate_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "provider": self.provider,
        }


@contextmanager
def stopwatch():
    """Tiny context manager that returns elapsed ms when exited."""
    t0 = time.perf_counter()
    yield lambda: (time.perf_counter() - t0) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helpers
# ─────────────────────────────────────────────────────────────────────────────

def doc_id(doc) -> str:
    """Extract ``chunk_id`` from a LangChain Document, falling back to content hash."""
    md = getattr(doc, "metadata", None) or {}
    cid = md.get("chunk_id")
    if cid:
        return str(cid)
    # Last-ditch stable identifier — content hash so reruns are comparable.
    return "hash:" + str(hash(doc.page_content) & 0xFFFFFFFF)


def docs_to_ids(docs) -> list[str]:
    return [doc_id(d) for d in (docs or [])]


def resolve_relevant_ids(
    engine,
    relevant_chunk_ids: list[str],
    expected_keywords: list[str] | None = None,
) -> list[str]:
    """Resolve ground-truth ids.

    When the test set stores exact ``chunk_id`` values (e.g.
    ``sample_p0_c3``) we look them up in the engine's index.  If the
    stored ids are stale (e.g. the corpus was re-indexed) we fall back
    to a keyword search over ``engine.all_chunks`` so the harness stays
    useful without manual curation.
    """
    if not engine.all_chunks:
        return list(relevant_chunk_ids)

    id_to_chunk = {doc_id(c): c for c in engine.all_chunks}
    direct = [cid for cid in relevant_chunk_ids if cid in id_to_chunk]
    if direct:
        return direct

    if not expected_keywords:
        return list(relevant_chunk_ids)

    # Substring-match fallback: a chunk is "relevant" if it contains
    # any of the expected keywords (case-insensitive).  This keeps the
    # test set resilient to chunk-id churn after re-indexing.
    matches: list[str] = []
    for chunk in engine.all_chunks:
        text = (chunk.page_content or "").lower()
        if any(kw.lower() in text for kw in expected_keywords):
            matches.append(doc_id(chunk))
    return matches


def retrieve_bm25(engine, query: str, k: int) -> list:
    """BM25-only baseline. Wraps ``rag.retrieval.bm25_retrieve``."""
    from rag import retrieval
    return retrieval.bm25_retrieve(engine, query, k=k)


def retrieve_dense(engine, query: str, k: int) -> list:
    """Dense-only baseline. Honours HyDE for academic papers (same as prod)."""
    from rag import retrieval
    use_hyde = getattr(engine, "doc_type", "general") == "academic_paper"
    if use_hyde:
        return retrieval.dense_retrieve_with_hyde(engine, query, k=k)
    return retrieval.dense_retrieve(engine, query, k=k)


def retrieve_hybrid(engine, query: str, k: int) -> list:
    """Full production pipeline: BM25 + dense (RRF) + cross-encoder rerank."""
    from rag import retrieval
    bm25_results = retrieval.bm25_retrieve(engine, query, k=k * 5)
    dense_results = retrieve_dense(engine, query, k=k * 5)
    fused = retrieval.rrf_fusion([bm25_results, dense_results], k=k * 3)
    return retrieval.rerank(engine, query, fused, top_k=k)


_RETRIEVAL_METHODS = {
    "bm25": retrieve_bm25,
    "dense": retrieve_dense,
    "hybrid": retrieve_hybrid,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-query evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryMetrics:
    """All metrics for ONE query across ONE retrieval method."""

    query: str
    method: str
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    n_retrieved: int = 0
    n_relevant_in_top5: int = 0
    latency: LatencyReport = field(default_factory=LatencyReport)
    answer: str = ""
    faithfulness: float | None = None  # 1-5 from LLM judge, None if skipped
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "method": self.method,
            "precision_at_5": round(self.precision_at_5, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "n_retrieved": self.n_retrieved,
            "n_relevant_in_top5": self.n_relevant_in_top5,
            "latency": self.latency.to_dict(),
            "answer_preview": (self.answer or "")[:160],
            "faithfulness": self.faithfulness,
            "error": self.error,
        }


def evaluate_single_query(
    engine,
    query: str,
    relevant_ids: list[str],
    method: str = "hybrid",
    k: int = 5,
    measure_latency: bool = True,
    generate_answer: bool = False,
    expected_keywords: list[str] | None = None,
) -> QueryMetrics:
    """Run one query through one retrieval method and score it.

    ``relevant_ids`` is resolved against the engine's index when the stored
    ``chunk_id`` values are out of date; in that case ``expected_keywords``
    is used as a substring-match fallback.
    """
    method_fn = _RETRIEVAL_METHODS.get(method)
    if method_fn is None:
        raise ValueError(
            f"Unknown retrieval method {method!r}. "
            f"Choose from {sorted(_RETRIEVAL_METHODS)}."
        )

    resolved_relevant = resolve_relevant_ids(engine, relevant_ids, expected_keywords)

    metrics = QueryMetrics(query=query, method=method)
    retrieval_ms = 0.0
    rerank_ms = 0.0
    answer_text = ""
    provider = "none"

    try:
        # ── Retrieval ──────────────────────────────────────────────────────
        with stopwatch() as get_elapsed:
            docs = method_fn(engine, query, k=k)
        retrieval_ms = get_elapsed()

        # For hybrid, attribute the rerank portion: reranker.predict cost
        # is small but useful to expose when comparing baselines fairly.
        if method == "hybrid" and docs:
            with stopwatch() as get_elapsed_rr:
                from rag import retrieval as _r
                _ = _r.rerank(engine, query, docs[: max(k, len(docs))], top_k=k)
            rerank_ms = get_elapsed_rr()

        retrieved_ids = docs_to_ids(docs)
        metrics.n_retrieved = len(retrieved_ids)
        metrics.n_relevant_in_top5 = len(set(retrieved_ids[:5]) & set(resolved_relevant))

        # ── Retrieval-quality metrics ─────────────────────────────────────
        metrics.precision_at_5 = precision_at_k(retrieved_ids, resolved_relevant, 5)
        metrics.recall_at_5 = recall_at_k(retrieved_ids, resolved_relevant, 5)
        metrics.mrr = reciprocal_rank(retrieved_ids, resolved_relevant)
        metrics.ndcg_at_5 = ndcg_at_k(retrieved_ids, resolved_relevant, 5)

        # ── Optional answer generation + latency for full pipeline ────────
        if generate_answer and docs:
            from rag import generator
            with stopwatch() as get_elapsed_gen:
                answer_text, _ = generator.generate_answer_from_docs(engine, query, docs)
                # Sniff which provider fired (best-effort).
                provider = _detect_provider(engine, answer_text)
            generate_ms = get_elapsed_gen()
        else:
            generate_ms = 0.0

        metrics.answer = answer_text
        metrics.latency = LatencyReport(
            retrieve_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generate_ms=generate_ms,
            total_ms=retrieval_ms + rerank_ms + generate_ms,
            provider=provider,
        )
    except Exception as exc:  # noqa: BLE001 — keep one bad query from killing the run
        logger.exception("evaluate_single_query failed for method=%s", method)
        metrics.error = str(exc)
        metrics.latency = LatencyReport(total_ms=retrieval_ms + rerank_ms)

    if not measure_latency:
        metrics.latency = LatencyReport()

    return metrics


def _detect_provider(engine, answer: str) -> str:
    """Best-effort guess of which LLM provider actually answered."""
    if getattr(engine, "openrouter_client", None) is not None:
        return "openrouter"
    if getattr(engine, "local_base_url", None) and getattr(engine, "local_model", None):
        return "lmstudio"
    if getattr(engine, "client", None) is not None:
        return "google"
    return "none"


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness (LLM-as-judge)
# ─────────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """Bạn là giám khảo đánh giá độ trung thực (faithfulness) của một câu trả lời
được sinh ra bởi hệ thống RAG. Hãy chấm điểm từ 1 đến 5 dựa trên tiêu chí:

  5 — Mọi khẳng định trong câu trả lời đều có căn cứ trong NGỮ CẢNH.
  4 — Gần như hoàn toàn trung thực, có thể có 1 chi tiết nhỏ suy diễn thêm.
  3 — Phần lớn trung thực nhưng có 1-2 ý không rõ ràng từ ngữ cảnh.
  2 — Nhiều ý không có căn cứ, bắt đầu hallucinate.
  1 — Trả lời hoàn toàn bịa hoặc mâu thuẫn với ngữ cảnh.

NGỮ CẢNH TRUY XUẤT:
{context}

CÂU HỎI: {question}

CÂU TRẢ LỜI CỦA HỆ THỐNG:
{answer}

Chỉ trả về MỘT SỐ NGUYÊN từ 1 đến 5, không giải thích gì thêm.
Điểm:"""


def faithfulness_score(
    engine,
    question: str,
    answer: str,
    context_docs,
    max_context_chars: int = 2400,
) -> float | None:
    """LLM-as-judge faithfulness in [1, 5].  Returns None on judge failure.

    Truncates the context so the judge's prompt stays small.  We use the
    same generator path the production pipeline uses, so the judge is
    the same family of model as the one being judged.
    """
    if not answer or not context_docs:
        return None
    context_text = "\n---\n".join(d.page_content for d in context_docs)
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + "\n[…truncated]"

    prompt = _FAITHFULNESS_PROMPT.format(
        context=context_text, question=question, answer=answer
    )

    try:
        from rag import generator
        raw = generator.generate_text(engine, prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Faithfulness judge call failed: %s", exc)
        return None

    if not raw:
        return None

    # Extract the first integer 1-5 from the response.  Models occasionally
    # add preamble or trailing punctuation — be lenient.
    match = re.search(r"\b([1-5])\b", raw)
    if not match:
        logger.debug("Faithfulness judge returned no score: %r", raw[:120])
        return None
    return float(match.group(1))


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MethodReport:
    """Mean of all per-query metrics for a single retrieval method."""

    method: str
    n_queries: int
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    mean_retrieve_ms: float
    mean_rerank_ms: float
    mean_generate_ms: float
    mean_total_ms: float
    provider: str
    mean_faithfulness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_queries": self.n_queries,
            "precision_at_5": round(self.precision_at_5, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "latency": {
                "retrieve_ms": round(self.mean_retrieve_ms, 2),
                "rerank_ms": round(self.mean_rerank_ms, 2),
                "generate_ms": round(self.mean_generate_ms, 2),
                "total_ms": round(self.mean_total_ms, 2),
            },
            "provider": self.provider,
            "mean_faithfulness": (
                round(self.mean_faithfulness, 3) if self.mean_faithfulness is not None
                else None
            ),
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(per_query: list[QueryMetrics], method: str) -> MethodReport:
    rows = [r for r in per_query if r.method == method]
    if not rows:
        return MethodReport(
            method=method,
            n_queries=0,
            precision_at_5=0.0, recall_at_5=0.0, mrr=0.0, ndcg_at_5=0.0,
            mean_retrieve_ms=0.0, mean_rerank_ms=0.0, mean_generate_ms=0.0,
            mean_total_ms=0.0, provider="none", mean_faithfulness=None,
        )
    faithfulness_scores = [r.faithfulness for r in rows if r.faithfulness is not None]
    providers = [r.latency.provider for r in rows if r.latency.provider and r.latency.provider != "none"]
    provider = providers[0] if providers else "none"
    return MethodReport(
        method=method,
        n_queries=len(rows),
        precision_at_5=_mean([r.precision_at_5 for r in rows]),
        recall_at_5=_mean([r.recall_at_5 for r in rows]),
        mrr=_mean([r.mrr for r in rows]),
        ndcg_at_5=_mean([r.ndcg_at_5 for r in rows]),
        mean_retrieve_ms=_mean([r.latency.retrieve_ms for r in rows]),
        mean_rerank_ms=_mean([r.latency.rerank_ms for r in rows]),
        mean_generate_ms=_mean([r.latency.generate_ms for r in rows]),
        mean_total_ms=_mean([r.latency.total_ms for r in rows]),
        provider=provider,
        mean_faithfulness=_mean(faithfulness_scores) if faithfulness_scores else None,
    )


def build_full_report(per_query: list[QueryMetrics]) -> dict[str, Any]:
    """Aggregate per-method reports + per-query details for the API response."""
    methods = sorted({r.method for r in per_query})
    summary = [aggregate(per_query, m).to_dict() for m in methods]
    return {
        "summary": summary,
        "queries": [r.to_dict() for r in per_query],
        "n_queries": len({r.query for r in per_query}),
    }


def run_evaluation(
    engine,
    test_set: list[dict[str, Any]],
    methods: list[str] | None = None,
    k: int = 5,
    measure_latency: bool = True,
    generate_answers: bool = False,
    judge_faithfulness: bool = False,
    on_progress: callable | None = None,
) -> dict[str, Any]:
    """End-to-end evaluation harness.

    For each query in ``test_set``, run every method in ``methods`` (default:
    all three — bm25, dense, hybrid), then optionally:

      * generate an answer (``generate_answers=True`` — only for the hybrid
        baseline to keep the run short)
      * score answer faithfulness with the LLM judge (``judge_faithfulness``)
    """
    if methods is None:
        methods = ["bm25", "dense", "hybrid"]
    methods = [m for m in methods if m in _RETRIEVAL_METHODS]
    if not methods:
        raise ValueError("No valid retrieval methods supplied.")

    per_query: list[QueryMetrics] = []
    total = len(test_set)
    for idx, item in enumerate(test_set, start=1):
        query = item["query"]
        relevant_ids = item.get("relevant_chunk_ids", [])
        keywords = item.get("expected_keywords", [])

        for method in methods:
            metrics = evaluate_single_query(
                engine,
                query=query,
                relevant_ids=relevant_ids,
                expected_keywords=keywords,
                method=method,
                k=k,
                measure_latency=measure_latency,
                # Only run the answer generator on the hybrid pipeline — the
                # whole point is to measure the production path.
                generate_answer=generate_answers and method == "hybrid",
            )

            if judge_faithfulness and method == "hybrid" and metrics.answer:
                # Re-pull the same hybrid docs the judge will see.
                hybrid_docs = retrieve_hybrid(engine, query, k=k)
                metrics.faithfulness = faithfulness_score(
                    engine,
                    question=query,
                    answer=metrics.answer,
                    context_docs=hybrid_docs,
                )

            per_query.append(metrics)

        if on_progress is not None:
            on_progress(idx, total)

    return build_full_report(per_query)


# ─────────────────────────────────────────────────────────────────────────────
# Test set loader
# ─────────────────────────────────────────────────────────────────────────────

def load_test_set(path: str | Path) -> list[dict[str, Any]]:
    """Load a test set of ``{"query": str, "relevant_chunk_ids": [str, ...]}``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Test set at {path} must be a JSON list of queries.")
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if "query" not in item:
            logger.warning("Test set entry %d missing 'query' — skipped", i)
            continue
        cleaned.append(
            {
                "query": str(item["query"]).strip(),
                "relevant_chunk_ids": [str(x) for x in item.get("relevant_chunk_ids", [])],
                "expected_keywords": [str(x).lower() for x in item.get("expected_keywords", [])],
                "note": item.get("note", ""),
            }
        )
    return cleaned


def save_test_set(path: str | Path, items: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
