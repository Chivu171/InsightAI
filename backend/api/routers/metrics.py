"""Evaluation / metrics endpoints.

These power the "Hybrid search đạt X% trên bộ test Y câu hỏi, so với
Z% khi chỉ dùng keyword search" numbers that go into the report.

Disabled in production — same convention as ``chat.py`` debug endpoints.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from config import settings
from rag.engine import RAGEngine

from ..deps import get_rag_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


class EvaluateRequest(BaseModel):
    test_set_path: str = Field(
        default="data/test_set.json",
        description="Path to JSON test set, relative to repo root.",
    )
    methods: list[str] | None = Field(
        default=None,
        description="Retrieval methods to evaluate. Defaults to all three.",
    )
    k: int = Field(default=5, ge=1, le=20)
    measure_latency: bool = True
    generate_answers: bool = Field(
        default=False,
        description="Run the LLM on every hybrid query (slow, but exposes "
        "answer quality + generation latency).",
    )
    judge_faithfulness: bool = Field(
        default=False,
        description="Use the LLM-as-judge to score answer faithfulness (1-5).",
    )


@router.post("/metrics/evaluate")
async def evaluate(
    request: EvaluateRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
) -> dict[str, Any]:
    """Run the full evaluation harness and return a JSON report.

    Each call to this endpoint can take anywhere from a few seconds
    (retrieval only, 25 queries × 3 methods) to a few minutes
    (with ``generate_answers`` or ``judge_faithfulness``).  It is
    intended for batch runs and CI checks, not live traffic.
    """
    if settings.env == "production":
        raise HTTPException(
            status_code=404,
            detail="Endpoint disabled in production.",
        )

    if rag_engine.retriever is None and not rag_engine.load_index():
        raise HTTPException(
            status_code=400,
            detail="RAG engine has no index loaded. Upload a document first.",
        )

    from rag import metrics as rag_metrics

    test_set_path = Path(request.test_set_path)
    if not test_set_path.is_absolute():
        # Resolve relative paths against the repo root (one level above backend/).
        test_set_path = Path(__file__).resolve().parents[2] / request.test_set_path
    if not test_set_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Test set not found at {test_set_path}",
        )

    try:
        test_set = rag_metrics.load_test_set(test_set_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to load test set: {exc}")

    if not test_set:
        raise HTTPException(status_code=400, detail="Test set is empty.")

    logger.info(
        "[Metrics] Running eval: %d queries, methods=%s, answers=%s, judge=%s",
        len(test_set), request.methods, request.generate_answers, request.judge_faithfulness,
    )

    def _progress(done: int, total: int) -> None:
        logger.info("[Metrics] progress %d/%d", done, total)

    report = rag_metrics.run_evaluation(
        rag_engine,
        test_set=test_set,
        methods=request.methods,
        k=request.k,
        measure_latency=request.measure_latency,
        generate_answers=request.generate_answers,
        judge_faithfulness=request.judge_faithfulness,
        on_progress=_progress,
    )
    report["test_set_path"] = str(test_set_path)
    report["k"] = request.k
    return report


@router.get("/metrics/test-set")
async def list_test_set() -> dict[str, Any]:
    """Return the test set metadata (no need to load the engine for this)."""
    if settings.env == "production":
        raise HTTPException(status_code=404, detail="Endpoint disabled in production.")

    from rag import metrics as rag_metrics

    default_path = Path(__file__).resolve().parents[2] / "data" / "test_set.json"
    if not default_path.exists():
        raise HTTPException(status_code=404, detail=f"Test set not found at {default_path}")
    items = rag_metrics.load_test_set(default_path)
    return {
        "path": str(default_path),
        "n_queries": len(items),
        "items": [
            {"query": it["query"], "expected_keywords": it.get("expected_keywords", [])}
            for it in items
        ],
    }
