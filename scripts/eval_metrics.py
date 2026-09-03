"""Command-line harness for evaluating the RAG retrieval baselines.

Usage (from the repo root, with the backend virtualenv activated):

    python -m scripts.eval_metrics                     # retrieval-only
    python -m scripts.eval_metrics --answers           # also generate answers
    python -m scripts.eval_metrics --judge             # also score faithfulness
    python -m scripts.eval_metrics --methods bm25 dense
    python -m scripts.eval_metrics --test-set data/test_set.json
    python -m scripts.eval_metrics --out report.json   # dump JSON report

The script:
  1. Boots a ``RAGEngine`` and loads ``vector_db/``.
  2. Reads the test set (default: ``data/test_set.json``).
  3. Runs each retrieval method (``bm25``, ``dense``, ``hybrid``) on every query.
  4. Aggregates Precision@5, Recall@5, MRR, NDCG@5 and per-stage latency.
  5. Optionally generates answers and scores faithfulness with the LLM judge.
  6. Prints a compact, human-readable table to stdout, optionally writes JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make sure ``backend/`` is importable regardless of CWD.
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _build_engine():
    from rag.engine import RAGEngine  # local import after sys.path tweak
    engine = RAGEngine()
    if engine.retriever is None:
        engine.load_index()
    return engine


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG retrieval evaluation harness")
    p.add_argument(
        "--test-set",
        default=str(ROOT / "data" / "test_set.json"),
        help="Path to the JSON test set.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "dense", "hybrid"],
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval methods to evaluate.",
    )
    p.add_argument("--k", type=int, default=5, help="Cut-off for P@k, R@k, NDCG@k.")
    p.add_argument(
        "--answers",
        action="store_true",
        help="Run the LLM on every hybrid query (adds generation latency).",
    )
    p.add_argument(
        "--judge",
        action="store_true",
        help="Use the LLM-as-judge to score answer faithfulness (1-5).",
    )
    p.add_argument(
        "--no-latency",
        action="store_true",
        help="Skip per-stage timing.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write the full JSON report.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    return p.parse_args()


def _print_table(report: dict) -> None:
    summary = report.get("summary", [])
    if not summary:
        print("(empty report)")
        return

    cols = [
        ("method", 8),
        ("n", 4),
        ("P@5", 7),
        ("R@5", 7),
        ("MRR", 7),
        ("NDCG@5", 8),
        ("ret ms", 8),
        ("rerank ms", 10),
        ("gen ms", 8),
        ("total ms", 10),
        ("faith", 6),
        ("prov", 8),
    ]
    header = "  ".join(f"{name:<{w}}" for name, w in cols)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in summary:
        lat = row.get("latency", {})
        faith = row.get("mean_faithfulness")
        faith_str = f"{faith:.2f}" if faith is not None else "—"
        line = "  ".join(
            [
                f"{row.get('method', ''):<{cols[0][1]}}",
                f"{row.get('n_queries', 0):<{cols[1][1]}}",
                f"{row.get('precision_at_5', 0):<{cols[2][1]}.3f}",
                f"{row.get('recall_at_5', 0):<{cols[3][1]}.3f}",
                f"{row.get('mrr', 0):<{cols[4][1]}.3f}",
                f"{row.get('ndcg_at_5', 0):<{cols[5][1]}.3f}",
                f"{lat.get('retrieve_ms', 0):<{cols[6][1]}.1f}",
                f"{lat.get('rerank_ms', 0):<{cols[7][1]}.1f}",
                f"{lat.get('generate_ms', 0):<{cols[8][1]}.1f}",
                f"{lat.get('total_ms', 0):<{cols[9][1]}.1f}",
                f"{faith_str:<{cols[10][1]}}",
                f"{row.get('provider', ''):<{cols[11][1]}}",
            ]
        )
        print(line)
    print(sep)


def _print_winner(report: dict) -> None:
    summary = report.get("summary", [])
    if not summary:
        return
    by_p5 = sorted(summary, key=lambda r: r.get("precision_at_5", 0), reverse=True)
    by_mrr = sorted(summary, key=lambda r: r.get("mrr", 0), reverse=True)
    by_lat = sorted(summary, key=lambda r: r.get("latency", {}).get("total_ms", 0))
    print()
    print("Winners:")
    print(f"  Precision@5 : {by_p5[0]['method']:<7}  ({by_p5[0]['precision_at_5']:.3f})")
    print(f"  MRR         : {by_mrr[0]['method']:<7}  ({by_mrr[0]['mrr']:.3f})")
    print(f"  Latency     : {by_lat[0]['method']:<7}  ({by_lat[0]['latency']['total_ms']:.1f} ms)")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_set_path = Path(args.test_set)
    if not test_set_path.is_absolute():
        test_set_path = ROOT / test_set_path
    if not test_set_path.exists():
        print(f"Test set not found: {test_set_path}", file=sys.stderr)
        return 2

    print(f"Loading RAG engine...")
    t0 = time.perf_counter()
    try:
        engine = _build_engine()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build engine: {exc}", file=sys.stderr)
        return 1
    print(f"Engine ready in {(time.perf_counter() - t0) * 1000:.0f} ms.")

    if engine.retriever is None:
        print(
            "No index loaded. Run the upload flow once (or call "
            "`engine.build_index(...)`) before evaluating.",
            file=sys.stderr,
        )
        return 1

    from rag import metrics as rag_metrics

    test_set = rag_metrics.load_test_set(test_set_path)
    print(f"Test set: {test_set_path}  ({len(test_set)} queries)")

    def _progress(done: int, total: int) -> None:
        print(f"  ... {done}/{total} queries", file=sys.stderr)

    report = rag_metrics.run_evaluation(
        engine,
        test_set=test_set,
        methods=args.methods,
        k=args.k,
        measure_latency=not args.no_latency,
        generate_answers=args.answers,
        judge_faithfulness=args.judge,
        on_progress=_progress,
    )
    report["test_set_path"] = str(test_set_path)
    report["k"] = args.k

    print()
    _print_table(report)
    _print_winner(report)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nFull report written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
