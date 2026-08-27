from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

from rag import generator
from rag import retrieval
from rag.fact_service import FactService
from rag.summarization_service import SummarizationService


# Words/phrases that strongly suggest a query needs prior context to make sense.
# If NONE of these appear, the query is almost certainly self-contained and
# we can skip the (expensive) LLM-based rewrite step.
_REFERENTIAL_MARKERS = [
    # Vietnamese pronouns / demonstratives
    r"\bnó\b", r"\bđó\b", r"\bnày\b", r"\bkia\b", r"\bấy\b",
    r"\bchúng\b", r"\bhọ\b", r"\bchỗ\b", r"\bngười\b",
    r"\bvậy\b", r"\bthế\b", r"\bnhư vậy\b", r"\bnhư thế\b",
    r"\bở trên\b", r"\bphía trên\b", r"\btrước\b", r"\bsau\b",
    # English
    r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b", r"\bit\b",
    r"\bthey\b", r"\bthem\b", r"\bthe above\b", r"\bprevious\b",
    r"\bthe former\b", r"\bthe latter\b",
    # Trailing references like "là gì?", "bao nhiêu?" without a clear subject
    # are ambiguous only if there's prior context; we keep these (they don't
    # *prove* dependency) — true dependency is shown by demonstrative pronouns.
]


def _is_self_contained(query: str) -> bool:
    """Return True if the query is clearly self-contained and does not
    need a rewrite against prior conversation history."""
    if not query:
        return True
    text = query.lower().strip()
    if len(text) < 5:
        return True
    for pattern in _REFERENTIAL_MARKERS:
        if re.search(pattern, text):
            return False
    return True


class QueryRouter:
    def __init__(self, engine):
        self.engine = engine
        self.fact_service = FactService(engine)
        self.summarization = SummarizationService(engine)

    def _rewrite_query_with_history(self, user_query: str, session_id: str) -> str:
        """Rewrite a conversational query into a standalone one using recent history.

        Optimization (LLM call budget): skip the LLM rewrite when the query is
        already self-contained — i.e. it does NOT contain a referential pronoun
        that depends on prior context. This is the common case for direct
        fact questions and saves 1 LLM call per turn for most queries.
        """
        history_pairs = self.engine.conversation_store.get_rewrite_history(session_id)
        if not history_pairs:
            return user_query

        if _is_self_contained(user_query):
            return user_query

        return generator.rewrite_query_with_history(self.engine, user_query, history_pairs)

    def _classify_query(self, query: str) -> str:
        # ── Fast keyword path (0 LLM calls) ─────────────────────────────────
        # Covers ~95% of real queries without burning a classification call.
        summary_keywords = [
            "tóm tắt", "tổng hợp", "ý chính", "nội dung chính", "bài nói về gì",
            "vấn đề chính", "main idea", "overview", "summary", "summarize",
            "important points", "ngắn gọn", "bao quát", "so sánh", "compare",
            "điểm chung", "điểm khác", "tổng quan", "toàn bộ",
        ]
        fact_question_markers = [
            "là gì", "là ai", "là bao nhiêu", "bao nhiêu", "khi nào", "ở đâu",
            "như thế nào", "tại sao", "vì sao", "what is", "who is", "when",
            "where", "how", "why", "which", "?", "gì?", "ai?", "nào?",
        ]
        query_lower = (query or "").lower()
        if any(kw in query_lower for kw in summary_keywords):
            return "summary"
        if any(mk in query_lower for mk in fact_question_markers):
            return "fact"

        # ── Slow LLM path (1 call) — only for ambiguous queries ────────────
        prompt = f"""Phân loại câu hỏi sau thành 1 trong 2 loại: 'fact' hoặc 'summary'.
- 'fact': Câu hỏi tra cứu thông tin cụ thể, con số, định nghĩa, sự kiện đơn lẻ.
- 'summary': Câu hỏi yêu cầu tóm tắt, lấy ý chính, cái nhìn tổng quan, so sánh giữa các phần.

Chỉ trả về duy nhất 1 từ 'fact' hoặc 'summary'.

Câu hỏi: {query}
Trả lời:"""

        try:
            res = generator.generate_text(self.engine, prompt).strip().lower()
            if "summary" in res:
                return "summary"
            if "fact" in res:
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
        logger.info("Query type: %s", q_type)

        if q_type == "summary":
            _, _, diverse_chunks = self.summarization.run_summarize_pipeline(user_query, k=k)
            answer, sources = self.summarization.generate_summary_from_chunks(user_query, diverse_chunks)
        else:
            answer, sources = self.fact_service.process_query(user_query, k=k)

        if session_id:
            logger.info("[Memory] appending turn %s", session_id)
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
            blocks, candidate_chunks, diverse_chunks = self.summarization.run_summarize_pipeline(
                user_query, k=k
            )
            answer, sources = self.summarization.generate_summary_from_chunks(user_query, diverse_chunks)

            if blocks is not None:
                debug = {
                    "query_type": "summary",
                    "method": "block-based",
                    "blocks_retrieved": [
                        {"id": b.metadata.get("block_id"), "content": (b.page_content or "")[:200]}
                        for b in blocks
                    ],
                    "candidates_preview": retrieval.docs_to_debug_items(candidate_chunks[:10], content_chars=150),
                    "diverse_chunks": retrieval.docs_to_debug_items(diverse_chunks, content_chars=200),
                }
            else:
                debug = {
                    "query_type": "summary",
                    "method": "fallback-mmr",
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

    def _stream_query(self, user_query: str, k: int = 3, session_id: str | None = None):
        """
        Streaming variant: runs retrieval synchronously, then yields SSE events
        from the LLM token-by-token.
        Yields strings formatted as SSE (see generator._sse).
        """
        if self.engine.retriever is None and not self.engine.load_index():
            yield generator._sse("token", {"content": "Chưa có dữ liệu. Vui lòng tải file và index trước."})
            yield generator._sse("done", {})
            return

        original_query = user_query
        if session_id:
            user_query = self._rewrite_query_with_history(user_query, session_id)

        q_type = self._classify_query(user_query)

        # ── Retrieval (synchronous) ───────────────────────────────────────────
        if q_type == "summary":
            _, _, docs = self.summarization.run_summarize_pipeline(user_query, k=k)
        else:
            _, _, _, docs = self.fact_service.run_fact_pipeline(user_query, k=k)

        # ── Streaming generation ──────────────────────────────────────────────
        collected_answer = []
        for sse_line in generator.stream_answer_from_docs(self.engine, user_query, docs):
            collected_answer.append(sse_line)
            yield sse_line

        # Persist turn to conversation memory
        if session_id:
            import json as _json
            full_answer = "".join(
                _json.loads(line[6:]).get("content", "")
                for line in collected_answer
                if line.startswith("data:") and '"token"' in line
            )
            self.engine.conversation_store.append_turn(session_id, original_query, full_answer)
