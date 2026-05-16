from __future__ import annotations

import json
from datetime import datetime, timezone

from infrastructure.redis_client import UpstashRestClient


class ConversationStore:
    def __init__(
        self,
        max_turns: int = 15,
        rewrite_history_turns: int = 5,
        ttl_seconds: int = 86400,
        redis_url: str | None = None,
        redis_token: str | None = None,
    ):
        self.max_turns = max_turns
        self.rewrite_history_turns = rewrite_history_turns
        self.ttl_seconds = ttl_seconds
        self._conversations: dict[str, list[tuple[str, str]]] = {}
        self._redis = None

        if redis_url and redis_token:
            self._redis = UpstashRestClient(redis_url, redis_token)

    def _turns_key(self, session_id: str) -> str:
        return f"session:{session_id}:turns"

    def _summary_key(self, session_id: str) -> str:
        return f"session:{session_id}:summary"

    def _use_memory_fallback(self) -> bool:
        return self._redis is None

    def clear(self) -> None:
        if self._use_memory_fallback():
            self._conversations.clear()
            return

        cursor = "0"
        keys_to_delete: list[str] = []
        while True:
            result = self._redis.unwrap(self._redis.pipeline([["SCAN", cursor, "MATCH", "session:*", "COUNT", 1000]])[0])
            if isinstance(result, list) and len(result) == 2:
                cursor = str(result[0])
                keys = result[1] or []
                keys_to_delete.extend([str(key) for key in keys])
            else:
                break

            if cursor == "0":
                break

        if keys_to_delete:
            self._redis.pipeline([["DEL", key] for key in keys_to_delete])

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        if self._use_memory_fallback():
            return list(self._conversations.get(session_id, []))

        key = self._turns_key(session_id)
        result = self._redis.unwrap(self._redis.pipeline([["LRANGE", key, 0, -1]])[0])
        if not isinstance(result, list):
            return []

        history: list[tuple[str, str]] = []
        for item in result:
            if not item:
                continue
            try:
                payload = json.loads(item)
                user_text = str(payload.get("user", ""))
                assistant_text = str(payload.get("assistant", ""))
                history.append((user_text, assistant_text))
            except Exception:
                continue
        return history

    def get_rewrite_history(self, session_id: str) -> list[tuple[str, str]]:
        history = self.get_history(session_id)
        return history[-self.rewrite_history_turns :]

    def append_turn(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        turn = {
            "user": user_query,
            "assistant": assistant_answer,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        if self._use_memory_fallback():
            history = self._conversations.setdefault(session_id, [])
            history.append((user_query, assistant_answer))
            if len(history) > self.max_turns:
                self._conversations[session_id] = history[-self.max_turns :]
            return

        key = self._turns_key(session_id)
        response = self._redis.pipeline(
            [
                ["RPUSH", key, json.dumps(turn, ensure_ascii=False)],
                ["LTRIM", key, -self.max_turns, -1],
                ["EXPIRE", key, self.ttl_seconds],
            ]
        )
        print("[Redis] append_turn response:", response)
        for idx, item in enumerate(response):
            print(f"[Redis] append_turn cmd {idx}:", item)

    def set_summary(self, session_id: str, summary: str) -> None:
        if self._use_memory_fallback():
            return
        self._redis.pipeline(
            [
                ["SET", self._summary_key(session_id), summary],
                ["EXPIRE", self._summary_key(session_id), self.ttl_seconds],
            ]
        )

    def get_summary(self, session_id: str) -> str | None:
        if self._use_memory_fallback():
            return None
        result = self._redis.unwrap(self._redis.pipeline([["GET", self._summary_key(session_id)]])[0])
        return None if result in (None, "") else str(result)
