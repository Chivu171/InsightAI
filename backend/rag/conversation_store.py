from __future__ import annotations


class ConversationStore:
    def __init__(self, max_turns: int = 15, rewrite_history_turns: int = 5):
        self.max_turns = max_turns
        self.rewrite_history_turns = rewrite_history_turns
        self._conversations: dict[str, list[tuple[str, str]]] = {}

    def clear(self) -> None:
        self._conversations.clear()

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        return list(self._conversations.get(session_id, []))

    def get_rewrite_history(self, session_id: str) -> list[tuple[str, str]]:
        history = self._conversations.get(session_id, [])
        return history[-self.rewrite_history_turns :]

    def append_turn(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        history = self._conversations.setdefault(session_id, [])
        history.append((user_query, assistant_answer))
        if len(history) > self.max_turns:
            self._conversations[session_id] = history[-self.max_turns :]
