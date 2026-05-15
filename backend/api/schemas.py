from __future__ import annotations

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

