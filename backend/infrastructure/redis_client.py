from __future__ import annotations

from typing import Any

import requests


class UpstashRestClient:
    def __init__(self, rest_url: str, rest_token: str, timeout: int = 30):
        self.rest_url = rest_url.rstrip("/")
        self.rest_token = rest_token
        self.timeout = timeout

    def pipeline(self, commands: list[list[Any]]) -> list[Any]:
        response = requests.post(
            f"{self.rest_url}/pipeline",
            headers={"Authorization": f"Bearer {self.rest_token}"},
            json=commands,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Upstash pipeline response: {payload!r}")
        return payload

    @staticmethod
    def unwrap(item: Any) -> Any:
        if isinstance(item, dict):
            if "error" in item and item["error"]:
                raise RuntimeError(str(item["error"]))
            return item.get("result")
        return item

