from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer env value: {value!r}") from exc


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "InsightAI API"
    env: str = "development"
    debug: bool = False

    cors_origins: list[str] = field(default_factory=list)
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = field(default_factory=lambda: ["*"])

    google_api_key: str | None = None
    google_model: str = "gemini-2.5-flash"
    lmstudio_base_url: str = ""
    lmstudio_model: str | None = None

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_db_path: str = "vector_db"
    upload_dir: str = "uploads"
    chunk_size: int = 600
    chunk_overlap: int = 120
    top_k: int = 3
    max_turns: int = 15
    rewrite_history_turns: int = 5
    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: str | None = None
    session_ttl_seconds: int = 86400

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()

        cors_origins = _parse_csv(os.getenv("CORS_ORIGINS"))

        return cls(
            cors_origins=cors_origins,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "").rstrip("/"),
            lmstudio_model=os.getenv("LMSTUDIO_MODEL"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            vector_db_path=os.getenv("VECTOR_DB_PATH", "vector_db"),
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            chunk_size=_parse_int(os.getenv("CHUNK_SIZE"), 600),
            chunk_overlap=_parse_int(os.getenv("CHUNK_OVERLAP"), 120),
            top_k=_parse_int(os.getenv("RETRIEVAL_TOP_K"), 3),
            max_turns=_parse_int(os.getenv("MAX_TURNS"), 15),
            rewrite_history_turns=_parse_int(os.getenv("REWRITE_HISTORY_TURNS"), 5),
            debug=_parse_bool(os.getenv("DEBUG"), False),
            env=os.getenv("ENV", "development"),
            upstash_redis_rest_url=os.getenv("UPSTASH_REDIS_REST_URL"),
            upstash_redis_rest_token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
            session_ttl_seconds=_parse_int(os.getenv("SESSION_TTL_SECONDS"), 86400),
        )


settings = AppConfig.from_env()
