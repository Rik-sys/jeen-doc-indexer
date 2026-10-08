"""Configuration loaded from environment variables (a local ``.env`` file is supported).

Secrets never live in code. Only two variables are required:

* ``GEMINI_API_KEY`` - Google AI Studio API key.
* ``POSTGRES_URL``   - e.g. ``postgresql://user:password@localhost:5432/doc_index``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .errors import ConfigError

EMBEDDING_MODEL = "gemini-embedding-001"
# 768 instead of the 3072 default: pgvector's HNSW index supports up to 2,000
# dimensions for the `vector` type, storage is 4x smaller, and Google reports
# only a marginal quality loss at 768 for retrieval tasks.
EMBEDDING_DIM = 768


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    postgres_url: str | None
    embedding_model: str = EMBEDDING_MODEL
    embedding_dim: int = EMBEDDING_DIM

    @property
    def safe_db_target(self) -> str:
        """``host:port/dbname`` without user or password - safe to print."""
        if not self.postgres_url:
            return "<not set>"
        parts = urlsplit(self.postgres_url)
        host = parts.hostname or "?"
        port = parts.port or 5432
        db = (parts.path or "/").lstrip("/") or "?"
        return f"{host}:{port}/{db}"


def load_settings(*, need_gemini: bool = True, need_db: bool = True) -> Settings:
    """Read settings from the environment and validate the ones this command needs."""
    load_dotenv()  # does not override variables that are already set
    key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
    url = (os.getenv("POSTGRES_URL") or "").strip() or None

    missing = []
    if need_gemini and not key:
        missing.append("GEMINI_API_KEY")
    if need_db and not url:
        missing.append("POSTGRES_URL")
    if missing:
        raise ConfigError(
            f"Missing {' and '.join(missing)}. Copy .env.example to .env and fill in the value(s)."
        )
    if need_db and url and urlsplit(url).scheme not in ("postgresql", "postgres"):
        raise ConfigError("POSTGRES_URL must start with postgresql:// (see .env.example).")
    return Settings(gemini_api_key=key, postgres_url=url)
