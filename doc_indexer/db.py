"""PostgreSQL + pgvector storage and similarity search.

The schema is created automatically on first use (``CREATE ... IF NOT EXISTS``),
so a fresh database only needs the pgvector extension to be available
(the ``pgvector/pgvector`` Docker image ships with it).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import Settings
from .errors import DatabaseError

TABLE = "document_chunks"


def schema_sql(dim: int) -> list[str]:
    """DDL statements (also documented in sql/schema.sql)."""
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id             BIGSERIAL    PRIMARY KEY,
            chunk_text     TEXT         NOT NULL,
            embedding      VECTOR({int(dim)}) NOT NULL,
            filename       TEXT         NOT NULL,
            split_strategy TEXT         NOT NULL CHECK (split_strategy IN ('fixed', 'sentence', 'paragraph')),
            chunk_index    INTEGER      NOT NULL,
            content_hash   TEXT         NOT NULL,
            created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (filename, split_strategy, content_hash)
        )""",
        # Approximate nearest-neighbour index for cosine distance (<=>).
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_embedding ON {TABLE} USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_file_strategy ON {TABLE} (filename, split_strategy)",
    ]


@dataclass(frozen=True)
class SearchHit:
    id: int
    filename: str
    split_strategy: str
    chunk_index: int
    chunk_text: str
    score: float  # cosine similarity: 1.0 = same meaning, ~0 = unrelated


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def connect(settings: Settings, dim: int) -> psycopg.Connection:
    """Open a connection, make sure the schema exists, and register the vector type."""
    target = settings.safe_db_target
    try:
        # autocommit: every `with conn.transaction()` below is a real BEGIN/COMMIT, not a savepoint.
        conn = psycopg.connect(settings.postgres_url, connect_timeout=10, autocommit=True)
        conn.prepare_threshold = None  # short-lived CLI: server-side prepared statements add nothing
    except psycopg.OperationalError as exc:
        raise DatabaseError(
            f"Cannot connect to PostgreSQL at {target}. Is the database running? "
            f"(docker compose up -d) Check POSTGRES_URL in .env."
        ) from exc
    try:
        with conn.transaction(), conn.cursor() as cur:
            for stmt in schema_sql(dim):
                cur.execute(stmt)
        register_vector(conn)
        _check_dimension(conn, dim)
    except DatabaseError:
        conn.close()
        raise
    except psycopg.Error as exc:
        conn.close()
        detail = (exc.diag.message_primary if getattr(exc, "diag", None) else None) or str(exc)
        if "vector" in detail and ("extension" in detail or "control file" in detail):
            detail += " - the pgvector extension is not installed on this server (use the pgvector/pgvector image)"
        raise DatabaseError(f"Database setup failed at {target}: {detail}") from exc
    return conn


def _check_dimension(conn: psycopg.Connection, dim: int) -> None:
    """Fail clearly if an existing table was created with a different vector size."""
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
        (TABLE,),
    ).fetchone()
    if row and row[0] not in (-1, dim):
        raise DatabaseError(
            f"Table {TABLE} stores {row[0]}-dimension vectors but this code uses {dim}. "
            f"Drop the table or use a new database."
        )


def existing_hashes(conn: psycopg.Connection, filename: str, strategy: str) -> set[str]:
    rows = conn.execute(
        f"SELECT content_hash FROM {TABLE} WHERE filename = %s AND split_strategy = %s",
        (filename, strategy),
    ).fetchall()
    return {r[0] for r in rows}


def delete_document(conn: psycopg.Connection, filename: str, strategy: str) -> int:
    cur = conn.execute(f"DELETE FROM {TABLE} WHERE filename = %s AND split_strategy = %s", (filename, strategy))
    return cur.rowcount


def insert_chunks(
    conn: psycopg.Connection,
    filename: str,
    strategy: str,
    rows: Iterable[tuple[int, str, np.ndarray]],
) -> int:
    """Insert ``(chunk_index, chunk_text, embedding)`` rows in one transaction; returns rows inserted."""
    sql = (
        f"INSERT INTO {TABLE} (chunk_text, embedding, filename, split_strategy, chunk_index, content_hash) "
        f"VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (filename, split_strategy, content_hash) DO NOTHING"
    )
    params = [(text, vec, filename, strategy, idx, content_hash(text)) for idx, text, vec in rows]
    if not params:
        return 0
    try:
        with conn.transaction(), conn.cursor() as cur:  # all-or-nothing
            cur.executemany(sql, params)
            inserted = cur.rowcount  # total rows inserted (conflicts count as 0)
    except psycopg.Error as exc:
        raise DatabaseError(f"Failed to store chunks: {exc.diag.message_primary or exc}") from exc
    return inserted


def count_rows(conn: psycopg.Connection) -> int:
    return conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]


def search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    top_k: int = 5,
    strategy: str | None = None,
    filename: str | None = None,
    min_score: float | None = None,
) -> list[SearchHit]:
    """Return the ``top_k`` chunks closest to ``query_vec`` by cosine similarity."""
    sql = f"""
        SELECT id, filename, split_strategy, chunk_index, chunk_text,
               1 - (embedding <=> %(q)s) AS score
        FROM {TABLE}
        WHERE (%(strategy)s::text IS NULL OR split_strategy = %(strategy)s)
          AND (%(filename)s::text IS NULL OR filename = %(filename)s)
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
    """
    try:
        rows = conn.execute(sql, {"q": query_vec, "strategy": strategy, "filename": filename, "k": top_k}).fetchall()
    except psycopg.Error as exc:
        raise DatabaseError(f"Search query failed: {exc.diag.message_primary or exc}") from exc
    hits = [SearchHit(r[0], r[1], r[2], r[3], r[4], float(r[5])) for r in rows]
    if min_score is not None:
        hits = [h for h in hits if h.score >= min_score]
    return hits


def indexed_files(conn: psycopg.Connection) -> Sequence[tuple[str, str, int]]:
    return conn.execute(
        f"SELECT filename, split_strategy, count(*) FROM {TABLE} GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
