-- Schema for the document index. The application creates it automatically
-- (doc_indexer/db.py, CREATE ... IF NOT EXISTS); this file documents it and
-- can be run manually:  psql "$POSTGRES_URL" -f sql/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id             BIGSERIAL    PRIMARY KEY,
    chunk_text     TEXT         NOT NULL,
    embedding      VECTOR(768)  NOT NULL,          -- gemini-embedding-001, output_dimensionality=768, L2-normalized
    filename       TEXT         NOT NULL,          -- base name only, e.g. example.pdf
    split_strategy TEXT         NOT NULL CHECK (split_strategy IN ('fixed', 'sentence', 'paragraph')),
    chunk_index    INTEGER      NOT NULL,          -- position of the chunk in the document
    content_hash   TEXT         NOT NULL,          -- sha256(chunk_text): makes re-indexing idempotent
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (filename, split_strategy, content_hash)
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_document_chunks_file_strategy
    ON document_chunks (filename, split_strategy);
