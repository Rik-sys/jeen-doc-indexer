"""Embeddings with Google's ``gemini-embedding-001`` via the ``google-genai`` SDK.

* Documents and queries use different task types (``RETRIEVAL_DOCUMENT`` /
  ``RETRIEVAL_QUERY``); the model optimizes each side of the search accordingly.
* Vectors are requested at 768 dimensions and L2-normalized here: Google only
  normalizes the full 3,072-dimension output, so truncated vectors must be
  normalized by the caller for cosine similarity to be meaningful.
* Texts are sent in batches, and rate-limit / server errors are retried with
  exponential backoff (the free tier returns HTTP 429 quickly).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Protocol, Sequence

import numpy as np

from .errors import EmbeddingError

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class Embedder(Protocol):
    """Anything that turns texts into vectors (lets tests swap in a fake)."""

    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        batch_size: int = 50,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        client=None,
    ):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.batches_sent = 0

    # public -----------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(list(texts[i : i + self.batch_size]), "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]

    # internals --------------------------------------------------------------
    def _embed_batch(self, batch: list[str], task_type: str) -> list[np.ndarray]:
        from google.genai import errors, types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._client.models.embed_content(model=self.model, contents=batch, config=config)
                self.batches_sent += 1
                return self._to_vectors(resp, expected=len(batch))
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                last_error = f"HTTP {code}: {_short(getattr(exc, 'message', None) or str(exc))}"
                if code not in RETRYABLE_STATUS:
                    raise EmbeddingError(f"Embedding request rejected ({last_error}).{_hint(code)}") from exc
            except EmbeddingError:
                raise
            except Exception as exc:  # network errors, timeouts
                last_error = f"{type(exc).__name__}: {_short(str(exc))}"
            if attempt < self.max_attempts:
                delay = self.base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.5)
                log.warning("Embedding attempt %d/%d failed (%s). Retrying in %.1fs",
                            attempt, self.max_attempts, last_error, delay)
                time.sleep(delay)
        raise EmbeddingError(f"Embedding failed after {self.max_attempts} attempts: {last_error}")

    def _to_vectors(self, resp, expected: int) -> list[np.ndarray]:
        embeddings = getattr(resp, "embeddings", None) or []
        if len(embeddings) != expected:
            raise EmbeddingError(f"Embedding API returned {len(embeddings)} vectors for {expected} texts")
        out = []
        for e in embeddings:
            v = np.asarray(e.values, dtype=np.float32)
            if v.shape != (self.dim,):
                raise EmbeddingError(f"Expected {self.dim}-dimension vectors, got {v.shape[0]}")
            out.append(normalize(v))
        return out


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise EmbeddingError("Embedding API returned an all-zero vector")
    return (v / norm).astype(np.float32)


def _short(msg: str, limit: int = 160) -> str:
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


def _hint(code) -> str:
    if code in (400, 401, 403):
        return " Check that GEMINI_API_KEY in .env is valid."
    if code == 404:
        return " Check the model name and that the API is enabled for your key."
    return ""
