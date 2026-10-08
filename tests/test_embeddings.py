"""Embedding client tests with a fake Gemini client - no network or API key needed."""

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors

from doc_indexer.embeddings import GeminiEmbedder
from doc_indexer.errors import EmbeddingError


class FakeModels:
    def __init__(self, dim=768, fail_times=0, fail_code=429):
        self.dim, self.fail_times, self.fail_code = dim, fail_times, fail_code
        self.calls = []

    def embed_content(self, model, contents, config):
        self.calls.append((model, list(contents), config.task_type, config.output_dimensionality))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise errors.ClientError(self.fail_code, {"error": {"code": self.fail_code, "message": "quota", "status": "X"}})
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0] + [0.0] * (self.dim - 2)) for _ in contents])


def make(models, **kw):
    return GeminiEmbedder("test-key", client=SimpleNamespace(models=models), base_delay=0, **kw)


def test_vectors_are_normalized_and_sized():
    vecs = make(FakeModels()).embed_documents(["a", "b"])
    assert len(vecs) == 2 and vecs[0].shape == (768,)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0)


def test_batching_and_task_types():
    m = FakeModels()
    e = make(m, batch_size=2)
    e.embed_documents(["a", "b", "c", "d", "e"])
    e.embed_query("q")
    assert [len(c[1]) for c in m.calls] == [2, 2, 1, 1]
    assert {c[2] for c in m.calls[:3]} == {"RETRIEVAL_DOCUMENT"} and m.calls[3][2] == "RETRIEVAL_QUERY"
    assert all(c[3] == 768 for c in m.calls)


def test_retries_on_rate_limit_then_succeeds():
    m = FakeModels(fail_times=2, fail_code=429)
    assert len(make(m).embed_documents(["a"])) == 1
    assert len(m.calls) == 3


def test_gives_up_after_max_attempts():
    with pytest.raises(EmbeddingError, match="after 3 attempts"):
        make(FakeModels(fail_times=10, fail_code=503), max_attempts=3).embed_documents(["a"])


def test_invalid_key_is_not_retried():
    m = FakeModels(fail_times=10, fail_code=400)
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        make(m).embed_documents(["a"])
    assert len(m.calls) == 1


def test_wrong_dimension_is_rejected():
    with pytest.raises(EmbeddingError, match="768"):
        make(FakeModels(dim=3072)).embed_documents(["a"])


def test_certificate_error_is_not_retried():
    class CertFail(FakeModels):
        def embed_content(self, model, contents, config):
            self.calls.append(1)
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    m = CertFail()
    with pytest.raises(EmbeddingError, match="certificate"):
        make(m).embed_documents(["a"])
    assert len(m.calls) == 1