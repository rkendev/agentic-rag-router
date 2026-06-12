"""Unit tests for `vector_search`: orchestration plus the real adapters.

The orchestration is exercised with a fake embedder and a fake repository; the
real `SentenceTransformerEmbedder` and `PgVectorRepository` run against
injected fake modules so their lazy-import paths are covered without
sentence-transformers or psycopg installed.
"""

from __future__ import annotations

import pytest

from agentic_rag_router.tools.envelope import ERROR_BACKEND, TOOL_VECTOR_SEARCH
from agentic_rag_router.tools.vector_search import (
    EMBEDDING_DIM,
    MODEL_NAME,
    MODEL_REVISION,
    PgVectorRepository,
    SentenceTransformerEmbedder,
    vector_search,
)
from tests.unit.tools.fakes import (
    FakeEmbedder,
    FakeSentenceTransformer,
    FakeVectorRepository,
    install_fake_psycopg,
    install_fake_sentence_transformers,
)

_ROWS: list[dict[str, object]] = [
    {
        "arxiv_id": "2506.00001",
        "title": "A",
        "abstract": "...",
        "published_date": "2026-06-01",
        "similarity": 0.91,
    },
    {
        "arxiv_id": "2506.00002",
        "title": "B",
        "abstract": "...",
        "published_date": "2026-05-30",
        "similarity": 0.88,
    },
]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_vector_search_success() -> None:
    embedder = FakeEmbedder()
    repo = FakeVectorRepository(_ROWS)
    result = vector_search("transformers for retrieval", k=2, embedder=embedder, repository=repo)

    assert result.ok is True
    assert result.tool == TOOL_VECTOR_SEARCH
    assert result.data == _ROWS
    assert result.latency_ms >= 0
    assert embedder.calls == ["transformers for retrieval"]
    assert repo.last_k == 2
    assert repo.last_embedding == [0.1] * EMBEDDING_DIM


def test_vector_search_respects_k() -> None:
    repo = FakeVectorRepository(_ROWS)
    result = vector_search("q", k=1, embedder=FakeEmbedder(), repository=repo)
    assert result.data == _ROWS[:1]


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
def test_vector_search_empty_query_raises(bad_query: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        vector_search(bad_query, embedder=FakeEmbedder(), repository=FakeVectorRepository())


def test_vector_search_bad_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        vector_search("q", k=0, embedder=FakeEmbedder(), repository=FakeVectorRepository())


def test_vector_search_backend_error_envelope() -> None:
    repo = FakeVectorRepository(error=RuntimeError("connection refused"))
    result = vector_search("q", embedder=FakeEmbedder(), repository=repo)
    assert result.ok is False
    assert result.error_code == ERROR_BACKEND
    assert result.error_message is not None
    assert "connection refused" in result.error_message


# ---------------------------------------------------------------------------
# Real SentenceTransformerEmbedder over a fake module
# ---------------------------------------------------------------------------


def test_sentence_transformer_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder()
    vector = embedder.embed("a query")

    assert len(vector) == EMBEDDING_DIM
    assert all(isinstance(v, float) for v in vector)
    # pinned model + revision were passed to the loader
    assert len(FakeSentenceTransformer.instances) == 1
    loaded = FakeSentenceTransformer.instances[0]
    assert loaded.model_name == MODEL_NAME
    assert loaded.revision == MODEL_REVISION


def test_sentence_transformer_embedder_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder()
    embedder.embed("first")
    embedder.embed("second")
    # model constructed exactly once (lazy load cached), encoded twice
    assert len(FakeSentenceTransformer.instances) == 1
    assert FakeSentenceTransformer.instances[0].encode_calls == 2


# ---------------------------------------------------------------------------
# Real PgVectorRepository over a fake psycopg
# ---------------------------------------------------------------------------


def test_pgvector_repository_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    record = install_fake_psycopg(monkeypatch, _ROWS)
    repo = PgVectorRepository()
    out = repo.top_k([0.1] * EMBEDDING_DIM, k=2)

    assert out == _ROWS
    # connected as the dev role (NOT router_ro, which lacks SELECT on corpus_docs)
    assert record.conninfo is not None
    assert "user=dev" in record.conninfo
    # query bound the vector literal twice and k once
    sql, params = record.cursor.executed[0]
    assert "corpus_docs" in sql
    assert "<=>" in sql
    assert isinstance(params, tuple)
    assert params[0].startswith("[") and params[0].endswith("]")
    assert params[2] == 2
    assert record.connection.rolled_back is True
