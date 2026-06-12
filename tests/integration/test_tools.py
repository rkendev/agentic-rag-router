"""Integration tests for the D3 tool adapters.

Marked `integration`, so excluded from `make check`. They require the live
data layer (docker-compose pgvector with both ingest scripts run) and, for the
vector tool, the `ingest` dependency group (`sentence-transformers`):

    docker compose up -d postgres
    uv sync --group ingest
    python -m scripts.init_db
    python -m scripts.ingest_taxi
    python -m scripts.ingest_corpus
    uv run pytest tests/integration/test_tools.py -m integration

The web_search test performs one real Tavily call and is skipped when
`TAVILY_API_KEY` is absent (the request/response shape is also pinned offline by
the recorded cassette in `tests/unit/tools/`).
"""

from __future__ import annotations

import os

import pytest

# Skip the whole module at collection time when psycopg is absent (CI omits the
# `ingest` group); importorskip must precede the psycopg import below.
pytest.importorskip("psycopg")

import psycopg  # noqa: E402 - intentionally imported after the importorskip guard

from agentic_rag_router.tools import sql_query, vector_search, web_search  # noqa: E402
from agentic_rag_router.tools.envelope import ERROR_VALIDATION  # noqa: E402
from agentic_rag_router.tools.sql_query import RouterRoExecutor  # noqa: E402
from agentic_rag_router.tools.vector_search import (  # noqa: E402
    PgVectorRepository,
    SentenceTransformerEmbedder,
)

pytestmark = pytest.mark.integration

TAXI_MIN_ROWS = 500_000


def test_vector_search_returns_k_results_descending() -> None:
    pytest.importorskip("sentence_transformers")
    result = vector_search(
        "retrieval augmented generation with dense embeddings",
        k=5,
        embedder=SentenceTransformerEmbedder(),
        repository=PgVectorRepository(),
    )
    assert result.ok, result.error_message
    assert result.data is not None
    assert len(result.data) == 5

    first = result.data[0]
    assert set(first) == {"arxiv_id", "title", "abstract", "published_date", "similarity"}

    sims = [row["similarity"] for row in result.data]
    assert all(isinstance(s, float) for s in sims)
    floats = [s for s in sims if isinstance(s, float)]
    assert floats == sorted(floats, reverse=True)


def test_sql_query_real_aggregate_as_router_ro() -> None:
    result = sql_query("SELECT count(*) AS n FROM taxi_trips", executor=RouterRoExecutor())
    assert result.ok, result.error_message
    assert result.data is not None
    count = result.data[0]["n"]
    assert isinstance(count, int)
    assert count >= TAXI_MIN_ROWS


def test_sql_query_rejects_write_at_the_front_door() -> None:
    result = sql_query("DELETE FROM taxi_trips WHERE false", executor=RouterRoExecutor())
    assert result.ok is False
    assert result.error_code == ERROR_VALIDATION
    assert result.data is None


def test_router_ro_grant_blocks_an_unvalidated_write() -> None:
    # Defence in depth: bypass the validator and hit the executor directly ---
    # the `router_ro` SELECT-only grant must still refuse the write.
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        RouterRoExecutor().execute("DELETE FROM taxi_trips WHERE false")


@pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"), reason="web_search integration needs TAVILY_API_KEY"
)
def test_web_search_real_tavily_call() -> None:
    result = web_search("latest arXiv cs.CL retrieval papers", max_results=3)
    assert result.ok, result.error_message
    assert result.data is not None
    assert len(result.data) >= 1
    first = result.data[0]
    assert set(first) == {"title", "url", "snippet", "published"}
