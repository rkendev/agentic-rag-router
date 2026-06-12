"""Integration tests for the D2 data layer.

Marked `integration`, so excluded from `make check`. They require the
docker-compose pgvector service up and both ingest scripts run:

    docker compose up -d postgres
    uv sync --group ingest
    python -m scripts.init_db
    python -m scripts.ingest_taxi
    python -m scripts.ingest_corpus
    uv run pytest tests/integration -m integration
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from scripts import _db

# Skip this whole module at *collection* time when psycopg is absent — pytest
# imports the module before marker deselection runs, and CI omits the `ingest`
# dependency group. importorskip must precede the psycopg import below.
pytest.importorskip("psycopg")

import psycopg  # noqa: E402 - intentionally imported after the importorskip guard

pytestmark = pytest.mark.integration

TAXI_MIN_ROWS = 500_000
CORPUS_MIN_DOCS = 10_000
EMBEDDING_DIM = 384


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection[Any]]:
    with _db.connect() as connection:
        yield connection


def test_taxi_trips_has_at_least_500k_rows(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM taxi_trips")
        (count,) = cur.fetchone()  # type: ignore[misc]
    assert count >= TAXI_MIN_ROWS


def test_corpus_docs_has_at_least_10k_docs(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM corpus_docs")
        (count,) = cur.fetchone()  # type: ignore[misc]
    assert count >= CORPUS_MIN_DOCS


def test_corpus_embedding_dimension(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT vector_dims(embedding) FROM corpus_docs LIMIT 1")
        (dim,) = cur.fetchone()  # type: ignore[misc]
    assert dim == EMBEDDING_DIM


def test_pgvector_cosine_query_returns_k_results(conn: psycopg.Connection[Any]) -> None:
    k = 5
    with conn.cursor() as cur:
        # Use a real stored embedding as the probe, then rank by cosine distance.
        cur.execute("SELECT embedding FROM corpus_docs LIMIT 1")
        (probe,) = cur.fetchone()  # type: ignore[misc]
        cur.execute(
            "SELECT arxiv_id FROM corpus_docs ORDER BY embedding <=> %s::vector LIMIT %s",
            (probe, k),
        )
        hits = cur.fetchall()
    assert len(hits) == k


def test_router_ro_can_select(conn: psycopg.Connection[Any]) -> None:
    user, password = _db.router_ro_credentials()
    with _db.connect(user=user, password=password) as ro, ro.cursor() as cur:
        cur.execute("SELECT count(*) FROM taxi_trips")
        (count,) = cur.fetchone()  # type: ignore[misc]
    assert count >= TAXI_MIN_ROWS


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO taxi_trips (vendor_id) VALUES (1)",
        "UPDATE taxi_trips SET vendor_id = 1 WHERE false",
        "DELETE FROM taxi_trips WHERE false",
    ],
)
def test_router_ro_cannot_write(statement: str) -> None:
    user, password = _db.router_ro_credentials()
    with _db.connect(user=user, password=password) as ro:
        with ro.cursor() as cur, pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(statement)
        ro.rollback()
