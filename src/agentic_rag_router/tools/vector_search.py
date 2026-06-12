"""The `vector_search` tool --- cosine top-k over the arXiv abstract corpus.

Embeds the query with the *same* pinned MiniLM model and revision used at
ingestion (`scripts/ingest_corpus.py`), then ranks `corpus_docs` by cosine
distance and returns the nearest `k`. Using the identical model/revision is
not optional: a different model --- or even a different revision of the same
model --- produces an embedding space the stored vectors do not live in, and
the cosine ranking becomes noise.

The embedder and the repository are both ports, so the orchestration logic is
exercised in unit tests with a fake 384-dim embedder and a fake repository ---
no `sentence-transformers` (and the torch it pulls) and no live database.

Unlike `sql_query`, this tool connects as the **dev** role, not `router_ro`:
`router_ro` was granted `SELECT` on `taxi_trips` only, not `corpus_docs`. The
read-only-role backstop is specific to the SQL tool; the vector tool's query
is a fixed, parameterised `SELECT` that never interpolates caller input.
"""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

from agentic_rag_router.tools.envelope import (
    ERROR_BACKEND,
    TOOL_VECTOR_SEARCH,
    ToolResult,
    error_result,
    ok_result,
)

# Pinned to match ingestion. Keep in lockstep with `scripts/ingest_corpus.py`
# and `docs/DATA_SOURCES.md`; the revision pins the exact weights.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIM = 384

DEFAULT_K = 5

# Fixed, fully parameterised ranking query. Cosine distance (`<=>`) over the
# HNSW `vector_cosine_ops` index; similarity is `1 - distance` for the
# L2-normalised vectors the corpus stores.
_TOP_K_SQL = (
    "SELECT arxiv_id, title, abstract, published_date, "
    "1 - (embedding <=> %s::vector) AS similarity "
    "FROM corpus_docs "
    "ORDER BY embedding <=> %s::vector "
    "LIMIT %s"
)


class EmbedderPort(Protocol):
    """Turns a string into a dense embedding vector.

    The returned vector must match the dimensionality and the model the
    corpus was embedded with (`EMBEDDING_DIM`, `MODEL_NAME`/`MODEL_REVISION`).
    """

    def embed(self, text: str) -> list[float]:
        """Return the embedding of `text` as a list of floats."""
        ...


class VectorRepository(Protocol):
    """Ranks the corpus against a query embedding and returns the nearest rows."""

    def top_k(self, embedding: list[float], k: int) -> list[dict[str, object]]:
        """Return the `k` nearest rows (arxiv_id/title/abstract/published_date/similarity)."""
        ...


def vector_search(
    query: str,
    k: int = DEFAULT_K,
    *,
    embedder: EmbedderPort,
    repository: VectorRepository,
) -> ToolResult:
    """Embed `query` and return the `k` most similar corpus abstracts.

    `query` must be non-empty and `k` must be >= 1 --- violating either is a
    caller bug and raises `ValueError`. A failure inside the embedder or the
    repository is an operational failure and comes back as a `ToolResult` with
    ``ok=False`` and `ERROR_BACKEND`.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty or whitespace-only")
    if k < 1:
        raise ValueError("k must be >= 1")

    start = time.perf_counter()
    try:
        embedding = embedder.embed(query)
        rows = repository.top_k(embedding, k)
    except Exception as exc:  # embedder/repository failure becomes a failed envelope
        latency_ms = int((time.perf_counter() - start) * 1000)
        return error_result(TOOL_VECTOR_SEARCH, ERROR_BACKEND, str(exc), latency_ms)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return ok_result(TOOL_VECTOR_SEARCH, rows, latency_ms)


class SentenceTransformerEmbedder:
    """`EmbedderPort` backed by the pinned local MiniLM model.

    `sentence_transformers` (and torch) is imported lazily and the model is
    loaded once on first use, so constructing this adapter --- and importing
    the tools package --- costs nothing until an embedding is actually needed.
    """

    def __init__(self, *, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION) -> None:
        self._model_name = model_name
        self._revision = revision
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, revision=self._revision)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in vector]


def _vector_conn_params() -> dict[str, str]:
    """libpq connection keywords for the read substrate (dev role).

    Mirrors `scripts/_db.py` defaults. `router_ro` is intentionally *not* used
    here --- it has no `SELECT` on `corpus_docs`.
    """
    return {
        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "port": os.environ.get("POSTGRES_PORT", "5436"),
        "dbname": os.environ.get("POSTGRES_DB", "dev"),
        "user": os.environ.get("POSTGRES_USER", "dev"),
        "password": os.environ.get("POSTGRES_PASSWORD", "dev"),
    }


class PgVectorRepository:
    """`VectorRepository` over the `corpus_docs` pgvector table.

    `psycopg` is imported lazily (the `ingest` group). The query is the fixed
    `_TOP_K_SQL` with the embedding bound as a parameter twice (once for the
    similarity projection, once for the ORDER BY) and `k` as the LIMIT.
    """

    def top_k(self, embedding: list[float], k: int) -> list[dict[str, object]]:
        import psycopg
        from psycopg.conninfo import make_conninfo
        from psycopg.rows import dict_row

        vector_literal = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        conninfo = make_conninfo(**_vector_conn_params())
        with psycopg.connect(conninfo, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(_TOP_K_SQL, (vector_literal, vector_literal, k))
                rows = cur.fetchall()
            conn.rollback()
        return [dict(row) for row in rows]
