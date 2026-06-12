"""Live end-to-end router tests --- real Claude Sonnet routing two goldens.

Two variants, both routing a `vector_only` golden and a `sql_only` golden read
verbatim from the frozen eval set:

* **Port-fake** (``live``) --- real Sonnet + the real router/dispatch/loop, with
  the substrate ports as in-memory fakes. Proves the live model routes into the
  loop correctly without needing a database. Needs only ``ANTHROPIC_API_KEY``.
* **True-substrate** (``live`` + ``integration``) --- real Sonnet + the real
  `Dispatcher` over the pinned MiniLM embedder + pgvector repository
  (`vector_search`) and the read-only ``router_ro`` executor (`sql_query`),
  against the live pgvector database. Proves the whole stack end to end.

Both are opt-in: they skip unless ``RUN_LIVE=1`` and ``ANTHROPIC_API_KEY`` are
set. The true-substrate variant additionally needs the ``ingest`` group
(psycopg + sentence-transformers) and a reachable database, guarded with
``importorskip`` + a connectivity probe.

    RUN_LIVE=1 uv run pytest tests/live/test_router_live.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.router.client import AnthropicRouterClient
from agentic_rag_router.router.dispatch import Dispatcher
from agentic_rag_router.router.loop import RouterResponse, run_router
from agentic_rag_router.router.schema import TOOLS
from agentic_rag_router.tools.envelope import TOOL_SQL_QUERY, TOOL_VECTOR_SEARCH
from tests.unit.tools.fakes import FakeEmbedder, FakeSqlExecutor, FakeVectorRepository

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GOLDENS = _REPO_ROOT / "data" / "eval" / "golden_questions.jsonl"

_LIVE = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE") == "1" and os.environ.get("ANTHROPIC_API_KEY")),
    reason="router live tests need RUN_LIVE=1 and ANTHROPIC_API_KEY",
)


def _first_question(label: str) -> str:
    """Return the first non-adversarial golden of ``label``, read verbatim."""
    for line in _GOLDENS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["label"] == label and not record.get("adversarial"):
            question: str = record["question"]
            return question
    raise AssertionError(f"no non-adversarial golden for label {label!r}")


class _FakeWebClient:
    """A web client that returns nothing --- vector/sql goldens never hit it."""

    def search(self, query: str, max_results: int) -> list[dict[str, object]]:
        return []


def _fake_dispatcher() -> Dispatcher:
    """Dispatcher backed entirely by in-memory port fakes (no substrates)."""
    vector_rows: list[dict[str, object]] = [
        {
            "arxiv_id": "2401.00001",
            "title": "On Self-Attention",
            "abstract": "Self-attention relates each token to every other token.",
            "published_date": "2026-01-01",
            "similarity": 0.91,
        }
    ]
    sql_rows: list[dict[str, object]] = [{"avg_trip_distance": 3.21}]
    return Dispatcher(
        embedder=FakeEmbedder(),
        repository=FakeVectorRepository(rows=vector_rows),
        executor=FakeSqlExecutor(rows=sql_rows),
        web_client=_FakeWebClient(),
    )


def _assert_envelope(result: RouterResponse) -> None:
    """Assert the RouterResponse shape D4 promises (no grade field yet)."""
    assert isinstance(result, RouterResponse)
    assert isinstance(result.citations, list)
    assert isinstance(result.trajectory, list)
    assert isinstance(result.iterations, int) and result.iterations >= 1
    assert result.refusal_reason is None


def _tools_hit(result: RouterResponse) -> list[str]:
    return [step.tool for step in result.trajectory]


# ---------------------------------------------------------------------------
# Variant 1 --- real Sonnet, fake substrates
# ---------------------------------------------------------------------------


@_LIVE
@pytest.mark.live
def test_live_vector_only_routes_to_vector_search_with_fakes() -> None:
    client = AnthropicRouterClient.from_settings(Settings(), max_tokens=512)
    result = run_router(
        _first_question("vector_only"),
        client=client,
        tools=TOOLS,
        dispatcher=_fake_dispatcher(),
    )
    assert TOOL_VECTOR_SEARCH in _tools_hit(result)
    assert result.trajectory[0].tool == TOOL_VECTOR_SEARCH
    assert result.trajectory[0].ok is True
    _assert_envelope(result)


@_LIVE
@pytest.mark.live
def test_live_sql_only_routes_to_sql_query_with_fakes() -> None:
    client = AnthropicRouterClient.from_settings(Settings(), max_tokens=512)
    result = run_router(
        _first_question("sql_only"),
        client=client,
        tools=TOOLS,
        dispatcher=_fake_dispatcher(),
    )
    assert TOOL_SQL_QUERY in _tools_hit(result)
    assert result.trajectory[0].tool == TOOL_SQL_QUERY
    assert result.trajectory[0].ok is True
    _assert_envelope(result)


# ---------------------------------------------------------------------------
# Variant 2 --- real Sonnet, real substrates (pgvector + router_ro)
# ---------------------------------------------------------------------------


def _real_dispatcher() -> Dispatcher:
    """Dispatcher over the real adapters; skips if the data layer is absent."""
    pytest.importorskip("psycopg")
    pytest.importorskip("sentence_transformers")

    import psycopg
    from psycopg.conninfo import make_conninfo

    from agentic_rag_router.tools.sql_query import RouterRoExecutor
    from agentic_rag_router.tools.vector_search import (
        PgVectorRepository,
        SentenceTransformerEmbedder,
    )

    # Connectivity probe: skip (don't fail) when the DB isn't reachable.
    params = {
        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "port": os.environ.get("POSTGRES_PORT", "5436"),
        "dbname": os.environ.get("POSTGRES_DB", "dev"),
        "user": os.environ.get("POSTGRES_USER", "dev"),
        "password": os.environ.get("POSTGRES_PASSWORD", "dev"),
    }
    try:
        with psycopg.connect(make_conninfo(**params), connect_timeout=3) as conn:
            conn.rollback()
    except psycopg.OperationalError as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"pgvector database not reachable: {exc}")

    return Dispatcher(
        embedder=SentenceTransformerEmbedder(),
        repository=PgVectorRepository(),
        executor=RouterRoExecutor(),
    )


@_LIVE
@pytest.mark.live
@pytest.mark.integration
def test_live_vector_only_end_to_end_true_substrate() -> None:
    client = AnthropicRouterClient.from_settings(Settings(), max_tokens=1024)
    result = run_router(
        _first_question("vector_only"),
        client=client,
        tools=TOOLS,
        dispatcher=_real_dispatcher(),
    )
    assert TOOL_VECTOR_SEARCH in _tools_hit(result)
    assert any(step.tool == TOOL_VECTOR_SEARCH and step.ok for step in result.trajectory)
    assert result.answer is not None
    _assert_envelope(result)


@_LIVE
@pytest.mark.live
@pytest.mark.integration
def test_live_sql_only_end_to_end_true_substrate() -> None:
    client = AnthropicRouterClient.from_settings(Settings(), max_tokens=1024)
    result = run_router(
        _first_question("sql_only"),
        client=client,
        tools=TOOLS,
        dispatcher=_real_dispatcher(),
    )
    assert TOOL_SQL_QUERY in _tools_hit(result)
    assert any(step.tool == TOOL_SQL_QUERY and step.ok for step in result.trajectory)
    assert result.answer is not None
    _assert_envelope(result)
