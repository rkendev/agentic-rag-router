"""Keyless offline replay of the two README examples.

The README shows two live ``POST /ask`` calls: an answerable taxi-statistics
question that the router answers with a citation, and an unanswerable
("next Saturday") question that the router refuses with zero citations. These
tests reproduce both from recorded Anthropic cassettes plus in-memory substrate
fakes, so a reviewer can run them with no API key, no database, and no network:

    make demo-replay

The fakes return the same evidence the README prints, so the replayed trajectory
matches the documented one. SQL and vector calls are in-process fakes (not HTTP),
so the cassette only carries the Sonnet turns.

Recording (the one step that costs API credit, a few cents, run once):

    RUN_LIVE=1 ANTHROPIC_API_KEY=sk-... \
      uv run pytest tests/replay/ --record-mode=once

That writes scrubbed cassettes under ``tests/replay/cassettes/``; commit them.
Afterwards the tests replay offline and are part of the normal suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.router.client import AnthropicRouterClient
from agentic_rag_router.router.dispatch import Dispatcher
from agentic_rag_router.router.loop import (
    EVIDENCE_REFUSAL_REASONS,
    RouterResponse,
    run_router,
)
from agentic_rag_router.router.schema import TOOLS
from agentic_rag_router.tools.envelope import TOOL_SQL_QUERY
from tests.unit.tools.fakes import FakeEmbedder, FakeSqlExecutor, FakeVectorRepository

# The two README questions, read verbatim from the front-door doc examples.
_ANSWERABLE_Q = "What was the average trip distance across all taxi trips in the dataset?"
_REFUSAL_Q = "Exactly how many taxi trips will occur in New York City next Saturday?"

_CASSETTE_DIR = Path(__file__).parent / "cassettes" / "test_router_replay"


def _skip_unless_runnable(name: str) -> pytest.MarkDecorator:
    """Skip in replay only when the cassette is absent.

    During a recording run (``RUN_LIVE=1``) the cassette does not exist yet, so
    the test must NOT be skipped or there is nothing to record. In every other
    run the test is skipped until its cassette has been committed.
    """
    runnable = (_CASSETTE_DIR / f"{name}.yaml").is_file() or os.environ.get("RUN_LIVE") == "1"
    return pytest.mark.skipif(
        not runnable,
        reason="cassette not recorded yet; see module docstring for the record command",
    )


class _FakeWebClient:
    """Web client that returns nothing: the taxi questions never route to web."""

    def search(self, query: str, max_results: int) -> list[dict[str, object]]:
        return []


def _dispatcher_with(sql_rows: list[dict[str, object]]) -> Dispatcher:
    """Dispatcher backed by in-memory fakes; SQL rows mirror the README output."""
    vector_rows: list[dict[str, object]] = [
        {
            "arxiv_id": "2401.00001",
            "title": "On Self-Attention",
            "abstract": "Self-attention relates each token to every other token.",
            "published_date": "2026-01-01",
            "similarity": 0.91,
        }
    ]
    return Dispatcher(
        embedder=FakeEmbedder(),
        repository=FakeVectorRepository(rows=vector_rows),
        executor=FakeSqlExecutor(rows=sql_rows),
        web_client=_FakeWebClient(),
    )


def _run(question: str, sql_rows: list[dict[str, object]]) -> RouterResponse:
    client = AnthropicRouterClient.from_settings(Settings(), max_tokens=512)
    return run_router(
        question,
        client=client,
        tools=TOOLS,
        dispatcher=_dispatcher_with(sql_rows),
    )


@_skip_unless_runnable("test_answerable_taxi_question_is_answered_with_a_citation")
@pytest.mark.vcr
def test_answerable_taxi_question_is_answered_with_a_citation() -> None:
    """README example 1: a real route + answer, carrying at least one citation."""
    result = _run(_ANSWERABLE_Q, sql_rows=[{"avg_trip_distance": 3.65}])

    assert result.refusal_reason is None
    assert result.answer is not None and result.answer.strip()
    assert len(result.citations) >= 1
    assert TOOL_SQL_QUERY in [step.tool for step in result.trajectory]


@_skip_unless_runnable("test_unanswerable_future_question_is_refused_with_zero_citations")
@pytest.mark.vcr
def test_unanswerable_future_question_is_refused_with_zero_citations() -> None:
    """README example 2: a sufficient tool result, but the model still refuses."""
    result = _run(_REFUSAL_Q, sql_rows=[{"count": 2_964_624}])

    assert result.refusal_reason in EVIDENCE_REFUSAL_REASONS
    assert result.answer is None
    assert result.citations == []
