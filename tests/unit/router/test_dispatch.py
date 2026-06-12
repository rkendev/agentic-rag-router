"""Unit tests for `Dispatcher` --- model tool call to adapter result.

Exercises every routing branch with the in-memory tool port fakes
(`tests/unit/tools/fakes.py`): each of the three tools on success and failure,
an unknown tool name, and a malformed call (missing key / empty value). No
substrate, no network.
"""

from __future__ import annotations

import json
from typing import Any

from agentic_rag_router.router.dispatch import ERROR_UNKNOWN_TOOL, Dispatcher
from agentic_rag_router.router.grading import GRADE_NONE, GRADE_SUFFICIENT, GRADE_WEAK
from agentic_rag_router.tools.envelope import (
    ERROR_VALIDATION,
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
)
from agentic_rag_router.tools.vector_search import DEFAULT_K
from agentic_rag_router.tools.web_search import DEFAULT_MAX_RESULTS
from tests.unit.tools.fakes import FakeEmbedder, FakeSqlExecutor, FakeVectorRepository


class FakeWebClient:
    """`WebSearchClient` returning preset rows; records query + max_results."""

    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self._rows = rows if rows is not None else []
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int) -> list[dict[str, object]]:
        self.calls.append((query, max_results))
        return self._rows


def _dispatcher(
    *,
    vector_rows: list[dict[str, object]] | None = None,
    sql_rows: list[dict[str, object]] | None = None,
    web_rows: list[dict[str, object]] | None = None,
) -> tuple[Dispatcher, FakeVectorRepository, FakeSqlExecutor, FakeWebClient]:
    repository = FakeVectorRepository(rows=vector_rows if vector_rows is not None else [])
    executor = FakeSqlExecutor(rows=sql_rows if sql_rows is not None else [])
    web_client = FakeWebClient(rows=web_rows if web_rows is not None else [])
    dispatcher = Dispatcher(
        embedder=FakeEmbedder(),
        repository=repository,
        executor=executor,
        web_client=web_client,
    )
    return dispatcher, repository, executor, web_client


def _payload(content: str) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(content)
    return parsed


def test_vector_search_success_cites_each_row() -> None:
    rows = [
        {"arxiv_id": "2401.1", "title": "A", "abstract": "...", "similarity": 0.9},
        {"arxiv_id": "2401.2", "title": "B", "abstract": "...", "similarity": 0.8},
    ]
    dispatcher, repository, _, _ = _dispatcher(vector_rows=rows)

    outcome = dispatcher.dispatch(TOOL_VECTOR_SEARCH, {"query": "attention", "k": 2})

    assert outcome.ok is True
    assert outcome.is_error is False
    assert outcome.error_code is None
    assert outcome.grade == GRADE_SUFFICIENT  # top-1 similarity 0.9 clears the floor
    assert repository.last_k == 2
    body = _payload(outcome.content)
    assert body["ok"] is True
    assert body["tool"] == TOOL_VECTOR_SEARCH
    assert len(body["data"]) == 2
    assert outcome.citations == [
        {"tool": TOOL_VECTOR_SEARCH, "source": "2401.1", "title": "A"},
        {"tool": TOOL_VECTOR_SEARCH, "source": "2401.2", "title": "B"},
    ]


def test_vector_search_defaults_k_when_absent() -> None:
    dispatcher, repository, _, _ = _dispatcher(vector_rows=[])

    dispatcher.dispatch(TOOL_VECTOR_SEARCH, {"query": "concept"})

    assert repository.last_k == DEFAULT_K


def test_vector_search_low_similarity_grades_weak() -> None:
    # A top hit below the similarity floor is real-but-not-strong evidence.
    rows = [{"arxiv_id": "2401.9", "title": "Z", "abstract": "...", "similarity": 0.20}]
    dispatcher, _, _, _ = _dispatcher(vector_rows=rows)

    outcome = dispatcher.dispatch(TOOL_VECTOR_SEARCH, {"query": "off topic", "k": 1})

    assert outcome.ok is True
    assert outcome.grade == GRADE_WEAK
    # The row is still cited at the dispatch layer; the loop drops it (not sufficient).
    assert outcome.citations == [{"tool": TOOL_VECTOR_SEARCH, "source": "2401.9", "title": "Z"}]


def test_vector_search_zero_rows_grades_none() -> None:
    dispatcher, _, _, _ = _dispatcher(vector_rows=[])

    outcome = dispatcher.dispatch(TOOL_VECTOR_SEARCH, {"query": "nothing", "k": 3})

    assert outcome.ok is True
    assert outcome.grade == GRADE_NONE
    assert outcome.citations == []


def test_sql_query_success_cites_table_once() -> None:
    dispatcher, _, executor, _ = _dispatcher(sql_rows=[{"n": 5}])

    outcome = dispatcher.dispatch(TOOL_SQL_QUERY, {"sql": "SELECT count(*) AS n FROM taxi_trips"})

    assert outcome.ok is True
    assert outcome.grade == GRADE_SUFFICIENT
    assert outcome.citations == [{"tool": TOOL_SQL_QUERY, "source": "taxi_trips"}]
    assert executor.executed == ["SELECT count(*) AS n FROM taxi_trips"]


def test_sql_query_success_with_no_rows_is_sufficient_but_uncited() -> None:
    # An empty aggregate still answers the question ("0 trips") -> sufficient,
    # but there is no row to cite.
    dispatcher, _, _, _ = _dispatcher(sql_rows=[])

    outcome = dispatcher.dispatch(TOOL_SQL_QUERY, {"sql": "SELECT 1 WHERE false"})

    assert outcome.ok is True
    assert outcome.grade == GRADE_SUFFICIENT
    assert outcome.citations == []


def test_sql_query_validation_failure_is_an_error_result() -> None:
    dispatcher, _, executor, _ = _dispatcher()

    outcome = dispatcher.dispatch(TOOL_SQL_QUERY, {"sql": "DELETE FROM taxi_trips WHERE false"})

    assert outcome.ok is False
    assert outcome.is_error is True
    assert outcome.error_code == ERROR_VALIDATION
    assert outcome.grade == GRADE_NONE
    body = _payload(outcome.content)
    assert body["ok"] is False
    assert body["error_code"] == ERROR_VALIDATION
    # The validator rejected it before the executor ran.
    assert executor.executed == []


def test_web_search_success_cites_each_url() -> None:
    rows: list[dict[str, object]] = [
        {"title": "T1", "url": "https://a", "snippet": "...", "published": None},
        {"title": "T2", "url": "https://b", "snippet": "...", "published": None},
    ]
    dispatcher, _, _, web_client = _dispatcher(web_rows=rows)

    outcome = dispatcher.dispatch(TOOL_WEB_SEARCH, {"query": "latest news", "max_results": 2})

    assert outcome.ok is True
    assert outcome.grade == GRADE_SUFFICIENT  # top result carries a URL
    assert web_client.calls == [("latest news", 2)]
    assert outcome.citations == [
        {"tool": TOOL_WEB_SEARCH, "source": "https://a", "title": "T1"},
        {"tool": TOOL_WEB_SEARCH, "source": "https://b", "title": "T2"},
    ]


def test_web_search_defaults_max_results_when_absent() -> None:
    dispatcher, _, _, web_client = _dispatcher(web_rows=[])

    dispatcher.dispatch(TOOL_WEB_SEARCH, {"query": "now"})

    assert web_client.calls == [("now", DEFAULT_MAX_RESULTS)]


def test_unknown_tool_name_is_an_error_result() -> None:
    dispatcher, _, _, _ = _dispatcher()

    outcome = dispatcher.dispatch("frobnicate", {"x": 1})

    assert outcome.ok is False
    assert outcome.is_error is True
    assert outcome.error_code == ERROR_UNKNOWN_TOOL
    assert outcome.grade == GRADE_NONE
    assert outcome.latency_ms == 0
    assert _payload(outcome.content)["error_code"] == ERROR_UNKNOWN_TOOL


def test_malformed_call_missing_key_is_validation_error() -> None:
    dispatcher, _, _, _ = _dispatcher()

    outcome = dispatcher.dispatch(TOOL_VECTOR_SEARCH, {})  # no "query"

    assert outcome.ok is False
    assert outcome.error_code == ERROR_VALIDATION
    assert outcome.grade == GRADE_NONE
    assert outcome.latency_ms == 0


def test_malformed_call_empty_value_is_validation_error() -> None:
    # vector_search raises ValueError on a whitespace-only query; the dispatcher
    # catches it and reports a validation error rather than letting it escape.
    dispatcher, _, _, _ = _dispatcher()

    outcome = dispatcher.dispatch(TOOL_VECTOR_SEARCH, {"query": "   "})

    assert outcome.ok is False
    assert outcome.error_code == ERROR_VALIDATION
    assert outcome.latency_ms == 0
