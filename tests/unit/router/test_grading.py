"""Unit tests for deterministic evidence grading (`router.grading`).

Drives `grade_result` directly with crafted `ToolResult` envelopes --- the rule
tables per tool, the threshold boundary, and the failure-always-none rule. No
substrate, no model.
"""

from __future__ import annotations

import pytest

from agentic_rag_router.router.grading import (
    GRADE_NONE,
    GRADE_SUFFICIENT,
    GRADE_WEAK,
    VECTOR_SUFFICIENCY_THRESHOLD,
    grade_result,
)
from agentic_rag_router.tools.envelope import (
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
    error_result,
    ok_result,
)

_VS = TOOL_VECTOR_SEARCH
_SQL = TOOL_SQL_QUERY
_WEB = TOOL_WEB_SEARCH


def _vector_rows(similarity: object) -> list[dict[str, object]]:
    return [{"arxiv_id": "2401.1", "title": "T", "abstract": "...", "similarity": similarity}]


# --- failure is always `none`, regardless of tool ---------------------------


@pytest.mark.parametrize("tool", [_VS, _SQL, _WEB])
def test_failed_result_is_always_none(tool: str) -> None:
    result = error_result(tool, "backend_error", "boom", latency_ms=1)
    assert grade_result(result) == GRADE_NONE


# --- vector_search: none / weak / sufficient on the similarity floor ---------


def test_vector_zero_rows_is_none() -> None:
    assert grade_result(ok_result(_VS, [], latency_ms=1)) == GRADE_NONE


def test_vector_top_similarity_at_threshold_is_sufficient() -> None:
    # The floor is inclusive (>=).
    rows = _vector_rows(VECTOR_SUFFICIENCY_THRESHOLD)
    assert grade_result(ok_result(_VS, rows, latency_ms=1)) == GRADE_SUFFICIENT


def test_vector_top_similarity_above_threshold_is_sufficient() -> None:
    rows = _vector_rows(VECTOR_SUFFICIENCY_THRESHOLD + 0.2)
    assert grade_result(ok_result(_VS, rows, latency_ms=1)) == GRADE_SUFFICIENT


def test_vector_top_similarity_below_threshold_is_weak() -> None:
    rows = _vector_rows(VECTOR_SUFFICIENCY_THRESHOLD - 0.01)
    assert grade_result(ok_result(_VS, rows, latency_ms=1)) == GRADE_WEAK


def test_vector_ranks_on_top_row_only() -> None:
    # Top row clears the floor even if later rows do not.
    rows = _vector_rows(VECTOR_SUFFICIENCY_THRESHOLD + 0.1)
    rows.append({"arxiv_id": "2401.2", "similarity": 0.01})
    assert grade_result(ok_result(_VS, rows, latency_ms=1)) == GRADE_SUFFICIENT


def test_vector_missing_or_nonnumeric_similarity_is_weak() -> None:
    assert grade_result(ok_result(_VS, [{"arxiv_id": "x"}], latency_ms=1)) == GRADE_WEAK
    assert grade_result(ok_result(_VS, _vector_rows(None), latency_ms=1)) == GRADE_WEAK
    assert grade_result(ok_result(_VS, _vector_rows("high"), latency_ms=1)) == GRADE_WEAK


# --- sql_query: executed == sufficient, even with zero rows ------------------


def test_sql_with_rows_is_sufficient() -> None:
    assert grade_result(ok_result(_SQL, [{"n": 5}], latency_ms=1)) == GRADE_SUFFICIENT


def test_sql_empty_aggregate_is_still_sufficient() -> None:
    # "0 trips" answers the question --- an executed empty result is evidence.
    assert grade_result(ok_result(_SQL, [], latency_ms=1)) == GRADE_SUFFICIENT


# --- web_search: none / weak / sufficient on URL presence -------------------


def test_web_zero_results_is_none() -> None:
    assert grade_result(ok_result(_WEB, [], latency_ms=1)) == GRADE_NONE


def test_web_top_result_with_url_is_sufficient() -> None:
    rows: list[dict[str, object]] = [
        {"title": "T", "url": "https://example.com/a", "snippet": "...", "published": None}
    ]
    assert grade_result(ok_result(_WEB, rows, latency_ms=1)) == GRADE_SUFFICIENT


def test_web_top_result_without_url_is_weak() -> None:
    for bad_url in (None, "", "   "):
        rows: list[dict[str, object]] = [{"title": "T", "url": bad_url, "snippet": "..."}]
        assert grade_result(ok_result(_WEB, rows, latency_ms=1)) == GRADE_WEAK
