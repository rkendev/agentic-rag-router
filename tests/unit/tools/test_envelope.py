"""Unit tests for the shared `ToolResult` envelope and its builders."""

from __future__ import annotations

import dataclasses

import pytest

from agentic_rag_router.tools.envelope import (
    ERROR_VALIDATION,
    TOOL_SQL_QUERY,
    ToolResult,
    error_result,
    ok_result,
)


def test_ok_result_fields() -> None:
    rows: list[dict[str, object]] = [{"a": 1}]
    result = ok_result(TOOL_SQL_QUERY, rows, latency_ms=12)
    assert result.ok is True
    assert result.tool == TOOL_SQL_QUERY
    assert result.data == rows
    assert result.error_code is None
    assert result.error_message is None
    assert result.latency_ms == 12


def test_error_result_fields() -> None:
    result = error_result(TOOL_SQL_QUERY, ERROR_VALIDATION, "bad sql", latency_ms=3)
    assert result.ok is False
    assert result.tool == TOOL_SQL_QUERY
    assert result.data is None
    assert result.error_code == ERROR_VALIDATION
    assert result.error_message == "bad sql"
    assert result.latency_ms == 3


def test_tool_result_is_frozen() -> None:
    result = ok_result(TOOL_SQL_QUERY, [], latency_ms=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ok = False  # type: ignore[misc]


def test_tool_result_has_no_evidence_field() -> None:
    # The evidence-grade field is D5; it must not be stubbed here.
    field_names = {f.name for f in dataclasses.fields(ToolResult)}
    assert field_names == {
        "ok",
        "tool",
        "data",
        "error_code",
        "error_message",
        "latency_ms",
    }
