"""Unit tests for the routing-contract tool schema.

The descriptions are the router, so these tests pin the structural invariants
(three tools, correct names, required inputs) and the routing anchors each
description must carry (the corpus cutoff in the vector/web contracts, the taxi
columns in the SQL contract, the anti-fabrication instruction in the system
prompt). They do not pin the prose --- that is tuned against the probe.
"""

from __future__ import annotations

from typing import Any

from agentic_rag_router.router.schema import (
    CORPUS_CUTOFF,
    SYSTEM_PROMPT,
    TAXI_SCHEMA_BLOCK,
    TOOLS,
    build_tools,
)
from agentic_rag_router.tools.envelope import (
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
)


def _by_name() -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for tool in TOOLS}


def test_corpus_cutoff_matches_data_sources() -> None:
    # The contract value documented in docs/DATA_SOURCES.md.
    assert CORPUS_CUTOFF == "2026-06-11"


def test_exactly_three_tools_with_expected_names() -> None:
    assert [tool["name"] for tool in TOOLS] == [
        TOOL_VECTOR_SEARCH,
        TOOL_SQL_QUERY,
        TOOL_WEB_SEARCH,
    ]


def test_build_tools_returns_independent_copies() -> None:
    first = build_tools()
    second = build_tools()
    assert first == second
    assert first is not second


def test_input_schemas_declare_required_fields() -> None:
    tools = _by_name()
    assert tools[TOOL_VECTOR_SEARCH]["input_schema"]["required"] == ["query"]
    assert "k" in tools[TOOL_VECTOR_SEARCH]["input_schema"]["properties"]
    assert tools[TOOL_SQL_QUERY]["input_schema"]["required"] == ["sql"]
    assert tools[TOOL_WEB_SEARCH]["input_schema"]["required"] == ["query"]
    assert "max_results" in tools[TOOL_WEB_SEARCH]["input_schema"]["properties"]


def test_vector_and_web_descriptions_anchor_on_the_cutoff() -> None:
    tools = _by_name()
    assert CORPUS_CUTOFF in tools[TOOL_VECTOR_SEARCH]["description"]
    assert CORPUS_CUTOFF in tools[TOOL_WEB_SEARCH]["description"]


def test_sql_description_embeds_the_taxi_schema() -> None:
    description = _by_name()[TOOL_SQL_QUERY]["description"]
    assert TAXI_SCHEMA_BLOCK in description
    # A couple of representative columns the model needs to author SQL.
    assert "payment_type" in description
    assert "trip_distance" in description
    assert "taxi_trips" in description


def test_system_prompt_biases_against_fabrication() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "prior knowledge" in lowered or "memory" in lowered
    assert "cannot answer" in lowered or "do not answer" in lowered
    # Names all three tools so the model knows the routing surface.
    for name in (TOOL_VECTOR_SEARCH, TOOL_SQL_QUERY, TOOL_WEB_SEARCH):
        assert name in SYSTEM_PROMPT
