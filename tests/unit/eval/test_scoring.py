"""Unit tests for the pure eval scoring functions.

Drives every branch of `agentic_rag_router.eval.scoring` with synthetic rows ---
no API, no DB, no router. The scoring contract (what counts as a correct route, a
clean refusal, an over-refusal, citation coverage, and a baseline) is exercised
directly so the D6 gate rests on tested arithmetic.
"""

from __future__ import annotations

from agentic_rag_router.eval.scoring import (
    EvalRow,
    Golden,
    confusion,
    derive_refused,
    naive_baselines,
    score_run,
)
from agentic_rag_router.router.loop import (
    REFUSAL_BACKSTOP,
    REFUSAL_ITERATION_BUDGET,
    REFUSAL_SENTINEL,
)
from agentic_rag_router.tools.envelope import (
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
)


def _answer_row(
    row_id: str,
    label: str,
    acceptable: tuple[str, ...],
    first_tool: str | None,
    *,
    citations: int = 1,
) -> EvalRow:
    """An answered (non-refused) row."""
    return EvalRow(
        id=row_id,
        label=label,
        acceptable_tools=acceptable,
        first_tool=first_tool,
        refusal_reason=None,
        citation_count=citations,
        answer_is_none=False,
    )


def _refuse_row(row_id: str, label: str, reason: str | None, *, first_tool: str | None) -> EvalRow:
    """A refused row (no answer, zero citations) with the given reason/layer."""
    return EvalRow(
        id=row_id,
        label=label,
        acceptable_tools=(),
        first_tool=first_tool,
        refusal_reason=reason,
        citation_count=0,
        answer_is_none=True,
    )


# --- derive_refused -------------------------------------------------------


def test_derive_refused_true_for_clean_evidence_refusal() -> None:
    row = _refuse_row("G043", "no_answer", REFUSAL_SENTINEL, first_tool=TOOL_SQL_QUERY)
    assert derive_refused(row) is True


def test_derive_refused_false_for_iteration_budget() -> None:
    # iter_budget is a loop-cap fallback, not an evidence-based refusal.
    row = _refuse_row("G044", "no_answer", REFUSAL_ITERATION_BUDGET, first_tool=TOOL_SQL_QUERY)
    assert derive_refused(row) is False


def test_derive_refused_false_when_citations_leaked() -> None:
    row = EvalRow("G045", "no_answer", (), TOOL_WEB_SEARCH, REFUSAL_BACKSTOP, 2, True)
    assert derive_refused(row) is False


def test_derive_refused_false_when_answer_present() -> None:
    row = EvalRow("G046", "no_answer", (), TOOL_SQL_QUERY, REFUSAL_SENTINEL, 0, False)
    assert derive_refused(row) is False


# --- routing accuracy -----------------------------------------------------


def test_routing_accuracy_perfect() -> None:
    rows = [
        _answer_row("G001", "vector_only", (TOOL_VECTOR_SEARCH,), TOOL_VECTOR_SEARCH),
        _answer_row("G015", "sql_only", (TOOL_SQL_QUERY,), TOOL_SQL_QUERY),
        _answer_row("G029", "web_only", (TOOL_WEB_SEARCH,), TOOL_WEB_SEARCH),
        _answer_row("G055", "hybrid", (TOOL_SQL_QUERY, TOOL_WEB_SEARCH), TOOL_WEB_SEARCH),
    ]
    result = score_run(rows)
    assert result["routing_accuracy"] == 1.0
    assert result["routing_correct"] == 4
    assert result["routing_total"] == 4
    assert result["per_class"]["hybrid"] == {"correct": 1, "total": 1, "accuracy": 1.0}


def test_routing_accuracy_counts_misroute_and_missing_tool() -> None:
    rows = [
        _answer_row("G001", "vector_only", (TOOL_VECTOR_SEARCH,), TOOL_SQL_QUERY),  # misroute
        _answer_row("G002", "vector_only", (TOOL_VECTOR_SEARCH,), None),  # no tool chosen
        _answer_row("G015", "sql_only", (TOOL_SQL_QUERY,), TOOL_SQL_QUERY),  # correct
    ]
    result = score_run(rows)
    assert result["routing_correct"] == 1
    assert result["routing_total"] == 3
    assert result["routing_accuracy"] == round(1 / 3, 6)
    assert result["per_class"]["vector_only"] == {"correct": 0, "total": 2, "accuracy": 0.0}
    # A class with no rows reports accuracy None (nothing to divide).
    assert result["per_class"]["web_only"] == {"correct": 0, "total": 0, "accuracy": None}


def test_empty_rows_yield_zero_routing_and_refusal() -> None:
    result = score_run([])
    assert result["routing_accuracy"] == 0.0
    assert result["refusal_correctness"] == 0.0
    assert result["citation_coverage"] is None
    assert result["over_refusals"] == 0


# --- refusal correctness + attribution ------------------------------------


def test_refusal_correctness_perfect_with_attribution() -> None:
    rows = [
        _refuse_row("G043", "no_answer", REFUSAL_SENTINEL, first_tool=TOOL_SQL_QUERY),
        _refuse_row("G044", "no_answer", REFUSAL_SENTINEL, first_tool=TOOL_SQL_QUERY),
        _refuse_row("G047", "no_answer", REFUSAL_BACKSTOP, first_tool=TOOL_VECTOR_SEARCH),
    ]
    result = score_run(rows)
    assert result["refusal_correctness"] == 1.0
    assert result["refused"] == 3
    assert result["no_answer_total"] == 3
    assert result["refusal_attribution"] == {
        "sentinel": 2,
        "backstop": 1,
        "iter_budget": 0,
        "other": 0,
    }


def test_refusal_attribution_counts_iter_budget_other_and_non_refusal() -> None:
    rows = [
        _refuse_row("G049", "no_answer", REFUSAL_ITERATION_BUDGET, first_tool=TOOL_WEB_SEARCH),
        _refuse_row("G050", "no_answer", "some_unmapped_reason", first_tool=TOOL_WEB_SEARCH),
        # A no_answer that leaked an answer entirely (refusal_reason None) ->
        # not refused, and skipped by attribution.
        _answer_row("G051", "no_answer", (), TOOL_WEB_SEARCH, citations=0),
    ]
    result = score_run(rows)
    assert result["refused"] == 0  # iter_budget + unmapped are not evidence refusals
    assert result["refusal_correctness"] == 0.0
    assert result["refusal_attribution"] == {
        "sentinel": 0,
        "backstop": 0,
        "iter_budget": 1,
        "other": 1,
    }


# --- over-refusals --------------------------------------------------------


def test_over_refusals_listed_individually() -> None:
    rows = [
        _refuse_row("G003", "vector_only", REFUSAL_BACKSTOP, first_tool=TOOL_VECTOR_SEARCH),
        _answer_row("G004", "vector_only", (TOOL_VECTOR_SEARCH,), TOOL_VECTOR_SEARCH),
        _refuse_row("G043", "no_answer", REFUSAL_SENTINEL, first_tool=TOOL_SQL_QUERY),  # not over
    ]
    result = score_run(rows)
    assert result["over_refusals"] == 1
    assert result["over_refusal_ids"] == ["G003"]


# --- citation coverage ----------------------------------------------------


def test_citation_coverage_proxy() -> None:
    rows = [
        _answer_row("G001", "vector_only", (TOOL_VECTOR_SEARCH,), TOOL_VECTOR_SEARCH, citations=2),
        _answer_row("G015", "sql_only", (TOOL_SQL_QUERY,), TOOL_SQL_QUERY, citations=0),
        _refuse_row("G043", "no_answer", REFUSAL_SENTINEL, first_tool=TOOL_SQL_QUERY),
    ]
    result = score_run(rows)
    assert result["answered"] == 2
    assert result["answered_with_citations"] == 1
    assert result["citation_coverage"] == 0.5


# --- confusion table ------------------------------------------------------


def test_confusion_places_rows_by_outcome() -> None:
    rows = [
        _answer_row("G001", "vector_only", (TOOL_VECTOR_SEARCH,), TOOL_VECTOR_SEARCH),
        _refuse_row("G043", "no_answer", REFUSAL_SENTINEL, first_tool=TOOL_SQL_QUERY),
        # answered but never chose a tool -> `none` column
        _answer_row("G002", "vector_only", (TOOL_VECTOR_SEARCH,), None),
    ]
    table = confusion(rows)
    assert table["vector_only"][TOOL_VECTOR_SEARCH] == 1
    assert table["vector_only"]["none"] == 1
    assert table["no_answer"]["refuse"] == 1
    # Every known label has a full column set even when unobserved.
    assert set(table["sql_only"]) >= {TOOL_VECTOR_SEARCH, "refuse", "none"}


def test_confusion_handles_unknown_label_and_tool() -> None:
    rows = [_answer_row("X1", "surprise_label", (), "mystery_tool")]
    table = confusion(rows)
    assert table["surprise_label"]["mystery_tool"] == 1


# --- naive baselines ------------------------------------------------------


def test_naive_baselines_over_answerable_only() -> None:
    goldens = [
        Golden("G001", "vector_only", (TOOL_VECTOR_SEARCH,)),
        Golden("G015", "sql_only", (TOOL_SQL_QUERY,)),
        Golden("G055", "hybrid", (TOOL_SQL_QUERY, TOOL_WEB_SEARCH)),
        Golden("G043", "no_answer", ()),  # excluded from the denominator
    ]
    result = naive_baselines(goldens)
    assert result["denominator"] == 3
    assert result["policies"][TOOL_SQL_QUERY]["correct"] == 2  # sql_only + hybrid
    assert result["policies"][TOOL_VECTOR_SEARCH]["correct"] == 1
    assert result["policies"][TOOL_WEB_SEARCH]["correct"] == 1
    assert result["best"]["policy"] == TOOL_SQL_QUERY
    assert result["best"]["accuracy"] == round(2 / 3, 6)


def test_naive_baselines_empty_goldens() -> None:
    result = naive_baselines([])
    assert result["denominator"] == 0
    assert result["policies"][TOOL_VECTOR_SEARCH]["accuracy"] is None
    # No answerable goldens: every policy ties at 0, best falls to the first.
    assert result["best"]["policy"] == TOOL_VECTOR_SEARCH
    assert result["best"]["accuracy"] is None
