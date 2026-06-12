"""Deterministic eval scoring --- pure functions over run rows + frozen goldens.

These functions are the measurement contract in code. They take rows derived
solely from `router.loop.RouterResponse` fields (no LLM-as-judge, no I/O) and
produce the metrics the D6 gate asserts on:

- ``routing_accuracy`` --- first-tool correctness over the 48 *answerable*
  goldens (a question's first tool must be in its ``acceptable_tools``; for a
  ``hybrid`` that means any listed tool). Reported with a per-class breakdown.
- ``refusal_correctness`` --- the 12 ``no_answer`` goldens, each of which must be
  *refused* with zero citations and no answer. A refusal counts only when it
  fires at an evidence layer (`EVIDENCE_REFUSAL_REASONS`); the layer attribution
  (sentinel vs backstop) is reported alongside.
- ``over_refusals`` --- answerable goldens that refused. Rubric §5.2's *separate*
  error class (opposite fix to a misroute), counted and listed, never folded into
  routing accuracy.
- ``citation_coverage`` --- answered goldens carrying >= 1 citation, over answered
  goldens. A proxy (citations derive from ``sufficient``-graded evidence), not
  faithfulness; the report says so.

`confusion` builds the per-class outcome table (rows = true label, columns = the
three tools plus ``refuse`` / ``none``). `naive_baselines` scores the three
constant single-tool policies against the labels with no API calls, to
contextualize the routing gate for an outside reader.

The refusal-reason constants are imported from `router.loop` (not re-spelled) so
the metric's definition of "an evidence-based refusal" can never drift from the
router that produces them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentic_rag_router.router.loop import (
    EVIDENCE_REFUSAL_REASONS,
    REFUSAL_BACKSTOP,
    REFUSAL_ITERATION_BUDGET,
    REFUSAL_SENTINEL,
)
from agentic_rag_router.tools.envelope import (
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
)

# Golden labels. The four answerable classes are scored for routing accuracy;
# `no_answer` is scored for refusal correctness instead.
LABEL_VECTOR_ONLY = "vector_only"
LABEL_SQL_ONLY = "sql_only"
LABEL_WEB_ONLY = "web_only"
LABEL_HYBRID = "hybrid"
LABEL_NO_ANSWER = "no_answer"
ANSWERABLE_LABELS: tuple[str, ...] = (
    LABEL_VECTOR_ONLY,
    LABEL_SQL_ONLY,
    LABEL_WEB_ONLY,
    LABEL_HYBRID,
)

# Confusion-table route columns: the three tools, plus a refusal column and a
# `none` catch-all for a run that never chose a tool.
ROUTE_REFUSE = "refuse"
ROUTE_NONE = "none"
ROUTE_COLUMNS: tuple[str, ...] = (
    TOOL_VECTOR_SEARCH,
    TOOL_SQL_QUERY,
    TOOL_WEB_SEARCH,
    ROUTE_REFUSE,
    ROUTE_NONE,
)

# The three constant single-tool baseline policies (one per substrate).
BASELINE_POLICIES: tuple[str, ...] = (
    TOOL_VECTOR_SEARCH,
    TOOL_SQL_QUERY,
    TOOL_WEB_SEARCH,
)

# Refusal-reason -> short layer label, for attribution counts. Sentinel and
# backstop are the two evidence layers; iter_budget is the loop-cap fallback.
REFUSAL_LAYER: dict[str, str] = {
    REFUSAL_SENTINEL: "sentinel",
    REFUSAL_BACKSTOP: "backstop",
    REFUSAL_ITERATION_BUDGET: "iter_budget",
}


@dataclass(frozen=True, slots=True)
class Golden:
    """The frozen facts about one golden question (no run outcome).

    Only ``label`` and ``acceptable_tools`` are read by `naive_baselines`; ``id``
    is carried for traceability.
    """

    id: str
    label: str
    acceptable_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalRow:
    """One golden's run outcome, reduced to the fields scoring needs.

    Built by the live runner from a `RouterResponse`:

    - ``first_tool`` --- the first tool the router invoked (``None`` if it never
      called one).
    - ``refusal_reason`` --- the `RouterResponse.refusal_reason` (``None`` on a
      normal answer).
    - ``citation_count`` --- ``len(response.citations)``.
    - ``answer_is_none`` --- ``response.answer is None``.
    """

    id: str
    label: str
    acceptable_tools: tuple[str, ...]
    first_tool: str | None
    refusal_reason: str | None
    citation_count: int
    answer_is_none: bool


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """``numerator / denominator`` rounded, or ``None`` when there is nothing to divide."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def derive_refused(row: EvalRow) -> bool:
    """True when ``row`` is a clean, evidence-based refusal.

    The rubric contract for a correct ``no_answer`` outcome: the router declined
    at an evidence layer (sentinel or backstop --- `EVIDENCE_REFUSAL_REASONS`),
    returned no answer, and leaked zero citations. An iteration-budget refusal is
    deliberately *not* counted: it is a loop-cap fallback, not a grounded "I have
    no supporting evidence" decision.
    """
    return (
        row.refusal_reason in EVIDENCE_REFUSAL_REASONS
        and row.citation_count == 0
        and row.answer_is_none
    )


def _route_of(row: EvalRow) -> str:
    """The confusion-table column a row lands in: ``refuse`` / a tool / ``none``."""
    if row.refusal_reason is not None:
        return ROUTE_REFUSE
    if row.first_tool is not None:
        return row.first_tool
    return ROUTE_NONE


def confusion(rows: Sequence[EvalRow]) -> dict[str, dict[str, int]]:
    """Per-class outcome table: ``table[label][route] = count``.

    Rows are the five golden labels; columns are `ROUTE_COLUMNS`. Each golden is
    placed by its *outcome* --- a refusal lands in ``refuse`` regardless of which
    tool it tried first, so an over-refused answerable golden is visible as a
    count off the tool columns. A tool name a row reports that is not one of the
    three known tools (should not happen) still gets its own column key.
    """
    labels = (*ANSWERABLE_LABELS, LABEL_NO_ANSWER)
    table: dict[str, dict[str, int]] = {label: dict.fromkeys(ROUTE_COLUMNS, 0) for label in labels}
    for row in rows:
        bucket = table.setdefault(row.label, dict.fromkeys(ROUTE_COLUMNS, 0))
        route = _route_of(row)
        bucket[route] = bucket.get(route, 0) + 1
    return table


def naive_baselines(goldens: Sequence[Golden]) -> dict[str, Any]:
    """Score the three constant single-tool policies against the frozen labels.

    Each policy ("always call tool X") is scored over the *answerable* goldens
    only --- the same denominator as ``routing_accuracy`` --- so the numbers are a
    like-for-like floor for the routing gate. A constant policy can never refuse,
    so including ``no_answer`` would only depress every policy uniformly and break
    the comparison. The best policy is reported as "the naive baseline".
    """
    answerable = [g for g in goldens if g.label in ANSWERABLE_LABELS]
    total = len(answerable)
    policies: dict[str, dict[str, Any]] = {}
    for tool in BASELINE_POLICIES:
        correct = sum(1 for g in answerable if tool in g.acceptable_tools)
        policies[tool] = {
            "correct": correct,
            "total": total,
            "accuracy": _safe_ratio(correct, total),
        }

    # `policies` always has one entry per BASELINE_POLICIES, so there is always a
    # best; with no answerable goldens every policy ties at 0 and `max` returns
    # the first by insertion order (accuracy then `None`).
    best_tool = max(policies, key=lambda t: policies[t]["correct"])
    best = {"policy": best_tool, "accuracy": policies[best_tool]["accuracy"]}
    return {"denominator": total, "policies": policies, "best": best}


def _routing(rows: Sequence[EvalRow]) -> dict[str, Any]:
    """First-tool routing accuracy over the answerable goldens, with a per-class split."""
    per_class: dict[str, dict[str, Any]] = {}
    correct_total = 0
    scored_total = 0
    for label in ANSWERABLE_LABELS:
        items = [r for r in rows if r.label == label]
        correct = sum(1 for r in items if r.first_tool in r.acceptable_tools)
        correct_total += correct
        scored_total += len(items)
        per_class[label] = {
            "correct": correct,
            "total": len(items),
            "accuracy": _safe_ratio(correct, len(items)),
        }
    accuracy = _safe_ratio(correct_total, scored_total)
    return {
        "routing_accuracy": 0.0 if accuracy is None else accuracy,
        "routing_correct": correct_total,
        "routing_total": scored_total,
        "per_class": per_class,
    }


def _refusal(rows: Sequence[EvalRow]) -> dict[str, Any]:
    """Refusal correctness over the no_answer goldens, with layer attribution."""
    no_answer = [r for r in rows if r.label == LABEL_NO_ANSWER]
    refused = [r for r in no_answer if derive_refused(r)]
    attribution: dict[str, int] = {"sentinel": 0, "backstop": 0, "iter_budget": 0, "other": 0}
    for row in no_answer:
        if row.refusal_reason is None:
            continue
        attribution[REFUSAL_LAYER.get(row.refusal_reason, "other")] += 1
    correctness = _safe_ratio(len(refused), len(no_answer))
    return {
        "refusal_correctness": 0.0 if correctness is None else correctness,
        "refused": len(refused),
        "no_answer_total": len(no_answer),
        "refusal_attribution": attribution,
    }


def _over_refusals(rows: Sequence[EvalRow]) -> dict[str, Any]:
    """Answerable goldens that refused --- rubric §5.2's separate error class."""
    ids = [r.id for r in rows if r.label in ANSWERABLE_LABELS and r.refusal_reason is not None]
    return {"over_refusals": len(ids), "over_refusal_ids": ids}


def _citation_coverage(rows: Sequence[EvalRow]) -> dict[str, Any]:
    """Answered goldens carrying >= 1 citation, over answered goldens (a proxy)."""
    answered = [r for r in rows if r.refusal_reason is None and not r.answer_is_none]
    with_citations = sum(1 for r in answered if r.citation_count >= 1)
    return {
        "citation_coverage": _safe_ratio(with_citations, len(answered)),
        "answered_with_citations": with_citations,
        "answered": len(answered),
    }


def score_run(rows: Sequence[EvalRow]) -> dict[str, Any]:
    """Aggregate every metric for a run into one JSON-ready dict.

    Combines first-tool routing accuracy (+ per-class), refusal correctness (+
    sentinel/backstop attribution), the separate over-refusal count and id list,
    and citation coverage. All values are plain ints / floats / lists / dicts so
    the result serializes straight into ``report.json``.
    """
    return {
        **_routing(rows),
        **_refusal(rows),
        **_over_refusals(rows),
        **_citation_coverage(rows),
    }
