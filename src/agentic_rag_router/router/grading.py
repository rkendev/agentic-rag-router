"""Deterministic evidence grading --- the rubric in code, never an LLM (D5).

Every tool result the router collects is graded into one of three buckets so the
loop can decide whether it has evidence good enough to answer on. Grading is
*deterministic code* by council-locked decision: an LLM-as-grader would double
latency, cost, and cassette surface and would itself need an eval. The grade is
computed purely from the `ToolResult` envelope (`tools/envelope.py`).

Three grades:

- ``sufficient`` --- evidence good enough to ground an answer and to cite.
- ``weak`` --- the tool returned something, but not strong enough to answer on;
  an answer resting on *only* weak/none evidence is converted to a refusal by
  the loop's grade-based backstop (`loop.py`).
- ``none`` --- the tool failed or returned nothing usable.

Per-tool rules (see the T005 report for the empirical probe behind the vector
threshold):

- ``vector_search``: ``none`` if the call failed or returned zero rows; else a
  ``sufficient``/``weak`` split on the top result's cosine similarity.
- ``sql_query``: ``none`` if the call failed (validation/backend error); else
  ``sufficient`` --- an executed aggregate is evidence even when it returns zero
  rows ("0 trips" answers the question). ``weak`` is unused for SQL.
- ``web_search``: ``none`` if the call failed or returned zero results; else
  ``sufficient`` when the top result carries a real source URL, else ``weak``.
  Web is the hardest substrate to grade deterministically, so this rule is
  deliberately lenient: the system-prompt *sentinel* (`schema.py`), not the
  grade, is the primary refusal mechanism for web near-misses --- a strict web
  grade would convert genuine web answers into over-refusals via the backstop.

Citations flow only from ``sufficient`` results (the loop enforces this); every
refusal carries zero citations (the rubric contract).
"""

from __future__ import annotations

from agentic_rag_router.tools.envelope import (
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
    ToolResult,
)

# Grade values. Plain string constants (not a StrEnum) to match the codebase's
# `TOOL_*` / `ERROR_*` style; `TrajectoryStep.grade` is typed `str`.
GRADE_SUFFICIENT = "sufficient"
GRADE_WEAK = "weak"
GRADE_NONE = "none"

# Cosine-similarity floor above which the top vector hit is `sufficient`.
#
# EMPIRICAL --- pinned against the frozen goldens via the live MiniLM embedder +
# pgvector substrate (T005 probe). Top-1 similarity observed:
#   answerable `vector_only` (G001-G014): min 0.4397, max 0.6491
#   `no_answer`               (G043-G054): min 0.2588, max 0.5184
# The two bands OVERLAP: the adversarial vector near-misses are topically
# on-corpus by design (G047 "transformer hyperparameters" 0.5184, G045
# "passenger satisfaction" 0.5078, G048 "GPU-hours" 0.4656), scoring as high as
# genuine concept questions --- so no similarity threshold can separate them.
# That separation is the sentinel's job, not grading's. The threshold's narrow
# role is to (a) keep genuine vector answers `sufficient` so they are not
# over-refused, and (b) grade clearly off-topic mis-routes (restaurant 0.259,
# meaning-of-life 0.277, crypto 0.289, haiku 0.348, S&P 0.356, medallion 0.298)
# as `weak` so the backstop fires. 0.40 sits just below the answerable floor
# (0.4397) to protect against over-refusal while still catching those.
VECTOR_SUFFICIENCY_THRESHOLD = 0.40


def _grade_vector(result: ToolResult) -> str:
    """`sufficient` if the top hit clears the similarity floor, else `weak`."""
    rows = result.data or []
    if not rows:
        return GRADE_NONE
    top_similarity = rows[0].get("similarity")
    if isinstance(top_similarity, int | float) and top_similarity >= VECTOR_SUFFICIENCY_THRESHOLD:
        return GRADE_SUFFICIENT
    # Below the floor, or a row without a usable similarity: real but not strong.
    return GRADE_WEAK


def _grade_web(result: ToolResult) -> str:
    """`sufficient` when the top result carries a source URL, else `weak`."""
    rows = result.data or []
    if not rows:
        return GRADE_NONE
    top_url = rows[0].get("url")
    if isinstance(top_url, str) and top_url.strip():
        return GRADE_SUFFICIENT
    return GRADE_WEAK


def grade_result(result: ToolResult) -> str:
    """Grade one tool result into ``sufficient`` / ``weak`` / ``none``.

    A failed result (``ok=False``) is always ``none`` regardless of tool. A
    successful result is graded per-tool: vector on similarity, web on URL
    presence, and SQL is ``sufficient`` whenever it executed (an empty aggregate
    still answers the question).
    """
    if not result.ok:
        return GRADE_NONE
    if result.tool == TOOL_VECTOR_SEARCH:
        return _grade_vector(result)
    if result.tool == TOOL_WEB_SEARCH:
        return _grade_web(result)
    # sql_query: executed successfully -> evidence, even with zero rows.
    return GRADE_SUFFICIENT
