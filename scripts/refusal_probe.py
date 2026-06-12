"""Refusal probe --- live D5 validation of the refusal path against the goldens.

Runs the REAL router (live Sonnet + real pgvector / ``router_ro`` / Tavily
substrates) over two slices of the frozen golden set and prints two tables:

* all 12 ``no_answer`` goldens --- target 12/12 refused with zero leaked
  citations; and
* 6 answerable spot-checks (2 ``vector_only`` / 2 ``sql_only`` / 2 ``web_only``,
  read verbatim) --- target 0 over-refusals, answers carrying citations.

Every row carries the full trace, not just a yes/no: the refusal **layer**
(``sentinel`` / ``backstop`` / ``iter_budget``, or ``LEAK`` / ``over-refusal``)
and the **per-step grade trace** (``tool(ok=.. -> grade)`` chain). This makes the
sql-error -> web-fallback -> sufficient-junk escape visible by construction: a
leaked taxi near-miss shows its SQL ``none`` step, the web ``sufficient`` step,
and the citations it carried.

Frozen fidelity: questions are read VERBATIM from
``data/eval/golden_questions.jsonl``; the script prints the file's sha256 + line
count so a run is traceable to the exact frozen set. This is a dev probe, NOT the
D6 gate.

Usage (source ``.env`` so ``TAVILY_API_KEY`` reaches ``os.environ``)::

    set -a && . ./.env && set +a && uv run python scripts/refusal_probe.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.router.client import AnthropicRouterClient
from agentic_rag_router.router.dispatch import Dispatcher
from agentic_rag_router.router.loop import (
    REFUSAL_BACKSTOP,
    REFUSAL_ITERATION_BUDGET,
    REFUSAL_SENTINEL,
    RouterResponse,
    run_router,
)
from agentic_rag_router.router.schema import TOOLS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDENS = _REPO_ROOT / "data" / "eval" / "golden_questions.jsonl"

# How many answerable spot-checks per single-tool class.
_PER_CLASS = 2
_ANSWERABLE_CLASSES = ("vector_only", "sql_only", "web_only")

# refusal_reason -> short layer label for the tables.
_LAYER = {
    REFUSAL_SENTINEL: "sentinel",
    REFUSAL_BACKSTOP: "backstop",
    REFUSAL_ITERATION_BUDGET: "iter_budget",
}


@dataclass(frozen=True)
class Golden:
    """One frozen golden question, read verbatim."""

    id: str
    question: str
    label: str
    adversarial: bool


def load_goldens() -> list[Golden]:
    """Read the frozen golden set verbatim (no paraphrasing)."""
    goldens: list[Golden] = []
    for line in _GOLDENS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        goldens.append(
            Golden(
                id=record["id"],
                question=record["question"],
                label=record["label"],
                adversarial=bool(record.get("adversarial", False)),
            )
        )
    return goldens


def frozen_fingerprint() -> tuple[str, int]:
    """Return ``(sha256, line_count)`` of the frozen goldens file."""
    raw = _GOLDENS.read_bytes()
    line_count = len([line for line in raw.decode("utf-8").splitlines() if line.strip()])
    return hashlib.sha256(raw).hexdigest(), line_count


def _trace(result: RouterResponse) -> str:
    """Render the per-step grade trace, e.g. ``sql_query(ok=F,backend_error->none)``."""
    if not result.trajectory:
        return "(no tool calls)"
    parts = []
    for step in result.trajectory:
        ok = "T" if step.ok else "F"
        err = f",{step.error_code}" if step.error_code else ""
        parts.append(f"{step.tool}(ok={ok}{err}->{step.grade})")
    return " -> ".join(parts)


def _layer(result: RouterResponse) -> str:
    """The refusal layer label, or ``ANSWER`` when the router answered."""
    if result.refusal_reason is None:
        return "ANSWER"
    return _LAYER.get(result.refusal_reason, result.refusal_reason)


def build_dispatcher() -> Dispatcher:
    """Real Dispatcher over pgvector + ``router_ro`` + live Tavily."""
    from agentic_rag_router.tools.sql_query import RouterRoExecutor
    from agentic_rag_router.tools.vector_search import (
        PgVectorRepository,
        SentenceTransformerEmbedder,
    )

    return Dispatcher(
        embedder=SentenceTransformerEmbedder(),
        repository=PgVectorRepository(),
        executor=RouterRoExecutor(),
    )


def select_answerable(goldens: list[Golden]) -> list[Golden]:
    """The first ``_PER_CLASS`` non-adversarial goldens of each answerable class."""
    chosen: list[Golden] = []
    for label in _ANSWERABLE_CLASSES:
        items = [g for g in goldens if g.label == label and not g.adversarial][:_PER_CLASS]
        chosen.extend(items)
    return chosen


def run_no_answer(
    client: AnthropicRouterClient, dispatcher: Dispatcher, goldens: list[Golden]
) -> int:
    """Run the 12 no_answer goldens; print the table; return refused count."""
    no_answer = [g for g in goldens if g.label == "no_answer"]
    print(f"NO_ANSWER  ({len(no_answer)} goldens; target: all refused, zero leaked citations)")
    print(f"{'id':<6}{'outcome':<10}{'layer':<12}{'cites':<7}question / trace")
    refused = 0
    leaked_cites = 0
    for g in no_answer:
        result = run_router(g.question, client=client, tools=TOOLS, dispatcher=dispatcher)
        is_refused = result.refusal_reason is not None and result.answer is None
        outcome = "REFUSED" if is_refused else "LEAK"
        refused += int(is_refused)
        leaked_cites += int(bool(result.citations))
        print(f"{g.id:<6}{outcome:<10}{_layer(result):<12}{len(result.citations):<7}{g.question}")
        print(f"{'':<6}trace: {_trace(result)}")
        if result.citations:
            print(f"{'':<6}LEAKED CITATIONS: {result.citations}")
    print(f"-> refused {refused}/{len(no_answer)} ; rows with leaked citations: {leaked_cites}\n")
    return refused


def run_answerable(
    client: AnthropicRouterClient, dispatcher: Dispatcher, goldens: list[Golden]
) -> int:
    """Run the 6 answerable spot-checks; print the table; return over-refusal count."""
    answerable = select_answerable(goldens)
    print(f"ANSWERABLE SPOT-CHECKS  ({len(answerable)}; target: 0 over-refusals, cites present)")
    print(f"{'id':<6}{'label':<13}{'outcome':<12}{'cites':<7}question / trace")
    over_refusals = 0
    for g in answerable:
        result = run_router(g.question, client=client, tools=TOOLS, dispatcher=dispatcher)
        over = result.refusal_reason is not None
        over_refusals += int(over)
        outcome = f"OVER-REFUSE({_layer(result)})" if over else "ANSWERED"
        print(f"{g.id:<6}{g.label:<13}{outcome:<12}{len(result.citations):<7}{g.question}")
        print(f"{'':<6}trace: {_trace(result)}")
    print(f"-> over-refusals {over_refusals}/{len(answerable)}\n")
    return over_refusals


def main() -> None:
    """Run both slices and print the validation tables."""
    goldens = load_goldens()
    sha, line_count = frozen_fingerprint()

    client = AnthropicRouterClient.from_settings(Settings())
    dispatcher = build_dispatcher()

    print("Refusal probe --- live router over the frozen goldens (D5 validation)")
    print(f"file:   {_GOLDENS.relative_to(_REPO_ROOT)}")
    print(f"sha256: {sha}")
    print(f"lines:  {line_count}")
    print(f"model:  {client._model}\n")

    refused = run_no_answer(client, dispatcher, goldens)
    over_refusals = run_answerable(client, dispatcher, goldens)

    na_total = len([g for g in goldens if g.label == "no_answer"])
    ans_total = _PER_CLASS * len(_ANSWERABLE_CLASSES)
    print("SUMMARY")
    print(f"  no_answer refusals: {refused}/{na_total}  (target {na_total}/{na_total})")
    print(f"  over-refusals:      {over_refusals}/{ans_total}  (target 0)")


if __name__ == "__main__":
    main()
