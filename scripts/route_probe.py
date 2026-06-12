"""Routing probe --- a dev tool for tuning tool DESCRIPTIONS (NOT the D6 gate).

Runs the 60 frozen golden questions through a single *forced* tool call (the
router's iteration-0 decision: ``tool_choice={"type":"any"}``) and prints a
per-class confusion table plus the overall first-tool routing accuracy. Use it
to tune the descriptions in ``agentic_rag_router.router.schema`` against the
frozen eval set, then re-run to measure the delta.

Why forced mode: ``tool_choice`` any mirrors the loop's iteration 0 and isolates
the routing decision (which tool) from the answer-vs-tool decision. A forced
call always picks a tool, so ``no_answer`` questions can never be "correct" here
(refusal is impossible under forced choice and is D5's concern) --- they are
reported as a separate *leakage* row and excluded from the accuracy denominator.

Frozen-file fidelity: questions are read VERBATIM from
``data/eval/golden_questions.jsonl`` (never paraphrased); the script prints the
file's sha256 + line count so a run is traceable to the exact frozen set.

This is a routing-only probe: it makes one model call per question and reads the
chosen tool name. It never executes a tool, so it needs no database or web key
--- only ``ANTHROPIC_API_KEY``. Cost: ~60 live calls per pass, max_tokens kept
small.

Usage::

    uv run python scripts/route_probe.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.router.client import AnthropicRouterClient
from agentic_rag_router.router.schema import TOOLS
from agentic_rag_router.tools.envelope import (
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDENS = _REPO_ROOT / "data" / "eval" / "golden_questions.jsonl"

# Columns of the confusion table (the three tools plus a "none" catch-all for a
# forced call that somehow returned no tool_use block).
_NONE = "none"
_TOOL_COLUMNS = [TOOL_VECTOR_SEARCH, TOOL_SQL_QUERY, TOOL_WEB_SEARCH, _NONE]

# Labels scored for first-tool accuracy (no_answer is reported but not scored).
_ANSWERABLE_LABELS = ["vector_only", "sql_only", "web_only", "hybrid"]
_ALL_LABELS = [*_ANSWERABLE_LABELS, "no_answer"]

# Routing-only probe: tiny output ceiling keeps the forced call cheap.
_PROBE_MAX_TOKENS = 256
_FORCE_ANY: dict[str, str] = {"type": "any"}


@dataclass(frozen=True)
class Golden:
    """One frozen golden question, read verbatim from the JSONL."""

    id: str
    question: str
    label: str
    acceptable_tools: list[str]


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
                acceptable_tools=list(record["acceptable_tools"]),
            )
        )
    return goldens


def frozen_fingerprint() -> tuple[str, int]:
    """Return ``(sha256, line_count)`` of the frozen goldens file."""
    raw = _GOLDENS.read_bytes()
    line_count = len([line for line in raw.decode("utf-8").splitlines() if line.strip()])
    return hashlib.sha256(raw).hexdigest(), line_count


def chosen_tool(client: AnthropicRouterClient, question: str) -> str:
    """Force a tool call for ``question`` and return the first tool's name."""
    response = client.create_message(
        messages=[{"role": "user", "content": question}],
        tools=TOOLS,
        tool_choice=_FORCE_ANY,
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            name: str = block.name
            return name
    return _NONE


@dataclass(frozen=True)
class ProbeResult:
    """One golden and the tool the forced call routed it to."""

    golden: Golden
    chosen: str


def run_probe(client: AnthropicRouterClient, goldens: list[Golden]) -> list[ProbeResult]:
    """Route every golden once, keeping the per-question result.

    Per-question results (not an aggregated counter) are kept because ``hybrid``
    scoring depends on *each* question's own ``acceptable_tools``, not the
    class's --- the rubric's acceptable-overlap rule.
    """
    return [ProbeResult(golden, chosen_tool(client, golden.question)) for golden in goldens]


def _confusion(results: list[ProbeResult]) -> dict[str, Counter[str]]:
    confusion: dict[str, Counter[str]] = {label: Counter() for label in _ALL_LABELS}
    for result in results:
        confusion[result.golden.label][result.chosen] += 1
    return confusion


def _format_table(results: list[ProbeResult]) -> str:
    confusion = _confusion(results)
    header = f"{'label':<13}" + "".join(f"{col:>15}" for col in _TOOL_COLUMNS) + f"{'N':>6}"
    lines = [header]
    for label in _ALL_LABELS:
        row = confusion[label]
        total = sum(row.values())
        cells = "".join(f"{row.get(col, 0):>15}" for col in _TOOL_COLUMNS)
        suffix = "  (leakage; not scored)" if label == "no_answer" else ""
        lines.append(f"{label:<13}{cells}{total:>6}{suffix}")
    return "\n".join(lines)


def _accuracy_report(results: list[ProbeResult]) -> str:
    lines: list[str] = []
    correct_total = 0
    scored_total = 0
    for label in _ANSWERABLE_LABELS:
        items = [r for r in results if r.golden.label == label]
        # Each question scored against its OWN acceptable_tools (hybrids differ).
        correct = sum(1 for r in items if r.chosen in r.golden.acceptable_tools)
        correct_total += correct
        scored_total += len(items)
        lines.append(f"  {label:<12} {correct}/{len(items)}")

    overall = correct_total / scored_total if scored_total else 0.0
    leakage = dict(_confusion(results)["no_answer"])
    return (
        f"First-tool accuracy (answerable classes; hybrids count any acceptable tool): "
        f"{correct_total}/{scored_total} = {overall:.2f}\n" + "\n".join(lines) + "\n"
        f"no_answer leakage (forced choice cannot refuse): {leakage}"
    )


def main() -> None:
    """Run the probe and print the confusion table + accuracy."""
    goldens = load_goldens()
    sha, line_count = frozen_fingerprint()

    client = AnthropicRouterClient.from_settings(Settings(), max_tokens=_PROBE_MAX_TOKENS)
    results = run_probe(client, goldens)

    print("Routing probe — frozen goldens (forced iteration-0 tool choice)")
    print(f"file:   {_GOLDENS.relative_to(_REPO_ROOT)}")
    print(f"sha256: {sha}")
    print(f"lines:  {line_count}")
    print(f"model:  {client._model}")
    print()
    print("Confusion (rows = true label, cols = chosen tool):")
    print(_format_table(results))
    print()
    print(_accuracy_report(results))


if __name__ == "__main__":
    main()
