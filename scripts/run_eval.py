"""Live eval runner --- the D6 evaluation harness (NOT the CI gate itself).

Runs the REAL router (live Sonnet + real pgvector / ``router_ro`` / Tavily
substrates) over all 60 frozen golden questions and writes two committed
artifacts that the CI gate (`tests/test_eval_gates.py`) asserts against:

* ``eval/report.json`` --- machine-readable: goldens sha256 + count, run
  timestamp, model id, per-question rows (id, label, first_tool,
  acceptable_tools, refused, refusal_reason, refusal_layer, citation_count,
  per-step grade trace, iterations), the aggregate metrics
  (`agentic_rag_router.eval.scoring.score_run`), the per-class confusion table,
  and the three naive single-tool baselines.
* ``eval/EVAL_REPORT.md`` --- human-readable: the confusion table, metrics vs.
  the locked gates, the baseline comparison, sentinel/backstop refusal
  attribution, the citation-coverage proxy note, and the two disclosure
  paragraphs.

Scoring is deterministic and computed purely from ``RouterResponse`` fields ---
there is no LLM-as-judge anywhere in this harness. Re-running the eval is a
documented manual step: a fresh report must be committed for CI to stay green if
the goldens or the router change (the gate sha-pins the report to the frozen
set, so drift is visible by construction).

Frozen fidelity: questions are read VERBATIM from
``data/eval/golden_questions.jsonl``; the script prints (and the report records)
the file's sha256 + line count so a run is traceable to the exact frozen set.

Usage (source ``.env`` so ``TAVILY_API_KEY`` reaches ``os.environ``)::

    set -a && . ./.env && set +a && uv run python scripts/run_eval.py

Cost: ~60-180 live Sonnet calls plus Tavily/DB per pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_rag_router.eval.scoring import (
    ANSWERABLE_LABELS,
    BASELINE_POLICIES,
    LABEL_NO_ANSWER,
    REFUSAL_LAYER,
    ROUTE_COLUMNS,
    EvalRow,
    Golden,
    confusion,
    naive_baselines,
    score_run,
)
from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.router.client import AnthropicRouterClient
from agentic_rag_router.router.dispatch import Dispatcher
from agentic_rag_router.router.loop import RouterResponse, run_router
from agentic_rag_router.router.schema import TOOLS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDENS = _REPO_ROOT / "data" / "eval" / "golden_questions.jsonl"
_GOLDENS_REL = _GOLDENS.relative_to(_REPO_ROOT).as_posix()
_OUT_DIR = _REPO_ROOT / "eval"
_REPORT_JSON = _OUT_DIR / "report.json"
_REPORT_MD = _OUT_DIR / "EVAL_REPORT.md"

SCHEMA_VERSION = 1

# Tuning-iteration log. Every eval run and what changed on the *sanctioned* tuning
# surface (tool descriptions / system prompt in `router/schema.py`; the frozen
# golden set and rubric are never touched). Recorded in the report so the routing
# and refusal numbers are always traceable to a known router surface, per the D6
# acceptance protocol ("record every iteration: run count, what changed").
RUN_LOG: tuple[dict[str, Any], ...] = (
    {
        "run": 1,
        "change": (
            "Baseline --- D5-shipped tool descriptions + system prompt. "
            "over_refusals=2: G035 refused at the sentinel layer and G042 hit the "
            "iteration budget, both answerable web_only current-events questions."
        ),
    },
    {
        "run": 2,
        "change": (
            "System prompt (sanctioned surface): (1) reframed the future-value "
            "refusal rule so current or recent events that are reported, announced, "
            "enacted, or scheduled are answerable from sourced web results, reserving "
            "refusal for genuinely unpredictable values (e.g. tomorrow's index close); "
            "(2) added a synthesize-don't-re-search rule (never call the same "
            "substrate more than twice). Fixed G035's sentinel over-refusal, but both "
            "G035 and G042 then exhausted the iteration budget (the soft 'stop "
            "re-searching' instruction was not honoured); over_refusals still 2."
        ),
    },
    {
        "run": 3,
        "change": (
            "Router loop (loop.py): the final allowed iteration now forbids tools "
            "(tool_choice 'none'), so the model must answer or emit the sentinel from "
            "the evidence already gathered instead of re-searching to the cap. "
            "MAX_ITERATIONS stays 5; the iteration-budget refusal is now a fallback "
            "only when the forced final turn yields no usable text. Fixed G035, but "
            "boundary goldens flickered under sampling noise: G042 over-refused "
            "(sentinel) and G048 (no_answer near-miss) answered, so both refusal "
            "gates failed."
        ),
    },
    {
        "run": 4,
        "change": (
            "Router client (client.py): set temperature=0 on messages.create so the "
            "tool choice and refusal decision are deterministic --- removes the "
            "run-to-run sampling noise that flickered G042 and G048. Result is now "
            "stable: over_refusals=0 (G042 fixed), but G048 (no_answer near-miss) "
            "deterministically answers an exact-compute question from on-topic "
            "abstracts -> refusal_correctness 0.9167. Keeps the run-2 prompt and "
            "run-3 loop changes; frozen set untouched."
        ),
    },
    {
        "run": 5,
        "change": (
            "System prompt (sanctioned surface): sharpened the specific-fact refusal "
            "rule to state that abstract-level corpus text never contains exact "
            "training compute or hardware figures (GPU model, GPU-hours, FLOPs, cost) "
            "or full hyperparameter tables, so retrieving on-topic abstracts is not "
            "grounds to answer --- targets G048 deterministically. Conceptual "
            "vector_only goldens (what-is / how-does) are untouched; frozen set "
            "untouched."
        ),
    },
)


@dataclass(frozen=True)
class GoldenQuestion:
    """One frozen golden, read verbatim (question included so we can run it)."""

    id: str
    question: str
    label: str
    acceptable_tools: tuple[str, ...]


def load_goldens() -> list[GoldenQuestion]:
    """Read the frozen golden set verbatim (no paraphrasing)."""
    goldens: list[GoldenQuestion] = []
    for line in _GOLDENS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        goldens.append(
            GoldenQuestion(
                id=record["id"],
                question=record["question"],
                label=record["label"],
                acceptable_tools=tuple(record["acceptable_tools"]),
            )
        )
    return goldens


def frozen_fingerprint() -> tuple[str, int]:
    """Return ``(sha256, line_count)`` of the frozen goldens file."""
    raw = _GOLDENS.read_bytes()
    line_count = len([line for line in raw.decode("utf-8").splitlines() if line.strip()])
    return hashlib.sha256(raw).hexdigest(), line_count


def build_dispatcher() -> Dispatcher:
    """Real Dispatcher over pgvector + ``router_ro`` + live Tavily (mirrors the probe)."""
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


def _first_tool(result: RouterResponse) -> str | None:
    """The first tool the router invoked, or ``None`` if it never called one."""
    return result.trajectory[0].tool if result.trajectory else None


def _grade_trace(result: RouterResponse) -> list[dict[str, Any]]:
    """Per-step grade trace: the chain that explains a refusal (the leak path)."""
    return [
        {"tool": step.tool, "ok": step.ok, "error_code": step.error_code, "grade": step.grade}
        for step in result.trajectory
    ]


def to_eval_row(golden: GoldenQuestion, result: RouterResponse) -> EvalRow:
    """Reduce a run to the fields `scoring` needs."""
    return EvalRow(
        id=golden.id,
        label=golden.label,
        acceptable_tools=golden.acceptable_tools,
        first_tool=_first_tool(result),
        refusal_reason=result.refusal_reason,
        citation_count=len(result.citations),
        answer_is_none=result.answer is None,
    )


def to_audit_row(golden: GoldenQuestion, result: RouterResponse) -> dict[str, Any]:
    """The full per-question record committed to ``report.json`` for the audit trail."""
    reason = result.refusal_reason
    return {
        "id": golden.id,
        "label": golden.label,
        "acceptable_tools": list(golden.acceptable_tools),
        "first_tool": _first_tool(result),
        "refused": reason is not None,
        "refusal_reason": reason,
        "refusal_layer": None if reason is None else REFUSAL_LAYER.get(reason, "other"),
        "citation_count": len(result.citations),
        "answer_is_none": result.answer is None,
        "iterations": result.iterations,
        "grades": _grade_trace(result),
    }


def build_report(
    goldens: list[GoldenQuestion],
    results: list[RouterResponse],
    *,
    model: str,
    sha: str,
    count: int,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the machine-readable report dict from the run."""
    eval_rows = [to_eval_row(g, r) for g, r in zip(goldens, results, strict=True)]
    golden_specs = [Golden(g.id, g.label, g.acceptable_tools) for g in goldens]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "model": model,
        "goldens": {"path": _GOLDENS_REL, "sha256": sha, "count": count},
        "metrics": score_run(eval_rows),
        "confusion": confusion(eval_rows),
        "naive_baseline": naive_baselines(golden_specs),
        "run_log": list(RUN_LOG),
        "rows": [to_audit_row(g, r) for g, r in zip(goldens, results, strict=True)],
    }


def _fmt(value: float | None) -> str:
    """Render a metric value for the markdown tables."""
    return "n/a" if value is None else f"{value:.4f}"


def _gate_rows(metrics: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(metric, value, gate, pass/FAIL) tuples for the gates table."""
    routing = metrics["routing_accuracy"]
    refusal = metrics["refusal_correctness"]
    over = metrics["over_refusals"]
    return [
        ("routing_accuracy", _fmt(routing), ">= 0.85", "PASS" if routing >= 0.85 else "FAIL"),
        ("refusal_correctness", _fmt(refusal), "== 1.0", "PASS" if refusal == 1.0 else "FAIL"),
        ("over_refusals", str(over), "== 0", "PASS" if over == 0 else "FAIL"),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    """Render ``EVAL_REPORT.md`` from the report dict."""
    metrics: dict[str, Any] = report["metrics"]
    table: dict[str, dict[str, int]] = report["confusion"]
    baseline: dict[str, Any] = report["naive_baseline"]
    goldens_meta: dict[str, Any] = report["goldens"]

    lines: list[str] = []
    lines.append("# Evaluation report --- agentic-rag-router routing & refusal (D6)")
    lines.append("")
    lines.append(
        "Generated by `scripts/run_eval.py` over the live router (real Sonnet + "
        "pgvector / `router_ro` / Tavily). Scoring is deterministic, computed purely "
        "from `RouterResponse` fields --- there is no LLM-as-judge in this harness."
    )
    lines.append("")
    lines.append(f"- model: `{report['model']}`")
    lines.append(f"- generated_at: `{report['generated_at']}`")
    lines.append(f"- goldens: `{goldens_meta['path']}` ({goldens_meta['count']} questions)")
    lines.append(f"- goldens sha256: `{goldens_meta['sha256']}`")
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("Locked gate values --- not relaxable without a logged CHANGELOG decision.")
    lines.append("")
    lines.append("| metric | value | gate | result |")
    lines.append("| --- | --- | --- | --- |")
    for name, value, gate, verdict in _gate_rows(metrics):
        lines.append(f"| {name} | {value} | {gate} | {verdict} |")
    lines.append("")

    lines.append("## Routing accuracy by class (answerable goldens)")
    lines.append("")
    lines.append(
        f"First-tool accuracy: **{_fmt(metrics['routing_accuracy'])}** "
        f"({metrics['routing_correct']}/{metrics['routing_total']}). "
        "A route is correct when the first tool invoked is in the question's "
        "`acceptable_tools` (for a `hybrid`, any listed tool)."
    )
    lines.append("")
    lines.append("| class | correct | total | accuracy |")
    lines.append("| --- | --- | --- | --- |")
    for label in ANSWERABLE_LABELS:
        stats = metrics["per_class"][label]
        lines.append(
            f"| {label} | {stats['correct']} | {stats['total']} | {_fmt(stats['accuracy'])} |"
        )
    lines.append("")

    lines.append("## Confusion table (rows = true label, columns = outcome)")
    lines.append("")
    lines.append("A refusal lands in the `refuse` column regardless of which tool it tried first.")
    lines.append("")
    header = "| label | " + " | ".join(ROUTE_COLUMNS) + " |"
    lines.append(header)
    lines.append("| --- " * (len(ROUTE_COLUMNS) + 1) + "|")
    for label in (*ANSWERABLE_LABELS, LABEL_NO_ANSWER):
        row = table.get(label, {})
        cells = " | ".join(str(row.get(col, 0)) for col in ROUTE_COLUMNS)
        lines.append(f"| {label} | {cells} |")
    lines.append("")

    lines.append("## Refusal correctness")
    lines.append("")
    attribution: dict[str, int] = metrics["refusal_attribution"]
    lines.append(
        f"Refused **{metrics['refused']}/{metrics['no_answer_total']}** `no_answer` goldens "
        f"(refusal_correctness = **{_fmt(metrics['refusal_correctness'])}**), each with no "
        "answer and zero citations. Layer attribution (sentinel = the model emitted the "
        "`REFUSE:` sentinel; backstop = the grade-based suppression; iter_budget = loop-cap "
        "fallback):"
    )
    lines.append("")
    lines.append(
        f"- sentinel: {attribution['sentinel']} | backstop: {attribution['backstop']} "
        f"| iter_budget: {attribution['iter_budget']} | other: {attribution['other']}"
    )
    lines.append("")
    over_ids = metrics["over_refusal_ids"]
    over_note = "none" if not over_ids else ", ".join(over_ids)
    lines.append(
        f"Over-refusals (answerable goldens that refused --- rubric §5.2's separate error "
        f"class): **{metrics['over_refusals']}** ({over_note})."
    )
    lines.append("")

    lines.append("## Citation coverage")
    lines.append("")
    lines.append(
        f"**{_fmt(metrics['citation_coverage'])}** "
        f"({metrics['answered_with_citations']}/{metrics['answered']} answered goldens carry "
        "at least one citation). This is `citation_coverage`, NOT faithfulness: citations "
        "derive from `sufficient`-graded evidence, which is a proxy for groundedness, not a "
        "check that the answer's claims are actually supported."
    )
    lines.append("")

    lines.append("## Naive single-tool baselines")
    lines.append("")
    lines.append(
        "Each constant policy ('always call tool X') scored over the same answerable "
        f"denominator ({baseline['denominator']}) as routing accuracy --- a like-for-like "
        "floor that contextualizes the >= 0.85 gate for an outside reader. A constant policy "
        "can never refuse, so `no_answer` is excluded from the denominator."
    )
    lines.append("")
    lines.append("| policy | correct | total | accuracy |")
    lines.append("| --- | --- | --- | --- |")
    for tool in BASELINE_POLICIES:
        stats = baseline["policies"][tool]
        lines.append(
            f"| always-{tool} | {stats['correct']} | {stats['total']} | {_fmt(stats['accuracy'])} |"
        )
    best = baseline["best"]
    lines.append("")
    lines.append(f"Best naive baseline: **always-{best['policy']}** at {_fmt(best['accuracy'])}.")
    lines.append("")

    lines.append("## Tuning iterations")
    lines.append("")
    lines.append(
        "Every run and what changed on the sanctioned tuning surface (tool descriptions "
        "/ system prompt). The frozen golden set and rubric are never touched."
    )
    lines.append("")
    for entry in report["run_log"]:
        lines.append(f"- **Run {entry['run']}** --- {entry['change']}")
    lines.append("")

    lines.append("## Disclosures")
    lines.append("")
    lines.append(
        "1. The tool descriptions and the system prompt were iterated against this same "
        "frozen golden set (the sanctioned tuning process; the set was authored and frozen "
        "before any router code existed). The eval set therefore doubles as the tuning set, "
        "so these numbers measure fit to a target the router was tuned toward; a held-out "
        "set is future work."
    )
    lines.append("")
    lines.append(
        "2. In live runs all `no_answer` refusals fired at the sentinel layer; the "
        "deterministic grade-based backstop is defense-in-depth that has fired only in unit "
        "tests. Per-refusal layer attribution is in `report.json` (`rows[].refusal_layer`) "
        "and summarized above."
    )
    lines.append("")
    return "\n".join(lines)


def _print_summary(report: dict[str, Any]) -> None:
    """Console summary mirroring the probes' style."""
    metrics: dict[str, Any] = report["metrics"]
    print("\nSUMMARY")
    for name, value, gate, verdict in _gate_rows(metrics):
        print(f"  {name:<22}{value:<10}{gate:<10}{verdict}")
    print(
        f"  citation_coverage     {_fmt(metrics['citation_coverage'])}  "
        f"({metrics['answered_with_citations']}/{metrics['answered']})"
    )
    best = report["naive_baseline"]["best"]
    print(f"  naive baseline        always-{best['policy']} = {_fmt(best['accuracy'])}")
    print(f"\n  wrote {_REPORT_JSON.relative_to(_REPO_ROOT).as_posix()}")
    print(f"  wrote {_REPORT_MD.relative_to(_REPO_ROOT).as_posix()}")


def main() -> None:
    """Run all 60 goldens through the live router and write the two artifacts."""
    goldens = load_goldens()
    sha, count = frozen_fingerprint()

    client = AnthropicRouterClient.from_settings(Settings())
    dispatcher = build_dispatcher()

    print("Eval runner --- live router over the frozen goldens (D6)")
    print(f"file:   {_GOLDENS_REL}")
    print(f"sha256: {sha}")
    print(f"lines:  {count}")
    print(f"model:  {client._model}\n")

    results: list[RouterResponse] = []
    for golden in goldens:
        result = run_router(golden.question, client=client, tools=TOOLS, dispatcher=dispatcher)
        results.append(result)
        outcome = "REFUSE" if result.refusal_reason is not None else "ANSWER"
        print(f"  {golden.id} {golden.label:<12} {outcome:<7} first={_first_tool(result)}")

    report = build_report(
        goldens,
        results,
        model=client._model,
        sha=sha,
        count=count,
        generated_at=datetime.now(UTC).isoformat(),
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _REPORT_MD.write_text(render_markdown(report), encoding="utf-8")

    _print_summary(report)


if __name__ == "__main__":
    main()
