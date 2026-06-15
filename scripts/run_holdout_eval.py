"""Score a held-out or red-team question set with the live router.

This is the same deterministic scoring the frozen-set harness uses
(`agentic_rag_router.eval.scoring.score_run`), pointed at a *non-frozen* question
file so the numbers can be regenerated as the held-out and red-team sets evolve.
It writes its own report (`eval/<stem>_report.json` / `.md`) and never touches the
gated `eval/report.json`, so a held-out run cannot turn the frozen-set CI gate red.

Routing accuracy and refusal correctness are scored automatically. Some red-team
probes (for example SQL semantic correctness) are deliberately *not* fully
auto-scorable; their per-question row carries the `probes` and `expected_behavior`
labels so a human can judge the cases the deterministic scorer cannot.

Usage (source .env so TAVILY_API_KEY reaches the environment; export a real
ANTHROPIC_API_KEY first, since .env ships it blank)::

    export ANTHROPIC_API_KEY=sk-ant-...
    set -a && . ./.env && set +a
    uv run python -m scripts.run_holdout_eval data/eval/holdout_questions.jsonl
    uv run python -m scripts.run_holdout_eval data/eval/redteam_questions.jsonl

Cost: one live Sonnet call per question (plus Tavily/DB), a few cents per set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_rag_router.eval.scoring import (
    REFUSAL_LAYER,
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
_OUT_DIR = _REPO_ROOT / "eval"
_DEFAULT = _REPO_ROOT / "data" / "eval" / "holdout_questions.jsonl"


@dataclass(frozen=True)
class Question:
    """One non-frozen question, plus optional red-team annotations."""

    id: str
    question: str
    label: str
    acceptable_tools: tuple[str, ...]
    probes: str | None
    expected_behavior: str | None


def load_questions(path: Path) -> list[Question]:
    """Read a .jsonl question file verbatim."""
    questions: list[Question] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        questions.append(
            Question(
                id=record["id"],
                question=record["question"],
                label=record["label"],
                acceptable_tools=tuple(record["acceptable_tools"]),
                probes=record.get("probes"),
                expected_behavior=record.get("expected_behavior"),
            )
        )
    return questions


def build_dispatcher() -> Dispatcher:
    """Real Dispatcher over pgvector + ``router_ro`` + live Tavily (mirrors run_eval)."""
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
    return result.trajectory[0].tool if result.trajectory else None


def to_eval_row(question: Question, result: RouterResponse) -> EvalRow:
    """Reduce a run to the fields `scoring` needs (identical mapping to run_eval)."""
    return EvalRow(
        id=question.id,
        label=question.label,
        acceptable_tools=question.acceptable_tools,
        first_tool=_first_tool(result),
        refusal_reason=result.refusal_reason,
        citation_count=len(result.citations),
        answer_is_none=result.answer is None,
    )


def to_audit_row(question: Question, result: RouterResponse) -> dict[str, Any]:
    """Full per-question record, including red-team annotations."""
    reason = result.refusal_reason
    return {
        "id": question.id,
        "label": question.label,
        "acceptable_tools": list(question.acceptable_tools),
        "probes": question.probes,
        "expected_behavior": question.expected_behavior,
        "first_tool": _first_tool(result),
        "refused": reason is not None,
        "refusal_reason": reason,
        "refusal_layer": None if reason is None else REFUSAL_LAYER.get(reason, "other"),
        "citation_count": len(result.citations),
        "answer_is_none": result.answer is None,
        "iterations": result.iterations,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _rel(path: Path) -> str:
    """Repo-relative display path, falling back to the path as-is if outside the repo."""
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def render_markdown(report: dict[str, Any]) -> str:
    """A concise human-readable report: metrics plus a per-question outcome table."""
    m = report["metrics"]
    lines: list[str] = []
    lines.append(f"# Held-out / red-team eval --- {report['source']}")
    lines.append("")
    lines.append(
        f"Live router ({report['model']}), {report['count']} questions, "
        f"generated {report['generated_at']}. Same deterministic scoring as the "
        "frozen-set harness; this report is not a CI gate."
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(
        f"| routing_accuracy | {_fmt(m['routing_accuracy'])} "
        f"({m['routing_correct']}/{m['routing_total']}) |"
    )
    lines.append(
        f"| refusal_correctness | {_fmt(m['refusal_correctness'])} "
        f"({m['refused']}/{m['no_answer_total']}) |"
    )
    lines.append(f"| over_refusals | {m['over_refusals']} |")
    lines.append(
        f"| citation_coverage | {_fmt(m['citation_coverage'])} "
        f"({m['answered_with_citations']}/{m['answered']}) |"
    )
    lines.append("")
    lines.append("## Per-question outcomes")
    lines.append("")
    lines.append("| id | label | probes | first_tool | refused | reason | cites |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {row['label']} | {row['probes'] or ''} | "
            f"{row['first_tool'] or ''} | {'yes' if row['refused'] else 'no'} | "
            f"{row['refusal_reason'] or ''} | {row['citation_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run a question set through the live router and write a standalone report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "questions",
        nargs="?",
        default=str(_DEFAULT),
        help="Path to a .jsonl question file (default: data/eval/holdout_questions.jsonl)",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Report filename stem (default: derived from the input file name)",
    )
    args = parser.parse_args()

    path = Path(str(args.questions)).resolve()
    stem = str(args.stem) if args.stem else path.stem.replace("_questions", "")
    questions = load_questions(path)

    client = AnthropicRouterClient.from_settings(Settings())
    dispatcher = build_dispatcher()

    print(f"Held-out / red-team runner --- {_rel(path)}")
    print(f"model:  {client._model}")
    print(f"count:  {len(questions)}\n")

    rows: list[EvalRow] = []
    audit: list[dict[str, Any]] = []
    for question in questions:
        result = run_router(question.question, client=client, tools=TOOLS, dispatcher=dispatcher)
        rows.append(to_eval_row(question, result))
        audit.append(to_audit_row(question, result))
        outcome = "REFUSE" if result.refusal_reason is not None else "ANSWER"
        print(f"  {question.id} {question.label:<12} {outcome:<7} first={_first_tool(result)}")

    goldens = [Golden(q.id, q.label, q.acceptable_tools) for q in questions]
    report: dict[str, Any] = {
        "source": _rel(path),
        "model": client._model,
        "count": len(questions),
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": score_run(rows),
        "confusion": confusion(rows),
        "naive_baseline": naive_baselines(goldens),
        "rows": audit,
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = _OUT_DIR / f"{stem}_report.json"
    out_md = _OUT_DIR / f"{stem}_report.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    m = report["metrics"]
    print(f"\n  routing_accuracy      {_fmt(m['routing_accuracy'])}")
    print(f"  refusal_correctness   {_fmt(m['refusal_correctness'])}")
    print(f"  over_refusals         {m['over_refusals']}")
    print(f"  wrote {_rel(out_json)}")
    print(f"  wrote {_rel(out_md)}")


if __name__ == "__main__":
    main()
