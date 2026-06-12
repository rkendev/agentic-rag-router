"""Reference eval runner — copy to ``evals/run_eval.py`` and fill in TODOs.

Generalized from F4's ``nl2sql-copilot/evals/run_eval.py`` (v0.1.0 commit
``cbb2c29``). The skeleton ships the metric-registry abstraction, the
golden-set loader, the aggregator, the threshold checker, and the report
writer. Project-specific bits (golden schema import, SUT call,
domain-specific metrics) are marked with ``# TODO``.

Anti-bias rule (HARD): when the SUT is Claude Sonnet, the judge must be
Claude Haiku. See ``SKILL.md`` for the rationale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, quantiles
from typing import Any

import structlog

# TODO: replace with your project's settings + SUT entrypoint.
# from agentic_rag_router.infrastructure.settings import Settings
# from agentic_rag_router.application.driver import run as run_sut

_logger = structlog.get_logger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_PATH: Path = REPO_ROOT / "evals" / "golden_v1.jsonl"
DEFAULT_THRESHOLDS_PATH: Path = REPO_ROOT / "evals" / "thresholds.json"
DEFAULT_REPORT_DIR: Path = REPO_ROOT / "evals" / "reports"

# TODO: extend or shrink to the metrics this project actually scores.
METRIC_NAMES: tuple[str, ...] = (
    "execution_accuracy",
    "intent_match_judge",
    "out_of_schema_accuracy",
    "destructive_query_rejection_rate",
    "latency_p95_ms",
)

_FILENAME_TIMESTAMP_SCRUB = re.compile(r"[:]")
_SCORE_PARSE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_goldens(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records into a list of dicts."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_thresholds(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"thresholds file {path} must be a JSON object"
        raise ValueError(msg)
    return {str(k): dict(v) for k, v in raw.items()}


def parse_judge_score(raw: str) -> float | None:
    """Extract first decimal from judge response, clamp to [0.0, 1.0]."""
    match = _SCORE_PARSE_RE.search(raw)
    if match is None:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return max(0.0, min(1.0, value))


# Metric-registry abstraction. Each entry is (name, scorer). The scorer
# takes (record, sut_result, judge) and returns float | None. Returning
# None means the metric does not apply to this record's band.
ScorerFn = Callable[[dict[str, Any], Any, Any], float | None]


def evaluate_record(
    record: dict[str, Any],
    *,
    scorers: dict[str, ScorerFn],
    judge: Any,
) -> dict[str, Any]:
    """Run one golden record end-to-end and score it on every metric.

    The metric a record contributes to is determined by which scorers
    return non-None for that record. ``None`` keeps the means honest
    (a destructive record does not pull execution_accuracy toward zero).
    """
    t_start = time.perf_counter()
    # TODO: replace stub with your project's SUT call.
    # sut_result = run_sut(record["question"], ...)
    sut_result: Any = None
    latency_ms = int((time.perf_counter() - t_start) * 1000)

    metrics: dict[str, float | None] = {}
    for name, scorer in scorers.items():
        if name == "latency_p95_ms":
            continue
        metrics[name] = scorer(record, sut_result, judge)

    return {
        "id": record.get("id"),
        "band": record.get("expected_band"),
        "question": record.get("question"),
        "latency_ms": latency_ms,
        "metrics": metrics,
    }


def _p95(values: list[int]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    qs = quantiles(sorted(values), n=100, method="inclusive")
    return float(qs[94])


def aggregate_results(per_record: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for metric in METRIC_NAMES:
        if metric == "latency_p95_ms":
            latencies = [int(r["latency_ms"]) for r in per_record]
            summary[metric] = {"value": _p95(latencies), "n": len(latencies)}
            continue
        values = [r["metrics"][metric] for r in per_record if r["metrics"].get(metric) is not None]
        summary[metric] = {
            "mean": fmean(values) if values else None,
            "n": len(values),
        }
    return {"summary": summary}


def check_thresholds(
    summary: dict[str, dict[str, Any]],
    thresholds: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    annotated: dict[str, dict[str, Any]] = {}
    all_gating_passed = True
    for metric, stats in summary.items():
        spec = thresholds.get(metric, {})
        min_value = spec.get("min")
        gating = bool(spec.get("gating", False))

        new_stats = dict(stats)
        new_stats["min_threshold"] = min_value
        new_stats["gating"] = gating

        if metric == "latency_p95_ms":
            value = stats.get("value")
            passes = (
                True
                if min_value is None
                else (False if value is None else bool(value <= min_value))
            )
        else:
            mean = stats.get("mean")
            passes = False if (mean is None or min_value is None) else bool(mean >= min_value)

        new_stats["passes_threshold"] = passes
        annotated[metric] = new_stats
        if gating and not passes:
            all_gating_passed = False
    return all_gating_passed, annotated


def _utc_iso8601_now() -> str:
    raw = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _FILENAME_TIMESTAMP_SCRUB.sub("-", raw)


def write_report(report: dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{_utc_iso8601_now()}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run eval against golden_v1.jsonl.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    records = load_goldens(args.golden_path)
    if args.limit is not None:
        records = records[: args.limit]
    thresholds = load_thresholds(args.thresholds_path)

    # TODO: build SUT + judge ports here. Anti-bias: judge MUST be Haiku
    # when SUT is Sonnet. See SKILL.md.
    # settings = Settings()
    # sut = build_sut(settings)
    # judge = build_judge(settings)  # AnthropicAdapter(model=eval_judge_anthropic_model)
    judge: Any = None

    # TODO: register the project's scorers. Each returns float | None.
    scorers: dict[str, ScorerFn] = {
        # "execution_accuracy": score_execution_accuracy,
        # "intent_match_judge": score_intent_match_judge,
        # "out_of_schema_accuracy": score_out_of_schema_accuracy,
        # "destructive_query_rejection_rate": score_destructive_rejection,
    }

    started_at = datetime.now(UTC)
    t_start = time.perf_counter()
    per_record = [evaluate_record(r, scorers=scorers, judge=judge) for r in records]
    elapsed_seconds = round(time.perf_counter() - t_start, 2)

    aggregates = aggregate_results(per_record)
    all_gating_passed, annotated = check_thresholds(aggregates["summary"], thresholds)

    report = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        # TODO: pull from settings — the report MUST record both models.
        "sut_model": "claude-sonnet-4-5",
        "judge_model": "claude-haiku-4-5",
        "golden_set_path": str(args.golden_path),
        "n_records": len(records),
        "thresholds": thresholds,
        "all_gating_passed": all_gating_passed,
        "summary": annotated,
        "per_record": per_record,
    }
    report_path = write_report(report, args.report_dir)

    _logger.info(
        "eval_run_done",
        elapsed_seconds=elapsed_seconds,
        report=str(report_path),
        all_gating_passed=all_gating_passed,
    )
    return 0 if all_gating_passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
