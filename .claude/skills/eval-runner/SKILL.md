---
name: eval-runner
description: Run an eval suite and produce a JSON report. Use when the user asks to "run evals", "eval", "score the model", "run the baseline", or "measure quality". Reads a golden JSONL set, dispatches per-record scoring (execution-accuracy or LLM-as-judge), writes a frozen report to evals/reports/<UTC>.json. Anti-bias rule: judge model defaults to Haiku when SUT is Sonnet (per F3 T019c precedent and F4 T021 baseline).
---

# eval-runner — produce a JSON eval report from a golden set

Use this skill when running an eval suite for a project that has:

- A `evals/golden_v1.jsonl` (or similar) with hand-labeled records.
- A `Settings.eval_sut_anthropic_model` and `Settings.eval_judge_anthropic_model`
  (per F3 T019c pattern; if absent, fall back to `claude-sonnet-4-5` and
  `claude-haiku-4-5` respectively).
- At least one runnable system-under-test entrypoint (e.g.
  `application.query_driver.run_query` for NL→SQL projects, or any callable
  returning a structured result).

## Process

1. **Discover the golden set.** Default path: `evals/golden_v1.jsonl`. Each
   record must conform to a project-defined Pydantic schema (typically
   `evals/golden_schema.py::GoldenRecord`).
2. **Discover or build the metric registry.** The skill ships with a default
   5-metric pattern (`execution_accuracy`, `intent_match_judge`,
   `out_of_schema_accuracy`, `destructive_query_rejection_rate`,
   `latency_p95_ms`) — projects extend the registry by adding metrics in
   their own `evals/run_eval.py` per the F4 reference.
3. **Per-record scoring.** For each record:
   - Call the project's SUT (the runnable that returns a structured result).
   - For `executed`-band records, run BOTH the gold SQL AND the generated
     output against the same fixture; compute `execution_accuracy` as
     multiset comparison (Spider/BIRD semantics — ignore column names by
     default).
   - Call the **Haiku judge** (per the anti-bias rule below) for
     `intent_match_judge` if the band is executable.
   - For `out_of_schema` band: assert the canonical no-knowledge response
     byte-equal.
   - For `destructive_intent` band: assert
     `error_code in ("destructive_query_rejected", "no_knowledge")` (Sonnet
     often refuses via OOS path which counts).
4. **Aggregate.** Mean of each metric across records where it applies (None
   elsewhere). `passes_threshold` flag per metric vs the thresholds JSON.
5. **Write the report** at `evals/reports/<ISO-8601-UTC>.json` with `summary`
   + `per_record` + `sut_model` + `judge_model` + `golden_set_path` fields.

## Anti-bias rule (HARD — DO NOT VIOLATE)

When the SUT is Claude Sonnet, the judge MUST be Claude Haiku (not Sonnet).
Same-model self-grading inflates scores and the inflation is non-trivial
(F3 T019c found ~10pp lift on judge metrics when SUT and judge were the
same model). The Haiku judge is also ~5x cheaper. There is no scenario in
v0.1.0 evals where same-model self-grading is acceptable.

## Reference implementation

See `evals/run_eval.py` in any project copier-instantiated from
`agentic-rag-router@v0.5.0+`. The reference is F4
(`nl2sql-copilot/evals/run_eval.py` at v0.1.0 commit `cbb2c29`). A starting
skeleton ships alongside this skill at
`.claude/skills/eval-runner/_reference_run_eval.py` — copy it into
`evals/run_eval.py`, fill in the project-specific TODOs (golden_schema
import, SUT call, custom metrics), and iterate.

## Cost discipline

Each eval run on a 50-question golden set with Sonnet SUT + Haiku judge
costs ~$0.30. Budget for 3-4 tune iterations during T021-equivalent phases.
If 3 iterations don't close the gating gap, claim ADR D11 and document the
relaxation honestly (`feedback_eval_dataset_size.md` discipline; F3 D11
precedent).

## What this skill does NOT do

- Build the golden set. That's a project-author task (e.g., F4 T020).
- Tune the SUT prompt. That's a separate skill (or just inline iteration).
- Compare against a baseline. Use `scripts/check_eval_regression.py` (also
  from F4 T022 — promote in v0.6.0).
