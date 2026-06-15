# Verification

Every claim this project makes about itself is tied to a command you can run.
This file is that mapping. Run each from the repo root after
`uv sync --all-extras`.

## How to read each entry

| Field | Meaning |
| --- | --- |
| **Command** | Exact invocation. Copy, paste, run. |
| **Expected** | The snippet that signals "pass". Surrounding noise varies by environment. |
| **Proves** | The claim the check stands in for. |

If a command fails, the failure is the signal. Don't paper over it.

---

## V1 — The offline gate is green

**Command**

```bash
make check
```

**Expected**

```
387 passed ... in ...s
TOTAL ... 100%
```

Runs ruff (lint + format), mypy (strict), bandit, and the unit + contract +
eval-guard suites with coverage.

**Proves**

The shipped code passes lint, type-check, security scan, and 387 tests at 100%
line and branch coverage on `src/`, with no network and no API keys.

---

## V2 — Routing and refusal meet their gates

**Command**

```bash
uv run pytest tests/test_eval_gates.py -v
```

**Expected**

```
test_routing_accuracy_meets_gate PASSED
test_refusal_correctness_is_perfect PASSED
test_no_over_refusals PASSED
test_report_is_pinned_to_the_frozen_goldens PASSED
```

**Proves**

The committed `eval/report.json` clears the locked gates (routing accuracy
>= 0.85, refusal correctness == 1.0, over-refusals == 0) and is pinned by hash to
the frozen golden set, so the report cannot drift from the questions it claims to
score. To regenerate the report against the live model and substrates:

```bash
set -a && . ./.env && set +a
uv run python scripts/run_eval.py
```

The full gate table and the naive single-tool baseline (0.40) are in
[`eval/EVAL_REPORT.md`](eval/EVAL_REPORT.md).

---

## V3 — SQL is read-only by construction

**Command**

```bash
uv run pytest tests/unit/tools/test_sql_query.py -v
```

**Expected**

```
test_validate_select_accepts[...] PASSED
test_validate_select_rejects[...] PASSED
test_sql_query_rejected_returns_validation_envelope PASSED
```

**Proves**

`validate_select` accepts a single `SELECT` and rejects everything else
(multiple statements, DML, comment-buried writes). Defense in depth: the tool
connects as the SELECT-only `router_ro` role, so a validator miss still cannot
write.

---

## V4 — Evidence grading is deterministic

**Command**

```bash
uv run pytest tests/unit/router/test_grading.py -v
```

**Expected**

```
test_vector_top_similarity_at_threshold_is_sufficient PASSED
test_vector_top_similarity_below_threshold_is_weak PASSED
test_sql_with_rows_is_sufficient PASSED
test_web_zero_results_is_none PASSED
... all passed
```

**Proves**

Grades are computed from the `ToolResult` envelope by fixed rules, not by a
model. The same input always grades the same way, which is what makes a route and
a refusal reproducible.

---

## V5 — Refusals carry zero citations

**Command**

```bash
uv run pytest tests/unit/router/test_loop.py -v -k "refus or backstop or sentinel or citation"
```

**Expected**

```
test_single_tool_path_returns_answer_and_citations PASSED
test_sentinel_refusal_wins_even_over_sufficient_evidence PASSED
test_final_iteration_answer_without_sufficient_is_backstopped PASSED
test_model_answers_without_calling_a_tool_is_backstopped PASSED
... all passed
```

**Proves**

The two refusal layers both work: the model sentinel (`REFUSE:`) forces a refusal
even when a tool graded `sufficient`, and the deterministic backstop converts any
ungrounded answer into a refusal. Citations flow only from `sufficient` evidence,
so refusals carry none.

---

## V6 — Pre-commit versions match `pyproject.toml`

**Command**

```bash
uv run python scripts/check_version_parity.py
```

**Expected**

```
OK pin parity: ruff=0.8.0, mypy=1.13.0, bandit=1.8.0
```

**Proves**

ruff / mypy / bandit run the exact same version locally (via the `dev` extras) as
the pre-commit hook chain does (via `.pre-commit-config.yaml`). A local-green /
CI-red version skew cannot happen.

---

## V7 — Every adapter conforms to `LLMPort`

**Command**

```bash
uv run pytest tests/contract/test_llm_port.py -v
```

**Expected**

```
tests/contract/test_llm_port.py::test_returns_response[anthropic] PASSED
tests/contract/test_llm_port.py::test_returns_response[openai] PASSED
tests/contract/test_llm_port.py::test_returns_response[ollama] PASSED
tests/contract/test_llm_port.py::test_returns_response[fake] PASSED
... 32 passed
```

**Proves**

The inherited adapter library (the substrate the examples and CLI use, not the
router itself) holds its contract: all three production adapters plus the
in-memory fake honour the same behaviour. A future change that breaks one shows
up as a vendor-tagged failure.

---

## V8 — The wheel builds with the package inside it

**Command**

```bash
uv build
unzip -l dist/*.whl | grep agentic_rag_router/__init__.py
```

**Expected**

```
Successfully built dist/agentic_rag_router-0.1.0-py3-none-any.whl
   ...  agentic_rag_router/__init__.py
```

**Proves**

The Hatch wheel target points at the package directory; the grep forces an
actual-file assertion rather than trusting the "Successfully built" line.

---

## V9 — No medium or high bandit findings

**Command**

```bash
uv run bandit -r src -ll
```

**Expected**

```
No issues identified.
... Medium: 0  High: 0
```

**Proves**

The shipped code is free of known-dangerous patterns (`eval`, hard-coded
credentials, unsafe deserialisation, insecure hashing). Bandit also runs in the
pre-commit chain; this target gives `make check` a single security signal without
relying on hooks being installed.

---

## Running the set

```bash
make check       # V1, V3, V4, V5, V6, V7, V9
make build       # V8
uv run pytest tests/test_eval_gates.py   # V2
```

The live routing and refusal numbers (V2's regeneration path) need
`ANTHROPIC_API_KEY` and `TAVILY_API_KEY` plus a loaded data layer; everything
else runs offline.
