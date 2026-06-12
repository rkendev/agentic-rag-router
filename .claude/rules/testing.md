# Rule: testing

Tests live in two buckets with different purposes. Keep them separate and
the signal stays clean.

## The two buckets

- **`tests/unit/`** — one adapter, one port, one domain model at a time.
  Mirror the `src/` layout (`tests/unit/domain/`, `tests/unit/application/`,
  `tests/unit/infrastructure/`). Mock the SDK boundary (e.g. `httpx` /
  `anthropic.Anthropic`) but never mock the layer under test. Coverage
  target: 100% line coverage on `src/`. Missing lines are either dead
  code (delete them) or untested behaviour (cover them).
- **`tests/contract/`** — one parametrized suite (`test_llm_port.py`) that
  runs against every adapter registered in `tests/contract/conftest.py::LLM_ADAPTERS`.
  Three production adapters + one in-memory fake = 4 parameter values ×
  8 tests = 32 cases. Adding a new adapter multiplies by 8, not adds by 1 —
  that's the leverage.

## Discipline

- **`make check` is the gate.** Run it before every commit. If it's red,
  don't commit. CI runs the same chain on every push and will reject
  anything that drifts.
- **No `skip` without a referenced DECISION.** `@pytest.mark.skip` or
  `skipif` without a link to a `docs/DECISIONS.md` entry is a latent bug.
  Either fix the test, delete it, or write the ADR explaining why it's
  frozen.
- **No mocks in contract tests.** The point of the contract suite is that
  every adapter honours the same behavioural contract for real. Mocking
  an adapter's internals defeats the suite's only job.
- **Fakes are honest doubles, not mocks.** `tests/contract/fakes.py::FakeLLMAdapter`
  implements the full port, with the same error semantics as a real
  adapter. It exists so contract-suite runs don't need network; it does
  not exist to sidestep behaviour.
- **Coverage includes error paths.** Every exception a production adapter
  can raise needs a unit test that triggers it. The `LLMError` hierarchy
  in `domain/exceptions.py` is the catalogue — if an adapter can hit a
  branch, the unit test matrix covers it.

## When you add a new adapter

Every new `LLMPort` implementation needs two things to ship:

1. **Unit tests** in `tests/unit/infrastructure/test_<vendor>_adapter.py`
   covering happy path + each error class the adapter can raise (retryable
   vs permanent vs rate-limit vs context-length).
2. **Contract enrolment** in `tests/contract/conftest.py::LLM_ADAPTERS` —
   a factory function that returns a ready-to-use instance. The 8
   parametrized tests auto-apply.

Skipping step 2 is the common failure mode. The unit tests go green,
`make check` goes green, and six months later someone changes the Protocol
signature and only three of the four adapters break — because the fourth
was never in the contract suite. Don't be that fourth.

## When a test fails and you're tempted to "just fix" it

First, write down what the test is asserting. If the assertion is still
correct, the code under test is the bug — fix the code. If the assertion
is wrong (spec changed, contract evolved, domain shifted), update the
spec chain first (`SPECIFICATION.md` / relevant port definition / relevant
ADR), *then* update the test, *then* update the code. Order matters —
drifting tests to match broken code hides regressions the next person
won't find.

## What `make check` covers end-to-end

```
ruff check src tests        # lint
ruff format --check src tests  # format
mypy src tests              # types
bandit -r src -ll           # security (medium+ severity)
uv run python scripts/check_version_parity.py  # pin parity (ADR D8)
uv run pytest -q tests/unit tests/contract     # 219 + 32 tests
```

Every entry has a reason and a DECISIONS.md anchor. If a step feels like
drag, the fix is the DECISIONS entry, not removing the step.

---

**Verified by:** [OT-2](../../VERIFICATION.md#ot-2--every-adapter-conforms-to-llmport) (contract tests green across every registered adapter), [OT-6](../../VERIFICATION.md#ot-6--ci-is-green-on-main) (CI re-runs the full gate on every push to `main`), [OT-10](../../VERIFICATION.md#ot-10--no-bandit-highmedium-findings) (bandit finds zero medium-or-higher issues in `src/`).
