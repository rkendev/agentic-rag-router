---
description: Walk every runnable row of VERIFICATION.md, run the command, and report pass/fail per OT-ID.
---

# /verify

Read `VERIFICATION.md` from the repo root. For each OT-ID row with a
runnable command in the "How to verify" column:

1. Execute the command exactly as written, in the repo root, from a
   clean shell (no inherited env beyond what the user has set). Capture
   stdout, stderr, and exit code.
2. Compare the actual output against the "Expected output" column. A
   verified row is one where the exit code is zero *and* the output
   matches the spec — check both.
3. Report one line per OT-ID:
   - `OT-N: PASS — <short reason>`
   - `OT-N: FAIL — <what diverged>` (include a 1-3 line excerpt of the
     diff)
   - `OT-N: SKIP — <why>` (e.g., row has no runnable command, or the
     environment required isn't available)

End with a summary line: `N passed, M failed, K skipped`.

## Guardrails

- Do not modify the project while verifying. If a command requires side
  effects (e.g., writes a temp file), note it but don't "fix" stale
  state the command itself was supposed to handle.
- Do not paraphrase commands. If the spec says `make check`, run
  exactly `make check` — not `uv run make check` or `cd . && make check`.
- If a command takes longer than 2 minutes, report that fact and move
  on; don't hang the whole pass on one slow row.
- If VERIFICATION.md is missing or has zero runnable rows, say so and
  stop — there's nothing to verify and that itself is a finding.

## Output shape

```
OT-1: PASS — tests/ layout matches spec
OT-2: PASS — 32 contract cases green (4 adapters × 8 tests)
OT-3: PASS — make check exits 0 with 219 tests
OT-4: SKIP — requires Docker, not available in this shell
OT-5: PASS — .claude/ directory renders cleanly via copier
OT-6: FAIL — pre-commit parity check: pyproject pins ruff 0.8.0, config pins 0.7.4
  ...
5 passed, 1 failed, 1 skipped
```

A single FAIL is a release blocker; never wave them through. A SKIP is
acceptable when the environment genuinely can't run the row — flag it
so the user knows coverage isn't complete.
