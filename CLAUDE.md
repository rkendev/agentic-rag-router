# agentic-rag-router

AI-augmented Python project scaffolded from an upstream template. The exact
source repository and commit are recorded in `.copier-answers.yml`; run
`copier update` to pull improvements.

Claude Code reads this file at the start of every session. Keep it under
200 lines; the ceiling exists so the agent brief stays focused (see
[`VERIFICATION.md`](VERIFICATION.md) OT-5). Detail belongs in linked files,
not here.

## What this project is

`agentic-rag-router` is a hexagonal / DDD-lite Python application built on
three strict layers: `domain/` (types, invariants, errors), `application/`
(ports, orchestration), `infrastructure/` (SDK adapters, config). The only
composition root is `src/agentic_rag_router/main.py`. Read
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the shape,
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the why of any non-obvious
choice.

The default backend stack is three LLM adapters (Anthropic / OpenAI /
Ollama) behind an `LLMPort` Protocol, orchestrated by a `FallbackModel`
that cascades primary → secondary → tertiary on retryable errors. Replace
or extend either side freely; the ports exist precisely so new backends
don't need domain or application changes.

## How to work here

Claude Code has a bundle of rules, skills, and slash commands installed
under `.claude/`. Read the rule file before starting work in its area;
invoke the skill when its trigger matches; use the slash commands when
their scope fits.

- **Architecture:** [`.claude/rules/architecture.md`](.claude/rules/architecture.md), the dependency rule and what each layer may import.
- **Testing:** [`.claude/rules/testing.md`](.claude/rules/testing.md), unit vs. contract split, 100% line coverage on `src/`, `make check` before any commit.
- **SDD discipline:** [`.claude/rules/sdd.md`](.claude/rules/sdd.md), the charter → spec → plan → tasks chain, and how to keep it honest.
- **Commits & PRs:** [`.claude/rules/commit.md`](.claude/rules/commit.md), message format, heredoc for anything with shell specials, branch conventions.

### Skills

- `add-adapter`: scaffold a new `LLMPort` implementation. Ships a full
  annotated adapter template (`_adapter_template.py.jinja`) the skill
  copies, renames, and fills in. Handles Settings fields, unit tests, and
  contract-suite enrolment. Read [`.claude/skills/add-adapter/SKILL.md`](.claude/skills/add-adapter/SKILL.md).
- `write-handoff`: draft a session-to-session handoff doc from a recent
  diff. Read [`.claude/skills/write-handoff/SKILL.md`](.claude/skills/write-handoff/SKILL.md).
- `ship-release`. Cut a release: bump CHANGELOG, tag, push, create GH
  release. Read [`.claude/skills/ship-release/SKILL.md`](.claude/skills/ship-release/SKILL.md).

### Slash commands

- `/verify`: walk every runnable row of [`VERIFICATION.md`](VERIFICATION.md), run the command, report pass/fail per OT-ID.
- `/handoff`: generate a VPS handoff doc from `git diff` against the last tag.
- `/sdd`: draft charter + spec skeleton + plan outline + task breakdown from a one-liner project description.

## Commands

The shared gate is `make check`. Run it before every commit. CI runs the
same chain on every push.

| Command | Purpose |
| --- | --- |
| `make check` | ruff + ruff-format + mypy + bandit + 219 unit + 32 contract tests (offline, fast) |
| `make smoke` | full docker compose + Ollama healthcheck + `/api/tags` probe (online, ~10s warm) |
| `make example-all-tiers` | runs all three example scripts across primary/secondary/tertiary tiers |
| `make build` | `uv build`: wheel + sdist into `dist/` |
| `make parity` | pre-commit version-pin parity check (pyproject vs `.pre-commit-config.yaml`) |

`VERIFICATION.md` maps each claim this project makes about itself to a
runnable command. If you change a claim, update the row in the same commit.

## Project conventions worth knowing upfront

- **`uv` is the unified tool.** `uv sync --all-extras` installs everything
  (runtime + dev). `uv run <cmd>` is the preferred invocation. Don't mix
  `python -m pip` or `python -m build` in; it drifts from the lockfile.
- **Dependency rule is strict.** `domain/` imports only stdlib + pydantic.
  `application/` imports from `domain/` only. `infrastructure/` imports
  from both. If you catch yourself pulling an adapter into `domain/`,
  stop. Add a port in `application/ports.py` and invert.
- **Contract suite auto-enrols adapters.** Registering a new
  `LLMPort` implementation in `tests/contract/conftest.py::LLM_ADAPTERS`
  multiplies the 8 parametrized contract tests across every adapter
  ("N adapters → 8N contract cases"). Never skip this step when adding a
  backend.
- **Pin parity is enforced.** `ruff`, `mypy`, `bandit` versions in
  `pyproject.toml` and `.pre-commit-config.yaml` must match; the parity
  script runs in `make check` and in CI.
- **Commit messages with backticks / `$` / `!` use heredoc.** `git commit
  -m "..."` with shell specials gets bash-interpreted. See the commit
  rule for the exact pattern.

## Where the spec chain lives

```
PROJECT_CHARTER.md   → the "why" (problem, users, non-goals)
SPECIFICATION.md     → the "what" (OT-1..OT-N observable truths)
TECHNICAL_PLAN.md    → the "how" (layer shape, dependencies, risks)
TASKS.md             → the "do" (T000..T0NN atomic work items)
docs/DECISIONS.md    → ADR log (D1..DN, dated, linked from code)
VERIFICATION.md      → every claim → runnable command, OT-1..OT-N
CHANGELOG.md         → Keep-a-Changelog, tagged per release
```

When you start work, read the relevant spec-chain file first. When you
finish, update the downstream links (TASKS → VERIFICATION → CHANGELOG).
Drift between the spec chain and the shipped code is the single most
common rot in templates like this one. Keep them in lockstep.

## When in doubt

Prefer doing less. The template is deliberately minimal: three adapters,
not ten; two docs services (opt-in), not a platform. Add a port before
adding an adapter. Write the spec row before writing the test.
