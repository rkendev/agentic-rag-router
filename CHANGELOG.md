# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
`[Unreleased]` collects changes landing on `main` ahead of the next tagged
release; each tagged version carries its release date and a stable anchor.

## [Unreleased]

### Added

- **Agentic router loop + tools schema + routing probe (T004 / D4).**
  `src/agentic_rag_router/router/` turns the three T003 adapters into a router.
  `schema.py` holds the three Anthropic tool definitions whose DESCRIPTIONS are
  the routing policy (tuned against the frozen goldens); they embed the NYC
  taxi schema (rubric §4) and the corpus cutoff (`2026-06-11`) the model routes
  on. `loop.run_router` drives Claude Sonnet (`ROUTER_MODEL`, default
  `claude-sonnet-4-6`) through a hand-written agentic loop behind an injected
  `AnthropicClientPort`: iteration 0 forces a tool choice then relaxes to
  `auto`, every parallel `tool_use` block is answered, and a 5-iteration cap
  returns a `RouterResponse` with `refusal_reason="iteration_budget_exhausted"`
  and zero citations. `dispatch.Dispatcher` maps tool calls onto the unchanged
  adapters and serializes each `ToolResult` (or error) back to the model.
  `RouterResponse` carries `answer`, `citations`, `trajectory` (per-tool
  `TrajectoryStep`s), `refusal_reason`, and `iterations` — no evidence-grade
  field (that is D5). `scripts/route_probe.py` routes the 60 frozen goldens in
  forced-choice mode and prints a per-class confusion table + first-tool
  accuracy (1.00 on the 48 answerable goldens with the shipped descriptions;
  routing-contract descriptions vs. a naive one-line baseline measured in the
  T004 report). Unit tests drive the loop with a fake Anthropic client and reach
  100% line coverage on the new `src/` lines; `tests/live/test_router_live.py`
  routes a `vector_only` and a `sql_only` golden end to end through real Sonnet
  — once with in-memory port fakes, once against the live pgvector substrates
  (`vector_search` over `corpus_docs`, `sql_query` as `router_ro`). Adds
  `Settings.router_model`. The T003 adapters were not modified.
- **Three tool adapters + result envelope (T003 / D3).** `src/agentic_rag_router/tools/`
  ships the three substrate adapters the router (D4) will route across, each
  returning a shared `ToolResult` envelope (`ok`, `tool`, `data`, `error_code`,
  `error_message`, `latency_ms` — the evidence-grade field stays out until D5).
  `vector_search` embeds the query with the same pinned MiniLM model/revision as
  ingestion and ranks `corpus_docs` by cosine distance; `sql_query` validates a
  single SELECT (rejecting multi-statement, CTE-wrapped writes, and
  comment-obfuscated writes) then executes it as the read-only `router_ro` role
  with a statement timeout and a 200-row cap; `web_search` calls Tavily over
  httpx with the API key scrubbed from the recorded VCR cassette. Heavy
  substrate deps (`psycopg`, `sentence-transformers`) are lazy-imported behind
  ports so `make check` and CI stay lean; unit tests use fakes and reach 100%
  line coverage on the new `src/` lines. The SELECT validator carries an
  adversarial reject table, and the cassette's scrub is grep-verified.
- **Data layer (T002 / D2) — retroactive entry.** Logged here as part of T003
  after a process slip omitted it from T002's own commit. The compose `postgres`
  service is the digest-pinned `pgvector/pgvector:pg16` on host loopback
  `127.0.0.1:5436`. `taxi_trips` (NYC TLC yellow 2024-01, ~2.96M rows) binds the
  frozen `EVAL_RUBRIC.md` §4 columns lowercased via `scripts/_taxi_mapping.py`;
  `corpus_docs` holds 11,194 arXiv cs.{CL,LG,AI,IR} abstracts with `vector(384)`
  embeddings from the pinned `all-MiniLM-L6-v2` (HNSW cosine index). The
  SELECT-only `router_ro` role and the idempotent `scripts/init_db.py` /
  `scripts/ingest_*.py` loaders land here, heavy deps isolated in the PEP-735
  `ingest` group, and `ci.yml`'s test step is scoped to
  `-m "not integration and not live"` so CI never needs Docker or the group.
  Provenance and the corpus cutoff date (2026-06-11) are recorded in
  `docs/DATA_SOURCES.md`.
- **Frozen evaluation set (T001 / D0).** `data/eval/golden_questions.jsonl` —
  60 hand-labelled routing questions (14 `vector_only`, 14 `sql_only`,
  14 `web_only`, 12 `no_answer` of which 8 are adversarial near-misses,
  6 `hybrid` with explicit `acceptable_tools` overlap). `docs/EVAL_RUBRIC.md`
  is the measurement contract: class definitions, the assumed NYC TLC
  yellow-taxi schema, the routing-error definition (misroute vs. the separate
  over-refusal error class), the hybrid overlap rule, and the freeze policy.
  The set is authored and frozen **before** any tool description, adapter, or
  router code so later tasks tune against a fixed target.
- **`tests/test_eval_set_frozen.py`** — freeze guard pinning the sha256 of both
  files and validating the JSONL schema (60 lines, label enum, adversarial
  count, hybrid `acceptable_tools`). Per the freeze policy, changes to the set
  require a new versioned file plus an entry here, not an in-place edit.

### Fixed

- **Scaffold CI baseline.** A fresh v0.5.0 generation failed
  `pre-commit run --all-files` (the CI lint-and-test gate): the rendered
  `_adapter_template.py` skill helper carries deliberate `<placeholder>` syntax
  and is not valid standalone Python. Excluded `*_template.py` under
  `.claude/skills` from ruff / ruff-format / mypy / bandit, and applied pending
  `end-of-file-fixer` / `ruff-format` autofixes.

## [0.5.0] — 2026-05-03

Council-driven Phase 2 retrofit (minimal). Promotes battle-tested artifacts from F4
nl2sql-copilot v0.1.0 (commit `cbb2c29`, ~21 hook-fires across T007-T024 with no
errors) plus a generalized eval-runner skill abstracted from F4's T021 pattern.

### Added

- **`.claude/hooks/SessionStart-context-prime.sh`** — primes context with `git log -3`,
  branch + working-tree status, and latest tag at session start. Saves the
  "where was I" reorientation cost per session.
- **`.claude/hooks/Stop-task-close-emit.sh`** — auto-emits the FF-merge + push +
  Windows-mirror-pull commands when a session ends on a non-`main` branch.
  Implements `feedback_task_close_sequence.md` deterministically; no LLM call.
  A 4-line comment block at the top documents that the `C:\Users\...` mirror
  path prefix is Roy's local convention — forkers on a different OS or layout
  edit the path or set `$WINDOWS_MIRROR_BASE`.
- **`.claude/commands/close.md`** — on-demand `/close` slash command; mirrors the
  Stop hook's behavior but invocable mid-session.
- **`.claude/agents/code-reviewer.md`** — senior-engineer second-pass review
  subagent. Read-only by tool constraint (Read/Grep/Glob/Bash); never edits files.
  **Hiring Manager rule preserved:** must be invoked ≥1× per project before that
  project's release tag (no ghost subagents).
- **`.claude/skills/eval-runner/SKILL.md`** + reference script
  (`_reference_run_eval.py.jinja`) — abstracts F4's custom 5-metric evaluator
  (no ragas, per ADR D10). Anti-bias rule pinned in the skill description:
  Haiku judge when SUT is Sonnet (F3 T019c precedent). Reference ships the
  metric-registry abstraction, golden-set loader, aggregator, threshold
  checker, and report writer; project-specific bits (golden schema, SUT
  call, scorer functions) are marked with `# TODO`.
- **`.claude/settings.json`** — `hooks` block added as a top-level sibling
  of `permissions`, registering both SessionStart and Stop hooks against
  `$CLAUDE_PROJECT_DIR/.claude/hooks/...`.

### Unchanged (intentional)

- `template/pyproject.toml.jinja` version remains `0.1.0` — that's the value forks instantiate with, not the template's own version. Template version lives in this git tag + this CHANGELOG entry. Per `feedback_ship_release_skill_template_version_by_tag.md`.

### Discipline rule

Per `feedback_phase2_retrofit_minimum.md`: only artifacts load-tested in a working
downstream project earn template promotion. The remaining Phase 2 items (PreToolUse
hooks, additional subagents, per-layer CLAUDE.md, `.mcp.json`) wait for v0.6.0 after
Module 3 ships and proves them.

## [0.4.0] — 2026-04-21

Ships the `/sdd` slash command — the one deliberate deferral from
v0.3.0's `.claude/` scaffold. Closes T022.5. The shipped-surface
promise that `rules/sdd.md` made in v0.3.0 ("there is a
`.claude/commands/sdd.md` that drafts the spec chain from a one-liner")
is now real; a cold-fork of the template no longer has to explain that
one of the documented commands is a placeholder.

### Added

- `.claude/commands/sdd.md` — full-featured `/sdd` slash command that
  drafts the top of the spec chain (charter + specification + technical
  plan + task breakdown) from a one-liner project description. Resolves
  the T022.5 deferral called out in v0.3.0's "Deferred" section; the
  `sdd.md` rule file (shipped v0.3.0) already documented the command as
  existing, so this closes the gap between promised and shipped
  `.claude/` surface. Produces four drafts at the repo root and cross-
  links tasks to OTs to charter success criteria. Draft-only — does
  *not* populate `docs/DECISIONS.md`, `VERIFICATION.md`, or
  `CHANGELOG.md` (those earn their entries once implementation exists).
  Includes a quality bar, a "don't fabricate stakeholders" discipline
  rule, and overwrite protection for existing spec-chain files.

## [0.3.1] — 2026-04-21

Docs-only patch. OT-6's verify command had silently gone stale after
T019's copier-wrap split and was asserting against a workflow that no
longer exists at the repo root. Caught during v0.3.0 OT-6 close-out
(2026-04-21) — exactly the "stale VERIFICATION row" rot the `sdd.md`
Red flags section exists to catch.

### Fixed

- `VERIFICATION.md` OT-6 command updated from `--workflow=ci.yml` to
  `--workflow=template-ci.yml`. The original command was written before
  T019 (ADR D13) moved `ci.yml` into `template/.github/workflows/`; it
  returned only pre-T019 historical runs at the repo root. OT-6 now
  correctly asserts against the root-level workflow that still exists
  (`template-ci.yml`), and the row documents why the per-fork `ci.yml`
  can only be asserted transitively via OT-1.

## [0.3.0] — 2026-04-20

Claude Code scaffold. Every fork now inherits a working `.claude/`
environment on day 1. Closes OT-5.

### Added

- `CLAUDE.md` at the project root — a 200-line orientation brief for
  the Claude Code agent: what the project is, where architecture / spec
  / testing rules live, what `make check` covers, where the spec chain
  lives, and what "when in doubt" looks like. Jinja-substituted so the
  project slug and package name render correctly in the fork.
- `.claude/rules/` — four rule files that encode project conventions:
  `architecture.md.jinja` (hexagonal layer rule, `domain → application
  → infrastructure` import direction, how to add a new capability),
  `testing.md` (two-bucket layout, `make check` gate, contract-suite
  enrolment discipline), `sdd.md` (charter → spec → plan → tasks chain
  discipline, absolute-date rule, feedback loop), `commit.md`
  (imperative-mood subject, `git commit -F - <<'EOF'` heredoc pattern
  for shell specials, branch conventions, what never to commit). Each
  file ends with a `**Verified by:** [OT-N](...)` footer that links
  to the corresponding VERIFICATION.md row.
- `.claude/skills/add-adapter/` — slash skill + full annotated Python
  code template. `SKILL.md.jinja` walks a seven-step process (confirm
  port contract → copy + search-and-replace → add Settings fields →
  write unit tests → enrol in contract suite → wire `main.py` if a new
  tier → close the loop). `_adapter_template.py.jinja` is a real
  working skeleton mirroring `AnthropicAdapter` — fail-fast `__init__`,
  transient-first-then-permanent exception ordering, `structlog.bind`
  pattern, with TODO markers for SDK-specific fill-in.
- `.claude/skills/write-handoff/SKILL.md` — five-section structure for
  drafting a session-to-session handoff from a recent `git diff`.
- `.claude/skills/ship-release/SKILL.md` — semver bump → CHANGELOG
  finalise → tag → push → `gh release create` sequence, with the
  heredoc pattern baked into the commit step.
- `.claude/commands/verify.md` — `/verify` slash command walks every
  runnable row in VERIFICATION.md and reports pass / fail / skip per
  OT-ID.
- `.claude/commands/handoff.md` — `/handoff` slash command invokes the
  `write-handoff` skill against the session's `git diff`.
- `.claude/settings.json` — permission baseline with safe-default
  allow list (`uv *`, `make *`, `pytest *`, `ruff *`, `mypy *`,
  `bandit *`, `pre-commit *`, non-destructive `git *`, common shell
  utilities, read-only `docker compose` subcommands, whitelisted
  `WebFetch` domains) and deny list (`rm -rf /`, `git push --force`,
  `git reset --hard`, `sudo *`, `curl * | sh`).
- `docs/DECISIONS.md` D14 — scaffold-shipping rationale (Option A/B/C
  tradeoffs, sed-sweep gotcha, per-file `.jinja` discipline).
- New VERIFICATION.md OT-5 body — replaces the v0.1.0 "not yet
  applicable" placeholder with `wc -l CLAUDE.md` + `find .claude -type
  f | sort` plus the expected file listing.

### Deferred

- `.claude/commands/sdd.md` (the full-featured `/sdd` slash command
  that drafts charter + spec skeleton + plan outline + task breakdown
  from a one-liner) is deferred to a follow-up. The rules file
  explaining SDD discipline (`.claude/rules/sdd.md`) ships now; the
  interactive drafter ships later. Tracked as T022.5.

## [0.2.0] — 2026-04-20

Copier wrap. The template now scaffolds fresh projects via
`copier copy gh:rkendev/agentic-rag-router my-new-project` — `v0.1.0`'s
fork-by-clone path still works (clone the repo, `cd template/`) but is no
longer the primary story. Closes OT-1.

### Added

- `copier.yml` at the repo root defining seven questions (`project_slug`,
  `package_name`, `project_description`, `author_name`, `author_email`,
  `year`, plus `include_postgres` / `include_redis` compose toggles).
  Uses stock jinja delimiters (`{{ }}` / `{% %}`); an earlier draft used
  `[[ ]]` which collided with TOML's `[[array.of.tables]]` syntax in
  `pyproject.toml.jinja` and was reverted before merge.
- `template/pyproject.toml.jinja`, `template/docker-compose.yml.jinja`,
  `template/LICENSE.jinja` — the three files that need runtime
  substitution. Everything else is copied verbatim and sed-renamed in a
  post-copy hook.
- `tests/template/test_fresh_generation.sh` + `make template-test` at
  the repo root. Runs `copier copy` into a scratch dir with `--defaults`
  and executes the generated project's `make check`. Green = OT-1
  passing.
- `.github/workflows/template-ci.yml` at the repo root. Runs
  `make template-test` on every push and PR so the copier wrap can't
  silently rot.
- `docs/DECISIONS.md` D13 — the two-tier layout rationale (`template/`
  subdirectory + post-copy sed rename), jinja delimiter choice, and the
  compose-only scope of the Postgres/Redis toggles.

### Changed

- **Repo layout.** All shippable files moved under `template/`. The
  repo root now only contains template-meta docs (`PROJECT_CHARTER`,
  `SPECIFICATION`, `TECHNICAL_PLAN`, `TASKS`, `DECISIONS`,
  `workflow-conversation.txt`), the outer `LICENSE`, `copier.yml`, a
  slim outer `Makefile` and `README`, the CI workflow, and the
  `tests/template/` harness.
- Plain-clone users now `cd template/ && uv sync --all-extras && make check`
  instead of running from the root directly. Documented in the new root
  README.
- `docker-compose.yml` Postgres / Redis services were previously always
  present behind `profiles: ["optional"]`. They now ship only when the
  corresponding copier question is answered `true`. Forks that don't
  need them get a slimmer compose file.

### Known limitations

- **No `copier update` migrations tested yet.** `_migrations: []` is a
  placeholder; v0.3.0's first compat-breaking change will add the first
  real migration entry and an integration test for the update path.
- **Windows sed.** `_tasks` relies on `sed -i` and `find`. Copier on
  pure Windows (no WSL) will fail the rename step. Roy's workflow runs
  generation on Linux (VPS) or WSL, so this is acceptable for v0.2.0;
  a cross-platform Python post-copy script lands when there's a real
  Windows-only user.

### Divergences from SPECIFICATION.md §6 / TASKS.md T019

- **Question set is 7, not 5.** Added `package_name` (snake_case
  identifier distinct from kebab-case `project_slug`), `author_email`,
  and `year`. Without these the generated `pyproject.toml` and `LICENSE`
  would have hard-coded placeholders the user has to manually sed after
  generation — defeats the point.
- **OT-1 command** is now `make template-test` wrapping
  `bash tests/template/test_fresh_generation.sh`, not the bare
  `copier copy ... && make check` the spec literals suggest. Semantically
  equivalent; Make target makes it CI-discoverable.

## [0.1.0] — 2026-04-20

Initial template release. F1 is feature-complete: the core hexagonal skeleton,
all three LLM adapters, the fallback composition, an offline contract suite,
CI, Docker, Makefile, and the docs stack ship together.

### Added

- **Domain layer** (`src/agentic_rag_router/domain/`) — frozen Pydantic
  `LLMResponse`, `LLMTier` StrEnum, and the three-branch `LLMError` hierarchy
  (`LLMTransientError`, `LLMPermanentError`, `LLMContentError`). No I/O, no
  third-party SDK imports.
- **Application layer** (`src/agentic_rag_router/application/`) — `@runtime_checkable`
  `LLMPort` protocol and the `FallbackModel` orchestrator. `FallbackModel`
  advances past `LLMTransientError` and re-raises permanent/content errors
  immediately.
- **Infrastructure adapters** (`src/agentic_rag_router/infrastructure/`) —
  `AnthropicAdapter` (Claude Haiku 4.5, PRIMARY), `OpenAIAdapter`
  (gpt-4o-mini, SECONDARY), `OllamaAdapter` (llama3.2:3b, TERTIARY). All
  three satisfy `LLMPort` and map SDK errors onto the domain error classes.
- **Settings** (`infrastructure/settings.py`) — `pydantic-settings` loader with
  `SecretStr` for API keys, empty-string-to-`None` coercion, and per-tier
  enable/disable based on credential presence.
- **Composition root** (`main.py`) — `build_llm(settings)` selects either a
  single adapter or a `FallbackModel` based on `LLM_TIER`. Only module
  allowed to import across all three layers.
- **Contract suite** (`tests/contract/`) — 32 parametrized tests (8 tests ×
  4 adapters including the fakes) validating that every `LLMPort`
  implementation honours the same behavioural contract. Runs fully offline
  via SDK monkeypatching.
- **Unit suite** — 187 tests across domain, application, infrastructure, and
  the composition root. Total: 219 tests, ~5s wall time, 100 % line coverage
  on `src/`.
- **Three runnable examples** (`examples/`) — `01_single_adapter.py`,
  `02_fallback_demo.py`, `03_custom_stack.py` exercise successively larger
  slices of the surface area.
- **Docker Compose + smoke script** (`docker-compose.yml`,
  `scripts/smoke.sh`) — Ollama default service + Postgres/Redis behind the
  `optional` profile. All images digest-pinned per ADR D10. `smoke.sh`
  brings up the stack, healthchecks Ollama, and tears down via an `EXIT`
  trap.
- **Pre-commit hook chain** (`.pre-commit-config.yaml`) — ruff, ruff-format,
  mypy-strict, bandit, and a local pin-parity guard that fails if
  `pyproject.toml` and `.pre-commit-config.yaml` disagree on tool versions
  (ADR D8).
- **GitHub Actions workflows** (`.github/workflows/`) — `ci.yml` (lint +
  unit tests on every push/PR) and `smoke.yml` (docker compose smoke on
  push-to-main + manual dispatch). Both pin `actions/checkout@v4` and
  `astral-sh/setup-uv@v4`.
- **Makefile** (`Makefile`) — twelve PHONY targets (`help`, `install`,
  `check`, `test`, `integration`, `live`, `smoke`, `example`,
  `example-all-tiers`, `build`, `clean`, `parity`). Every command runs
  through `uv run` so local and CI invocations use identical tool versions.
- **Docs stack** — `README.md`, `ARCHITECTURE.md` (mermaid layer diagram +
  dependency rule narrative), `VERIFICATION.md` (each observable truth
  from SPECIFICATION.md §6 mapped to a command plus expected output),
  `docs/DECISIONS.md` (ADR-lite, D1..D12), `PROJECT_CHARTER.md`,
  `SPECIFICATION.md`, `TECHNICAL_PLAN.md`, `TASKS.md`.
- **MIT license** (`LICENSE`).

### Known limitations

- `tests/integration/` directory exists but no integration tests are wired
  yet; `make integration` collects zero items and exits with pytest's
  "no tests collected" exit code (5). This is expected — the target is in
  place so the command surface stays complete, but it does not prove
  anything today.
- `live` pytest marker is registered and respected but no live tests are
  authored yet. `RUN_LIVE=1 make live` is a no-op until real-API tests land.
- `copier`-based template wrap (SPECIFICATION.md §9 acceptance item 5 /
  OT-1) is deferred to a follow-up task. The template is still usable as a
  plain clone target — `copier copy` just isn't wired yet.
- No `.claude/` agent-contract directory is shipped at `v0.1.0`. OT-5
  (`wc -l .claude/CLAUDE.md` ≤ 200) will first become applicable when that
  directory ships.

### Divergences from TECHNICAL_PLAN.md / SPECIFICATION.md

Each divergence is documented in-file at the point where it lives. The full
list for context:

- `make install` → `uv sync --all-extras` (plan said bare `uv sync`;
  that strips dev deps on a PEP-621 project).
- `make build` → `uv build` (plan said `python -m build --wheel`; uv is the
  project's unified tool).
- `make example` → `examples/02_fallback_demo.py` (plan referenced a
  `hello_llm/` package; three numbered scripts shipped instead).
- `make example-all-tiers` → runs all three scripts sequentially (plan said
  "diff outputs"; LLM text diffs are noisy by design).
- Added `make test` (pytest-only) for red-green iteration; not in the plan.
- `make clean` keeps `.venv/` (plan included it; blowing it away forces a
  full re-sync and is a surprise for a `clean` invocation).
- Smoke lives in its own workflow (`smoke.yml`) rather than a job inside
  `ci.yml` so PR iterations don't pay the Docker-pull cost.

[Unreleased]: https://github.com/rkendev/agentic-rag-router/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/rkendev/agentic-rag-router/releases/tag/v0.5.0
[0.4.0]: https://github.com/rkendev/agentic-rag-router/releases/tag/v0.4.0
[0.3.1]: https://github.com/rkendev/agentic-rag-router/releases/tag/v0.3.1
[0.3.0]: https://github.com/rkendev/agentic-rag-router/releases/tag/v0.3.0
[0.2.0]: https://github.com/rkendev/agentic-rag-router/releases/tag/v0.2.0
[0.1.0]: https://github.com/rkendev/agentic-rag-router/releases/tag/v0.1.0
