# Verification

Every claim the template makes about itself is tied to a command you can run.
This file is that mapping. The ten observable truths (OT-1..OT-10) are the
ones listed in [`SPECIFICATION.md §6`](SPECIFICATION.md); this document
rewrites them in the form they actually take against the shipped code, so a
reviewer (or future you) can walk through the list top to bottom and watch
each one go green.

Run from the repo root. Most commands assume `uv sync --all-extras` has
already landed the dev dependencies (see [`README.md`](README.md)).

## How to read each entry

| Field | Meaning |
| --- | --- |
| **Command** | Exact invocation. Copy, paste, run. |
| **Expected output** | The snippet that signals "pass". Surrounding noise varies by environment and can be ignored. |
| **What it proves** | The architectural claim the check is standing in for. |

If a command fails, the failure is the signal — don't paper over it. The
error and its fix belong in `docs/DECISIONS.md` if it reveals a drift in the
contract, or in `CHANGELOG.md` if it's a bug the next release needs to fix.

---

## OT-1 — Template forks cleanly

**Command**

Run from the template repo root (one level up from this file):

```bash
make template-test
```

Under the hood that invokes `bash tests/template/test_fresh_generation.sh`,
which does: `copier copy . /tmp/agentic-rag-router-scratch-$$ --defaults --trust`
→ `cd` into the scratch dir → `uv sync --all-extras` → `make check`.

**Expected output**

```
→ Scratch dir: /tmp/agentic-rag-router-scratch-NNNNN
→ Template root: /home/you/agentic-rag-router
→ copier copy ... --defaults
Copying from template ...
    create  src/my_ai_project/__init__.py
    create  pyproject.toml
    ...
→ (in scratch) uv sync --all-extras
Resolved NNN packages in ...
→ (in scratch) make check
[ruff + ruff-format + mypy + bandit + 219 tests all green]

✓ fresh generation ok: copier copy + uv sync + make check all green
```

**What it proves**

That the template is parameterised rather than hard-coded to Roy's
conventions — forking produces a working project without manual surgery.
The harness also exercises every substitution point: the `_tasks` sed
rename hits every Python import, the three `.jinja` files render without
delimiter collisions, and the generated project's own quality gate
(ruff / mypy / bandit + 219 unit + 32 contract tests) passes on a
codebase it has never seen before.

> **Note on the spec command.** SPECIFICATION.md §6 and §9 reference the
> bare `copier copy ... && make check` form. The Make target adds
> `uv sync --all-extras` between generation and `make check` (a fresh
> scratch dir has no virtualenv) and wraps the whole thing so CI has a
> single discoverable target. Same intent, end-to-end coverage.

---

## OT-2 — Every adapter conforms to `LLMPort`

**Command**

```bash
uv run pytest tests/contract/test_llm_port.py -v
```

**Expected output**

```
tests/contract/test_llm_port.py::test_returns_response[anthropic] PASSED
tests/contract/test_llm_port.py::test_returns_response[openai] PASSED
tests/contract/test_llm_port.py::test_returns_response[ollama] PASSED
tests/contract/test_llm_port.py::test_returns_response[fake] PASSED
...
============================== 32 passed in 0.XXs ==============================
```

**What it proves**

All three production adapters plus the in-memory fake honour the same
behavioural contract. A future change that breaks any one of them surfaces
as a vendor-tagged failure (`test_returns_response[anthropic]`) instead of
a subtle shape mismatch at call time.

---

## OT-3 — Pre-commit versions match `pyproject.toml`

**Command**

```bash
uv run python scripts/check_version_parity.py
```

**Expected output**

```
OK pin parity: ruff=0.8.0, mypy=1.13.0, bandit=1.8.0
```

The script exits `0` on match and non-zero with a diff on drift.

**What it proves**

The StockStream "local-green-CI-red" incident class can't happen here —
ruff / mypy / bandit run the exact same version locally (via
`pyproject.toml`'s `dev` extras) as the pre-commit hook chain does (via
`.pre-commit-config.yaml`). See ADR D8.

---

## OT-4 — Docker services come up healthy

**Command**

```bash
./scripts/smoke.sh
```

**Expected output**

```
→ docker compose up -d --wait ollama
 ✔ Network agentic-rag-router_default  Created
 ✔ Volume  agentic-rag-router_ollama   Created
 ✔ Container agentic-rag-router-ollama  Healthy
→ GET http://localhost:11434/api/tags
✓ smoke ok: compose up + ollama healthcheck + /api/tags all green
```

First cold run pulls ~1.5 GB for the Ollama image and takes ~60s; subsequent
runs hit the local Docker layer cache and finish in ~10s.

**What it proves**

The pinned digest (`ollama/ollama@sha256:2e7ce379…`, ADR D10) still resolves,
the compose file is syntactically correct, Ollama starts, and its healthcheck
clears. `smoke.sh`'s `EXIT` trap tears the stack down so repeated runs are
idempotent.

---

## OT-5 — agent tooling scaffold ships with the project

**Command**

```bash
wc -l CLAUDE.md
find .claude -type f | sort
```

**Expected output**

The first command prints `<N> CLAUDE.md` where `<N> ≤ 200`. The `find`
lists eleven files:

```
.claude/commands/handoff.md
.claude/commands/verify.md
.claude/rules/architecture.md
.claude/rules/commit.md
.claude/rules/sdd.md
.claude/rules/testing.md
.claude/settings.json
.claude/skills/add-adapter/SKILL.md
.claude/skills/add-adapter/_adapter_template.py
.claude/skills/ship-release/SKILL.md
.claude/skills/write-handoff/SKILL.md
```

Four rules (architecture, testing, SDD, commits), three skills
(`add-adapter` ships a full annotated code template alongside its
SKILL.md; `write-handoff` and `ship-release` are prose-only), two slash
commands (`/verify`, `/handoff`), and the permission baseline.
`CLAUDE.md` at the project root is the agent's orientation brief.

**What it proves**

Every fork gets a working agent-tooling environment on day 1 —
architecture / testing / SDD / commit discipline encoded as rules,
three ready-to-run skills, two slash commands, and a safe-default
permission set. Agent quality on a fresh fork no longer depends on
the developer remembering to paste conventions from elsewhere. The
200-line ceiling on `CLAUDE.md` is a forcing function — bloat in the
agent brief correlates with drift everywhere else.

See ADR D14 for why scaffold-shipping was preferred over "each fork
writes its own".

---

## OT-6 — CI is green on `main`

**Command**

```bash
gh run list --workflow=template-ci.yml --branch=main --limit=2
```

**Expected output**

```
STATUS  TITLE                                              WORKFLOW     BRANCH  EVENT  ID           ELAPSED  AGE
✓       chore(release): v0.3.0                             Template CI  main    push   24686344923  35s      14h ago
✓       Merge pull request #1 from rkendev/f1-t019-copier  Template CI  main    push   24681195259  39s      16h ago
```

Two consecutive green runs on `main` — the acceptance criterion in
SPECIFICATION.md §9.

**What it proves**

The copier wrap round-trips cleanly on the same environment a fresh
clone gets — not just on Roy's VPS. `template-ci.yml` at the repo root
runs `make template-test` on every push/PR: `copier copy` into a scratch
dir, `uv sync --all-extras`, then `make check` (ruff + ruff-format +
mypy + bandit + 219 unit + 32 contract tests) on the rendered project.
Two consecutive greens rules out "landed once by luck".

The per-fork `ci.yml` (shipped inside `template/.github/workflows/`)
runs the same quality gate against consumer code. It can't be exercised
directly against this repo — it's asserted transitively by OT-1, which
runs `make template-test` end-to-end.

> **Note on the spec command.** SPECIFICATION.md §6 references
> `--workflow=ci.yml`. T019 (the copier-wrap split, ADR D13) moved
> `ci.yml` into `template/.github/workflows/` so it ships with every
> fork; the root-level workflow is now `template-ci.yml`. A `ci.yml`
> query at the repo root returns only pre-T019 historical runs, which
> is the drift this row exists to catch.

---

## OT-7 — Example runs against Ollama with zero API cost

**Command**

```bash
# Ollama must be running — `./scripts/smoke.sh` or a background
# `docker compose up -d ollama` satisfies this.
LLM_TIER=tertiary uv run python -m agentic_rag_router.main \
  "Say hi in one sentence."
```

**Expected output**

```
Hi there! I'm happy to help — let me know what you need.
[tier=tertiary model=llama3.2:3b tokens_in=... tokens_out=... latency_ms=...]
```

The completion lands on stdout; the `[tier=tertiary ...]` metadata goes to
stderr so pipelines can consume `.text` cleanly.

**What it proves**

The tertiary tier is reachable without an API key, which is the whole point
of its inclusion — offline capability is not an abstract claim, it's a
command you can run with your network cable unplugged. Also verifies that
`LLM_TIER` steering short-circuits fail-over (a single adapter, no cascade).

> **Note on the spec command.** SPECIFICATION.md §6 references
> `python -m examples.hello_llm`; `examples/hello_llm/` was replaced during
> T011 with three numbered scripts. The canonical zero-API-cost example is
> either the command above (the CLI itself, forced to tertiary) or
> `LLM_TIER=tertiary uv run python examples/02_fallback_demo.py`.

---

## OT-8 — Example runs end-to-end against all three tiers

**Command**

```bash
# With ANTHROPIC_API_KEY and OPENAI_API_KEY in .env, plus local Ollama up.
make example-all-tiers
```

**Expected output**

```
→ examples/01_single_adapter.py
[header + completion from Claude Haiku]
[served by: tier=primary model=claude-haiku-4-5-20251001 ...]

→ examples/02_fallback_demo.py
[header + completion]
[served by: tier=primary ...]

→ examples/03_custom_stack.py
[header + completion]
[served by: tier=secondary ...]
```

Each script self-reports which tier served it via stderr. Any non-zero exit
fails the target — `make example-all-tiers` only returns green when all
three scripts succeed.

**What it proves**

The composition root works end-to-end for every tier. Missing keys for
scripts 01 or 03 cause an honest `LLMPermanentError` at construction and
the target goes red — a deliberate behaviour, not a bug.

> **Note on the spec command.** SPECIFICATION.md §6 phrases this as
> "diff outputs against each tier". A literal text diff of LLM completions
> is noisy by construction — tiers phrase answers differently by design —
> so the target instead runs all three scripts back-to-back and trusts each
> script's stderr tier report. Same intent, cleaner signal.

---

## OT-9 — Hatchling wheel builds

**Command**

```bash
uv build
unzip -l dist/*.whl | grep agentic_rag_router/__init__.py
```

**Expected output**

```
Successfully built dist/agentic_rag_router-0.1.0.tar.gz
Successfully built dist/agentic_rag_router-0.1.0-py3-none-any.whl
      401  1980-01-01 00:00   agentic_rag_router/__init__.py
```

**What it proves**

The `[tool.hatch.build.targets.wheel] packages = ["src/agentic_rag_router"]`
block in `pyproject.toml` is correctly pointing Hatch at the package dir.
Without that line Hatch silently builds an empty wheel — the StockStream
incident this check exists to prevent. `grep`ing for `__init__.py`
forces an actual-file assertion rather than trusting the "Successfully
built" line.

> **Note on the spec command.** SPECIFICATION.md §6 references
> `python -m build --wheel`. The Makefile and this doc use `uv build`
> because uv is the project's unified tool (`uv sync`, `uv run`, `uv build`).
> Both commands produce bytewise-compatible wheels in `dist/`.

---

## OT-10 — No bandit high/medium findings

**Command**

```bash
uv run bandit -r src -ll
```

**Expected output**

```
Run started: ...
Test results:
	No issues identified.
Code scanned:
	Total lines of code: 1184
...
Run metrics:
	Total issues (by severity):
		Undefined: 0
		Low: 0
		Medium: 0
		High: 0
```

The `-ll` flag filters to medium-or-higher severity; low-severity noise is
suppressed by design.

**What it proves**

The shipped code is free of known-dangerous patterns (`eval`, hard-coded
credentials, unsafe deserialisation, insecure hashing). Bandit also runs in
the pre-commit hook chain — this target exists so `make check` has a
single-pass "security touched" signal without relying on hooks being
installed.

---

## Running the full set

Every check above runs through a Make target:

```bash
make parity          # OT-3
make check           # OT-2, OT-10 (plus ruff + mypy)
make smoke           # OT-4
make build           # OT-9
make example-all-tiers   # OT-7, OT-8
```

All ten checks are now live. Two were deferred in earlier releases — OT-1
lit up at `v0.2.0` with the copier wrap; OT-5 lit up at `v0.3.0` with the
`.claude/` scaffold — and are now part of the default verification pass.
