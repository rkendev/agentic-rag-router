# agentic-rag-router — developer command surface
#
# Thin wrapper over the commands developers and CI already use. The
# Makefile is not authoritative — `pyproject.toml`, `.pre-commit-config.yaml`,
# and `ci.yml` are. This file only gives those invocations short names.
#
# All targets run through `uv run` so the version of every tool matches
# the lockfile, not whatever happens to be on the user's PATH. Matches
# what CI does; local and CI runs should never disagree.
#
# See TECHNICAL_PLAN.md §5 for the target list spec. Divergences from
# the spec are documented inline at each target and collectively in the
# T013 VPS handoff doc.

# Make the shell strict — fail on undefined vars or pipe errors — so a
# silent bug in one target doesn't cascade into a false-green elsewhere.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Make help the default when someone types just `make`. Every target
# below with a `## ` comment shows up in the generated help output.
.DEFAULT_GOAL := help

# Every target is a task, not a file — declare them PHONY so `make`
# doesn't get confused by a same-named file in the working directory.
.PHONY: help install check test integration live smoke example \
        example-all-tiers build clean parity demo-replay


help:  ## Show this help (default target)
	@echo "agentic-rag-router — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "See README.md for walkthrough. TECHNICAL_PLAN.md §5 is the spec."


install:  ## Sync dependencies from uv.lock (includes dev extras)
	# Divergence from §5: spec said `uv sync`. Bare `uv sync` on a PEP-621
	# project strips dev dependencies — see feedback_uv_sync_extras_vs_dev_flag
	# memory. `--all-extras` keeps ruff/mypy/bandit/pytest available locally.
	uv sync --all-extras


check:  ## Run the full local quality gate (ruff + mypy + bandit + unit tests)
	# Matches §5 spec exactly. Excludes `live` and `integration` markers so
	# this target runs offline with no Docker or API keys required. CI's
	# lint-and-test job runs the same chain via pre-commit; the two paths
	# should never disagree (pin-parity hook enforces the version lockstep).
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src tests
	uv run bandit -r src -ll
	uv run pytest tests/ -m "not live and not integration"
	# Auto-fix hooks (trailing-whitespace, end-of-file-fixer,
	# mixed-line-ending) aren't covered by the inline ruff/mypy/bandit
	# passes above, so a CI-side `pre-commit run --all-files` can fail
	# on bytes the local gate left alone. Firing the same chain here
	# catches that drift one push earlier (lesson from F2 T012 PR #13).
	# Guarded by a `.git/` check so the template-test scratch render
	# (which copier-renders into a non-git tmpdir) skips this step
	# gracefully; real forks always pass the guard since `git init`
	# is the first thing a fork does.
	@if [ -d .git ]; then \
		uv run pre-commit run --all-files; \
	else \
		echo "skipping pre-commit run --all-files (no .git/ — likely template-test scratch render)"; \
	fi


test:  ## Alias for `check` minus the lint/type/security passes (just pytest)
	# Not in §5, added for ergonomics during red-green iteration when you
	# only want to re-run tests. Still excludes live/integration for speed.
	uv run pytest tests/ -m "not live and not integration"


integration:  ## Run integration tests (requires `make smoke` stack up)
	# Matches §5 spec. Expects docker compose services to be running — the
	# caller is responsible for bringing them up and tearing them down,
	# because the integration suite may want to reuse a warm stack across
	# multiple runs without the ~60s Ollama pull every time.
	uv run pytest tests/ -m integration


live:  ## Run live tests against real LLM APIs (costs pennies; requires RUN_LIVE=1)
	# Matches §5 spec. RUN_LIVE=1 is the opt-in gate — without it, the live
	# tests skip themselves. ANTHROPIC_API_KEY and/or OPENAI_API_KEY must
	# also be set in the environment or .env. Costs ~$0.01 per full run.
	RUN_LIVE=1 uv run pytest tests/ -m live

demo-replay:  ## Replay the two README POST /ask examples offline (no API key)
	# Recorded Anthropic cassettes + in-memory substrate fakes, so a reviewer
	# reproduces the answered-with-citation and refused-with-zero-citations
	# examples with no key, no DB, no network. Record once with:
	#   RUN_LIVE=1 ANTHROPIC_API_KEY=sk-... uv run pytest tests/replay/ --record-mode=once
	uv run pytest tests/replay/ -v

smoke:  ## Bring up docker compose, healthcheck Ollama, tear down
	# Matches §5 spec. Delegates to scripts/smoke.sh which handles the full
	# lifecycle (up --wait, curl /api/tags, cleanup trap). First cold run
	# pulls ~1.5 GB for the Ollama image; subsequent runs are ~10s.
	./scripts/smoke.sh


example:  ## Run the headline composition example (02_fallback_demo.py)
	# Divergence from §5: spec said `python -m examples.hello_llm`. T011
	# shipped three numbered scripts (01/02/03) instead of a single
	# hello_llm package. 02_fallback_demo is the canonical headline demo
	# because it exercises the full composition root (build_llm + FallbackModel).
	uv run python examples/02_fallback_demo.py


example-all-tiers:  ## Run all three example scripts sequentially
	# Divergence from §5: spec mentioned "run example against each tier, diff
	# outputs." A literal diff of LLM outputs is noisy — different tiers will
	# always phrase answers differently, so a text diff surfaces nothing
	# useful. Instead we run all three scripts back-to-back; each self-reports
	# which tier served it via stderr. A `set -e`-style failure on any script
	# fails the whole target. Fits the intent of SPECIFICATION.md OT-8.
	@echo "→ examples/01_single_adapter.py"
	uv run python examples/01_single_adapter.py
	@echo ""
	@echo "→ examples/02_fallback_demo.py"
	uv run python examples/02_fallback_demo.py
	@echo ""
	@echo "→ examples/03_custom_stack.py"
	uv run python examples/03_custom_stack.py


build:  ## Build the wheel with uv
	# Divergence from §5: spec said `python -m build --wheel`. We use `uv
	# build` because uv is the project's unified tool (sync, run, build).
	# Output lands in dist/ — same filesystem location as python-build,
	# so downstream consumers of dist/*.whl see no difference.
	uv build


clean:  ## Remove build artifacts, test caches, and coverage output
	# Matches §5 spec, with minor additions for ruff/coverage caches that
	# the spec predates. Does NOT remove .venv/ — blowing that away forces
	# a full re-sync on the next `make check`, which is surprising. To
	# also drop .venv, run `rm -rf .venv` manually.
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf coverage.xml .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true


parity:  ## Verify pre-commit ruff/mypy/bandit pins match pyproject.toml
	# Matches §5 spec. Same script the pre-commit hook invokes, exposed as a
	# Make target for ad-hoc runs during dep upgrades. Exits non-zero with a
	# clear diff if .pre-commit-config.yaml and pyproject.toml have drifted
	# on any of the three tool versions. See ADR D8.
	uv run python scripts/check_version_parity.py
