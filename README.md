# agentic-rag-router

Hexagonal / DDD-lite Python 3.12 project with a three-tier LLM adapter
(Claude Haiku → gpt-4o-mini → Ollama), a parametrized `LLMPort` contract
suite, Docker compose for Ollama, a pinned pre-commit chain, and a CI
workflow — scaffolded from
[`agentic-rag-router`](https://github.com/rkendev/agentic-rag-router).

## Quick start

```bash
# Copy .env.example and fill in any keys you want to exercise.
cp .env.example .env
$EDITOR .env

# Install dependencies (including dev extras — ruff, mypy, pytest, etc.).
uv sync --all-extras

# Install pre-commit's git hook so trailing-whitespace / EOF / line-ending
# auto-fixes fire at commit time. Skipping this means CI catches drift
# (auto-fix hooks aren't part of `make check`).
uv run pre-commit install

# Run the full quality gate: lint + type + security + 219 tests + auto-fix hooks.
make check
```

Need offline Ollama backing? `./scripts/smoke.sh` brings up a
digest-pinned Ollama container and verifies it's healthy.

## What this gives you

A shaped starting point, not a framework. Three layers with a strict
dependency rule (see [`ARCHITECTURE.md`](ARCHITECTURE.md)):

- **`domain/`** — types, invariants, errors. Pure Python; Pydantic is
  the only third-party import allowed.
- **`application/`** — ports (`LLMPort`, `ConfigPort`, `LoggerPort`)
  and the `FallbackModel` orchestrator. Depends on `domain/` only.
- **`infrastructure/`** — SDK adapters (Anthropic, OpenAI, Ollama) and
  the `pydantic-settings` loader. Only layer that imports vendor SDKs
  or reads the environment.
- **`main.py`** — the single composition root. `build_llm(settings)`
  wires a single adapter or a `FallbackModel` stack depending on
  `LLM_TIER`.

The 32-case contract suite in `tests/contract/` is the architectural
drift detector: any new adapter registered with
`tests/contract/conftest.py::LLM_ADAPTERS` inherits eight behavioural
assertions automatically — vendor-tagged failures
(`test_returns_response[anthropic]`) pinpoint which implementation
drifted, not which test broke.

## Make targets

Run `make help` for the full list. The core surface:

| Target | What it does |
| --- | --- |
| `check` | ruff + ruff-format + mypy + bandit + 219 unit/contract tests. The default quality gate. |
| `fmt` | Auto-fix formatting with ruff. |
| `lint` | ruff lint only (no format pass). |
| `typecheck` | mypy strict on `src/` + `tests/`. |
| `security` | bandit -ll on `src/`. |
| `test` | pytest unit + contract, with coverage. |
| `integration` | pytest -m integration (requires docker-compose; skips if empty). |
| `smoke` | `./scripts/smoke.sh` — docker compose up + healthcheck for Ollama. |
| `build` | `uv build` — sdist + wheel. |
| `parity` | `scripts/check_version_parity.py` — asserts ruff/mypy/bandit pins match between `pyproject.toml` and `.pre-commit-config.yaml`. |
| `example-all-tiers` | Run all three examples back-to-back (needs API keys for cloud tiers). |

## Example usage

Three runnable scripts in `examples/` show the composition root from the
outside:

```bash
# Single adapter (Claude Haiku) — needs ANTHROPIC_API_KEY.
uv run python examples/01_single_adapter.py

# Fallback stack — uses whichever tier's credentials are present.
uv run python examples/02_fallback_demo.py

# Custom stack — secondary (OpenAI) only; demonstrates how to wire a
# subset of tiers manually.
uv run python examples/03_custom_stack.py
```

Each script prints the completion on stdout and a `[tier=... model=... ]`
metadata line on stderr so pipelines can consume `.text` cleanly.

Offline-only? Force the tertiary tier:

```bash
LLM_TIER=tertiary uv run python -m agentic_rag_router.main \
  "Say hi in one sentence."
```

No API key required; runs entirely against local Ollama.

## Configuration

All runtime configuration lives in `.env` (loaded by
`infrastructure/settings.py`). Variables:

| Var | Default | Purpose |
| --- | --- | --- |
| `LLM_TIER` | `fallback` | `primary` / `secondary` / `tertiary` / `fallback`. |
| `ANTHROPIC_API_KEY` | (unset) | Enables primary tier. |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Override model. |
| `OPENAI_API_KEY` | (unset) | Enables secondary tier. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Override model. |
| `OLLAMA_HOST` | `http://localhost:11434` | Where to find Ollama. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Override model. |

Empty strings coerce to `None` so a `.env` placeholder doesn't silently
become a zero-length API key.

## Verification

Every architectural claim is paired with a runnable command in
[`VERIFICATION.md`](VERIFICATION.md). OT-2 (`LLMPort` contract
conformance), OT-3 (pre-commit parity), OT-4 (Docker healthcheck), OT-7
(offline Ollama), OT-8 (all three tiers end-to-end), OT-9 (wheel build),
and OT-10 (bandit clean) each take one line to re-verify.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the Mermaid dependency
graph and the extension recipes (adding a tier, adding an unrelated
port). The short version: `domain/` knows nothing; `application/` knows
`domain/`; `infrastructure/` knows both; `main.py` knows all three and
is the only place allowed to wire them together.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) — Keep-a-Changelog 1.1.0 format. The
template's own release notes (`v0.1.0`, `v0.2.0`) are trimmed from the
fork's changelog so `[Unreleased]` is what you edit.

## License

MIT — see [`LICENSE`](LICENSE).
