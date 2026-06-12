# Examples

Three runnable walkthroughs of the template, ordered by increasing composition complexity. Each script is self-contained and exits cleanly (with a message to stderr) when required credentials are missing.

## Prerequisites

```bash
uv sync --all-extras
cp .env.example .env
# Edit .env and set at least ANTHROPIC_API_KEY (for 01 and 02) or
# OPENAI_API_KEY (for 03). Leave the others blank to keep them disabled.
```

Run all examples from the repository root so `.env` loads correctly.

## 01 — `01_single_adapter.py` — call one adapter directly

The minimum API surface. Constructs `AnthropicAdapter` with a `Settings`-derived API key and calls `generate()`. No fail-over: if the primary tier fails, you see the error. Good for debugging a specific tier or for library code that deliberately wants single-tier behaviour.

```bash
uv run python examples/01_single_adapter.py
```

**Needs:** `ANTHROPIC_API_KEY` set.

## 02 — `02_fallback_demo.py` — full three-tier FallbackModel

Uses `build_llm(settings)` — the composition root — to assemble every tier whose preconditions are met (cloud tiers if keys set, Ollama always). Prints the response plus a note telling you which tier actually served it.

```bash
uv run python examples/02_fallback_demo.py
```

To see a tier hop in action, temporarily sabotage the primary:

```bash
# Force a 401 from Anthropic, then watch SECONDARY or TERTIARY serve the request.
ANTHROPIC_API_KEY=sk-wrong uv run python examples/02_fallback_demo.py
```

**Needs:** at least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or a running local Ollama daemon.

## 03 — `03_custom_stack.py` — hand-built compositional stack

Builds a **cloud-only, cost-ordered** `FallbackModel`: gpt-4o-mini first, Claude Haiku as backup. No Ollama. Shows that `FallbackModel` is just a consumer of an ordered `list[LLMPort]` — any subset, any order, any user-written `LLMPort` wrapper is fair game.

```bash
uv run python examples/03_custom_stack.py
```

**Needs:** at least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Cost notes

Each run of 01 and 02 on their default tier costs a fraction of a cent (well under $0.01 per invocation at claude-haiku-4-5 and gpt-4o-mini prices as of April 2026). Ollama is free and runs offline. Running all three examples back-to-back is effectively free.

If you want to run the examples without touching a paid API at all, leave both cloud keys blank and run only `02_fallback_demo.py` — it will fall through to the local Ollama tier, provided the daemon is up and `llama3.2:3b` has been pulled. See the main README's Ollama section for setup.

## When to pick each tier directly

- **Primary only (`LLM_TIER=primary`):** You want the best default quality and are willing to surface provider outages as errors rather than silently degrade.
- **Fallback (`LLM_TIER=fallback`):** You want the template's built-in resilience story — cloud-first with a local safety net. This is the default.
- **Tertiary only (`LLM_TIER=tertiary`):** Offline dev, CI that can't hit external APIs, or cost-sensitive batch work where local-model quality is acceptable.
- **Custom stack (via `examples/03`):** Any of the above doesn't quite fit. The `FallbackModel([...])` constructor takes any ordered sequence of `LLMPort` — build what you need.
