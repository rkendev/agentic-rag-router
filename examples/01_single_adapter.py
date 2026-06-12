"""Example 01 — call a single adapter directly.

The simplest possible use of the template. No fail-over, no composition:
just construct one adapter and call `generate()`. Good for debugging a
specific tier, or for library code that doesn't want retries to hop
between providers.

Run from the repository root:

    uv run python examples/01_single_adapter.py

Reads `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` from `.env` or the shell
environment. If the key is missing, prints a clear message and exits 1.
"""

from __future__ import annotations

import sys

from agentic_rag_router.domain.errors import LLMError
from agentic_rag_router.infrastructure.anthropic_adapter import AnthropicAdapter
from agentic_rag_router.infrastructure.settings import Settings

PROMPT = "In one sentence, what is a hexagonal architecture?"


def main() -> int:
    settings = Settings()

    if settings.anthropic_api_key is None:
        print(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill "
            "in your key, or `export ANTHROPIC_API_KEY=sk-ant-...`.",
            file=sys.stderr,
        )
        return 1

    # Direct construction. `AnthropicAdapter` itself satisfies `LLMPort`,
    # so downstream code that takes an `LLMPort` can consume it directly
    # without wrapping it in `FallbackModel`.
    llm = AnthropicAdapter(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    try:
        response = llm.generate(PROMPT)
    except LLMError as exc:
        # Every adapter-layer failure is one of LLMTransientError /
        # LLMPermanentError / LLMContentError. They all inherit from
        # LLMError, so a single catch suffices when the caller doesn't
        # need to distinguish.
        print(f"LLM call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(response.text)
    print(
        f"\n[tier={response.tier.value} "
        f"model={response.model_name} "
        f"tokens_in={response.tokens_in} "
        f"tokens_out={response.tokens_out} "
        f"latency_ms={response.latency_ms}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
