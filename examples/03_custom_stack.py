"""Example 03 — hand-build a custom tier stack.

`main.build_llm` is one way to compose a stack. When you want a
different priority order, a subset of tiers, or your own `LLMPort`
wrapper in the chain, build the `FallbackModel` directly — the
composition root is not privileged.

This example builds a **cloud-only, cost-ordered** stack: gpt-4o-mini
first (cheapest for simple prompts), Claude Haiku as the backup. Ollama
is intentionally excluded — some workloads would rather fail loud than
fall back to a local model whose output is materially different.

Run from the repository root:

    uv run python examples/03_custom_stack.py

The example skips gracefully if neither cloud key is set.
"""

from __future__ import annotations

import sys

from agentic_rag_router.application.fallback import FallbackModel
from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.errors import LLMError
from agentic_rag_router.infrastructure.anthropic_adapter import AnthropicAdapter
from agentic_rag_router.infrastructure.openai_adapter import OpenAIAdapter
from agentic_rag_router.infrastructure.settings import Settings

PROMPT = "Name one advantage of protocols over abstract base classes in Python."


def main() -> int:
    settings = Settings()

    tiers: list[LLMPort] = []

    # Build in cost order for this demo: cheapest-first is the inverse
    # of the default quality-first order in main.build_llm. Either is
    # valid — FallbackModel doesn't care about tier labels, only the
    # sequence you hand it.
    if settings.openai_api_key is not None:
        tiers.append(
            OpenAIAdapter(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        )
    if settings.anthropic_api_key is not None:
        tiers.append(
            AnthropicAdapter(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        )

    if not tiers:
        print(
            "Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set — this "
            "example needs at least one cloud key. Set one in .env and "
            "re-run.",
            file=sys.stderr,
        )
        return 1

    # FallbackModel itself satisfies LLMPort, so it can be nested inside
    # another FallbackModel (e.g. priority groups) — useful when you
    # want "try all cheap tiers, then all expensive tiers" semantics.
    llm = FallbackModel(tiers)

    try:
        response = llm.generate(PROMPT)
    except LLMError as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(response.text)
    print(
        f"\n[served by: tier={response.tier.value} "
        f"model={response.model_name} "
        f"latency_ms={response.latency_ms}] "
        f"(stack: {[type(t).__name__ for t in tiers]})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
