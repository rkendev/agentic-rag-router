"""Example 02 — full three-tier FallbackModel via the composition root.

Uses `build_llm(settings)` to assemble whichever tiers have their
preconditions met: cloud tiers only if their API key is set, local
Ollama always appended. The `LLMResponse.tier` field on the returned
value tells you which tier actually served the request — critical for
production observability.

Run from the repository root:

    uv run python examples/02_fallback_demo.py

Exits 0 on any successful completion (regardless of which tier served
it). Exits 2 on an unrecoverable error after logging which tier raised.

To actually see a tier hop, temporarily break the primary tier
(e.g. `export ANTHROPIC_API_KEY=sk-wrong` to force a 401, or point
`ANTHROPIC_MODEL` at an unknown model) and re-run — the response will
come back tagged with `tier=secondary` or `tier=tertiary` depending on
what else is configured.
"""

from __future__ import annotations

import sys

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.main import build_llm

PROMPT = "Give me three reasons to unit-test the adapter layer."


def main() -> int:
    settings = Settings()

    # Force the full fallback stack regardless of whatever LLM_TIER the
    # caller has in their .env — this example is specifically about
    # demonstrating composition.
    settings = settings.model_copy(update={"llm_tier": "fallback"})

    # `build_llm` is the composition root. It inspects the settings and
    # returns either a single adapter or a `FallbackModel` composing
    # every tier whose credentials are present. Either way the return
    # type is `LLMPort` and the call site is identical.
    llm = build_llm(settings)

    try:
        response = llm.generate(PROMPT)
    except LLMTransientError as exc:
        # This path only fires if *every* configured tier failed with a
        # recoverable error — if you hit it in practice, your cloud
        # providers are both down and your Ollama daemon is unreachable.
        print(f"All tiers exhausted with transient failures: {exc}", file=sys.stderr)
        return 2
    except LLMPermanentError as exc:
        # One tier raised a non-retryable error. FallbackModel intentionally
        # re-raises instead of failing over because the same credentials /
        # request shape would fail everywhere.
        print(f"Permanent failure (no fail-over): {exc}", file=sys.stderr)
        return 2
    except LLMContentError as exc:
        print(f"Content refused by the model: {exc}", file=sys.stderr)
        return 2

    served_by = response.tier.value
    print(response.text)
    print(
        f"\n[served by: tier={served_by} "
        f"model={response.model_name} "
        f"tokens_out={response.tokens_out} "
        f"latency_ms={response.latency_ms}]",
        file=sys.stderr,
    )
    if served_by != "primary":
        print(
            f"(Note: the PRIMARY tier declined or was not configured; "
            f"response came from {served_by}.)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
