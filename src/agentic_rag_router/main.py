"""Composition root.

This is the only module allowed to import across all three architectural
layers: `domain`, `application`, and `infrastructure`. Every other module
observes the dependency rule (domain knows nothing, application depends
on domain, infrastructure depends on domain + application). Centralising
the cross-layer wiring here keeps the rest of the codebase testable in
isolation.

Wiring is driven by `Settings.llm_tier`:

- `"primary"` / `"secondary"` / `"tertiary"` return a single adapter
  directly, bypassing fail-over. This matches the contract documented
  in `.env.example` --- use a single tier when the caller wants
  diagnostic transparency (no retries, no tier hop) rather than
  resilience.
- `"fallback"` wraps every available tier in `FallbackModel` for the
  full transient-failure-to-next-tier behaviour (see D2). Cloud-hosted
  adapters (Anthropic, OpenAI) are skipped when their API key is
  absent --- a partial environment (e.g. only an Anthropic key) still
  gets a working fallback rather than failing loudly. Ollama is local
  and has no credentials, so it is always appended; if the daemon
  isn't running, the call surfaces as an `LLMTransientError` at the
  tail of the cascade instead of at construction time.

Scope today: T007a, T008, and T009 shipped all three adapters.
`"primary"` -> Claude Haiku. `"secondary"` -> gpt-4o-mini.
`"tertiary"` -> local Ollama. `"fallback"` composes every tier whose
preconditions are met: the cloud tiers when keys are set, plus Ollama
unconditionally as the always-free last-resort.
"""

from __future__ import annotations

import sys

from agentic_rag_router.application.fallback import FallbackModel
from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.infrastructure.anthropic_adapter import AnthropicAdapter
from agentic_rag_router.infrastructure.ollama_adapter import OllamaAdapter
from agentic_rag_router.infrastructure.openai_adapter import OpenAIAdapter
from agentic_rag_router.infrastructure.settings import Settings


def build_llm(settings: Settings) -> LLMPort:
    """Compose the `LLMPort` instance selected by `settings.llm_tier`.

    Parameters
    ----------
    settings:
        Already-validated `Settings` instance (constructed by the caller
        so tests can inject without touching env / .env).

    Returns
    -------
    LLMPort
        Either a concrete adapter (`primary` / `secondary` / `tertiary`)
        or a `FallbackModel` (`fallback`). Both satisfy the `LLMPort`
        protocol, so the caller never needs to distinguish.

    Raises
    ------
    LLMPermanentError
        Propagated from an adapter constructor when required credentials
        are missing for the selected tier (fail-fast per D4). Ollama
        does not raise here --- it has no credentials to fail on.
    """
    tier = settings.llm_tier

    if tier == "primary":
        return _build_anthropic(settings)

    if tier == "secondary":
        return _build_openai(settings)

    if tier == "tertiary":
        return _build_ollama(settings)

    # tier == "fallback" --- build every tier whose preconditions are met.
    # Order matters for fail-over: Anthropic first (primary), then OpenAI
    # (secondary), then Ollama (tertiary). Ollama is appended
    # unconditionally because it has no credentials to gate on --- a
    # missing daemon surfaces at the tail as a transient error, which is
    # the honest runtime signal.
    tiers: list[LLMPort] = []
    if settings.anthropic_api_key is not None:
        tiers.append(_build_anthropic(settings))
    if settings.openai_api_key is not None:
        tiers.append(_build_openai(settings))
    tiers.append(_build_ollama(settings))

    return FallbackModel(tiers)


def _build_anthropic(settings: Settings) -> AnthropicAdapter:
    """Construct the Anthropic adapter with settings-driven knobs."""
    return AnthropicAdapter(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _build_openai(settings: Settings) -> OpenAIAdapter:
    """Construct the OpenAI adapter with settings-driven knobs."""
    return OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _build_ollama(settings: Settings) -> OllamaAdapter:
    """Construct the Ollama adapter with settings-driven knobs.

    Note: `max_retries` is intentionally not threaded through --- the
    ollama SDK doesn't expose a client-level retry budget, and retrying
    a local daemon that isn't responding won't make it respond.
    `FallbackModel` handles the tier-level fail-over instead.
    """
    return OllamaAdapter(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def main() -> None:  # pragma: no cover
    """Smoke-test entry point: print a completion for an argv-supplied prompt.

    Usage::

        uv run python -m agentic_rag_router.main "Say hi in one sentence."

    With no argv, a default prompt runs. Credentials for the selected
    tier must be present in the environment or `.env`; a missing key
    exits with `LLMPermanentError`. The Ollama tier has no credentials
    but needs the daemon reachable at `OLLAMA_HOST`.

    Excluded from coverage because the `.generate()` call hits a live
    API --- the unit tests cover `build_llm` itself with fake keys.
    """
    default_prompt = "Introduce yourself in one sentence."
    prompt = sys.argv[1] if len(sys.argv) > 1 else default_prompt

    settings = Settings()
    llm = build_llm(settings)
    response = llm.generate(prompt)

    # Stdout = the completion itself; stderr = metadata so pipelines can
    # cleanly consume the text without parsing out a banner.
    print(response.text)
    print(
        f"[tier={response.tier.value} model={response.model_name} "
        f"tokens_in={response.tokens_in} tokens_out={response.tokens_out} "
        f"latency_ms={response.latency_ms}]",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
