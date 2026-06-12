"""Concrete `AnthropicClientPort` --- the real Sonnet call behind the router.

`AnthropicRouterClient` wraps `anthropic.Anthropic` and implements the one
method `run_router` needs (`create_message`). It owns the model id, the system
prompt, and ``max_tokens``; the loop supplies only the per-turn messages, the
tools schema, and the tool-choice payload.

Design notes:

- Single model: Claude Sonnet, id from `Settings.router_model` (env
  ``ROUTER_MODEL``, default ``claude-sonnet-4-6``). One model, no fallback ---
  the router is a routing experiment, not the resilience stack.
- No ``thinking`` parameter. Iteration 0 forces a tool call
  (``tool_choice={"type":"any"}``), which is incompatible with extended
  thinking; omitting it keeps the forced turn valid. Mirrors the existing
  `AnthropicAdapter`, which also sets no thinking.
- Temperature 0. Routing is a *decision*, not creative generation: deterministic
  sampling makes the tool choice and the refusal call reproducible, so an eval run
  is stable and the D6 gate measures the router rather than sampling noise.
- No prompt caching (the 3-tool system prompt is well under the caching floor).
- The kwargs dict is built explicitly and splatted into ``messages.create`` so
  mypy checks our call site without fighting the SDK's heavily-overloaded
  signature --- the same pattern `AnthropicAdapter.generate` uses.
- Construction is fail-fast: a missing API key raises immediately, so a
  misconfigured environment fails at build time rather than on the first call.
"""

from __future__ import annotations

from typing import Any

import anthropic
from pydantic import SecretStr

from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.router.schema import SYSTEM_PROMPT

# Default model + output ceiling. Sonnet per the D4 scope; 1024 tokens is ample
# for a routing answer and keeps non-streaming calls under the SDK's timeout.
DEFAULT_ROUTER_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024

# Deterministic routing: temperature 0 fixes the tool choice and the refusal
# decision so an eval run reproduces and the D6 gate is not chasing sampling noise.
ROUTER_TEMPERATURE = 0.0


class AnthropicRouterClient:
    """`AnthropicClientPort` backed by the Anthropic Python SDK (Claude Sonnet)."""

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str = DEFAULT_ROUTER_MODEL,
        system: str = SYSTEM_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ) -> None:
        """Build the client and its underlying SDK object.

        Parameters
        ----------
        api_key:
            Anthropic API key. ``None`` (the shape `Settings` produces when the
            env var is empty) raises ``ValueError`` immediately --- the router
            never silently proceeds without credentials.
        model:
            Claude model id. Defaults to Sonnet; overridable so a run can pin a
            different model via ``Settings.router_model``.
        system / max_tokens:
            The system prompt and per-response token ceiling baked into every
            turn. The probe lowers ``max_tokens`` to keep routing-only calls
            cheap.
        timeout_seconds / max_retries:
            Passed through to the SDK client (per-request timeout and its
            transient-error retry budget).
        """
        if api_key is None:
            msg = (
                "AnthropicRouterClient requires an API key; "
                "set ANTHROPIC_API_KEY in the environment or .env."
            )
            raise ValueError(msg)

        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key.get_secret_value(),
            timeout=float(timeout_seconds),
            max_retries=max_retries,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> AnthropicRouterClient:
        """Build from `Settings` --- the wiring the probe and live tests use."""
        return cls(
            api_key=settings.anthropic_api_key,
            model=settings.router_model,
            max_tokens=max_tokens,
        )

    def create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, str],
    ) -> Any:
        """Dispatch one Messages-API turn and return the raw SDK response."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "system": self._system,
            "max_tokens": self._max_tokens,
            "temperature": ROUTER_TEMPERATURE,
            "tools": tools,
            "tool_choice": tool_choice,
            "messages": messages,
        }
        return self._client.messages.create(**kwargs)
