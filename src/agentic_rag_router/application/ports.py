"""Port protocols for the application layer.

A port is a contract: infrastructure adapters implement it, application
orchestrators consume it. Using `typing.Protocol` (PEP 544) over an abstract
base class keeps the domain free of inheritance coupling to application code
and lets adapters conform structurally without importing this module.

Using a Protocol rather than an ABC is deliberate, as noted above.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentic_rag_router.domain.llm import LLMResponse


@runtime_checkable
class LLMPort(Protocol):
    """Contract every LLM adapter must satisfy.

    Adapters translate this signature into their SDK-specific call
    (`anthropic.messages.create`, `openai.chat.completions.create`,
    `ollama.chat`) and always return a domain `LLMResponse`. On failure
    they raise one of the three subclasses in `domain.errors` --- never
    an SDK-native exception. This is what lets `FallbackModel` route
    transient failures to the next tier without knowing anything about
    which vendor raised the error.

    `@runtime_checkable` enables `isinstance(obj, LLMPort)` for tests and
    defensive composition, but note that runtime checks only verify method
    *presence*, not signature --- mypy remains the primary contract
    enforcer.
    """

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Parameters
        ----------
        prompt:
            User-facing prompt text. Must be non-empty. Empty prompts are a
            programming error; adapters may raise `ValueError` or an
            `LLMPermanentError` subclass.
        system:
            Optional system prompt. Adapters whose underlying model does
            not natively accept a system role (some Ollama models) should
            prepend this to the prompt before dispatching.
        max_tokens:
            Upper bound on generated tokens. `None` means the adapter's
            default (which each adapter documents in its module docstring).
        temperature:
            Sampling temperature in `[0.0, 2.0]`. `None` means adapter
            default. Validation of the range is the adapter's
            responsibility; the port intentionally does not constrain it.

        Returns
        -------
        LLMResponse
            Frozen domain response carrying `text`, `tier`, token counts,
            `latency_ms`, and tz-aware `created_at`.

        Raises
        ------
        LLMTransientError
            Recoverable upstream failure --- safe to fail over to the next
            tier. Examples: HTTP 429, 5xx, network timeout.
        LLMPermanentError
            Unrecoverable --- do not retry, do not fail over. Examples:
            HTTP 4xx (auth, bad request), unknown model string.
        LLMContentError
            Call succeeded but returned unusable content (empty body,
            policy refusal, malformed tool-call payload).
        """
        ...
