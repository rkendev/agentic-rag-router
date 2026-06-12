"""Domain exception hierarchy for the LLM layer.

Adapters translate SDK-specific errors (e.g. `anthropic.APIStatusError`,
`openai.APIError`, `httpx.ConnectError`) into one of the three domain
subclasses below. `FallbackModel` relies on the transient-vs-permanent
split to decide whether to try the next tier.

See SPECIFICATION.md §2.3 and §7 (failure-mode contract).
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every LLM-layer failure.

    Catch this in callers that want to handle any LLM failure generically.
    Adapters and `FallbackModel` both raise one of the three subclasses
    defined below — never raise `LLMError` directly.
    """


class LLMTransientError(LLMError):
    """Recoverable failure — safe to retry or fail over to the next tier.

    Examples: HTTP 429 (rate limit), 5xx server errors, network timeouts,
    connection-refused against a local Ollama host. `FallbackModel` logs
    and advances to the next tier on this class.
    """


class LLMPermanentError(LLMError):
    """Unrecoverable failure — do not retry, do not fail over.

    Examples: HTTP 4xx (non-rate-limit) — invalid API key, bad request,
    unknown model string, missing required env var. Falling over to the
    next tier would not help (credentials or request shape are wrong)
    and could silently mask a configuration error, so `FallbackModel`
    re-raises immediately.
    """


class LLMContentError(LLMError):
    """The call succeeded but the content is unusable.

    Examples: empty response body, content-policy refusal, malformed
    function-call payload. `FallbackModel` treats this as permanent —
    the same prompt will likely trigger the same refusal on the next
    tier, and silently swallowing refusals would violate the observability
    contract (the caller needs to know the prompt was rejected).
    """
