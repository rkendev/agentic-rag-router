"""OpenAI (gpt-4o-mini) adapter --- SECONDARY tier of the fallback stack.

Implements `LLMPort` by wrapping `openai.OpenAI.chat.completions.create`
and translating the SDK's exception hierarchy into the three domain
error classes (`LLMTransientError` / `LLMPermanentError` / `LLMContentError`)
that `FallbackModel` knows how to route.

Truth table (see docs/DECISIONS.md D3):

- `APIConnectionError` (incl. `APITimeoutError`) -> transient
- `RateLimitError` (429)                         -> transient
- `InternalServerError` (5xx)                    -> transient
- `AuthenticationError` (401)                    -> permanent
- `PermissionDeniedError` (403)                  -> permanent
- `NotFoundError` (404, model missing)           -> permanent
- `ConflictError` (409)                          -> permanent
- `BadRequestError` (400)                        -> permanent
- `UnprocessableEntityError` (422)               -> permanent
- Other `APIStatusError`: classify by status_code (5xx -> transient)
- Any other `APIError`                           -> permanent (safety net)

Design notes (shared with `anthropic_adapter`, reiterated for local context):

- Optional SDK params (`temperature`) are added to the kwargs dict only
  when the caller supplied a value --- avoids the `NotGiven` / `Omit`
  sentinel drift across SDK versions and lets the SDK apply its own
  defaults for omitted keys.
- The OpenAI chat API takes `system` as a message with role `"system"`,
  NOT as a top-level kwarg like Anthropic. We prepend it to `messages`
  when the caller provides it.
- `messages: list[ChatCompletionMessageParam]` narrows the inline
  `{"role": "user", ...}` literal correctly for mypy strict mode.
- Exception order matters: the SDK uses inheritance (`RateLimitError`
  subclasses `APIStatusError` subclasses `APIError`). Specific handlers
  must precede general ones. The final `APIStatusError` branch catches
  subclasses we haven't enumerated and classifies them by status_code,
  so new SDK exception types land on the right side of the transient
  vs permanent split automatically.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import openai
import structlog
from openai.types.chat import ChatCompletionMessageParam
from pydantic import SecretStr

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier

_logger = structlog.get_logger(__name__)

_REFUSAL_FINISH_REASON = "content_filter"


class OpenAIAdapter:
    """LLMPort implementation backed by the OpenAI Python SDK.

    Construction is fail-fast: a missing API key raises `LLMPermanentError`
    immediately so the composition root catches the misconfiguration
    before any request is attempted (per D4). The underlying
    `openai.OpenAI` client is built once and reused across calls; its
    own retry / timeout settings are configured from the adapter's
    constructor args so operators tune behaviour via `Settings` rather
    than code.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        timeout_seconds: int = 30,
        max_retries: int = 2,
        default_max_tokens: int = 1024,
    ) -> None:
        """Build the adapter and its underlying SDK client.

        Parameters
        ----------
        api_key:
            OpenAI API key. `None` (the shape produced by `Settings`
            when the env var is empty) raises `LLMPermanentError` at
            construction --- adapters never silently proceed without
            credentials.
        model:
            OpenAI chat-completion model identifier (e.g.
            `"gpt-4o-mini"`). Configurable via `Settings.openai_model`
            so template forks can swap models without touching code.
        timeout_seconds:
            Per-request timeout passed to the SDK. The SDK converts
            this into an `httpx` timeout internally.
        max_retries:
            SDK-level retry budget for transient HTTP errors. Retries
            inside the SDK are distinct from tier fail-over in
            `FallbackModel` --- the SDK retries the same tier, the
            `FallbackModel` moves on.
        default_max_tokens:
            Used when the caller passes `max_tokens=None`. The OpenAI
            API treats `max_tokens` as optional, but setting a ceiling
            matches the Anthropic adapter's behaviour and keeps costs
            predictable for template users.
        """
        if api_key is None:
            msg = (
                "OpenAIAdapter requires an API key; "
                "set OPENAI_API_KEY in the environment or .env."
            )
            raise LLMPermanentError(msg)

        self._model = model
        self._default_max_tokens = default_max_tokens
        self._client = openai.OpenAI(
            api_key=api_key.get_secret_value(),
            timeout=float(timeout_seconds),
            max_retries=max_retries,
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Dispatch a single-turn prompt to gpt-4o-mini.

        See `LLMPort.generate` for the full contract. Raises the three
        domain exception subclasses exclusively --- no SDK exception
        ever escapes.
        """
        if not prompt.strip():
            msg = "prompt must not be empty or whitespace-only"
            raise ValueError(msg)

        effective_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        log = _logger.bind(
            adapter="OpenAIAdapter",
            model=self._model,
            max_tokens=effective_max_tokens,
        )

        # System prompt is a message in the chat API, NOT a top-level kwarg.
        # When absent, we send user-only messages and let the model respond
        # with no steering beyond its default system prompt.
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
        }
        # Only include optional params when the caller set them; otherwise
        # let the SDK apply its own defaults rather than threading sentinels.
        if temperature is not None:
            create_kwargs["temperature"] = temperature

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**create_kwargs)
        # --- transient: network + server-side ---------------------------
        except openai.APIConnectionError as exc:
            # Subsumes APITimeoutError (subclass).
            log.warning("openai.transient", error_class=type(exc).__name__)
            msg = f"OpenAI connection / timeout error: {exc}"
            raise LLMTransientError(msg) from exc
        except openai.RateLimitError as exc:
            log.warning("openai.transient", error_class=type(exc).__name__)
            msg = f"OpenAI rate limit (429): {exc}"
            raise LLMTransientError(msg) from exc
        except openai.InternalServerError as exc:
            log.warning("openai.transient", error_class=type(exc).__name__)
            msg = f"OpenAI server error (5xx): {exc}"
            raise LLMTransientError(msg) from exc
        # --- permanent: client-side configuration / request -------------
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.ConflictError,
            openai.BadRequestError,
            openai.UnprocessableEntityError,
        ) as exc:
            log.error("openai.permanent", error_class=type(exc).__name__)
            msg = f"OpenAI client error ({type(exc).__name__}): {exc}"
            raise LLMPermanentError(msg) from exc
        # --- unmapped status: classify by code, 5xx -> transient --------
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                log.warning(
                    "openai.transient.unmapped",
                    error_class=type(exc).__name__,
                    status_code=exc.status_code,
                )
                msg = f"OpenAI unmapped 5xx ({exc.status_code}): {exc}"
                raise LLMTransientError(msg) from exc
            log.error(
                "openai.permanent.unmapped",
                error_class=type(exc).__name__,
                status_code=exc.status_code,
            )
            msg = f"OpenAI unmapped 4xx ({exc.status_code}): {exc}"
            raise LLMPermanentError(msg) from exc
        # --- final safety net: any other SDK-native error ---------------
        except openai.APIError as exc:
            log.error("openai.permanent.unknown", error_class=type(exc).__name__)
            msg = f"OpenAI unknown API error ({type(exc).__name__}): {exc}"
            raise LLMPermanentError(msg) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        text = self._extract_text(response)
        return LLMResponse(
            text=text,
            model_name=response.model,
            tier=LLMTier.SECONDARY,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=latency_ms,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text payload out of a chat-completion response.

        Raises `LLMContentError` for any shape we can't turn into usable
        text: content-filter refusal, empty choices list, missing message
        content, or text that is empty / whitespace-only.
        """
        choices = response.choices
        if not choices:
            msg = "OpenAI returned an empty choices list"
            raise LLMContentError(msg)

        first = choices[0]
        if getattr(first, "finish_reason", None) == _REFUSAL_FINISH_REASON:
            msg = "OpenAI refused to generate content " "(finish_reason=content_filter)"
            raise LLMContentError(msg)

        message = getattr(first, "message", None)
        if message is None:
            msg = "OpenAI choice had no message attribute"
            raise LLMContentError(msg)

        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            msg = "OpenAI returned empty or whitespace-only text"
            raise LLMContentError(msg)

        return text
