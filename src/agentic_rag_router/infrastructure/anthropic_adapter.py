"""Anthropic (Claude Haiku) adapter --- PRIMARY tier of the fallback stack.

Implements `LLMPort` by wrapping `anthropic.Anthropic.messages.create` and
translating the SDK's exception hierarchy into the three domain error
classes (`LLMTransientError` / `LLMPermanentError` / `LLMContentError`)
that `FallbackModel` knows how to route.

See docs/DECISIONS.md D3 (error-classification contract) and the T007
handoff doc for the full truth table. Summary:

- Network and server-side failures (timeout, connection, 429, 5xx) are
  transient --- the next tier is likely to succeed.
- Client-side failures (auth, permission, 404, 400, 422) are permanent
  --- the same credentials or payload would fail elsewhere too.
- Successful responses with empty / whitespace-only text or a refusal
  stop_reason are content errors --- the model didn't produce usable
  output and retrying the same prompt elsewhere would likely repeat.

Design notes:

- Optional SDK params (`system`, `temperature`) are added to the kwargs
  dict only when the caller supplied a value. This sidesteps the
  SDK-evolving sentinel (`NotGiven` vs `Omit` across versions) and
  lets the SDK apply its own defaults when we omit the key entirely.
- `max_tokens` is REQUIRED by Anthropic's API; the port allows `None`
  meaning "adapter default". `default_max_tokens=1024` is a
  conservative ceiling for short-form template use and is overridable
  at construction.
- Exception order below is meaningful: the SDK uses inheritance
  (`RateLimitError` subclasses `APIStatusError` which subclasses
  `APIError`), so specific handlers must precede general ones. The
  final `APIStatusError` branch classifies by `status_code` to catch
  any subclass the SDK adds in the future that we haven't enumerated.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import anthropic
import structlog
from anthropic.types import MessageParam
from pydantic import SecretStr

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier

_logger = structlog.get_logger(__name__)

_TEXT_BLOCK_TYPE = "text"
_REFUSAL_STOP_REASON = "refusal"


class AnthropicAdapter:
    """LLMPort implementation backed by the Anthropic Python SDK.

    Construction is fail-fast: a missing API key raises `LLMPermanentError`
    immediately so the composition root catches the misconfiguration
    before any request is attempted (per D4). The underlying
    `anthropic.Anthropic` client is built once and reused across calls;
    its own retry / timeout settings are configured from the adapter's
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
            Anthropic API key. `None` (the shape produced by
            `Settings` when the env var is empty) raises
            `LLMPermanentError` at construction --- adapters never
            silently proceed without credentials.
        model:
            Anthropic model identifier (e.g.
            `"claude-haiku-4-5-20251001"`). Configurable via
            `Settings.anthropic_model` so template forks can swap
            models without touching code.
        timeout_seconds:
            Per-request timeout passed to the SDK. The SDK converts
            this into an `httpx` timeout internally.
        max_retries:
            SDK-level retry budget for transient HTTP errors. Retries
            inside the SDK are distinct from tier fail-over in
            `FallbackModel` --- the SDK retries the same tier, the
            `FallbackModel` moves on.
        default_max_tokens:
            Used when the caller passes `max_tokens=None`. Required by
            Anthropic's API; the port contract lets the adapter choose.
        """
        if api_key is None:
            msg = (
                "AnthropicAdapter requires an API key; "
                "set ANTHROPIC_API_KEY in the environment or .env."
            )
            raise LLMPermanentError(msg)

        self._model = model
        self._default_max_tokens = default_max_tokens
        self._client = anthropic.Anthropic(
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
        """Dispatch a single-turn prompt to Claude Haiku.

        See `LLMPort.generate` for the full contract. Raises the three
        domain exception subclasses exclusively --- no SDK exception
        ever escapes.
        """
        if not prompt.strip():
            msg = "prompt must not be empty or whitespace-only"
            raise ValueError(msg)

        effective_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        log = _logger.bind(
            adapter="AnthropicAdapter",
            model=self._model,
            max_tokens=effective_max_tokens,
        )

        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
        }
        # Only add optional params when the caller set them; otherwise let
        # the SDK apply its own defaults rather than threading sentinels.
        if system is not None:
            create_kwargs["system"] = system
        if temperature is not None:
            create_kwargs["temperature"] = temperature

        start = time.perf_counter()
        try:
            response = self._client.messages.create(**create_kwargs)
        # --- transient: network + server-side ---------------------------
        except anthropic.APIConnectionError as exc:
            # Subsumes APITimeoutError (subclass).
            log.warning("anthropic.transient", error_class=type(exc).__name__)
            msg = f"Anthropic connection / timeout error: {exc}"
            raise LLMTransientError(msg) from exc
        except anthropic.RateLimitError as exc:
            log.warning("anthropic.transient", error_class=type(exc).__name__)
            msg = f"Anthropic rate limit (429): {exc}"
            raise LLMTransientError(msg) from exc
        except anthropic.InternalServerError as exc:
            log.warning("anthropic.transient", error_class=type(exc).__name__)
            msg = f"Anthropic server error (5xx): {exc}"
            raise LLMTransientError(msg) from exc
        # --- permanent: client-side configuration / request -------------
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
            anthropic.UnprocessableEntityError,
        ) as exc:
            log.error("anthropic.permanent", error_class=type(exc).__name__)
            msg = f"Anthropic client error ({type(exc).__name__}): {exc}"
            raise LLMPermanentError(msg) from exc
        # --- unmapped status: classify by code, 5xx -> transient --------
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                log.warning(
                    "anthropic.transient.unmapped",
                    error_class=type(exc).__name__,
                    status_code=exc.status_code,
                )
                msg = f"Anthropic unmapped 5xx ({exc.status_code}): {exc}"
                raise LLMTransientError(msg) from exc
            log.error(
                "anthropic.permanent.unmapped",
                error_class=type(exc).__name__,
                status_code=exc.status_code,
            )
            msg = f"Anthropic unmapped 4xx ({exc.status_code}): {exc}"
            raise LLMPermanentError(msg) from exc
        # --- final safety net: any other SDK-native error ---------------
        except anthropic.APIError as exc:
            log.error("anthropic.permanent.unknown", error_class=type(exc).__name__)
            msg = f"Anthropic unknown API error ({type(exc).__name__}): {exc}"
            raise LLMPermanentError(msg) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        text = self._extract_text(response)
        return LLMResponse(
            text=text,
            model_name=response.model,
            tier=LLMTier.PRIMARY,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            latency_ms=latency_ms,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text payload out of a Messages API response.

        Raises `LLMContentError` for any shape we can't turn into usable
        text: refusal stop_reason, empty content list, first block that
        isn't a text block, or text that is empty / whitespace-only.
        """
        if getattr(response, "stop_reason", None) == _REFUSAL_STOP_REASON:
            msg = "Anthropic refused to generate content (stop_reason=refusal)"
            raise LLMContentError(msg)

        content = response.content
        if not content:
            msg = "Anthropic returned an empty content list"
            raise LLMContentError(msg)

        first = content[0]
        if getattr(first, "type", None) != _TEXT_BLOCK_TYPE:
            msg = (
                f"Anthropic returned a non-text first block "
                f"(type={getattr(first, 'type', '?')!r})"
            )
            raise LLMContentError(msg)

        text = first.text
        if not isinstance(text, str) or not text.strip():
            msg = "Anthropic returned empty or whitespace-only text"
            raise LLMContentError(msg)

        return text
