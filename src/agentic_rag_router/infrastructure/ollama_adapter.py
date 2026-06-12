"""Ollama (local) adapter --- TERTIARY tier of the fallback stack.

Implements `LLMPort` by wrapping `ollama.Client.chat` and translating the
SDK's exception shape into the three domain error classes
(`LLMTransientError` / `LLMPermanentError` / `LLMContentError`) that
`FallbackModel` knows how to route.

Truth table (see docs/DECISIONS.md D3):

- `httpx.RequestError` (timeouts, connection failures, other transport)
                                                 -> transient
- `ollama.ResponseError` with status >= 500      -> transient
- `ollama.ResponseError` with status 404         -> permanent (model not pulled)
- `ollama.ResponseError` with other 4xx / unknown
                                                 -> permanent

Design notes (how this adapter differs from Anthropic / OpenAI):

- Ollama is local and requires no authentication, so there is no
  fail-fast on a missing API key. The constructor always succeeds; a
  bad host or missing daemon surfaces at first `generate()` as an
  `LLMTransientError`.
- `max_retries` is not part of the constructor signature: the ollama
  Python SDK (0.6.x) does not expose a client-level retry budget, and
  adding a tenacity wrapper here would duplicate `FallbackModel`'s
  tier-level fail-over. A local daemon that can't answer won't start
  answering if we retry in place --- we let `FallbackModel` move on.
- `max_tokens` maps to `options["num_predict"]` --- Ollama's
  Modelfile-parameter convention rather than a top-level kwarg.
- `temperature`, when supplied, is passed via the same `options` dict.
- No refusal / content-filter branch: locally-hosted models don't
  produce structured refusals the way hosted models do. Empty or
  whitespace-only output is still treated as `LLMContentError`.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx
import ollama
import structlog

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier

_logger = structlog.get_logger(__name__)


class OllamaAdapter:
    """LLMPort implementation backed by the Ollama Python SDK.

    The underlying `ollama.Client` is built once and reused across calls;
    its timeout is configurable via `Settings`. The host defaults in
    `Settings` to `http://localhost:11434` (Ollama's ship-default) but
    is overridable so operators can point at a remote Ollama deployment
    without touching code.
    """

    def __init__(
        self,
        *,
        host: str,
        model: str,
        timeout_seconds: int = 30,
        default_max_tokens: int = 1024,
    ) -> None:
        """Build the adapter and its underlying SDK client.

        Parameters
        ----------
        host:
            Base URL of the Ollama HTTP API (e.g.
            ``http://localhost:11434``). Required; the class does not
            default it so the composition root is forced to thread the
            configured value through.
        model:
            Ollama model tag (e.g. ``llama3.2:3b``). Configurable via
            `Settings.ollama_model`.
        timeout_seconds:
            Per-request timeout. Passed to the underlying httpx client
            via the ollama SDK.
        default_max_tokens:
            Used when the caller passes `max_tokens=None`. Maps to
            `num_predict` in Ollama's Modelfile-parameter vocabulary.
        """
        self._host = host
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._client = ollama.Client(
            host=host,
            timeout=float(timeout_seconds),
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Dispatch a single-turn prompt to the local Ollama daemon.

        See `LLMPort.generate` for the full contract. Raises the three
        domain exception subclasses exclusively --- no SDK exception
        ever escapes.
        """
        if not prompt.strip():
            msg = "prompt must not be empty or whitespace-only"
            raise ValueError(msg)

        effective_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        log = _logger.bind(
            adapter="OllamaAdapter",
            host=self._host,
            model=self._model,
            max_tokens=effective_max_tokens,
        )

        # System prompt is prepended as a role="system" message --- Ollama
        # follows the OpenAI-compatible chat shape here, not Anthropic's
        # top-level kwarg.
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Ollama's sampling knobs go inside the `options` dict, matching
        # Modelfile parameter names. `num_predict` is Ollama's equivalent
        # of `max_tokens`; absent means "let the model's default win".
        options: dict[str, Any] = {"num_predict": effective_max_tokens}
        if temperature is not None:
            options["temperature"] = temperature

        start = time.perf_counter()
        try:
            response = self._client.chat(
                model=self._model,
                messages=messages,
                options=options,
                stream=False,
            )
        # --- transient: transport-level failures ------------------------
        # `httpx.RequestError` subsumes `TimeoutException`, `ConnectError`,
        # and every other transport-layer error. They all map to transient
        # because either the daemon isn't running (next tier is the only
        # escape) or the request never reached it.
        except httpx.RequestError as exc:
            log.warning("ollama.transient", error_class=type(exc).__name__)
            msg = f"Ollama transport error ({type(exc).__name__}): {exc}"
            raise LLMTransientError(msg) from exc
        # --- ollama.ResponseError: server returned a non-2xx ------------
        except ollama.ResponseError as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and status >= 500:
                log.warning(
                    "ollama.transient",
                    error_class=type(exc).__name__,
                    status_code=status,
                )
                msg = f"Ollama server error ({status}): {exc}"
                raise LLMTransientError(msg) from exc
            if status == 404:
                # Most common permanent failure: the model hasn't been
                # `ollama pull`-ed on this daemon. Nudge the operator.
                log.error(
                    "ollama.permanent.model_missing",
                    error_class=type(exc).__name__,
                    model=self._model,
                )
                msg = (
                    f"Ollama model not found (404): {exc}. "
                    f"Did you run `ollama pull {self._model}`?"
                )
                raise LLMPermanentError(msg) from exc
            log.error(
                "ollama.permanent",
                error_class=type(exc).__name__,
                status_code=status,
            )
            msg = f"Ollama client error (status={status}): {exc}"
            raise LLMPermanentError(msg) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        text = self._extract_text(response)
        # `response.model` is Optional[str] in the SDK's pydantic model;
        # `getattr(..., default)` only returns the default when the attr
        # is ABSENT, not when it's present-but-None. The `or` idiom falls
        # back to the configured model for either case (plus empty string,
        # which would also violate LLMResponse's min_length=1 on model_name).
        response_model = getattr(response, "model", None) or self._model
        return LLMResponse(
            text=text,
            model_name=response_model,
            tier=LLMTier.TERTIARY,
            tokens_in=_coerce_int(response, "prompt_eval_count"),
            tokens_out=_coerce_int(response, "eval_count"),
            latency_ms=latency_ms,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text payload out of an Ollama ChatResponse.

        Raises `LLMContentError` for any shape we can't turn into usable
        text: missing message, missing content, empty / whitespace-only
        string. Ollama models don't produce structured refusals (no
        equivalent of Anthropic's ``stop_reason=refusal`` or OpenAI's
        ``finish_reason=content_filter``), so empty output IS the signal
        that something went wrong content-wise.
        """
        message = getattr(response, "message", None)
        if message is None:
            msg = "Ollama response had no message attribute"
            raise LLMContentError(msg)

        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            msg = "Ollama returned empty or whitespace-only text"
            raise LLMContentError(msg)

        return text


def _coerce_int(response: Any, attr: str) -> int:
    """Read an int attribute off the SDK response, coercing ``None`` to 0.

    Ollama populates token counts on normal completions, but defensive
    ``None``-handling keeps the adapter from blowing up on unusual
    responses (e.g. a cached reply with no eval phase, or a future SDK
    shape change).
    """
    value = getattr(response, attr, None)
    if isinstance(value, int):
        return value
    return 0
