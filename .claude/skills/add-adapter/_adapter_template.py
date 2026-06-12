"""<Vendor> adapter — <TIER> tier of the fallback stack.

This file is a TEMPLATE. Copy it to
`src/agentic_rag_router/infrastructure/<vendor>_adapter.py` and do the
search-and-replace pass described in
`.claude/skills/add-adapter/SKILL.md`:

    Vendor  → PascalCase vendor name (e.g. Groq)
    vendor  → lowercase vendor name  (e.g. groq)
    VENDOR  → UPPERCASE vendor name  (e.g. GROQ)
    <TIER>  → PRIMARY / SECONDARY / TERTIARY

The template mirrors the shape of `AnthropicAdapter` (see
`src/agentic_rag_router/infrastructure/anthropic_adapter.py`). Keep
the error-classification order — specific SDK exceptions before
generic ones — and always translate SDK exceptions into one of the
three domain error classes from `domain/errors.py`. Never let an
SDK-native exception escape the adapter.

Design notes to preserve (don't remove when filling this in):

- Optional SDK params (`system`, `temperature`) go into `create_kwargs`
  only when the caller supplied a value. This sidesteps SDK-evolving
  sentinels and lets the SDK apply its own defaults.
- Fail-fast on missing API key at construction — never at first call.
  The composition root (`main.py`) should see misconfiguration
  immediately, not N seconds later when the first request fires.
- Exception order is meaningful: the SDK's exception hierarchy usually
  has `RateLimitError` ⊂ `APIStatusError` ⊂ `APIError`. Specific
  handlers must precede the general ones.
- `structlog.bind()` with `adapter=` / `model=` / `max_tokens=` gives
  the observability stack a consistent tag set across adapters.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

# TODO: replace with the real vendor SDK import, e.g.
#   import groq
#   import google.generativeai as genai
#   import boto3
import <vendor_sdk>  # noqa: F401 — placeholder for the real SDK

import structlog
from pydantic import SecretStr

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier

_logger = structlog.get_logger(__name__)


class VendorAdapter:
    """LLMPort implementation backed by the <Vendor> Python SDK.

    Construction is fail-fast: a missing API key raises
    `LLMPermanentError` immediately so the composition root catches the
    misconfiguration before any request is attempted. The underlying
    SDK client is built once and reused across calls.
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
            <Vendor> API key. `None` (the shape produced by `Settings`
            when the env var is empty) raises `LLMPermanentError` at
            construction.
        model:
            <Vendor> model identifier (configurable via
            `Settings.<vendor>_model`).
        timeout_seconds:
            Per-request timeout passed to the SDK.
        max_retries:
            SDK-level retry budget for transient HTTP errors.
        default_max_tokens:
            Used when the caller passes `max_tokens=None`.
        """
        if api_key is None:
            msg = (
                "VendorAdapter requires an API key; "
                "set VENDOR_API_KEY in the environment or .env."
            )
            raise LLMPermanentError(msg)

        self._model = model
        self._default_max_tokens = default_max_tokens

        # TODO: construct the real SDK client. Example shapes:
        #   self._client = groq.Groq(
        #       api_key=api_key.get_secret_value(),
        #       timeout=float(timeout_seconds),
        #       max_retries=max_retries,
        #   )
        self._client = object()  # placeholder — replace

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Dispatch a single-turn prompt to <Vendor>.

        See `LLMPort.generate` for the full contract. Raises the three
        domain exception subclasses exclusively — no SDK exception
        ever escapes.
        """
        if not prompt.strip():
            msg = "prompt must not be empty or whitespace-only"
            raise ValueError(msg)

        effective_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        log = _logger.bind(
            adapter="VendorAdapter",
            model=self._model,
            max_tokens=effective_max_tokens,
        )

        # Build the request kwargs. Only add optional params when the
        # caller set them; otherwise let the SDK apply its own defaults.
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": effective_max_tokens,
            # TODO: message shape is vendor-specific. Examples:
            #   OpenAI-style: "messages": [{"role": "user", "content": prompt}]
            #   Anthropic:    "messages": [{"role": "user", "content": prompt}]
            #   Ollama:       "messages": [{"role": "user", "content": prompt}]
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            # TODO: some vendors take `system` as a top-level kwarg
            # (Anthropic), others as a `system` role in `messages` array
            # (OpenAI). Adjust accordingly.
            create_kwargs["system"] = system
        if temperature is not None:
            create_kwargs["temperature"] = temperature

        start = time.perf_counter()
        try:
            # TODO: call the real SDK method, e.g.
            #   response = self._client.chat.completions.create(**create_kwargs)
            response = self._call_sdk(**create_kwargs)

        # --- transient: network + server-side ---------------------------
        # TODO: replace <vendor>.APIConnectionError etc. with the real
        # SDK's exception classes. Keep the transient/permanent split
        # from `domain/errors.py`.
        except (
            # e.g. <vendor_sdk>.APIConnectionError,
            # e.g. <vendor_sdk>.APITimeoutError,
            Exception,  # placeholder — replace with real SDK transport errors
        ) as exc:
            log.warning("vendor.transient.transport", error_class=type(exc).__name__)
            msg = f"<Vendor> connection / timeout error: {exc}"
            raise LLMTransientError(msg) from exc
        # except <vendor_sdk>.RateLimitError as exc:
        #     log.warning("vendor.transient.rate_limit", error_class=type(exc).__name__)
        #     raise LLMTransientError(f"<Vendor> rate limit (429): {exc}") from exc
        # except <vendor_sdk>.InternalServerError as exc:
        #     log.warning("vendor.transient.5xx", error_class=type(exc).__name__)
        #     raise LLMTransientError(f"<Vendor> server error (5xx): {exc}") from exc

        # --- permanent: client-side configuration / request -------------
        # except (
        #     <vendor_sdk>.AuthenticationError,
        #     <vendor_sdk>.PermissionDeniedError,
        #     <vendor_sdk>.NotFoundError,
        #     <vendor_sdk>.BadRequestError,
        #     <vendor_sdk>.UnprocessableEntityError,
        # ) as exc:
        #     log.error("vendor.permanent", error_class=type(exc).__name__)
        #     raise LLMPermanentError(
        #         f"<Vendor> client error ({type(exc).__name__}): {exc}"
        #     ) from exc

        # --- unmapped status: classify by code, 5xx → transient ---------
        # except <vendor_sdk>.APIStatusError as exc:
        #     if exc.status_code >= 500:
        #         log.warning("vendor.transient.unmapped", status_code=exc.status_code)
        #         raise LLMTransientError(
        #             f"<Vendor> unmapped 5xx ({exc.status_code}): {exc}"
        #         ) from exc
        #     log.error("vendor.permanent.unmapped", status_code=exc.status_code)
        #     raise LLMPermanentError(
        #         f"<Vendor> unmapped 4xx ({exc.status_code}): {exc}"
        #     ) from exc

        # --- final safety net: any other SDK-native error ---------------
        # except <vendor_sdk>.APIError as exc:
        #     log.error("vendor.permanent.unknown", error_class=type(exc).__name__)
        #     raise LLMPermanentError(
        #         f"<Vendor> unknown API error ({type(exc).__name__}): {exc}"
        #     ) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        text = self._extract_text(response)
        return LLMResponse(
            text=text,
            model_name=response.model,  # TODO: adjust to vendor's response shape
            tier=LLMTier.PRIMARY,  # TODO: set to the tier this adapter fills
            tokens_in=response.usage.input_tokens,  # TODO: adjust
            tokens_out=response.usage.output_tokens,  # TODO: adjust
            latency_ms=latency_ms,
            created_at=datetime.now(UTC),
        )

    def _call_sdk(self, **kwargs: Any) -> Any:
        """Placeholder for the real SDK call.

        Remove this method once you've inlined the real SDK invocation
        into `generate()`. It exists only so the template parses cleanly
        before the search-and-replace pass completes.
        """
        raise NotImplementedError("Replace this with the real SDK call — see TODOs above.")

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text payload out of a <Vendor> response.

        Raises `LLMContentError` for any shape we can't turn into usable
        text: refusal / content-policy stop reason, empty content, or
        empty / whitespace-only text.

        TODO: replace the body below with the vendor-specific extraction
        path. Canonical shapes:

            OpenAI:    response.choices[0].message.content
            Anthropic: response.content[0].text  (check .type == "text")
            Ollama:    response.message.content
        """
        # Placeholder implementation. Replace with vendor-specific logic
        # and matching content-policy / refusal checks.
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            msg = "<Vendor> returned empty or whitespace-only text"
            raise LLMContentError(msg)
        return text
