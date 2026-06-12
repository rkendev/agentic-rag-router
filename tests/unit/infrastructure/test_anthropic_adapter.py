"""Unit tests for `AnthropicAdapter`.

Every test runs fully offline. The adapter's real `anthropic.Anthropic`
client is constructed against a fake `sk-test` key (the SDK doesn't
validate until first request) and `client.messages.create` is patched
via `monkeypatch` to either return a fabricated response shape or
raise a specific SDK exception. This isolates our translation logic
from the network and from any particular SDK version's private
response schema.

Coverage goals:

  * Construction: fail-fast on missing key, success on valid key.
  * Happy path: all response fields map correctly onto `LLMResponse`.
  * Error-translation truth table (see T007 handoff): every SDK
    exception class maps to the right domain error, with the original
    exception chained via `__cause__`.
  * Content failures: refusal stop_reason, empty content list, non-text
    first block, whitespace-only text --- all become `LLMContentError`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import SecretStr

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier
from agentic_rag_router.infrastructure.anthropic_adapter import AnthropicAdapter

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    *,
    text: str = "Hello, Roy.",
    stop_reason: str = "end_turn",
    tokens_in: int = 12,
    tokens_out: int = 7,
    model: str = _DEFAULT_MODEL,
    block_type: str = "text",
    content: list[Any] | None = None,
) -> SimpleNamespace:
    """Fabricate a shape that looks like anthropic.types.Message.

    We use `SimpleNamespace` rather than `Mock` so attribute access has
    real semantics (no `Mock` spec quirks) and `getattr` returns real
    values --- the adapter's `_extract_text` uses `getattr` and needs
    truthful answers.
    """
    if content is None:
        content = [SimpleNamespace(type=block_type, text=text)]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


def _make_status_error(
    cls: type[anthropic.APIStatusError], status_code: int, message: str = "boom"
) -> anthropic.APIStatusError:
    """Build an `APIStatusError` subclass with a real httpx.Response.

    The SDK's `APIStatusError.__init__` requires a response object it can
    read `status_code` from; constructing via `httpx.Response` avoids the
    pain of mocking that shape.
    """
    request = httpx.Request("POST", _ANTHROPIC_URL)
    response = httpx.Response(status_code=status_code, request=request)
    return cls(message=message, response=response, body=None)


def _make_connection_error(
    cls: type[anthropic.APIConnectionError],
) -> anthropic.APIConnectionError:
    """Build an `APIConnectionError` (or subclass like `APITimeoutError`)."""
    request = httpx.Request("POST", _ANTHROPIC_URL)
    return cls(request=request)


@pytest.fixture
def adapter() -> AnthropicAdapter:
    """A live `AnthropicAdapter` against a fake key.

    Tests patch `adapter._client.messages.create` per-test to drive
    success / failure paths without hitting the network.
    """
    return AnthropicAdapter(
        api_key=SecretStr("sk-ant-test-fake"),
        model=_DEFAULT_MODEL,
    )


def _patch_create(monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter, fn: Any) -> None:
    """Replace `adapter._client.messages.create` with `fn`."""
    monkeypatch.setattr(adapter._client.messages, "create", fn)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Fail-fast per D4; don't let a missing key reach the first request."""

    def test_none_api_key_raises_permanent(self) -> None:
        with pytest.raises(LLMPermanentError, match="requires an API key"):
            AnthropicAdapter(api_key=None, model=_DEFAULT_MODEL)

    def test_valid_key_succeeds(self) -> None:
        # Doesn't call the network --- the SDK client defers validation.
        built = AnthropicAdapter(api_key=SecretStr("sk-ant-test"), model=_DEFAULT_MODEL)
        assert isinstance(built._client, anthropic.Anthropic)

    def test_timeout_and_retries_flow_to_client(self) -> None:
        built = AnthropicAdapter(
            api_key=SecretStr("sk-ant-test"),
            model=_DEFAULT_MODEL,
            timeout_seconds=45,
            max_retries=4,
        )
        # The SDK exposes timeout on the client; we just verify it's the
        # value we passed (coerced to float as the SDK expects).
        assert float(built._client.timeout) == 45.0  # type: ignore[arg-type]
        assert built._client.max_retries == 4


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestGenerateSuccess:
    """A normal response becomes a well-formed `LLMResponse`."""

    def test_returns_llmresponse(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response())
        result = adapter.generate("hello")
        assert isinstance(result, LLMResponse)

    def test_all_fields_mapped(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        _patch_create(
            monkeypatch,
            adapter,
            lambda **_: _make_response(text="Roy.", tokens_in=3, tokens_out=2),
        )
        result = adapter.generate("ping")

        assert result.text == "Roy."
        assert result.model_name == _DEFAULT_MODEL
        assert result.tier is LLMTier.PRIMARY
        assert result.tokens_in == 3
        assert result.tokens_out == 2
        assert result.latency_ms >= 0
        assert result.created_at.tzinfo is not None

    def test_passes_system_prompt_through(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi", system="You are laconic.")
        assert captured["system"] == "You are laconic."

    def test_omits_system_when_none(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi")
        # Key omitted entirely --- the SDK applies its own default.
        assert "system" not in captured
        assert "temperature" not in captured

    def test_default_max_tokens_applied_when_none(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi")
        assert captured["max_tokens"] == 1024  # adapter default

    def test_explicit_max_tokens_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi", max_tokens=256)
        assert captured["max_tokens"] == 256

    def test_temperature_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi", temperature=0.2)
        assert captured["temperature"] == 0.2

    def test_empty_prompt_rejected(self, adapter: AnthropicAdapter) -> None:
        # No patching needed --- fails before the SDK call.
        with pytest.raises(ValueError, match="empty"):
            adapter.generate("")

    def test_whitespace_prompt_rejected(self, adapter: AnthropicAdapter) -> None:
        with pytest.raises(ValueError, match="empty"):
            adapter.generate("   \n\t  ")


# ---------------------------------------------------------------------------
# Error translation (transient vs permanent vs content)
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    """Every SDK exception maps to exactly one domain error."""

    def test_timeout_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_connection_error(anthropic.APITimeoutError)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="connection"):
            adapter.generate("hi")

    def test_connection_error_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_connection_error(anthropic.APIConnectionError)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError):
            adapter.generate("hi")

    def test_rate_limit_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.RateLimitError, 429)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="rate limit"):
            adapter.generate("hi")

    def test_internal_server_error_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.InternalServerError, 500)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="server error"):
            adapter.generate("hi")

    def test_authentication_error_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.AuthenticationError, 401)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="AuthenticationError"):
            adapter.generate("hi")

    def test_permission_denied_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.PermissionDeniedError, 403)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_not_found_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.NotFoundError, 404)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_bad_request_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.BadRequestError, 400)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_unprocessable_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.UnprocessableEntityError, 422)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_unmapped_5xx_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        # An APIStatusError subclass we haven't enumerated with a 5xx code
        # should still route to transient via the status_code branch.
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.APIStatusError, 503)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="unmapped 5xx"):
            adapter.generate("hi")

    def test_unmapped_4xx_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        # 418 is not enumerated; the status_code branch must send it to
        # permanent (client-side).
        def boom(**_: Any) -> None:
            raise _make_status_error(anthropic.APIStatusError, 418)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="unmapped 4xx"):
            adapter.generate("hi")

    def test_unknown_api_error_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        # Construct a bare `anthropic.APIError` that isn't a status or
        # connection error --- falls into the final safety net.
        def boom(**_: Any) -> None:
            request = httpx.Request("POST", _ANTHROPIC_URL)
            raise anthropic.APIError(message="weird", request=request, body=None)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="unknown API error"):
            adapter.generate("hi")

    def test_exception_is_chained(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        """`raise ... from exc` must preserve the SDK exception as __cause__."""
        original = _make_status_error(anthropic.RateLimitError, 429)

        def boom(**_: Any) -> None:
            raise original

        _patch_create(monkeypatch, adapter, boom)
        try:
            adapter.generate("hi")
        except LLMTransientError as domain_exc:
            assert domain_exc.__cause__ is original
        else:
            pytest.fail("expected LLMTransientError")


# ---------------------------------------------------------------------------
# Content failures (successful call, unusable output)
# ---------------------------------------------------------------------------


class TestContentFailures:
    """Well-formed SDK success with unusable content raises `LLMContentError`."""

    def test_refusal_stop_reason(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        _patch_create(
            monkeypatch,
            adapter,
            lambda **_: _make_response(stop_reason="refusal"),
        )
        with pytest.raises(LLMContentError, match="refused"):
            adapter.generate("hi")

    def test_empty_content_list(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response(content=[]))
        with pytest.raises(LLMContentError, match="empty content"):
            adapter.generate("hi")

    def test_non_text_first_block(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        tool_use_block = SimpleNamespace(type="tool_use", text=None)
        _patch_create(
            monkeypatch,
            adapter,
            lambda **_: _make_response(content=[tool_use_block]),
        )
        with pytest.raises(LLMContentError, match="non-text"):
            adapter.generate("hi")

    def test_whitespace_only_text(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response(text="   \n"))
        with pytest.raises(LLMContentError, match="empty or whitespace"):
            adapter.generate("hi")

    def test_empty_string_text(
        self, monkeypatch: pytest.MonkeyPatch, adapter: AnthropicAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response(text=""))
        with pytest.raises(LLMContentError):
            adapter.generate("hi")
