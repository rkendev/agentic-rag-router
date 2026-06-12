"""Unit tests for `OpenAIAdapter`.

Every test runs fully offline. The adapter's real `openai.OpenAI` client
is constructed against a fake `sk-test` key (the SDK doesn't validate
until first request) and `client.chat.completions.create` is patched
via `monkeypatch` to either return a fabricated response shape or raise
a specific SDK exception. This isolates our translation logic from the
network and from any particular SDK version's private response schema.

Coverage goals:

  * Construction: fail-fast on missing key, success on valid key.
  * Happy path: all response fields map correctly onto `LLMResponse`;
    system prompt is prepended to `messages` (NOT a top-level kwarg);
    optional params are omitted when not supplied.
  * Error-translation truth table: every SDK exception class maps to
    the right domain error, with the original chained via `__cause__`.
  * Content failures: content_filter refusal, empty choices list,
    missing message, empty / whitespace-only content --- all become
    `LLMContentError`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import SecretStr

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier
from agentic_rag_router.infrastructure.openai_adapter import OpenAIAdapter

_DEFAULT_MODEL = "gpt-4o-mini"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    *,
    text: str = "Hello, Roy.",
    finish_reason: str = "stop",
    tokens_in: int = 12,
    tokens_out: int = 7,
    model: str = _DEFAULT_MODEL,
    choices: list[Any] | None = None,
) -> SimpleNamespace:
    """Fabricate a shape that looks like an OpenAI ChatCompletion.

    We use `SimpleNamespace` rather than `Mock` so attribute access has
    real semantics (no `Mock` spec quirks) and `getattr` returns real
    values --- `_extract_text` uses `getattr` and needs truthful answers.
    """
    if choices is None:
        message = SimpleNamespace(content=text)
        choices = [SimpleNamespace(message=message, finish_reason=finish_reason)]
    return SimpleNamespace(
        choices=choices,
        model=model,
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


def _make_status_error(
    cls: type[openai.APIStatusError], status_code: int, message: str = "boom"
) -> openai.APIStatusError:
    """Build an `APIStatusError` subclass with a real httpx.Response.

    The SDK's `APIStatusError.__init__` requires a response object it can
    read `status_code` from; constructing via `httpx.Response` avoids the
    pain of mocking that shape.
    """
    request = httpx.Request("POST", _OPENAI_URL)
    response = httpx.Response(status_code=status_code, request=request)
    return cls(message=message, response=response, body=None)


def _make_connection_error(
    cls: type[openai.APIConnectionError],
) -> openai.APIConnectionError:
    """Build an `APIConnectionError` (or subclass like `APITimeoutError`)."""
    request = httpx.Request("POST", _OPENAI_URL)
    return cls(request=request)


@pytest.fixture
def adapter() -> OpenAIAdapter:
    """A live `OpenAIAdapter` against a fake key.

    Tests patch `adapter._client.chat.completions.create` per-test to
    drive success / failure paths without hitting the network.
    """
    return OpenAIAdapter(
        api_key=SecretStr("sk-oai-test-fake"),
        model=_DEFAULT_MODEL,
    )


def _patch_create(monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter, fn: Any) -> None:
    """Replace `adapter._client.chat.completions.create` with `fn`."""
    monkeypatch.setattr(adapter._client.chat.completions, "create", fn)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Fail-fast per D4; don't let a missing key reach the first request."""

    def test_none_api_key_raises_permanent(self) -> None:
        with pytest.raises(LLMPermanentError, match="requires an API key"):
            OpenAIAdapter(api_key=None, model=_DEFAULT_MODEL)

    def test_valid_key_succeeds(self) -> None:
        # Doesn't call the network --- the SDK client defers validation.
        built = OpenAIAdapter(api_key=SecretStr("sk-oai-test"), model=_DEFAULT_MODEL)
        assert isinstance(built._client, openai.OpenAI)

    def test_timeout_and_retries_flow_to_client(self) -> None:
        built = OpenAIAdapter(
            api_key=SecretStr("sk-oai-test"),
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
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response())
        result = adapter.generate("hello")
        assert isinstance(result, LLMResponse)

    def test_all_fields_mapped(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        _patch_create(
            monkeypatch,
            adapter,
            lambda **_: _make_response(text="Roy.", tokens_in=3, tokens_out=2),
        )
        result = adapter.generate("ping")

        assert result.text == "Roy."
        assert result.model_name == _DEFAULT_MODEL
        assert result.tier is LLMTier.SECONDARY
        assert result.tokens_in == 3
        assert result.tokens_out == 2
        assert result.latency_ms >= 0
        assert result.created_at.tzinfo is not None

    def test_system_prompt_prepended_to_messages(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        # Unlike Anthropic, OpenAI takes system as a message with
        # role="system", NOT as a top-level kwarg.
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi", system="You are laconic.")

        messages = captured["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are laconic."}
        assert messages[1] == {"role": "user", "content": "hi"}
        # And NEVER as a top-level kwarg:
        assert "system" not in captured

    def test_omits_system_when_none(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi")

        messages = captured["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "hi"}
        # Temperature omitted entirely --- the SDK applies its own default.
        assert "temperature" not in captured

    def test_default_max_tokens_applied_when_none(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi")
        assert captured["max_tokens"] == 1024  # adapter default

    def test_explicit_max_tokens_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi", max_tokens=256)
        assert captured["max_tokens"] == 256

    def test_temperature_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_create(monkeypatch, adapter, fake_create)
        adapter.generate("hi", temperature=0.2)
        assert captured["temperature"] == 0.2

    def test_empty_prompt_rejected(self, adapter: OpenAIAdapter) -> None:
        # No patching needed --- fails before the SDK call.
        with pytest.raises(ValueError, match="empty"):
            adapter.generate("")

    def test_whitespace_prompt_rejected(self, adapter: OpenAIAdapter) -> None:
        with pytest.raises(ValueError, match="empty"):
            adapter.generate("   \n\t  ")


# ---------------------------------------------------------------------------
# Error translation (transient vs permanent vs content)
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    """Every SDK exception maps to exactly one domain error."""

    def test_timeout_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_connection_error(openai.APITimeoutError)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="connection"):
            adapter.generate("hi")

    def test_connection_error_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_connection_error(openai.APIConnectionError)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError):
            adapter.generate("hi")

    def test_rate_limit_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.RateLimitError, 429)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="rate limit"):
            adapter.generate("hi")

    def test_internal_server_error_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.InternalServerError, 500)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="server error"):
            adapter.generate("hi")

    def test_authentication_error_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.AuthenticationError, 401)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="AuthenticationError"):
            adapter.generate("hi")

    def test_permission_denied_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.PermissionDeniedError, 403)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_not_found_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.NotFoundError, 404)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_conflict_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        # OpenAI-specific 409 --- the Anthropic adapter's truth table
        # doesn't have this branch because the Anthropic SDK doesn't
        # expose a ConflictError class.
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.ConflictError, 409)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="ConflictError"):
            adapter.generate("hi")

    def test_bad_request_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.BadRequestError, 400)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_unprocessable_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.UnprocessableEntityError, 422)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_unmapped_5xx_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        # An APIStatusError subclass we haven't enumerated with a 5xx code
        # should still route to transient via the status_code branch.
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.APIStatusError, 503)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="unmapped 5xx"):
            adapter.generate("hi")

    def test_unmapped_4xx_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        # 418 is not enumerated; the status_code branch must send it to
        # permanent (client-side).
        def boom(**_: Any) -> None:
            raise _make_status_error(openai.APIStatusError, 418)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="unmapped 4xx"):
            adapter.generate("hi")

    def test_unknown_api_error_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        # Construct a bare `openai.APIError` that isn't a status or
        # connection error --- falls into the final safety net.
        def boom(**_: Any) -> None:
            request = httpx.Request("POST", _OPENAI_URL)
            raise openai.APIError(message="weird", request=request, body=None)

        _patch_create(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="unknown API error"):
            adapter.generate("hi")

    def test_exception_is_chained(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        """`raise ... from exc` must preserve the SDK exception as __cause__."""
        original = _make_status_error(openai.RateLimitError, 429)

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

    def test_content_filter_finish_reason(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        _patch_create(
            monkeypatch,
            adapter,
            lambda **_: _make_response(finish_reason="content_filter"),
        )
        with pytest.raises(LLMContentError, match="refused"):
            adapter.generate("hi")

    def test_empty_choices_list(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response(choices=[]))
        with pytest.raises(LLMContentError, match="empty choices"):
            adapter.generate("hi")

    def test_choice_without_message(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        # A choice that has no `message` attribute at all --- shouldn't
        # crash with AttributeError, should raise LLMContentError.
        bare_choice = SimpleNamespace(finish_reason="stop", message=None)
        _patch_create(
            monkeypatch,
            adapter,
            lambda **_: _make_response(choices=[bare_choice]),
        )
        with pytest.raises(LLMContentError, match="no message"):
            adapter.generate("hi")

    def test_none_content(self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter) -> None:
        # OpenAI returns message.content=None when the model emits a
        # tool_call or function_call instead of text --- we treat that
        # as unusable for the template's single-turn text-only use case.
        message = SimpleNamespace(content=None)
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        _patch_create(
            monkeypatch,
            adapter,
            lambda **_: _make_response(choices=[choice]),
        )
        with pytest.raises(LLMContentError, match="empty or whitespace"):
            adapter.generate("hi")

    def test_whitespace_only_text(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response(text="   \n"))
        with pytest.raises(LLMContentError, match="empty or whitespace"):
            adapter.generate("hi")

    def test_empty_string_text(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OpenAIAdapter
    ) -> None:
        _patch_create(monkeypatch, adapter, lambda **_: _make_response(text=""))
        with pytest.raises(LLMContentError):
            adapter.generate("hi")
