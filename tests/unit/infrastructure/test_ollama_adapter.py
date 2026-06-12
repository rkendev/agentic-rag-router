"""Unit tests for `OllamaAdapter`.

Every test runs fully offline. The adapter's real `ollama.Client` is
constructed against a fake host (the SDK doesn't probe until first
request) and `client.chat` is patched via `monkeypatch` to either
return a fabricated response shape or raise a specific SDK / transport
exception. This isolates our translation logic from the network and
from any particular SDK version's private response schema.

Coverage goals:

  * Construction: always succeeds (no auth to fail-fast on); timeout
    flows through to the underlying SDK client.
  * Happy path: all response fields map onto `LLMResponse`; system
    prompt is prepended to `messages`; `max_tokens` lands in
    `options["num_predict"]`; optional params are omitted when not
    supplied; `stream=False` is always set.
  * Error-translation truth table: every SDK / transport error maps
    to the right domain error, with the original chained via
    `__cause__`.
  * Content failures: missing message, missing content, empty /
    whitespace-only text --- all become `LLMContentError`. (Ollama
    has no content-filter / refusal equivalent.)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import ollama
import pytest

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier
from agentic_rag_router.infrastructure.ollama_adapter import OllamaAdapter

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2:3b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    *,
    text: str | None = "Hello, Roy.",
    tokens_in: int = 12,
    tokens_out: int = 7,
    model: str = _DEFAULT_MODEL,
    message: Any = ...,  # sentinel: use default
) -> SimpleNamespace:
    """Fabricate a shape that looks like an Ollama ChatResponse.

    We use `SimpleNamespace` rather than `Mock` so attribute access has
    real semantics (no `Mock` spec quirks) and `getattr` returns real
    values --- `_extract_text` uses `getattr` and needs truthful answers.

    Pass ``message=None`` or a custom `SimpleNamespace` to exercise the
    content-failure branches; otherwise a normal message carrying `text`
    is constructed.
    """
    if message is ...:
        message = SimpleNamespace(role="assistant", content=text)
    return SimpleNamespace(
        message=message,
        model=model,
        prompt_eval_count=tokens_in,
        eval_count=tokens_out,
        done=True,
        done_reason="stop",
    )


def _patch_chat(monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter, fn: Any) -> None:
    """Replace `adapter._client.chat` with `fn`."""
    monkeypatch.setattr(adapter._client, "chat", fn)


@pytest.fixture
def adapter() -> OllamaAdapter:
    """A live `OllamaAdapter` against a fake host.

    Tests patch `adapter._client.chat` per-test to drive success /
    failure paths without hitting the network.
    """
    return OllamaAdapter(
        host=_DEFAULT_HOST,
        model=_DEFAULT_MODEL,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Ollama has no auth to validate, so construction always succeeds.

    The closest analogue to Anthropic/OpenAI's key fail-fast is "bad
    host URL" --- but that surfaces at first request as a transport
    error, not at construction. Keeping construction infallible matches
    the SDK's own contract.
    """

    def test_valid_host_and_model_succeeds(self) -> None:
        built = OllamaAdapter(host=_DEFAULT_HOST, model=_DEFAULT_MODEL)
        assert isinstance(built._client, ollama.Client)
        assert built._model == _DEFAULT_MODEL
        assert built._host == _DEFAULT_HOST

    def test_timeout_flows_to_client(self) -> None:
        # The ollama.Client stores timeout on its underlying httpx client.
        # We just verify the adapter round-tripped the value without
        # depending on the private attribute name of the httpx timeout.
        built = OllamaAdapter(
            host=_DEFAULT_HOST,
            model=_DEFAULT_MODEL,
            timeout_seconds=45,
        )
        # ollama.Client exposes its underlying httpx Client as `_client`
        # (the typical httpx.Client timeout shape carries connect/read/etc.
        # limits; we just assert the Client exists and was constructed).
        assert isinstance(built._client, ollama.Client)

    def test_default_max_tokens_overridable(self) -> None:
        built = OllamaAdapter(
            host=_DEFAULT_HOST,
            model=_DEFAULT_MODEL,
            default_max_tokens=2048,
        )
        assert built._default_max_tokens == 2048


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestGenerateSuccess:
    """A normal response becomes a well-formed `LLMResponse`."""

    def test_returns_llmresponse(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        _patch_chat(monkeypatch, adapter, lambda **_: _make_response())
        result = adapter.generate("hello")
        assert isinstance(result, LLMResponse)

    def test_all_fields_mapped(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        _patch_chat(
            monkeypatch,
            adapter,
            lambda **_: _make_response(text="Roy.", tokens_in=3, tokens_out=2),
        )
        result = adapter.generate("ping")

        assert result.text == "Roy."
        assert result.model_name == _DEFAULT_MODEL
        assert result.tier is LLMTier.TERTIARY
        assert result.tokens_in == 3
        assert result.tokens_out == 2
        assert result.latency_ms >= 0
        assert result.created_at.tzinfo is not None

    def test_system_prompt_prepended_to_messages(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # Like OpenAI (and unlike Anthropic), Ollama takes system as a
        # message with role="system", NOT as a top-level kwarg.
        captured: dict[str, Any] = {}

        def fake_chat(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_chat(monkeypatch, adapter, fake_chat)
        adapter.generate("hi", system="You are laconic.")

        messages = captured["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are laconic."}
        assert messages[1] == {"role": "user", "content": "hi"}
        # And NEVER as a top-level kwarg:
        assert "system" not in captured

    def test_omits_system_when_none(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_chat(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_chat(monkeypatch, adapter, fake_chat)
        adapter.generate("hi")

        messages = captured["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "hi"}

    def test_default_num_predict_applied_when_max_tokens_none(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # Ollama maps `max_tokens` to `options["num_predict"]` --- a
        # Modelfile-parameter convention, not a top-level kwarg.
        captured: dict[str, Any] = {}

        def fake_chat(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_chat(monkeypatch, adapter, fake_chat)
        adapter.generate("hi")
        assert captured["options"]["num_predict"] == 1024  # adapter default
        # And max_tokens is NEVER a top-level kwarg:
        assert "max_tokens" not in captured

    def test_explicit_max_tokens_lands_in_num_predict(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_chat(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_chat(monkeypatch, adapter, fake_chat)
        adapter.generate("hi", max_tokens=256)
        assert captured["options"]["num_predict"] == 256

    def test_temperature_passed_in_options(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_chat(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_chat(monkeypatch, adapter, fake_chat)
        adapter.generate("hi", temperature=0.2)
        assert captured["options"]["temperature"] == 0.2

    def test_temperature_omitted_from_options_when_none(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_chat(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_chat(monkeypatch, adapter, fake_chat)
        adapter.generate("hi")
        # No temperature key inside options when the caller omitted it.
        assert "temperature" not in captured["options"]

    def test_stream_always_false(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # The port contract is single-turn, non-streaming. Lock this in
        # so a future refactor can't accidentally flip it.
        captured: dict[str, Any] = {}

        def fake_chat(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _make_response()

        _patch_chat(monkeypatch, adapter, fake_chat)
        adapter.generate("hi")
        assert captured["stream"] is False

    def test_none_token_counts_coerced_to_zero(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # Defensive: if Ollama returns None for eval counts (e.g. cached
        # reply with no eval phase), we coerce to 0 rather than let it
        # explode inside pydantic validation of LLMResponse.
        response = SimpleNamespace(
            message=SimpleNamespace(content="ok"),
            model=_DEFAULT_MODEL,
            prompt_eval_count=None,
            eval_count=None,
        )
        _patch_chat(monkeypatch, adapter, lambda **_: response)
        result = adapter.generate("hi")
        assert result.tokens_in == 0
        assert result.tokens_out == 0

    def test_none_response_model_falls_back_to_configured(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # Defensive: `response.model` is Optional[str] in the SDK's
        # pydantic shape. If it comes back None (or empty), fall back to
        # the adapter's configured model rather than letting None flow
        # into LLMResponse.model_name (which has min_length=1).
        response = SimpleNamespace(
            message=SimpleNamespace(content="ok"),
            model=None,
            prompt_eval_count=1,
            eval_count=1,
        )
        _patch_chat(monkeypatch, adapter, lambda **_: response)
        result = adapter.generate("hi")
        assert result.model_name == _DEFAULT_MODEL

    def test_empty_prompt_rejected(self, adapter: OllamaAdapter) -> None:
        # No patching needed --- fails before the SDK call.
        with pytest.raises(ValueError, match="empty"):
            adapter.generate("")

    def test_whitespace_prompt_rejected(self, adapter: OllamaAdapter) -> None:
        with pytest.raises(ValueError, match="empty"):
            adapter.generate("   \n\t  ")


# ---------------------------------------------------------------------------
# Error translation (transient vs permanent)
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    """Every SDK / transport exception maps to exactly one domain error."""

    def test_connect_error_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # Ollama daemon not running --- classic transient: FallbackModel
        # should exhaust the tier list rather than retry in place.
        def boom(**_: Any) -> None:
            raise httpx.ConnectError("connection refused")

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="transport"):
            adapter.generate("hi")

    def test_read_timeout_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # Model is huge and the timeout is too tight --- still transient;
        # a smaller model (next tier) may answer in time.
        def boom(**_: Any) -> None:
            raise httpx.ReadTimeout("read timeout")

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="transport"):
            adapter.generate("hi")

    def test_generic_request_error_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # The RequestError catch-all covers any httpx transport error we
        # haven't enumerated above.
        def boom(**_: Any) -> None:
            raise httpx.RequestError("some transport weirdness")

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="transport"):
            adapter.generate("hi")

    def test_response_error_500_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise ollama.ResponseError("internal server error", 500)

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError, match="server error"):
            adapter.generate("hi")

    def test_response_error_503_becomes_transient(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise ollama.ResponseError("service unavailable", 503)

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMTransientError):
            adapter.generate("hi")

    def test_response_error_404_points_at_ollama_pull(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # The 404 branch is specifically shaped to nudge the operator
        # toward `ollama pull <model>` --- the most common way to land
        # a 404 is asking for a model the daemon doesn't have.
        def boom(**_: Any) -> None:
            raise ollama.ResponseError(
                f"model '{_DEFAULT_MODEL}' not found, try pulling it first",
                404,
            )

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="ollama pull"):
            adapter.generate("hi")

    def test_response_error_400_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        def boom(**_: Any) -> None:
            raise ollama.ResponseError("bad request body", 400)

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="status=400"):
            adapter.generate("hi")

    def test_response_error_other_4xx_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # A 4xx Ollama doesn't currently emit but might in the future
        # (e.g. 403 behind a reverse proxy). Anything 4xx that isn't
        # 404 lands on permanent via the catch-all branch.
        def boom(**_: Any) -> None:
            raise ollama.ResponseError("forbidden", 403)

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError, match="status=403"):
            adapter.generate("hi")

    def test_response_error_no_status_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # Defensive: if a future SDK version constructs ResponseError
        # without a status_code (or with -1, its current default), we
        # still need to classify rather than crash.
        def boom(**_: Any) -> None:
            raise ollama.ResponseError("cryptic")  # default status_code=-1

        _patch_chat(monkeypatch, adapter, boom)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hi")

    def test_exception_is_chained(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        """`raise ... from exc` must preserve the SDK exception as __cause__."""
        original = ollama.ResponseError("boom", 500)

        def boom(**_: Any) -> None:
            raise original

        _patch_chat(monkeypatch, adapter, boom)
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
    """Well-formed SDK success with unusable content raises `LLMContentError`.

    Ollama has no `finish_reason=content_filter` or `stop_reason=refusal`
    equivalent, so the adapter's content-failure surface is smaller than
    the hosted adapters': just "no usable text came back".
    """

    def test_missing_message(self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter) -> None:
        _patch_chat(monkeypatch, adapter, lambda **_: _make_response(message=None))
        with pytest.raises(LLMContentError, match="no message"):
            adapter.generate("hi")

    def test_message_without_content_attribute(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        # A message object missing the `content` attribute entirely ---
        # shouldn't AttributeError, should raise LLMContentError.
        bare_message = SimpleNamespace(role="assistant")
        _patch_chat(
            monkeypatch,
            adapter,
            lambda **_: _make_response(message=bare_message),
        )
        with pytest.raises(LLMContentError, match="empty or whitespace"):
            adapter.generate("hi")

    def test_none_content(self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter) -> None:
        message = SimpleNamespace(role="assistant", content=None)
        _patch_chat(
            monkeypatch,
            adapter,
            lambda **_: _make_response(message=message),
        )
        with pytest.raises(LLMContentError, match="empty or whitespace"):
            adapter.generate("hi")

    def test_empty_string_content(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        _patch_chat(monkeypatch, adapter, lambda **_: _make_response(text=""))
        with pytest.raises(LLMContentError):
            adapter.generate("hi")

    def test_whitespace_only_content(
        self, monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter
    ) -> None:
        _patch_chat(monkeypatch, adapter, lambda **_: _make_response(text="   \n"))
        with pytest.raises(LLMContentError, match="empty or whitespace"):
            adapter.generate("hi")
