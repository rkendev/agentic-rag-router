"""Contract-test fixtures.

Parametrizes the contract suite over every `LLMPort` implementation we
ship: `FakeLLMAdapter` (always-on control group) plus the three real
adapters with their SDK clients monkeypatched so nothing hits a
network. Everything in this module is test-scaffolding; no runtime
code depends on it.

The spec-per-adapter pattern keeps `test_llm_port.py` SDK-agnostic:
the test body just calls `spec.build(monkeypatch)` and
`spec.inject_*(adapter, monkeypatch)`. All vendor-specific fakery
lives in the helpers below.

Adding a fourth real adapter later (e.g. Cohere, Mistral) means one
new block of helpers + one new `AdapterSpec` entry --- the tests
pick it up automatically.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import ollama
import openai
import pytest
from pydantic import SecretStr

from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.llm import LLMTier
from agentic_rag_router.infrastructure.anthropic_adapter import AnthropicAdapter
from agentic_rag_router.infrastructure.ollama_adapter import OllamaAdapter
from agentic_rag_router.infrastructure.openai_adapter import OpenAIAdapter
from tests.contract.fakes import AdapterSpec, FakeLLMAdapter, FakeMode

# ===========================================================================
# FakeLLMAdapter --- control group. No SDK, no monkeypatching needed;
# `inject_*` helpers just flip the adapter's `mode` attribute.
# ===========================================================================


def _fake_build(_: pytest.MonkeyPatch) -> FakeLLMAdapter:
    return FakeLLMAdapter(tier=LLMTier.PRIMARY)


def _fake_inject_transient(adapter: LLMPort, _: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, FakeLLMAdapter)
    adapter.mode = FakeMode.TRANSIENT


def _fake_inject_permanent(adapter: LLMPort, _: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, FakeLLMAdapter)
    adapter.mode = FakeMode.PERMANENT


def _fake_inject_content(adapter: LLMPort, _: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, FakeLLMAdapter)
    adapter.mode = FakeMode.CONTENT


# ===========================================================================
# AnthropicAdapter --- fake `client.messages.create`
# ===========================================================================

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _anthropic_healthy_response(text: str = "fake anthropic") -> SimpleNamespace:
    """Shape that mimics `anthropic.types.Message` with one text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        model=_ANTHROPIC_MODEL,
        usage=SimpleNamespace(input_tokens=3, output_tokens=3),
    )


def _anthropic_status_error(
    cls: type[anthropic.APIStatusError], status: int
) -> anthropic.APIStatusError:
    """Build an Anthropic `APIStatusError` with a real httpx.Response."""
    request = httpx.Request("POST", _ANTHROPIC_URL)
    response = httpx.Response(status_code=status, request=request)
    return cls(message="contract test", response=response, body=None)


def _anthropic_build(monkeypatch: pytest.MonkeyPatch) -> AnthropicAdapter:
    adapter = AnthropicAdapter(
        api_key=SecretStr("sk-ant-test-fake"),
        model=_ANTHROPIC_MODEL,
    )
    monkeypatch.setattr(
        adapter._client.messages,
        "create",
        lambda **_: _anthropic_healthy_response(),
    )
    return adapter


def _anthropic_inject_transient(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, AnthropicAdapter)
    err = _anthropic_status_error(anthropic.RateLimitError, 429)

    def boom(**_: Any) -> None:
        raise err

    monkeypatch.setattr(adapter._client.messages, "create", boom)


def _anthropic_inject_permanent(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, AnthropicAdapter)
    err = _anthropic_status_error(anthropic.AuthenticationError, 401)

    def boom(**_: Any) -> None:
        raise err

    monkeypatch.setattr(adapter._client.messages, "create", boom)


def _anthropic_inject_content(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, AnthropicAdapter)
    # Well-formed SDK response with whitespace-only text --- passes the
    # shape checks but trips the adapter's content validation.
    monkeypatch.setattr(
        adapter._client.messages,
        "create",
        lambda **_: _anthropic_healthy_response(text="   "),
    )


# ===========================================================================
# OpenAIAdapter --- fake `client.chat.completions.create`
# ===========================================================================

_OPENAI_MODEL = "gpt-4o-mini"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _openai_healthy_response(text: str = "fake openai") -> SimpleNamespace:
    """Shape that mimics an OpenAI `ChatCompletion` with one choice."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model=_OPENAI_MODEL,
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=3),
    )


def _openai_status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    """Build an OpenAI `APIStatusError` with a real httpx.Response."""
    request = httpx.Request("POST", _OPENAI_URL)
    response = httpx.Response(status_code=status, request=request)
    return cls(message="contract test", response=response, body=None)


def _openai_build(monkeypatch: pytest.MonkeyPatch) -> OpenAIAdapter:
    adapter = OpenAIAdapter(
        api_key=SecretStr("sk-oai-test-fake"),
        model=_OPENAI_MODEL,
    )
    monkeypatch.setattr(
        adapter._client.chat.completions,
        "create",
        lambda **_: _openai_healthy_response(),
    )
    return adapter


def _openai_inject_transient(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, OpenAIAdapter)
    err = _openai_status_error(openai.RateLimitError, 429)

    def boom(**_: Any) -> None:
        raise err

    monkeypatch.setattr(adapter._client.chat.completions, "create", boom)


def _openai_inject_permanent(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, OpenAIAdapter)
    err = _openai_status_error(openai.AuthenticationError, 401)

    def boom(**_: Any) -> None:
        raise err

    monkeypatch.setattr(adapter._client.chat.completions, "create", boom)


def _openai_inject_content(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, OpenAIAdapter)
    monkeypatch.setattr(
        adapter._client.chat.completions,
        "create",
        lambda **_: _openai_healthy_response(text="   "),
    )


# ===========================================================================
# OllamaAdapter --- fake `client.chat`
# ===========================================================================

_OLLAMA_HOST = "http://localhost:11434"
_OLLAMA_MODEL = "llama3.2:3b"


def _ollama_healthy_response(text: str = "fake ollama") -> SimpleNamespace:
    """Shape that mimics the `ollama.ChatResponse` pydantic model."""
    return SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=text),
        model=_OLLAMA_MODEL,
        prompt_eval_count=3,
        eval_count=3,
        done=True,
        done_reason="stop",
    )


def _ollama_build(monkeypatch: pytest.MonkeyPatch) -> OllamaAdapter:
    adapter = OllamaAdapter(host=_OLLAMA_HOST, model=_OLLAMA_MODEL)
    monkeypatch.setattr(
        adapter._client,
        "chat",
        lambda **_: _ollama_healthy_response(),
    )
    return adapter


def _ollama_inject_transient(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, OllamaAdapter)

    # httpx.ConnectError is an httpx.RequestError subclass --- the
    # catch-all transport-error branch maps it to transient.
    def boom(**_: Any) -> None:
        raise httpx.ConnectError("contract test: daemon unreachable")

    monkeypatch.setattr(adapter._client, "chat", boom)


def _ollama_inject_permanent(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, OllamaAdapter)
    # 404 is the canonical "model not pulled" signal --- permanent.
    err = ollama.ResponseError("model not found", 404)

    def boom(**_: Any) -> None:
        raise err

    monkeypatch.setattr(adapter._client, "chat", boom)


def _ollama_inject_content(adapter: LLMPort, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(adapter, OllamaAdapter)
    monkeypatch.setattr(
        adapter._client,
        "chat",
        lambda **_: _ollama_healthy_response(text="   "),
    )


# ===========================================================================
# Fixture list + parametrized fixture
# ===========================================================================


LLM_ADAPTERS: list[AdapterSpec] = [
    AdapterSpec(
        name="fake",
        tier=LLMTier.PRIMARY,
        build=_fake_build,
        inject_transient=_fake_inject_transient,
        inject_permanent=_fake_inject_permanent,
        inject_content=_fake_inject_content,
    ),
    AdapterSpec(
        name="anthropic",
        tier=LLMTier.PRIMARY,
        build=_anthropic_build,
        inject_transient=_anthropic_inject_transient,
        inject_permanent=_anthropic_inject_permanent,
        inject_content=_anthropic_inject_content,
    ),
    AdapterSpec(
        name="openai",
        tier=LLMTier.SECONDARY,
        build=_openai_build,
        inject_transient=_openai_inject_transient,
        inject_permanent=_openai_inject_permanent,
        inject_content=_openai_inject_content,
    ),
    AdapterSpec(
        name="ollama",
        tier=LLMTier.TERTIARY,
        build=_ollama_build,
        inject_transient=_ollama_inject_transient,
        inject_permanent=_ollama_inject_permanent,
        inject_content=_ollama_inject_content,
    ),
]


@pytest.fixture(params=LLM_ADAPTERS, ids=lambda spec: spec.name)
def adapter_spec(request: pytest.FixtureRequest) -> AdapterSpec:
    """The adapter spec currently under contract test.

    Parametrized: every test that takes this fixture runs once per
    adapter, giving `adapter_count * test_count` total cases. Test ids
    show up as `test_foo[fake]`, `test_foo[anthropic]`, etc.

    `request.param` is typed as `Any` in pytest's stubs, so we `cast`
    rather than return it raw --- strict mypy's `no-any-return` would
    otherwise flag this line. The parametrization on `LLM_ADAPTERS`
    (itself typed `list[AdapterSpec]`) makes the cast statically safe.
    """
    return cast(AdapterSpec, request.param)
