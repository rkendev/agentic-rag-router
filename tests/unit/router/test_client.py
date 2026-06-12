"""Unit tests for `AnthropicRouterClient` --- the concrete model call.

Runs fully offline: the SDK client is built against a fake key (the SDK does not
validate until first request) and its ``messages.create`` is replaced with a
recorder, so the test asserts exactly what the router forwards to the API
without a network call. Mirrors the offline approach in
``tests/unit/infrastructure/test_anthropic_adapter.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.router.client import DEFAULT_ROUTER_MODEL, AnthropicRouterClient
from agentic_rag_router.router.schema import SYSTEM_PROMPT


class _RecordingMessages:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(stop_reason="end_turn", content=[])


class _RecordingSDK:
    def __init__(self) -> None:
        self.messages = _RecordingMessages()


def test_missing_api_key_fails_fast() -> None:
    with pytest.raises(ValueError, match="API key"):
        AnthropicRouterClient(api_key=None)


def test_create_message_forwards_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicRouterClient(
        api_key=SecretStr("sk-test"),
        model="claude-sonnet-4-6",
        system="SYSTEM",
        max_tokens=256,
    )
    sdk = _RecordingSDK()
    monkeypatch.setattr(client, "_client", sdk)

    tools = [{"name": "vector_search"}]
    messages = [{"role": "user", "content": "hi"}]
    tool_choice = {"type": "any"}

    response = client.create_message(messages=messages, tools=tools, tool_choice=tool_choice)

    kwargs = sdk.messages.kwargs
    assert kwargs is not None
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["system"] == "SYSTEM"
    assert kwargs["max_tokens"] == 256
    assert kwargs["tools"] is tools
    assert kwargs["tool_choice"] == tool_choice
    assert kwargs["messages"] is messages
    # Deterministic routing: temperature 0 so the route/refusal reproduce.
    assert kwargs["temperature"] == 0
    # No thinking param: it is incompatible with the forced tool_choice on iter 0.
    assert "thinking" not in kwargs
    assert response.stop_reason == "end_turn"


def test_defaults_use_sonnet_and_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicRouterClient(api_key=SecretStr("sk-test"))
    sdk = _RecordingSDK()
    monkeypatch.setattr(client, "_client", sdk)

    client.create_message(messages=[], tools=[], tool_choice={"type": "auto"})

    kwargs = sdk.messages.kwargs
    assert kwargs is not None
    assert kwargs["model"] == DEFAULT_ROUTER_MODEL
    assert kwargs["system"] == SYSTEM_PROMPT


def test_from_settings_pins_router_model_and_key() -> None:
    # Init kwargs outrank env, so this is isolated from the ambient environment.
    settings = Settings(
        anthropic_api_key=SecretStr("sk-test"),
        router_model="claude-test-model",
    )

    client = AnthropicRouterClient.from_settings(settings, max_tokens=128)

    assert client._model == "claude-test-model"
    assert client._max_tokens == 128
