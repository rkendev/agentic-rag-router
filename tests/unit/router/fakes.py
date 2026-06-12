"""Honest test doubles for the router loop.

Two doubles plus a couple of response builders:

* `FakeAnthropicClient` --- an `AnthropicClientPort` scripted with a list of
  responses. It records the ``tool_choice`` passed on each turn (so a test can
  assert the force-then-relax rule) and repeats its last scripted response once
  exhausted (so the iteration-cap path can be driven with a single
  always-``tool_use`` response).
* `FakeDispatcher` --- a `DispatcherPort` returning a preset `DispatchOutcome`
  per tool name (or a default), recording every ``dispatch`` call.

The response builders fabricate the slim Messages-API shape the loop reads
(``stop_reason`` + ``content`` blocks) with `SimpleNamespace`, mirroring the
convention in ``tests/unit/infrastructure/test_anthropic_adapter.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentic_rag_router.router.dispatch import DispatchOutcome


def make_text_response(
    text: str = "Final answer.", stop_reason: str = "end_turn"
) -> SimpleNamespace:
    """A response whose content is a single text block."""
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


def make_tool_use_response(
    blocks: list[tuple[str, str, dict[str, Any]]],
    *,
    stop_reason: str = "tool_use",
) -> SimpleNamespace:
    """A response carrying one or more ``tool_use`` blocks.

    ``blocks`` is a list of ``(id, name, input)`` triples; pass two or more to
    simulate a parallel multi-tool turn.
    """
    content = [
        SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)
        for block_id, name, tool_input in blocks
    ]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def make_empty_response(stop_reason: str = "max_tokens") -> SimpleNamespace:
    """A non-end_turn response with no tool_use and no text block."""
    return SimpleNamespace(stop_reason=stop_reason, content=[])


def success_outcome(
    *,
    content: str = '{"ok": true}',
    latency_ms: int = 7,
    citations: list[dict[str, object]] | None = None,
) -> DispatchOutcome:
    """A successful `DispatchOutcome` for loop tests."""
    return DispatchOutcome(
        content=content,
        is_error=False,
        ok=True,
        error_code=None,
        latency_ms=latency_ms,
        citations=citations
        if citations is not None
        else [{"tool": "vector_search", "source": "x"}],
    )


def error_outcome(
    *,
    error_code: str = "backend_error",
    content: str = '{"ok": false}',
    latency_ms: int = 3,
) -> DispatchOutcome:
    """A failed `DispatchOutcome` for loop tests (is_error, no citations)."""
    return DispatchOutcome(
        content=content,
        is_error=True,
        ok=False,
        error_code=error_code,
        latency_ms=latency_ms,
        citations=[],
    )


class FakeAnthropicClient:
    """`AnthropicClientPort` scripted with a fixed list of responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.tool_choices: list[dict[str, str]] = []
        self.messages_seen: list[list[dict[str, Any]]] = []

    def create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, str],
    ) -> Any:
        self.tool_choices.append(tool_choice)
        self.messages_seen.append(messages)
        index = self.calls if self.calls < len(self._responses) else -1
        self.calls += 1
        return self._responses[index]


class FakeDispatcher:
    """`DispatcherPort` returning a preset outcome per tool name."""

    def __init__(
        self,
        *,
        outcomes: dict[str, DispatchOutcome] | None = None,
        default: DispatchOutcome | None = None,
    ) -> None:
        self._outcomes = outcomes if outcomes is not None else {}
        self._default = default if default is not None else success_outcome()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> DispatchOutcome:
        self.calls.append((name, tool_input))
        return self._outcomes.get(name, self._default)
