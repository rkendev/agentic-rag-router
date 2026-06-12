"""Fake `LLMPort` implementation + shared contract-test types.

`FakeLLMAdapter` is a canned-response stand-in for the real adapters.
It ships no SDK dependency: every `generate()` call is answered
directly from memory, controlled by a `FakeMode` field the test can
flip between calls. Purpose is two-fold:

  * a control group for the parametrized contract tests --- if a real
    adapter fails a contract test that the fake passes, the divergence
    is in the adapter's translation logic, not in the test;
  * a drop-in `LLMPort` for higher-level tests (e.g. `FallbackModel`)
    that need deterministic tier / error behaviour without SDK fakery.

`AdapterSpec` lives here rather than in `conftest.py` so it can be
imported by both the conftest wiring and the contract test module
without cross-conftest imports. Keeping the type in a plain module
sidesteps pytest's conftest-discovery magic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import pytest

from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier


class FakeMode(Enum):
    """Which behaviour the next `FakeLLMAdapter.generate()` call picks."""

    HEALTHY = "healthy"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    CONTENT = "content"


class FakeLLMAdapter:
    """`LLMPort` implementation that returns canned data.

    No SDK, no network, no monkeypatching required. Tests drive it via
    the `mode` attribute: `HEALTHY` returns a valid `LLMResponse`, the
    three error modes raise the matching domain exception. `ValueError`
    on empty / whitespace prompts matches the real adapters' contract
    (see `LLMPort.generate` docstring).
    """

    def __init__(
        self,
        *,
        tier: LLMTier = LLMTier.PRIMARY,
        model_name: str = "fake-model-1",
        mode: FakeMode = FakeMode.HEALTHY,
    ) -> None:
        self._tier = tier
        self._model_name = model_name
        self.mode = mode

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Answer the prompt per the current `mode`."""
        if not prompt.strip():
            msg = "prompt must not be empty or whitespace-only"
            raise ValueError(msg)

        if self.mode is FakeMode.TRANSIENT:
            msg = "fake transient failure"
            raise LLMTransientError(msg)
        if self.mode is FakeMode.PERMANENT:
            msg = "fake permanent failure"
            raise LLMPermanentError(msg)
        if self.mode is FakeMode.CONTENT:
            msg = "fake content failure"
            raise LLMContentError(msg)

        # HEALTHY: return a well-formed response. Text includes the
        # prompt prefix so tests that want to assert on round-tripping
        # have something to look at.
        return LLMResponse(
            text=f"fake response to: {prompt[:40]}",
            model_name=self._model_name,
            tier=self._tier,
            tokens_in=max(len(prompt.split()), 1),
            tokens_out=5,
            latency_ms=1,
            created_at=datetime.now(UTC),
        )


@dataclass(frozen=True)
class AdapterSpec:
    """Bundle of everything a contract test needs for one adapter.

    `build` returns the adapter with its underlying client already
    patched for a canned-healthy response. The three `inject_*`
    callables re-patch the same client to raise the SDK error the
    contract test wants to observe translated. Keeping the fakery
    centralised here lets the parametrized test bodies stay
    SDK-agnostic.
    """

    name: str  # pytest id (shows up in `-v` output)
    tier: LLMTier  # what LLMResponse.tier the adapter must declare
    build: Callable[[pytest.MonkeyPatch], LLMPort]
    inject_transient: Callable[[LLMPort, pytest.MonkeyPatch], None]
    inject_permanent: Callable[[LLMPort, pytest.MonkeyPatch], None]
    inject_content: Callable[[LLMPort, pytest.MonkeyPatch], None]
