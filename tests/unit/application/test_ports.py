"""Structural tests for the `LLMPort` Protocol.

Protocols are primarily a mypy construct --- mypy is what enforces that
adapters conform. These tests add a runtime-facing safety net:

  1. A conforming stub satisfies `isinstance(..., LLMPort)` thanks to
     `@runtime_checkable`. This is the mechanism `FallbackModel` uses
     to guard against mis-wired composition.
  2. A class missing `generate` is rejected by the same runtime check.
  3. A concrete stub returning a valid `LLMResponse` demonstrates the
     full contract end-to-end --- adapters in T007/T008/T009 use this
     shape as their reference.

No behavioural coverage here (no failover, no retries, no error routing)
--- that lives in `test_fallback.py` at T005b.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.llm import LLMResponse, LLMTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_response(text: str = "ok") -> LLMResponse:
    """Build a minimal valid LLMResponse for stub return values."""
    return LLMResponse(
        text=text,
        model_name="stub-model",
        tier=LLMTier.PRIMARY,
        tokens_in=0,
        tokens_out=0,
        latency_ms=0,
        created_at=datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Runtime Protocol conformance
# ---------------------------------------------------------------------------


class TestLLMPortStructural:
    """`@runtime_checkable` lets `isinstance` verify method presence."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        class ConformingStub:
            def generate(
                self,
                prompt: str,
                *,
                system: str | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
            ) -> LLMResponse:
                return _canonical_response()

        assert isinstance(ConformingStub(), LLMPort)

    def test_missing_method_fails_protocol_check(self) -> None:
        class NotAnLLM:
            """Deliberately omits `generate`."""

        assert not isinstance(NotAnLLM(), LLMPort)

    def test_unrelated_method_fails_protocol_check(self) -> None:
        # A class with a differently-named method does not satisfy the
        # Protocol --- runtime_checkable checks attribute presence, not
        # arbitrary callable shape.
        class DifferentlyNamed:
            def complete(self, prompt: str) -> str:
                return prompt

        assert not isinstance(DifferentlyNamed(), LLMPort)


# ---------------------------------------------------------------------------
# Contract demonstration
# ---------------------------------------------------------------------------


class TestLLMPortContract:
    """A concrete stub end-to-end: input goes in, domain response comes out."""

    def test_stub_returns_valid_llmresponse(self) -> None:
        class EchoStub:
            def generate(
                self,
                prompt: str,
                *,
                system: str | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
            ) -> LLMResponse:
                # Echo the prompt back as the response text, ignoring the
                # other kwargs. Exercises the keyword-only signature.
                return _canonical_response(text=prompt)

        port: LLMPort = EchoStub()
        resp = port.generate("hello world")

        assert isinstance(resp, LLMResponse)
        assert resp.text == "hello world"
        assert resp.tier is LLMTier.PRIMARY
        assert resp.created_at.tzinfo is UTC

    def test_stub_honours_keyword_only_signature(self) -> None:
        # The Protocol declares `system`, `max_tokens`, `temperature` as
        # keyword-only. A stub that accepts them positionally would still
        # satisfy isinstance (Protocol doesn't enforce kw-only at runtime)
        # but callers must pass them by name --- this test pins that
        # convention for future adapter authors.
        class KwOnlyStub:
            def generate(
                self,
                prompt: str,
                *,
                system: str | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
            ) -> LLMResponse:
                assert system == "be terse"
                assert max_tokens == 100
                assert temperature == 0.0
                return _canonical_response()

        port: LLMPort = KwOnlyStub()
        resp = port.generate(
            "hello",
            system="be terse",
            max_tokens=100,
            temperature=0.0,
        )
        assert resp.text == "ok"
