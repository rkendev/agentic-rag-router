"""Behavioural tests for `FallbackModel`.

These tests exercise the full error-routing truth table using stub
`LLMPort` adapters. No real I/O, no structlog output assertions (logging
is a side-effect observable via the CI log pipeline, not a unit-test
concern).

Test matrix (per docs/DECISIONS.md D2):

| Scenario                         | Expected outcome                        |
|----------------------------------|-----------------------------------------|
| tier[0] succeeds                 | return primary response, tier[1] skipped|
| tier[0] transient, tier[1] ok    | return secondary response, failover     |
| tier[0] permanent                | re-raise, tier[1] NOT called            |
| tier[0] content                  | re-raise, tier[1] NOT called            |
| all tiers transient              | raise final LLMTransientError, chained  |
| tier[0] transient, tier[1] perm  | re-raise permanent, no further failover |
| empty tiers list                 | ValueError at construction              |
| single tier succeeds             | return, no failover                     |
| single tier transient            | raise wrapped LLMTransientError         |
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentic_rag_router.application.fallback import FallbackModel
from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier

# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class StubAdapter:
    """Configurable `LLMPort` stub used throughout the test matrix.

    Either returns a canned `LLMResponse` carrying the configured tier, or
    raises the configured exception class. Tracks call count so tests can
    assert that a later tier was (or wasn't) invoked.
    """

    def __init__(
        self,
        *,
        tier: LLMTier = LLMTier.PRIMARY,
        raises: type[Exception] | None = None,
        raise_message: str = "stub failure",
        response_text: str = "ok",
    ) -> None:
        self._tier = tier
        self._raises = raises
        self._raise_message = raise_message
        self._response_text = response_text
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_kwargs: dict[str, object] = {}

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_kwargs = {
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._raises is not None:
            raise self._raises(self._raise_message)
        return LLMResponse(
            text=self._response_text,
            model_name=f"stub-{self._tier.value}",
            tier=self._tier,
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            created_at=datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC),
        )


def test_stub_is_structurally_a_port() -> None:
    """Sanity check: StubAdapter satisfies the LLMPort contract."""
    assert isinstance(StubAdapter(), LLMPort)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFallbackModelConstruction:
    """Empty tiers is invalid; anything non-empty is accepted."""

    def test_empty_tiers_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one tier"):
            FallbackModel([])

    def test_single_tier_accepted(self) -> None:
        fb = FallbackModel([StubAdapter()])
        assert fb is not None

    def test_three_tiers_accepted(self) -> None:
        fb = FallbackModel(
            [
                StubAdapter(tier=LLMTier.PRIMARY),
                StubAdapter(tier=LLMTier.SECONDARY),
                StubAdapter(tier=LLMTier.TERTIARY),
            ]
        )
        assert fb is not None

    def test_is_itself_an_llmport(self) -> None:
        # FallbackModel conforms structurally --- it can be composed into
        # another FallbackModel or passed anywhere an adapter is expected.
        fb = FallbackModel([StubAdapter()])
        assert isinstance(fb, LLMPort)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestFallbackModelHappyPath:
    """First-tier success is the common case --- must not touch later tiers."""

    def test_primary_success_returns_primary_response(self) -> None:
        primary = StubAdapter(tier=LLMTier.PRIMARY, response_text="from primary")
        secondary = StubAdapter(tier=LLMTier.SECONDARY)
        fb = FallbackModel([primary, secondary])

        resp = fb.generate("hello")

        assert resp.text == "from primary"
        assert resp.tier is LLMTier.PRIMARY
        assert primary.call_count == 1
        assert secondary.call_count == 0  # never reached

    def test_single_tier_success_returns_its_response(self) -> None:
        only = StubAdapter(tier=LLMTier.PRIMARY)
        fb = FallbackModel([only])

        resp = fb.generate("hello")

        assert resp.tier is LLMTier.PRIMARY
        assert only.call_count == 1

    def test_kwargs_are_forwarded_verbatim(self) -> None:
        primary = StubAdapter()
        fb = FallbackModel([primary])

        fb.generate(
            "a prompt",
            system="be terse",
            max_tokens=200,
            temperature=0.3,
        )

        assert primary.last_prompt == "a prompt"
        assert primary.last_kwargs == {
            "system": "be terse",
            "max_tokens": 200,
            "temperature": 0.3,
        }


# ---------------------------------------------------------------------------
# Transient failover
# ---------------------------------------------------------------------------


class TestFallbackModelTransientFailover:
    """Transient on tier N -> advance to tier N+1."""

    def test_transient_on_primary_then_secondary_succeeds(self) -> None:
        primary = StubAdapter(tier=LLMTier.PRIMARY, raises=LLMTransientError)
        secondary = StubAdapter(tier=LLMTier.SECONDARY, response_text="fallback win")
        fb = FallbackModel([primary, secondary])

        resp = fb.generate("hello")

        assert resp.text == "fallback win"
        assert resp.tier is LLMTier.SECONDARY
        assert primary.call_count == 1
        assert secondary.call_count == 1

    def test_transient_cascades_through_all_tiers_until_success(self) -> None:
        primary = StubAdapter(tier=LLMTier.PRIMARY, raises=LLMTransientError)
        secondary = StubAdapter(tier=LLMTier.SECONDARY, raises=LLMTransientError)
        tertiary = StubAdapter(tier=LLMTier.TERTIARY, response_text="last resort")
        fb = FallbackModel([primary, secondary, tertiary])

        resp = fb.generate("hello")

        assert resp.tier is LLMTier.TERTIARY
        assert resp.text == "last resort"
        assert primary.call_count == 1
        assert secondary.call_count == 1
        assert tertiary.call_count == 1

    def test_all_tiers_transient_raises_final_transient_chained(self) -> None:
        primary = StubAdapter(raises=LLMTransientError, raise_message="primary timeout")
        secondary = StubAdapter(raises=LLMTransientError, raise_message="secondary 429")
        tertiary = StubAdapter(
            raises=LLMTransientError, raise_message="tertiary connection refused"
        )
        fb = FallbackModel([primary, secondary, tertiary])

        with pytest.raises(LLMTransientError) as exc_info:
            fb.generate("hello")

        # The final raised error references the count and the last message.
        assert "All 3 tiers failed" in str(exc_info.value)
        assert "tertiary connection refused" in str(exc_info.value)
        # Chained to the last transient (exception chaining preserves
        # the original for log correlation).
        assert isinstance(exc_info.value.__cause__, LLMTransientError)
        assert str(exc_info.value.__cause__) == "tertiary connection refused"

        # All tiers were attempted exactly once.
        assert primary.call_count == 1
        assert secondary.call_count == 1
        assert tertiary.call_count == 1

    def test_single_tier_transient_raises_wrapped(self) -> None:
        only = StubAdapter(raises=LLMTransientError, raise_message="network down")
        fb = FallbackModel([only])

        with pytest.raises(LLMTransientError) as exc_info:
            fb.generate("hello")

        assert "All 1 tiers failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, LLMTransientError)


# ---------------------------------------------------------------------------
# Permanent and content: immediate re-raise, no failover
# ---------------------------------------------------------------------------


class TestFallbackModelNonFailoverErrors:
    """Permanent and content errors must NOT trigger failover."""

    def test_permanent_on_primary_reraises_immediately(self) -> None:
        primary = StubAdapter(raises=LLMPermanentError, raise_message="invalid API key")
        secondary = StubAdapter()  # should never be called
        fb = FallbackModel([primary, secondary])

        with pytest.raises(LLMPermanentError, match="invalid API key"):
            fb.generate("hello")

        assert primary.call_count == 1
        assert secondary.call_count == 0

    def test_content_on_primary_reraises_immediately(self) -> None:
        primary = StubAdapter(raises=LLMContentError, raise_message="policy refusal")
        secondary = StubAdapter()
        fb = FallbackModel([primary, secondary])

        with pytest.raises(LLMContentError, match="policy refusal"):
            fb.generate("hello")

        assert primary.call_count == 1
        assert secondary.call_count == 0

    def test_transient_then_permanent_reraises_permanent(self) -> None:
        # A mid-cascade permanent error stops the cascade --- later tiers
        # are not attempted.
        primary = StubAdapter(tier=LLMTier.PRIMARY, raises=LLMTransientError)
        secondary = StubAdapter(
            tier=LLMTier.SECONDARY,
            raises=LLMPermanentError,
            raise_message="bad request",
        )
        tertiary = StubAdapter(tier=LLMTier.TERTIARY)
        fb = FallbackModel([primary, secondary, tertiary])

        with pytest.raises(LLMPermanentError, match="bad request"):
            fb.generate("hello")

        assert primary.call_count == 1
        assert secondary.call_count == 1
        assert tertiary.call_count == 0  # cascade halted

    def test_transient_then_content_reraises_content(self) -> None:
        primary = StubAdapter(raises=LLMTransientError)
        secondary = StubAdapter(raises=LLMContentError, raise_message="empty response")
        tertiary = StubAdapter()
        fb = FallbackModel([primary, secondary, tertiary])

        with pytest.raises(LLMContentError, match="empty response"):
            fb.generate("hello")

        assert primary.call_count == 1
        assert secondary.call_count == 1
        assert tertiary.call_count == 0
