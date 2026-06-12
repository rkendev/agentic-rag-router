"""Unit tests for agentic_rag_router.domain.llm and agentic_rag_router.domain.errors.

These tests are pure: no I/O, no mocks, no third-party services. They cover
every validator branch in the domain so regressions to the core contract
surface immediately. Coverage goal per T004 DoD: 100% of domain lines.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse, LLMTier

# ---------------------------------------------------------------------------
# LLMTier
# ---------------------------------------------------------------------------


class TestLLMTier:
    """LLMTier is a StrEnum — members are their string values."""

    def test_three_tiers_exist(self) -> None:
        assert {t.value for t in LLMTier} == {"primary", "secondary", "tertiary"}

    def test_str_enum_equality(self) -> None:
        # StrEnum members compare equal to their string values (this is the
        # whole point of using StrEnum over Enum — serialization is free).
        # mypy doesn't see StrEnum's str inheritance and flags these as
        # non-overlapping; the runtime equality is exactly what we're testing.
        assert LLMTier.PRIMARY == "primary"  # type: ignore[comparison-overlap]
        assert LLMTier.SECONDARY == "secondary"  # type: ignore[comparison-overlap]
        assert LLMTier.TERTIARY == "tertiary"  # type: ignore[comparison-overlap]

    def test_comparison_is_lexicographic_not_semantic(self) -> None:
        # StrEnum inherits from `str`, so `<` / `>` are valid and compare the
        # underlying string values. This is NOT a tier-precedence ordering —
        # PRIMARY < SECONDARY happens to be True because "p" < "s"
        # alphabetically, which coincidentally aligns with tier order here.
        # Callers must never rely on `<` to mean "higher-priority tier";
        # if the string values are ever renamed the coincidence breaks.
        # Use explicit `tier is LLMTier.PRIMARY` checks instead.
        assert LLMTier.PRIMARY < LLMTier.SECONDARY  # "primary" < "secondary"
        assert LLMTier.SECONDARY < LLMTier.TERTIARY  # "secondary" < "tertiary"


# ---------------------------------------------------------------------------
# LLMResponse — valid construction
# ---------------------------------------------------------------------------


def _valid_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a kwargs dict that constructs a valid LLMResponse.

    Tests override individual fields to exercise specific validators.
    """
    base: dict[str, Any] = {
        "text": "Hello, world.",
        "model_name": "claude-haiku-4-5-20251001",
        "tier": LLMTier.PRIMARY,
        "tokens_in": 10,
        "tokens_out": 5,
        "latency_ms": 250,
        "created_at": datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return base


class TestLLMResponseValid:
    """Positive cases — a well-formed response round-trips cleanly."""

    def test_constructs_with_canonical_inputs(self) -> None:
        resp = LLMResponse(**_valid_kwargs())
        assert resp.text == "Hello, world."
        assert resp.tier is LLMTier.PRIMARY
        assert resp.tokens_in == 10
        assert resp.tokens_out == 5
        assert resp.latency_ms == 250
        assert resp.created_at.tzinfo is UTC

    def test_zero_tokens_and_zero_latency_allowed(self) -> None:
        # Zero is a legitimate value — some mocked adapters may report it,
        # and the contract is ">= 0" not "> 0".
        resp = LLMResponse(**_valid_kwargs(tokens_in=0, tokens_out=0, latency_ms=0))
        assert resp.tokens_in == resp.tokens_out == resp.latency_ms == 0

    def test_all_three_tiers_accepted(self) -> None:
        for tier in LLMTier:
            resp = LLMResponse(**_valid_kwargs(tier=tier))
            assert resp.tier is tier


# ---------------------------------------------------------------------------
# LLMResponse — frozen and extra-forbidding
# ---------------------------------------------------------------------------


class TestLLMResponseImmutability:
    """Frozen + extra="forbid" — the core invariants that keep callers safe."""

    def test_cannot_mutate_field_after_construction(self) -> None:
        resp = LLMResponse(**_valid_kwargs())
        with pytest.raises(ValidationError):
            resp.text = "mutated"

    def test_extra_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(**_valid_kwargs(extra_field="nope"))
        # Pydantic's error message mentions the offending key.
        assert "extra_field" in str(exc_info.value)


# ---------------------------------------------------------------------------
# LLMResponse — text validator
# ---------------------------------------------------------------------------


class TestLLMResponseText:
    """The text field must carry real content."""

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(**_valid_kwargs(text=""))

    @pytest.mark.parametrize("whitespace", [" ", "\t", "\n", " \n\t "])
    def test_whitespace_only_rejected(self, whitespace: str) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(**_valid_kwargs(text=whitespace))

    def test_leading_trailing_whitespace_kept(self) -> None:
        # The validator rejects empty/whitespace-only, but does not strip.
        # Adapters should return content verbatim.
        resp = LLMResponse(**_valid_kwargs(text="  real content  "))
        assert resp.text == "  real content  "


# ---------------------------------------------------------------------------
# LLMResponse — numeric validators
# ---------------------------------------------------------------------------


class TestLLMResponseNumbers:
    """tokens_in, tokens_out, latency_ms must all be non-negative."""

    @pytest.mark.parametrize("field", ["tokens_in", "tokens_out", "latency_ms"])
    def test_negative_value_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(**_valid_kwargs(**{field: -1}))

    def test_model_name_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(**_valid_kwargs(model_name=""))


# ---------------------------------------------------------------------------
# LLMResponse — created_at tz validator
# ---------------------------------------------------------------------------


class TestLLMResponseCreatedAt:
    """created_at must be tz-aware UTC — no naive, no non-UTC offsets."""

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(**_valid_kwargs(created_at=datetime(2026, 4, 18, 12, 0, 0)))
        assert "tz-aware" in str(exc_info.value)

    def test_non_utc_offset_rejected(self) -> None:
        est = timezone(timedelta(hours=-5))
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(**_valid_kwargs(created_at=datetime(2026, 4, 18, 12, 0, 0, tzinfo=est)))
        assert "UTC" in str(exc_info.value)

    def test_utc_explicit_accepted(self) -> None:
        resp = LLMResponse(**_valid_kwargs(created_at=datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)))
        assert resp.created_at.tzinfo is UTC


# ---------------------------------------------------------------------------
# LLMError hierarchy
# ---------------------------------------------------------------------------


class TestLLMErrorHierarchy:
    """All three concrete errors subclass LLMError and are distinct from each other."""

    @pytest.mark.parametrize(
        "cls",
        [LLMTransientError, LLMPermanentError, LLMContentError],
    )
    def test_subclass_of_llmerror(self, cls: type[LLMError]) -> None:
        assert issubclass(cls, LLMError)
        assert issubclass(cls, Exception)

    def test_sibling_classes_are_distinct(self) -> None:
        # A transient error is not a permanent error, and vice versa.
        # FallbackModel relies on this to decide whether to fail over.
        assert not issubclass(LLMTransientError, LLMPermanentError)
        assert not issubclass(LLMPermanentError, LLMTransientError)
        assert not issubclass(LLMContentError, LLMTransientError)
        assert not issubclass(LLMContentError, LLMPermanentError)

    def test_errors_can_carry_messages(self) -> None:
        err = LLMTransientError("rate limited by upstream")
        assert str(err) == "rate limited by upstream"
        assert isinstance(err, LLMError)

    def test_base_error_is_raisable(self) -> None:
        # Nothing prevents raising LLMError directly, but the contract is
        # that adapters raise a subclass. We verify the base is a real
        # exception class for `except LLMError:` to work.
        with pytest.raises(LLMError):
            raise LLMTransientError("boom")
