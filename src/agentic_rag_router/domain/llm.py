"""Domain primitives for the LLM layer.

Contains the tier enum and the frozen response model that every adapter must
return. Pure domain: no I/O, no third-party SDKs, no imports from
`application` or `infrastructure`. See SPECIFICATION.md §2 and docs/DECISIONS.md D2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMTier(StrEnum):
    """Which tier of the fallback stack served a request.

    Values are stable string identifiers so they can cross the boundary into
    logs, dashboards, and serialized responses without translation.
    """

    PRIMARY = "primary"  # Claude Haiku
    SECONDARY = "secondary"  # gpt-4o-mini
    TERTIARY = "tertiary"  # Ollama local


class LLMResponse(BaseModel):
    """The structured response every `LLMPort.generate()` call returns.

    Frozen (immutable) and extra-forbidding so callers can rely on the shape
    and adapters cannot smuggle extra fields. `tier` indicates which adapter
    actually served the request — critical for `FallbackModel` observability
    (D2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: Annotated[str, Field(min_length=1)]
    model_name: Annotated[str, Field(min_length=1)]
    tier: LLMTier
    tokens_in: Annotated[int, Field(ge=0)]
    tokens_out: Annotated[int, Field(ge=0)]
    latency_ms: Annotated[int, Field(ge=0)]
    created_at: datetime

    @field_validator("text")
    @classmethod
    def _text_must_not_be_whitespace_only(cls, value: str) -> str:
        """Reject text that is non-empty but contains only whitespace.

        Field(min_length=1) catches empty strings. This validator catches
        strings like "   " or "\n\t" that pass the length check but carry no
        signal. Empty content is a `LLMContentError` concern at the adapter
        layer; once it reaches the domain response it must be real content.
        """
        if not value.strip():
            msg = "text must not be whitespace-only"
            raise ValueError(msg)
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_utc(cls, value: datetime) -> datetime:
        """Require tz-aware UTC.

        Naive datetimes are ambiguous across hosts / CI runners — refusing
        them at construction prevents silent tz drift from reaching logs
        and dashboards. Non-UTC tz-aware values are also rejected to keep
        downstream arithmetic (latency windows, TTLs) unambiguous.
        """
        if value.tzinfo is None:
            msg = "created_at must be tz-aware (got naive datetime)"
            raise ValueError(msg)
        if value.utcoffset() != UTC.utcoffset(None):
            msg = f"created_at must be UTC (got offset {value.utcoffset()})"
            raise ValueError(msg)
        return value
