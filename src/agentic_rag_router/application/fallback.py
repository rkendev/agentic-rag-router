"""FallbackModel: tier-based orchestration across multiple LLMPort adapters.

This is the behavioural core of the three-tier strategy (Claude Haiku ->
gpt-4o-mini -> Ollama). `FallbackModel` itself implements the `LLMPort`
protocol, so callers can swap it in wherever a single adapter would go.

Failure routing:

- `LLMTransientError` -> log at WARNING, advance to the next tier.
- `LLMPermanentError` -> log at ERROR, re-raise immediately. Falling over
  would not help (auth / request shape is wrong everywhere) and would mask
  a configuration bug.
- `LLMContentError` -> log at ERROR, re-raise immediately. The same prompt
  will likely trigger the same refusal on the next tier; swallowing it
  would violate the observability contract.
- All tiers exhausted with transient errors -> raise a final
  `LLMTransientError` chained (`raise ... from last`) to preserve the root
  cause for log correlation.

Logging uses a bound `structlog` logger with `tier_index` (0-based position
in the composition list) and `tier_class` (the adapter class name). The
semantic tier label (`LLMTier.PRIMARY` etc.) comes from the adapter's
returned `LLMResponse.tier` and is NOT assumed to match position, so a
caller can reorder adapters without breaking observability.
"""

from __future__ import annotations

from collections.abc import Sequence

import structlog

from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse

_logger = structlog.get_logger(__name__)


class FallbackModel:
    """Composes an ordered list of `LLMPort` adapters with tier failover.

    `FallbackModel` itself conforms structurally to `LLMPort`, so it can be
    passed anywhere a single adapter is expected --- including, recursively,
    another `FallbackModel` (useful for priority groups, though the default
    composition root in `main.py` uses a flat list).
    """

    def __init__(self, tiers: Sequence[LLMPort]) -> None:
        """Construct with an ordered list of adapters.

        Parameters
        ----------
        tiers:
            Non-empty sequence of `LLMPort` instances. Order is priority
            order --- index 0 is tried first. The composition root
            typically passes `[haiku_adapter, gpt4omini_adapter,
            ollama_adapter]`.

        Raises
        ------
        ValueError
            If `tiers` is empty. A zero-tier fallback has no meaningful
            behaviour and silently rejecting every call would be worse
            than failing loud at construction.
        """
        if not tiers:
            msg = "FallbackModel requires at least one tier"
            raise ValueError(msg)
        self._tiers: list[LLMPort] = list(tiers)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Dispatch to the first tier; fail over on transient errors.

        See module docstring for the full failure-routing contract.
        """
        last_transient: LLMTransientError | None = None

        for idx, tier in enumerate(self._tiers):
            tier_log = _logger.bind(
                tier_index=idx,
                tier_class=type(tier).__name__,
            )
            try:
                response = tier.generate(
                    prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except LLMTransientError as exc:
                last_transient = exc
                tier_log.warning(
                    "llm.tier.transient_failure",
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                )
                continue
            except (LLMPermanentError, LLMContentError) as exc:
                tier_log.error(
                    "llm.tier.permanent_failure",
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                )
                raise
            else:
                if idx > 0:
                    # Only log recovery when we actually failed over --- a
                    # successful first-tier call is the common path and
                    # doesn't need a log line per request.
                    tier_log.info(
                        "llm.tier.recovered",
                        served_tier=response.tier.value,
                    )
                return response

        # Loop completed without returning: every tier raised transient.
        # `last_transient` is guaranteed non-None because the tier list is
        # non-empty (checked at __init__) and every iteration either
        # returned or captured a transient error. The guard below is
        # defensive against future refactors; it cannot be reached today.
        if last_transient is None:  # pragma: no cover
            msg = (
                "FallbackModel internal invariant violated: "
                "no transient error captured after non-empty tier iteration"
            )
            raise RuntimeError(msg)
        _logger.error(
            "llm.all_tiers_exhausted",
            tier_count=len(self._tiers),
            last_error_class=type(last_transient).__name__,
            last_error_message=str(last_transient),
        )
        msg = (
            f"All {len(self._tiers)} tiers failed with transient errors; "
            f"last error: {last_transient}"
        )
        raise LLMTransientError(msg) from last_transient
