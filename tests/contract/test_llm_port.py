"""Parametrized contract tests for every `LLMPort` implementation.

Proves the UNIFORM shape of the port across every adapter we ship:

  * `isinstance(adapter, LLMPort)` is structurally true for all of them
  * the healthy path returns an `LLMResponse` whose `tier` matches what
    the adapter declares (so `FallbackModel` can trust the tier field
    for observability)
  * empty / whitespace prompts raise `ValueError` or
    `LLMPermanentError` --- both are permitted by the `LLMPort`
    docstring, and the contract suite is agnostic about which one a
    given adapter picks
  * every SDK-level failure translates to exactly one of the three
    domain error classes: `LLMTransientError` (recoverable -->
    next tier), `LLMPermanentError` (don't retry), or
    `LLMContentError` (call succeeded but the content is unusable).
    No vendor SDK exception is allowed to escape.

The per-adapter unit tests
(`tests/unit/infrastructure/test_*_adapter.py`) still cover the
exhaustive per-vendor error-translation truth tables (5xx vs 4xx,
rate limit vs auth vs not-found, etc.) The contract suite is the
drift detector: if some future adapter edit breaks the `LLMPort`
contract, the matching contract test fails for that adapter while the
fake keeps passing --- and the failure pattern immediately points at
the adapter rather than the test scaffolding.

Parametrization is driven by `LLM_ADAPTERS` in `conftest.py`. Adding
another adapter is a one-entry change in that list and the whole test
grid runs against it automatically. Today's grid is
8 contract tests * 4 adapters = 32 test cases.
"""

from __future__ import annotations

import pytest

from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.errors import (
    LLMContentError,
    LLMPermanentError,
    LLMTransientError,
)
from agentic_rag_router.domain.llm import LLMResponse
from tests.contract.fakes import AdapterSpec


@pytest.mark.contract
class TestLLMPortContract:
    """Every `LLMPort` implementation must honour the same contract."""

    # -----------------------------------------------------------------
    # Structural / happy-path assertions
    # -----------------------------------------------------------------

    def test_is_llmport(self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        # `@runtime_checkable` lets `isinstance` inspect method presence.
        # It's a weaker guarantee than mypy's static structural check but
        # catches "adapter forgot to implement generate()" at test time.
        adapter = adapter_spec.build(monkeypatch)
        assert isinstance(adapter, LLMPort)

    def test_generate_returns_llm_response(
        self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = adapter_spec.build(monkeypatch)
        result = adapter.generate("hello")
        assert isinstance(result, LLMResponse)
        # `LLMResponse` already enforces min_length=1 and a
        # whitespace-only validator, so reaching here means the text is
        # real. Asserting anyway to make the contract explicit.
        assert result.text.strip() != ""

    def test_tier_attribute_matches(
        self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The tier declared by the adapter must flow through to the
        # returned response. FallbackModel relies on this for logging
        # which tier actually served a request.
        adapter = adapter_spec.build(monkeypatch)
        result = adapter.generate("hello")
        assert result.tier == adapter_spec.tier

    # -----------------------------------------------------------------
    # Prompt validation
    # -----------------------------------------------------------------

    def test_empty_prompt_raises(
        self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Per LLMPort.generate docstring, empty prompts may raise either
        # `ValueError` or `LLMPermanentError`. Either is compliant.
        adapter = adapter_spec.build(monkeypatch)
        with pytest.raises((ValueError, LLMPermanentError)):
            adapter.generate("")

    def test_whitespace_prompt_raises(
        self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = adapter_spec.build(monkeypatch)
        with pytest.raises((ValueError, LLMPermanentError)):
            adapter.generate("   \n\t")

    # -----------------------------------------------------------------
    # SDK error translation --- the core of the contract.
    # -----------------------------------------------------------------

    def test_transient_sdk_error_becomes_domain(
        self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each adapter's `inject_transient` picks a characteristic
        # SDK-layer transient failure (rate-limit for cloud, connection
        # error for local). The contract only cares that it translates
        # to `LLMTransientError` and nothing vendor-specific escapes.
        adapter = adapter_spec.build(monkeypatch)
        adapter_spec.inject_transient(adapter, monkeypatch)
        with pytest.raises(LLMTransientError):
            adapter.generate("hello")

    def test_permanent_sdk_error_becomes_domain(
        self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Characteristic permanent: auth failure (cloud) / model-not-pulled
        # (local). Both say "don't retry, don't fail over".
        adapter = adapter_spec.build(monkeypatch)
        adapter_spec.inject_permanent(adapter, monkeypatch)
        with pytest.raises(LLMPermanentError):
            adapter.generate("hello")

    def test_content_failure_becomes_domain(
        self, adapter_spec: AdapterSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Characteristic content failure: well-formed SDK success with
        # whitespace-only text. Works uniformly across all three real
        # adapters because each one's content validator rejects
        # whitespace-only output.
        adapter = adapter_spec.build(monkeypatch)
        adapter_spec.inject_content(adapter, monkeypatch)
        with pytest.raises(LLMContentError):
            adapter.generate("hello")
