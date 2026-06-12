"""Tests for the composition root (`build_llm`).

`main()` itself is a smoke-test CLI --- it's excluded from coverage
because exercising it requires a live API call. The composition
function `build_llm` is the one with non-trivial logic and is fully
testable offline: we construct `Settings` instances with explicit
kwargs (bypassing env / .env) and observe which `LLMPort` flavour
comes back.

Scope:

  * `primary` returns the concrete `AnthropicAdapter` (no fallback wrapper).
  * `secondary` returns the concrete `OpenAIAdapter` (no fallback wrapper).
  * `tertiary` returns the concrete `OllamaAdapter` (no fallback wrapper).
  * `fallback` returns a `FallbackModel` containing every tier whose
    preconditions are met. Ollama is appended unconditionally (no
    credentials to gate on) --- so the tier count is 1, 2, or 3
    depending on which cloud keys are present.
  * Missing credentials for a single-tier cloud selection are surfaced
    by the adapter's own fail-fast (not silently swallowed here). The
    tertiary tier has no equivalent fail-fast: `OllamaAdapter`
    construction always succeeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from agentic_rag_router.application.fallback import FallbackModel
from agentic_rag_router.application.ports import LLMPort
from agentic_rag_router.domain.errors import LLMPermanentError
from agentic_rag_router.infrastructure.anthropic_adapter import AnthropicAdapter
from agentic_rag_router.infrastructure.ollama_adapter import OllamaAdapter
from agentic_rag_router.infrastructure.openai_adapter import OpenAIAdapter
from agentic_rag_router.infrastructure.settings import Settings
from agentic_rag_router.main import build_llm

# ---------------------------------------------------------------------------
# Fixtures: isolate env so Roy's real shell can't bleed into these tests.
# ---------------------------------------------------------------------------

_SETTINGS_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_MODEL",
    "OPENAI_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LOG_LEVEL",
    "LLM_TIER",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Build Settings with `_env_file` pinned to an empty tmp file.

    Any field not in `overrides` takes its declared default. Fake
    Anthropic and OpenAI keys are provided by default so most tests
    don't have to repeat themselves; tests that specifically verify
    missing-key behaviour pass `anthropic_api_key=None` and/or
    `openai_api_key=None` explicitly. Ollama settings take their
    declared defaults (`http://localhost:11434`, `llama3.2:3b`).
    """
    empty_env = tmp_path / ".env.empty"
    empty_env.write_text("")
    base: dict[str, Any] = {
        "anthropic_api_key": SecretStr("sk-ant-test-fake"),
        "openai_api_key": SecretStr("sk-oai-test-fake"),
    }
    base.update(overrides)
    return Settings(_env_file=empty_env, **base)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# `primary` --- single-tier Anthropic, no fallback wrapper.
# ---------------------------------------------------------------------------


class TestBuildLlmPrimary:
    def test_returns_anthropic_adapter_directly(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings = _make_settings(tmp_path, llm_tier="primary")
        llm = build_llm(settings)
        assert isinstance(llm, AnthropicAdapter)

    def test_is_an_llmport(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Structural check: callers only rely on the LLMPort interface,
        # never the concrete adapter class, so this contract is what
        # matters day-to-day.
        settings = _make_settings(tmp_path, llm_tier="primary")
        assert isinstance(build_llm(settings), LLMPort)

    def test_missing_key_raises_permanent(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Fail-fast per D4: the AnthropicAdapter's own constructor
        # check fires before any network call.
        settings = _make_settings(tmp_path, llm_tier="primary", anthropic_api_key=None)
        with pytest.raises(LLMPermanentError, match="API key"):
            build_llm(settings)

    def test_uses_configured_model(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _make_settings(
            tmp_path,
            llm_tier="primary",
            anthropic_model="claude-sonnet-4-6",
        )
        adapter = build_llm(settings)
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter._model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# `secondary` --- single-tier OpenAI, no fallback wrapper.
# ---------------------------------------------------------------------------


class TestBuildLlmSecondary:
    def test_returns_openai_adapter_directly(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings = _make_settings(tmp_path, llm_tier="secondary")
        llm = build_llm(settings)
        assert isinstance(llm, OpenAIAdapter)

    def test_is_an_llmport(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, llm_tier="secondary")
        assert isinstance(build_llm(settings), LLMPort)

    def test_missing_key_raises_permanent(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Fail-fast per D4: OpenAIAdapter's own constructor check fires
        # before any network call.
        settings = _make_settings(tmp_path, llm_tier="secondary", openai_api_key=None)
        with pytest.raises(LLMPermanentError, match="API key"):
            build_llm(settings)

    def test_uses_configured_model(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _make_settings(
            tmp_path,
            llm_tier="secondary",
            openai_model="gpt-4o",
        )
        adapter = build_llm(settings)
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter._model == "gpt-4o"


# ---------------------------------------------------------------------------
# `tertiary` --- single-tier Ollama, no fallback wrapper.
# ---------------------------------------------------------------------------


class TestBuildLlmTertiary:
    def test_returns_ollama_adapter_directly(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings = _make_settings(tmp_path, llm_tier="tertiary")
        llm = build_llm(settings)
        assert isinstance(llm, OllamaAdapter)

    def test_is_an_llmport(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, llm_tier="tertiary")
        assert isinstance(build_llm(settings), LLMPort)

    def test_no_credentials_required(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Unlike primary/secondary, tertiary builds successfully even
        # with both cloud keys absent --- Ollama has no auth.
        settings = _make_settings(
            tmp_path,
            llm_tier="tertiary",
            anthropic_api_key=None,
            openai_api_key=None,
        )
        llm = build_llm(settings)
        assert isinstance(llm, OllamaAdapter)

    def test_uses_configured_host_and_model(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings = _make_settings(
            tmp_path,
            llm_tier="tertiary",
            ollama_host="http://vps-gpu:11434",
            ollama_model="llama3.1:8b",
        )
        adapter = build_llm(settings)
        assert isinstance(adapter, OllamaAdapter)
        assert adapter._host == "http://vps-gpu:11434"
        assert adapter._model == "llama3.1:8b"


# ---------------------------------------------------------------------------
# `fallback` --- wraps available tiers in FallbackModel.
# ---------------------------------------------------------------------------


class TestBuildLlmFallback:
    def test_returns_fallback_model(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, llm_tier="fallback")
        llm = build_llm(settings)
        assert isinstance(llm, FallbackModel)

    def test_three_tiers_when_both_cloud_keys_present(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # With both cloud keys set the composition produces a
        # FallbackModel whose tiers are Anthropic (primary), OpenAI
        # (secondary), then Ollama (tertiary), in that order.
        settings = _make_settings(tmp_path, llm_tier="fallback")
        llm = build_llm(settings)
        assert isinstance(llm, FallbackModel)
        assert len(llm._tiers) == 3
        assert isinstance(llm._tiers[0], AnthropicAdapter)
        assert isinstance(llm._tiers[1], OpenAIAdapter)
        assert isinstance(llm._tiers[2], OllamaAdapter)

    def test_two_tiers_when_only_anthropic_key_present(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Anthropic (primary) + Ollama (tertiary), skipping OpenAI.
        settings = _make_settings(tmp_path, llm_tier="fallback", openai_api_key=None)
        llm = build_llm(settings)
        assert isinstance(llm, FallbackModel)
        assert len(llm._tiers) == 2
        assert isinstance(llm._tiers[0], AnthropicAdapter)
        assert isinstance(llm._tiers[1], OllamaAdapter)

    def test_two_tiers_when_only_openai_key_present(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # OpenAI (secondary) + Ollama (tertiary), skipping Anthropic.
        settings = _make_settings(tmp_path, llm_tier="fallback", anthropic_api_key=None)
        llm = build_llm(settings)
        assert isinstance(llm, FallbackModel)
        assert len(llm._tiers) == 2
        assert isinstance(llm._tiers[0], OpenAIAdapter)
        assert isinstance(llm._tiers[1], OllamaAdapter)

    def test_one_tier_when_no_cloud_keys_present(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # With neither cloud key set, Ollama alone composes a single-tier
        # FallbackModel. No NotImplementedError --- Ollama is always
        # available as the no-credentials local option. If the daemon
        # isn't actually running, the failure surfaces at generate()
        # time as LLMTransientError, which is the honest runtime signal.
        settings = _make_settings(
            tmp_path,
            llm_tier="fallback",
            anthropic_api_key=None,
            openai_api_key=None,
        )
        llm = build_llm(settings)
        assert isinstance(llm, FallbackModel)
        assert len(llm._tiers) == 1
        assert isinstance(llm._tiers[0], OllamaAdapter)
