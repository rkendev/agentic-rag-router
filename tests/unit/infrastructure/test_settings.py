"""Tests for the `Settings` pydantic-settings model.

Each test isolates env via pytest's `monkeypatch` so results don't depend
on Roy's shell or whatever CI runner happens to be in use. When testing
`.env` file loading we use `tmp_path` + the `_env_file` kwarg.

The scope is:

  * Defaults apply when no env is set.
  * Env var override works for every declared field.
  * Empty-string API keys coerce to `None`.
  * `SecretStr` obscures values in `repr` / `str`.
  * Validation errors fire for out-of-range / invalid-enum input.
  * `.env` file loading works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from agentic_rag_router.infrastructure.settings import Settings

# ---------------------------------------------------------------------------
# Fixture: a monkeypatch that clears every Settings-relevant env var.
# Prevents Roy's real shell environment from bleeding into tests.
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
    """Strip every settings-relevant env var before each test."""
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Also point _env_file at a non-existent path so ambient .env files in
    # the test runner's CWD (e.g. Roy's real .env) cannot bleed in.
    return monkeypatch


def _build_settings(tmp_path: Path) -> Settings:
    """Construct Settings with env_file pinned to an empty temp file.

    Guarantees no ambient .env interferes with the test environment.
    """
    empty_env = tmp_path / ".env.empty"
    empty_env.write_text("")
    return Settings(_env_file=empty_env)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    """No env set + no .env file -> all defaults apply."""

    def test_api_keys_default_to_none(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _build_settings(tmp_path)
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None

    def test_ollama_defaults(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _build_settings(tmp_path)
        assert settings.ollama_host == "http://localhost:11434"
        assert settings.ollama_model == "llama3.2:3b"

    def test_adapter_behaviour_defaults(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings = _build_settings(tmp_path)
        assert settings.llm_timeout_seconds == 30
        assert settings.llm_max_retries == 2

    def test_observability_defaults(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _build_settings(tmp_path)
        assert settings.log_level == "INFO"

    def test_tier_selection_default(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _build_settings(tmp_path)
        assert settings.llm_tier == "fallback"

    def test_anthropic_model_default(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _build_settings(tmp_path)
        assert settings.anthropic_model == "claude-haiku-4-5-20251001"

    def test_openai_model_default(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        settings = _build_settings(tmp_path)
        assert settings.openai_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------


class TestSettingsEnvOverrides:
    """Env vars override the declared defaults for every field."""

    def test_api_keys_from_env(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        clean_env.setenv("OPENAI_API_KEY", "sk-oai-fake")
        settings = _build_settings(tmp_path)

        assert settings.anthropic_api_key is not None
        assert settings.anthropic_api_key.get_secret_value() == "sk-ant-fake"
        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-oai-fake"

    def test_ollama_from_env(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("OLLAMA_HOST", "http://ollama.internal:11434")
        clean_env.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        settings = _build_settings(tmp_path)

        assert settings.ollama_host == "http://ollama.internal:11434"
        assert settings.ollama_model == "qwen2.5:7b"

    def test_numeric_fields_from_env(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("LLM_TIMEOUT_SECONDS", "60")
        clean_env.setenv("LLM_MAX_RETRIES", "5")
        settings = _build_settings(tmp_path)

        assert settings.llm_timeout_seconds == 60
        assert settings.llm_max_retries == 5

    def test_log_level_from_env(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("LOG_LEVEL", "DEBUG")
        settings = _build_settings(tmp_path)
        assert settings.log_level == "DEBUG"

    def test_tier_selection_from_env(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("LLM_TIER", "primary")
        settings = _build_settings(tmp_path)
        assert settings.llm_tier == "primary"

    def test_anthropic_model_from_env(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        settings = _build_settings(tmp_path)
        assert settings.anthropic_model == "claude-sonnet-4-6"

    def test_openai_model_from_env(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("OPENAI_MODEL", "gpt-4o")
        settings = _build_settings(tmp_path)
        assert settings.openai_model == "gpt-4o"

    def test_case_insensitive_env_matching(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # `case_sensitive=False` in model_config: lowercase works too.
        clean_env.setenv("anthropic_api_key", "sk-ant-lower")
        settings = _build_settings(tmp_path)

        assert settings.anthropic_api_key is not None
        assert settings.anthropic_api_key.get_secret_value() == "sk-ant-lower"


# ---------------------------------------------------------------------------
# Empty-string API key coercion (the fail-fast design per D4)
# ---------------------------------------------------------------------------


class TestEmptyStringCoercion:
    """Empty API keys become `None` so adapters fail-fast instead of 401."""

    def test_empty_anthropic_key_coerces_to_none(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clean_env.setenv("ANTHROPIC_API_KEY", "")
        settings = _build_settings(tmp_path)
        assert settings.anthropic_api_key is None

    def test_whitespace_anthropic_key_coerces_to_none(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A tab or spaces should count as empty too --- users sometimes
        # accidentally `export KEY=" "` when clearing.
        clean_env.setenv("ANTHROPIC_API_KEY", "   ")
        settings = _build_settings(tmp_path)
        assert settings.anthropic_api_key is None

    def test_empty_openai_key_coerces_to_none(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clean_env.setenv("OPENAI_API_KEY", "")
        settings = _build_settings(tmp_path)
        assert settings.openai_api_key is None


# ---------------------------------------------------------------------------
# SecretStr behaviour
# ---------------------------------------------------------------------------


class TestSecretStrObscures:
    """API keys must NOT appear in repr or str output."""

    def test_repr_does_not_leak_key(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
        settings = _build_settings(tmp_path)

        rendered = repr(settings) + str(settings)
        assert "supersecret" not in rendered
        # SecretStr canonical redaction
        assert "**********" in rendered

    def test_get_secret_value_returns_plaintext(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clean_env.setenv("OPENAI_API_KEY", "sk-oai-plain")
        settings = _build_settings(tmp_path)

        assert settings.openai_api_key is not None
        assert isinstance(settings.openai_api_key, SecretStr)
        assert settings.openai_api_key.get_secret_value() == "sk-oai-plain"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Bad input fails fast at construction."""

    def test_negative_timeout_rejected(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("LLM_TIMEOUT_SECONDS", "-1")
        with pytest.raises(ValidationError):
            _build_settings(tmp_path)

    def test_zero_timeout_rejected(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # `gt=0` --- zero is not a valid timeout.
        clean_env.setenv("LLM_TIMEOUT_SECONDS", "0")
        with pytest.raises(ValidationError):
            _build_settings(tmp_path)

    def test_excessive_timeout_rejected(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # `le=600` --- anything larger is a likely misconfiguration.
        clean_env.setenv("LLM_TIMEOUT_SECONDS", "601")
        with pytest.raises(ValidationError):
            _build_settings(tmp_path)

    def test_negative_retries_rejected(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        clean_env.setenv("LLM_MAX_RETRIES", "-1")
        with pytest.raises(ValidationError):
            _build_settings(tmp_path)

    def test_invalid_log_level_rejected(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clean_env.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            _build_settings(tmp_path)

    def test_invalid_tier_selection_rejected(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clean_env.setenv("LLM_TIER", "quaternary")
        with pytest.raises(ValidationError):
            _build_settings(tmp_path)

    def test_empty_ollama_host_rejected(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clean_env.setenv("OLLAMA_HOST", "")
        with pytest.raises(ValidationError):
            _build_settings(tmp_path)


# ---------------------------------------------------------------------------
# .env file loading
# ---------------------------------------------------------------------------


class TestEnvFileLoading:
    """A `.env` file at the configured path populates the settings."""

    def test_env_file_values_loaded(self, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "ANTHROPIC_API_KEY=sk-from-envfile\n" "OLLAMA_MODEL=phi3:mini\n" "LOG_LEVEL=WARNING\n"
        )

        settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

        assert settings.anthropic_api_key is not None
        assert settings.anthropic_api_key.get_secret_value() == "sk-from-envfile"
        assert settings.ollama_model == "phi3:mini"
        assert settings.log_level == "WARNING"

    def test_env_var_overrides_env_file(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Real env vars take precedence over .env (standard
        # pydantic-settings layering).
        env_file = tmp_path / ".env"
        env_file.write_text("LOG_LEVEL=WARNING\n")
        clean_env.setenv("LOG_LEVEL", "ERROR")

        settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

        assert settings.log_level == "ERROR"
