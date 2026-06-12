"""Runtime configuration loaded from environment variables and `.env`.

API keys are wrapped in `pydantic.SecretStr` so that `print(settings)` and
the structlog default serialization cannot leak them. Adapters that need a
key call `.get_secret_value()` explicitly at construction time.

Design choices (see docs/DECISIONS.md D4):

- Empty-string API keys coerce to `None` so users can `export KEY=` to
  disable a tier without removing the variable. Adapters check for `None`
  and raise `LLMPermanentError` at build-time if the tier is requested
  but its key is missing --- fail-fast over silent misconfiguration.
- `extra="ignore"` means stray env vars don't break the app (e.g. when
  running alongside other tools that set their own `ANTHROPIC_*` vars).
- Bounds on `llm_timeout_seconds` and `llm_max_retries` are loose
  guardrails, not policy --- operators can tune within the range without
  touching code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
"""Valid values for `log_level`. Matches stdlib `logging` level names."""

TierSelection = Literal["primary", "secondary", "tertiary", "fallback"]
"""Valid values for `llm_tier` (the example CLI's tier selector).

`fallback` runs the full three-tier `FallbackModel` composition; the other
three run a single adapter directly, bypassing failover.
"""


class Settings(BaseSettings):
    """Application configuration.

    Constructing `Settings()` reads environment variables first, then any
    values in a `.env` file next to the current working directory. The
    test suite overrides via `monkeypatch.setenv(...)`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Case-insensitive matching: ANTHROPIC_API_KEY and anthropic_api_key
        # both populate `anthropic_api_key`. Keeps .env portable.
        case_sensitive=False,
    )

    # --- API keys (optional; empty string -> None) -------------------------

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Claude API key. Empty means PRIMARY tier is disabled.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key. Empty means SECONDARY tier is disabled.",
    )

    # --- Hosted model identifiers -------------------------------------------

    anthropic_model: str = Field(
        default="claude-haiku-4-5-20251001",
        min_length=1,
        description=(
            "Anthropic model identifier for the PRIMARY tier. Overridable "
            "so template forks can point at a different Claude model "
            "without touching code."
        ),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description=(
            "OpenAI chat-completion model identifier for the SECONDARY "
            "tier. Overridable so template forks can point at a different "
            "OpenAI model (e.g. gpt-4o) without touching code."
        ),
    )

    # --- Ollama (local, no auth) -------------------------------------------

    ollama_host: str = Field(
        default="http://localhost:11434",
        min_length=1,
        description="Base URL for the local Ollama HTTP API.",
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        min_length=1,
        description="Model identifier pulled into Ollama.",
    )

    # --- Adapter behaviour --------------------------------------------------

    llm_timeout_seconds: int = Field(
        default=30,
        gt=0,
        le=600,
        description="Per-request timeout. Upper bound guards against hangs.",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Retry budget per adapter before surfacing transient.",
    )

    # --- Observability ------------------------------------------------------

    log_level: LogLevel = Field(
        default="INFO",
        description="structlog / stdlib logging threshold.",
    )

    # --- Example CLI tier selection -----------------------------------------

    llm_tier: TierSelection = Field(
        default="fallback",
        description=(
            "Which tier (or `fallback` for the full composition) the "
            "example CLI invokes. Does not affect library use."
        ),
    )

    @field_validator("anthropic_api_key", "openai_api_key", mode="before")
    @classmethod
    def _empty_string_is_none(cls, value: str | None) -> str | None:
        """Coerce empty-string env values to `None`.

        Without this, `ANTHROPIC_API_KEY=` in `.env` would deserialize into
        `SecretStr("")`, which would silently make `.get_secret_value()`
        return an empty string --- and the adapter would then hit a 401
        at runtime instead of failing at build-time. Coercing to `None`
        means the adapter's `None`-guard fires immediately.
        """
        if isinstance(value, str) and value.strip() == "":
            return None
        return value
