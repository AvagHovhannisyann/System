"""Application configuration via pydantic-settings.

All configuration is environment-driven. Field names map to upper-case
environment variables (``database_url`` <- ``DATABASE_URL``); a local
``.env`` file is read when present. Assumes the process is launched from
the repository root (the ``.env`` path is resolved relative to the CWD).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DISTRIBUTION_NAME = "quant-research-platform"


def _package_version() -> str:
    """Return the installed version of this distribution.

    Assumes the project was installed (``uv sync`` installs it into the
    virtualenv). Falls back to ``"0.0.0+unknown"`` when the distribution
    metadata is unavailable (e.g. sources used without installation) so
    that configuration loading never fails on a metadata lookup.
    """
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


class Settings(BaseSettings):
    """Runtime settings for the backend.

    Values come from (highest precedence first): constructor arguments,
    environment variables, a ``.env`` file, then the defaults below.
    Unknown environment entries are ignored. ``app_version`` defaults to
    the installed package version but may be overridden via ``APP_VERSION``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://quant:quant@localhost:5432/quant"
    redis_url: str = "redis://localhost:6379/0"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_version: str = Field(default_factory=_package_version)

    sec_user_agent: str | None = None
    """Contact string sent as ``User-Agent`` to SEC EDGAR.

    SEC's fair-access policy requires every automated requester to identify
    itself with contact information. There is deliberately **no default**:
    the EDGAR connector refuses to run when this is unset rather than
    sending an anonymous or invented identifier (policy compliance is
    fail-closed, and a fabricated contact would be worse than none).
    """

    secrets_kek: SecretStr | None = None
    """Key-encryption key (Fernet, urlsafe-base64 32 bytes) from the environment.

    Wraps provider API keys at rest (§7, I5). Sourced **only** from the
    environment — never from the database, never committed — so a database
    dump alone cannot decrypt stored credentials. ``SecretStr`` so the value
    cannot leak through ``repr``/logs. Components needing it raise when it is
    unset rather than falling back to an unencrypted path.
    """

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        """Upper-case string log levels so ``LOG_LEVEL=debug`` is accepted."""
        if isinstance(value, str):
            return value.upper()
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    Assumes the environment does not change mid-process; tests that mutate
    the environment must call ``get_settings.cache_clear()`` first.
    """
    return Settings()
