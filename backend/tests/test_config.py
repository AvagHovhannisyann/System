"""Tests for backend.core.config: defaults, env overrides, caching, version."""

from __future__ import annotations

from importlib import metadata

import pytest
from pydantic import ValidationError

from backend.core.config import Settings, get_settings

_ENV_VARS = ("DATABASE_URL", "REDIS_URL", "ENVIRONMENT", "LOG_LEVEL", "APP_VERSION")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all settings-related variables from the ambient environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults() -> None:
    """With no environment set, every field takes its documented default."""
    settings = Settings()
    assert settings.database_url == "postgresql+asyncpg://quant:quant@localhost:5432/quant"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.environment == "dev"
    assert settings.log_level == "INFO"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override every default, including APP_VERSION."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/other")
    monkeypatch.setenv("REDIS_URL", "redis://cache:6379/3")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "debug")  # lower-case must normalize
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    settings = Settings()
    assert settings.database_url == "postgresql+asyncpg://u:p@db:5432/other"
    assert settings.redis_url == "redis://cache:6379/3"
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.app_version == "9.9.9"


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment outside dev/test/prod fails validation loudly."""
    monkeypatch.setenv("ENVIRONMENT", "staging")
    with pytest.raises(ValidationError, match="environment"):
        Settings()


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown log level fails validation rather than being silently accepted."""
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError, match="log_level"):
        Settings()


def test_app_version_matches_installed_package() -> None:
    """Without an APP_VERSION override, the version comes from package metadata."""
    assert Settings().app_version == metadata.version("quant-research-platform")


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings returns one cached instance until the cache is cleared."""
    monkeypatch.setenv("ENVIRONMENT", "prod")
    first = get_settings()
    assert first.environment == "prod"
    assert get_settings() is first
    monkeypatch.setenv("ENVIRONMENT", "dev")
    assert get_settings() is first  # still cached
    get_settings.cache_clear()
    assert get_settings().environment == "dev"
