"""Shared fixtures: isolate settings cache and structlog contextvars per test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import structlog

from backend.core.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Clear the cached Settings before and after each test.

    Keeps environment mutations made with ``monkeypatch`` from leaking
    between tests through the ``lru_cache`` on ``get_settings``.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_structlog_contextvars() -> Iterator[None]:
    """Clear structlog contextvars before and after each test."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
