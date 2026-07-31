"""Tests for backend.core.logging: JSON output, contextvars, level, idempotency."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
import structlog

from backend.core.config import Settings
from backend.core.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _restore_logging_config() -> Iterator[None]:
    """Leave logging configured with default settings after each test."""
    yield
    configure_logging(Settings(), force=True)


def test_emits_valid_json_with_bound_request_id(capsys: pytest.CaptureFixture[str]) -> None:
    """A log line is one JSON object with timestamp, level, logger, and request_id."""
    configure_logging(Settings(log_level="INFO"), force=True)
    structlog.contextvars.bind_contextvars(request_id="req-123")
    logger = get_logger("test.logger")
    logger.info("something_happened", answer=42)

    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["event"] == "something_happened"
    assert payload["level"] == "info"
    assert payload["logger"] == "test.logger"
    assert payload["request_id"] == "req-123"
    assert payload["answer"] == 42
    # ISO-8601 timestamp must parse.
    datetime.fromisoformat(payload["timestamp"])


def test_level_filtering_from_settings(capsys: pytest.CaptureFixture[str]) -> None:
    """Events below the configured level are dropped; at or above are emitted."""
    configure_logging(Settings(log_level="WARNING"), force=True)
    logger = get_logger("test.level")
    logger.info("too_quiet")
    logger.warning("loud_enough")

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    events = [entry["event"] for entry in lines]
    assert "too_quiet" not in events
    assert events == ["loud_enough"]


def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    """A second call without force is a no-op: the first configuration wins."""
    configure_logging(Settings(log_level="ERROR"), force=True)
    configure_logging(Settings(log_level="DEBUG"))  # must NOT take effect
    assert structlog.is_configured()

    logger = get_logger("test.idempotent")
    logger.debug("invisible")
    assert capsys.readouterr().out.strip() == ""
