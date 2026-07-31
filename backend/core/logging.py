"""Structured JSON logging built on structlog (DECISIONS.md D-003).

Every log line is a single JSON object on stdout carrying an ISO-8601 UTC
timestamp, the log level, the logger name, the event, any bound key/values,
and everything merged from ``structlog.contextvars`` (notably ``request_id``
bound by the correlation-ID middleware).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

    from backend.core.config import Settings

_configured = False

_LOGGER_NAME_KEY = "logger_name"
"""Internal context key carrying the logger name until it is promoted to ``logger``."""


def _promote_logger_name(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Rename the internal ``logger_name`` key to ``logger`` in the event dict.

    :func:`get_logger` cannot bind ``logger`` directly because that keyword
    collides with ``structlog.wrap_logger``'s first parameter, so the name
    travels as ``logger_name`` and is promoted here.
    """
    name = event_dict.pop(_LOGGER_NAME_KEY, None)
    if name is not None:
        event_dict["logger"] = name
    return event_dict


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure structlog for JSON output at the level given by settings.

    Idempotent: after the first successful call, subsequent calls are
    no-ops unless ``force=True`` (intended for tests and deliberate
    reconfiguration). When ``settings`` is omitted the cached application
    settings are used. Output goes to the *current* ``sys.stdout`` at emit
    time, one JSON object per line.
    """
    global _configured  # module-level idempotency latch
    if _configured and not force:
        return
    if settings is None:
        from backend.core.config import get_settings

        settings = get_settings()
    level = logging.getLevelNamesMapping()[settings.log_level]
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _promote_logger_name,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # No explicit file: each logger resolves the current sys.stdout when it
        # is created, which keeps output correct under pytest's capture.
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a structlog logger with ``logger=name`` bound lazily.

    Safe to call at module import time: the returned proxy materializes on
    first use with whatever configuration is active then, so importing a
    module before :func:`configure_logging` runs does not freeze defaults.
    The name is carried as a lazily-bound initial value and emitted under
    the ``logger`` key by :func:`_promote_logger_name`.
    """
    logger: FilteringBoundLogger = structlog.get_logger(**{_LOGGER_NAME_KEY: name})
    return logger
