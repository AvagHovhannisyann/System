"""Structured JSON logging built on structlog (DECISIONS.md D-003).

Every log line is a single JSON object on stdout carrying an ISO-8601 UTC
timestamp, the log level, the logger name, the event, any bound key/values,
and everything merged from ``structlog.contextvars`` (notably ``request_id``
bound by the correlation-ID middleware).

The last processor before rendering is :func:`redact_secrets`, which strips
credential-shaped data out of the whole event — nested containers, rendered
tracebacks (including frame locals) and exception arguments included — so that
no key pattern can reach stdout (DIRECTIVE §7, I5).
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, cast

import structlog

if TYPE_CHECKING:
    from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

    from backend.core.config import Settings

_configured = False

_LOGGER_NAME_KEY = "logger_name"
"""Internal context key carrying the logger name until it is promoted to ``logger``."""

REDACTED: Final = "***REDACTED***"
"""Replacement emitted in place of anything that looks like a credential."""

_DEPTH_LIMIT_MARKER: Final = "<redaction depth limit reached>"
_UNRENDERABLE_MARKER: Final = "<unrepresentable value>"

_MAX_DEPTH: Final = 16
"""Container nesting the redactor walks before truncating.

Bounds both runaway recursion and self-referential structures: a cyclic
``dict`` in a log call must not hang or blow the stack.
"""

_MAX_REDACTION_PASSES: Final = 4
"""Substitution passes run before falling back to replacing the whole string.

Repeating to a fixed point guarantees the invariant the property test asserts:
no emitted string still matches a credential pattern.
"""

_CREDENTIAL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # PEM private-key blocks of any flavour. The `\Z` alternative catches a
    # block that was truncated before its END line, which must still be scrubbed.
    re.compile(
        r"-----BEGIN[A-Z ]{0,32}PRIVATE KEY-----"
        r".*?(?:-----END[A-Z ]{0,32}PRIVATE KEY-----|\Z)",
        re.DOTALL,
    ),
    # OpenAI / Anthropic shapes: sk-..., sk-proj-..., sk-ant-api03-...
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    # Stripe secret / restricted keys.
    re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"),
    # AWS access key identifiers.
    re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA|AIDA|AROA)[A-Z0-9]{16}\b"),
    # GitHub classic tokens (ghp_/gho_/ghu_/ghs_/ghr_) and fine-grained PATs.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    # Google API keys.
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    # Slack tokens.
    re.compile(r"\bxox[abeprs]-[A-Za-z0-9-]{10,}"),
    # Bearer credentials appearing inside free text (header dumps, curl echoes).
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{16,}={0,2}"),
)
"""Regexes for credential shapes that must never be emitted, whatever field carries them."""

_SENSITIVE_KEY_PARTS: Final[frozenset[str]] = frozenset(
    {
        "accesskey",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "kek",
        "passphrase",
        "passwd",
        "password",
        "privatekey",
        "secret",
        "sessionkey",
        "signature",
        "token",
    }
)
"""Key-name fragments whose textual values are redacted regardless of shape.

Matched against the key with everything non-alphanumeric removed, so
``X-API-Key``, ``api_key`` and ``apiKey`` all resolve to ``apikey``.
"""

_NON_ALNUM = re.compile(r"[^a-z0-9]")


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


def contains_credential(text: str) -> bool:
    """Return True when *text* contains something shaped like a credential.

    The same patterns :func:`redact_secrets` substitutes on, exposed so callers
    (and tests) can assert the absence of credential shapes without reaching
    into module internals.
    """
    return any(pattern.search(text) for pattern in _CREDENTIAL_PATTERNS)


def _redact_text(text: str) -> str:
    """Replace every credential-shaped span in *text* with :data:`REDACTED`.

    Substitution repeats to a fixed point because replacing one span can splice
    together neighbours into a new match. If a match somehow survives
    :data:`_MAX_REDACTION_PASSES` the whole string is discarded — the invariant
    "nothing credential-shaped is emitted" outranks preserving the message.
    """
    redacted = text
    for _ in range(_MAX_REDACTION_PASSES):
        previous = redacted
        for pattern in _CREDENTIAL_PATTERNS:
            redacted = pattern.sub(REDACTED, redacted)
        if redacted == previous:
            return redacted
    return REDACTED if contains_credential(redacted) else redacted


def _safe_repr(value: object) -> str:
    """Return ``repr(value)``, or a marker when the object's ``__repr__`` raises."""
    try:
        return repr(value)
    except Exception:  # a hostile __repr__ must not break logging
        return _UNRENDERABLE_MARKER


def _is_sensitive_key(key: object) -> bool:
    """Return True when *key* names a field whose textual value must be redacted."""
    if not isinstance(key, str):
        return False
    normalized = _NON_ALNUM.sub("", key.lower())
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_mapping(mapping: Mapping[Any, Any], *, sensitive: bool, depth: int) -> dict[Any, Any]:
    """Return a copy of *mapping* with keys and values redacted.

    String keys are scrubbed too: a dict whose *key* is the credential leaks
    just as effectively as one whose value is. ``sensitive`` propagates
    downward so everything nested under, say, ``credentials`` is treated as
    secret even when the leaf key names look innocuous.
    """
    result: dict[Any, Any] = {}
    for key, value in list(mapping.items()):
        redacted_key = _redact_text(key) if isinstance(key, str) else key
        result[redacted_key] = _redact_obj(
            value,
            sensitive=sensitive or _is_sensitive_key(key),
            depth=depth + 1,
        )
    return result


def _rebuild_sequence(original: object, items: list[Any]) -> object:
    """Return *items* in the container flavour of *original* (list/tuple/set/frozenset)."""
    if isinstance(original, list):
        return items
    if isinstance(original, tuple):
        return tuple(items)
    if isinstance(original, frozenset):
        return frozenset(items)
    return set(items)


def _redact_exception(exc: BaseException, *, depth: int) -> object:
    """Return *exc* untouched when it carries no credential, else a redacted rendering.

    Exception instances cannot be rebuilt safely (arbitrary subclasses have
    arbitrary ``__init__`` signatures), so an exception whose ``args`` or
    ``repr`` carry a secret is replaced by a string rendering built from its
    redacted arguments rather than passed on to the renderer.
    """
    try:
        args: tuple[Any, ...] = tuple(exc.args)
    except Exception:  # exotic exceptions may not expose usable args
        args = ()
    redacted_args = [_redact_obj(arg, sensitive=False, depth=depth + 1) for arg in args]
    plain = f"{type(exc).__name__}({', '.join(_safe_repr(arg) for arg in args)})"
    rendered = _redact_text(f"{type(exc).__name__}({', '.join(map(_safe_repr, redacted_args))})")
    own_repr = _safe_repr(exc)
    if rendered == plain and _redact_text(own_repr) == own_repr:
        return exc
    return rendered


def _redact_obj(value: object, *, sensitive: bool, depth: int) -> object:
    """Return *value* with every credential-shaped part replaced.

    ``sensitive`` means "an enclosing key named this secret", which redacts
    textual leaves outright instead of only pattern matches. Numbers and
    ``None`` are returned untouched even when sensitive: credentials are
    textual, while counters legitimately live under names like ``tokens_used``
    and destroying them would blind cost governance for no security gain.
    """
    if depth > _MAX_DEPTH:
        return _DEPTH_LIMIT_MARKER
    if isinstance(value, str):
        return REDACTED if sensitive else _redact_text(value)
    if isinstance(value, bytes | bytearray):
        return REDACTED if sensitive else _redact_text(repr(bytes(value)))
    if isinstance(value, Mapping):
        return _redact_mapping(value, sensitive=sensitive, depth=depth)
    if isinstance(value, list | tuple | set | frozenset):
        items = [_redact_obj(item, sensitive=sensitive, depth=depth + 1) for item in value]
        return _rebuild_sequence(value, items)
    if isinstance(value, BaseException):
        return REDACTED if sensitive else _redact_exception(value, depth=depth)
    if value is None or isinstance(value, bool | int | float | complex):
        return value
    if sensitive:
        return REDACTED
    # Unknown types reach the renderer through ``repr``; scan that rendering and
    # keep the object itself whenever it is clean, so output shape is unchanged.
    text = _safe_repr(value)
    redacted = _redact_text(text)
    return value if redacted == text else redacted


def _redaction_failure_event(exc: BaseException, event_dict: Mapping[Any, Any]) -> EventDict:
    """Return a minimal, safe event describing a redactor failure.

    Fail closed: the original payload is dropped rather than emitted
    unredacted, and only correlation metadata that survives its own redaction
    pass is carried over.
    """
    fallback: dict[str, Any] = {
        "event": "log_redaction_failed",
        "level": "error",
        "redaction_error": type(exc).__name__,
    }
    for key in ("timestamp", "logger", "request_id"):
        with contextlib.suppress(Exception):
            value = event_dict[key]
            if isinstance(value, str):
                fallback[key] = _redact_text(value)
    return cast("EventDict", fallback)


def redact_secrets(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Strip credential-shaped data out of a log event (DIRECTIVE §7, I5).

    Runs immediately before rendering so it sees the fully-built event: bound
    values, merged context, and the structured traceback produced by
    ``dict_tracebacks`` (whose frame locals are a prime leak vector). Redaction
    reaches into nested dicts, lists, tuples, sets, exception arguments and the
    ``repr`` of unknown objects — not just top-level fields.

    Never raises: a redactor that propagated an exception would take the whole
    logging path down with it. On internal failure the event is replaced by a
    ``log_redaction_failed`` marker, dropping the payload rather than risking an
    unredacted emission.
    """
    try:
        return cast("EventDict", _redact_mapping(event_dict, sensitive=False, depth=0))
    except Exception as exc:  # logging must survive any input
        return _redaction_failure_event(exc, event_dict)


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure structlog for JSON output at the level given by settings.

    Idempotent: after the first successful call, subsequent calls are
    no-ops unless ``force=True`` (intended for tests and deliberate
    reconfiguration). When ``settings`` is omitted the cached application
    settings are used. Output goes to the *current* ``sys.stdout`` at emit
    time, one JSON object per line.

    :func:`redact_secrets` is installed as the final processor before
    rendering, so no configuration produces an unredacted log line (§7, I5).
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
            # Last before rendering: everything the renderer will serialise has
            # been built by now, tracebacks included (§7, I5).
            redact_secrets,
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
