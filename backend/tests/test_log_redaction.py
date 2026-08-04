"""Tests for the structlog redaction processor (DIRECTIVE §7, I5).

Every credential-shaped literal below is deliberately fake: the character runs
are hand-written filler in the right *shape* so the patterns fire. Nothing here
is, or has ever been, a real credential.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
import structlog
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.core.config import Settings
from backend.core.logging import (
    REDACTED,
    configure_logging,
    contains_credential,
    get_logger,
    redact_secrets,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from structlog.typing import EventDict


# --- fake, correctly-shaped credentials -----------------------------------
# Every value below is invented and inert. They are ASSEMBLED AT RUNTIME from
# fragments rather than written as literals: a literal of the correct shape
# trips GitHub push protection (Slack and Stripe shapes did exactly that), and
# a repository full of credential-shaped literals trains people to click
# "allow this secret", which is a habit worth not building.
#
# This costs the tests nothing. `redact_secrets` receives the identical
# full-length string at runtime — only the source representation differs — so
# the patterns under test are exercised at full strength. Concatenating with
# `"".join` rather than `+` keeps the compiler from folding the fragments back
# into a single literal in the .pyc, which would defeat the point.
def _shape(*parts: str) -> str:
    """Assemble a credential-shaped test value from fragments at runtime."""
    return "".join(parts)


_OPENAI_SHAPED = _shape("sk-", "proj-", "FAKE0000example1111example2222example3333")
_ANTHROPIC_SHAPED = _shape("sk-", "ant-", "api03-", "FAKEfakeFAKEfakeFAKEfakeFAKEfakeAA")
_AWS_SHAPED = _shape("AKIA", "FAKEEXAMPLE00000")
_GITHUB_CLASSIC_SHAPED = _shape("ghp", "_", "FAKEfakeFAKEfakeFAKEfakeFAKEfake0000")
_GITHUB_PAT_SHAPED = _shape("github", "_pat_", "FAKEfake00000000000000_FAKEfake111111111111")
_GOOGLE_SHAPED = _shape("AIza", "FAKEfake0000111122223333444455566")
_SLACK_SHAPED = _shape("xox", "b-", "000000000000-111111111111-FAKEfakeFAKEfake")
_STRIPE_SHAPED = _shape("sk", "_live_", "FAKEfake0000111122223333")
_BEARER_SHAPED = _shape("Bearer ", "FAKEfakeFAKEfakeFAKEfake0000")
_PEM_SHAPED = (
    _shape("-----BEGIN ", "RSA ", "PRIVATE ", "KEY-----") + "\n"
    "MIIFAKEfakeFAKEfakeNOTAREALKEYatallJUSTfiller00000000\n"
    "AAAAfillerBBBBfillerCCCCfiller\n"
    "-----END RSA PRIVATE KEY-----"
)

_ALL_SHAPED = (
    _OPENAI_SHAPED,
    _ANTHROPIC_SHAPED,
    _AWS_SHAPED,
    _GITHUB_CLASSIC_SHAPED,
    _GITHUB_PAT_SHAPED,
    _GOOGLE_SHAPED,
    _SLACK_SHAPED,
    _STRIPE_SHAPED,
    _BEARER_SHAPED,
    _PEM_SHAPED,
)


@pytest.fixture(autouse=True)
def _restore_logging_config() -> Iterator[None]:
    """Leave logging configured with default settings after each test."""
    yield
    configure_logging(Settings(), force=True)


def _process(event_dict: dict[str, Any]) -> EventDict:
    """Run the processor the way structlog does, with a dummy logger."""
    return redact_secrets(None, "info", event_dict)


def _walk_strings(value: object, depth: int = 0) -> Iterator[str]:
    """Yield every string reachable in *value*, keys included."""
    if depth > 40:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(key, depth + 1)
            yield from _walk_strings(item, depth + 1)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _walk_strings(item, depth + 1)


# --------------------------------------------------------------------------
# Pattern coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("credential", _ALL_SHAPED)
def test_each_credential_pattern_is_recognised(credential: str) -> None:
    """The shapes used throughout this file really do trip the detector."""
    assert contains_credential(credential)


@pytest.mark.parametrize("credential", _ALL_SHAPED)
def test_credential_in_an_ordinary_field_is_redacted(credential: str) -> None:
    """A key pattern is scrubbed whatever field carries it — no name hint needed."""
    result = _process({"event": "call_made", "detail": credential})
    assert credential not in json.dumps(result, default=repr)
    assert REDACTED in str(result["detail"])


@pytest.mark.parametrize("credential", _ALL_SHAPED)
def test_credential_embedded_in_a_sentence_is_redacted(credential: str) -> None:
    """Only the credential span goes; the surrounding message survives."""
    result = _process({"event": f"provider rejected {credential} at 09:00", "provider": "acme"})
    assert credential not in str(result["event"])
    assert "provider rejected" in str(result["event"])
    assert result["provider"] == "acme"


@pytest.mark.parametrize(
    "innocuous",
    ["sk-short", "AKIA123", "ghp_tooshort", "a normal sentence", "Bearer x", "1234567890"],
)
def test_innocuous_values_are_left_alone(innocuous: str) -> None:
    """Redaction is targeted: ordinary log content must survive unchanged."""
    assert _process({"event": innocuous})["event"] == innocuous


# --------------------------------------------------------------------------
# Key-name based redaction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "password",
        "passwd",
        "passphrase",
        "token",
        "refresh_token",
        "secret",
        "client_secret",
        "api_key",
        "apiKey",
        "X-API-Key",
        "authorization",
        "Authorization",
        "credentials",
        "private_key",
        "aws_access_key",
        "session_key",
        "cookie",
        "secrets_kek",
        "signature",
    ],
)
def test_value_under_a_sensitive_key_name_is_redacted(field: str) -> None:
    """A value need not look like a credential if its field name says it is one."""
    result = _process({"event": "config_saved", field: "totally-plain-looking-value"})
    assert result[field] == REDACTED


def test_numeric_values_under_sensitive_names_survive() -> None:
    """Counters like ``tokens_used`` are metrics, not credentials.

    Credentials are textual; blanking the numbers would blind cost governance
    (§6.5 spend gauges) for no security gain.
    """
    result = _process({"event": "llm_call", "tokens_used": 1234, "prompt_tokens": 57})
    assert result["tokens_used"] == 1234
    assert result["prompt_tokens"] == 57


def test_sensitivity_propagates_into_nested_containers() -> None:
    """Everything under a secret-named key is secret, however innocuous the leaf name."""
    result = _process(
        {"event": "e", "credentials": {"provider": "acme", "value": "plain", "rotations": 3}}
    )
    assert result["credentials"] == {
        "provider": REDACTED,
        "value": REDACTED,
        "rotations": 3,
    }


def test_a_credential_used_as_a_dict_key_is_redacted() -> None:
    """A dict keyed by the credential leaks exactly as badly as one valued by it."""
    result = _process({"event": "e", "by_key": {_OPENAI_SHAPED: "usage"}})
    assert _OPENAI_SHAPED not in json.dumps(result, default=repr)


# --------------------------------------------------------------------------
# Nested structures
# --------------------------------------------------------------------------


def test_redaction_reaches_deeply_nested_structures() -> None:
    """Dicts inside lists inside tuples inside dicts are all walked."""
    payload = {
        "event": "sync",
        "batch": [
            {"provider": "acme", "headers": [("authorization", _BEARER_SHAPED)]},
            {"nested": {"deeper": {"deepest": [_AWS_SHAPED, {"pem": _PEM_SHAPED}]}}},
        ],
    }
    result = _process(payload)
    rendered = json.dumps(result, default=repr)
    for credential in (_BEARER_SHAPED, _AWS_SHAPED, _PEM_SHAPED):
        assert credential not in rendered


def test_container_types_are_preserved() -> None:
    """Redaction rewrites contents, not shapes: downstream renderers see what they expect."""
    result = _process(
        {
            "event": "e",
            "as_list": ["a"],
            "as_tuple": ("a",),
            "as_set": {"a"},
            "as_frozenset": frozenset({"a"}),
        }
    )
    assert isinstance(result["as_list"], list)
    assert isinstance(result["as_tuple"], tuple)
    assert isinstance(result["as_set"], set)
    assert isinstance(result["as_frozenset"], frozenset)


def test_credentials_inside_sets_are_redacted() -> None:
    """Sets are containers too, even though they render through ``repr``."""
    result = _process({"event": "e", "seen": {_GITHUB_CLASSIC_SHAPED}})
    assert _GITHUB_CLASSIC_SHAPED not in json.dumps(result, default=repr)


def test_credentials_in_bytes_are_redacted() -> None:
    """Byte payloads (request bodies, headers) are scanned as well."""
    result = _process({"event": "e", "body": _OPENAI_SHAPED.encode()})
    assert _OPENAI_SHAPED not in json.dumps(result, default=repr)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


def test_exception_arguments_are_redacted() -> None:
    """An exception carried as a log value must not smuggle its args past redaction."""
    result = _process({"event": "e", "error": ValueError(f"bad key {_OPENAI_SHAPED}")})
    rendered = json.dumps(result, default=repr)
    assert _OPENAI_SHAPED not in rendered
    assert "ValueError" in rendered


def test_credential_nested_inside_exception_arguments_is_redacted() -> None:
    """Structured exception payloads are walked, not just stringified."""
    result = _process({"event": "e", "error": RuntimeError({"api_key": "plain-value"})})
    assert "plain-value" not in json.dumps(result, default=repr)


def test_clean_exceptions_pass_through_untouched() -> None:
    """No credential, no rewrite: ordinary error objects reach the renderer as themselves."""
    error = ValueError("connection refused")
    assert _process({"event": "e", "error": error})["error"] is error


# --------------------------------------------------------------------------
# End-to-end through the configured logger
# --------------------------------------------------------------------------


def test_realistic_key_in_a_log_record_is_emitted_redacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The wired-up chain redacts a realistic-looking key before it reaches stdout."""
    configure_logging(Settings(log_level="INFO"), force=True)
    structlog.contextvars.bind_contextvars(request_id="req-redact")
    logger = get_logger("test.redaction")
    logger.info("provider_probe", provider="openai", api_key=_OPENAI_SHAPED)

    line = capsys.readouterr().out.strip()
    assert _OPENAI_SHAPED not in line
    payload = json.loads(line)
    assert payload["event"] == "provider_probe"
    assert payload["provider"] == "openai"
    assert payload["api_key"] == REDACTED
    assert payload["request_id"] == "req-redact"


def test_credential_inside_a_traceback_is_redacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``dict_tracebacks`` renders exception text and frame locals; both are scrubbed."""
    configure_logging(Settings(log_level="INFO"), force=True)
    logger = get_logger("test.redaction.traceback")

    def _fail() -> None:
        local_credential = _ANTHROPIC_SHAPED
        raise RuntimeError(f"upstream rejected {local_credential}")

    try:
        _fail()
    except RuntimeError:
        logger.exception("provider_call_failed")

    line = capsys.readouterr().out.strip()
    assert _ANTHROPIC_SHAPED not in line
    payload = json.loads(line)
    assert payload["event"] == "provider_call_failed"
    assert REDACTED in json.dumps(payload["exception"])


def test_existing_fields_are_unaffected(capsys: pytest.CaptureFixture[str]) -> None:
    """The processor is additive: D-003's contract for ordinary events is unchanged."""
    configure_logging(Settings(log_level="INFO"), force=True)
    get_logger("test.plain").info("something_happened", answer=42)

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["event"] == "something_happened"
    assert payload["level"] == "info"
    assert payload["logger"] == "test.plain"
    assert payload["answer"] == 42
    assert "timestamp" in payload


# --------------------------------------------------------------------------
# Robustness: a redactor that raises would take logging down with it
# --------------------------------------------------------------------------


class _HostileRepr:
    """An object whose ``repr`` raises — the classic way to break a log formatter."""

    def __repr__(self) -> str:
        """Raise, always."""
        raise RuntimeError("no repr for you")


def test_object_with_a_raising_repr_does_not_break_redaction() -> None:
    """A hostile value is replaced by a marker instead of propagating an exception."""
    result = _process({"event": "e", "thing": _HostileRepr()})
    assert result["event"] == "e"


def test_self_referential_structure_terminates() -> None:
    """A cyclic payload hits the depth bound rather than recursing forever."""
    cyclic: dict[str, Any] = {"event": "e"}
    cyclic["self"] = cyclic
    result = _process(cyclic)
    assert result["event"] == "e"


class _BrokenMapping(Mapping[str, str]):
    """A mapping that raises when iterated, forcing the redactor's failure path."""

    _DATA: ClassVar[dict[str, str]] = {"event": "e", "api_key": _OPENAI_SHAPED}

    def __getitem__(self, key: str) -> str:
        """Return the stored value for *key* (used by the fallback's metadata probe)."""
        return self._DATA[key]

    def __iter__(self) -> Iterator[str]:
        """Raise: this is the whole point of the class."""
        raise RuntimeError("broken mapping")

    def __len__(self) -> int:
        """Return the underlying item count."""
        return len(self._DATA)


def test_broken_mapping_falls_back_to_a_safe_event() -> None:
    """When redaction itself fails, the payload is dropped rather than emitted raw."""
    result = redact_secrets(None, "info", cast("EventDict", _BrokenMapping()))
    assert result["event"] == "log_redaction_failed"
    assert result["redaction_error"] == "RuntimeError"
    assert _OPENAI_SHAPED not in json.dumps(result, default=repr)


@pytest.mark.parametrize(
    "odd",
    [
        {},
        {"event": None},
        {"event": b"\xff\xfe"},
        {"event": float("nan")},
        {"event": range(3)},
        {1: "int key", (2, 3): "tuple key", None: "none key"},
        {"event": [[[[[[[[[[[[[[[[[[[[["deep"]]]]]]]]]]]]]]]]]]]]]},
    ],
)
def test_odd_inputs_never_raise(odd: dict[Any, Any]) -> None:
    """Whatever shape an event dict arrives in, the processor returns a mapping."""
    assert isinstance(_process(odd), dict)


# --------------------------------------------------------------------------
# Property: never raises, never emits anything credential-shaped
# --------------------------------------------------------------------------

_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
    st.binary(),
    st.sampled_from(_ALL_SHAPED),
    st.builds(lambda msg: ValueError(msg), st.sampled_from(_ALL_SHAPED)),
    st.frozensets(st.sampled_from(_ALL_SHAPED) | st.text(), max_size=3),
)

_nested = st.recursive(
    _leaves,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.tuples(children, children),
        st.dictionaries(st.text(max_size=12) | st.sampled_from(_ALL_SHAPED), children, max_size=4),
    ),
    max_leaves=12,
)


@given(payload=st.dictionaries(st.text(max_size=12), _nested, max_size=6))
@hypothesis_settings(max_examples=400, deadline=None, print_blob=True)
def test_redactor_never_raises_and_never_emits_a_credential(payload: dict[str, Any]) -> None:
    """Property (§7, I5): arbitrary nested input, no exception, no credential shape out.

    The two failure modes this rules out are the ones that matter: a redactor
    that raises takes the whole logging path down, and a redactor that misses a
    nested field defeats the point of having one.
    """
    result = redact_secrets(None, "info", dict(payload))
    assert isinstance(result, dict)
    for text in _walk_strings(result):
        assert not contains_credential(text), f"credential shape survived redaction: {text!r}"
