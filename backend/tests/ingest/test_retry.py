"""Retry semantics: transient failures retry, everything else does not (P3.1).

The load-bearing assertion in this file is the negative one: a
``PermanentSourceError`` — what a 4xx meaning "your request is wrong" becomes —
must be attempted exactly once. A retry loop on a malformed request burns the
source's rate-limit budget and hides a real defect behind several sleeps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from backend.ingest.errors import (
    PermanentSourceError,
    TransientSourceError,
    source_error_for_status,
)
from backend.ingest.retry import RetryPolicy, full_jitter, retry_async
from backend.tests.ingest.clock import ControlledClock

if TYPE_CHECKING:
    from collections.abc import Callable


def _no_jitter(ceiling_s: float) -> float:
    """Identity jitter: wait exactly the ceiling, so the schedule is assertable."""
    return ceiling_s


_NO_JITTER = _no_jitter


def _policy(
    max_attempts: int = 4,
    initial_backoff_s: float = 1.0,
    max_backoff_s: float = 8.0,
    multiplier: float = 2.0,
) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_backoff_s=initial_backoff_s,
        max_backoff_s=max_backoff_s,
        multiplier=multiplier,
    )


# --- policy validation ------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: RetryPolicy(max_attempts=0), "max_attempts"),
        (lambda: RetryPolicy(initial_backoff_s=0.0), "initial_backoff_s"),
        (lambda: RetryPolicy(initial_backoff_s=1.0, max_backoff_s=0.5), "max_backoff_s"),
        (lambda: RetryPolicy(multiplier=0.5), "multiplier"),
    ],
)
def test_invalid_policy_is_rejected_at_construction(
    build: Callable[[], RetryPolicy], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build()


def test_backoff_ceiling_grows_exponentially_and_is_capped() -> None:
    policy = _policy()
    ceilings = [policy.backoff_ceiling_s(attempt) for attempt in range(1, 6)]
    assert ceilings == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_backoff_ceiling_rejects_a_zeroth_attempt() -> None:
    with pytest.raises(ValueError, match="attempt must be >= 1"):
        _policy().backoff_ceiling_s(0)


def test_full_jitter_stays_within_its_ceiling() -> None:
    """Full jitter draws in [0, ceiling]; the bound is what backoff relies on."""
    for _ in range(200):
        assert 0.0 <= full_jitter(3.0) <= 3.0


# --- retry behavior ---------------------------------------------------------


async def test_successful_operation_is_attempted_once_and_never_sleeps() -> None:
    clock = ControlledClock()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        return "payload"

    result = await retry_async(
        operation, policy=_policy(), description="probe", sleep=clock.sleep, jitter=_NO_JITTER
    )
    assert result == "payload"
    assert attempts == 1
    assert clock.sleeps == []


async def test_transient_failure_is_retried_and_then_succeeds() -> None:
    clock = ControlledClock()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientSourceError("connection reset")
        return "payload"

    result = await retry_async(
        operation, policy=_policy(), description="probe", sleep=clock.sleep, jitter=_NO_JITTER
    )
    assert result == "payload"
    assert attempts == 3
    assert clock.sleeps == [1.0, 2.0]


async def test_exhausted_retries_reraise_the_last_transient_error() -> None:
    """The error propagates unchanged — it is never converted into a value (I3)."""
    clock = ControlledClock()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        msg = f"gateway timeout #{attempts}"
        raise TransientSourceError(msg)

    with pytest.raises(TransientSourceError, match="gateway timeout #4"):
        await retry_async(
            operation, policy=_policy(), description="probe", sleep=clock.sleep, jitter=_NO_JITTER
        )
    assert attempts == 4
    assert clock.sleeps == [1.0, 2.0, 4.0]


async def test_permanent_source_error_is_never_retried() -> None:
    """The whole point of the transient/permanent split: 4xx does not loop."""
    clock = ControlledClock()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise PermanentSourceError("404 not found")

    with pytest.raises(PermanentSourceError, match="404"):
        await retry_async(
            operation, policy=_policy(), description="probe", sleep=clock.sleep, jitter=_NO_JITTER
        )
    assert attempts == 1
    assert clock.sleeps == []


async def test_unexpected_exception_is_not_retried_either() -> None:
    """Only TransientSourceError is retryable; a bug must surface immediately."""
    clock = ControlledClock()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        await retry_async(
            operation, policy=_policy(), description="probe", sleep=clock.sleep, jitter=_NO_JITTER
        )
    assert attempts == 1
    assert clock.sleeps == []


async def test_single_attempt_policy_does_not_retry() -> None:
    clock = ControlledClock()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise TransientSourceError("503")

    with pytest.raises(TransientSourceError):
        await retry_async(
            operation,
            policy=RetryPolicy(max_attempts=1),
            description="probe",
            sleep=clock.sleep,
            jitter=_NO_JITTER,
        )
    assert attempts == 1
    assert clock.sleeps == []


async def test_on_retry_callback_reports_each_backoff() -> None:
    clock = ControlledClock()
    observed: list[tuple[int, float]] = []
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientSourceError("429")
        return "payload"

    await retry_async(
        operation,
        policy=_policy(),
        description="probe",
        sleep=clock.sleep,
        jitter=_NO_JITTER,
        on_retry=lambda attempt, waited: observed.append((attempt, waited)),
    )
    assert observed == [(1, 1.0), (2, 2.0)]


# --- HTTP status classification --------------------------------------------


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504, 599])
def test_transient_statuses_classify_as_retryable(status: int) -> None:
    error = source_error_for_status(status, source="probe", detail="GET /x")
    assert isinstance(error, TransientSourceError)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422, 451])
def test_bad_request_statuses_classify_as_permanent(status: int) -> None:
    error = source_error_for_status(status, source="probe", detail="GET /x")
    assert isinstance(error, PermanentSourceError)
    assert not isinstance(error, TransientSourceError)


@pytest.mark.parametrize("status", [200, 204, 302, 399, 600])
def test_non_error_status_is_a_programming_error(status: int) -> None:
    """Asking for the failure of a success must not manufacture one."""
    with pytest.raises(ValueError, match="non-error HTTP status"):
        source_error_for_status(status, source="probe", detail="GET /x")
