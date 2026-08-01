"""Token-bucket rate limiting (P3.1).

Asserted against a controlled clock, so the pacing is checked as an exact
sequence of waits rather than by sleeping and measuring — a wall-clock test of
a rate limiter is both slow and flaky, and would in practice be weakened to
"roughly" until it asserted nothing.
"""

from __future__ import annotations

import pytest

from backend.ingest.ratelimit import RateLimit, TokenBucket
from backend.tests.ingest.clock import ControlledClock


def _bucket(clock: ControlledClock, *, requests_per_second: float, burst: int) -> TokenBucket:
    return RateLimit(requests_per_second=requests_per_second, burst=burst).bucket(
        monotonic=clock.monotonic, sleep=clock.sleep
    )


# --- configuration validation ----------------------------------------------


@pytest.mark.parametrize(
    ("requests_per_second", "burst", "match"),
    [
        (0.0, 1, "requests_per_second"),
        (-1.0, 1, "requests_per_second"),
        (1.0, 0, "burst"),
        (1.0, -3, "burst"),
    ],
)
def test_invalid_rate_limit_is_rejected(requests_per_second: float, burst: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RateLimit(requests_per_second=requests_per_second, burst=burst)


# --- pacing behavior --------------------------------------------------------


async def test_burst_is_served_without_waiting() -> None:
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=1.0, burst=3)
    waits = [await bucket.acquire() for _ in range(3)]
    assert waits == [0.0, 0.0, 0.0]
    assert clock.sleeps == []


async def test_requests_beyond_the_burst_are_paced_at_the_sustained_rate() -> None:
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=2.0, burst=2)
    for _ in range(2):
        assert await bucket.acquire() == 0.0
    assert await bucket.acquire() == pytest.approx(0.5)
    assert await bucket.acquire() == pytest.approx(0.5)
    assert clock.sleeps == pytest.approx([0.5, 0.5])


async def test_elapsed_time_refills_the_bucket() -> None:
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=4.0, burst=1)
    assert await bucket.acquire() == 0.0
    clock.advance(0.25)  # exactly one token's worth at 4/s
    assert await bucket.acquire() == 0.0
    assert clock.sleeps == []


async def test_refill_is_capped_at_the_burst_capacity() -> None:
    """An idle hour buys ``burst`` requests, not an hour's worth of them."""
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=1.0, burst=2)
    clock.advance(3600.0)
    assert await bucket.acquire() == 0.0
    assert await bucket.acquire() == 0.0
    assert await bucket.acquire() == pytest.approx(1.0)


async def test_fractional_rates_are_honored() -> None:
    """A '10 requests per minute' source is 10/60 per second, not a special case."""
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=10.0 / 60.0, burst=1)
    assert await bucket.acquire() == 0.0
    assert await bucket.acquire() == pytest.approx(6.0)


async def test_available_tokens_reports_accounted_state_only() -> None:
    """Reading availability must not silently refill: it reports, it does not grant."""
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=1.0, burst=2)
    await bucket.acquire()
    assert bucket.available_tokens == pytest.approx(1.0)
    clock.advance(5.0)
    assert bucket.available_tokens == pytest.approx(1.0)
    await bucket.acquire()
    assert bucket.available_tokens == pytest.approx(1.0)


@pytest.mark.parametrize("tokens", [0.0, -1.0])
async def test_non_positive_acquisition_is_rejected(tokens: float) -> None:
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=1.0, burst=1)
    with pytest.raises(ValueError, match="tokens must be > 0"):
        await bucket.acquire(tokens)


async def test_acquiring_more_than_capacity_is_rejected_rather_than_hanging() -> None:
    """An unsatisfiable request must raise; waiting forever is not a rate limit."""
    clock = ControlledClock()
    bucket = _bucket(clock, requests_per_second=1.0, burst=2)
    with pytest.raises(ValueError, match="could never be satisfied"):
        await bucket.acquire(3.0)
