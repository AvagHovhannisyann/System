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


class _CountingClock:
    """Injected clock that advances only when slept on, and refuses to spin.

    ``TokenBucket`` is given a monotonic source and a sleep function. This
    clock advances virtual time by exactly the requested delay, and raises once
    the number of sleeps passes a bound — so a wait loop that fails to make
    progress fails the test in milliseconds instead of hanging the suite (the
    original defect consumed several GB before being killed).
    """

    def __init__(self, max_sleeps: int = 64) -> None:
        """Start at t=0 with no sleeps recorded."""
        self.now = 0.0
        self.sleeps = 0
        self.max_sleeps = max_sleeps

    def monotonic(self) -> float:
        """Return the current virtual time in seconds."""
        return self.now

    async def sleep(self, seconds: float) -> None:
        """Advance virtual time, bounding the total number of waits."""
        self.sleeps += 1
        if self.sleeps > self.max_sleeps:
            msg = (
                f"acquire() slept {self.sleeps} times without completing: the wait "
                f"loop is not making progress (virtual time {self.now:.9f}s)"
            )
            raise AssertionError(msg)
        self.now += seconds


@pytest.mark.parametrize("rate", [10.0, 3.0, 7.0, 1.0 / 3.0, 0.7, 1.1])
async def test_acquire_terminates_for_rates_without_exact_reciprocals(rate: float) -> None:
    """Refill rounding must not stall the wait loop (regression).

    Rates whose reciprocal is not exactly representable in binary floating
    point (1/3, 0.7, 1.1, and 10.0 among them) can leave the balance one ulp
    below the requested token after sleeping precisely long enough. The loop
    then computed a ~0 delay and spun forever under an injected clock. Each
    acquisition here must finish in a small number of sleeps.
    """
    clock = _CountingClock()
    bucket = TokenBucket(
        requests_per_second=rate, burst=1, monotonic=clock.monotonic, sleep=clock.sleep
    )
    for _ in range(20):
        await bucket.acquire()
    assert clock.sleeps <= 20 + 1


async def test_acquire_waits_about_the_expected_time_at_a_steady_rate() -> None:
    """The tolerance must not turn into free capacity: pacing still holds."""
    clock = _CountingClock(max_sleeps=256)
    bucket = TokenBucket(
        requests_per_second=10.0, burst=1, monotonic=clock.monotonic, sleep=clock.sleep
    )
    for _ in range(11):
        await bucket.acquire()
    # 1 immediate (full bucket) + 10 paced at 0.1s each, within a nanosecond.
    assert clock.now == pytest.approx(1.0, abs=1e-6)
