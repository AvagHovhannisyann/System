"""Per-source rate limiting: an async token bucket (P3.1).

Every connector declares a :class:`RateLimit` and the framework paces *every*
outbound request through it, so a source's published limit is honored by
construction rather than by each connector remembering to sleep. Data vendors
answer an exceeded limit with 429s or a ban; both are expensive, and the ban is
not something a retry loop can fix.

Token-bucket semantics, in the units the vendors themselves publish:

- the bucket refills continuously at ``requests_per_second`` tokens/second;
- it holds at most ``burst`` tokens, so a connector idle for a while may fire
  ``burst`` requests back to back and is then paced at the sustained rate;
- :meth:`TokenBucket.acquire` waits until the requested tokens exist and
  returns how long it waited (seconds), which the connector reports as a
  data-quality metric.

The clock and the sleep function are injectable so the pacing behavior can be
tested deterministically instead of by wall-clock sleeping (which would make
the test both slow and flaky). Production uses :func:`time.monotonic` —
monotonic, not wall clock, so an NTP step cannot hand out free tokens or hang
the bucket.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["RateLimit", "TokenBucket"]


@dataclass(frozen=True, slots=True)
class RateLimit:
    """A source's declared request budget.

    Attributes:
        requests_per_second: sustained refill rate, requests per second
            (strictly positive; fractional values are meaningful — a "10
            requests per minute" source is ``10 / 60``).
        burst: bucket capacity in requests (>= 1). The largest number of
            requests that may be issued with no pacing after an idle period.
    """

    requests_per_second: float
    burst: int = 1

    def __post_init__(self) -> None:
        """Validate the budget; a nonsensical limit must not silently pass."""
        if self.requests_per_second <= 0:
            msg = f"requests_per_second must be > 0; got {self.requests_per_second}"
            raise ValueError(msg)
        if self.burst < 1:
            msg = f"burst must be >= 1 request; got {self.burst}"
            raise ValueError(msg)

    def bucket(
        self,
        *,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> TokenBucket:
        """Create a fresh :class:`TokenBucket` enforcing this limit.

        Args:
            monotonic: monotonic clock returning seconds; defaults to
                :func:`time.monotonic`. Injected only by tests.
            sleep: awaitable sleep taking seconds; defaults to
                :func:`asyncio.sleep`. Injected only by tests.

        Returns:
            A bucket that starts full (``burst`` tokens available).
        """
        return TokenBucket(
            requests_per_second=self.requests_per_second,
            burst=self.burst,
            monotonic=monotonic if monotonic is not None else time.monotonic,
            sleep=sleep if sleep is not None else asyncio.sleep,
        )


class TokenBucket:
    """Async token bucket pacing requests to a sustained rate with burst.

    One bucket instance per connector instance; it is safe to share across
    concurrent tasks in one event loop (an :class:`asyncio.Lock` serializes
    accounting) but is **not** shared across processes — a multi-worker
    deployment therefore enforces the limit per worker, which is why the
    declared limit should be the per-worker share of the vendor's budget.
    That is a deployment constraint, stated here rather than discovered from
    a vendor ban.
    """

    def __init__(
        self,
        *,
        requests_per_second: float,
        burst: int,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create a full bucket refilling at ``requests_per_second``.

        Args:
            requests_per_second: refill rate, tokens per second (> 0).
            burst: capacity in tokens (>= 1); the bucket starts full.
            monotonic: monotonic clock in seconds.
            sleep: awaitable sleep in seconds.
        """
        self._rate = requests_per_second
        self._capacity = float(burst)
        self._monotonic = monotonic
        self._sleep = sleep
        self._tokens = float(burst)
        self._updated_at = monotonic()
        self._lock = asyncio.Lock()

    @property
    def available_tokens(self) -> float:
        """Tokens present at the last accounting point (dimensionless).

        Does **not** refill on read — it reports the accounted state, so a
        caller cannot mistake elapsed-time refill for capacity actually
        reserved. Refill happens inside :meth:`acquire`, under the lock.
        """
        return self._tokens

    async def acquire(self, tokens: float = 1.0) -> float:
        """Wait until ``tokens`` are available, consume them, and report the wait.

        Args:
            tokens: tokens to consume (default one request's worth). Must be
                > 0 and <= the bucket capacity.

        Returns:
            Seconds spent waiting (0.0 when the tokens were already
            available). Reported by the connector as a data-quality metric so
            an operator can see when a source is throttling throughput.

        Raises:
            ValueError: if ``tokens`` is not positive, or exceeds capacity —
                which could never be satisfied and would otherwise wait
                forever.
        """
        if tokens <= 0:
            msg = f"tokens must be > 0; got {tokens}"
            raise ValueError(msg)
        if tokens > self._capacity:
            msg = (
                f"cannot acquire {tokens} token(s) from a bucket of capacity "
                f"{self._capacity}: the request could never be satisfied"
            )
            raise ValueError(msg)
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                delay = (tokens - self._tokens) / self._rate
                await self._sleep(delay)
                waited += delay

    def _refill(self) -> None:
        """Credit tokens for elapsed time, capped at capacity."""
        now = self._monotonic()
        elapsed = now - self._updated_at
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated_at = now
