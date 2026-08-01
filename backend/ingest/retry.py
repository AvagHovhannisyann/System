"""Retry with exponential backoff and jitter — transient failures only (P3.1).

The rule this module exists to enforce: **only** a
:class:`~backend.ingest.errors.TransientSourceError` is retried. Every other
exception — most importantly
:class:`~backend.ingest.errors.PermanentSourceError`, which is what a 4xx that
means "your request is wrong" becomes — propagates on the first attempt.
Retrying a malformed, unauthorized or not-found request cannot succeed; it
burns the source's rate-limit budget and delays the operator seeing a genuine
defect behind several backoff sleeps.

Backoff is exponential with **full jitter**: attempt *n* waits a uniform draw
from ``[0, min(max_backoff_s, initial_backoff_s * multiplier**(n-1))]``.
Full jitter rather than a fixed schedule because deterministic backoff
synchronizes retries across concurrently failing connectors and re-creates the
burst that caused the throttling. The jitter function is injectable so tests
assert the *schedule* (the ceiling sequence) deterministically instead of
sampling randomness.

Nothing here swallows an error: when attempts are exhausted, the last
transient error is re-raised unchanged, with its retry history attached to the
structured log record. A retry helper that returned a default on exhaustion
would be exactly the fabrication invariant I3 forbids.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.core.logging import get_logger
from backend.ingest.errors import TransientSourceError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["RetryPolicy", "full_jitter", "retry_async"]

_logger = get_logger(__name__)


def full_jitter(ceiling_s: float) -> float:
    """Return a uniform random wait in ``[0, ceiling_s]`` seconds.

    The default jitter strategy ("full jitter"). Uses the ``random`` module
    deliberately: this is scheduling noise to decorrelate retries, not a
    security primitive, and a cryptographic source would buy nothing here.
    """
    return random.uniform(0.0, ceiling_s)  # noqa: S311 — backoff jitter, not a security primitive


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff retry budget for one connector.

    Attributes:
        max_attempts: total attempts including the first (>= 1). ``1`` means
            no retry at all.
        initial_backoff_s: backoff ceiling before the second attempt, seconds
            (> 0).
        max_backoff_s: cap on the backoff ceiling, seconds (>=
            ``initial_backoff_s``).
        multiplier: growth factor applied per attempt (>= 1).
    """

    max_attempts: int = 5
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 30.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Validate the budget; an unusable policy must fail at construction."""
        if self.max_attempts < 1:
            msg = f"max_attempts must be >= 1; got {self.max_attempts}"
            raise ValueError(msg)
        if self.initial_backoff_s <= 0:
            msg = f"initial_backoff_s must be > 0 seconds; got {self.initial_backoff_s}"
            raise ValueError(msg)
        if self.max_backoff_s < self.initial_backoff_s:
            msg = (
                f"max_backoff_s ({self.max_backoff_s}s) must be >= initial_backoff_s "
                f"({self.initial_backoff_s}s)"
            )
            raise ValueError(msg)
        if self.multiplier < 1:
            msg = f"multiplier must be >= 1; got {self.multiplier}"
            raise ValueError(msg)

    def backoff_ceiling_s(self, attempt: int) -> float:
        """Return the backoff ceiling in seconds for the wait *after* ``attempt``.

        Args:
            attempt: 1-based number of the attempt that just failed.

        Returns:
            ``min(max_backoff_s, initial_backoff_s * multiplier ** (attempt - 1))``
            seconds — the upper end of the jitter draw, not the wait itself.

        Raises:
            ValueError: if ``attempt`` is < 1.
        """
        if attempt < 1:
            msg = f"attempt must be >= 1; got {attempt}"
            raise ValueError(msg)
        return min(self.max_backoff_s, self.initial_backoff_s * self.multiplier ** (attempt - 1))


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    description: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float], float] = full_jitter,
    on_retry: Callable[[int, float], None] | None = None,
) -> T:
    """Await ``operation``, retrying **only** transient source failures.

    Args:
        operation: zero-argument coroutine factory performing one attempt. It
            is called afresh per attempt, so it must be idempotent with
            respect to the source (a GET, a signed download — never a
            non-idempotent mutation).
        policy: attempt and backoff budget.
        description: short human-readable label for logs (e.g.
            ``"edgar daily-index 2024-01-05"``). Must contain no secrets
            (invariant I5).
        sleep: awaitable sleep in seconds; injected by tests.
        jitter: maps a backoff ceiling in seconds to the actual wait in
            seconds; injected by tests to make the schedule deterministic.
        on_retry: optional callback invoked as ``(attempt, waited_s)`` after
            each backoff, used by the connector to count retries for its
            data-quality metrics.

    Returns:
        Whatever ``operation`` returned on the first successful attempt.

    Raises:
        TransientSourceError: the last one seen, re-raised unchanged once
            ``policy.max_attempts`` attempts have failed. Never converted into
            a default value (invariant I3).
        Exception: any non-transient exception, propagated immediately from
            the attempt that raised it — in particular
            :class:`~backend.ingest.errors.PermanentSourceError` is **not**
            retried.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await operation()
        except TransientSourceError as exc:
            if attempt >= policy.max_attempts:
                _logger.error(
                    "ingest.retry.exhausted",
                    operation=description,
                    attempts=attempt,
                    error=str(exc),
                )
                raise
            waited = jitter(policy.backoff_ceiling_s(attempt))
            _logger.warning(
                "ingest.retry.transient",
                operation=description,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                backoff_s=waited,
                error=str(exc),
            )
            await sleep(waited)
            if on_retry is not None:
                on_retry(attempt, waited)
