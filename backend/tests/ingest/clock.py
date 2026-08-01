"""A controlled clock for the ingestion unit tests.

Retry backoff and rate limiting are both defined in terms of *time*, so
testing them against the wall clock would mean either sleeping for real (slow,
and flaky under load) or asserting nothing precise. Every timing-dependent
component in :mod:`backend.ingest` therefore takes its clock and its sleep as
injected callables; this module supplies a pair where sleeping advances the
clock instantly and every requested delay is recorded.

That makes the assertions exact — the backoff *schedule* and the rate
limiter's pacing are checked as sequences of numbers, not as elapsed
wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControlledClock:
    """A monotonic clock that only advances when something sleeps on it.

    Attributes:
        seconds: current reading of the monotonic clock, in seconds.
        sleeps: every delay passed to :meth:`sleep`, in order, in seconds.
    """

    seconds: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        """Return the current clock reading in seconds."""
        return self.seconds

    async def sleep(self, delay: float) -> None:
        """Record ``delay`` seconds and advance the clock by exactly that much."""
        self.sleeps.append(delay)
        self.seconds += delay

    def advance(self, delay: float) -> None:
        """Advance the clock by ``delay`` seconds without recording a sleep."""
        self.seconds += delay
