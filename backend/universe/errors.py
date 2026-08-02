"""Failure taxonomy for point-in-time universe construction (directive §5 Phase 4).

Every failure here is loud, and the reason is narrower than "good hygiene". A
universe is a *filter*, and the characteristic failure of a filter is that it
stops filtering. If the market-capitalisation input is missing and the
market-cap screen quietly passes every name, the universe is still produced,
still plausible, still roughly the right size — and every backtest run on it
silently includes microcaps the strategy could never have traded. Nothing
downstream reports that: the Sharpe ratio goes *up*, because microcaps are where
the apparent alpha lives and where the real costs are.

So this module exists to make the two failure modes that matter impossible to
miss:

**A missing input source refuses the whole build**
(:class:`UniverseInputUnavailableError`). Not "excludes everything", not
"includes everything", not "skips the filter" — refuses, naming the source and
the blocker that owns it. Today the market-capitalisation and borrow-availability
inputs do not exist in this repository (BLOCKERS.md **B1**, **B2**), and this is
the error that says so.

**A missing input for one name excludes that name**, and is recorded against the
filter that could not be evaluated, so the exclusion appears in the waterfall
rather than in nobody's count. That is the one direction of error a universe may
take unilaterally: excluding a name we cannot show is eligible removes a trade,
while including one we cannot show is eligible invents a trade.

The remaining errors guard the arithmetic that surrounds those two decisions —
criteria that do not describe a filter (:class:`UniverseCriteriaError`), a
session that is not scoped by ``as_of()`` (:class:`UniverseSessionError`), and
internal inconsistencies between a snapshot and the counts derived from it
(:class:`UniverseConsistencyError`).
"""

from __future__ import annotations

__all__ = [
    "UniverseConsistencyError",
    "UniverseCriteriaError",
    "UniverseError",
    "UniverseInputUnavailableError",
    "UniverseSessionError",
]


class UniverseError(Exception):
    """Base class for every failure raised by :mod:`backend.universe`."""


class UniverseCriteriaError(UniverseError, ValueError):
    """Raised when a :class:`~backend.universe.criteria.UniverseCriteria` is not a filter.

    Examples: a non-positive floor (nothing can fail ``value >= 0``, so a zero
    floor records a screen in the criteria hash that never screened anything),
    an empty set of allowed exchanges (no name could ever qualify), a
    non-positive ADV lookback (the median of an empty window is undefined).

    Subclasses :class:`ValueError` so ordinary caller-side validation and
    ``pytest.raises(ValueError)`` keep working.
    """


class UniverseInputUnavailableError(UniverseError, RuntimeError):
    """Raised when a filter's input data source does not exist in this system.

    The refusal is **whole-build**, deliberately. A universe that quietly drops
    the market-cap screen because nothing supplies market caps is not a smaller
    universe or a noisier one — it is a *different* universe, one whose members
    were never screened on size, and it is indistinguishable from a correct one
    by inspection. Directive I3 forbids stubbing the input to plausible values;
    the only remaining honest behaviour is to refuse, name the source, and name
    the blocker whose resolution would make the build possible.

    Raised before any database read, so the caller cannot be left wondering
    whether a partial universe was computed. See
    :func:`backend.universe.builder.build_universe`.

    Attributes:
        filter_name: the screen that cannot be evaluated, one of
            :data:`~backend.universe.criteria.FILTER_ORDER`.
        source: prose name of the missing data source, e.g. ``"securities
            master (shares outstanding)"``. Names what would have to exist, not
            the code that is absent.
        blocker: the ``BLOCKERS.md`` identifier tracking it, e.g. ``"B1"``.
    """

    def __init__(self, *, filter_name: str, source: str, blocker: str) -> None:
        """Build the error from the filter, the missing source, and its blocker."""
        self.filter_name = filter_name
        self.source = source
        self.blocker = blocker
        super().__init__(
            f"universe filter {filter_name!r} cannot be evaluated: its input comes from "
            f"{source}, which does not exist in this system (BLOCKERS.md {blocker}). "
            f"The build is refused rather than run without this screen. Skipping an "
            f"unavailable filter would produce a universe that looks correct, is a "
            f"different universe than the criteria describe, and silently admits names "
            f"the strategy could not have traded (I3, directive §9.1-9.2). Supply a "
            f"real source once {blocker} is resolved, or narrow the criteria so this "
            f"filter is not requested."
        )


class UniverseSessionError(UniverseError, RuntimeError):
    """Raised when the session handed to the builder is not scoped by ``as_of()``.

    Invariant I1 requires every read of a bitemporal fact table to be pinned to a
    knowledge instant. :mod:`backend.db.asof` enforces that at the ORM and SQL
    boundaries, so an unscoped session would fail there anyway — but it would
    fail *after* the caller believed a universe was being built, and the message
    would be about statement rewriting rather than about the universe. This
    checks the precondition at the entry point instead, and it also supplies the
    ``as_of`` instant the snapshot records for reproducibility (I2).
    """


class UniverseConsistencyError(UniverseError, ValueError):
    """Raised when a snapshot or a history is internally inconsistent.

    Examples: a member that has no outcome, an outcome marked included whose
    ``failed_filters`` is non-empty, two identity versions in force for one
    security at one instant, a waterfall whose steps do not sum back to the
    member count, or a history whose snapshots were built under different
    criteria (their sizes and turnover would not be comparable).

    These are assertions about this package's own arithmetic rather than about
    the caller's data, which is why they raise instead of returning a flag: a
    universe that cannot reconcile its own counts has no honest number to report.
    """
