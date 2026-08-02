"""Feature declarations: what a number means and when it becomes knowable (P5.1).

A feature in this system is not a function. It is a **declaration** — a name, a
definition in prose, a unit, an availability lag, and the set of fact tables it
reads — to which a computation is attached. The declaration is the part that
matters for correctness, and it exists separately from the code because two of
the questions it answers cannot be recovered by reading that code:

**What unit is this number in?** Directive §8: *"basis points versus percent
versus fraction is the most common bug class in this domain and it is silent"*.
A momentum feature and a book-to-price feature both arrive as float64 columns;
nothing downstream can tell that one is a dimensionless log return and the
other a ratio unless the declaration says so. ``units`` is required and may not
be empty.

**When could this number first have been computed?** This is invariant I1 at
the feature layer. A quarterly fundamental is not knowable on the quarter end;
it is knowable when the filing is accepted, some weeks later. A feature that
uses the quarter-end value on the quarter-end date is not slightly optimistic,
it is prescient, and the resulting backtest is worthless in a way no
distribution check reveals. ``availability_lag`` records the delay, and
:mod:`backend.features.compute` turns it into the only as-of instant the
computation is ever given.

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

**``compute_date`` is a calendar date; the instant it denotes is midnight UTC
opening that date.** A feature computed for rebalance date ``D`` is a number an
operator could have had in hand *before trading on D began*, so the reference
instant is the start of ``D``, not its end. A zero-lag feature therefore sees
facts stamped up to and including ``00:00:00Z`` on ``D`` and nothing published
during ``D`` itself. This is the conservative direction, and it is the only
choice that makes a daily rebalance implementable: a feature that needs a fact
published at 16:05 on ``D`` cannot inform a trade placed on ``D``.

**``availability_lag`` is a wall-clock delay, not a count of trading days.**
Knowledge times in the store are absolute instants (an EDGAR acceptance
timestamp, a vendor delivery time), so the arithmetic that has to line up with
them is wall-clock arithmetic. A lag expressed in trading days would need a
calendar, and this module has no business holding an opinion about market
holidays. Declare 45 days, not "30 sessions".

**The lag is a floor on staleness, never a ceiling.** Declaring 45 days does
not promise the value is 45 days old; it promises nothing fresher than that was
read. Over-declaring is safe (the feature is merely staler than necessary);
under-declaring is a lookahead bug. When the true lag is uncertain, round up.

**The boundary is inclusive**, matching :func:`backend.db.as_of`: a fact whose
``knowledge_time`` equals the cutoff instant is visible.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Final

from backend.features.errors import FeatureComputeError, FeatureSpecError

__all__ = [
    "MAX_AVAILABILITY_LAG",
    "MAX_FEATURES",
    "FeatureSpec",
    "compute_instant",
]

MAX_FEATURES: Final = 30
"""Hard cap on the number of registered features (count), directive §5 Phase 5.

Includes the Phase 7 LLM-derived features. Enforced by
:class:`backend.features.registry.FeatureRegistry`, which does not accept an
override, and proven by :mod:`backend.tests.features.test_registry`.
"""

MAX_AVAILABILITY_LAG: Final = dt.timedelta(days=3652)
"""Largest declarable availability lag (10 years), a guard against unit errors.

No equity feature is first knowable a decade after its event; a lag this large
is a typo or a units confusion. The bound also keeps
``compute_instant(date) - lag`` inside the representable range of
:class:`datetime.datetime` for any date the platform will ever be handed, so an
absurd declaration fails with a named error at construction rather than an
``OverflowError`` somewhere down the compute path.
"""

_NAME_PATTERN: Final = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
"""``snake_case``: lowercase, digits allowed, single underscores between parts."""

_TABLE_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
"""A physical table name as PostgreSQL holds it: lowercase, unquoted."""


def compute_instant(compute_date: dt.date) -> dt.datetime:
    """Return the UTC instant a compute date denotes: midnight opening the date.

    Args:
        compute_date: the calendar date a feature is being computed for. Must
            be a :class:`datetime.date` and **not** a
            :class:`datetime.datetime`; ``datetime`` is a subclass of ``date``
            and would have its time component silently discarded, turning a
            caller who meant 16:00 into one who gets 00:00 without being told.

    Returns:
        Midnight UTC opening ``compute_date``, timezone-aware.

    Raises:
        FeatureComputeError: if a ``datetime`` is passed instead of a ``date``.

    Example:
        >>> compute_instant(dt.date(2026, 3, 1)).isoformat()
        '2026-03-01T00:00:00+00:00'
    """
    if isinstance(compute_date, dt.datetime):
        msg = (
            f"compute_date must be a date, not a datetime; got {compute_date!r}. "
            f"A datetime's time component would be silently dropped, so the "
            f"instant you get would not be the instant you passed. Pass "
            f"{compute_date!r}.date() if that is what you mean."
        )
        raise FeatureComputeError(msg)
    return dt.datetime.combine(compute_date, dt.time.min, tzinfo=dt.UTC)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The declaration of one feature: what it is, its unit, and its lag.

    Frozen and hashable, so a declaration can be embedded in a config hash (I2)
    and cannot be edited after registration.

    Attributes:
        name: stable ``snake_case`` identifier, unique across the registry
            (e.g. ``momentum_12_1``). Used verbatim in configs, model
            artifacts, the dashboard and ``DECISIONS.md``, which is why the
            spelling is pinned rather than normalized.
        definition: prose stating what the number is and how it is computed,
            non-empty. This is what the Features page (§6.4) displays and what
            a reviewer reads before deciding whether the feature is redundant.
        units: **required**, free text, non-empty. Examples: ``"dimensionless
            z-score"``, ``"log return, fraction"``, ``"USD"``, ``"basis points
            of notional"``. There is no default and no ``None``: a number
            without a unit is not a measurement.
        availability_lag: wall-clock delay between an event and the moment its
            value is knowable. Non-negative, at most
            :data:`MAX_AVAILABILITY_LAG`. Zero is legal and means "knowable
            immediately" — appropriate for a feature built only from prices
            that are already in the store at the previous close.
        source_tables: physical fact tables the computation reads, non-empty,
            lowercase. Documentation and impact analysis: when a connector's
            revision policy changes, this is what says which features are
            affected. It is not an access-control list — enforcement of what a
            computation may read is temporal and lives in
            :mod:`backend.features.compute`.

    Raises:
        FeatureSpecError: if any field is malformed. Every check is at
            construction, so an invalid declaration cannot reach the registry.
    """

    name: str
    definition: str
    units: str
    availability_lag: dt.timedelta
    source_tables: frozenset[str]

    def __post_init__(self) -> None:
        """Validate the declaration and freeze ``source_tables``.

        Raises:
            FeatureSpecError: naming the offending field and what it must be.
        """
        if not _NAME_PATTERN.fullmatch(self.name):
            msg = (
                f"feature name {self.name!r} is not snake_case. Expected "
                f"lowercase words separated by single underscores, starting "
                f"with a letter (e.g. 'momentum_12_1'). The name is a stable "
                f"identifier across configs, artifacts and the dashboard, so it "
                f"is pinned to one spelling rather than normalized."
            )
            raise FeatureSpecError(msg)
        if not self.definition.strip():
            msg = (
                f"feature {self.name!r} has an empty definition. The catalog is "
                f"read by a human deciding whether this feature earns its place "
                f"against the 30-feature cap; an empty definition makes that "
                f"judgement impossible."
            )
            raise FeatureSpecError(msg)
        if not self.units.strip():
            msg = (
                f"feature {self.name!r} has empty units. Units are required "
                f"(directive §8): a float64 column whose unit is unrecorded is "
                f"the silent bug class this project spends the most effort "
                f"avoiding. Say 'dimensionless z-score' if that is the answer."
            )
            raise FeatureSpecError(msg)
        # Widened to `object` so the isinstance guards below are live code
        # rather than statements mypy proves unreachable from the annotations.
        # They are not redundant: declarations also arrive from config files and
        # test fixtures, where nothing has type-checked them.
        declared_lag: object = self.availability_lag
        if not isinstance(declared_lag, dt.timedelta):
            msg = (
                f"feature {self.name!r} declares availability_lag "
                f"{declared_lag!r}, which is not a timedelta. A bare number has "
                f"no unit; pass dt.timedelta(days=45)."
            )
            raise FeatureSpecError(msg)
        if self.availability_lag < dt.timedelta(0):
            msg = (
                f"feature {self.name!r} declares a negative availability_lag "
                f"{self.availability_lag}. A negative lag says the value is "
                f"knowable before the event that produces it, which is the "
                f"definition of lookahead (I1)."
            )
            raise FeatureSpecError(msg)
        if self.availability_lag > MAX_AVAILABILITY_LAG:
            msg = (
                f"feature {self.name!r} declares availability_lag "
                f"{self.availability_lag}, above the {MAX_AVAILABILITY_LAG} "
                f"limit. No equity feature is first knowable that long after "
                f"its event; this is a unit error, not a data property."
            )
            raise FeatureSpecError(msg)
        declared_tables: object = self.source_tables
        if isinstance(declared_tables, str):
            msg = (
                f"feature {self.name!r} declares source_tables as the string "
                f"{declared_tables!r}. A bare string is iterable, so it would "
                f"become a set of single characters rather than a set of one "
                f"table name; pass frozenset({{{declared_tables!r}}})."
            )
            raise FeatureSpecError(msg)
        tables = frozenset(self.source_tables)
        if not tables:
            msg = (
                f"feature {self.name!r} declares no source_tables. A feature "
                f"that reads nothing has no availability lag to enforce and "
                f"nothing to be point-in-time about; name the fact tables it "
                f"reads."
            )
            raise FeatureSpecError(msg)
        bad = sorted(table for table in tables if not _TABLE_PATTERN.fullmatch(table))
        if bad:
            msg = (
                f"feature {self.name!r} declares source table name(s) "
                f"{', '.join(repr(table) for table in bad)} that are not "
                f"lowercase unquoted identifiers. Name tables as PostgreSQL "
                f"holds them, so the declaration can be matched against the "
                f"schema by string equality."
            )
            raise FeatureSpecError(msg)
        # Freeze a caller's mutable set into the declaration: a spec that could
        # change its source tables after registration would make the config
        # hash a lie (I2).
        object.__setattr__(self, "source_tables", tables)

    def knowledge_cutoff(self, compute_date: dt.date) -> dt.datetime:
        """Return the freshest instant this feature may read, computing for a date.

        This is the whole content of the availability lag, expressed as the one
        number the compute path needs::

            midnight_utc(compute_date) - availability_lag

        Args:
            compute_date: the calendar date the feature is computed for. Must
                be a ``date``, not a ``datetime`` (see :func:`compute_instant`).

        Returns:
            A timezone-aware UTC instant. Facts whose ``knowledge_time`` is at
            or before it are visible; anything later is not. The boundary is
            inclusive, matching :func:`backend.db.as_of`.

        Raises:
            FeatureComputeError: if ``compute_date`` is a ``datetime``.

        Example:
            >>> spec = FeatureSpec(
            ...     name="book_to_price",
            ...     definition="Common equity over market capitalization.",
            ...     units="dimensionless ratio",
            ...     availability_lag=dt.timedelta(days=45),
            ...     source_tables=frozenset({"fundamentals"}),
            ... )
            >>> spec.knowledge_cutoff(dt.date(2026, 3, 1)).isoformat()
            '2026-01-15T00:00:00+00:00'
        """
        return compute_instant(compute_date) - self.availability_lag
