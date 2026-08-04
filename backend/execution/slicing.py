"""VWAP and TWAP slicing: one parent order becomes a schedule of child orders (P11.4).

What this module is, and what it deliberately is not
----------------------------------------------------

:func:`plan_twap` and :func:`plan_vwap` are **pure functions from a parent
:class:`~backend.execution.orders.OrderIntent` to a tuple of child intents**.
They read no clock, open no session, touch no database and take no destination —
the same structural claim :mod:`backend.execution.orders` and
:mod:`backend.execution.reconciliation` make, for the same reason (directive
§1.1, §9.5). A schedule is *content*: given the same parent and the same
forecast it is byte-for-byte the same schedule, which is what lets every child
carry a content-derived idempotency key.

A schedule carries **no timestamps**. The children are ordered, and the ordering
is the schedule; when each one is released is the caller's decision. That is not
squeamishness about scope, it is I3: this platform has daily bars and nothing
finer (B1 — Sharadar SEP is end-of-day), so a module that stamped
``10:00:00``/``10:30:00`` onto its slices would be manufacturing a precision no
data in the system supports. Buckets here are *equal-length intervals of one
regular session*, indexed and nothing more.

TWAP is trivial. VWAP is not, and the difference is a forecast
--------------------------------------------------------------

**TWAP** splits the parent evenly across ``n`` equal-duration buckets. It needs
no information about the market at all — that is its entire appeal, and it is why
it is implemented here as the degenerate case of the VWAP path under a forecast
that declares no shape (:func:`uniform_volume_forecast`). There is no separate
TWAP arithmetic to get wrong.

**VWAP** weights each bucket by the volume the market is *expected* to trade in
it, so that the order participates evenly in volume rather than evenly in time.
Every word of that sentence except "expected" is arithmetic; "expected" is a
forecast, and a forecast is a claim about the future that has to come from
somewhere, be checkable, and be allowed to be wrong.

Where the forecast comes from, and why it cannot look ahead (I1)
----------------------------------------------------------------

**The trap.** A VWAP schedule built from the volume that *actually* traded in
each bucket of the same session is not a forecast — it is the answer. It would
also flatter every backtest that used it, silently and enormously, because
knowing the day's volume profile is knowing where the day's liquidity was. This
is invariant I1 (directive §2) in its purest form, and it is the single thing
this module is built to make unrepresentable.

Three defences, none of them a convention:

1. :class:`VolumeForecast` states the date it is **for** (``as_of``) and the last
   session it **consumed** (``observed_through``), and refuses construction
   unless ``observed_through < as_of``. Strictly before: same-day realised
   volume is exactly the prohibited input, so equality is a violation, not a
   boundary case.
2. :func:`observed_volume_forecast` does not accept a claim about its own
   temporal bound — it takes the :class:`SessionVolume` observations themselves,
   each carrying its session date, and *derives* ``observed_through`` as their
   maximum. A caller cannot understate what the forecast saw, because the
   forecast computes it from what it was given, and refuses outright if any
   session is dated on or after ``as_of``
   (:class:`~backend.execution.errors.ForecastLookaheadError`).
3. :class:`SliceSchedule` re-checks the same inequality against its own
   ``rebalance_date``. A schedule that survives construction has, as a property
   of its type, consumed nothing dated on or after the day it trades.

**Today no fitted forecast can exist, and that is the honest state.**
:attr:`ForecastBasis.OBSERVED_PROFILE` is the only basis that is a forecast
rather than an assumption, and it requires intraday volume history the platform
does not have (B1). So the shapes available now are ``UNIFORM`` (no claim) and
``STYLISED_U_CURVE`` (an assumption, see below), and
:attr:`VolumeForecast.fitted` is ``False`` for both. That flag travels onto the
schedule and onto every cost estimate derived from it, in the same shape as
:attr:`~backend.costs.model.TradeCost.uncalibrated`, so "VWAP" never reads as
"fitted to observed volume" anywhere downstream.

The U-curve is a stylised assumption, not a fitted model (I3)
-------------------------------------------------------------

Intraday volume in US equities is a well-documented U (heavier at the open and
the close, lighter around midday) — Wood, McInish & Ord (1985); Harris (1986);
Admati & Pfleiderer (1988); Jain & Joh (1988). That is a **stylised fact from the
literature**, and :func:`stylised_u_curve_forecast` reproduces its shape.

It is not a model of anything in this system. Nothing here was fitted, because
there is no intraday data to fit it to (B1). The curve is a symmetric parabola
with one visible, chosen constant
(:data:`U_CURVE_EDGE_TO_MIDDLE_RATIO`), and its two most important properties are
stated rather than discovered:

- The real curve is **asymmetric** — the close is heavier than the open. This one
  is symmetric, deliberately, because modelling the asymmetry means choosing a
  second unfitted number and the second number would look like a measurement.
- The ratio is an assumption. It is a module constant rather than an argument so
  that it is read once, in one place, by anyone auditing a schedule.

Every forecast carries a ``basis_detail`` string saying which of these it is, and
the string is on the schedule and in every summary, not in a document.

**When the forecast is wrong.** The schedule is *static*: it is computed once and
never re-forecast, because re-forecasting requires observing the session as it
happens and this package observes nothing. If the day's volume arrives in a
different shape, the order over-participates in some buckets and
under-participates in others, and the realised participation-weighted price
drifts from the day's true VWAP by an amount **this system cannot measure**,
having no intraday prints to measure it against. So: no tracking-error claim may
be made about this module, now or after P11.1. It produces a defensible
*participation shape*, not a benchmark-tracking algorithm, and calling it VWAP is
a description of the weighting, not a promise about the outcome.

What stops the optimizer from slicing infinitely (I4)
------------------------------------------------------

:mod:`backend.costs.impact` is square-root in participation, so splitting a
parent into ``n`` equal slices multiplies each slice's impact **in basis points**
by ``1/sqrt(n)`` while leaving total notional unchanged — total impact **in
dollars** therefore falls as ``1/sqrt(n)``, without limit. Spread and commission
dollars do not move at all, because the model quotes both as a flat rate on
notional. A cost-minimising search over ``n`` under this model does not converge:
it slices until it runs out of shares. That is not a finding about markets, it is
a property of a model that charges nothing for slicing.

Three things stop it here, and only the first two are enforceable arithmetic:

1. **A minimum slice size, in whole shares.** Slices are integers, so ``n`` can
   never exceed the parent quantity; the operative bound is tighter. The default
   minimum is one round lot (:data:`ROUND_LOT_SHARES`, 100 shares — the US equity
   convention), so a 1,000-share parent admits at most ten slices. An odd-lot
   child is not a smaller round-lot child: odd lots are handled differently by
   venues and have not historically been displayed in the protected quotation,
   so the model's assumption that cost scales smoothly with size is least
   defensible exactly where slicing pushes it.
2. **A schedule horizon of one session.** :data:`MAX_SLICE_COUNT` is
   ``REGULAR_SESSION_MINUTES // BUCKET_MINUTES`` — thirteen 30-minute buckets in
   the 390-minute US regular session, 30 minutes being the standard bin of the
   intraday-volume literature cited above. And every child carries the parent's
   ``rebalance_date`` unchanged, so a schedule *structurally* cannot span
   sessions: an order still working tomorrow is a different position from the one
   the optimizer chose today.
3. **A per-slice fixed cost, which this system's cost model does not have.**
   Real per-order costs exist — IBKR's published US equity schedules carry a
   per-order commission minimum, and each additional child is another message,
   another acknowledgement and another thing reconciliation must account for. The
   model in :mod:`backend.costs.model` expresses commission as a flat
   ``commission_bps`` of notional and has **no per-order term at all**, so it
   cannot see any of that. Neither does it have a timing-risk term: stretching an
   order over the session trades impact for exposure to price drift, which is the
   whole content of the Almgren-Chriss trade-off and is absent here.

So the slice-count bound in this module is **exogenous and stated**, never
optimised. :func:`modelled_schedule_cost` exists to make that checkable rather
than rhetorical: it costs a schedule through
:func:`~backend.costs.model.estimate_trade_cost` — the real model, with its real
``uncalibrated`` flag propagated (I4) — and the test suite uses it to demonstrate
that modelled cost decreases monotonically in ``n`` all the way to the ceiling.
A number that only ever goes down is not a recommendation to slice; it is the
model telling you it is missing a term.

Exact summation, in integers only
----------------------------------

A schedule must sum to **exactly** the parent quantity. Rounding leaks are the
characteristic slicing bug: ten slices of a 1,003-share parent that sum to 1,000
leave three shares that no order will ever trade and that reconciliation will
report as a break days later.

:func:`apportion_shares` therefore never touches a float. Weights are integers,
every slice receives the minimum first, and the remaining shares are distributed
by **largest remainder** (Hamilton apportionment) with ties broken by the lowest
bucket index. The sum is exact by construction rather than by rounding luck:
``sum(floor(D*w_i/W))`` differs from ``D`` by an integer strictly between ``0``
and ``n``, and that many single shares are handed out. Two further consequences,
both asserted in the tests: no allocation differs from its exact quota by a whole
share, and the allocation is monotone in the weights (a bucket with more expected
volume never receives fewer shares than one with less).

:class:`SliceSchedule` re-checks the sum in ``__post_init__``, so the property is
enforced by the type rather than only by the function that usually builds it.

Idempotency
-----------

Each child gets its own key, from P11.2's recipe with nothing added:
:func:`~backend.execution.idempotency.idempotency_key` hashes the intent's
content, which already includes ``slice_index`` and ``slice_count``. Two children
of one parent therefore differ in the preimage and cannot collide, and two runs
of the same plan — same parent, same forecast, same I2 stamp — produce the same
keys in the same order, so re-running a rebalance is absorbed by the database's
unique constraint instead of doubling the book.

Units
-----

Quantities are **whole shares** throughout. Weights are **dimensionless integer
units** whose scale is irrelevant (only ratios are used). Prices are **US dollars
per share**. Costs are **US dollars**, converted from the cost model's basis
points at the boundary and nowhere else. Bucket durations are **minutes**.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from backend.costs.model import (
    UNCALIBRATED_DEFAULTS,
    CostModelParams,
    estimate_trade_cost,
)
from backend.costs.model import Order as CostedOrder
from backend.costs.units import bps_of_notional_to_usd
from backend.execution.errors import (
    ForecastLookaheadError,
    SliceScheduleError,
    VolumeForecastError,
)
from backend.execution.idempotency import idempotency_key
from backend.execution.orders import OrderIntent

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BUCKET_MINUTES",
    "MAX_SLICE_COUNT",
    "MINIMUM_SLICE_SHARES_FLOOR",
    "REGULAR_SESSION_MINUTES",
    "ROUND_LOT_SHARES",
    "U_CURVE_EDGE_TO_MIDDLE_RATIO",
    "ForecastBasis",
    "ScheduleCost",
    "SessionVolume",
    "SliceAlgorithm",
    "SliceSchedule",
    "VolumeForecast",
    "apportion_shares",
    "feasible_slice_count",
    "modelled_schedule_cost",
    "observed_volume_forecast",
    "plan_twap",
    "plan_vwap",
    "stylised_u_curve_forecast",
    "stylised_u_curve_weights",
    "uniform_volume_forecast",
]

REGULAR_SESSION_MINUTES: Final = 390
"""Length of one US equity regular session, in **minutes** (09:30-16:00).

A stated market convention, not a calendar: this package has no session calendar
and no clock, and holidays and early closes are the caller's problem. The number
exists so that :data:`MAX_SLICE_COUNT` is derived from something rather than
picked.
"""

BUCKET_MINUTES: Final = 30
"""Length of one schedule bucket, in **minutes**.

Thirty minutes is the standard bin of the intraday-volume literature (Wood,
McInish & Ord 1985; Jain & Joh 1988), which is the only reason it is this number
and not another. It is emphatically *not* a claim that this platform resolves
thirty-minute intervals: it has daily bars and nothing finer (B1). Any bucket
finer than a day is an assumption, and this one is the assumption the published
stylised facts are stated in.
"""

MAX_SLICE_COUNT: Final = REGULAR_SESSION_MINUTES // BUCKET_MINUTES
"""Largest number of slices any schedule may have. Derived: 390 / 30 = 13.

The horizon bound from the module docstring, made arithmetic. A schedule cannot
have more buckets than the session has, and it cannot reach into the next session
because every child carries the parent's ``rebalance_date``. Raising this
constant is a claim that the session is longer or the buckets are shorter, and
neither is something a configuration should be able to assert.
"""

ROUND_LOT_SHARES: Final = 100
"""The US equity round lot, in **whole shares**. Default minimum slice size.

The other half of the anti-infinite-slicing bound. It is a market convention
rather than a derivation, and it is the default rather than a hard floor —
:func:`plan_twap` and :func:`plan_vwap` take the minimum as an argument so a
caller trading a low-priced name can lower it — but it cannot go below
:data:`MINIMUM_SLICE_SHARES_FLOOR`.
"""

MINIMUM_SLICE_SHARES_FLOOR: Final = 1
"""Smallest minimum-slice size any caller may ask for, in **whole shares**.

Zero is not available. A zero-share child is not an order (the
:class:`~backend.execution.orders.OrderIntent` constructor refuses it), and a
minimum of zero would re-open the unbounded slicing the module docstring is
about.
"""

U_CURVE_EDGE_TO_MIDDLE_RATIO: Final = 3
"""Expected volume in an edge bucket over a middle bucket. **An assumption.**

Dimensionless. It says the first and last buckets of a session are modelled as
carrying three times the volume of the midday trough, which is the right order of
magnitude for the published U shape and is otherwise **a number that was chosen,
not estimated** — there is no intraday volume in this system to estimate it from
(B1, I3).

It is a module constant rather than a parameter on purpose: a knob would invite
tuning, and tuning an unfitted constant against backtest results is how an
assumption is laundered into a finding.
"""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`SliceScheduleError` when ``condition`` is false.

    Args:
        condition: the requirement that must hold.
        message: what the caller got wrong, quoted verbatim in the error.

    Raises:
        SliceScheduleError: when ``condition`` is false.
    """
    if not condition:
        raise SliceScheduleError(message)


def _require_forecast(condition: bool, message: str) -> None:
    """Raise :class:`VolumeForecastError` when ``condition`` is false.

    Args:
        condition: the requirement that must hold.
        message: what the caller got wrong, quoted verbatim in the error.

    Raises:
        VolumeForecastError: when ``condition`` is false.
    """
    if not condition:
        raise VolumeForecastError(message)


def _whole(value: int) -> bool:
    """Return whether ``value`` is a plain ``int`` and not a ``bool``.

    ``bool`` is excluded explicitly: ``True`` is an ``int`` and would silently
    read as a quantity or a weight of one.

    Args:
        value: the value to classify.

    Returns:
        ``True`` if the value is an ``int`` that is not a ``bool``.
    """
    checked: object = value
    return isinstance(checked, int) and not isinstance(checked, bool)


def _require_no_lookahead(forecast: VolumeForecast, trading_date: dt.date) -> None:
    """Refuse a forecast that consumed a session on or after the day it schedules.

    The I1 check, factored out because it is made twice on two different dates and
    neither is redundant. :class:`VolumeForecast` guarantees only that its
    observations predate its own ``as_of``; this guarantees that they predate the
    *session actually being traded*, which is a different date whenever a forecast
    built for one day is handed to a schedule for another. That mismatch is caught
    on its own terms elsewhere, but the lookahead is checked first: a forecast
    reaching past the trading day is an invariant violation, not a filing error.

    Args:
        forecast: the weighting being used.
        trading_date: the session the schedule trades in.

    Raises:
        ForecastLookaheadError: if ``forecast.observed_through`` is on or after
            ``trading_date``.
    """
    observed = forecast.observed_through
    if observed is not None and observed >= trading_date:
        message = (
            f"the forecast consumed sessions through {observed.isoformat()}, which is on or "
            f"after the session it would schedule, {trading_date.isoformat()}. A volume "
            f"forecast may only consume sessions strictly earlier than the one it is for "
            f"(I1): weighting a schedule by volume that had not traded when the schedule was "
            f"built is not a forecast, it is the answer, and every backtest built on it would "
            f"look excellent and be worthless"
        )
        raise ForecastLookaheadError(message)


def _is_plain_date(value: dt.date) -> bool:
    """Return whether ``value`` is a calendar date rather than an instant.

    :class:`datetime.datetime` is a subclass of :class:`datetime.date`, so it
    would pass an ``isinstance`` check silently. It is refused because a
    timestamp on a forecast implies an intraday resolution this platform does not
    have (B1), and because the temporal bound the forecast enforces is a
    *session* comparison.

    Args:
        value: the date to classify.

    Returns:
        ``True`` if the value is a date and not a datetime.
    """
    return isinstance(value, dt.date) and not isinstance(value, dt.datetime)


class ForecastBasis(StrEnum):
    """Where a volume forecast's shape came from. Values are persisted and displayed.

    The distinction this enum exists for is the one between a **measurement** and
    an **assumption**, which is the same distinction
    :attr:`~backend.costs.model.CostModelParams.uncalibrated` draws for costs. It
    is a value on the object rather than a note in a document, so it survives
    into every summary, artifact and operator view.

    - ``UNIFORM`` — every bucket weighted equally. This is not a forecast; it is
      the explicit refusal to make one, and it is what TWAP uses.
    - ``STYLISED_U_CURVE`` — the published intraday U shape (Wood, McInish & Ord
      1985; Admati & Pfleiderer 1988; Jain & Joh 1988), reproduced as a symmetric
      parabola. **An assumption, fitted to nothing.**
    - ``OBSERVED_PROFILE`` — normalised realised per-bucket volume from sessions
      strictly earlier than the schedule's date. The only member that denotes a
      forecast, and the only one that cannot be produced today: the platform has
      daily bars and no intraday history (B1). It exists so that when such
      history arrives, the fitted case has a name that was always distinct from
      the assumed one, rather than being retrofitted onto it.
    """

    UNIFORM = "uniform"
    STYLISED_U_CURVE = "stylised_u_curve"
    OBSERVED_PROFILE = "observed_profile"

    @property
    def fitted(self) -> bool:
        """Whether a forecast on this basis was derived from observed volume.

        ``True`` only for :attr:`OBSERVED_PROFILE`. Everything else is a stated
        shape, and a consumer that shows a VWAP schedule must be able to say
        which it is looking at without consulting prose.
        """
        return self is ForecastBasis.OBSERVED_PROFILE


class SliceAlgorithm(StrEnum):
    """Which schedule shape produced a set of children. Values are persisted.

    ``TWAP`` and ``VWAP`` differ only in the forecast they were planned against —
    ``TWAP`` is ``VWAP`` under :attr:`ForecastBasis.UNIFORM` — so this is a label
    for the blotter and the audit trail rather than a switch anything branches on.
    Keeping it explicit means a schedule can say what it was asked to be, not only
    what it turned out to be: a VWAP planned against a uniform forecast and a
    TWAP are the same numbers and are not the same decision.
    """

    TWAP = "twap"
    VWAP = "vwap"


@dataclass(frozen=True, slots=True)
class SessionVolume:
    """Realised per-bucket share volume for one past session.

    The input to :func:`observed_volume_forecast`, and the reason that function
    cannot be lied to about its own temporal bound: it derives
    ``observed_through`` from these dates rather than accepting a claim about
    them.

    Units: ``bucket_volume_shares`` is **whole shares per bucket**, in bucket
    order, one entry per bucket of the session.

    Attributes:
        session_date: the session these volumes were traded in. A calendar date;
            a :class:`datetime.datetime` is refused.
        bucket_volume_shares: shares traded in each bucket, non-negative whole
            numbers, at least one bucket and no more than
            :data:`MAX_SLICE_COUNT`. The session total must be strictly positive
            — a session in which nothing traded is not evidence about where
            volume falls within a session.
    """

    session_date: dt.date
    bucket_volume_shares: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the session date and the per-bucket volumes.

        Raises:
            VolumeForecastError: if the date is not a plain calendar date, if the
                volumes are not a tuple, if there are none or more than
                :data:`MAX_SLICE_COUNT`, if any is not a non-negative whole
                number, or if the session total is zero.
        """
        _require_forecast(
            _is_plain_date(self.session_date),
            f"session_date must be a calendar date, got {self.session_date!r}; a timestamp "
            f"implies an intraday resolution this platform does not have",
        )
        volumes: object = self.bucket_volume_shares
        _require_forecast(
            isinstance(volumes, tuple),
            f"bucket_volume_shares must be a tuple, got {type(volumes).__name__}; a mutable "
            f"sequence inside a frozen value object is a value that can change under its own "
            f"identity",
        )
        _require_forecast(
            1 <= len(self.bucket_volume_shares) <= MAX_SLICE_COUNT,
            f"a session must carry between 1 and {MAX_SLICE_COUNT} buckets, got "
            f"{len(self.bucket_volume_shares)}",
        )
        for index, volume in enumerate(self.bucket_volume_shares):
            _require_forecast(
                _whole(volume) and volume >= 0,
                f"bucket {index} volume must be a non-negative whole number of shares "
                f"(bool is refused), got {volume!r}",
            )
        _require_forecast(
            sum(self.bucket_volume_shares) > 0,
            f"session {self.session_date.isoformat()} traded no shares in any bucket; a "
            f"session with no volume says nothing about where volume falls within a session",
        )

    @property
    def bucket_count(self) -> int:
        """Number of buckets this session was observed in."""
        return len(self.bucket_volume_shares)


@dataclass(frozen=True, slots=True)
class VolumeForecast:
    """Expected share of volume per bucket, and the provenance of that expectation.

    The weights are **dimensionless integers** and only their ratios matter, so
    no normalisation to 1 ever happens and no float is ever introduced. That is
    what lets :func:`apportion_shares` be exact.

    The two dates are the invariant. ``as_of`` is the session this forecast is
    *for*; ``observed_through`` is the last session whose realised volume went
    into it. Construction refuses ``observed_through >= as_of``, strictly, because
    same-day realised volume is precisely the input I1 exists to prohibit and
    "the schedule was built before the open" is not a property a type can check.

    Attributes:
        weight_units: expected relative volume per bucket, in bucket order. Each
            entry is a strictly positive whole number; a bucket expected to trade
            nothing is not a bucket to schedule into, and the constructors say so
            rather than allocating zero shares to it.
        basis: which kind of thing this shape is — see :class:`ForecastBasis`.
        basis_detail: prose stating where the shape came from, displayed wherever
            a schedule is. Must be non-empty, for the reason
            :attr:`~backend.costs.model.CostModelParams.calibration_basis` must
            be: an empty provenance makes an assumption indistinguishable from a
            measurement.
        as_of: the session this forecast applies to. Must equal the parent
            order's ``rebalance_date`` when the schedule is planned.
        observed_through: the latest session consumed, or ``None`` when nothing
            was. Present exactly when :attr:`fitted` is ``True``.
    """

    weight_units: tuple[int, ...]
    basis: ForecastBasis
    basis_detail: str
    as_of: dt.date
    observed_through: dt.date | None

    def __post_init__(self) -> None:
        """Validate the weights, the provenance, and the I1 temporal bound.

        Raises:
            VolumeForecastError: if the weights are not a tuple of strictly
                positive whole numbers, if there are none or more than
                :data:`MAX_SLICE_COUNT`, if ``basis_detail`` is blank, if either
                date is not a plain calendar date, or if the presence of
                ``observed_through`` disagrees with :attr:`fitted`.
            ForecastLookaheadError: if ``observed_through`` is on or after
                ``as_of``. This is invariant I1 and it is a distinct type
                because it is not a validation slip — a forecast built from the
                session it is forecasting is the lookahead that makes a backtest
                worthless.
        """
        weights: object = self.weight_units
        _require_forecast(
            isinstance(weights, tuple),
            f"weight_units must be a tuple, got {type(weights).__name__}",
        )
        _require_forecast(
            1 <= len(self.weight_units) <= MAX_SLICE_COUNT,
            f"a forecast must carry between 1 and {MAX_SLICE_COUNT} buckets, got "
            f"{len(self.weight_units)}; the ceiling is one regular session divided into "
            f"{BUCKET_MINUTES}-minute buckets",
        )
        for index, weight in enumerate(self.weight_units):
            _require_forecast(
                _whole(weight) and weight > 0,
                f"bucket {index} weight must be a strictly positive whole number (bool is "
                f"refused), got {weight!r}; a bucket with no expected volume is not a bucket "
                f"to schedule into — build the forecast with fewer buckets instead",
            )
        _require_forecast(
            self.basis_detail.strip() != "",
            "basis_detail must state where this shape came from; an empty provenance makes "
            "an assumed volume curve indistinguishable from a measured one",
        )
        _require_forecast(
            _is_plain_date(self.as_of),
            f"as_of must be a calendar date, got {self.as_of!r}",
        )
        self._validate_temporal_bound()

    def _validate_temporal_bound(self) -> None:
        """Check ``observed_through`` against :attr:`fitted` and against ``as_of``.

        Raises:
            VolumeForecastError: if the presence of an observation date
                disagrees with the basis, or if it is not a plain calendar date.
            ForecastLookaheadError: if it is on or after ``as_of``.
        """
        observed = self.observed_through
        _require_forecast(
            (observed is not None) == self.fitted,
            f"basis {self.basis.value!r} is "
            f"{'fitted' if self.fitted else 'not fitted'}, so observed_through must be "
            f"{'present' if self.fitted else 'None'}, got {observed!r}; a forecast that "
            f"claims observations it did not make, or hides ones it did, cannot be checked "
            f"against I1 at all",
        )
        if observed is None:
            return
        _require_forecast(
            _is_plain_date(observed),
            f"observed_through must be a calendar date, got {observed!r}",
        )
        if observed >= self.as_of:
            message = (
                f"observed_through={observed.isoformat()} is on or after "
                f"as_of={self.as_of.isoformat()}. A volume forecast may only consume "
                f"sessions strictly earlier than the one it is for (I1): weighting a "
                f"schedule by the volume that actually traded on the day it trades is not a "
                f"forecast, it is the answer, and every backtest built on it would look "
                f"excellent and be worthless"
            )
            raise ForecastLookaheadError(message)

    @property
    def bucket_count(self) -> int:
        """Number of buckets this forecast covers."""
        return len(self.weight_units)

    @property
    def total_weight_units(self) -> int:
        """Sum of the weights, in dimensionless integer units. Strictly positive."""
        return sum(self.weight_units)

    @property
    def fitted(self) -> bool:
        """Whether this shape was derived from observed volume rather than assumed."""
        return self.basis.fitted

    def summary(self) -> dict[str, object]:
        """Return a JSON-safe description for operator display and run artifacts.

        Returns:
            Mapping of the weights, the basis and its prose, both dates, and a
            ``units`` key stating explicitly that the weights are dimensionless
            — a reader who mistakes them for shares gets a schedule that looks
            plausible and is not.
        """
        return {
            "basis": self.basis.value,
            "basis_detail": self.basis_detail,
            "fitted": self.fitted,
            "as_of": self.as_of.isoformat(),
            "observed_through": (
                None if self.observed_through is None else self.observed_through.isoformat()
            ),
            "bucket_count": self.bucket_count,
            "weight_units": list(self.weight_units),
            "units": "weight_units are dimensionless integers; only their ratios are used",
        }


def uniform_volume_forecast(*, bucket_count: int, as_of: dt.date) -> VolumeForecast:
    """Return a forecast that weights every bucket equally — the TWAP shape.

    This is the explicit refusal to forecast, and it is the honest default when
    nothing is known about the session's shape. It consumes no observations, so
    it carries no ``observed_through`` and cannot look ahead by construction.

    Args:
        bucket_count: number of equal-duration buckets, between 1 and
            :data:`MAX_SLICE_COUNT`.
        as_of: the session the schedule will trade in.

    Returns:
        A :class:`VolumeForecast` on :attr:`ForecastBasis.UNIFORM` whose weights
        are all ``1``.

    Raises:
        VolumeForecastError: if ``bucket_count`` is outside its range or
            ``as_of`` is not a calendar date.
    """
    _require_forecast(
        _whole(bucket_count) and bucket_count >= 1,
        f"bucket_count must be a whole number of at least 1, got {bucket_count!r}",
    )
    return VolumeForecast(
        weight_units=(1,) * bucket_count,
        basis=ForecastBasis.UNIFORM,
        basis_detail=(
            "uniform across buckets — no volume forecast was made; this is the TWAP "
            "weighting and it claims nothing about the session's shape"
        ),
        as_of=as_of,
        observed_through=None,
    )


def stylised_u_curve_weights(bucket_count: int) -> tuple[int, ...]:
    """Return the assumed U-shaped intraday volume weights, as integers.

    A symmetric parabola in the bucket index, scaled so that an edge bucket
    carries :data:`U_CURVE_EDGE_TO_MIDDLE_RATIO` times the weight of the midpoint.
    With ``span = bucket_count - 1`` and ``d = |2i - span|`` (the doubled distance
    from the midpoint, an integer), the weight of bucket ``i`` is::

        span**2 + (U_CURVE_EDGE_TO_MIDDLE_RATIO - 1) * d**2

    which is ``span**2`` at the midpoint and ``U_CURVE_EDGE_TO_MIDDLE_RATIO *
    span**2`` at either edge. Integer arithmetic throughout, so the weights are
    exact and identical on every machine.

    **This is a stylised assumption, not a fitted model** (I3). The U shape is a
    published stylised fact (Wood, McInish & Ord 1985; Harris 1986; Admati &
    Pfleiderer 1988; Jain & Joh 1988); the ratio is a chosen constant; and the
    real curve is asymmetric in a way this deliberately is not — see the module
    docstring.

    Args:
        bucket_count: number of buckets, between 1 and :data:`MAX_SLICE_COUNT`.
            One bucket is the degenerate case and returns ``(1,)``: a single
            bucket has no shape.

    Returns:
        Strictly positive integer weights in bucket order, symmetric about the
        midpoint. Dimensionless — only the ratios are used.

    Raises:
        VolumeForecastError: if ``bucket_count`` is not a whole number of at
            least 1.
    """
    _require_forecast(
        _whole(bucket_count) and bucket_count >= 1,
        f"bucket_count must be a whole number of at least 1, got {bucket_count!r}",
    )
    if bucket_count == 1:
        return (1,)
    span = bucket_count - 1
    trough = span * span
    curvature = U_CURVE_EDGE_TO_MIDDLE_RATIO - 1
    return tuple(trough + curvature * (2 * index - span) ** 2 for index in range(bucket_count))


def stylised_u_curve_forecast(*, bucket_count: int, as_of: dt.date) -> VolumeForecast:
    """Return a forecast on the assumed intraday U shape.

    Consumes no observations — there are none to consume (B1) — so it carries no
    ``observed_through`` and :attr:`VolumeForecast.fitted` is ``False``. That flag
    and the ``basis_detail`` below travel onto the schedule and into every cost
    estimate, so nothing downstream can present this as a measured volume profile.

    Args:
        bucket_count: number of equal-duration buckets, between 1 and
            :data:`MAX_SLICE_COUNT`.
        as_of: the session the schedule will trade in.

    Returns:
        A :class:`VolumeForecast` on :attr:`ForecastBasis.STYLISED_U_CURVE`.

    Raises:
        VolumeForecastError: if ``bucket_count`` is outside its range or
            ``as_of`` is not a calendar date.
    """
    return VolumeForecast(
        weight_units=stylised_u_curve_weights(bucket_count),
        basis=ForecastBasis.STYLISED_U_CURVE,
        basis_detail=(
            f"STYLISED ASSUMPTION, NOT FITTED — symmetric parabolic U with an edge-to-middle "
            f"ratio of {U_CURVE_EDGE_TO_MIDDLE_RATIO}:1. The U shape is a published stylised "
            f"fact for US equity intraday volume (Wood, McInish & Ord 1985; Harris 1986; "
            f"Admati & Pfleiderer 1988; Jain & Joh 1988); the ratio is a chosen constant and "
            f"nothing here was estimated from data, because this platform has daily bars and "
            f"no intraday history (BLOCKERS.md B1). The real curve is asymmetric — the close "
            f"is heavier than the open — and this one is not"
        ),
        as_of=as_of,
        observed_through=None,
    )


def observed_volume_forecast(
    sessions: Sequence[SessionVolume],
    *,
    as_of: dt.date,
) -> VolumeForecast:
    """Return a forecast fitted to realised per-bucket volume from earlier sessions.

    The only basis that is a forecast rather than an assumption, and the only one
    that can look ahead — so this is where I1 is enforced hardest. **Every session
    must be dated strictly before ``as_of``**, and the resulting forecast's
    ``observed_through`` is *derived* as the maximum session date rather than
    accepted from the caller. There is no argument through which a caller could
    understate what the forecast consumed.

    Weights are the per-bucket **sum** across sessions. Summing and averaging give
    the same weights because only ratios are used, so no division and no float is
    introduced; and sessions are weighted equally rather than decayed, because a
    decay rate is another unfitted constant and one assumption at a time is the
    limit.

    **Nothing in this platform can call this today.** It needs intraday volume
    history, which the data stack does not have (B1, Sharadar SEP is end-of-day).
    It exists so the fitted case has always been a distinct value rather than one
    retrofitted onto the assumed case later.

    Args:
        sessions: realised per-bucket volumes, at least one session, every
            session covering the same number of buckets.
        as_of: the session the schedule will trade in. Every observation must
            predate it strictly.

    Returns:
        A :class:`VolumeForecast` on :attr:`ForecastBasis.OBSERVED_PROFILE` whose
        ``observed_through`` is the latest session date supplied.

    Raises:
        VolumeForecastError: if no sessions are given, if they disagree about the
            bucket count, if ``as_of`` is not a calendar date, or if some bucket
            traded nothing in any session (that bucket is not one to schedule
            into — supply fewer buckets).
        ForecastLookaheadError: if any session is dated on or after ``as_of``.
    """
    _require_forecast(
        len(sessions) >= 1,
        "at least one observed session is required; a forecast fitted to nothing is not a forecast",
    )
    _require_forecast(
        _is_plain_date(as_of),
        f"as_of must be a calendar date, got {as_of!r}",
    )
    bucket_count = sessions[0].bucket_count
    for session in sessions:
        _require_forecast(
            session.bucket_count == bucket_count,
            f"sessions disagree about the bucket count: {sessions[0].session_date.isoformat()} "
            f"has {bucket_count} and {session.session_date.isoformat()} has "
            f"{session.bucket_count}; a profile summed across different griddings is not a "
            f"profile of anything",
        )
        if session.session_date >= as_of:
            message = (
                f"session {session.session_date.isoformat()} is on or after "
                f"as_of={as_of.isoformat()}. A volume forecast may only consume sessions "
                f"strictly earlier than the one it is for (I1); same-day realised volume is "
                f"the answer, not a forecast of it"
            )
            raise ForecastLookaheadError(message)
    totals = tuple(
        sum(session.bucket_volume_shares[index] for session in sessions)
        for index in range(bucket_count)
    )
    for index, total in enumerate(totals):
        _require_forecast(
            total > 0,
            f"bucket {index} traded nothing in any of the {len(sessions)} observed sessions; "
            f"a bucket with no observed volume is not a bucket to schedule into — build the "
            f"forecast with fewer buckets instead",
        )
    observed_through = max(session.session_date for session in sessions)
    earliest = min(session.session_date for session in sessions)
    return VolumeForecast(
        weight_units=totals,
        basis=ForecastBasis.OBSERVED_PROFILE,
        basis_detail=(
            f"fitted to realised per-bucket share volume summed over {len(sessions)} "
            f"session(s), {earliest.isoformat()} to {observed_through.isoformat()}, weighted "
            f"equally; every session is strictly earlier than {as_of.isoformat()} (I1)"
        ),
        as_of=as_of,
        observed_through=observed_through,
    )


def apportion_shares(
    *,
    total_shares: int,
    weight_units: Sequence[int],
    minimum_shares: int,
) -> tuple[int, ...]:
    """Split a whole-share quantity across weighted buckets, summing exactly.

    Largest-remainder (Hamilton) apportionment over a floor, in integers only:

    1. Every bucket receives ``minimum_shares``.
    2. The remaining ``D = total_shares - n * minimum_shares`` shares are divided
       by weight: bucket ``i`` gets ``floor(D * w_i / W)``, where ``W`` is the
       total weight.
    3. The ``D - sum(floor(...))`` shares still unassigned — an integer strictly
       less than ``n`` — are handed out one each to the buckets with the largest
       fractional remainder, ties broken by the **lowest bucket index**.

    The sum is exact by construction, never by rounding luck, and no float is
    involved at any step. Two properties follow and are asserted in the tests: no
    bucket differs from its exact quota ``minimum_shares + D * w_i / W`` by a
    whole share, and the allocation is monotone in the weights.

    **Why the floor comes first.** Apportioning the whole quantity by weight and
    hoping every bucket clears the minimum does not work: with a skewed shape and
    a small parent, a low-weight bucket rounds to zero, and a zero-share child is
    not an order. Giving every bucket its minimum first makes the floor exact
    rather than probable, at the cost of flattening the shape when the parent is
    small — which is the minimum-slice rule doing precisely its job, visibly.

    **Why ties go to the earliest bucket.** It has to be deterministic, or one
    plan produces two sets of idempotency keys. Given a choice, the earlier bucket
    wins because a schedule can be stopped part-way — the kill switch (P11.5) is a
    real event here — and quantity left for later is quantity that may never
    trade.

    Args:
        total_shares: the parent quantity, in **whole shares**, strictly
            positive.
        weight_units: dimensionless strictly-positive integer weights, one per
            bucket, at least one and no more than :data:`MAX_SLICE_COUNT`.
        minimum_shares: shares every bucket receives before weighting, in **whole
            shares**, at least :data:`MINIMUM_SLICE_SHARES_FLOOR`.

    Returns:
        Whole-share allocations in bucket order. Sums to exactly
        ``total_shares``; every entry is at least ``minimum_shares``.

    Raises:
        SliceScheduleError: if any argument is not a whole number in range, or if
            ``total_shares < len(weight_units) * minimum_shares`` — in which case
            the requested schedule cannot exist and the message says how many
            buckets would fit.
    """
    bucket_count = len(weight_units)
    _require(
        1 <= bucket_count <= MAX_SLICE_COUNT,
        f"weight_units must hold between 1 and {MAX_SLICE_COUNT} buckets, got {bucket_count}",
    )
    _require(
        _whole(total_shares) and total_shares > 0,
        f"total_shares must be a strictly positive whole number of shares (bool is refused), "
        f"got {total_shares!r}",
    )
    _require(
        _whole(minimum_shares) and minimum_shares >= MINIMUM_SLICE_SHARES_FLOOR,
        f"minimum_shares must be a whole number of at least {MINIMUM_SLICE_SHARES_FLOOR}, got "
        f"{minimum_shares!r}; a minimum of zero re-opens unbounded slicing",
    )
    for index, weight in enumerate(weight_units):
        _require(
            _whole(weight) and weight > 0,
            f"bucket {index} weight must be a strictly positive whole number, got {weight!r}",
        )
    _require(
        total_shares >= bucket_count * minimum_shares,
        f"{total_shares} shares cannot fill {bucket_count} buckets at a minimum of "
        f"{minimum_shares} shares each; at this minimum the parent supports at most "
        f"{total_shares // minimum_shares} bucket(s)",
    )

    discretionary = total_shares - bucket_count * minimum_shares
    total_weight = sum(weight_units)
    base = [(discretionary * weight) // total_weight for weight in weight_units]
    remainder = [(discretionary * weight) % total_weight for weight in weight_units]
    unassigned = discretionary - sum(base)
    ranked = sorted(range(bucket_count), key=lambda index: (-remainder[index], index))
    extra = set(ranked[:unassigned])
    return tuple(
        minimum_shares + base[index] + (1 if index in extra else 0) for index in range(bucket_count)
    )


def feasible_slice_count(
    *,
    quantity_shares: int,
    requested_count: int,
    minimum_slice_shares: int = ROUND_LOT_SHARES,
) -> int:
    """Return the number of slices a parent of this size can actually support.

    The smallest of three bounds, floored at one:

    - what the caller asked for;
    - :data:`MAX_SLICE_COUNT`, the schedule horizon (one regular session in
      :data:`BUCKET_MINUTES`-minute buckets);
    - ``quantity_shares // minimum_slice_shares``, the minimum-slice rule.

    The last two are the module's answer to "the cost model says slice
    infinitely". Neither is optimised and neither is negotiable through a
    configuration; see the module docstring.

    Args:
        quantity_shares: the parent quantity, in **whole shares**, strictly
            positive.
        requested_count: buckets the caller would like, at least 1.
        minimum_slice_shares: shares each slice must carry, at least
            :data:`MINIMUM_SLICE_SHARES_FLOOR`. Defaults to one round lot.

    Returns:
        A slice count between 1 and :data:`MAX_SLICE_COUNT` inclusive. Never
        zero: a parent too small to slice still has to be traded, as one order.

    Raises:
        SliceScheduleError: if any argument is not a whole number in range.
    """
    _require(
        _whole(quantity_shares) and quantity_shares > 0,
        f"quantity_shares must be a strictly positive whole number of shares, got "
        f"{quantity_shares!r}",
    )
    _require(
        _whole(requested_count) and requested_count >= 1,
        f"requested_count must be a whole number of at least 1, got {requested_count!r}",
    )
    _require(
        _whole(minimum_slice_shares) and minimum_slice_shares >= MINIMUM_SLICE_SHARES_FLOOR,
        f"minimum_slice_shares must be a whole number of at least "
        f"{MINIMUM_SLICE_SHARES_FLOOR}, got {minimum_slice_shares!r}",
    )
    supported = quantity_shares // minimum_slice_shares
    return max(1, min(requested_count, MAX_SLICE_COUNT, supported))


@dataclass(frozen=True, slots=True)
class SliceSchedule:
    """A parent order divided into ordered child orders, and the shape that divided it.

    The type carries the invariants rather than trusting the function that
    usually builds it, so a schedule assembled by any route — a test, a future
    caller, a deserialiser — either satisfies them or does not exist:

    - the children sum to **exactly** the parent quantity;
    - every child carries at least :attr:`minimum_slice_shares`;
    - the children are indexed ``0..n-1`` and all declare a ``slice_count`` of
      ``n``, which is what makes their idempotency keys distinct;
    - every child agrees with every other on instrument, side, pricing
      instruction, rebalance date and I2 stamp — a schedule is one decision;
    - the forecast covers exactly ``n`` buckets, is *for* this rebalance date, and
      consumed nothing dated on or after it (I1, restated here so it is a
      property of the schedule and not only of the forecast).

    Units: every quantity is in **whole shares**.

    Attributes:
        algorithm: which schedule shape was asked for.
        parent_quantity_shares: the parent's quantity, which the children sum to.
        minimum_slice_shares: the minimum actually applied. Equals the requested
            minimum, except for a one-slice schedule of a parent smaller than it:
            a single slice *is* the parent order, and there is nothing to
            constrain.
        forecast: the weighting the children were apportioned by, carrying its own
            provenance and its ``fitted`` flag.
        slices: the child orders, in bucket order.
    """

    algorithm: SliceAlgorithm
    parent_quantity_shares: int
    minimum_slice_shares: int
    forecast: VolumeForecast
    slices: tuple[OrderIntent, ...]

    def __post_init__(self) -> None:
        """Validate every invariant the class documents.

        Raises:
            SliceScheduleError: if the children do not sum to the parent, if any
                child is below the minimum, if the slice coordinates are not
                ``0..n-1`` over a ``slice_count`` of ``n``, if the children
                disagree about the order they implement, or if the forecast does
                not cover exactly these buckets for exactly this date.
            ForecastLookaheadError: if the forecast consumed a session on or
                after the rebalance date (I1).
        """
        children: object = self.slices
        _require(
            isinstance(children, tuple),
            f"slices must be a tuple, got {type(children).__name__}",
        )
        count = len(self.slices)
        _require(
            1 <= count <= MAX_SLICE_COUNT,
            f"a schedule must hold between 1 and {MAX_SLICE_COUNT} slices, got {count}",
        )
        _require(
            _whole(self.parent_quantity_shares) and self.parent_quantity_shares > 0,
            f"parent_quantity_shares must be a strictly positive whole number, got "
            f"{self.parent_quantity_shares!r}",
        )
        _require(
            _whole(self.minimum_slice_shares)
            and self.minimum_slice_shares >= MINIMUM_SLICE_SHARES_FLOOR,
            f"minimum_slice_shares must be a whole number of at least "
            f"{MINIMUM_SLICE_SHARES_FLOOR}, got {self.minimum_slice_shares!r}",
        )
        self._validate_quantities(count)
        self._validate_coordinates(count)
        self._validate_one_decision()
        self._validate_forecast(count)

    def _validate_quantities(self, count: int) -> None:
        """Check the exact-summation and minimum-slice properties.

        Args:
            count: number of slices.

        Raises:
            SliceScheduleError: if the sum is not exact or a slice is too small.
        """
        allocated = sum(child.quantity_shares for child in self.slices)
        _require(
            allocated == self.parent_quantity_shares,
            f"the {count} slices sum to {allocated} shares but the parent is "
            f"{self.parent_quantity_shares}; a schedule that does not sum exactly leaves "
            f"shares no order will ever trade, and reconciliation reports them as a break "
            f"days later",
        )
        for index, child in enumerate(self.slices):
            _require(
                child.quantity_shares >= self.minimum_slice_shares,
                f"slice {index} carries {child.quantity_shares} shares, below the minimum of "
                f"{self.minimum_slice_shares}",
            )

    def _validate_coordinates(self, count: int) -> None:
        """Check that the slice coordinates make every child's key distinct.

        Args:
            count: number of slices.

        Raises:
            SliceScheduleError: if any index or count is wrong for its position.
        """
        for index, child in enumerate(self.slices):
            _require(
                child.slice_index == index,
                f"slice at position {index} declares slice_index={child.slice_index}; the "
                f"coordinates are part of the idempotency preimage, so a wrong one either "
                f"collides with a sibling or names an order nobody planned",
            )
            _require(
                child.slice_count == count,
                f"slice {index} declares slice_count={child.slice_count} in a schedule of {count}",
            )

    def _validate_one_decision(self) -> None:
        """Check that every child implements the same parent order.

        Raises:
            SliceScheduleError: if the children disagree on instrument, side,
                pricing instruction, rebalance date or I2 stamp.
        """
        first = self.slices[0]
        for index, child in enumerate(self.slices[1:], start=1):
            mismatched = [
                name
                for name in (
                    "security_id",
                    "side",
                    "order_type",
                    "time_in_force",
                    "limit_price_usd",
                    "rebalance_date",
                    "stamp",
                )
                if getattr(child, name) != getattr(first, name)
            ]
            _require(
                not mismatched,
                f"slice {index} disagrees with slice 0 on {mismatched}; a schedule is one "
                f"decision divided over time, not a basket",
            )

    def _validate_forecast(self, count: int) -> None:
        """Check the forecast covers these buckets, for this date, without lookahead.

        Args:
            count: number of slices.

        Raises:
            SliceScheduleError: if the forecast's bucket count or ``as_of`` does
                not match the schedule.
            ForecastLookaheadError: if it consumed a session on or after the
                rebalance date. Checked **before** the ``as_of`` match, because a
                forecast whose observations reach past the trading day is an
                invariant violation whichever date it claims to be for — and
                because checking the match first would make this branch
                unreachable, which is a defence that only looks like one.
        """
        _require(
            self.forecast.bucket_count == count,
            f"the forecast covers {self.forecast.bucket_count} buckets but the schedule has "
            f"{count} slices",
        )
        _require_no_lookahead(self.forecast, self.rebalance_date)
        _require(
            self.forecast.as_of == self.rebalance_date,
            f"the forecast is for {self.forecast.as_of.isoformat()} but the schedule trades "
            f"on {self.rebalance_date.isoformat()}",
        )

    @property
    def slice_count(self) -> int:
        """Number of child orders in this schedule."""
        return len(self.slices)

    @property
    def rebalance_date(self) -> dt.date:
        """The single session every child trades in.

        A property rather than a field: it is read off the children, so it cannot
        disagree with them. It is also the structural reason a schedule cannot
        span sessions — there is one date and every child carries it.
        """
        return self.slices[0].rebalance_date

    @property
    def quantities(self) -> tuple[int, ...]:
        """Child quantities in bucket order, in **whole shares**."""
        return tuple(child.quantity_shares for child in self.slices)

    def idempotency_keys(self) -> tuple[str, ...]:
        """Return each child's content-derived idempotency key, in bucket order.

        P11.2's recipe with nothing added: the key already covers ``slice_index``
        and ``slice_count``, so siblings cannot collide, and two runs of one plan
        — same parent, same forecast, same I2 stamp — produce the same keys in
        the same order. The second run is then absorbed by the ``UNIQUE``
        constraint on ``execution_order.idempotency_key`` rather than doubling the
        position.

        Returns:
            One lowercase 64-character SHA-256 hex digest per slice.
        """
        return tuple(idempotency_key(child) for child in self.slices)

    def summary(self) -> dict[str, object]:
        """Return a JSON-safe description for operator display and run artifacts.

        Returns:
            Mapping of the algorithm, the quantities, the sum, the minimum
            applied, the rebalance date and the full forecast provenance —
            including whether the forecast was fitted, which is the fact a reader
            most needs and is least likely to go looking for.
        """
        return {
            "algorithm": self.algorithm.value,
            "slice_count": self.slice_count,
            "parent_quantity_shares": self.parent_quantity_shares,
            "quantities_shares": list(self.quantities),
            "allocated_shares": sum(self.quantities),
            "minimum_slice_shares": self.minimum_slice_shares,
            "rebalance_date": self.rebalance_date.isoformat(),
            "forecast": self.forecast.summary(),
            "units": "every quantity is a whole number of shares",
        }


def _child_intents(parent: OrderIntent, allocation: Sequence[int]) -> tuple[OrderIntent, ...]:
    """Build the child intents from a parent and an allocation.

    Every field except the quantity and the slice coordinates is copied from the
    parent, so the children are the same decision and their idempotency keys
    differ in exactly the two coordinates that distinguish them.

    Args:
        parent: the order being sliced.
        allocation: whole-share quantities in bucket order.

    Returns:
        The child intents, in bucket order.

    Raises:
        OrderValidationError: propagated from :class:`OrderIntent` if any
            allocation is not a tradeable quantity.
    """
    count = len(allocation)
    return tuple(
        replace(
            parent,
            quantity_shares=quantity,
            slice_index=index,
            slice_count=count,
        )
        for index, quantity in enumerate(allocation)
    )


def _plan(
    parent: OrderIntent,
    *,
    algorithm: SliceAlgorithm,
    forecast: VolumeForecast,
    minimum_slice_shares: int,
) -> SliceSchedule:
    """Apportion a parent against a forecast and assemble the schedule.

    The one place slicing actually happens; :func:`plan_twap` and
    :func:`plan_vwap` differ only in the forecast they arrive with.

    Args:
        parent: the unsliced order. Must itself be unsliced.
        algorithm: the label recorded on the schedule.
        forecast: the weighting, whose ``as_of`` must be the parent's rebalance
            date.
        minimum_slice_shares: requested minimum per slice.

    Returns:
        The assembled :class:`SliceSchedule`.

    Raises:
        SliceScheduleError: if the parent is already a slice, if the forecast is
            for another date, or if the quantity cannot fill the forecast's
            buckets at this minimum.
        ForecastLookaheadError: if the forecast consumed a session on or after
            the parent's rebalance date.
    """
    _require(
        parent.slice_count == 1 and parent.slice_index == 0,
        f"the parent is already slice {parent.slice_index} of {parent.slice_count}; slicing "
        f"a slice would need nested coordinates the order schema cannot express, and the "
        f"resulting keys would not identify anything a reader could reconstruct",
    )
    _require_no_lookahead(forecast, parent.rebalance_date)
    _require(
        forecast.as_of == parent.rebalance_date,
        f"the forecast is for {forecast.as_of.isoformat()} but the parent rebalances on "
        f"{parent.rebalance_date.isoformat()}; a volume shape for another session is not "
        f"evidence about this one",
    )
    quantity = parent.quantity_shares
    bucket_count = forecast.bucket_count
    minimum = minimum_slice_shares if bucket_count > 1 else min(minimum_slice_shares, quantity)
    allocation = apportion_shares(
        total_shares=quantity,
        weight_units=forecast.weight_units,
        minimum_shares=minimum,
    )
    return SliceSchedule(
        algorithm=algorithm,
        parent_quantity_shares=quantity,
        minimum_slice_shares=minimum,
        forecast=forecast,
        slices=_child_intents(parent, allocation),
    )


def plan_twap(
    parent: OrderIntent,
    *,
    bucket_count: int,
    minimum_slice_shares: int = ROUND_LOT_SHARES,
) -> SliceSchedule:
    """Split a parent order evenly across equal-duration buckets.

    TWAP is the degenerate VWAP: it is planned against
    :func:`uniform_volume_forecast`, which declares no shape at all, so there is
    no separate TWAP arithmetic that could disagree with the VWAP path. Every
    slice therefore receives the same quantity except for the remainder, which
    goes one share each to the earliest buckets (see :func:`apportion_shares`) —
    so the difference between the largest and smallest slice is never more than
    one share.

    Unlike :func:`plan_vwap` this **never refuses for being too ambitious**: a
    uniform forecast has no shape to preserve, so an infeasible ``bucket_count``
    is silently reduced by :func:`feasible_slice_count` rather than rejected. A
    caller that needs to know how many slices it got reads
    :attr:`SliceSchedule.slice_count`.

    Args:
        parent: the order to slice, itself unsliced. Its quantity is in **whole
            shares**.
        bucket_count: buckets the caller would like, at least 1. Reduced to what
            the horizon and the minimum-slice rule permit.
        minimum_slice_shares: shares each slice must carry, at least
            :data:`MINIMUM_SLICE_SHARES_FLOOR`. Defaults to one round lot
            (:data:`ROUND_LOT_SHARES`).

    Returns:
        A :class:`SliceSchedule` on :attr:`SliceAlgorithm.TWAP` whose children sum
        to exactly the parent quantity.

    Raises:
        SliceScheduleError: if the parent is already a slice, or an argument is
            not a whole number in range.
        VolumeForecastError: propagated from the uniform forecast.
    """
    _require(
        parent.slice_count == 1 and parent.slice_index == 0,
        f"the parent is already slice {parent.slice_index} of {parent.slice_count}",
    )
    count = feasible_slice_count(
        quantity_shares=parent.quantity_shares,
        requested_count=bucket_count,
        minimum_slice_shares=minimum_slice_shares,
    )
    return _plan(
        parent,
        algorithm=SliceAlgorithm.TWAP,
        forecast=uniform_volume_forecast(bucket_count=count, as_of=parent.rebalance_date),
        minimum_slice_shares=minimum_slice_shares,
    )


def plan_vwap(
    parent: OrderIntent,
    *,
    forecast: VolumeForecast,
    minimum_slice_shares: int = ROUND_LOT_SHARES,
) -> SliceSchedule:
    """Split a parent order in proportion to expected volume per bucket.

    The bucket count comes from the forecast, not from an argument, and this
    function **will not reshape a forecast to make it fit**. If the parent is too
    small to give every bucket its minimum, it raises and names the largest bucket
    count that would work, rather than truncating a thirteen-bucket U into three
    buckets — which is not a three-bucket U and would quietly become one.

    Whether this is a forecast or an assumption is the forecast's business and it
    says so: :attr:`VolumeForecast.fitted` and its ``basis_detail`` travel onto
    the returned schedule. Today nothing in the platform can produce a fitted one
    (B1), so every VWAP schedule this system builds is planned against a stated
    assumption, and the schedule carries that admission rather than a footnote.

    Args:
        parent: the order to slice, itself unsliced. Its quantity is in **whole
            shares**.
        forecast: the per-bucket weighting. Its ``as_of`` must equal the parent's
            ``rebalance_date``, and it must have consumed nothing dated on or
            after that (enforced at its own construction and again here, I1).
        minimum_slice_shares: shares each slice must carry, at least
            :data:`MINIMUM_SLICE_SHARES_FLOOR`. Defaults to one round lot.

    Returns:
        A :class:`SliceSchedule` on :attr:`SliceAlgorithm.VWAP` whose children sum
        to exactly the parent quantity and are monotone in the forecast weights.

    Raises:
        SliceScheduleError: if the parent is already a slice, if the forecast is
            for another session, or if the parent cannot fill the forecast's
            buckets at this minimum.
        ForecastLookaheadError: if the forecast consumed a session on or after
            the rebalance date.
    """
    return _plan(
        parent,
        algorithm=SliceAlgorithm.VWAP,
        forecast=forecast,
        minimum_slice_shares=minimum_slice_shares,
    )


@dataclass(frozen=True, slots=True)
class ScheduleCost:
    """The modelled cost of executing a whole schedule, in **US dollars**.

    Produced by :func:`modelled_schedule_cost` from
    :func:`~backend.costs.model.estimate_trade_cost` — the platform's one cost
    model, not a second one written for this module (I4). The calibration flags
    come straight through, so a number derived from this object can state that its
    costs are assumptions, and the forecast's provenance rides along for the same
    reason.

    Attributes:
        slice_count: number of child orders costed.
        notional_usd: total traded notional across the schedule, in US dollars.
            Invariant in the slice count — slicing moves the same shares.
        impact_usd: total square-root market impact, in US dollars. The only
            component that responds to slicing, and it falls as ``1/sqrt(n)``.
        spread_and_commission_usd: total half-spread plus commission, in US
            dollars. **Invariant in the slice count**, because the model quotes
            both as flat rates on notional and has no per-order term — the
            demonstration that the model cannot see what slicing costs.
        total_usd: the sum of the two, in US dollars.
        uncalibrated: from :attr:`~backend.costs.model.CostModelParams.uncalibrated`.
        calibration_basis: the cost parameters' statement of provenance.
        forecast_is_fitted: :attr:`VolumeForecast.fitted` of the schedule's
            forecast.
        forecast_basis_detail: that forecast's provenance prose.
    """

    slice_count: int
    notional_usd: float
    impact_usd: float
    spread_and_commission_usd: float
    total_usd: float
    uncalibrated: bool
    calibration_basis: str
    forecast_is_fitted: bool
    forecast_basis_detail: str

    def summary(self) -> dict[str, object]:
        """Return a JSON-safe breakdown for operator display and run artifacts.

        Returns:
            Mapping of every component in US dollars, both provenance flags, and
            an ``omits`` key naming what the model does not charge for. The
            omissions are in the summary rather than only in the docstring
            because a reader comparing two schedules' costs is exactly the reader
            about to draw a conclusion the model cannot support.
        """
        return {
            "slice_count": self.slice_count,
            "notional_usd": self.notional_usd,
            "impact_usd": self.impact_usd,
            "spread_and_commission_usd": self.spread_and_commission_usd,
            "total_usd": self.total_usd,
            "uncalibrated": self.uncalibrated,
            "calibration_basis": self.calibration_basis,
            "forecast_is_fitted": self.forecast_is_fitted,
            "forecast_basis_detail": self.forecast_basis_detail,
            "omits": (
                "no per-order fixed cost (commission is a flat rate on notional, so this "
                "figure falls without limit as the schedule is cut finer); no timing or "
                "volatility risk from stretching execution over the session; no cross-impact "
                "between names; no borrow, which is a holding cost identical for every "
                "schedule of one parent and therefore not a function of the schedule"
            ),
            "units": "US dollars",
        }


def modelled_schedule_cost(
    schedule: SliceSchedule,
    *,
    reference_price_usd: float,
    adv_usd: float,
    daily_volatility_bps: float | None = None,
    params: CostModelParams = UNCALIBRATED_DEFAULTS,
) -> ScheduleCost:
    """Cost a schedule through the platform's cost model, slice by slice.

    Each child is costed on its own notional and therefore its own participation
    rate, which is the entire mechanism by which slicing changes a modelled cost:
    :mod:`backend.costs.impact` is square-root in participation, so ``n`` equal
    slices each cost ``1/sqrt(n)`` times the parent's impact **in basis points**
    on ``1/n`` of the notional, and the dollar total falls as ``1/sqrt(n)``.

    **This is a modelled number and it is not net of everything.** Read
    :meth:`ScheduleCost.summary`'s ``omits`` before comparing two schedules: the
    model has no per-order cost and no timing-risk term, so it prefers more
    slices unconditionally and would prefer infinitely many. That preference is an
    artifact of the missing terms, which is why the slice-count bound in this
    module is a stated constant rather than the output of a search
    (:data:`MAX_SLICE_COUNT`, :data:`ROUND_LOT_SHARES`).

    Borrow is deliberately excluded: it accrues on a held short position over a
    holding period, is identical for every schedule of one parent, and is
    therefore not a function of the schedule. Including it would inflate every
    figure here by a constant and invite the comparison to be read as if it
    covered financing.

    Args:
        schedule: the schedule to cost.
        reference_price_usd: price used to turn share counts into notional, in
            **US dollars per share**, strictly positive. An arrival or decision
            price — not a fill price, and nothing here is a realised cost.
        adv_usd: average daily dollar volume for the name, in **US dollars**,
            strictly positive.
        daily_volatility_bps: the name's daily return standard deviation in
            **basis points** (``200.0`` is 2% per day). ``None`` falls back to
            the parameter set's default.
        params: cost parameters. Defaults to
            :data:`~backend.costs.model.UNCALIBRATED_DEFAULTS`, whose
            ``uncalibrated`` flag propagates into the result.

    Returns:
        A :class:`ScheduleCost` in **US dollars**, carrying both the cost
        parameters' calibration status and the forecast's provenance.

    Raises:
        SliceScheduleError: if ``reference_price_usd`` is not strictly positive.
        CostParameterError: propagated from the cost model for a non-finite
            price, a non-positive ``adv_usd``, or a negative volatility.
    """
    _require(
        reference_price_usd > 0,
        f"reference_price_usd must be strictly positive, got {reference_price_usd!r}; a "
        f"zero price would report every schedule as free",
    )
    costs = tuple(
        estimate_trade_cost(
            CostedOrder(
                side=child.side,
                notional_usd=child.quantity_shares * reference_price_usd,
                adv_usd=adv_usd,
                daily_volatility_bps=daily_volatility_bps,
            ),
            params,
        )
        for child in schedule.slices
    )
    impact_usd = sum(bps_of_notional_to_usd(cost.impact_bps, cost.notional_usd) for cost in costs)
    spread_and_commission_usd = sum(
        bps_of_notional_to_usd(cost.half_spread_bps + cost.commission_bps, cost.notional_usd)
        for cost in costs
    )
    return ScheduleCost(
        slice_count=schedule.slice_count,
        notional_usd=sum(cost.notional_usd for cost in costs),
        impact_usd=impact_usd,
        spread_and_commission_usd=spread_and_commission_usd,
        total_usd=sum(cost.total_usd for cost in costs),
        uncalibrated=params.uncalibrated,
        calibration_basis=params.calibration_basis,
        forecast_is_fitted=schedule.forecast.fitted,
        forecast_basis_detail=schedule.forecast.basis_detail,
    )
