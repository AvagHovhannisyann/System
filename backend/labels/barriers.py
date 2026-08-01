"""Triple-barrier labelling (P6.1, directive §5 Phase 6).

An event at bar ``t0`` is labelled by which of three barriers the path touches
first:

- an **upper** barrier at ``+u`` cumulative log return from the event's close,
- a **lower** barrier at ``-d`` cumulative log return, and
- a **vertical** barrier at bar ``t0 + horizon`` — a deadline, not a price.

The horizontal barriers are sized by **trailing** volatility, so a quiet stock
and a volatile stock are asked the same question in their own units rather than
in absolute percent.

--------------------------------------------------------------------------
Units, conventions, and the exact arithmetic
--------------------------------------------------------------------------

**Everything is in natural-log return space, as a fraction.** ``0.02`` is a 2%
move, near enough. Cumulative return from the event to bar ``s`` is
``ln(P_s) - ln(P_t0)``. See :mod:`backend.labels.volatility` for why log space
rather than simple returns.

**Barrier width** (:func:`barrier_log_return_width`)::

    width = multiple * sigma * sqrt(horizon)

with ``sigma`` the trailing per-bar log-return standard deviation at ``t0``
(a fraction) and ``horizon`` the vertical barrier in bars. The upper barrier
sits at ``+upper_multiple * sigma * sqrt(horizon)``, the lower at
``-lower_multiple * sigma * sqrt(horizon)``. Barrier distance is therefore
**linear in trailing volatility**: double the volatility, double the distance.

*Why the* ``sqrt(horizon)`` *factor.* Under a driftless random walk with i.i.d.
per-bar log returns of standard deviation ``sigma``, the cumulative return over
``H`` bars has standard deviation ``sigma * sqrt(H)``. Scaling the barrier the
same way makes ``multiple`` the barrier measured in units of *the horizon's own*
return distribution, so the three-class balance is roughly comparable across the
5-day, 21-day and 63-day horizons this project uses. The textbook alternative
(a barrier at a fixed multiple of *daily* volatility, independent of horizon) is
touched almost surely well before a 63-day deadline, which collapses the
vertical barrier into an empty class and makes the 63-day label a slower copy of
the 5-day one. Nothing is lost by this choice: passing
``upper_multiple = k / sqrt(H)`` reproduces the fixed-daily convention exactly.

**Entry, scan window, and the vertical barrier.** The event is stamped at the
**close of bar** ``t0``. The path scanned is bars ``t0+1 … t0+horizon``
inclusive; bar ``t0`` itself is not scanned, since its cumulative return is zero
by construction. The vertical barrier is the close of bar ``t0 + horizon``.
A label therefore consumes information from bars ``t0+1 … resolution_index``
and from the ``volatility_window`` returns ending at ``t0`` — and from nothing
else, which is the property :mod:`backend.tests.labels.test_lookahead` asserts
by recomputing on truncated series.

**Touching is inclusive.** ``cumulative >= upper`` and ``cumulative <= lower``.
Reaching the barrier exactly counts as touching it.

**With ``high``/``low``, a barrier is touched intrabar; without them, only at a
close.** Close-only labelling misses a barrier that was pierced during the bar
and given back by the close. That is a *biased*, not merely noisier, estimator:
it systematically under-counts touches and over-counts vertical-barrier
outcomes. It is the honest default when only closes exist (as for a residualized
return path, where an intrabar high has no meaning — see
:mod:`backend.labels.residualize`), and the bias is stated here rather than
discovered later.

--------------------------------------------------------------------------
The two decisions that had to be made deliberately
--------------------------------------------------------------------------

**1. The vertical barrier is its own class, not the sign of the return.**
When neither horizontal barrier is touched, the outcome is
:attr:`TripleBarrierOutcome.VERTICAL` (numeric value ``0``) — *not*
``sign(realized_return)``. The common alternative labels a ``+0.05 * sigma``
drift identically to a ``+2 * sigma`` breakout, which injects the maximum amount
of noise at exactly the point where the price path said the least. Whether such
an observation should be dropped, treated as a neutral third class, or used as a
meta-labelling negative is a modelling decision for Phase 8, and it needs the
three cases to be distinguishable to make it. No information is destroyed:
:attr:`LabelSet.realized_log_return` carries the signed return, so a caller who
wants the sign convention can take it in one line — but they take it knowingly.

**2. Both barriers pierced inside one bar is flagged, not tie-broken.**
If a bar's high clears the upper barrier and the same bar's low clears the
lower barrier, daily data does not say which came first. The default
:attr:`IntrabarPolicy.FLAG_AMBIGUOUS` records
:attr:`TripleBarrierOutcome.AMBIGUOUS` and leaves the choice to the caller;
:meth:`LabelSet.class_labels` refuses to hand out a numeric class while any
remain. The alternative :attr:`IntrabarPolicy.LOWER_FIRST` resolves such bars to
the lower barrier — the pessimistic assumption **for a long position**, and
therefore *not* pessimistic for a short, which is why it is opt-in and named for
what it does rather than "conservative". Both are legitimate; inventing an
ordering from the close's direction is not, because it fabricates the one fact
the data is missing. The count of ambiguous bars is itself a diagnostic: a large
one means the barriers are narrow relative to typical bar range, and the label
design — not the tie-break — is what needs fixing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

import numpy as np

from backend.labels._arrays import (
    as_float_1d,
    as_index_1d,
    require_positive,
    require_same_length,
)
from backend.labels.errors import (
    AmbiguousLabelError,
    DegenerateVolatilityError,
    InsufficientHistoryError,
    LabelConfigurationError,
    LabelInputError,
)
from backend.labels.volatility import daily_log_returns, trailing_volatility

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from backend.labels._arrays import FloatArray, IntArray

    type _CumulativePathFn = Callable[[int, int], tuple[FloatArray, FloatArray, FloatArray]]
    """``(event_bar, deadline_bar) -> (high, low, close)`` cumulative log-return paths."""

__all__ = [
    "BarrierSpec",
    "IntrabarPolicy",
    "LabelBasis",
    "LabelSet",
    "TripleBarrierOutcome",
    "barrier_log_return_width",
    "resolve_barrier_touch",
    "triple_barrier_labels",
    "usable_event_indices",
]


class TripleBarrierOutcome(IntEnum):
    """Which barrier resolved the label.

    The three resolvable values are signed so they can be used directly as a
    classification target: ``-1`` down, ``0`` deadline, ``+1`` up.

    Attributes:
        LOWER_FIRST: the lower barrier was touched first (value ``-1``).
        VERTICAL: the deadline arrived with neither horizontal barrier touched
            (value ``0``). This is *not* "the return was zero" and it is *not*
            the sign of the return at the deadline; see the module docstring.
        UPPER_FIRST: the upper barrier was touched first (value ``+1``).
        AMBIGUOUS: both barriers were pierced within one bar and the data
            cannot order them (value ``-128``). The value is deliberately
            outside any sane class encoding: an observation that leaks into a
            model as a numeric target produces visibly insane arithmetic rather
            than quietly joining the ``VERTICAL`` class.
    """

    LOWER_FIRST = -1
    VERTICAL = 0
    UPPER_FIRST = 1
    AMBIGUOUS = -128


class IntrabarPolicy(Enum):
    """How to treat a bar that pierces both barriers.

    Attributes:
        FLAG_AMBIGUOUS: record :attr:`TripleBarrierOutcome.AMBIGUOUS` and let
            the caller decide. The default.
        LOWER_FIRST: assume the lower barrier was reached first. Pessimistic for
            a long position and optimistic for a short; opt-in, and named for
            the assumption it makes rather than for a virtue it only sometimes
            has.
    """

    FLAG_AMBIGUOUS = "flag_ambiguous"
    LOWER_FIRST = "lower_first"


class LabelBasis(Enum):
    """Which return series the barriers were applied to.

    Attributes:
        PRICE: raw close-to-close log returns of the security itself.
        RESIDUAL: log returns residualized against market and sector factors —
            see :mod:`backend.labels.residualize`. Always close-only, because a
            residual has no intrabar high or low.
    """

    PRICE = "price"
    RESIDUAL = "residual"


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    """The configuration that turns a volatility into three barriers.

    Frozen and carried on every :class:`LabelSet` it produces, so a label set is
    self-describing and a result is regenerable from its configuration
    (invariant I2).

    Attributes:
        horizon: vertical barrier, in bars after the event. Must be at least 1.
            The project's horizons are 5, 21 and 63 trading days.
        upper_multiple: upper barrier in units of ``sigma * sqrt(horizon)``
            (dimensionless). Strictly positive.
        lower_multiple: lower barrier in the same units, given as a **positive**
            number; the barrier itself is placed at minus this distance.
            Strictly positive. Asymmetric barriers are permitted and are how a
            caller expresses an asymmetric payoff — nothing here assumes
            ``upper_multiple == lower_multiple``.
        volatility_window: trailing bars used to estimate the volatility that
            sizes the barrier (count). At least 2.
        intrabar_policy: how to treat a bar piercing both barriers. Only
            reachable when ``high``/``low`` are supplied.
    """

    horizon: int
    upper_multiple: float = 1.0
    lower_multiple: float = 1.0
    volatility_window: int = 20
    intrabar_policy: IntrabarPolicy = IntrabarPolicy.FLAG_AMBIGUOUS

    def __post_init__(self) -> None:
        """Validate the specification at construction.

        Raises:
            LabelConfigurationError: if the horizon is below 1, either multiple
                is not finite and strictly positive, or the volatility window is
                below 2.
        """
        if self.horizon < 1:
            msg = (
                f"horizon must be >= 1 bar; got {self.horizon}. A horizon of 0 puts the "
                f"vertical barrier on the event bar itself, where the cumulative return "
                f"is zero by construction."
            )
            raise LabelConfigurationError(msg)
        for name, multiple in (
            ("upper_multiple", self.upper_multiple),
            ("lower_multiple", self.lower_multiple),
        ):
            if not math.isfinite(multiple) or multiple <= 0.0:
                msg = (
                    f"{name} must be finite and strictly positive (it is a barrier width "
                    f"in units of sigma*sqrt(horizon)); got {multiple!r}. A zero-width "
                    f"barrier is touched by construction on the first bar."
                )
                raise LabelConfigurationError(msg)
        if self.volatility_window < 2:
            msg = (
                f"volatility_window must be >= 2 bars; got {self.volatility_window}. "
                f"A sample standard deviation with ddof=1 needs two observations."
            )
            raise LabelConfigurationError(msg)

    def barrier_log_returns(self, volatility: float) -> tuple[float, float]:
        """Place the two horizontal barriers for one event.

        Args:
            volatility: trailing per-bar log-return standard deviation at the
                event (a fraction; 0.02 means 2% per bar). Must be finite and
                strictly positive — the caller is expected to have rejected
                degenerate events already.

        Returns:
            ``(upper, lower)`` as **signed** cumulative log returns from the
            event's close: ``upper`` is positive, ``lower`` is negative. Both
            are fractions.
        """
        upper = barrier_log_return_width(
            volatility, horizon=self.horizon, multiple=self.upper_multiple
        )
        lower = barrier_log_return_width(
            volatility, horizon=self.horizon, multiple=self.lower_multiple
        )
        return upper, -lower


def barrier_log_return_width(volatility: float, *, horizon: int, multiple: float) -> float:
    """Return the barrier's distance from the event, in cumulative log return.

    The one place the barrier-sizing formula is written down. Everything else in
    the package calls this, so there is a single definition to test and a single
    place a change would have to be made.

    Args:
        volatility: trailing per-bar log-return standard deviation (a fraction
            per bar, not annualized; 0.02 means 2% per bar).
        horizon: bars to the vertical barrier (count).
        multiple: barrier width in units of ``sigma * sqrt(horizon)``
            (dimensionless).

    Returns:
        The barrier's distance from the event as a **non-negative** cumulative
        log return (a fraction). The caller applies the sign. Linear in
        ``volatility``: doubling the volatility doubles the distance.
    """
    return multiple * volatility * math.sqrt(horizon)


def resolve_barrier_touch(
    cumulative_high: FloatArray,
    cumulative_low: FloatArray,
    *,
    upper_log_return: float,
    lower_log_return: float,
    policy: IntrabarPolicy,
) -> tuple[TripleBarrierOutcome, int]:
    """Decide which barrier a single event's forward path touches first.

    The core rule of the package, exposed so it can be tested on its own rather
    than only through the series-level entry points.

    Args:
        cumulative_high: for each forward bar ``t0+1 … t0+horizon`` in order,
            the highest cumulative log return reached at that bar (the bar's
            high for a price path; the bar's close when only closes are
            available). Fractions.
        cumulative_low: the same, for the lowest cumulative log return reached.
            Same length as ``cumulative_high``.
        upper_log_return: the upper barrier as a signed cumulative log return
            (positive).
        lower_log_return: the lower barrier as a signed cumulative log return
            (negative).
        policy: how to resolve a bar that pierces both barriers.

    Returns:
        ``(outcome, offset)``. ``offset`` is a 0-based position into the two
        input arrays, so the resolving bar is ``t0 + 1 + offset``. For
        :attr:`TripleBarrierOutcome.VERTICAL` the offset is the last bar (the
        deadline). For :attr:`TripleBarrierOutcome.AMBIGUOUS` it is the bar that
        pierced both.
    """
    touches_upper = cumulative_high >= upper_log_return
    touches_lower = cumulative_low <= lower_log_return

    any_upper = bool(touches_upper.any())
    any_lower = bool(touches_lower.any())
    if not any_upper and not any_lower:
        return TripleBarrierOutcome.VERTICAL, cumulative_high.shape[0] - 1

    horizon_end = cumulative_high.shape[0]
    first_upper = int(np.argmax(touches_upper)) if any_upper else horizon_end
    first_lower = int(np.argmax(touches_lower)) if any_lower else horizon_end

    if first_upper < first_lower:
        return TripleBarrierOutcome.UPPER_FIRST, first_upper
    if first_lower < first_upper:
        return TripleBarrierOutcome.LOWER_FIRST, first_lower
    # Same bar pierced both. Which came first is not in the data.
    if policy is IntrabarPolicy.LOWER_FIRST:
        return TripleBarrierOutcome.LOWER_FIRST, first_lower
    return TripleBarrierOutcome.AMBIGUOUS, first_upper


@dataclass(frozen=True, slots=True)
class LabelSet:
    """Triple-barrier labels for a set of events on one security.

    Every array is parallel to :attr:`event_index` and in the order the caller
    supplied its events, so results line up with whatever the caller carries
    alongside. Nothing is sorted or de-duplicated.

    Attributes:
        event_index: bar position of each event in the input series
            (dimensionless index). The event is stamped at that bar's **close**.
        outcome: :class:`TripleBarrierOutcome` value per event, as ``int8``.
        resolution_index: bar position at which the label resolved
            (dimensionless index). Always in ``[event_index + 1, event_index +
            horizon]``. For :attr:`TripleBarrierOutcome.VERTICAL` it is the
            deadline bar; for :attr:`TripleBarrierOutcome.AMBIGUOUS` it is the
            bar that pierced both barriers.
        realized_log_return: cumulative log return from the event's close to the
            **close** of the resolution bar (a fraction). Note this is a
            close-to-close return and is *not* the barrier level: a barrier
            touched intrabar can be given back by the close, so
            ``|realized_log_return|`` may be smaller than the barrier it
            triggered. Use the barrier fields, not this one, to model execution
            at the barrier.
        upper_barrier_log_return: the upper barrier that was applied, as a
            signed cumulative log return (positive fraction).
        lower_barrier_log_return: the lower barrier, as a signed cumulative log
            return (negative fraction).
        trailing_volatility: the per-bar log-return standard deviation used to
            size the barriers (a fraction per bar, not annualized). For
            :attr:`LabelBasis.RESIDUAL` this is the trailing *residual*
            volatility.
        spec: the configuration used, carried for reproducibility (I2).
        basis: whether the barriers were applied to raw or residualized returns.
    """

    event_index: IntArray
    outcome: npt.NDArray[np.int8]
    resolution_index: IntArray
    realized_log_return: FloatArray
    upper_barrier_log_return: FloatArray
    lower_barrier_log_return: FloatArray
    trailing_volatility: FloatArray
    spec: BarrierSpec
    basis: LabelBasis

    @property
    def n_labels(self) -> int:
        """Number of labelled events (count)."""
        return int(self.event_index.shape[0])

    @property
    def first_information_bar(self) -> IntArray:
        """First bar whose return the label consumes: ``event_index + 1``.

        The event is stamped at the close of ``event_index``, so the earliest
        bar the label can depend on is the next one. This, together with
        :attr:`resolution_index`, is the label's information span — the interval
        that :mod:`backend.labels.uniqueness` overlaps to compute sample weights
        and that Phase 8's purged cross-validation purges against.
        """
        return np.asarray(self.event_index + 1, dtype=np.intp)

    @property
    def is_ambiguous(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of events whose bar pierced both barriers at once."""
        return np.asarray(self.outcome == int(TripleBarrierOutcome.AMBIGUOUS), dtype=np.bool_)

    @property
    def n_ambiguous(self) -> int:
        """Number of unresolvable events (count)."""
        return int(np.count_nonzero(self.is_ambiguous))

    def outcome_counts(self) -> dict[TripleBarrierOutcome, int]:
        """Count events by outcome.

        Returns:
            A mapping with an entry for every member of
            :class:`TripleBarrierOutcome`, including zeros. The zeros are
            deliberate: an empty class is a finding (barriers too wide, or too
            narrow) and a caller iterating the mapping should see it rather than
            a missing key.
        """
        return {
            outcome: int(np.count_nonzero(self.outcome == int(outcome)))
            for outcome in TripleBarrierOutcome
        }

    def drop_ambiguous(self) -> LabelSet:
        """Return a copy without the events that pierced both barriers.

        Returns:
            A new :class:`LabelSet` over the resolvable events only, preserving
            order. Returns an equivalent copy when there are none.
        """
        keep = ~self.is_ambiguous
        return replace(
            self,
            event_index=self.event_index[keep],
            outcome=self.outcome[keep],
            resolution_index=self.resolution_index[keep],
            realized_log_return=self.realized_log_return[keep],
            upper_barrier_log_return=self.upper_barrier_log_return[keep],
            lower_barrier_log_return=self.lower_barrier_log_return[keep],
            trailing_volatility=self.trailing_volatility[keep],
        )

    def class_labels(self) -> npt.NDArray[np.int8]:
        """Return the labels as a ``{-1, 0, +1}`` classification target.

        Returns:
            An ``int8`` array parallel to :attr:`event_index`: ``-1`` lower
            barrier, ``0`` vertical barrier, ``+1`` upper barrier.

        Raises:
            AmbiguousLabelError: if any event is unresolvable. Ambiguous events
                are not folded into the ``0`` class, because "we cannot tell"
                and "neither barrier was touched" are different facts and
                merging them would put fabricated observations into the
                deadline class. Call :meth:`drop_ambiguous` first, or choose an
                explicit :class:`IntrabarPolicy`.
        """
        n_ambiguous = self.n_ambiguous
        if n_ambiguous:
            raise AmbiguousLabelError(n_ambiguous=n_ambiguous, n_labels=self.n_labels)
        # A copy, not a view: the label set is frozen, and a caller who
        # in-place-shuffles its training target must not silently rewrite the
        # labels it was derived from.
        return np.array(self.outcome, dtype=np.int8, copy=True)


def usable_event_indices(
    n_bars: int, spec: BarrierSpec, *, min_trailing_bars: int | None = None
) -> IntArray:
    """Return every bar index that can carry a label under ``spec``.

    A label needs ``min_trailing_bars`` bars of history before the event to
    estimate its volatility, and ``spec.horizon`` bars after it to reach the
    vertical barrier. Events failing either requirement are refused by
    :func:`triple_barrier_labels` rather than labelled from a shortened window,
    so this function is the intended way to pick events.

    Args:
        n_bars: length of the price series (count of bars).
        spec: the barrier configuration whose horizon and volatility window
            define the requirements.
        min_trailing_bars: override for the trailing requirement (count of
            bars). Defaults to ``spec.volatility_window``. Residualized
            labelling passes its longer estimation window here.

    Returns:
        Ascending bar indices (dimensionless), possibly empty when the series
        is too short to carry any label.

    Raises:
        LabelConfigurationError: if ``min_trailing_bars`` is negative.
    """
    trailing = spec.volatility_window if min_trailing_bars is None else min_trailing_bars
    if trailing < 0:
        msg = f"min_trailing_bars must be >= 0; got {trailing}"
        raise LabelConfigurationError(msg)
    last_usable = n_bars - spec.horizon
    if last_usable <= trailing:
        return np.empty(0, dtype=np.intp)
    return np.arange(trailing, last_usable, dtype=np.intp)


def triple_barrier_labels(
    close: npt.ArrayLike,
    event_index: npt.ArrayLike,
    spec: BarrierSpec,
    *,
    high: npt.ArrayLike | None = None,
    low: npt.ArrayLike | None = None,
    volatility: npt.ArrayLike | None = None,
) -> LabelSet:
    """Label events on one security's price path with the triple-barrier method.

    Args:
        close: close prices, one per bar, finite and strictly positive.
        event_index: bar positions to label (dimensionless indices). Each event
            is stamped at that bar's close. Order is preserved; duplicates are
            permitted (they produce duplicate labels, which
            :mod:`backend.labels.uniqueness` will then correctly treat as fully
            overlapping).
        spec: the barrier configuration.
        high: optional per-bar high prices. When supplied together with ``low``,
            a barrier counts as touched if the bar's high or low reached it,
            even if the close did not. Without them, only closing breaches are
            detected — a systematic under-count of touches, documented in the
            module docstring.
        low: optional per-bar low prices. Must be supplied with ``high``.
        volatility: optional per-bar trailing volatility (per-bar log-return
            standard deviation, a fraction), aligned to ``close``. Defaults to
            :func:`~backend.labels.volatility.trailing_volatility` of
            ``close``'s log returns over ``spec.volatility_window``. Supply it
            only to substitute a different *trailing* estimator; supplying a
            forward-looking one silently destroys the whole construction.

    Returns:
        A :class:`LabelSet` with :attr:`LabelBasis.PRICE` basis, parallel to
        ``event_index``.

    Raises:
        LabelInputError: if the series are malformed — non-positive prices,
            mismatched lengths, only one of ``high``/``low`` supplied, a bar
            where ``high < low``, or an event index outside the series.
        InsufficientHistoryError: if any event lacks the trailing bars its
            volatility estimate needs or the forward bars its horizon needs.
        DegenerateVolatilityError: if any event's trailing volatility is
            non-finite or not strictly positive.
    """
    prices = as_float_1d(close, name="close")
    require_positive(prices, name="close")
    n_bars = prices.shape[0]

    supplied_high, supplied_low = _prepare_intrabar_logs(high=high, low=low, close=prices)
    log_close = np.log(prices)
    # Without intrabar extremes, a "high" and a "low" are both just the close:
    # only closing breaches are detectable. See the module docstring.
    log_high = log_close if supplied_high is None else supplied_high
    log_low = log_close if supplied_low is None else supplied_low

    sigma = _resolve_volatility(volatility, prices=prices, spec=spec)
    require_same_length({"close": prices, "volatility": sigma})

    events = as_index_1d(event_index, name="event_index")
    _require_events_in_range(events, n_bars=n_bars)

    def cumulative(t0: int, stop: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Cumulative log return from the event's close over bars ``t0+1 … stop``."""
        anchor = log_close[t0]
        return (
            log_high[t0 + 1 : stop + 1] - anchor,
            log_low[t0 + 1 : stop + 1] - anchor,
            log_close[t0 + 1 : stop + 1] - anchor,
        )

    return _label_events(
        events=events,
        n_bars=n_bars,
        spec=spec,
        sigma=sigma,
        min_trailing_bars=spec.volatility_window,
        cumulative=cumulative,
        basis=LabelBasis.PRICE,
    )


def _prepare_intrabar_logs(
    *, high: npt.ArrayLike | None, low: npt.ArrayLike | None, close: FloatArray
) -> tuple[FloatArray | None, FloatArray | None]:
    """Validate and log-transform the optional intrabar extremes.

    Args:
        high: candidate high prices, or ``None``.
        low: candidate low prices, or ``None``.
        close: the already-validated close prices, for the bracketing check.

    Returns:
        ``(log_high, log_low)``, or ``(None, None)`` when neither was supplied.

    Raises:
        LabelInputError: if exactly one of the two is supplied, if either is
            non-positive or the wrong length, if any bar has ``high < low``, or
            if any bar's close falls outside ``[low, high]``.
    """
    if high is None and low is None:
        return None, None
    if high is None or low is None:
        msg = (
            "high and low must be supplied together: with only one of them, a barrier "
            "would be detected intrabar in one direction and only at the close in the "
            "other, which biases the label towards that direction."
        )
        raise LabelInputError(msg)

    highs = as_float_1d(high, name="high")
    lows = as_float_1d(low, name="low")
    require_positive(highs, name="high")
    require_positive(lows, name="low")
    require_same_length({"close": close, "high": highs, "low": lows})

    inverted = np.flatnonzero(highs < lows)
    if inverted.size:
        first = int(inverted[0])
        msg = (
            f"high must be >= low at every bar; {inverted.size} bar(s) violate this, "
            f"first at bar {first} (high={highs[first]!r}, low={lows[first]!r})"
        )
        raise LabelInputError(msg)

    outside = np.flatnonzero((close > highs) | (close < lows))
    if outside.size:
        first = int(outside[0])
        msg = (
            f"close must lie within [low, high] at every bar; {outside.size} bar(s) "
            f"violate this, first at bar {first} (low={lows[first]!r}, "
            f"close={close[first]!r}, high={highs[first]!r}). A close outside its own "
            f"bar's range means the three series are misaligned or differently adjusted, "
            f"which would place barriers against one series and measure returns against "
            f"another."
        )
        raise LabelInputError(msg)
    return np.log(highs), np.log(lows)


def _resolve_volatility(
    volatility: npt.ArrayLike | None, *, prices: FloatArray, spec: BarrierSpec
) -> FloatArray:
    """Return the trailing volatility series, computing it if not supplied.

    Args:
        volatility: caller-supplied series, or ``None``.
        prices: validated close prices.
        spec: the barrier configuration, for its volatility window.

    Returns:
        Per-bar trailing volatility (per-bar log-return standard deviation, a
        fraction), aligned to ``prices``.
    """
    if volatility is None:
        return trailing_volatility(daily_log_returns(prices), window=spec.volatility_window)
    return as_float_1d(volatility, name="volatility")


def _require_events_in_range(events: IntArray, *, n_bars: int) -> None:
    """Check that every event index addresses a bar that exists.

    Args:
        events: event bar positions.
        n_bars: length of the series (count).

    Raises:
        LabelInputError: if any index is at or beyond ``n_bars``.
    """
    if events.size and int(events.max()) >= n_bars:
        offending = int(events.max())
        msg = (
            f"event_index contains bar {offending} but the series has {n_bars} bars "
            f"(valid indices are 0..{n_bars - 1})"
        )
        raise LabelInputError(msg)


def _label_events(
    *,
    events: IntArray,
    n_bars: int,
    spec: BarrierSpec,
    sigma: FloatArray,
    min_trailing_bars: int,
    cumulative: _CumulativePathFn,
    basis: LabelBasis,
) -> LabelSet:
    """Resolve every event against its barriers and assemble the label set.

    The loop is per-event on purpose. Each event's window is a different slice
    of the series, and for residualized labelling each event has its own frozen
    regression coefficients; a fused array formulation would be faster and
    considerably harder to read, and this is a component where a silent error
    invalidates everything downstream (directive §0.3). Cost is
    ``O(n_events * horizon)``.

    Args:
        events: event bar positions, in the caller's order.
        n_bars: length of the underlying series (count).
        spec: the barrier configuration.
        sigma: per-bar trailing volatility aligned to the series (a fraction).
        min_trailing_bars: bars of history each event requires before it
            (count), for the error message when one is missing.
        cumulative: callable ``(t0, stop) -> (high, low, close)`` returning the
            three cumulative log-return paths over bars ``t0+1 … stop``
            inclusive, measured from the event's close.
        basis: which return series the barriers are being applied to.

    Returns:
        The assembled :class:`LabelSet`.

    Raises:
        InsufficientHistoryError: if an event lacks trailing or forward bars.
        DegenerateVolatilityError: if an event's volatility is non-positive or
            non-finite.
    """
    n_events = int(events.shape[0])
    outcome = np.empty(n_events, dtype=np.int8)
    resolution = np.empty(n_events, dtype=np.intp)
    realized = np.empty(n_events, dtype=np.float64)
    upper_barrier = np.empty(n_events, dtype=np.float64)
    lower_barrier = np.empty(n_events, dtype=np.float64)
    event_sigma = np.empty(n_events, dtype=np.float64)

    for position, raw_event in enumerate(events):
        t0 = int(raw_event)
        stop = t0 + spec.horizon
        if stop >= n_bars:
            raise InsufficientHistoryError(
                event_index=t0,
                n_bars=n_bars,
                bars_required_before=min_trailing_bars,
                bars_required_after=spec.horizon,
                detail=(f"the vertical barrier falls at bar {stop}, past the end of the series"),
            )
        if t0 < min_trailing_bars:
            raise InsufficientHistoryError(
                event_index=t0,
                n_bars=n_bars,
                bars_required_before=min_trailing_bars,
                bars_required_after=spec.horizon,
                detail=(
                    f"only {t0} bar(s) of history precede it, fewer than the "
                    f"{min_trailing_bars} the trailing estimate requires"
                ),
            )

        volatility = float(sigma[t0])
        if not math.isfinite(volatility) or volatility <= 0.0:
            raise DegenerateVolatilityError(event_index=t0, volatility=volatility)

        upper, lower = spec.barrier_log_returns(volatility)
        cumulative_high, cumulative_low, cumulative_close = cumulative(t0, stop)
        resolved, offset = resolve_barrier_touch(
            cumulative_high,
            cumulative_low,
            upper_log_return=upper,
            lower_log_return=lower,
            policy=spec.intrabar_policy,
        )

        outcome[position] = int(resolved)
        resolution[position] = t0 + 1 + offset
        realized[position] = float(cumulative_close[offset])
        upper_barrier[position] = upper
        lower_barrier[position] = lower
        event_sigma[position] = volatility

    return LabelSet(
        event_index=events,
        outcome=outcome,
        resolution_index=resolution,
        realized_log_return=realized,
        upper_barrier_log_return=upper_barrier,
        lower_barrier_log_return=lower_barrier,
        trailing_volatility=event_sigma,
        spec=spec,
        basis=basis,
    )
