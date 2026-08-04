"""Failure taxonomy for label construction (directive §5 Phase 6).

Every failure mode in this package is a *loud* failure. That is a deliberate
choice and it is the whole reason this module exists rather than a scattering of
bare ``ValueError``: directive §0.3 names label construction as a component where
"a silent error invalidates the entire system and will not surface as a test
failure". A label that is quietly wrong trains a model that is quietly wrong,
and every downstream number — the information coefficient, the backtest Sharpe,
the Deflated Sharpe — inherits the error while looking entirely healthy.

So there are no defaults here. An event without enough history to size its
barrier does not get a barrier sized from a shorter window; it raises. A stock
whose trailing volatility is exactly zero does not get a zero-width barrier that
is touched instantly in both directions; it raises. A factor design matrix whose
columns are collinear does not get a minimum-norm coefficient vector that is
then applied to future returns; it raises.

Each error carries the numbers that produced it, so the message alone is enough
to diagnose the cause without re-running under a debugger.
"""

from __future__ import annotations

__all__ = [
    "AmbiguousLabelError",
    "DegenerateVolatilityError",
    "InsufficientHistoryError",
    "LabelConfigurationError",
    "LabelError",
    "LabelInputError",
    "RankDeficientFactorError",
]


class LabelError(Exception):
    """Base class for every failure raised by :mod:`backend.labels`."""


class LabelConfigurationError(LabelError):
    """Raised when a specification cannot describe a well-posed label.

    Examples: a horizon of zero bars (the vertical barrier would coincide with
    the event), a non-positive barrier multiple (a zero-width barrier is not a
    barrier), a volatility window of one bar (a sample standard deviation with
    ``ddof=1`` is undefined on a single observation).
    """


class LabelInputError(LabelError):
    """Raised when the input series are malformed or mutually inconsistent.

    Examples: a non-positive close price (its logarithm is undefined), arrays of
    differing lengths, an event index outside the series, ``high < low`` on some
    bar.
    """


class InsufficientHistoryError(LabelError):
    """Raised when an event lacks the bars its label requires.

    A label needs history on both sides of its event: trailing bars to estimate
    the volatility that sizes the barrier, and forward bars up to the vertical
    barrier. Rather than shortening either window — which would make the barrier
    of an early event incomparable with the barrier of a later one, silently —
    the event is refused and the caller must drop it.

    :func:`backend.labels.barriers.usable_event_indices` returns exactly the
    events that satisfy both requirements, so the caller's filter is a one-liner
    rather than a guess.

    Attributes:
        event_index: position of the offending event in the input series
            (dimensionless bar index).
        n_bars: length of the input series (count of bars).
        bars_required_before: trailing bars the event needs strictly before it
            (count).
        bars_required_after: forward bars the event needs strictly after it
            (count).
    """

    def __init__(
        self,
        *,
        event_index: int,
        n_bars: int,
        bars_required_before: int,
        bars_required_after: int,
        detail: str,
    ) -> None:
        """Build the error from the event's position and its window demands.

        Args:
            event_index: position of the offending event (bar index).
            n_bars: length of the series the event indexes into (count).
            bars_required_before: trailing bars required (count).
            bars_required_after: forward bars required (count).
            detail: which requirement failed, in words.
        """
        self.event_index = event_index
        self.n_bars = n_bars
        self.bars_required_before = bars_required_before
        self.bars_required_after = bars_required_after
        super().__init__(
            f"event at bar {event_index} cannot be labelled: {detail}. "
            f"The series has {n_bars} bars; this event needs {bars_required_before} bar(s) "
            f"before it and {bars_required_after} bar(s) after it. "
            f"Filter events with usable_event_indices() rather than shortening a window — "
            f"a barrier sized from a shorter window is not comparable with the others."
        )


class DegenerateVolatilityError(LabelError):
    """Raised when an event's trailing volatility is zero or non-finite.

    A zero-volatility event produces a zero-width barrier, which every bar
    touches in both directions at once. The resulting "label" is an artefact of
    the arithmetic, not a fact about the security. In practice a run of
    identical closes means a halt, a stale feed, or a non-trading stub row — all
    of which are data problems that must be seen, not labelled around.

    Attributes:
        event_index: position of the offending event (bar index).
        volatility: the offending trailing volatility (standard deviation of
            daily log returns, expressed as a fraction per day).
    """

    def __init__(self, *, event_index: int, volatility: float, detail: str | None = None) -> None:
        """Build the error from the event position and its volatility.

        Args:
            event_index: position of the offending event (bar index).
            volatility: trailing daily log-return standard deviation (fraction
                per day) that was rejected.
            detail: optional extra sentence naming *why* it was rejected, for
                the cases that are not simply zero — a residual volatility that
                is nothing but floating-point noise, for instance.
        """
        self.event_index = event_index
        self.volatility = volatility
        super().__init__(
            f"event at bar {event_index} has trailing volatility {volatility!r} "
            f"(daily log-return standard deviation, fraction). A barrier sized by a "
            f"volatility that is zero, non-finite, or indistinguishable from zero has no "
            f"usable width and is touched by construction, not by price movement. "
            f"This event is refused: a flat trailing window means a halt, a stale feed or "
            f"a padded row, and labelling it would launder a data defect into training "
            f"data." + (f" {detail}" if detail else "")
        )


class RankDeficientFactorError(LabelError):
    """Raised when the residualization design matrix is (near-)collinear.

    The residual itself is well defined under collinearity — it is the
    projection onto the orthogonal complement of the factor span, which does not
    depend on how that span is parameterized. The *coefficients* are not: any
    point on a continuum of ``(alpha, beta_market, beta_sector)`` triples
    reproduces the same in-sample fit. Because those coefficients are then
    applied to **future** factor returns to build the forward residual path,
    an arbitrary choice among them silently produces an arbitrary label.

    The usual cause is a sector index that is a near-duplicate of the market
    index (a one-stock sector, or a sector proxy that *is* the market), or a
    constant factor over the estimation window.

    Attributes:
        event_index: position of the offending event (bar index).
        condition_number: condition number of the column-normalized design
            matrix (dimensionless). Columns are scaled to unit L2 norm first, so
            this measures collinearity only and is not inflated by the different
            scales of an intercept column and a return column.
        limit: the threshold that was exceeded (dimensionless).
    """

    def __init__(self, *, event_index: int, condition_number: float, limit: float) -> None:
        """Build the error from the event position and the conditioning.

        Args:
            event_index: position of the offending event (bar index).
            condition_number: condition number of the column-normalized design
                matrix (dimensionless).
            limit: the configured maximum (dimensionless).
        """
        self.event_index = event_index
        self.condition_number = condition_number
        self.limit = limit
        super().__init__(
            f"event at bar {event_index}: the residualization design matrix "
            f"[1, market, sector] has column-normalized condition number "
            f"{condition_number:.6g}, above the limit {limit:.6g}. The betas are not "
            f"identified, and they are applied to future factor returns to build the "
            f"forward residual path — so an arbitrary choice among the equivalent "
            f"coefficient vectors would produce an arbitrary label. Supply a sector "
            f"series that is not a near-duplicate of the market series."
        )


class AmbiguousLabelError(LabelError):
    """Raised when unresolvable labels are used where a class value is required.

    A bar whose high pierces the upper barrier *and* whose low pierces the lower
    barrier does not say which came first; daily bars do not carry that
    information. Those observations are marked
    :attr:`~backend.labels.barriers.TripleBarrierOutcome.AMBIGUOUS` rather than
    resolved by a coin-flip, and this error fires if a caller then asks for a
    numeric class label without deciding what to do with them.

    Attributes:
        n_ambiguous: number of ambiguous observations present (count).
        n_labels: total observations in the label set (count).
    """

    def __init__(self, *, n_ambiguous: int, n_labels: int) -> None:
        """Build the error from the ambiguous and total counts.

        Args:
            n_ambiguous: ambiguous observations present (count).
            n_labels: total observations in the label set (count).
        """
        self.n_ambiguous = n_ambiguous
        self.n_labels = n_labels
        super().__init__(
            f"{n_ambiguous} of {n_labels} labels are AMBIGUOUS: both barriers were "
            f"pierced within a single bar, and daily data cannot order them. "
            f"Call drop_ambiguous() to exclude them, or set the barrier spec's "
            f"intrabar_policy explicitly. They are not silently folded into the "
            f"vertical-barrier class, because 'we do not know' and 'neither barrier "
            f"was touched' are different facts."
        )
