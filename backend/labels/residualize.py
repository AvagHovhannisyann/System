"""Residualizing returns against market and sector before labelling (P6.2).

A raw triple-barrier label on a high-beta stock in a rising market is mostly a
label on *the market*. Feed enough of them to a cross-sectional ranker and it
learns beta, which is neither an edge nor tradeable in a beta-neutral portfolio
(directive §5 Phase 9 constrains the optimizer to beta and sector neutrality).
Residualizing first makes the label a statement about the stock's own movement.

--------------------------------------------------------------------------
The model
--------------------------------------------------------------------------

Over a trailing estimation window ending at the event bar ``t0`` (inclusive),
ordinary least squares::

    r_i,t  =  alpha  +  beta_m * r_m,t  +  beta_s * r_s,t  +  eps_i,t

with all three series per-bar natural-log returns as fractions. The window is
``[t0 - estimation_window + 1, t0]`` — **trailing**, ending at the event, never
spanning the labelling window. Fitting over the full sample, or over a window
that reaches past ``t0``, would let the label know the future beta; that is the
same class of lookahead as sizing a barrier with future volatility, and equally
invisible in any distribution check.

The coefficients are then **frozen** and applied forward, bar by bar, over the
labelling window::

    eps_i,s  =  r_i,s  -  beta_m * r_m,s  -  beta_s * r_s,s      for s > t0

and the barrier engine runs on the cumulative sum of those forward residuals,
which starts at zero at the event by construction. This is an out-of-sample
residual: the betas come from before the event, the returns from after it.

--------------------------------------------------------------------------
Three decisions that had to be made deliberately
--------------------------------------------------------------------------

**1. The intercept is estimated but not subtracted forward.** ``alpha`` is in
the regression because omitting it forces the fit through the origin and biases
both betas whenever the factors have non-zero mean over the window — which they
do, over 252 days. But the forward residual does **not** subtract ``alpha``.
Subtracting a trailing drift estimate from future returns would remove exactly
the idiosyncratic drift the label is meant to capture, and it would import a
noisy 252-day estimate of past drift into every forward bar. The fitted
``alpha`` is returned in :class:`ResidualFit` so a caller who wants the other
convention can apply it explicitly.

**2. Barriers are sized by trailing *residual* volatility.** The label measures
idiosyncratic movement, so its barrier is stated in idiosyncratic units: the
sample standard deviation of the last ``spec.volatility_window`` **in-sample**
residuals. Using total volatility would put a wide barrier around a stock whose
variance is nearly all market, and that stock's label would almost always be the
vertical barrier for reasons having nothing to do with it.

Stated rather than hidden: the in-sample residuals are alpha-inclusive (OLS
residuals sum to zero by construction) while the forward residuals are not, so
for a stock with strong trailing alpha the forward residual path has a non-zero
mean that the barrier width does not account for. That is intended — it is
decision 1 — but it means the residual barrier is a dispersion scale, not a
symmetric-probability statement.

**3. Near-collinear factors raise instead of resolving.** The residual is well
defined under collinearity (it is the projection onto the orthogonal complement
of the factor span, which does not depend on the parameterization) but the
*coefficients* are not, and it is the coefficients that get applied to future
factor returns. A minimum-norm solution would therefore produce an arbitrary
forward path with no visible symptom. See
:class:`~backend.labels.errors.RankDeficientFactorError`.

--------------------------------------------------------------------------
Necessarily close-only
--------------------------------------------------------------------------

A residual has no intrabar high or low: subtracting ``beta * r_market`` from a
stock's daily *high* is not a quantity that exists. Residualized labelling
therefore detects barrier touches at closes only, which under-counts touches
relative to a price-path label — the bias documented in
:mod:`backend.labels.barriers`. It also means
:class:`~backend.labels.barriers.TripleBarrierOutcome.AMBIGUOUS` cannot arise
here: a single close cannot be on both sides at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.labels._arrays import (
    FloatArray,
    as_float_1d,
    as_index_1d,
    require_positive,
    require_same_length,
)
from backend.labels.barriers import (
    BarrierSpec,
    LabelBasis,
    LabelSet,
    _label_events,
    usable_event_indices,
)
from backend.labels.errors import (
    DegenerateVolatilityError,
    InsufficientHistoryError,
    LabelConfigurationError,
    LabelInputError,
    RankDeficientFactorError,
)
from backend.labels.volatility import daily_log_returns

if TYPE_CHECKING:
    import numpy.typing as npt

    from backend.labels._arrays import IntArray

__all__ = [
    "N_REGRESSION_PARAMETERS",
    "ResidualFit",
    "ResidualSpec",
    "ResidualizedLabels",
    "fit_residual_model",
    "residualized_triple_barrier_labels",
    "usable_residual_event_indices",
]

N_REGRESSION_PARAMETERS: Final = 3
"""Columns in the design matrix: intercept, market return, sector return."""

_MIN_ESTIMATION_WINDOW: Final = N_REGRESSION_PARAMETERS + 2
"""Hard floor on the estimation window: three parameters plus two residual dof."""

MIN_RESIDUAL_VOLATILITY_RATIO: Final = 1.0e-8
"""Smallest residual volatility, as a fraction of the asset's own volatility.

Below this the "residual" is the arithmetic's rounding error rather than the
security's idiosyncratic movement: a stock whose returns are an exact linear
combination of the two factors over the window leaves residuals of order 1e-18,
which is a *positive* number and would size a barrier that the very first
forward bar clears in whichever direction the noise points. Double precision
carries about sixteen significant digits, so a ratio of 1e-8 leaves eight orders
of magnitude of margin between "small idiosyncratic variance" and "no
idiosyncratic variance at all".

This guard is specific to residualization because the residual is computed *by
this package*; a raw price series with a genuinely tiny volatility is a data
question, answered upstream by the ingestion layer's quality report.
"""


@dataclass(frozen=True, slots=True)
class ResidualSpec:
    """Configuration for the trailing factor regression.

    Attributes:
        estimation_window: trailing bars used to fit the betas, ending at and
            including the event bar (count). Default 252 — roughly one trading
            year, long enough for a two-factor beta to be estimated with useful
            precision and short enough to track a changing beta. The hard floor
            is 5 (three parameters plus two residual degrees of freedom); tests
            use short windows, but anything below about 60 gives betas too noisy
            to be worth applying forward, and that is a judgement, not a check.
        condition_number_limit: maximum condition number of the
            **column-normalized** design matrix ``[1, r_market, r_sector]``
            (dimensionless), above which the fit is refused. Columns are scaled
            to unit L2 norm before the condition number is taken, so this
            measures collinearity between the factors and is not inflated by the
            different natural scales of an intercept column and a return column.
            Default ``1e8``.
    """

    estimation_window: int = 252
    condition_number_limit: float = 1.0e8

    def __post_init__(self) -> None:
        """Validate the specification at construction.

        Raises:
            LabelConfigurationError: if the estimation window is below the hard
                floor, or the condition-number limit is not finite and greater
                than 1 (a condition number is never below 1).
        """
        if self.estimation_window < _MIN_ESTIMATION_WINDOW:
            msg = (
                f"estimation_window must be >= {_MIN_ESTIMATION_WINDOW} bars; got "
                f"{self.estimation_window}. The regression has "
                f"{N_REGRESSION_PARAMETERS} parameters, so a shorter window leaves "
                f"fewer than two residual degrees of freedom and the residual standard "
                f"deviation stops being an estimate of anything."
            )
            raise LabelConfigurationError(msg)
        if not math.isfinite(self.condition_number_limit) or self.condition_number_limit <= 1.0:
            msg = (
                f"condition_number_limit must be finite and > 1; got "
                f"{self.condition_number_limit!r}. A condition number is never below 1."
            )
            raise LabelConfigurationError(msg)


@dataclass(frozen=True, slots=True)
class ResidualFit:
    """One event's trailing factor regression.

    Attributes:
        alpha: fitted intercept — mean per-bar idiosyncratic drift over the
            estimation window (a fraction per bar). Reported but **not**
            subtracted from the forward residual; see the module docstring.
        beta_market: sensitivity to the market factor (dimensionless).
        beta_sector: sensitivity to the sector factor (dimensionless).
        residual_volatility: sample standard deviation (``ddof=1``) of the last
            ``volatility_window`` in-sample residuals — a per-bar idiosyncratic
            volatility as a fraction, not annualized. This is what sizes the
            barrier.
        condition_number: condition number of the column-normalized design
            matrix (dimensionless). Near 1 means the factors are close to
            orthogonal; large means they carry nearly the same information.
        n_observations: bars in the estimation window (count).
    """

    alpha: float
    beta_market: float
    beta_sector: float
    residual_volatility: float
    condition_number: float
    n_observations: int


@dataclass(frozen=True, slots=True)
class ResidualizedLabels:
    """Residualized triple-barrier labels together with their factor fits.

    The fits are returned rather than discarded because a label whose betas are
    absurd is a label to distrust, and there is no way to see that from the
    label alone.

    Attributes:
        labels: the label set, with :attr:`~backend.labels.barriers.LabelBasis`
            ``RESIDUAL``. Its ``trailing_volatility`` is residual volatility.
        alpha: fitted intercept per event (a fraction per bar), parallel to
            ``labels.event_index``.
        beta_market: fitted market beta per event (dimensionless).
        beta_sector: fitted sector beta per event (dimensionless).
        condition_number: design-matrix condition number per event
            (dimensionless).
    """

    labels: LabelSet
    alpha: FloatArray
    beta_market: FloatArray
    beta_sector: FloatArray
    condition_number: FloatArray


def usable_residual_event_indices(
    n_bars: int, spec: BarrierSpec, residual_spec: ResidualSpec
) -> IntArray:
    """Return every bar index that can carry a residualized label.

    Same purpose as :func:`~backend.labels.barriers.usable_event_indices`, with
    the longer trailing requirement of the factor regression.

    Args:
        n_bars: length of the series (count of bars).
        spec: the barrier configuration (for its horizon).
        residual_spec: the regression configuration (for its estimation
            window).

    Returns:
        Ascending bar indices (dimensionless), possibly empty.
    """
    return usable_event_indices(n_bars, spec, min_trailing_bars=residual_spec.estimation_window)


def fit_residual_model(
    asset_log_returns: FloatArray,
    market_log_returns: FloatArray,
    sector_log_returns: FloatArray,
    *,
    event_index: int,
    residual_spec: ResidualSpec,
    volatility_window: int,
) -> ResidualFit:
    """Fit the trailing two-factor model for one event.

    Args:
        asset_log_returns: per-bar log returns of the security (fractions),
            element ``0`` typically ``NaN``.
        market_log_returns: per-bar log returns of the market factor
            (fractions), aligned to ``asset_log_returns``.
        sector_log_returns: per-bar log returns of the sector factor
            (fractions), aligned likewise.
        event_index: the event's bar position. The estimation window is
            ``[event_index - estimation_window + 1, event_index]`` inclusive —
            trailing, ending at the event.
        residual_spec: window length and conditioning limit.
        volatility_window: how many of the most recent in-sample residuals
            define the residual volatility (count of bars). Must not exceed the
            estimation window.

    Returns:
        The :class:`ResidualFit` for this event.

    Raises:
        LabelConfigurationError: if ``volatility_window`` exceeds the estimation
            window.
        InsufficientHistoryError: if the estimation window would start before
            bar 1 (bar 0 has no return).
        LabelInputError: if any return in the estimation window is non-finite.
            Refused rather than dropped: silently fitting a 252-day beta on 180
            observations makes one event's barrier incomparable with another's.
        RankDeficientFactorError: if the design matrix is near-collinear.
        DegenerateVolatilityError: if the residual volatility is zero or
            non-finite — a perfect in-sample fit, which over a real window means
            the "sector" series *is* the security.
    """
    if volatility_window > residual_spec.estimation_window:
        msg = (
            f"volatility_window ({volatility_window}) cannot exceed estimation_window "
            f"({residual_spec.estimation_window}): the residual volatility is measured on "
            f"the in-sample residuals, and there are only estimation_window of them."
        )
        raise LabelConfigurationError(msg)

    start = event_index - residual_spec.estimation_window + 1
    if start < 1:
        raise InsufficientHistoryError(
            event_index=event_index,
            n_bars=int(asset_log_returns.shape[0]),
            bars_required_before=residual_spec.estimation_window,
            bars_required_after=0,
            detail=(
                f"the {residual_spec.estimation_window}-bar estimation window would start "
                f"at bar {start}, before the first bar that has a return"
            ),
        )

    stop = event_index + 1
    asset = asset_log_returns[start:stop]
    market = market_log_returns[start:stop]
    sector = sector_log_returns[start:stop]
    for name, series in (("asset", asset), ("market", market), ("sector", sector)):
        if not np.all(np.isfinite(series)):
            msg = (
                f"the {name} return series has non-finite values in the estimation window "
                f"[{start}, {event_index}] for the event at bar {event_index}. Refusing to "
                f"drop them: a beta fitted on a subset of the window is not comparable with "
                f"one fitted on the whole window, and nothing downstream would show it."
            )
            raise LabelInputError(msg)

    design = np.column_stack((np.ones_like(market), market, sector))
    condition_number = _normalized_condition_number(design)
    ill_conditioned = (
        not math.isfinite(condition_number)
        or condition_number > residual_spec.condition_number_limit
    )
    if ill_conditioned:
        raise RankDeficientFactorError(
            event_index=event_index,
            condition_number=condition_number,
            limit=residual_spec.condition_number_limit,
        )

    coefficients, *_ = np.linalg.lstsq(design, asset, rcond=None)
    in_sample_residuals = asset - design @ coefficients
    residual_volatility = float(np.std(in_sample_residuals[-volatility_window:], ddof=1))
    asset_volatility = float(np.std(asset[-volatility_window:], ddof=1))
    noise_floor = MIN_RESIDUAL_VOLATILITY_RATIO * asset_volatility
    if (
        not math.isfinite(residual_volatility)
        or residual_volatility <= 0.0
        or residual_volatility < noise_floor
    ):
        raise DegenerateVolatilityError(
            event_index=event_index,
            volatility=residual_volatility,
            detail=(
                f"The factors explain the security's returns to within rounding error over "
                f"this window (asset volatility {asset_volatility:.6g}, residual volatility "
                f"{residual_volatility:.6g}, floor {noise_floor:.6g}): what is left is the "
                f"arithmetic's noise, not idiosyncratic movement, and a barrier sized by it "
                f"would be resolved by the sign of that noise."
            ),
        )

    return ResidualFit(
        alpha=float(coefficients[0]),
        beta_market=float(coefficients[1]),
        beta_sector=float(coefficients[2]),
        residual_volatility=residual_volatility,
        condition_number=condition_number,
        n_observations=int(asset.shape[0]),
    )


def _normalized_condition_number(design: FloatArray) -> float:
    """Condition number of the design matrix after scaling columns to unit norm.

    Scaling first is what makes the number a measure of *collinearity* rather
    than of units: an intercept column of ones and a column of daily returns
    differ in norm by a factor of ~50 for no reason that matters, and that
    factor would otherwise dominate the condition number and set the threshold
    against the wrong thing.

    Args:
        design: the ``(n_observations, n_parameters)`` design matrix.

    Returns:
        The condition number (dimensionless, never below 1), or ``inf`` when a
        column is identically zero — a constant factor carries no information
        and is exactly collinear with the intercept.
    """
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms == 0.0):
        return math.inf
    return float(np.linalg.cond(design / norms))


def residualized_triple_barrier_labels(
    close: npt.ArrayLike,
    market_close: npt.ArrayLike,
    sector_close: npt.ArrayLike,
    event_index: npt.ArrayLike,
    spec: BarrierSpec,
    residual_spec: ResidualSpec | None = None,
) -> ResidualizedLabels:
    """Label events on the security's *idiosyncratic* return path.

    Args:
        close: the security's close prices, one per bar, finite and strictly
            positive.
        market_close: market index closes, aligned bar-for-bar with ``close``.
            A level series, not returns; it is differenced internally so the two
            cannot disagree about the return convention.
        sector_close: sector index closes, aligned likewise.
        event_index: bar positions to label. Use
            :func:`usable_residual_event_indices` to pick them.
        spec: the barrier configuration. ``spec.volatility_window`` selects how
            many recent in-sample residuals set the residual volatility.
            ``spec.intrabar_policy`` is unreachable here (close-only), and is
            carried through only for provenance.
        residual_spec: the regression configuration. Defaults to
            :class:`ResidualSpec` (252-bar trailing window).

    Returns:
        A :class:`ResidualizedLabels` whose ``labels`` carry
        :attr:`~backend.labels.barriers.LabelBasis.RESIDUAL`.

    Raises:
        LabelInputError: if the series are malformed, mismatched in length, or
            contain a non-finite return inside an estimation window.
        InsufficientHistoryError: if an event lacks the estimation window before
            it or the horizon after it.
        DegenerateVolatilityError: if an event's residual volatility is zero or
            non-finite.
        RankDeficientFactorError: if market and sector are near-collinear over
            an event's estimation window.
        LabelConfigurationError: if the barrier spec's volatility window exceeds
            the estimation window.
    """
    resolved_spec = ResidualSpec() if residual_spec is None else residual_spec

    prices = as_float_1d(close, name="close")
    market_prices = as_float_1d(market_close, name="market_close")
    sector_prices = as_float_1d(sector_close, name="sector_close")
    require_positive(prices, name="close")
    require_positive(market_prices, name="market_close")
    require_positive(sector_prices, name="sector_close")
    n_bars = require_same_length(
        {"close": prices, "market_close": market_prices, "sector_close": sector_prices}
    )

    asset_returns = daily_log_returns(prices)
    market_returns = daily_log_returns(market_prices)
    sector_returns = daily_log_returns(sector_prices)

    events = as_index_1d(event_index, name="event_index")
    if events.size and int(events.max()) >= n_bars:
        msg = (
            f"event_index contains bar {int(events.max())} but the series has {n_bars} bars "
            f"(valid indices are 0..{n_bars - 1})"
        )
        raise LabelInputError(msg)

    fits: dict[int, ResidualFit] = {}
    sigma = np.full(n_bars, np.nan, dtype=np.float64)
    for raw_event in events:
        t0 = int(raw_event)
        if t0 in fits:
            continue
        if t0 + spec.horizon >= n_bars or t0 < resolved_spec.estimation_window:
            # Let the shared event loop raise the history error, so the message
            # is identical whichever entry point the caller used.
            continue
        fits[t0] = fit_residual_model(
            asset_returns,
            market_returns,
            sector_returns,
            event_index=t0,
            residual_spec=resolved_spec,
            volatility_window=spec.volatility_window,
        )
        sigma[t0] = fits[t0].residual_volatility

    def cumulative(t0: int, stop: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Cumulative forward residual log return over bars ``t0+1 … stop``."""
        fit = fits[t0]
        forward = (
            asset_returns[t0 + 1 : stop + 1]
            - fit.beta_market * market_returns[t0 + 1 : stop + 1]
            - fit.beta_sector * sector_returns[t0 + 1 : stop + 1]
        )
        if not np.all(np.isfinite(forward)):
            msg = (
                f"the forward residual path for the event at bar {t0} contains non-finite "
                f"values over bars [{t0 + 1}, {stop}]; a gap in the factor or asset series "
                f"cannot be labelled around"
            )
            raise LabelInputError(msg)
        path: FloatArray = np.cumsum(forward)
        return path, path, path

    labels = _label_events(
        events=events,
        n_bars=n_bars,
        spec=spec,
        sigma=sigma,
        min_trailing_bars=resolved_spec.estimation_window,
        cumulative=cumulative,
        basis=LabelBasis.RESIDUAL,
    )

    return ResidualizedLabels(
        labels=labels,
        alpha=np.array([fits[int(t0)].alpha for t0 in events], dtype=np.float64),
        beta_market=np.array([fits[int(t0)].beta_market for t0 in events], dtype=np.float64),
        beta_sector=np.array([fits[int(t0)].beta_sector for t0 in events], dtype=np.float64),
        condition_number=np.array(
            [fits[int(t0)].condition_number for t0 in events], dtype=np.float64
        ),
    )
