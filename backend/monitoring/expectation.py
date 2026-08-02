"""Live-versus-expected against the CPCV distribution, with auto-halt (P12.1).

--------------------------------------------------------------------------
0. What the band is, and — more importantly — what it is not
--------------------------------------------------------------------------

The band this module builds is the **empirical spread of the strategy's own
backtest**, cut into windows the length of the live evaluation window. It is
computed from the out-of-sample paths of
:class:`~backend.backtest.cpcv.CombinatorialPurgedCV`, which is the strongest
backtest statement this platform can make — and it is still a backtest.

**It is not a confidence interval for live performance.** A confidence interval
answers "given the truth, where do estimates land?". This answers "across the
resamplings of one historical sample, where did this strategy's out-of-sample
statistic land?". Those coincide only if the future is drawn from the same
distribution as the sample, which is the assumption the band exists to test and
therefore cannot also assume.

**The null is contaminated, and this is the part that matters.** The strategy
being monitored was *selected* using this sample. Model configurations were
compared on it, features were kept or dropped on it, and the surviving
configuration is the one that did well on it. The CPCV path distribution is
therefore the distribution of the winner's in-sample-selected performance, and
its centre is biased upward by the selection. Two consequences, in opposite
directions, and both must be stated wherever the band is displayed:

* A live result **inside** the band is weak evidence. It is consistent with a
  strategy that works, and equally consistent with a strategy whose backtest was
  overfit by exactly the amount the live period happened to give back. "In band"
  is the absence of a specific alarm, not a validation, and this module never
  labels it one — :class:`ExpectationComparison` has no ``validated`` attribute
  and :meth:`ExpectationComparison.to_dict` emits the disclosures alongside every
  verdict.
* A live result **below** the band is strong evidence, and stronger than it looks
  — the comparison is being made against a benchmark that selection has already
  tilted in the strategy's favour, so falling out of the bottom of it means the
  live period disagrees with a distribution built to flatter.

The trial count that produced that selection travels on the band
(:attr:`CPCVArtefactRef.trials`, directive §6.7) precisely so a reader can see
how much flattering there was. A band from a search of 400 configurations with a
Deflated Sharpe Ratio near zero is an overfit band, and living inside it means
nothing at all. :mod:`backend.backtest.dsr` is where that correction lives; this
module does not repeat it and does not silently do without it.

**It is not corrected for regime.** The paths come from one historical sample.
A band cut from 2015-2020 says nothing about a market that has not happened yet;
it says what *this* sample would have produced.

--------------------------------------------------------------------------
1. How wide, and why
--------------------------------------------------------------------------

Width is not chosen. It falls out of three decisions that are chosen, each of
which is a field on :class:`HaltPolicy` and travels in the payload:

1. **The horizon is matched.** A live quarter is compared against the
   distribution of *quarters* inside the CPCV paths, never against the
   distribution of whole-sample path Sharpe ratios. The two differ by roughly
   ``sqrt(T_path / T_window)`` in spread — a factor of four or five in practice —
   and comparing a 63-day live estimate against a 1260-day band is not a
   conservative approximation, it is a category error that produces a band so
   narrow it halts constantly. So :meth:`ExpectationBand.from_paths` slides a
   window of exactly the live length along every path and bands the resulting
   statistics. Every window it uses is real out-of-sample backtest output; no
   distribution is fitted and nothing is simulated (I3).

2. **The tail mass is derived from a false-halt budget**, not picked. The
   operator declares how often a *working* strategy may be halted by chance in a
   year (:attr:`HaltPolicy.annual_false_halt_budget`); the per-evaluation tail
   mass follows from the number of evaluations in that year (§2). Under the null
   the band's own quantile is the false-halt rate by construction, which is the
   advantage of an empirical band over a parametric one: no normality assumption
   sits between the budget and the behaviour.

3. **A quantile the sample cannot resolve is refused, not approximated.** The
   number of *effectively independent* windows behind the band is
   ``n_paths * (T // window)`` — an upper bound, since CPCV paths share group
   forecasts with one another and are not independent either. If the requested
   tail mass times that count is below 1, the band's edge would be an extreme
   order statistic rather than a quantile, and
   :class:`~backend.monitoring.errors.ExpectationBandError` is raised.

Overlapping windows are pooled for the quantile estimate itself, because they
sharpen the shape estimate; they are *not* counted as independent draws for the
resolution test above. That split is the honest treatment of overlap.

--------------------------------------------------------------------------
2. How long before judging, and the multiple-testing decision
--------------------------------------------------------------------------

**The observation window is 63 trading periods by default — one quarter — and
evaluations are disjoint.** Both halves of that sentence are the decision.

*Why not shorter.* A halt on one bad day is a halt on noise. The daily Sharpe
estimate of a strategy with an annualised Sharpe of 1 has a standard error of
about 1 per day in the same units; there is no band a one-day statistic falls
outside of for a reason. Below :data:`MINIMUM_WINDOW_PERIODS` the window
statistic's sampling error exceeds any deterioration worth halting for, so this
module refuses those windows rather than offering a knob that produces noise.

*Why not longer.* A year-long window is one evaluation per year: the first halt
a broken strategy can produce arrives after it has already lost a year. That is
theatre — a control that reports after the loss it exists to prevent.

*The trade-off, stated in numbers rather than adjectives.* At the default
configuration (63-period window, 252 periods per year, a 10% annual false-halt
budget, two-sided) each evaluation carries a tail mass of about 0.013 per side,
and :meth:`ExpectationBand.detection_power` reports what that buys: a
deterioration of one band standard deviation is caught in a single quarter with
probability ~0.11, two with ~0.41, three with ~0.78. Expected time to halt is
one over that, in quarters. **This is the honest headline: a quarterly
live-versus-expected test is a weak test.** It reliably catches a strategy that
has broken badly and takes years to catch one that has merely decayed. The
alternative — a narrower band that catches decay — halts a working strategy
several times a year, and an auto-halt an operator learns to override is worse
than none. Numbers, so the choice can be argued with rather than deferred to.

**Multiple testing is handled by two decisions, not one.**

*Cadence.* Evaluations use **disjoint** windows. Checking a rolling window every
day means ~252 overlapping tests a year, of which the band is breached by chance
eventually and whose dependence structure has no closed form. Disjoint windows
make the count of tests per year small (four, by default) and their statistics
approximately independent, which is what makes the next step exact rather than
decorative. The cost is latency: a deterioration starting the day after an
evaluation waits a full window. That is accepted, and it is the reason the
window is a quarter rather than a year.

*Correction.* The per-evaluation tail mass is the Šidák correction of the annual
budget: ``alpha_eval = 1 - (1 - budget) ** (1 / m)`` with ``m`` the number of
evaluations per year. Bonferroni over 252 daily rolling tests was rejected: the
tests would be enormously dependent, so the correction would be far too
conservative, and a band that never fires is the same as no band at all.

*What is left unhandled, said plainly.* The correction controls the false-halt
rate **within one year of one strategy**. It does **not** control it across
strategies monitored in parallel, nor across years, nor jointly with the other
Phase 12 monitors (PSI drift, extraction quality) that can also raise alerts on
the same data. A platform running ten strategies under a 10% annual budget each
expects roughly one false halt a year somewhere. That is deliberate — halts are
per strategy and a halt of one is not a halt of all — but it is not the same
statement as "one false halt per platform-decade", and nothing here should be
read as claiming that.

--------------------------------------------------------------------------
3. Auto-halt fails closed
--------------------------------------------------------------------------

:func:`decide` has exactly two outcomes and no third. It returns
:attr:`HaltAction.CONTINUE` **only** when a comparison was actually performed and
the live statistic was not outside the halting side of the band. Every other
path — no band, no live data, a live window of the wrong length, stale live data,
a cost basis that is not net on both sides, an arithmetic refusal, or an
unexpected exception from anywhere inside the comparison — returns
:attr:`HaltAction.HALT` with the reason attached.

That is enforced structurally rather than by discipline:
:class:`HaltDecision` refuses to be constructed with ``CONTINUE`` unless it is
carrying a completed :class:`ExpectationComparison` that did not breach the
halting side. There is no way to express "continue, comparison unavailable" in
the type; the object cannot be built.

The direction is deliberate under I3. A monitor that cannot see is
indistinguishable, from the outside, from a monitor that sees nothing wrong, and
those two must not produce the same action. Halting a healthy strategy costs the
spread on an unwound position and an operator's afternoon; not halting a broken
one costs the account.

*Halting side.* The default is **both** sides. The lower breach is the obvious
one. The upper breach halts too, because a live result far above a distribution
built from the same data is not good news: the overwhelmingly likely causes are
an accounting error, a missed cost, an unintended position scale, or a data
feed that is not what the model was trained on. Continuing to trade on
accounting that visibly disagrees with the model that sized the positions is not
the conservative choice. An operator who wants the classic one-sided control
sets :attr:`HaltSide.LOWER`, and the upper breach then still raises an alert
through P12.4 rather than passing silently.

--------------------------------------------------------------------------
4. Net of costs on both sides, or the comparison is meaningless (I4)
--------------------------------------------------------------------------

Both sides declare a :class:`CostTreatment` and anything but net on both raises
:class:`~backend.monitoring.errors.CostBasisError`. The failure this prevents is
silent and directional: gross live returns compared against a net-of-cost
backtest band sit *above* the band by roughly the cost drag, for as long as the
costs are positive, which reads as "outperforming" and is the reading that keeps
a losing strategy funded.

The two nets are still not the same net, and the comparison says so rather than
pretending: the backtest side is net of **modelled** costs, the live side net of
**realised** ones. A live shortfall inside the band may be entirely a cost-model
error rather than an alpha decay, and :attr:`ExpectationComparison.cost_note`
carries that sentence into every payload so the P9/P11 cost calibration is
looked at before the model is blamed.

--------------------------------------------------------------------------
5. No live track record exists in this repository (I3)
--------------------------------------------------------------------------

Nothing in this module fabricates, simulates, extends or infills a live series.
:class:`LiveWindow` is a container for returns a caller already has, and every
entry point refuses an empty, short, non-finite or wrong-length one. Blocker B2
(no IBKR paper credentials) means no live series exists here at all, so every
test in this package is explicitly synthetic and named so. There is no default
live sample, and no code path that produces one.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.backtest.metrics import (
    FloatArray,
    as_float_array,
    sharpe_ratio,
    standard_normal_cdf,
    standard_normal_ppf,
)
from backend.monitoring.errors import (
    ComparisonUnavailableError,
    CostBasisError,
    ExpectationBandError,
)
from backend.monitoring.psi import JsonValue
from backend.tracking.stamp import ReproducibilityStamp, canonical_config_hash

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "DEFAULT_ANNUAL_FALSE_HALT_BUDGET",
    "DEFAULT_MAX_LIVE_AGE_DAYS",
    "DEFAULT_WINDOW_PERIODS",
    "MINIMUM_WINDOW_PERIODS",
    "SELECTION_CONTAMINATION_DISCLOSURE",
    "CPCVArtefactRef",
    "CostTreatment",
    "ExpectationBand",
    "ExpectationComparison",
    "HaltAction",
    "HaltCause",
    "HaltDecision",
    "HaltPolicy",
    "HaltSide",
    "LiveWindow",
    "WindowStatistic",
    "compare",
    "decide",
    "path_digest",
]

DEFAULT_WINDOW_PERIODS: Final = 63
"""Evaluation window, in observation periods. 63 trading days is one quarter.

Chosen, not derived — see the module docstring §2 for the power/latency argument
and :meth:`ExpectationBand.detection_power` for the numbers at this setting.
"""

MINIMUM_WINDOW_PERIODS: Final = 21
"""Shortest admissible evaluation window (periods). One trading month.

Below this the window statistic is dominated by its own sampling error: a
21-period Sharpe estimate already has a standard error near ``1/sqrt(21)`` per
period, and a shorter one produces a band no plausible deterioration falls
outside of. Offering the knob and letting it produce noise would be worse than
refusing it.
"""

DEFAULT_ANNUAL_FALSE_HALT_BUDGET: Final = 0.10
"""Probability that a *working* strategy is halted by chance at least once a year.

A budget, not a significance level: it is the family-wise rate across the
disjoint evaluations of one strategy in one year, and the per-evaluation tail
mass is derived from it (module docstring §2). Ten percent is a judgement — one
false halt per decade of strategy-years would need a band so wide it detects
nothing at a quarterly horizon.
"""

DEFAULT_MAX_LIVE_AGE_DAYS: Final = 4
"""How stale the live window's last observation may be at decision time (days).

Four calendar days covers a Friday close evaluated on the following Tuesday.
Beyond that the data pipeline has stopped rather than the market having been
quiet, and a monitor reading stale returns reports the *last* verdict forever —
which is the failure mode where a halt condition arrives and nothing happens.
Stale data halts (module docstring §3).
"""

SELECTION_CONTAMINATION_DISCLOSURE: Final = (
    "This band is the spread of the strategy's own backtest, not a confidence interval for "
    "live performance. The CPCV distribution it comes from was computed on the sample the "
    "strategy was SELECTED on, so its centre is biased upward by that selection: a live "
    "result inside the band is consistent with a working strategy AND with a backtest "
    "overfit by exactly the amount the live period gave back. 'Inside the band' is the "
    "absence of an alarm, never a validation. Read it next to the trial count and the "
    "Deflated Sharpe Ratio of the run that produced it (directive §6.7)."
)
"""The §0 caveat, in the payload rather than only in this docstring."""

_HORIZON_DISCLOSURE: Final = (
    "The band is cut from windows of exactly the live window's length, taken from the CPCV "
    "out-of-sample paths. Quantiles are estimated from overlapping windows (which sharpen "
    "the shape) but the resolution check counts only non-overlapping ones (which are the "
    "independent draws). CPCV paths share group forecasts with each other, so even that "
    "count is an upper bound on independence."
)

_COST_NOTE: Final = (
    "Both sides are net of costs (I4), but not of the same costs: the band is net of "
    "MODELLED costs (P9.3) and the live window is net of REALISED ones. A live shortfall "
    "inside or below the band may be a cost-model error rather than alpha decay — check the "
    "P11.8 cost calibration before concluding the model has stopped working."
)

_TWO_SIDED: Final = 2
_MIN_PATHS_FOR_SPREAD: Final = 2


class CostTreatment(StrEnum):
    """How costs were handled in a return series (invariant I4).

    ``StrEnum`` so the value stored on a row and the value compared in code are
    the same object.

    Attributes:
        NET_MODELLED: net of the modelled cost stack — half-spread, commission,
            square-root impact, borrow (:mod:`backend.costs`). What a backtest
            path carries.
        NET_REALISED: net of costs actually charged by the venue. What a paper or
            live fill series carries.
        GROSS: costs not deducted. Refused on both sides.
        UNKNOWN: cost handling not declared. Refused — an undeclared basis is
            not evidence of a net one.
    """

    NET_MODELLED = "net_modelled"
    NET_REALISED = "net_realised"
    GROSS = "gross"
    UNKNOWN = "unknown"

    @property
    def is_net(self) -> bool:
        """Whether this treatment deducts costs at all."""
        return self in {CostTreatment.NET_MODELLED, CostTreatment.NET_REALISED}


class WindowStatistic(StrEnum):
    """The statistic compared between the live window and the band.

    Whatever is chosen is applied by the *same function* to the backtest windows
    and to the live window, so the two sides cannot be computed differently.

    Attributes:
        SHARPE: per-period Sharpe ratio, ``ddof=1``, never annualised — the
            scale-free choice, and the noisiest at short horizons.
        MEAN_RETURN: arithmetic mean per-period return, as a fraction.
        CUMULATIVE_RETURN: compounded return over the window, as a fraction.
    """

    SHARPE = "sharpe"
    MEAN_RETURN = "mean_return"
    CUMULATIVE_RETURN = "cumulative_return"


def _window_statistic(
    values: FloatArray, *, statistic: WindowStatistic, risk_free_rate: float
) -> float:
    """Compute one window's statistic.

    The single implementation both sides go through, so a band and the live
    value it is compared against cannot be produced by different arithmetic.

    Args:
        values: per-period simple returns as fractions, **net of costs** (I4),
            finite, at least :data:`MINIMUM_WINDOW_PERIODS` long.
        statistic: which statistic to compute.
        risk_free_rate: per-period risk-free rate as a fraction, used only by
            :attr:`WindowStatistic.SHARPE`.

    Returns:
        The statistic, dimensionless for ``SHARPE`` and a fraction otherwise.

    Raises:
        ValueError: propagated from :func:`~backend.backtest.metrics.sharpe_ratio`
            when the window has zero dispersion, which leaves a Sharpe ratio
            undefined.
    """
    if statistic is WindowStatistic.SHARPE:
        return sharpe_ratio(values, risk_free_rate=risk_free_rate, periods_per_year=None, ddof=1)
    if statistic is WindowStatistic.MEAN_RETURN:
        return float(np.mean(values))
    return float(np.prod(1.0 + values) - 1.0)


class HaltSide(StrEnum):
    """Which breaches halt trading.

    Attributes:
        LOWER: only underperformance halts. The upper breach still alerts.
        BOTH: either breach halts — the default. See the module docstring §3 on
            why an upside breach is not good news.
    """

    LOWER = "lower"
    BOTH = "both"


class HaltAction(StrEnum):
    """The decision an evaluation produces.

    Attributes:
        CONTINUE: keep trading. Constructible only with a completed, in-band
            comparison attached (:class:`HaltDecision`).
        HALT: stop trading and require an operator.
    """

    CONTINUE = "continue"
    HALT = "halt"


class HaltCause(StrEnum):
    """Why a halt was raised. Every value names a distinct operator response.

    Attributes:
        BELOW_EXPECTED_BAND: the live statistic fell below the band's lower
            edge. The intended signal.
        ABOVE_EXPECTED_BAND: it rose above the upper edge — usually an
            accounting, scale or data error rather than good fortune.
        COMPARISON_UNAVAILABLE: the comparison could not be made. Fail-closed
            (module docstring §3); never a pass.
        STALE_LIVE_DATA: the live window's last observation is older than the
            policy permits, so the monitor is reporting an old verdict.
        COST_BASIS: one side was not net of costs (I4).
        INTERNAL_ERROR: an unexpected exception inside the decision path. Halts
            for the same reason as the rest: a monitor that crashed did not
            check anything.
    """

    BELOW_EXPECTED_BAND = "below_expected_band"
    ABOVE_EXPECTED_BAND = "above_expected_band"
    COMPARISON_UNAVAILABLE = "comparison_unavailable"
    STALE_LIVE_DATA = "stale_live_data"
    COST_BASIS = "cost_basis"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class CPCVArtefactRef:
    """Identity of the CPCV artefact a band was cut from (invariant I2).

    A band without this is a set of numbers nobody can re-derive. Every field is
    required, and :attr:`trials` in particular is not optional: directive §6.7
    requires the trial count beside every backtest result, and the whole §0
    caveat about selection is unreadable without it.

    Attributes:
        artefact_id: stable identifier of the backtest run artefact (the
            ``results_digest`` of a
            :class:`~backend.backtest.artifact.RunArtifact`, an MLflow run id,
            or whatever the caller stores runs under).
        stamp: the I2 stamp of the run that produced the paths — git commit,
            data version, config hash, seed.
        path_digest: SHA-256 over the path matrix the band was cut from, so two
            bands quoting the same artefact id can be told apart if the paths
            were not in fact the same numbers.
        n_paths: number of CPCV backtest paths, ``C(N-1, k-1)`` (count).
        n_observations: length of each path, in observation periods (count).
        n_groups: the CPCV ``N`` (count).
        n_test_groups: the CPCV ``k`` (count).
        trials: configurations evaluated during selection, from
            ``TESTING_LEDGER.md``. At least 1.
        cost_treatment: how the paths handle costs. Must be net (I4).
        periods_per_year: annualisation factor of the paths' periodicity (252
            for daily bars). Used to translate evaluation cadence into a year.
    """

    artefact_id: str
    stamp: ReproducibilityStamp
    path_digest: str
    n_paths: int
    n_observations: int
    n_groups: int
    n_test_groups: int
    trials: int
    cost_treatment: CostTreatment
    periods_per_year: float

    def __post_init__(self) -> None:
        """Validate the artefact identity.

        Raises:
            ExpectationBandError: if the identifier or digest is blank, the
                stamp is not a :class:`~backend.tracking.stamp.ReproducibilityStamp`,
                any count is below its minimum, ``periods_per_year`` is not
                positive and finite, or the CPCV shape is impossible.
            CostBasisError: if the paths are not net of costs (I4).
        """
        for field_name, value in (
            ("artefact_id", self.artefact_id),
            ("path_digest", self.path_digest),
        ):
            supplied: object = value
            if not isinstance(supplied, str) or not supplied.strip():
                msg = (
                    f"{field_name} must be a non-empty string; a band that cannot name the "
                    f"CPCV artefact it came from is not reproducible (I2)"
                )
                raise ExpectationBandError(msg)
        supplied_stamp: object = self.stamp
        if not isinstance(supplied_stamp, ReproducibilityStamp):
            msg = (
                f"stamp must be a ReproducibilityStamp, got {type(supplied_stamp).__name__}. "
                f"A halt decision that cannot say which commit, data version, config and "
                f"seed produced its band cannot be audited (I2)."
            )
            raise ExpectationBandError(msg)
        for field_name, count, minimum in (
            ("n_paths", self.n_paths, 1),
            ("n_observations", self.n_observations, MINIMUM_WINDOW_PERIODS),
            ("n_groups", self.n_groups, 2),
            ("n_test_groups", self.n_test_groups, 1),
            ("trials", self.trials, 1),
        ):
            supplied_count: object = count
            if isinstance(supplied_count, bool) or not isinstance(supplied_count, int):
                msg = f"{field_name} must be an int, got {type(supplied_count).__name__}"
                raise ExpectationBandError(msg)
            if count < minimum:
                msg = f"{field_name} must be at least {minimum}; got {count}"
                raise ExpectationBandError(msg)
        if self.n_test_groups >= self.n_groups:
            msg = (
                f"n_test_groups={self.n_test_groups} must be below n_groups={self.n_groups}; "
                f"a split with no training groups produces no out-of-sample forecast"
            )
            raise ExpectationBandError(msg)
        if not math.isfinite(self.periods_per_year) or self.periods_per_year <= 0.0:
            msg = f"periods_per_year must be finite and positive; got {self.periods_per_year!r}"
            raise ExpectationBandError(msg)
        treatment: object = self.cost_treatment
        if not isinstance(treatment, CostTreatment) or not self.cost_treatment.is_net:
            msg = (
                f"the CPCV paths declare cost_treatment={self.cost_treatment!r}. A band cut "
                f"from gross backtest returns sits above any net live series by the cost "
                f"drag, for as long as costs are positive — which reads as 'live is fine' "
                f"and is the most dangerous direction for this comparison to be wrong in "
                f"(I4, directive §9.6)."
            )
            raise CostBasisError(msg)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the artefact identity as a JSON-safe mapping."""
        return {
            "artefact_id": self.artefact_id,
            "path_digest": self.path_digest,
            "git_reference": self.stamp.git_reference,
            "data_version": self.stamp.data_version,
            "config_hash": self.stamp.config_hash,
            "seed": self.stamp.seed,
            "reproducible": self.stamp.reproducible,
            "n_paths": self.n_paths,
            "n_observations": self.n_observations,
            "cpcv_n_groups": self.n_groups,
            "cpcv_n_test_groups": self.n_test_groups,
            "trials": self.trials,
            "cost_treatment": str(self.cost_treatment),
            "periods_per_year": self.periods_per_year,
        }


def path_digest(paths: npt.ArrayLike) -> str:
    """Return a SHA-256 digest of a CPCV path matrix.

    Values are hashed as ``float.hex()`` strings so the digest captures the
    exact doubles rather than a decimal rendering of them.

    Args:
        paths: shape ``(n_paths, n_observations)`` net-of-cost per-period
            returns, as produced by
            :meth:`~backend.backtest.cpcv.CPCVSplits.assemble_paths`.

    Returns:
        A 64-character lowercase hex digest.

    Raises:
        ExpectationBandError: if ``paths`` is not a finite two-dimensional
            array.
    """
    matrix = _validated_paths(paths)
    return canonical_config_hash(
        {
            "shape": list(matrix.shape),
            "values": [float(value).hex() for value in matrix.reshape(-1)],
        }
    )


def _validated_paths(paths: npt.ArrayLike) -> FloatArray:
    """Coerce and validate a CPCV path matrix.

    Args:
        paths: shape ``(n_paths, n_observations)`` per-period returns.

    Returns:
        A two-dimensional ``float64`` array.

    Raises:
        ExpectationBandError: if the array is not two-dimensional, is empty, or
            holds a NaN or infinity. A NaN in a path would propagate into a
            quantile as a silently misplaced band edge.
    """
    matrix = np.asarray(paths, dtype=np.float64)
    if matrix.ndim != _TWO_SIDED:
        msg = f"paths must be two-dimensional (n_paths, n_observations); got shape {matrix.shape}"
        raise ExpectationBandError(msg)
    if matrix.size == 0:
        msg = "paths must be non-empty; there is no band without a distribution to cut it from"
        raise ExpectationBandError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = (
            "paths must be finite; found NaN or infinity. A non-finite path value would "
            "propagate into the band edge as a silently misplaced quantile"
        )
        raise ExpectationBandError(msg)
    return matrix


@dataclass(frozen=True, slots=True)
class HaltPolicy:
    """The cadence, budget and side of the auto-halt test.

    Every derived quantity below is a property rather than a stored field, so a
    stored policy cannot disagree with its own arithmetic.

    Attributes:
        window_periods: evaluation window length in observation periods. At
            least :data:`MINIMUM_WINDOW_PERIODS`. Default
            :data:`DEFAULT_WINDOW_PERIODS`.
        periods_per_year: annualisation factor of the observation periodicity
            (252 for daily bars).
        annual_false_halt_budget: probability that a working strategy is halted
            by chance at least once in a year, strictly inside ``(0, 1)``.
            Default :data:`DEFAULT_ANNUAL_FALSE_HALT_BUDGET`.
        side: which breaches halt. Default :attr:`HaltSide.BOTH`.
        max_live_age_days: how stale the live window's last observation may be
            at decision time (days). Default :data:`DEFAULT_MAX_LIVE_AGE_DAYS`.
    """

    window_periods: int = DEFAULT_WINDOW_PERIODS
    periods_per_year: float = 252.0
    annual_false_halt_budget: float = DEFAULT_ANNUAL_FALSE_HALT_BUDGET
    side: HaltSide = HaltSide.BOTH
    max_live_age_days: int = DEFAULT_MAX_LIVE_AGE_DAYS

    def __post_init__(self) -> None:
        """Validate the policy.

        Raises:
            ExpectationBandError: if the window is shorter than
                :data:`MINIMUM_WINDOW_PERIODS`, the periodicity is not positive
                and finite, the budget is not strictly inside ``(0, 1)``, the
                side is not a :class:`HaltSide`, the staleness allowance is
                negative, or the window is longer than a year (which leaves
                fewer than one evaluation per year and makes the Šidák
                correction meaningless).
        """
        supplied_window: object = self.window_periods
        if isinstance(supplied_window, bool) or not isinstance(supplied_window, int):
            msg = f"window_periods must be an int, got {type(supplied_window).__name__}"
            raise ExpectationBandError(msg)
        if self.window_periods < MINIMUM_WINDOW_PERIODS:
            msg = (
                f"window_periods={self.window_periods} is below the minimum of "
                f"{MINIMUM_WINDOW_PERIODS}. A shorter window's statistic is dominated by its "
                f"own sampling error, so the band it is compared against is one no plausible "
                f"deterioration falls outside of — a halt on one bad stretch is a halt on "
                f"noise (module docstring §2)."
            )
            raise ExpectationBandError(msg)
        if not math.isfinite(self.periods_per_year) or self.periods_per_year <= 0.0:
            msg = f"periods_per_year must be finite and positive; got {self.periods_per_year!r}"
            raise ExpectationBandError(msg)
        if self.window_periods > self.periods_per_year:
            msg = (
                f"window_periods={self.window_periods} exceeds periods_per_year="
                f"{self.periods_per_year!r}: fewer than one evaluation per year makes the "
                f"annual false-halt budget uninterpretable, and a control that reports after "
                f"the year it was meant to protect is theatre (module docstring §2)."
            )
            raise ExpectationBandError(msg)
        if not 0.0 < self.annual_false_halt_budget < 1.0:
            msg = (
                f"annual_false_halt_budget must lie strictly inside (0, 1); got "
                f"{self.annual_false_halt_budget!r}. Zero means no band edge exists; one "
                f"means halting is certain."
            )
            raise ExpectationBandError(msg)
        supplied_side: object = self.side
        if not isinstance(supplied_side, HaltSide):
            msg = f"side must be a HaltSide, got {type(supplied_side).__name__}"
            raise ExpectationBandError(msg)
        supplied_age: object = self.max_live_age_days
        if isinstance(supplied_age, bool) or not isinstance(supplied_age, int):
            msg = f"max_live_age_days must be an int, got {type(supplied_age).__name__}"
            raise ExpectationBandError(msg)
        if self.max_live_age_days < 0:
            msg = f"max_live_age_days must be non-negative; got {self.max_live_age_days}"
            raise ExpectationBandError(msg)

    @property
    def evaluations_per_year(self) -> float:
        """Disjoint evaluations in a year (count, fractional).

        Disjoint is the point: overlapping daily evaluations would be ~252
        dependent tests a year with no closed-form correction (module docstring
        §2).
        """
        return self.periods_per_year / self.window_periods

    @property
    def per_evaluation_alpha(self) -> float:
        """Tail mass spent per evaluation, total across the halting sides.

        Šidák: ``1 - (1 - budget) ** (1 / m)``. Exact when the ``m`` evaluations
        are independent, which is what the disjoint cadence buys; disjoint
        windows of a serially correlated return series are approximately, not
        exactly, independent, so this is close rather than exact.
        """
        survival = math.pow(1.0 - self.annual_false_halt_budget, 1.0 / self.evaluations_per_year)
        return 1.0 - survival

    @property
    def n_halting_sides(self) -> int:
        """How many band edges can trigger a halt (1 or 2)."""
        return _TWO_SIDED if self.side is HaltSide.BOTH else 1

    @property
    def tail_mass(self) -> float:
        """Probability mass outside **each** band edge (a fraction in ``(0, 1)``).

        The per-evaluation budget split evenly over the halting sides. With
        :attr:`HaltSide.LOWER` the whole budget goes to the lower tail and the
        upper edge is reported at the same mass for symmetry of display — it
        alerts rather than halts, so it spends no halt budget.
        """
        return self.per_evaluation_alpha / self.n_halting_sides

    @property
    def minimum_independent_windows(self) -> float:
        """Independent windows needed to resolve :attr:`tail_mass` (count).

        ``1 / tail_mass``: below this the band edge is the most extreme order
        statistic available rather than a quantile, and halting on it means
        halting on an order statistic's sampling error (module docstring §1.3).
        """
        return 1.0 / self.tail_mass

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the policy and its derived quantities as a JSON-safe mapping."""
        return {
            "window_periods": self.window_periods,
            "periods_per_year": self.periods_per_year,
            "annual_false_halt_budget": self.annual_false_halt_budget,
            "side": str(self.side),
            "max_live_age_days": self.max_live_age_days,
            "evaluations_per_year": self.evaluations_per_year,
            "per_evaluation_alpha": self.per_evaluation_alpha,
            "tail_mass_per_side": self.tail_mass,
            "correction": (
                "Sidak over disjoint evaluations within one year of one strategy. Does NOT "
                "cover strategies monitored in parallel, multiple years, or the other Phase "
                "12 monitors (module docstring §2)."
            ),
        }


@dataclass(frozen=True, slots=True)
class LiveWindow:
    """One evaluation window of realised live performance.

    A container, never a producer. Nothing in this module creates, extends or
    infills a live return series (I3): no live track record exists in this
    repository, and a plausible-looking one would be the single most damaging
    artefact it could contain.

    Attributes:
        returns: realised simple per-period returns as fractions, in
            chronological order, **net of realised costs** (I4). Finite and
            non-empty.
        as_of: the date of the window's **last** observation.
        cost_treatment: how costs were handled. Must be net.
        source: where the returns came from, in words — the paper account and
            reconciliation run they were computed from. Carried into the payload
            so a halt can be traced to the numbers that caused it.
    """

    returns: FloatArray
    as_of: dt.date
    cost_treatment: CostTreatment
    source: str

    def __post_init__(self) -> None:
        """Validate and normalise the live sample.

        Raises:
            ComparisonUnavailableError: if the returns are empty, not
                one-dimensional, or contain a NaN or infinity — all of which
                mean there is no window to compare, which is a halting
                condition rather than a value.
            ExpectationBandError: if ``as_of`` is not a date or ``source`` is
                blank.
            CostBasisError: if the returns are not net of costs (I4).
        """
        try:
            values = as_float_array(self.returns, name="returns")
        except ValueError as exc:
            msg = (
                f"the live return window is unusable ({exc}). There is no comparison to "
                f"make, and 'no comparison' is never reported as 'within band' (I3)."
            )
            raise ComparisonUnavailableError(msg) from exc
        object.__setattr__(self, "returns", values)
        supplied_date: object = self.as_of
        if not isinstance(supplied_date, dt.date) or isinstance(supplied_date, dt.datetime):
            msg = (
                f"as_of must be a datetime.date (not a datetime), got "
                f"{type(supplied_date).__name__}"
            )
            raise ExpectationBandError(msg)
        supplied_source: object = self.source
        if not isinstance(supplied_source, str) or not supplied_source.strip():
            msg = (
                "source must say where the live returns came from; a halt whose evidence "
                "cannot be traced to a reconciliation run cannot be reviewed"
            )
            raise ExpectationBandError(msg)
        treatment: object = self.cost_treatment
        if not isinstance(treatment, CostTreatment) or not self.cost_treatment.is_net:
            msg = (
                f"the live window declares cost_treatment={self.cost_treatment!r}. A gross "
                f"live series sits above a net-of-cost band by the cost drag and reads as "
                f"outperformance for as long as costs are positive (I4, directive §9.6)."
            )
            raise CostBasisError(msg)

    @property
    def n_periods(self) -> int:
        """Length of the window in observation periods (count)."""
        return int(self.returns.size)

    def age_days(self, now: dt.date) -> int:
        """Days between the window's last observation and ``now``.

        Args:
            now: the decision date.

        Returns:
            The difference in days; negative if ``as_of`` is in the future.
        """
        return (now - self.as_of).days


@dataclass(frozen=True, slots=True)
class ExpectationBand:
    """The expected range of the window statistic, cut from a CPCV distribution.

    Read :mod:`this module's docstring §0 <backend.monitoring.expectation>`
    before reading a number off this object. Summarised there and repeated in
    :attr:`disclosures`: this is the spread of the strategy's own backtest, on
    the sample the strategy was selected on, and "inside it" is not validation.

    Attributes:
        statistic: the statistic banded, applied identically to both sides.
        policy: the cadence, budget and side that set the edges.
        artefact: identity of the CPCV artefact the windows came from (I2).
        lower: lower band edge, in the statistic's units.
        upper: upper band edge.
        median: the pooled windows' median, for display beside the edges.
        dispersion: standard deviation across the pooled windows, used by
            :meth:`detection_power`. In the statistic's units.
        n_pooled_windows: overlapping windows the quantiles were estimated from
            (count).
        n_independent_windows: non-overlapping windows behind the band (count) —
            the honest resolution figure, itself an upper bound because CPCV
            paths are not independent of one another.
        risk_free_rate: per-period risk-free rate as a fraction, used for
            :attr:`WindowStatistic.SHARPE` on **both** sides.
    """

    statistic: WindowStatistic
    policy: HaltPolicy
    artefact: CPCVArtefactRef
    lower: float
    upper: float
    median: float
    dispersion: float
    n_pooled_windows: int
    n_independent_windows: int
    risk_free_rate: float

    def __post_init__(self) -> None:
        """Validate the band.

        Raises:
            ExpectationBandError: if any edge is non-finite, the edges are out
                of order, the dispersion is negative, or the window counts are
                not positive.
        """
        for field_name, value in (
            ("lower", self.lower),
            ("upper", self.upper),
            ("median", self.median),
            ("dispersion", self.dispersion),
            ("risk_free_rate", self.risk_free_rate),
        ):
            if not math.isfinite(value):
                msg = (
                    f"{field_name}={value!r} is not finite. An infinite band edge is a band "
                    f"nothing ever falls outside of, which is an auto-halt that never halts."
                )
                raise ExpectationBandError(msg)
        if not self.lower <= self.median <= self.upper:
            msg = (
                f"band edges are out of order: lower={self.lower!r}, median={self.median!r}, "
                f"upper={self.upper!r}"
            )
            raise ExpectationBandError(msg)
        if self.dispersion < 0.0:
            msg = f"dispersion must be non-negative; got {self.dispersion!r}"
            raise ExpectationBandError(msg)
        for field_name, count in (
            ("n_pooled_windows", self.n_pooled_windows),
            ("n_independent_windows", self.n_independent_windows),
        ):
            if count < 1:
                msg = f"{field_name} must be at least 1; got {count}"
                raise ExpectationBandError(msg)

    @property
    def window_periods(self) -> int:
        """Horizon the band describes, in observation periods (count)."""
        return self.policy.window_periods

    @property
    def width(self) -> float:
        """Distance between the edges, in the statistic's units."""
        return self.upper - self.lower

    @property
    def disclosures(self) -> tuple[str, ...]:
        """What must be shown wherever this band is shown.

        Modelled on :attr:`backend.backtest.artifact.RunArtifact.disclosures`:
        the caveat travels with the number rather than living in a document the
        reader has not opened.
        """
        return (
            SELECTION_CONTAMINATION_DISCLOSURE,
            _HORIZON_DISCLOSURE,
            _COST_NOTE,
            (
                f"Selection search size behind this band: {self.artefact.trials} trial(s) "
                f"(TESTING_LEDGER.md). Deflation for that search is backend.backtest.dsr's "
                f"job and is not applied here."
            ),
        )

    def detection_power(self, deterioration_sigmas: float) -> dict[str, float]:
        """Estimate the power and latency of the halt at a given deterioration.

        Under a normal approximation to the window statistic, a strategy whose
        true window statistic has fallen by ``d`` band standard deviations is
        caught in one evaluation with probability ``Z[z_alpha + d]``, where
        ``z_alpha = Z^-1[tail_mass]`` is negative. Expected evaluations to halt
        is one over that.

        **This is guidance, not a guarantee.** Three approximations, all in the
        optimistic direction: the pooled window distribution is not normal (a
        63-period Sharpe is skewed), a deteriorated strategy's dispersion is not
        the healthy one's, and the band edge is itself estimated. Treat the
        numbers as an order of magnitude for the power/latency trade-off, which
        is the decision they exist to inform (module docstring §2).

        Args:
            deterioration_sigmas: how far the statistic's centre has fallen, in
                units of :attr:`dispersion`. Must be finite and non-negative.

        Returns:
            ``power`` (probability of halting in one evaluation),
            ``expected_evaluations`` (``inf`` when power is 0),
            ``expected_periods`` and ``expected_years``.

        Raises:
            ExpectationBandError: if ``deterioration_sigmas`` is negative or not
                finite, or the band has zero dispersion (every window identical,
                so no shift is expressible in its units).
        """
        if not math.isfinite(deterioration_sigmas) or deterioration_sigmas < 0.0:
            msg = (
                f"deterioration_sigmas must be finite and non-negative; got "
                f"{deterioration_sigmas!r}"
            )
            raise ExpectationBandError(msg)
        if self.dispersion == 0.0:
            msg = (
                "the band has zero dispersion, so a deterioration cannot be expressed in its "
                "units; power is undefined rather than perfect"
            )
            raise ExpectationBandError(msg)
        z_alpha = standard_normal_ppf(self.policy.tail_mass)
        power = standard_normal_cdf(z_alpha + deterioration_sigmas)
        evaluations = math.inf if power <= 0.0 else 1.0 / power
        periods = evaluations * self.window_periods
        return {
            "deterioration_sigmas": deterioration_sigmas,
            "power": power,
            "expected_evaluations": evaluations,
            "expected_periods": periods,
            "expected_years": periods / self.artefact.periods_per_year,
        }

    @classmethod
    def from_paths(
        cls,
        paths: npt.ArrayLike,
        *,
        artefact: CPCVArtefactRef,
        policy: HaltPolicy | None = None,
        statistic: WindowStatistic = WindowStatistic.SHARPE,
        risk_free_rate: float = 0.0,
    ) -> ExpectationBand:
        """Cut a band from CPCV paths at the live evaluation horizon.

        Slides a window of :attr:`HaltPolicy.window_periods` along every path,
        computes ``statistic`` on each, pools them, and takes the empirical
        quantiles at :attr:`HaltPolicy.tail_mass` and its complement. Every
        input is real backtest output; nothing is fitted or simulated (I3).

        Args:
            paths: shape ``(n_paths, n_observations)`` **net-of-cost** per-period
                returns, from
                :meth:`~backend.backtest.cpcv.CPCVSplits.assemble_paths`.
            artefact: identity of the run these paths came from (I2). Its
                ``n_paths``/``n_observations`` must match ``paths``, so a band
                cannot be labelled with another run's identity.
            policy: cadence, budget and side. Defaults to :class:`HaltPolicy`.
            statistic: which statistic to band. Default
                :attr:`WindowStatistic.SHARPE`.
            risk_free_rate: per-period risk-free rate as a fraction, applied to
                both sides. Default 0.

        Returns:
            An :class:`ExpectationBand`.

        Raises:
            ExpectationBandError: if the paths are malformed, the artefact's
                declared shape disagrees with them, the window is longer than a
                path, a window statistic is undefined (a zero-dispersion window
                leaves a Sharpe ratio undefined), or the requested tail mass is
                finer than the sample can resolve.
        """
        matrix = _validated_paths(paths)
        band_policy = HaltPolicy() if policy is None else policy
        n_paths, n_observations = matrix.shape
        if (n_paths, n_observations) != (artefact.n_paths, artefact.n_observations):
            msg = (
                f"paths have shape {(n_paths, n_observations)} but the artefact declares "
                f"{(artefact.n_paths, artefact.n_observations)}. A band labelled with "
                f"another run's identity is worse than an unlabelled one (I2)."
            )
            raise ExpectationBandError(msg)
        window = band_policy.window_periods
        if window > n_observations:
            msg = (
                f"window_periods={window} exceeds the {n_observations}-period CPCV paths; "
                f"there is no matched-horizon window to cut"
            )
            raise ExpectationBandError(msg)

        independent = n_paths * (n_observations // window)
        required = band_policy.minimum_independent_windows
        if independent < required:
            msg = (
                f"tail mass {band_policy.tail_mass:.5f} per side needs at least "
                f"{math.ceil(required)} effectively independent windows to be a quantile; "
                f"{n_paths} path(s) of {n_observations} periods give {independent} "
                f"non-overlapping windows of {window}. The band edge would be the most "
                f"extreme order statistic available rather than a quantile, and halting on "
                f"it would be halting on that order statistic's own sampling error. Lengthen "
                f"the sample, shorten the window, or raise the annual false-halt budget — "
                f"widening the band to whatever the data supports is the same error one "
                f"level down."
            )
            raise ExpectationBandError(msg)

        values: list[float] = []
        for path_index in range(n_paths):
            row = matrix[path_index]
            for start in range(n_observations - window + 1):
                try:
                    values.append(
                        _window_statistic(
                            row[start : start + window],
                            statistic=statistic,
                            risk_free_rate=risk_free_rate,
                        )
                    )
                except ValueError as exc:
                    msg = (
                        f"the {statistic} of path {path_index} window [{start}, "
                        f"{start + window}) is undefined ({exc}). A band cannot be cut with "
                        f"windows dropped: dropping the ones the statistic refuses removes "
                        f"exactly the degenerate stretches and narrows the band."
                    )
                    raise ExpectationBandError(msg) from exc
        pooled = np.asarray(values, dtype=np.float64)
        dispersion = float(np.std(pooled, ddof=1)) if pooled.size >= _MIN_PATHS_FOR_SPREAD else 0.0
        return cls(
            statistic=statistic,
            policy=band_policy,
            artefact=artefact,
            lower=float(np.quantile(pooled, band_policy.tail_mass)),
            upper=float(np.quantile(pooled, 1.0 - band_policy.tail_mass)),
            median=float(np.median(pooled)),
            dispersion=dispersion,
            n_pooled_windows=int(pooled.size),
            n_independent_windows=int(independent),
            risk_free_rate=risk_free_rate,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the band as a JSON-safe mapping, disclosures included."""
        return {
            "statistic": str(self.statistic),
            "lower": self.lower,
            "upper": self.upper,
            "median": self.median,
            "dispersion": self.dispersion,
            "window_periods": self.window_periods,
            "n_pooled_windows": self.n_pooled_windows,
            "n_independent_windows": self.n_independent_windows,
            "risk_free_rate": self.risk_free_rate,
            "policy": self.policy.to_dict(),
            "artefact": self.artefact.to_dict(),
            "disclosures": list(self.disclosures),
            "power": [
                dict(self.detection_power(sigmas)) if self.dispersion > 0.0 else {}
                for sigmas in (1.0, 2.0, 3.0)
            ],
        }


@dataclass(frozen=True, slots=True)
class ExpectationComparison:
    """One completed live-versus-expected comparison.

    Deliberately has **no** ``validated``, ``passed`` or ``healthy`` attribute.
    The strongest thing this object can say about a live result inside the band
    is that no alarm fired, and it says exactly that
    (:attr:`~ExpectationBand.disclosures`, module docstring §0).

    Attributes:
        band: the band compared against, carrying its own artefact identity.
        live: the live window compared.
        value: the live window's statistic, in the band's units.
        percentile: the fraction of pooled backtest windows at or below
            ``value``, estimated by linear interpolation between the band's
            edges and median — a coarse position indicator for display, not a
            p-value.
        below_band: whether ``value`` is strictly below :attr:`ExpectationBand.lower`.
        above_band: whether ``value`` is strictly above :attr:`ExpectationBand.upper`.
        evaluated_at: when the comparison was made (UTC).
    """

    band: ExpectationBand
    live: LiveWindow
    value: float
    percentile: float
    below_band: bool
    above_band: bool
    evaluated_at: dt.datetime

    @property
    def within_band(self) -> bool:
        """Whether the live statistic sits between the edges, inclusive.

        Never read this as "the strategy is validated" — see
        :attr:`ExpectationBand.disclosures`.
        """
        return not (self.below_band or self.above_band)

    @property
    def breaches_halting_side(self) -> bool:
        """Whether the breach, if any, is on a side the policy halts on."""
        if self.below_band:
            return True
        return self.above_band and self.band.policy.side is HaltSide.BOTH

    @property
    def cost_note(self) -> str:
        """The I4 caveat about modelled-versus-realised costs, for the payload."""
        return _COST_NOTE

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the comparison as a JSON-safe mapping, disclosures included."""
        return {
            "value": self.value,
            "percentile": self.percentile,
            "below_band": self.below_band,
            "above_band": self.above_band,
            "within_band": self.within_band,
            "breaches_halting_side": self.breaches_halting_side,
            "evaluated_at": self.evaluated_at.isoformat(),
            "live_as_of": self.live.as_of.isoformat(),
            "live_n_periods": self.live.n_periods,
            "live_source": self.live.source,
            "live_cost_treatment": str(self.live.cost_treatment),
            "cost_note": self.cost_note,
            "band": self.band.to_dict(),
            "in_band_is_not_validation": SELECTION_CONTAMINATION_DISCLOSURE,
        }


def _percentile_position(value: float, band: ExpectationBand) -> float:
    """Locate ``value`` in the band by interpolating between its known points.

    A display aid, not a p-value: the band stores three order statistics
    (lower edge, median, upper edge) rather than the whole pooled sample, so
    the position between them is linear by assumption. Values outside the edges
    are reported at the edge mass, never extrapolated into a smaller tail
    probability the sample cannot support.

    Args:
        value: the live statistic.
        band: the band it is measured against.

    Returns:
        A fraction in ``[tail_mass, 1 - tail_mass]``.
    """
    tail = band.policy.tail_mass
    if value <= band.lower:
        return tail
    if value >= band.upper:
        return 1.0 - tail
    if value <= band.median:
        span = band.median - band.lower
        share = 0.0 if span == 0.0 else (value - band.lower) / span
        return tail + share * (0.5 - tail)
    span = band.upper - band.median
    share = 1.0 if span == 0.0 else (value - band.median) / span
    return 0.5 + share * (0.5 - tail)


def compare(
    *,
    band: ExpectationBand | None,
    live: LiveWindow | None,
    now: dt.datetime | None = None,
) -> ExpectationComparison:
    """Compare a live window against its expectation band.

    Raises on every condition under which the comparison cannot be made. It
    **never** returns a result meaning "unavailable", because the only shape
    that could take is a within-band verdict, which is the most dangerous wrong
    answer this system can produce (I3, module docstring §3).

    Args:
        band: the expectation band for the running strategy. ``None`` is an
            explicit "no band exists" and raises.
        live: the realised window. ``None`` raises.
        now: decision time (UTC). Defaults to the current instant. Used only for
            the record; staleness is enforced by :func:`decide`, which is the
            entry point that can act on it.

    Returns:
        A completed :class:`ExpectationComparison`.

    Raises:
        ComparisonUnavailableError: if either side is missing, the live window's
            length does not equal the band's horizon, or the live statistic is
            undefined.
        CostBasisError: if either side is not net of costs (I4).
    """
    if band is None:
        msg = (
            "no expectation band is available for the running strategy. A strategy trading "
            "without a band to be judged against is unmonitored, which halts rather than "
            "passes (I3)."
        )
        raise ComparisonUnavailableError(msg)
    if live is None:
        msg = (
            "no live return window was supplied. Nothing here manufactures one: an absent "
            "live sample is a halt, never an in-band verdict (I3)."
        )
        raise ComparisonUnavailableError(msg)
    if not band.artefact.cost_treatment.is_net or not live.cost_treatment.is_net:
        msg = (
            f"live-versus-expected requires both sides net of costs (I4): band is "
            f"{band.artefact.cost_treatment!r}, live is {live.cost_treatment!r}"
        )
        raise CostBasisError(msg)
    if live.n_periods != band.window_periods:
        msg = (
            f"the live window holds {live.n_periods} periods but the band was cut at "
            f"{band.window_periods}. Comparing a statistic to a band built at another "
            f"horizon is not conservative in either direction — the spread scales roughly "
            f"with 1/sqrt(periods), so a short window against a long band halts constantly "
            f"and a long window against a short band never halts at all."
        )
        raise ComparisonUnavailableError(msg)
    try:
        value = _window_statistic(
            live.returns, statistic=band.statistic, risk_free_rate=band.risk_free_rate
        )
    except ValueError as exc:
        msg = (
            f"the live window's {band.statistic} is undefined ({exc}). No number means no "
            f"comparison, and no comparison halts."
        )
        raise ComparisonUnavailableError(msg) from exc
    if not math.isfinite(value):
        msg = f"the live window's {band.statistic} is {value!r}, which cannot be banded"
        raise ComparisonUnavailableError(msg)
    return ExpectationComparison(
        band=band,
        live=live,
        value=value,
        percentile=_percentile_position(value, band),
        below_band=value < band.lower,
        above_band=value > band.upper,
        evaluated_at=dt.datetime.now(dt.UTC) if now is None else now,
    )


@dataclass(frozen=True, slots=True)
class HaltDecision:
    """The auto-halt verdict, which cannot express "continue, unchecked".

    :attr:`HaltAction.CONTINUE` is refused at construction unless a completed
    :class:`ExpectationComparison` is attached **and** it did not breach a
    halting side. That is what "fails closed" means structurally rather than by
    convention: there is no value of this type meaning "keep trading, the
    comparison did not happen" for any code path to return by accident.

    Attributes:
        action: continue or halt.
        cause: why, when halting. ``None`` exactly when continuing.
        detail: the reason in words, including the numbers behind it. Always
            populated — a continue says what it compared, a halt says what it
            found.
        comparison: the completed comparison, when there was one. ``None`` on a
            halt whose comparison never happened.
        stamp: the I2 stamp of the monitoring run that produced this decision.
        decided_at: when (UTC).
        as_of: the date of the live window's last observation, when known.
    """

    action: HaltAction
    cause: HaltCause | None
    detail: str
    comparison: ExpectationComparison | None
    stamp: ReproducibilityStamp
    decided_at: dt.datetime
    as_of: dt.date | None

    def __post_init__(self) -> None:
        """Refuse a decision that claims more than its evidence supports.

        Raises:
            ExpectationBandError: if ``CONTINUE`` carries no comparison, carries
                one that breached a halting side, or carries a cause; if
                ``HALT`` carries no cause; if the stamp is not a
                :class:`~backend.tracking.stamp.ReproducibilityStamp` (I2); or
                if ``detail`` is blank.
        """
        supplied_stamp: object = self.stamp
        if not isinstance(supplied_stamp, ReproducibilityStamp):
            msg = (
                f"stamp must be a ReproducibilityStamp, got {type(supplied_stamp).__name__}. "
                f"A halt decision is an operational event and must be regenerable (I2)."
            )
            raise ExpectationBandError(msg)
        supplied_detail: object = self.detail
        if not isinstance(supplied_detail, str) or not supplied_detail.strip():
            msg = "detail must say what was compared or what could not be"
            raise ExpectationBandError(msg)
        if self.action is HaltAction.HALT:
            if self.cause is None:
                msg = "a halt must name its cause; 'halted' with no cause cannot be reviewed"
                raise ExpectationBandError(msg)
            return
        if self.cause is not None:
            msg = f"a continue decision cannot carry a halt cause; got {self.cause!r}"
            raise ExpectationBandError(msg)
        if self.comparison is None:
            msg = (
                "CONTINUE requires a completed comparison. A decision to keep trading with "
                "no evidence is exactly the failure auto-halt exists to prevent, so it is "
                "unconstructible rather than merely discouraged (I3)."
            )
            raise ExpectationBandError(msg)
        if self.comparison.breaches_halting_side:
            msg = (
                f"CONTINUE was constructed with a comparison that breached the halting side "
                f"(value={self.comparison.value!r}, band=[{self.comparison.band.lower!r}, "
                f"{self.comparison.band.upper!r}], side={self.comparison.band.policy.side})"
            )
            raise ExpectationBandError(msg)

    @property
    def should_halt(self) -> bool:
        """Whether trading must stop."""
        return self.action is HaltAction.HALT

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the decision as a JSON-safe mapping."""
        return {
            "action": str(self.action),
            "cause": None if self.cause is None else str(self.cause),
            "detail": self.detail,
            "decided_at": self.decided_at.isoformat(),
            "as_of": None if self.as_of is None else self.as_of.isoformat(),
            "git_reference": self.stamp.git_reference,
            "data_version": self.stamp.data_version,
            "config_hash": self.stamp.config_hash,
            "seed": self.stamp.seed,
            "reproducible": self.stamp.reproducible,
            "comparison": None if self.comparison is None else self.comparison.to_dict(),
        }


def _halt(
    *,
    cause: HaltCause,
    detail: str,
    stamp: ReproducibilityStamp,
    decided_at: dt.datetime,
    as_of: dt.date | None,
    comparison: ExpectationComparison | None = None,
) -> HaltDecision:
    """Build a halt decision. The single constructor every failure path uses."""
    return HaltDecision(
        action=HaltAction.HALT,
        cause=cause,
        detail=detail,
        comparison=comparison,
        stamp=stamp,
        decided_at=decided_at,
        as_of=as_of,
    )


def decide(
    *,
    band: ExpectationBand | None,
    live: LiveWindow | None,
    stamp: ReproducibilityStamp,
    now: dt.datetime | None = None,
) -> HaltDecision:
    """Decide whether to keep trading, failing closed on every unavailability.

    The auto-halt entry point. It returns :attr:`HaltAction.CONTINUE` only when
    a comparison was actually performed and did not breach a halting side; every
    other outcome — including an unexpected exception raised anywhere inside the
    comparison — is a halt carrying its cause.

    Args:
        band: the expectation band, or ``None`` when none exists (halts).
        live: the realised window, or ``None`` when none exists (halts).
        stamp: the I2 stamp of this monitoring run. Required — every decision is
            an operational event and must be regenerable.
        now: decision time (UTC). Defaults to the current instant.

    Returns:
        A :class:`HaltDecision`. Never ``None``, and never a continue without
        evidence.

    Raises:
        ExpectationBandError: only if ``stamp`` is not a valid I2 stamp, which
            is a defect in the calling code rather than a monitoring condition —
            a decision that cannot be recorded must not be silently produced.
    """
    decided_at = dt.datetime.now(dt.UTC) if now is None else now
    as_of = live.as_of if live is not None else None
    try:
        if live is not None:
            age = live.age_days(decided_at.date())
            limit = band.policy.max_live_age_days if band is not None else DEFAULT_MAX_LIVE_AGE_DAYS
            if age > limit:
                return _halt(
                    cause=HaltCause.STALE_LIVE_DATA,
                    detail=(
                        f"the live window's last observation is {live.as_of.isoformat()}, "
                        f"{age} day(s) before the decision date, above the {limit}-day "
                        f"allowance. A monitor reading stale returns repeats its last "
                        f"verdict forever, so staleness halts rather than passes."
                    ),
                    stamp=stamp,
                    decided_at=decided_at,
                    as_of=as_of,
                )
        comparison = compare(band=band, live=live, now=decided_at)
    except CostBasisError as exc:
        return _halt(
            cause=HaltCause.COST_BASIS,
            detail=f"live-versus-expected is not net of costs on both sides (I4): {exc}",
            stamp=stamp,
            decided_at=decided_at,
            as_of=as_of,
        )
    except ComparisonUnavailableError as exc:
        return _halt(
            cause=HaltCause.COMPARISON_UNAVAILABLE,
            detail=(
                f"the live-versus-expected comparison could not be made, so trading is "
                f"halted rather than assumed healthy: {exc.reason}"
            ),
            stamp=stamp,
            decided_at=decided_at,
            as_of=as_of,
        )
    except Exception as exc:  # fail closed: an unexpected failure is still a halt
        return _halt(
            cause=HaltCause.INTERNAL_ERROR,
            detail=(
                f"the monitoring decision path raised {type(exc).__name__}: {exc}. A monitor "
                f"that crashed checked nothing, which is indistinguishable from the outside "
                f"from a monitor that found nothing wrong — so it halts."
            ),
            stamp=stamp,
            decided_at=decided_at,
            as_of=as_of,
        )

    if comparison.below_band:
        return _halt(
            cause=HaltCause.BELOW_EXPECTED_BAND,
            detail=(
                f"live {comparison.band.statistic} {comparison.value:.6g} over "
                f"{comparison.live.n_periods} periods to {comparison.live.as_of.isoformat()} "
                f"is below the expected band [{comparison.band.lower:.6g}, "
                f"{comparison.band.upper:.6g}] cut from CPCV artefact "
                f"{comparison.band.artefact.artefact_id!r}. That band was built on the "
                f"sample this strategy was selected on, so falling out of the bottom of it "
                f"is stronger evidence than the tail mass alone suggests."
            ),
            stamp=stamp,
            decided_at=decided_at,
            as_of=as_of,
            comparison=comparison,
        )
    if comparison.above_band and comparison.band.policy.side is HaltSide.BOTH:
        return _halt(
            cause=HaltCause.ABOVE_EXPECTED_BAND,
            detail=(
                f"live {comparison.band.statistic} {comparison.value:.6g} over "
                f"{comparison.live.n_periods} periods to {comparison.live.as_of.isoformat()} "
                f"is above the expected band [{comparison.band.lower:.6g}, "
                f"{comparison.band.upper:.6g}]. Outperforming a distribution built from the "
                f"same data is not good news: check position scale, missed costs and the "
                f"reconciliation before concluding the strategy is exceeding its backtest."
            ),
            stamp=stamp,
            decided_at=decided_at,
            as_of=as_of,
            comparison=comparison,
        )
    position = (
        "above the expected band, on a side this policy does not halt on (HaltSide.LOWER); "
        "P12.4 raises an alert for it instead, because an upside breach is usually an "
        "accounting or scale error rather than good fortune"
        if comparison.above_band
        else "inside the expected band"
    )
    return HaltDecision(
        action=HaltAction.CONTINUE,
        cause=None,
        detail=(
            f"live {comparison.band.statistic} {comparison.value:.6g} over "
            f"{comparison.live.n_periods} periods to {comparison.live.as_of.isoformat()} is "
            f"{position} [{comparison.band.lower:.6g}, {comparison.band.upper:.6g}]. This is "
            f"the absence of an alarm, not a validation: the band comes from the sample the "
            f"strategy was selected on ({comparison.band.artefact.trials} trial(s) in the "
            f"ledger)."
        ),
        comparison=comparison,
        stamp=stamp,
        decided_at=decided_at,
        as_of=as_of,
    )
