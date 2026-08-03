"""Synthetic-truth harness — the framework measured against a known answer (P10.6).

Directive §5, gate G10:

    *"Framework reproduces known results on synthetic data with injected signal
    of known strength — and correctly reports near-zero signal on pure noise.
    This second test is the critical one. If your framework finds alpha in
    random data, it is broken and everything downstream is fiction."*

:mod:`backend.tests.backtest.test_synthetic_truth` already tests the *statistics
layer* — CPCV, the Deflated Sharpe Ratio and PBO applied directly to return
matrices. This module tests the *framework*: it builds a seeded synthetic market,
drives :func:`backend.backtest.engine.run_backtest` over it end to end, and hands
the engine's own net-of-cost output to the same statistics. What it reports is
what the framework concluded, not what a shortcut through the framework
concluded.

It is a module rather than a test file because it is the regression instrument
for the validation stack: re-run it after any change to the engine, the cost
model, CPCV, DSR or PBO, and it will say whether the thing that tells you you are
wrong still works. :mod:`backend.tests.backtest.test_gate_g10` is one caller.

The data-generating process, and what "known strength" means
------------------------------------------------------------

``n_assets`` independent synthetic instruments. Asset *i*'s simple return in
period *t* is

    r[i, t] = a[i] + sigma * e[i, t],     e[i, t] ~ N(0, 1) iid

* ``sigma`` is ``idiosyncratic_volatility_bps_per_day``, in **basis points per
  period**.
* ``a[i]`` is drawn once per run from ``N(0, tau^2)``. **``tau`` —
  ``alpha_dispersion_bps_per_day``, the cross-sectional standard deviation of
  true expected return in basis points per period — is the injected signal
  strength.** ``tau = 0`` is the pure-noise mode: every asset has expected
  return exactly zero and there is nothing whatsoever to find.

The strategy is the obvious estimator: at each decision instant it averages each
asset's last ``lookback_periods`` **knowable** returns and holds
``sign(estimate) / n_assets`` — gross exposure exactly 1.0, no leverage (§1.1).
It reaches the data only through the snapshot the engine hands it, exactly like
any other strategy.

That combination has a closed form. With ``w = lookback_periods``, the estimator
is ``a[i] + eta[i]`` with ``eta ~ N(0, sigma^2 / w)`` exactly, so writing
``rho = tau / sqrt(tau^2 + sigma^2 / w)`` for the correlation between what the
strategy believes and what is true,

    E[per-period P&L] = sqrt(2/pi) * tau * rho
    Var[per-period P&L] = (tau^2 + sigma^2 - (2/pi) * tau^2 * rho^2) / n_assets

    SR = sqrt(2 * n_assets / pi) * tau * rho
         / sqrt(tau^2 + sigma^2 - (2/pi) * tau^2 * rho^2)      per period

(:meth:`SyntheticSpec.expected_gross_sharpe_per_period`). ``rho = 1`` gives the
oracle that knows every ``a[i]`` exactly
(:meth:`SyntheticSpec.oracle_gross_sharpe_per_period`), which no estimator can
beat. Both are derived from the DGP alone and reference nothing in
:mod:`backend.backtest`, so comparing a measured result against them is a real
test rather than a comparison of the framework with itself.

Because ``SR`` is proportional to ``tau * rho`` and ``rho`` is itself monotone in
``tau``, doubling the injected strength has a predicted — not merely "larger" —
effect. A framework with a scale error passes "signal is recovered" and fails
this.

Gross and net (invariant I4)
----------------------------

I4 forbids reporting a gross number, and nothing here reports one: the engine has
no gross path, and every :class:`SyntheticRun` carries the artifact's own
:class:`~backend.backtest.artifact.CostProvenance`.

But the noise direction has to be checked **before** costs as well as after. A
framework that finds alpha in random data and then loses it to the spread has
still found alpha in random data; it would pass a net-only test and fail the
moment costs were reduced. So the harness runs each configuration in two modes
(:class:`CostMode`): the reportable one charged the shipped uncalibrated cost
parameters, and :data:`ZERO_COST_PARAMS`, a **diagnostic** whose every parameter
is zero. The diagnostic is not a result and says so in three places — the mode on
the run, the ``calibration_basis`` string on the artifact's cost provenance, and
:attr:`SyntheticRun.disclosure`.

Synthetic data, permanently labelled (invariant I3)
---------------------------------------------------

I3 forbids presenting generated data as real. Three mechanisms, none of which
depends on anyone remembering:

* every dataset this module builds carries a ``data_version`` beginning
  :data:`SYNTHETIC_DATA_VERSION_PREFIX`, and that string travels into the I2
  reproducibility stamp of every artifact produced from it, so an artifact that
  escapes into a report still declares what it is;
* :func:`require_synthetic_source` refuses any source whose ``data_version``
  does not carry that prefix, and :class:`SyntheticMarket` calls it on
  construction — the harness cannot be pointed at a real feed;
* :attr:`SyntheticRun.is_research_finding` is ``False``, unconditionally, and
  :meth:`SyntheticRun.to_dict` leads with the disclosure.

Reproducibility (invariant I2)
------------------------------

Every run carries a :class:`backend.tracking.stamp.ReproducibilityStamp` — git
commit, dirty flag, data version, config hash, **and the seed** — alongside the
artifact's own stamp. Nothing in the harness draws from an unseeded source.

Units, stated once
------------------

Volatilities and drifts are **basis points per period**; returns and weights are
**fractions**; Sharpe ratios are **per period** unless a name says annualized;
``adv_usd`` and capital are **US dollars**. One period is one calendar day and
``periods_per_year`` defaults to 365, so the annualization factor and the cost
model's ACT/360 borrow accrual describe the same calendar. Nothing here claims to
be a trading calendar.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.backtest.artifact import (
    DEFAULT_CONFIDENCE_LEVEL,
    JsonValue,
    RunArtifact,
)
from backend.backtest.benchmark import BuyAndHoldSpec
from backend.backtest.cpcv import (
    CombinatorialPurgedCV,
    PathDistribution,
    path_sharpe_ratios,
)
from backend.backtest.dsr import DeflatedSharpeResult, deflated_sharpe_ratio_from_trials
from backend.backtest.engine import (
    BacktestConfig,
    InjectedMarketData,
    MarketSnapshot,
    Observation,
    PointInTimeMarketData,
    Strategy,
    run_backtest,
)
from backend.backtest.metrics import FloatArray, sharpe_ratio, standard_normal_cdf
from backend.backtest.pbo import PBOResult, probability_of_backtest_overfitting
from backend.costs.model import UNCALIBRATED_DEFAULTS, CostModelParams
from backend.tracking.stamp import (
    GitState,
    ReproducibilityStamp,
    canonical_config_hash,
    git_state,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "SYNTHETIC_DATA_VERSION_PREFIX",
    "SYNTHETIC_DISCLOSURE",
    "ZERO_COST_PARAMS",
    "ConfigurationSweep",
    "CostMode",
    "SharpeSweep",
    "StrategyCandidate",
    "SyntheticDataError",
    "SyntheticMarket",
    "SyntheticRun",
    "SyntheticSpec",
    "build_synthetic_market",
    "disjoint_asset_blocks",
    "format_report",
    "require_synthetic_source",
    "run_synthetic_backtest",
    "sweep_configurations",
    "sweep_seeds",
]

SYNTHETIC_DATA_VERSION_PREFIX: Final = "SYNTHETIC-NOT-A-RESULT"
"""Prefix on every dataset identifier this module produces.

It is the first thing in the ``data_version`` component of the I2 stamp, so an
artifact built from harness data announces that it is synthetic wherever it is
displayed, logged or stored — including in places nobody anticipated.
"""

SYNTHETIC_DISCLOSURE: Final = (
    "SYNTHETIC DATA — GENERATED, NOT OBSERVED. Every number here comes from a "
    "seeded pseudo-random data-generating process with a known answer, and exists "
    "only to test whether the validation framework reports that answer. It is not "
    "a research finding, describes no security, and must never be presented as a "
    "backtest result (directive invariant I3, §9.1)."
)
"""Statement attached to every harness result, in text and in ``to_dict``."""

ZERO_COST_PARAMS: Final = CostModelParams(
    half_spread_bps=0.0,
    commission_bps=0.0,
    impact_coefficient=0.0,
    default_daily_volatility_bps=0.0,
    borrow_rate_bps_per_year=0.0,
    uncalibrated=True,
    calibration_basis=(
        "ZERO-COST DIAGNOSTIC — every cost parameter deliberately set to zero. "
        "Figures produced with these parameters are GROSS and directive invariant "
        "I4 forbids reporting them. They exist for one purpose: to check whether "
        "the framework finds alpha in pure noise *before* costs, because a "
        "framework that does and then loses it to the spread is still broken."
    ),
)
"""Cost parameters for the gross diagnostic. Never for a reported figure.

``uncalibrated`` stays ``True``: these are not measurements, and the
:class:`~backend.costs.model.CostModelParams` calibration fence would refuse them
if they claimed to be (D-013).
"""

_BPS_PER_UNIT: Final = 10_000.0
_ASSET_PREFIX: Final = "SYNTH"
_BENCHMARK_ASSET: Final = "SYNTH-INDEX"
_TWO_OVER_PI: Final = 2.0 / math.pi


class SyntheticDataError(RuntimeError):
    """Raised when synthetic and real data are about to be confused for each other.

    Fatal in both directions. Pointing the harness at a real feed would produce a
    "gate passed" that says nothing about the framework; letting a harness result
    out as a research finding would put a fabricated number in front of a human.
    Directive §9.1 forbids the second and invariant I3 forbids both.
    """


class CostMode(StrEnum):
    """Which cost parameters a harness run is charged.

    Attributes:
        NET_OF_MODELLED_COSTS: the shipped conservative uncalibrated parameters
            (:data:`~backend.costs.model.UNCALIBRATED_DEFAULTS`). The only mode
            whose numbers are of the kind invariant I4 permits reporting.
        ZERO_COST_DIAGNOSTIC: :data:`ZERO_COST_PARAMS`. Gross. A diagnostic that
            answers "does the framework find an edge in noise before costs?",
            which is a question the net figure cannot answer.
    """

    NET_OF_MODELLED_COSTS = "net-of-modelled-costs"
    ZERO_COST_DIAGNOSTIC = "zero-cost-diagnostic"

    @property
    def cost_params(self) -> CostModelParams:
        """Return the cost parameters this mode charges."""
        if self is CostMode.ZERO_COST_DIAGNOSTIC:
            return ZERO_COST_PARAMS
        return UNCALIBRATED_DEFAULTS

    @property
    def is_reportable_basis(self) -> bool:
        """Return whether figures from this mode are net of modelled costs (I4)."""
        return self is CostMode.NET_OF_MODELLED_COSTS


@cache
def _harness_git_state() -> GitState:
    """Return the repository's commit and dirty flag, read once per process.

    Cached because a sweep stamps hundreds of runs and every one of them is
    produced by the same working tree; re-reading git per run would cost more
    than the simulation. The cache is per process, so a stamp taken after an edit
    within the same process reports the state at first call — stated here rather
    than discovered later.

    Returns:
        The :class:`~backend.tracking.stamp.GitState` for this checkout.

    Raises:
        GitStateUnavailableError: if the commit cannot be determined. Not
            downgraded to a placeholder: an unstamped result is not reproducible.
    """
    return git_state()


# ---------------------------------------------------------------------------
# The data-generating process
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticSpec:
    """The data-generating process, its seed, and its closed-form truth.

    Attributes:
        seed: seed for :func:`numpy.random.default_rng`. Non-negative. Recorded
            on the I2 stamp of every run built from this spec.
        alpha_dispersion_bps_per_day: **the injected signal strength** — the
            cross-sectional standard deviation of true per-period expected
            return, in basis points per period. ``0.0`` is the pure-noise mode.
        idiosyncratic_volatility_bps_per_day: per-asset return standard
            deviation in basis points per period. Must be strictly positive.
        n_assets: number of synthetic instruments. At least 2.
        n_periods: number of simulated return periods ``T``. The calendar holds
            ``T + 1`` instants.
        lookback_periods: the estimator's window ``w``, in periods. The market
            carries ``w`` extra bars before the calendar starts, so the strategy
            has a full window at the very first decision and no period is traded
            on a short estimate.
        benchmark_volatility_bps_per_day: the buy-and-hold index's return
            standard deviation, basis points per period.
        benchmark_drift_bps_per_day: the index's expected return, basis points
            per period. Zero by default: a drifting index would dominate the
            active return with its own trend and say nothing about the strategy.
        adv_usd: average daily dollar volume carried on every observation, US
            dollars. Large by default so that square-root impact is a small part
            of the cost rather than the whole of it; the harness is about the
            statistics, and impact is exercised by the P9.3 suite.
        initial_capital_usd: starting capital, US dollars.
        periods_per_year: annualization factor. 365 because one period is one
            calendar day here.
        bootstrap_resamples: bootstrap replicates behind each artifact interval.
            200 rather than the artifact default of 1000: a sweep aggregates
            across seeds and never reads a single run's interval, and 1000 would
            add about 40% to the harness's runtime for precision nothing here
            consumes. A single run wanting a displayable interval should raise
            it.
        confidence_level: central mass of each artifact interval.
        lookahead_leak_fraction: **a deliberate defect injector, zero for any
            honest run.** When non-zero the strategy adds
            ``fraction * r[i, t+1] / lookback_periods`` to its estimate — the
            next period's return leaking in with the weight of a single bar of
            the window, which is what an off-by-one in a rolling mean does. It
            exists so a gate can prove it fails on a leaky framework; a gate that
            cannot do that proves nothing. Any run with a non-zero value has
            :attr:`SyntheticRun.is_honest` ``False``.
    """

    seed: int
    alpha_dispersion_bps_per_day: float = 0.0
    idiosyncratic_volatility_bps_per_day: float = 100.0
    n_assets: int = 8
    n_periods: int = 504
    lookback_periods: int = 126
    benchmark_volatility_bps_per_day: float = 80.0
    benchmark_drift_bps_per_day: float = 0.0
    adv_usd: float = 5e9
    initial_capital_usd: float = 1_000_000.0
    periods_per_year: float = 365.0
    bootstrap_resamples: int = 200
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    lookahead_leak_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate the process parameters.

        Raises:
            ValueError: if the seed is not a non-negative int (``bool`` refused,
                since ``True`` would silently record as seed 1), if any size is
                too small to define the statistics the harness computes, if a
                volatility, ADV, capital or annualization factor is not strictly
                positive, if the injected strength is negative or non-finite, or
                if the leak fraction falls outside ``[0, 1]``.
        """
        seed: object = self.seed
        if isinstance(seed, bool) or not isinstance(seed, int) or self.seed < 0:
            msg = f"seed must be a non-negative int (bool refused); got {self.seed!r}"
            raise ValueError(msg)
        if self.n_assets < 2:
            msg = f"n_assets must be at least 2 for a cross-sectional book; got {self.n_assets}"
            raise ValueError(msg)
        if self.n_periods < 32:
            msg = (
                f"n_periods must be at least 32 for a Sharpe ratio to mean anything; "
                f"got {self.n_periods}"
            )
            raise ValueError(msg)
        if self.lookback_periods < 2:
            msg = f"lookback_periods must be at least 2; got {self.lookback_periods}"
            raise ValueError(msg)
        for name in ("idiosyncratic_volatility_bps_per_day", "benchmark_volatility_bps_per_day"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                msg = f"{name} must be finite and strictly positive; got {value!r}"
                raise ValueError(msg)
        for name in ("adv_usd", "initial_capital_usd", "periods_per_year"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                msg = f"{name} must be finite and strictly positive; got {value!r}"
                raise ValueError(msg)
        if (
            not math.isfinite(self.alpha_dispersion_bps_per_day)
            or self.alpha_dispersion_bps_per_day < 0.0
        ):
            msg = (
                "alpha_dispersion_bps_per_day is the injected signal strength and must be "
                f"finite and non-negative; got {self.alpha_dispersion_bps_per_day!r}"
            )
            raise ValueError(msg)
        if not math.isfinite(self.benchmark_drift_bps_per_day):
            drift = self.benchmark_drift_bps_per_day
            msg = f"benchmark_drift_bps_per_day must be finite; got {drift!r}"
            raise ValueError(msg)
        if not 0.0 <= self.lookahead_leak_fraction <= 1.0:
            msg = (
                "lookahead_leak_fraction is a defect injector and must lie in [0, 1]; "
                f"got {self.lookahead_leak_fraction!r}"
            )
            raise ValueError(msg)
        if self.bootstrap_resamples < 2:
            msg = f"bootstrap_resamples must be at least 2; got {self.bootstrap_resamples}"
            raise ValueError(msg)

    # -- the injected truth, computed from the process and nothing else ------

    @property
    def is_pure_noise(self) -> bool:
        """Return whether no signal at all was injected (``tau == 0``)."""
        return self.alpha_dispersion_bps_per_day == 0.0

    @property
    def injects_lookahead(self) -> bool:
        """Return whether this spec deliberately leaks the future into the signal."""
        return self.lookahead_leak_fraction != 0.0

    @property
    def alpha_dispersion(self) -> float:
        """Return ``tau`` as a fraction per period rather than basis points."""
        return self.alpha_dispersion_bps_per_day / _BPS_PER_UNIT

    @property
    def idiosyncratic_volatility(self) -> float:
        """Return ``sigma`` as a fraction per period rather than basis points."""
        return self.idiosyncratic_volatility_bps_per_day / _BPS_PER_UNIT

    @property
    def estimator_standard_error(self) -> float:
        """Return ``sigma / sqrt(w)``: the noise in one asset's alpha estimate.

        A fraction per period. The window mean of ``w`` iid normal returns has
        exactly this standard deviation about the asset's true expected return,
        which is what makes :attr:`signal_fidelity` exact rather than asymptotic.
        """
        return self.idiosyncratic_volatility / math.sqrt(self.lookback_periods)

    @property
    def signal_fidelity(self) -> float:
        """Return ``rho``, the correlation between the estimate and the truth.

        ``rho = tau / sqrt(tau^2 + sigma^2 / w)``, dimensionless in ``[0, 1)``.
        It is the entire difference between what this strategy can achieve and
        what an oracle holding the true alphas would: recovered Sharpe is the
        oracle's, multiplied by ``rho``. Zero in the pure-noise mode, where there
        is nothing for an estimate to be correlated with.
        """
        if self.is_pure_noise:
            return 0.0
        tau = self.alpha_dispersion
        error = self.estimator_standard_error
        return tau / math.hypot(tau, error)

    def _gross_sharpe(self, fidelity: float, n_held: int) -> float:
        """Return the closed-form per-period Sharpe ratio at a given fidelity.

        Args:
            fidelity: correlation between the position-driving estimate and the
                asset's true expected return. ``1.0`` is the oracle.
            n_held: how many independent assets the book holds.

        Returns:
            The per-period Sharpe ratio, before costs.
        """
        tau = self.alpha_dispersion
        sigma = self.idiosyncratic_volatility
        expected = math.sqrt(_TWO_OVER_PI) * tau * fidelity
        variance = (tau**2 + sigma**2 - _TWO_OVER_PI * (tau * fidelity) ** 2) / n_held
        return expected / math.sqrt(variance)

    def expected_gross_sharpe_per_period(self, *, n_held: int | None = None) -> float:
        """Return the strategy's closed-form per-period Sharpe ratio, before costs.

        Derived in this module's docstring from the data-generating process
        alone. This is the number a correct framework must recover from a
        zero-cost run, up to sampling error; a net-of-cost run must land below it
        by the cost drag and never above it.

        Args:
            n_held: how many assets the book holds. Defaults to ``n_assets``;
                pass the subset size for a candidate that trades fewer.

        Returns:
            Per-period Sharpe ratio (multiply by ``sqrt(periods_per_year)`` to
            annualize). Exactly ``0.0`` in the pure-noise mode.
        """
        return self._gross_sharpe(self.signal_fidelity, n_held or self.n_assets)

    def oracle_gross_sharpe_per_period(self, *, n_held: int | None = None) -> float:
        """Return the per-period Sharpe ratio of a book that knows every true alpha.

        The ceiling. An out-of-sample result above it is not a better strategy,
        it is a leak: no estimate is more aligned with the outcome than the truth.

        Args:
            n_held: how many assets the book holds. Defaults to ``n_assets``.

        Returns:
            Per-period Sharpe ratio, before costs.
        """
        return self._gross_sharpe(1.0, n_held or self.n_assets)

    def annualize(self, per_period: float) -> float:
        """Return ``per_period * sqrt(periods_per_year)``.

        Args:
            per_period: a per-period Sharpe ratio.

        Returns:
            The annualized Sharpe ratio. Dimensionless either way; the square
            root is the only conversion, and stating it here keeps every caller
            from picking its own.
        """
        return per_period * math.sqrt(self.periods_per_year)

    # -- identity -----------------------------------------------------------

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the process parameters as a JSON-safe mapping.

        This is what the config hash on the I2 stamp is computed over, so two
        harness runs share a hash exactly when they describe the same process.
        """
        return {
            "harness": "backend.backtest.synthetic",
            "task": "P10.6",
            "seed": int(self.seed),
            "alpha_dispersion_bps_per_day": float(self.alpha_dispersion_bps_per_day),
            "idiosyncratic_volatility_bps_per_day": float(
                self.idiosyncratic_volatility_bps_per_day
            ),
            "n_assets": int(self.n_assets),
            "n_periods": int(self.n_periods),
            "lookback_periods": int(self.lookback_periods),
            "benchmark_volatility_bps_per_day": float(self.benchmark_volatility_bps_per_day),
            "benchmark_drift_bps_per_day": float(self.benchmark_drift_bps_per_day),
            "adv_usd": float(self.adv_usd),
            "initial_capital_usd": float(self.initial_capital_usd),
            "periods_per_year": float(self.periods_per_year),
            "lookahead_leak_fraction": float(self.lookahead_leak_fraction),
        }

    @property
    def data_version(self) -> str:
        """Return the dataset identifier this process produces.

        Begins with :data:`SYNTHETIC_DATA_VERSION_PREFIX` and carries the seed
        and the injected strength in the clear, followed by a digest of the whole
        parameter set. Contains no pipe or newline, so it survives a
        ``TESTING_LEDGER.md`` cell intact.
        """
        digest = canonical_config_hash(self.to_dict())[:16]
        return (
            f"{SYNTHETIC_DATA_VERSION_PREFIX}:p10.6"
            f":seed={self.seed}"
            f":alpha={self.alpha_dispersion_bps_per_day:g}bps-per-period"
            f":leak={self.lookahead_leak_fraction:g}"
            f":{digest}"
        )


def require_synthetic_source(data: PointInTimeMarketData) -> None:
    """Refuse any market data this harness did not generate (invariant I3).

    The harness exists to measure a framework against a known answer. Pointed at
    a real feed it would measure nothing — there is no known answer — while
    producing output that looks exactly like a passing gate. So the source is
    checked rather than assumed.

    Args:
        data: the point-in-time source about to be simulated over.

    Raises:
        SyntheticDataError: if ``data.data_version`` does not begin with
            :data:`SYNTHETIC_DATA_VERSION_PREFIX`.
    """
    version = data.data_version
    if not version.startswith(SYNTHETIC_DATA_VERSION_PREFIX):
        msg = (
            f"the synthetic-truth harness refuses data_version {version!r}: it only runs on "
            f"data it generated itself, which is tagged {SYNTHETIC_DATA_VERSION_PREFIX!r}. "
            "Running it against a real feed would produce a gate result with no known "
            "answer behind it (invariant I3)."
        )
        raise SyntheticDataError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class SyntheticMarket:
    """One realization of the process: a calendar, a source, and the hidden truth.

    Construct it with :func:`build_synthetic_market` rather than directly; the
    constructor validates but does not generate.

    Attributes:
        spec: the process that produced this realization.
        calendar: ``n_periods + 1`` timezone-aware UTC instants. Observations
            exist before ``calendar[0]`` so the estimator starts with a full
            window.
        data: the point-in-time source, tagged synthetic.
        assets: the tradeable synthetic instruments, in name order.
        benchmark_asset: the buy-and-hold index's name.
        true_alpha_bps_per_day: each asset's realized draw of ``a[i]``, in basis
            points per period. This is the answer the framework is being asked to
            find; it is deliberately never given to a strategy.
        forward_returns: for each asset, the return of the period *following*
            each decision instant, keyed by decision instant. Present so the
            harness can build the deliberately-leaky control strategy that proves
            a gate is not vacuous, and reachable only through
            :meth:`strategy` — no honest strategy ever sees it, and the engine
            would not hand it over.
    """

    spec: SyntheticSpec
    calendar: tuple[dt.datetime, ...]
    data: InjectedMarketData
    assets: tuple[str, ...]
    benchmark_asset: str
    true_alpha_bps_per_day: Mapping[str, float]
    forward_returns: Mapping[str, Mapping[dt.datetime, float]] = field(repr=False)

    def __post_init__(self) -> None:
        """Refuse a market holding anything but this module's own synthetic data.

        Raises:
            SyntheticDataError: if the source is not tagged synthetic.
            ValueError: if the calendar length disagrees with the spec.
        """
        require_synthetic_source(self.data)
        if len(self.calendar) != self.spec.n_periods + 1:
            msg = (
                f"calendar holds {len(self.calendar)} instants but the spec asks for "
                f"{self.spec.n_periods + 1} (n_periods + 1)"
            )
            raise ValueError(msg)

    def benchmark(self) -> BuyAndHoldSpec:
        """Return the buy-and-hold baseline: the whole book in the synthetic index."""
        return BuyAndHoldSpec(
            weights={self.benchmark_asset: 1.0},
            description=(
                f"buy-and-hold of {self.benchmark_asset}, a SYNTHETIC zero-drift index "
                "generated by backend.backtest.synthetic — not a market index"
            ),
        )

    def conditional_gross_sharpe_per_period(
        self,
        *,
        assets: Sequence[str] | None = None,
        lookback_periods: int | None = None,
    ) -> float:
        """Return the closed-form Sharpe ratio **for the alphas this seed drew**.

        :meth:`SyntheticSpec.expected_gross_sharpe_per_period` averages over the
        draw of ``a``; this conditions on the one that happened. Both come from
        the data-generating process and neither consults
        :mod:`backend.backtest`, but the conditional form predicts *this run*
        rather than the average of infinitely many, which removes the largest
        source of seed-to-seed scatter and makes a tighter claim possible.

        With ``s = sigma / sqrt(w)``, the sign the strategy takes on asset ``i``
        is ``sign(a[i] + eta)`` with ``eta ~ N(0, s^2)``, so its expected tilt is
        ``2 * Phi(a[i] / s) - 1`` and

            E[P&L | a]   = (1/n) * sum_i (2 Phi(a[i]/s) - 1) * a[i]
            Var[P&L | a] = (1/n^2) * sum_i (a[i]^2 + sigma^2 - (tilt_i a[i])^2)

        Args:
            assets: the instruments the book holds. Defaults to all of them.
            lookback_periods: the estimator window. Defaults to the spec's.

        Returns:
            The per-period Sharpe ratio before costs, for this realization.
            Exactly ``0.0`` in the pure-noise mode, where every ``a[i]`` is zero.
        """
        held = tuple(assets) if assets is not None else self.assets
        window = self.spec.lookback_periods if lookback_periods is None else lookback_periods
        sigma = self.spec.idiosyncratic_volatility
        error = sigma / math.sqrt(window)
        total_mean = 0.0
        total_variance = 0.0
        for asset in held:
            alpha = self.true_alpha_bps_per_day[asset] / _BPS_PER_UNIT
            tilt = 2.0 * standard_normal_cdf(alpha / error) - 1.0
            contribution = tilt * alpha
            total_mean += contribution
            total_variance += alpha**2 + sigma**2 - contribution**2
        count = len(held)
        return (total_mean / count) / math.sqrt(total_variance / count**2)

    def strategy(
        self,
        *,
        assets: Sequence[str] | None = None,
        lookback_periods: int | None = None,
        lookahead_leak_fraction: float | None = None,
    ) -> Strategy:
        """Build the trailing-mean sign strategy over this market.

        At each decision instant the strategy averages each held asset's last
        ``lookback`` **knowable** returns and takes ``sign(estimate) / n_held``.
        Gross exposure is exactly 1.0, so the run never breaches the engine's
        no-leverage cap. The only data it reads is the snapshot the engine hands
        it.

        Args:
            assets: which instruments to hold. Defaults to all of them; a subset
                is how :func:`sweep_configurations` builds exchangeable candidate
                configurations.
            lookback_periods: window override. Defaults to the spec's.
            lookahead_leak_fraction: defect injector, defaulting to the spec's.
                Non-zero adds ``fraction * next_period_return / lookback`` to
                every estimate — one bar of the window replaced by a bar from the
                future, which is exactly what an off-by-one in a rolling window
                does. Used only to prove a gate fails on a leaky framework.

        Returns:
            A :class:`~backend.backtest.engine.Strategy`.

        Raises:
            ValueError: if ``assets`` is empty or names an instrument this market
                does not hold, or if the leak fraction is outside ``[0, 1]``.
        """
        held = tuple(assets) if assets is not None else self.assets
        if not held:
            msg = "a strategy must hold at least one asset"
            raise ValueError(msg)
        unknown = sorted(set(held) - set(self.assets))
        if unknown:
            msg = f"this synthetic market has no assets named {unknown}"
            raise ValueError(msg)
        window = self.spec.lookback_periods if lookback_periods is None else lookback_periods
        if window < 2:
            msg = f"lookback_periods must be at least 2; got {window}"
            raise ValueError(msg)
        leak = (
            self.spec.lookahead_leak_fraction
            if lookahead_leak_fraction is None
            else lookahead_leak_fraction
        )
        if not 0.0 <= leak <= 1.0:
            msg = f"lookahead_leak_fraction must lie in [0, 1]; got {leak!r}"
            raise ValueError(msg)
        weight = 1.0 / len(held)
        forward = self.forward_returns

        def _decide(snapshot: MarketSnapshot) -> Mapping[str, float]:
            """Return sign-of-trailing-mean weights for one decision instant."""
            targets: dict[str, float] = {}
            for asset in held:
                history = snapshot.history(asset)[-window:]
                estimate = sum(item.total_return for item in history) / len(history)
                if leak:
                    future = forward[asset].get(snapshot.as_of)
                    if future is None:
                        msg = (
                            f"no period follows {snapshot.as_of.isoformat()}, so the leak "
                            "instrument has nothing to leak. The engine handed a strategy a "
                            "snapshot at an instant that is not a decision instant, which is "
                            "itself a defect."
                        )
                        raise SyntheticDataError(msg)
                    estimate += leak * future / window
                targets[asset] = weight if estimate >= 0.0 else -weight
            return targets

        return _decide


def build_synthetic_market(spec: SyntheticSpec) -> SyntheticMarket:
    """Generate one seeded realization of the data-generating process.

    Draw order is fixed and the noise is drawn independently of the alphas, so
    two specs differing **only** in ``alpha_dispersion_bps_per_day`` share every
    noise draw at the same seed. That is what makes "same seeds, same sample, the
    only change is the injected signal" a valid comparison rather than two
    unrelated experiments.

    Args:
        spec: the process to realize.

    Returns:
        A :class:`SyntheticMarket`.
    """
    generator = np.random.default_rng(spec.seed)
    n_bars = spec.lookback_periods + spec.n_periods
    assets = tuple(f"{_ASSET_PREFIX}-{index:02d}" for index in range(spec.n_assets))

    standardized_alpha = generator.standard_normal(spec.n_assets)
    shocks = generator.standard_normal((spec.n_assets, n_bars))
    benchmark_shocks = generator.standard_normal(n_bars)

    alpha = standardized_alpha * spec.alpha_dispersion
    returns = alpha[:, None] + spec.idiosyncratic_volatility * shocks
    benchmark_returns = (
        spec.benchmark_drift_bps_per_day / _BPS_PER_UNIT
        + spec.benchmark_volatility_bps_per_day / _BPS_PER_UNIT * benchmark_shocks
    )

    origin = dt.datetime(2015, 1, 1, tzinfo=dt.UTC)
    bar_dates = tuple(origin + dt.timedelta(days=index) for index in range(n_bars))
    calendar = bar_dates[spec.lookback_periods - 1 :]

    observations: dict[str, tuple[Observation, ...]] = {}
    forward: dict[str, Mapping[dt.datetime, float]] = {}
    for index, asset in enumerate(assets):
        series = returns[index]
        observations[asset] = _observations(
            bar_dates, series, adv_usd=spec.adv_usd, volatility_bps=spec.idiosyncratic_volatility
        )
        forward[asset] = {
            calendar[position]: float(series[spec.lookback_periods + position])
            for position in range(spec.n_periods)
        }
    observations[_BENCHMARK_ASSET] = _observations(
        bar_dates,
        benchmark_returns,
        adv_usd=spec.adv_usd,
        volatility_bps=spec.benchmark_volatility_bps_per_day / _BPS_PER_UNIT,
    )

    return SyntheticMarket(
        spec=spec,
        calendar=calendar,
        data=InjectedMarketData(observations=observations, data_version=spec.data_version),
        assets=assets,
        benchmark_asset=_BENCHMARK_ASSET,
        true_alpha_bps_per_day={
            asset: float(alpha[index] * _BPS_PER_UNIT) for index, asset in enumerate(assets)
        },
        forward_returns=forward,
    )


def _observations(
    dates: tuple[dt.datetime, ...],
    returns: FloatArray,
    *,
    adv_usd: float,
    volatility_bps: float,
) -> tuple[Observation, ...]:
    """Build one observation per date, published at the close of its own period.

    Args:
        dates: period-end instants, timezone-aware UTC and strictly increasing.
        returns: one simple return per date, as a fraction.
        adv_usd: average daily dollar volume to carry, US dollars.
        volatility_bps: daily volatility as a **fraction**, converted to the
            basis points the observation records.

    Returns:
        The observations, oldest first.
    """
    knowledge_lag = dt.timedelta(0)
    return tuple(
        Observation(
            date=date,
            knowledge_time=date + knowledge_lag,
            total_return=float(value),
            adv_usd=adv_usd,
            daily_volatility_bps=volatility_bps * _BPS_PER_UNIT,
        )
        for date, value in zip(dates, returns, strict=True)
    )


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class SyntheticRun:
    """What the framework concluded about one seeded synthetic sample.

    Attributes:
        spec: the process, including the seed and the injected strength.
        cost_mode: which cost parameters the run was charged.
        stamp: the I2 reproducibility stamp — commit, dirty flag, synthetic data
            version, config hash, and the seed.
        artifact: the engine's own artifact, carrying both track records, the
            benchmark comparison, intervals on every metric, and the cost
            provenance.
        held_assets: the instruments the strategy traded.
        lookback_periods: the estimator window the strategy used.
        predicted_gross_sharpe_per_period: the closed-form per-period Sharpe
            ratio **for the alphas this seed actually drew**, from
            :meth:`SyntheticMarket.conditional_gross_sharpe_per_period`. Recorded
            at run time because the market is discarded afterwards, and because a
            prediction stored beside its measurement cannot be recomputed later
            against a different definition.
    """

    spec: SyntheticSpec
    cost_mode: CostMode
    stamp: ReproducibilityStamp
    artifact: RunArtifact
    held_assets: tuple[str, ...]
    lookback_periods: int
    predicted_gross_sharpe_per_period: float

    @property
    def is_research_finding(self) -> bool:
        """Return ``False``. Always, and by construction (invariant I3).

        A property rather than a comment so that any consumer deciding whether to
        display a number has something to branch on, and so that the answer
        cannot drift.
        """
        return False

    @property
    def is_honest(self) -> bool:
        """Return whether this run's strategy was free of injected lookahead.

        ``False`` marks a control run built by :attr:`SyntheticSpec.
        lookahead_leak_fraction` to prove a gate is not vacuous. Its numbers
        describe a deliberately broken framework and nothing else.
        """
        return not self.spec.injects_lookahead

    @property
    def disclosure(self) -> str:
        """Return the statement that must accompany every figure from this run."""
        parts = [SYNTHETIC_DISCLOSURE]
        if not self.cost_mode.is_reportable_basis:
            parts.append(self.cost_mode.cost_params.calibration_basis)
        if not self.is_honest:
            parts.append(
                "LOOKAHEAD DELIBERATELY INJECTED — this run's strategy was fed "
                f"{self.spec.lookahead_leak_fraction:g} of the next period's return. It is a "
                "control for the gate's non-vacuity, not a strategy."
            )
        return " ".join(parts)

    @property
    def returns(self) -> FloatArray:
        """Return the strategy's realized per-period returns, as fractions.

        Net of modelled costs when :attr:`cost_mode` is
        :attr:`CostMode.NET_OF_MODELLED_COSTS`; gross when it is the zero-cost
        diagnostic, which is the whole point of that mode and why its figures are
        not reportable.
        """
        return self.artifact.strategy.equity.net_returns

    @property
    def sharpe_per_period(self) -> float:
        """Return the realized per-period Sharpe ratio of :attr:`returns`."""
        return sharpe_ratio(self.returns, periods_per_year=None, ddof=1)

    @property
    def sharpe_annualized(self) -> float:
        """Return the realized Sharpe ratio annualized by the spec's factor."""
        return self.spec.annualize(self.sharpe_per_period)

    @property
    def mean_return_bps_per_period(self) -> float:
        """Return the realized mean per-period return in basis points."""
        return float(np.mean(self.returns)) * _BPS_PER_UNIT

    @property
    def turnover_per_period(self) -> float:
        """Return traded notional per rebalance as a fraction of portfolio value."""
        return self.artifact.strategy.summary.accounting.turnover_per_rebalance

    @property
    def total_cost_bps_of_capital(self) -> float:
        """Return every dollar of cost paid, in basis points of initial capital."""
        accounting = self.artifact.strategy.summary.accounting
        return accounting.total_cost_usd / accounting.initial_capital_usd * _BPS_PER_UNIT

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the run as a JSON-safe mapping, disclosure first.

        The disclosure leads and :attr:`is_research_finding` is present, so a
        payload that reaches a UI or a log carries its own refusal to be read as
        a result.
        """
        return {
            "disclosure": self.disclosure,
            "is_research_finding": self.is_research_finding,
            "is_honest": self.is_honest,
            "cost_mode": str(self.cost_mode),
            "spec": self.spec.to_dict(),
            "reproducibility": {
                "git_commit": self.stamp.git_commit,
                "git_dirty": self.stamp.git_dirty,
                "data_version": self.stamp.data_version,
                "config_hash": self.stamp.config_hash,
                "seed": self.stamp.seed,
            },
            "held_assets": list(self.held_assets),
            "lookback_periods": int(self.lookback_periods),
            "measured": {
                "sharpe_per_period": self.sharpe_per_period,
                "sharpe_annualized": self.sharpe_annualized,
                "mean_return_bps_per_period": self.mean_return_bps_per_period,
                "turnover_per_period": self.turnover_per_period,
                "total_cost_bps_of_capital": self.total_cost_bps_of_capital,
            },
            "predicted": {
                "conditional_gross_sharpe_per_period": self.predicted_gross_sharpe_per_period,
                "expected_gross_sharpe_per_period": self.spec.expected_gross_sharpe_per_period(
                    n_held=len(self.held_assets)
                ),
                "oracle_gross_sharpe_per_period": self.spec.oracle_gross_sharpe_per_period(
                    n_held=len(self.held_assets)
                ),
                "signal_fidelity": self.spec.signal_fidelity,
            },
            "artifact": self.artifact.to_dict(),
        }


def _stamp_for(
    spec: SyntheticSpec,
    *,
    cost_mode: CostMode,
    held_assets: Sequence[str],
    lookback_periods: int,
) -> ReproducibilityStamp:
    """Build the I2 stamp for one harness run.

    Args:
        spec: the data-generating process, supplying the seed and data version.
        cost_mode: which cost parameters the run is charged.
        held_assets: the instruments the strategy trades.
        lookback_periods: the estimator window.

    Returns:
        A :class:`~backend.tracking.stamp.ReproducibilityStamp` carrying all four
        I2 components plus the dirty-tree marker.
    """
    configuration: dict[str, object] = dict(spec.to_dict())
    configuration["cost_mode"] = str(cost_mode)
    configuration["held_assets"] = list(held_assets)
    configuration["strategy_lookback_periods"] = int(lookback_periods)
    state = _harness_git_state()
    return ReproducibilityStamp(
        git_commit=state.commit,
        git_dirty=state.dirty,
        data_version=spec.data_version,
        config_hash=canonical_config_hash(configuration),
        seed=spec.seed,
    )


async def run_synthetic_backtest(
    spec: SyntheticSpec,
    *,
    cost_mode: CostMode = CostMode.NET_OF_MODELLED_COSTS,
    market: SyntheticMarket | None = None,
    assets: Sequence[str] | None = None,
    lookback_periods: int | None = None,
) -> SyntheticRun:
    """Run the real engine over one seeded synthetic sample.

    Nothing about the simulation is special-cased for being synthetic: the same
    :func:`~backend.backtest.engine.run_backtest`, the same point-in-time
    snapshot boundary, the same cost model, the same mandatory benchmark. Only
    the data is generated, and it says so in its own ``data_version``.

    Args:
        spec: the data-generating process and seed.
        cost_mode: which cost parameters to charge. Defaults to the reportable
            net-of-modelled-costs basis.
        market: a market already realized from ``spec``, to avoid regenerating it
            across candidates. Must have been built from an equal spec.
        assets: subset of instruments to trade. Defaults to all.
        lookback_periods: estimator window override. Defaults to the spec's.

    Returns:
        A :class:`SyntheticRun`.

    Raises:
        SyntheticDataError: if ``market`` was not built from ``spec``, or holds
            anything but harness-generated data.
        ValueError: propagated from :meth:`SyntheticMarket.strategy` for an
            unusable asset subset or window.
    """
    realized = build_synthetic_market(spec) if market is None else market
    if realized.spec != spec:
        msg = (
            "the supplied market was built from a different spec than the one being run; "
            "reusing a market across specs would silently attribute one process's data to "
            "another process's stamp"
        )
        raise SyntheticDataError(msg)
    require_synthetic_source(realized.data)

    held = tuple(assets) if assets is not None else realized.assets
    window = spec.lookback_periods if lookback_periods is None else lookback_periods
    strategy = realized.strategy(assets=held, lookback_periods=window)
    stamp = _stamp_for(spec, cost_mode=cost_mode, held_assets=held, lookback_periods=window)
    config = BacktestConfig(
        benchmark=realized.benchmark(),
        seed=spec.seed,
        strategy_config={
            "family": "trailing-mean-sign",
            "lookback_periods": int(window),
            "held_assets": list(held),
            "lookahead_leak_fraction": float(spec.lookahead_leak_fraction),
        },
        initial_capital_usd=spec.initial_capital_usd,
        cost_params=cost_mode.cost_params,
        periods_per_year=spec.periods_per_year,
        max_gross_exposure=1.0,
        confidence_level=spec.confidence_level,
        n_bootstrap=spec.bootstrap_resamples,
        git_commit=stamp.git_commit,
    )
    run = await run_backtest(
        data=realized.data,
        calendar=realized.calendar,
        strategy=strategy,
        config=config,
    )
    return SyntheticRun(
        spec=spec,
        cost_mode=cost_mode,
        stamp=stamp,
        artifact=run.artifact,
        held_assets=held,
        lookback_periods=window,
        predicted_gross_sharpe_per_period=realized.conditional_gross_sharpe_per_period(
            assets=held, lookback_periods=window
        ),
    )


# ---------------------------------------------------------------------------
# Many seeds — because one seed is not evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class SharpeSweep:
    """The distribution of realized Sharpe ratios over independent seeds.

    A single synthetic run says almost nothing: at ``T`` periods the sampling
    standard deviation of a Sharpe ratio is about ``1 / sqrt(T)`` whatever the
    truth, so one noise run landing at ``+0.3`` annualized and one signal run
    landing at ``-0.1`` are both ordinary. Every claim the harness makes is about
    this object, never about one of its members.

    Attributes:
        runs: one run per seed, in seed order.
    """

    runs: tuple[SyntheticRun, ...]

    def __post_init__(self) -> None:
        """Validate that the sweep is comparable across its members.

        Raises:
            ValueError: if there are fewer than two runs (a distribution needs a
                spread), if the runs disagree about the cost mode, or if any two
                differ in anything but the seed, or if a seed is repeated. A
                sweep that mixes processes would report a mixture and call it a
                distribution, and a repeated seed would count one sample twice.
        """
        if len(self.runs) < 2:
            msg = f"a sweep needs at least 2 seeds to have a distribution; got {len(self.runs)}"
            raise ValueError(msg)
        first = self.runs[0]
        reference = replace(first.spec, seed=0)
        for run in self.runs:
            if run.cost_mode is not first.cost_mode:
                msg = (
                    f"sweep mixes cost modes {first.cost_mode} and {run.cost_mode}; a "
                    "distribution over two different cost bases is not a distribution"
                )
                raise ValueError(msg)
            if replace(run.spec, seed=0) != reference:
                msg = "sweep members differ in more than their seed"
                raise ValueError(msg)
        if len({run.stamp.seed for run in self.runs}) != len(self.runs):
            msg = "sweep seeds must be distinct; a repeated seed is a repeated sample"
            raise ValueError(msg)

    @property
    def spec(self) -> SyntheticSpec:
        """Return the process the sweep varied the seed of (the first run's)."""
        return self.runs[0].spec

    @property
    def cost_mode(self) -> CostMode:
        """Return the cost basis every member was charged."""
        return self.runs[0].cost_mode

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the seeds, in run order."""
        return tuple(run.stamp.seed for run in self.runs)

    @property
    def values(self) -> FloatArray:
        """Return each seed's realized **per-period** Sharpe ratio."""
        return np.array([run.sharpe_per_period for run in self.runs], dtype=np.float64)

    @property
    def n_seeds(self) -> int:
        """Return how many independent samples the distribution holds."""
        return len(self.runs)

    @property
    def mean(self) -> float:
        """Return the mean per-period Sharpe ratio over seeds."""
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        """Return the seed-to-seed sample standard deviation (``ddof=1``)."""
        return float(np.std(self.values, ddof=1))

    @property
    def standard_error(self) -> float:
        """Return the standard error of :attr:`mean` over seeds.

        The quantity every quantified claim about the mean is stated in multiples
        of. Seeds are independent draws of the whole experiment, so this is an
        honest standard error rather than a within-sample one.
        """
        return self.std / math.sqrt(self.n_seeds)

    @property
    def t_statistic(self) -> float:
        """Return ``mean / standard_error``: how many standard errors from zero."""
        return self.mean / self.standard_error

    @property
    def annualized_mean(self) -> float:
        """Return the mean per-period Sharpe ratio annualized by the spec's factor."""
        return self.spec.annualize(self.mean)

    @property
    def annualized_standard_error(self) -> float:
        """Return :attr:`standard_error` on the annualized scale."""
        return self.spec.annualize(self.standard_error)

    @property
    def predictions(self) -> FloatArray:
        """Return each seed's closed-form prediction, conditional on its own alphas.

        The companion of :attr:`values`, seed for seed. Predicted from the
        data-generating process alone
        (:meth:`SyntheticMarket.conditional_gross_sharpe_per_period`) and before
        costs, so a net-of-cost sweep should sit *below* these by the cost drag
        and a zero-cost sweep should sit on them.
        """
        return np.array(
            [run.predicted_gross_sharpe_per_period for run in self.runs], dtype=np.float64
        )

    @property
    def mean_prediction(self) -> float:
        """Return the mean of the per-seed closed-form predictions."""
        return float(np.mean(self.predictions))

    @property
    def residuals(self) -> FloatArray:
        """Return ``measured - predicted`` for each seed.

        Pairing each measurement with the prediction for *its own* alpha draw
        removes the dominant source of seed-to-seed scatter — which draw of ``a``
        happened — and leaves the sampling error of a Sharpe ratio over ``T``
        periods. It is what makes a recovery claim tight enough to be worth
        asserting.
        """
        return self.values - self.predictions

    @property
    def residual_mean(self) -> float:
        """Return the mean of :attr:`residuals`: the systematic recovery error."""
        return float(np.mean(self.residuals))

    @property
    def residual_standard_error(self) -> float:
        """Return the standard error of :attr:`residual_mean` over seeds."""
        return float(np.std(self.residuals, ddof=1)) / math.sqrt(self.n_seeds)

    @property
    def recovery_ratio(self) -> float:
        """Return measured mean divided by predicted mean, or ``nan`` for pure noise.

        Dimensionless. ``1.0`` is exact recovery; a framework with a scale error
        lands on a constant other than one, and one that recovers "some" signal
        lands on a different constant at every injected strength.
        """
        predicted = self.mean_prediction
        if predicted == 0.0:
            return math.nan
        return self.mean / predicted

    @property
    def mean_turnover_per_period(self) -> float:
        """Return the mean traded notional per rebalance, as a fraction of value."""
        return float(np.mean([run.turnover_per_period for run in self.runs]))

    @property
    def mean_total_cost_bps_of_capital(self) -> float:
        """Return the mean total cost paid, in basis points of initial capital."""
        return float(np.mean([run.total_cost_bps_of_capital for run in self.runs]))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the distribution as a JSON-safe mapping, disclosure first."""
        return {
            "disclosure": SYNTHETIC_DISCLOSURE,
            "is_research_finding": False,
            "cost_mode": str(self.cost_mode),
            "n_seeds": self.n_seeds,
            "seeds": list(self.seeds),
            "alpha_dispersion_bps_per_day": float(self.spec.alpha_dispersion_bps_per_day),
            "predicted_gross_sharpe_per_period": self.spec.expected_gross_sharpe_per_period(
                n_held=len(self.runs[0].held_assets)
            ),
            "predicted_conditional_mean": self.mean_prediction,
            "residual_mean": self.residual_mean,
            "residual_standard_error": self.residual_standard_error,
            "recovery_ratio": self.recovery_ratio,
            "measured": {
                "mean_sharpe_per_period": self.mean,
                "std_sharpe_per_period": self.std,
                "standard_error": self.standard_error,
                "t_statistic": self.t_statistic,
                "annualized_mean": self.annualized_mean,
                "min": float(np.min(self.values)),
                "max": float(np.max(self.values)),
                "mean_turnover_per_period": self.mean_turnover_per_period,
                "mean_total_cost_bps_of_capital": self.mean_total_cost_bps_of_capital,
            },
        }


async def sweep_seeds(
    spec: SyntheticSpec,
    *,
    seeds: Sequence[int],
    cost_mode: CostMode = CostMode.NET_OF_MODELLED_COSTS,
    assets: Sequence[str] | None = None,
    lookback_periods: int | None = None,
) -> SharpeSweep:
    """Run the engine once per seed and return the distribution of the results.

    Args:
        spec: the process to sweep. Its own ``seed`` is replaced by each entry of
            ``seeds``, so it acts as a template.
        seeds: the seeds to run, distinct and non-negative.
        cost_mode: which cost parameters to charge every member.
        assets: subset of instruments to trade. Defaults to all.
        lookback_periods: estimator window override.

    Returns:
        A :class:`SharpeSweep`.

    Raises:
        ValueError: if fewer than two seeds are supplied, or they are not
            distinct.
    """
    runs = [
        await run_synthetic_backtest(
            replace(spec, seed=seed),
            cost_mode=cost_mode,
            assets=assets,
            lookback_periods=lookback_periods,
        )
        for seed in seeds
    ]
    return SharpeSweep(runs=tuple(runs))


# ---------------------------------------------------------------------------
# Many configurations — what the overfitting statistics are for
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """One configuration in a search, which is what PBO and DSR count.

    Attributes:
        label: a short name, used in reports and as the column identity.
        assets: the instruments this candidate trades.
        lookback_periods: its estimator window, or ``None`` for the spec's.
    """

    label: str
    assets: tuple[str, ...]
    lookback_periods: int | None = None

    def __post_init__(self) -> None:
        """Validate the candidate.

        Raises:
            ValueError: if the label is blank or the candidate holds nothing.
        """
        if not self.label.strip():
            msg = "a candidate configuration must be named; a blank label is untraceable"
            raise ValueError(msg)
        if not self.assets:
            msg = f"candidate {self.label!r} holds no assets"
            raise ValueError(msg)


def disjoint_asset_blocks(
    market: SyntheticMarket, *, block_size: int
) -> tuple[StrategyCandidate, ...]:
    """Cut the market's assets into non-overlapping candidate configurations.

    Disjoint blocks of equal size are the one candidate family that is *exactly*
    exchangeable under the null: the assets are iid, so with no signal injected
    the candidates are independent draws of the same distribution and nothing
    distinguishes them but luck. That is the setting in which the Probability of
    Backtest Overfitting has a known answer — one half — so any departure from it
    is the framework's, not the family's.

    Args:
        market: the market whose assets to partition.
        block_size: assets per candidate. Must divide ``n_assets`` exactly and
            leave at least two candidates, since ranking one candidate is
            meaningless.

    Returns:
        One :class:`StrategyCandidate` per block, in asset order.

    Raises:
        ValueError: if ``block_size`` does not divide the universe into at least
            two equal blocks.
    """
    total = len(market.assets)
    if block_size < 1 or total % block_size != 0 or total // block_size < 2:
        msg = (
            f"block_size={block_size} does not cut {total} assets into at least two equal "
            "blocks; unequal or single blocks break the exchangeability the null relies on"
        )
        raise ValueError(msg)
    return tuple(
        StrategyCandidate(
            label=f"block-{start // block_size:02d}",
            assets=market.assets[start : start + block_size],
        )
        for start in range(0, total, block_size)
    )


@dataclass(frozen=True, slots=True, eq=False)
class ConfigurationSweep:
    """Every configuration of a search, run through the engine over one sample.

    This is the object the overfitting statistics are defined on. Its trial count
    is not an estimate or a ledger lookup — it is how many engine runs actually
    happened, so the Deflated Sharpe Ratio computed from it cannot understate the
    search (directive §9.7).

    Attributes:
        spec: the process every candidate was run over.
        cost_mode: the cost basis every candidate was charged.
        runs: one run per candidate, in candidate order.
        candidates: the configurations, in the same order.
    """

    spec: SyntheticSpec
    cost_mode: CostMode
    runs: tuple[SyntheticRun, ...]
    candidates: tuple[StrategyCandidate, ...]

    def __post_init__(self) -> None:
        """Validate the sweep.

        Raises:
            ValueError: if fewer than two candidates were run (there is no rank
                among one), or if the run and candidate counts disagree.
        """
        if len(self.runs) != len(self.candidates):
            msg = f"{len(self.runs)} runs for {len(self.candidates)} candidates"
            raise ValueError(msg)
        if len(self.runs) < 2:
            msg = f"a configuration sweep needs at least 2 candidates to rank; got {len(self.runs)}"
            raise ValueError(msg)

    @property
    def trials(self) -> int:
        """Return the number of configurations evaluated — the honest trial count."""
        return len(self.runs)

    @property
    def return_matrix(self) -> FloatArray:
        """Return the ``(T, N)`` matrix of per-period returns, one column per candidate.

        Net of modelled costs under :attr:`CostMode.NET_OF_MODELLED_COSTS`; gross
        under the zero-cost diagnostic.
        """
        return np.column_stack([run.returns for run in self.runs])

    @property
    def trial_sharpes(self) -> FloatArray:
        """Return each candidate's full-sample **per-period** Sharpe ratio."""
        return np.array([run.sharpe_per_period for run in self.runs], dtype=np.float64)

    @property
    def winner_index(self) -> int:
        """Return the index of the candidate with the best full-sample Sharpe ratio."""
        return int(np.argmax(self.trial_sharpes))

    @property
    def winner(self) -> SyntheticRun:
        """Return the run a naive researcher would report — the best of the search."""
        return self.runs[self.winner_index]

    def probability_of_backtest_overfitting(self, *, n_partitions: int = 10) -> PBOResult:
        """Return the PBO of selecting the in-sample winner among these candidates.

        Args:
            n_partitions: ``S`` for the combinatorially symmetric cross-validation.
                Even, at least 2, at most ``T``.

        Returns:
            The :class:`~backend.backtest.pbo.PBOResult`.
        """
        return probability_of_backtest_overfitting(self.return_matrix, n_partitions=n_partitions)

    def deflated_sharpe_of_winner(self) -> DeflatedSharpeResult:
        """Deflate the winner's Sharpe ratio against the search that produced it.

        The trial count is :attr:`trials` and the trial variance is the spread of
        :attr:`trial_sharpes`, both taken from runs that actually happened. This
        is the number that has to see through a search over worthless candidates.

        Returns:
            The :class:`~backend.backtest.dsr.DeflatedSharpeResult`.
        """
        return deflated_sharpe_ratio_from_trials(
            self.winner.returns, trial_sharpes=self.trial_sharpes
        )

    def undeflated_sharpe_of_winner(self) -> DeflatedSharpeResult:
        """Return what the winner certifies if the search is denied — trials of 1.

        Identical data, identical returns, identical formula; only the declared
        trial count is falsified. Kept here beside the honest number because the
        gap between them is the entire value of recording the search, and because
        a harness that only ever computed the honest one could not demonstrate
        what dishonesty costs.

        Returns:
            The :class:`~backend.backtest.dsr.DeflatedSharpeResult` a single-trial
            claim would produce.
        """
        return deflated_sharpe_ratio_from_trials(
            self.winner.returns, trial_sharpes=[self.winner.sharpe_per_period]
        )

    def selection_paths(
        self,
        *,
        n_groups: int = 6,
        n_test_groups: int = 2,
        label_horizon: float = 1.0,
    ) -> PathDistribution:
        """Return the CPCV distribution of Sharpe ratios of the *selection procedure*.

        On each combinatorially purged split the best candidate on the training
        rows is chosen, and its returns on that split's test rows are recorded.
        Reassembling those per-split out-of-sample stretches gives
        ``C(n_groups - 1, n_test_groups - 1)`` complete paths, each covering the
        whole sample once. What is being measured is not one configuration but
        the *rule* "pick the in-sample winner" — which is the thing an operator
        actually applies, and the thing that overfits.

        Labels here are point-in-time: a period's outcome is that period's
        return, so ``label_horizon`` is one period and purging removes only the
        observations straddling a test block's boundary.

        Args:
            n_groups: ``N``, contiguous groups the sample is cut into.
            n_test_groups: ``k``, groups per test set.
            label_horizon: the label's resolution horizon in periods, used for
                purging and as the embargo length.

        Returns:
            A :class:`~backend.backtest.cpcv.PathDistribution` of per-period
            Sharpe ratios, one per path.
        """
        matrix = self.return_matrix
        n_observations = matrix.shape[0]
        cv = CombinatorialPurgedCV(
            n_groups=n_groups, n_test_groups=n_test_groups, embargo=label_horizon
        )
        starts = np.arange(n_observations, dtype=np.float64)
        splits = cv.split(starts, starts + label_horizon)
        per_split: list[FloatArray] = []
        for split in splits:
            train = matrix[split.train_indices, :]
            in_sample = np.array(
                [
                    sharpe_ratio(train[:, column], periods_per_year=None, ddof=1)
                    for column in range(train.shape[1])
                ],
                dtype=np.float64,
            )
            chosen = int(np.argmax(in_sample))
            per_split.append(matrix[split.test_indices, chosen])
        return path_sharpe_ratios(splits.assemble_paths(per_split))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the sweep's conclusions as a JSON-safe mapping, disclosure first."""
        pbo = self.probability_of_backtest_overfitting()
        deflated = self.deflated_sharpe_of_winner()
        return {
            "disclosure": SYNTHETIC_DISCLOSURE,
            "is_research_finding": False,
            "cost_mode": str(self.cost_mode),
            "trials": self.trials,
            "candidates": [candidate.label for candidate in self.candidates],
            "alpha_dispersion_bps_per_day": float(self.spec.alpha_dispersion_bps_per_day),
            "winner": self.candidates[self.winner_index].label,
            "winner_sharpe_per_period": self.winner.sharpe_per_period,
            "pbo": pbo.pbo,
            "deflated_sharpe_of_winner": deflated.value,
            "undeflated_sharpe_of_winner": self.undeflated_sharpe_of_winner().value,
            "selection_path_mean_sharpe_per_period": self.selection_paths().mean,
        }


async def sweep_configurations(
    spec: SyntheticSpec,
    *,
    candidates: Sequence[StrategyCandidate],
    cost_mode: CostMode = CostMode.NET_OF_MODELLED_COSTS,
    market: SyntheticMarket | None = None,
) -> ConfigurationSweep:
    """Run every candidate configuration over one seeded sample.

    Every candidate sees the same market, which is what makes their returns
    comparable row by row — the requirement PBO's combinatorially symmetric
    cross-validation places on its input matrix.

    Args:
        spec: the process to realize once and run every candidate over.
        candidates: the configurations. At least two.
        cost_mode: which cost parameters to charge.
        market: a market already realized from ``spec``, reused across
            candidates.

    Returns:
        A :class:`ConfigurationSweep`.
    """
    realized = build_synthetic_market(spec) if market is None else market
    runs = [
        await run_synthetic_backtest(
            spec,
            cost_mode=cost_mode,
            market=realized,
            assets=candidate.assets,
            lookback_periods=candidate.lookback_periods,
        )
        for candidate in candidates
    ]
    return ConfigurationSweep(
        spec=spec,
        cost_mode=cost_mode,
        runs=tuple(runs),
        candidates=tuple(candidates),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(sweeps: Sequence[SharpeSweep]) -> str:
    """Render a set of seed sweeps as readable text, disclosure first.

    Intended for the operator re-running the harness after a change to the
    validation stack: it puts the injected strength, the closed-form prediction
    and the measured distribution on adjacent lines so a scale error is visible
    rather than inferable.

    Args:
        sweeps: the distributions to report, in the order to display them.

    Returns:
        A multi-line string. Every figure in it is synthetic and the first line
        says so.

    Raises:
        ValueError: if no sweeps are supplied — an empty report reads like a
            clean run.
    """
    if not sweeps:
        msg = "format_report needs at least one sweep; an empty report looks like a pass"
        raise ValueError(msg)
    lines = [SYNTHETIC_DISCLOSURE, ""]
    header = (
        f"{'injected tau':>14} {'cost basis':>22} {'predicted SR':>13} "
        f"{'measured SR':>12} {'std err':>9} {'annualized':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sweep in sweeps:
        predicted = sweep.spec.expected_gross_sharpe_per_period(
            n_held=len(sweep.runs[0].held_assets)
        )
        lines.append(
            f"{sweep.spec.alpha_dispersion_bps_per_day:>11.2f}bps "
            f"{sweep.cost_mode!s:>22} "
            f"{predicted:>13.5f} "
            f"{sweep.mean:>12.5f} "
            f"{sweep.standard_error:>9.5f} "
            f"{sweep.annualized_mean:>11.3f}"
        )
    lines.append("")
    lines.append(
        "Sharpe ratios are per period unless the column says annualized; tau is the "
        "cross-sectional standard deviation of true expected return in basis points per "
        "period. 'predicted' is the closed form in backend.backtest.synthetic, derived "
        "from the data-generating process and not from anything in backend.backtest."
    )
    return "\n".join(lines)
