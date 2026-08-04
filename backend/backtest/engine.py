"""Event-driven daily backtest loop (P10.1).

What makes this loop safe
-------------------------

The failure this module is built against is lookahead: a simulation that uses,
at simulated date *d*, a fact that was not knowable at *d*. It does not fail
loudly. It produces an excellent backtest.

Three structural properties, in order of how much they buy:

1. **Every read goes through an ``as_of`` gate.** The engine holds a
   :class:`PointInTimeMarketData`, whose only method is
   ``as_of(instant) -> MarketSnapshot``. There is no other accessor, no "give me
   the whole series" call, and no handle on the underlying store. The name and
   the semantics mirror :func:`backend.db.asof.as_of` (DECISIONS.md D-011,
   D-018): timezone-aware UTC instants, ``knowledge_time <= as_of`` inclusive at
   the boundary.

2. **The snapshot verifies itself.** :class:`MarketSnapshot` raises
   :class:`LookaheadError` at construction if it holds any observation whose
   ``knowledge_time`` — or whose ``date`` — is after its own ``as_of``. A future
   implementation of the protocol that filters incorrectly fails loudly at the
   boundary rather than silently returning the answer. This is the D-018
   principle applied one layer up: do not trust that the filter produced the
   right thing, verify the thing that comes out.

3. **The strategy is handed a snapshot, never the source.** The strategy
   callable receives exactly one argument, the snapshot for the decision
   instant. It cannot ask for another date because it has nothing to ask. And
   the loop fetches the next period's returns **only after** the decision has
   been made and the trades have been costed, so the value that would constitute
   lookahead does not exist in the enclosing scope while the decision is being
   made.

Where the data comes from — and where it does not
-------------------------------------------------

The vendor feeds are blocked (BLOCKERS.md **B1**: Sharadar credentials pending).
Invariant I3 forbids inventing a source, so this module **takes price and return
series as injected inputs**: :class:`InjectedMarketData` holds observations the
caller supplies and does nothing else. It generates nothing, interpolates
nothing, and defaults nothing. When B1 clears, a database-backed implementation
of :class:`PointInTimeMarketData` — wrapping :func:`backend.db.asof.as_of` — is
the only thing that needs to be written; the engine does not change, because the
engine never knew where the data came from.

Net of costs, always
--------------------

Invariant I4: no result is reported gross. There is no gross return anywhere in
the public surface of this module. The loop deducts modelled execution costs and
borrow *before* applying each period's market return, and the only equity path
that leaves the module is the net one. A caller can, of course, pass cost
parameters that are all zero — no library can stop that — so the artifact
carries the full parameter snapshot
(:class:`~backend.backtest.artifact.CostProvenance`), which makes a
cost-suppressed run visible in its own output rather than indistinguishable from
a costed one.

Accounting conventions, stated so they can be checked
-----------------------------------------------------

* The calendar is a sequence of ``T + 1`` timezone-aware UTC instants. A
  decision is taken at each of the first ``T``; the return over
  ``(calendar[i], calendar[i+1]]`` is the observation dated ``calendar[i+1]``.
* A decision at ``calendar[i]`` may use every observation whose
  ``knowledge_time <= calendar[i]`` — including the bar dated ``calendar[i]``
  itself, if the injected series says that bar was knowable by then. Whether it
  was is a **property of the data, not of this engine**: the trade-at-the-close
  convention is only honest if the feed's ``knowledge_time`` policy (D-011, one
  per connector) says the close was published at the close. A feed that
  publishes late must say so by carrying a later ``knowledge_time``, and the
  engine will then withhold the bar. This is the one place where the loop's
  point-in-time guarantee is only as good as its input, so it is stated rather
  than assumed.
* Weights are fractions of portfolio value; the remainder is cash at **zero**
  interest. Gross exposure is capped at 1.0 by default because leverage is an
  explicit non-goal (§1.1).
* Order of operations within a period: decide → cost the trades → accrue borrow
  on the post-trade short book → apply the market return. Costs are therefore
  paid out of the pre-return portfolio, which is the conservative ordering.
* The book starts **flat**, so the first period pays a full entry cost. The
  benchmark pays its own entry cost on the same basis.
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from backend.backtest.artifact import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    CostProvenance,
    EquityCurve,
    IntervalConfig,
    JsonValue,
    RealizedAccounting,
    ReproducibilityStamp,
    RunArtifact,
    TrackRecord,
    compare_to_benchmark,
    config_hash,
    current_git_commit,
    summarize_track_record,
)
from backend.backtest.benchmark import (
    SECONDS_PER_DAY,
    BuyAndHoldSpec,
    ExecutionCosts,
    PortfolioWipeoutError,
    WeightError,
    accrue_borrow_usd,
    buy_and_hold,
    cost_weight_changes,
    drift_weights,
    gross_exposure,
    validate_weights,
)
from backend.costs.model import UNCALIBRATED_DEFAULTS, CostModelParams

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

__all__ = [
    "BacktestConfig",
    "BacktestRun",
    "CalendarError",
    "InjectedMarketData",
    "LookaheadError",
    "MarketSnapshot",
    "Observation",
    "PointInTimeMarketData",
    "Strategy",
    "market_parameters",
    "run_backtest",
    "validated_calendar",
]


class LookaheadError(RuntimeError):
    """Raised when data that was not knowable at the simulated instant is in scope.

    Fatal, never a warning. A backtest that has seen the future is not "slightly
    optimistic"; it is measuring something else entirely, and every number
    derived from it — Sharpe, Deflated Sharpe, PBO — is about that other thing.
    """


class CalendarError(ValueError):
    """Raised when a simulation calendar is unusable.

    Covers naive or non-UTC instants, non-monotonic dates, and a calendar too
    short to produce the two return periods any dispersion statistic needs.
    """


def _require_utc(moment: dt.datetime, *, label: str) -> dt.datetime:
    """Return ``moment`` if it is timezone-aware UTC, else raise.

    Naive datetimes are refused rather than localized: D-012 records an
    hours-scale silent temporal error in this codebase caused by exactly that
    reinterpretation, and a backtest is the last place to repeat it.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        msg = f"{label} must be timezone-aware UTC; got naive datetime {moment!r}"
        raise CalendarError(msg)
    if moment.utcoffset() != dt.timedelta(0):
        msg = f"{label} must be UTC (offset 0); got offset {moment.utcoffset()} in {moment!r}"
        raise CalendarError(msg)
    return moment


# ---------------------------------------------------------------------------
# The point-in-time data boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """One asset's knowable facts for one period.

    Attributes:
        date: the instant the period **ends**, timezone-aware UTC. The return
            covers ``(previous date, date]``.
        knowledge_time: when this observation became knowable to the market,
            timezone-aware UTC. Must not precede ``date`` — a period's return
            cannot be known before the period is over. For a daily close-to-close
            bar this is the close itself, or later if the vendor publishes late;
            D-011 fixes the policy per connector.
        total_return: simple total return over the period, as a **fraction**
            (``0.01`` is 1%), including dividends and adjusted for corporate
            actions. Must be finite and at least ``-1.0``.
        adv_usd: average daily dollar volume as known at ``knowledge_time``, in
            **US dollars**, strictly positive. Feeds the cost model's
            participation rate.
        daily_volatility_bps: the asset's daily return standard deviation in
            **basis points** (``200.0`` is 2% per day), or ``None`` to let the
            cost model use its documented default.
    """

    date: dt.datetime
    knowledge_time: dt.datetime
    total_return: float
    adv_usd: float
    daily_volatility_bps: float | None = None

    def __post_init__(self) -> None:
        """Validate temporal ordering, units and signs.

        Raises:
            CalendarError: if either instant is naive or not UTC.
            LookaheadError: if ``knowledge_time`` precedes ``date``, which would
                assert that a period's outcome was known before it ended.
            ValueError: if the return is non-finite or below ``-1``, if
                ``adv_usd`` is not strictly positive, or if a supplied
                volatility is negative.
        """
        _require_utc(self.date, label="Observation.date")
        _require_utc(self.knowledge_time, label="Observation.knowledge_time")
        if self.knowledge_time < self.date:
            msg = (
                f"knowledge_time {self.knowledge_time.isoformat()} precedes the period end "
                f"{self.date.isoformat()}: a period's return cannot be knowable before the "
                "period is over"
            )
            raise LookaheadError(msg)
        value = float(self.total_return)
        if not np.isfinite(value) or value < -1.0:
            msg = f"total_return must be finite and at least -1.0; got {self.total_return!r}"
            raise ValueError(msg)
        object.__setattr__(self, "total_return", value)
        if not np.isfinite(self.adv_usd) or self.adv_usd <= 0.0:
            msg = (
                f"adv_usd must be finite and strictly positive; got {self.adv_usd!r}. "
                "An asset with no volume has no defined participation rate."
            )
            raise ValueError(msg)
        object.__setattr__(self, "adv_usd", float(self.adv_usd))
        if self.daily_volatility_bps is not None:
            volatility = float(self.daily_volatility_bps)
            if not np.isfinite(volatility) or volatility < 0.0:
                msg = f"daily_volatility_bps must be finite and non-negative; got {volatility!r}"
                raise ValueError(msg)
            object.__setattr__(self, "daily_volatility_bps", volatility)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """What a strategy may see at one instant — and never anything later.

    This is the *only* object a strategy ever sees. Its constructor re-verifies
    the point-in-time filter that produced it: an observation whose ``date`` —
    or whose ``knowledge_time`` — is after ``as_of`` raises
    :class:`LookaheadError` here, no matter which implementation of
    :class:`PointInTimeMarketData` built it. Structural verification of the
    filter is not enough; the value that comes out is checked (D-018).

    The invariant is an **upper bound, not an equality**: a snapshot holds *at
    most* what was knowable at ``as_of``, and holding less is legitimate. The
    walk-forward trainer relies on that when it hands a fit callback a snapshot
    restricted to one training window
    (:func:`backend.backtest.walkforward.run_walk_forward`). Nothing downstream
    may infer "this is the complete history" from a snapshot; it may only infer
    "nothing here postdates ``as_of``", which is the property that matters.

    Attributes:
        as_of: the simulated instant, timezone-aware UTC. Facts may be visible
            iff ``knowledge_time <= as_of`` (inclusive at the boundary, matching
            :func:`backend.db.asof.as_of`).
        observations: asset to its visible observations, in ascending date
            order.
    """

    as_of: dt.datetime
    observations: Mapping[str, tuple[Observation, ...]]

    def __post_init__(self) -> None:
        """Verify that nothing in the snapshot postdates its own ``as_of``.

        The two lookahead checks are ordered deliberately. ``date > as_of`` is
        tested first because it is the stronger statement — the snapshot holds a
        period that has not finished yet — and because
        :class:`Observation` already forbids ``knowledge_time < date``, so a
        future-dated bar always fails this check before the weaker one. The
        ``knowledge_time`` check then catches what remains: a bar dated in the
        past that was **published** later than ``as_of``, which is the ordinary
        late-vendor case and the one a real feed produces.

        Raises:
            CalendarError: if ``as_of`` is naive or not UTC, or if an asset's
                observations are not in strictly ascending date order.
            LookaheadError: if any observation postdates ``as_of`` or was not
                yet knowable at it.
        """
        _require_utc(self.as_of, label="MarketSnapshot.as_of")
        frozen = {asset: tuple(series) for asset, series in self.observations.items()}
        object.__setattr__(self, "observations", frozen)
        for asset, series in frozen.items():
            previous: dt.datetime | None = None
            for observation in series:
                if previous is not None and observation.date <= previous:
                    msg = (
                        f"observations for {asset!r} must be in strictly ascending date order; "
                        f"{observation.date.isoformat()} follows {previous.isoformat()}"
                    )
                    raise CalendarError(msg)
                previous = observation.date
                if observation.date > self.as_of:
                    msg = (
                        f"snapshot as of {self.as_of.isoformat()} holds an observation for "
                        f"{asset!r} dated {observation.date.isoformat()}, which is in its future"
                    )
                    raise LookaheadError(msg)
                if observation.knowledge_time > self.as_of:
                    msg = (
                        f"snapshot as of {self.as_of.isoformat()} holds an observation for "
                        f"{asset!r} that only became knowable at "
                        f"{observation.knowledge_time.isoformat()}"
                    )
                    raise LookaheadError(msg)

    @property
    def assets(self) -> frozenset[str]:
        """Return the assets with at least one observation knowable at ``as_of``."""
        return frozenset(asset for asset, series in self.observations.items() if series)

    def history(self, asset: str) -> tuple[Observation, ...]:
        """Return every knowable observation for ``asset``, oldest first.

        Args:
            asset: the asset identifier.

        Returns:
            The observations, possibly empty for an asset with no knowable history.
        """
        return self.observations.get(asset, ())

    def latest(self, asset: str) -> Observation | None:
        """Return the most recent knowable observation for ``asset``, or ``None``."""
        series = self.observations.get(asset, ())
        return series[-1] if series else None

    def observation_on(self, asset: str, date: dt.datetime) -> Observation | None:
        """Return ``asset``'s observation dated exactly ``date``, or ``None``.

        Args:
            asset: the asset identifier.
            date: the period-end instant to look for, timezone-aware UTC.

        Returns:
            The matching observation, or ``None`` if the asset has no
            observation for that period that is knowable at ``as_of``.
        """
        for observation in reversed(self.observations.get(asset, ())):
            if observation.date == date:
                return observation
            if observation.date < date:
                return None
        return None

    def since(self, earliest: dt.datetime) -> MarketSnapshot:
        """Return a snapshot holding only the observations dated at or after ``earliest``.

        Narrowing a snapshot is always safe — the result still holds nothing
        that postdates ``as_of`` — and it is how a *rolling* walk-forward
        training window is expressed: the upper bound is the point-in-time
        invariant, the lower bound is the scheme's choice of how much history to
        fit on. Discarding old observations can never manufacture lookahead, so
        no new verification is needed beyond the constructor's.

        Args:
            earliest: the oldest observation date to keep, timezone-aware UTC,
                inclusive.

        Returns:
            A new :class:`MarketSnapshot` with the same ``as_of``.

        Raises:
            CalendarError: if ``earliest`` is naive or not UTC.
        """
        _require_utc(earliest, label="earliest")
        return MarketSnapshot(
            as_of=self.as_of,
            observations={
                asset: tuple(item for item in series if item.date >= earliest)
                for asset, series in self.observations.items()
            },
        )

    def returns_history(self, asset: str) -> tuple[float, ...]:
        """Return ``asset``'s knowable per-period total returns, oldest first.

        Convenience for strategies that compute trailing statistics. Units:
        simple returns as fractions.
        """
        return tuple(observation.total_return for observation in self.history(asset))


@runtime_checkable
class PointInTimeMarketData(Protocol):
    """The engine's only door to historical data.

    Deliberately minimal: one asynchronous accessor taking an instant, plus a
    version string. There is no method that returns a whole series, because such
    a method is exactly how a strategy acquires the future.

    Implementations must return a :class:`MarketSnapshot` containing precisely
    the observations with ``knowledge_time <= as_of_ts``. The snapshot
    constructor re-checks this, so an implementation that gets it wrong fails at
    the boundary rather than silently leaking.

    Asynchronous because the implementation that matters — the one wrapping
    :func:`backend.db.asof.as_of` once BLOCKERS.md B1 clears — is asynchronous.
    Making the protocol synchronous now would force either a rewrite or a thread
    pool later.
    """

    @property
    def data_version(self) -> str:
        """Return the identifier of the dataset this source serves (invariant I2).

        Recorded on every :class:`~backend.backtest.artifact.ReproducibilityStamp`.
        It comes from the source rather than from the caller so that a run cannot
        mislabel its own inputs.
        """
        ...

    async def as_of(self, as_of_ts: dt.datetime) -> MarketSnapshot:
        """Return everything knowable at ``as_of_ts``.

        Args:
            as_of_ts: the simulated instant, timezone-aware UTC.

        Returns:
            A :class:`MarketSnapshot` holding only observations with
            ``knowledge_time <= as_of_ts``.
        """
        ...


@dataclass(frozen=True, slots=True, eq=False)
class InjectedMarketData:
    """A point-in-time source over observations the **caller** supplies.

    This exists because BLOCKERS.md **B1** blocks the vendor feeds and invariant
    I3 forbids inventing a substitute. It is a filter, not a source: it holds
    exactly the observations it was given, returns the subset knowable at each
    requested instant, and fabricates, interpolates and extrapolates nothing. An
    asset with no observation for a date simply has none, and the engine raises
    rather than filling a gap.

    The boundary is deliberately visible in the name. When a database-backed
    implementation lands, tests that inject known series stay valid — they are
    testing the engine — while anything reporting a *result* must move to the
    real source.

    Attributes:
        observations: asset to its observations. Each series must be in strictly
            ascending date order with **non-decreasing** ``knowledge_time``,
            which is what a daily bar feed produces and what allows the as-of
            filter to be a binary search rather than a scan.
        data_version: identifier of this dataset, recorded on every artifact
            (invariant I2). Must be non-empty — an unnamed dataset makes a run
            unreproducible even with the commit and the config.
    """

    observations: Mapping[str, tuple[Observation, ...]]
    data_version: str
    _knowledge_epochs: Mapping[str, tuple[float, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the injected series and precompute the as-of search index.

        Raises:
            ValueError: if ``data_version`` is blank.
            CalendarError: if any series is not in strictly ascending date order
                or has a decreasing ``knowledge_time``. A restatement — a later
                correction of an earlier period — is a legitimate bitemporal
                event that this simple source does not represent; it belongs in
                the database-backed implementation, where D-011's
                latest-knowledge-wins rule handles it. Refusing it here is
                better than silently mis-ordering it.
        """
        if not self.data_version.strip():
            msg = "data_version must name the injected dataset (invariant I2)"
            raise ValueError(msg)
        frozen = {asset: tuple(series) for asset, series in self.observations.items()}
        epochs: dict[str, tuple[float, ...]] = {}
        for asset, series in frozen.items():
            for index, observation in enumerate(series):
                if index and observation.date <= series[index - 1].date:
                    msg = (
                        f"observations for {asset!r} must be in strictly ascending date order; "
                        f"{observation.date.isoformat()} follows "
                        f"{series[index - 1].date.isoformat()}"
                    )
                    raise CalendarError(msg)
                if index and observation.knowledge_time < series[index - 1].knowledge_time:
                    msg = (
                        f"observations for {asset!r} must have non-decreasing knowledge_time; "
                        f"{observation.knowledge_time.isoformat()} follows "
                        f"{series[index - 1].knowledge_time.isoformat()}. Restatements need "
                        "the bitemporal store (D-011), not this injected source."
                    )
                    raise CalendarError(msg)
            epochs[asset] = tuple(item.knowledge_time.timestamp() for item in series)
        object.__setattr__(self, "observations", frozen)
        object.__setattr__(self, "_knowledge_epochs", epochs)

    async def as_of(self, as_of_ts: dt.datetime) -> MarketSnapshot:
        """Return the observations knowable at ``as_of_ts``.

        Args:
            as_of_ts: the simulated instant, timezone-aware UTC. The boundary is
                inclusive: ``knowledge_time == as_of_ts`` is visible, matching
                :func:`backend.db.asof.as_of`.

        Returns:
            A :class:`MarketSnapshot` for that instant.

        Raises:
            CalendarError: if ``as_of_ts`` is naive or not UTC.
        """
        _require_utc(as_of_ts, label="as_of_ts")
        cutoff = as_of_ts.timestamp()
        visible = {
            asset: series[: bisect_right(self._knowledge_epochs[asset], cutoff)]
            for asset, series in self.observations.items()
        }
        return MarketSnapshot(as_of=as_of_ts, observations=visible)


type Strategy = Callable[[MarketSnapshot], Mapping[str, float]]
"""Target weights from one point-in-time snapshot.

The single argument is the whole contract: a strategy has no other way to reach
data, and therefore no way to reach the future. The return is a mapping from
asset to target weight as a fraction of portfolio value (negative is short);
assets omitted are targeted flat.
"""


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class BacktestConfig:
    """Everything about a run that is not the data or the strategy code.

    Attributes:
        benchmark: the buy-and-hold baseline, simulated over the same dates and
            charged the same costs. Required — directive §5-P10 says the
            benchmark is "always shown alongside every strategy result", so
            there is no way to configure a run without one.
        seed: pseudo-random seed, recorded on the artifact (invariant I2). The
            simulation itself is deterministic; the seed drives interval
            estimation, so two runs differing only in seed agree on every point
            estimate and differ on every interval.
        strategy_config: the strategy's own parameters, as JSON values. Required
            and hashed into ``config_hash``: the engine cannot introspect a
            callable, so a run whose strategy parameters were not declared would
            carry a config hash that silently omits the most important part of
            the configuration. Pass ``{}`` only for a genuinely parameterless
            strategy.
        initial_capital_usd: starting portfolio value in **US dollars**.
        cost_params: the P9.3 cost parameters. Defaults to the conservative,
            **uncalibrated** shipped defaults, whose flag and basis string are
            propagated onto the artifact.
        periods_per_year: annualization factor. 252 for daily bars.
        max_gross_exposure: cap on ``sum(|w|)``. 1.0 by default: leverage is an
            explicit non-goal (§1.1).
        confidence_level: central mass of every reported interval.
        n_bootstrap: bootstrap replicates per metric.
        block_length: bootstrap block length in periods, or ``None`` for the
            ``n ** (1/3)`` rule of thumb.
        git_commit: the commit to stamp. ``None`` resolves it from the checkout
            (see :func:`~backend.backtest.artifact.current_git_commit`), which
            raises rather than guessing if it cannot.
    """

    benchmark: BuyAndHoldSpec
    seed: int
    strategy_config: Mapping[str, JsonValue]
    initial_capital_usd: float = 1_000_000.0
    cost_params: CostModelParams = UNCALIBRATED_DEFAULTS
    periods_per_year: float = 252.0
    max_gross_exposure: float = 1.0
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES
    block_length: int | None = None
    git_commit: str | None = None

    def __post_init__(self) -> None:
        """Validate the run parameters, including the benchmark's own leverage.

        Raises:
            ValueError: if the initial capital is not positive, if
                ``periods_per_year`` is not positive, or if
                ``max_gross_exposure`` is negative.
            WeightError: if the benchmark's gross exposure exceeds
                ``max_gross_exposure``. The benchmark is held to the same
                leverage rule as the strategy: a levered baseline is not a
                buy-and-hold index, and comparing against one silently changes
                what "beat the benchmark" means.
        """
        object.__setattr__(self, "strategy_config", dict(self.strategy_config))
        if self.initial_capital_usd <= 0.0:
            msg = f"initial_capital_usd must be positive; got {self.initial_capital_usd!r}"
            raise ValueError(msg)
        if self.periods_per_year <= 0.0:
            msg = f"periods_per_year must be positive; got {self.periods_per_year!r}"
            raise ValueError(msg)
        if self.max_gross_exposure < 0.0:
            msg = f"max_gross_exposure must be non-negative; got {self.max_gross_exposure!r}"
            raise ValueError(msg)
        benchmark_gross = gross_exposure(self.benchmark.weights)
        if benchmark_gross > self.max_gross_exposure + 1e-9:
            msg = (
                f"benchmark gross exposure {benchmark_gross:.6f} exceeds the run's cap "
                f"{self.max_gross_exposure:.6f}. The baseline is held to the same leverage "
                "rule as the strategy (directive §1.1)."
            )
            raise WeightError(msg)

    @property
    def intervals(self) -> IntervalConfig:
        """Return the interval-estimation configuration this run uses."""
        return IntervalConfig(
            seed=self.seed,
            level=self.confidence_level,
            n_resamples=self.n_bootstrap,
            block_length=self.block_length,
        )

    def hashable_configuration(
        self,
        *,
        data_version: str,
        calendar: Sequence[dt.datetime],
    ) -> dict[str, JsonValue]:
        """Return the canonical mapping that ``config_hash`` is computed over.

        Includes the dataset identity and the calendar's extent alongside every
        run parameter, because a result is only reproducible if the span it was
        computed over is pinned too.

        Args:
            data_version: the source's own dataset identifier.
            calendar: the simulation calendar.

        Returns:
            A JSON-safe mapping suitable for
            :func:`~backend.backtest.artifact.config_hash`.
        """
        return {
            "data_version": data_version,
            "calendar_start": calendar[0].isoformat(),
            "calendar_end": calendar[-1].isoformat(),
            "calendar_points": len(calendar),
            "initial_capital_usd": float(self.initial_capital_usd),
            "periods_per_year": float(self.periods_per_year),
            "max_gross_exposure": float(self.max_gross_exposure),
            "seed": int(self.seed),
            "confidence_level": float(self.confidence_level),
            "n_bootstrap": int(self.n_bootstrap),
            "block_length": self.block_length,
            "benchmark_weights": {
                asset: float(weight) for asset, weight in sorted(self.benchmark.weights.items())
            },
            "benchmark_description": self.benchmark.description,
            "cost_params": {
                "half_spread_bps": float(self.cost_params.half_spread_bps),
                "commission_bps": float(self.cost_params.commission_bps),
                "impact_coefficient": float(self.cost_params.impact_coefficient),
                "default_daily_volatility_bps": float(
                    self.cost_params.default_daily_volatility_bps
                ),
                "borrow_rate_bps_per_year": float(self.cost_params.borrow_rate_bps_per_year),
                "uncalibrated": bool(self.cost_params.uncalibrated),
                "calibration_basis": self.cost_params.calibration_basis,
            },
            "strategy_config": dict(self.strategy_config),
        }


@dataclass(frozen=True, slots=True, eq=False)
class BacktestRun:
    """The output of one engine run.

    Attributes:
        artifact: the reportable record — reproducibility stamp, cost
            provenance, both track records with intervals, and the comparison.
            This is the only thing that should ever be displayed.
        dates: the simulation calendar, ``T + 1`` instants.
        benchmark_asset_returns: per-asset realized returns for the benchmark's
            assets, ``T`` per asset, aligned so entry ``i`` covers
            ``(dates[i], dates[i+1]]``. These are **market facts read through
            the as-of gate**, not strategy results — no strategy return, gross
            or otherwise, is exposed here. They are carried so that walk-forward
            analysis can run a single continuous buy-and-hold across window
            boundaries instead of restarting (and re-charging) the benchmark
            once per window, which would flatter the strategy.
    """

    artifact: RunArtifact
    dates: tuple[dt.datetime, ...]
    benchmark_asset_returns: Mapping[str, tuple[float, ...]]


def validated_calendar(calendar: Sequence[dt.datetime]) -> tuple[dt.datetime, ...]:
    """Return the calendar as a validated tuple of strictly increasing UTC instants.

    Args:
        calendar: the simulation calendar, ``T + 1`` instants.

    Returns:
        The same instants as a tuple.

    Raises:
        CalendarError: if there are fewer than three instants, if any instant is
            naive or not UTC, or if the sequence does not strictly increase.
    """
    dates = tuple(calendar)
    if len(dates) < 3:
        msg = (
            f"a backtest calendar needs at least 3 instants (2 return periods) before any "
            f"dispersion statistic is defined; got {len(dates)}"
        )
        raise CalendarError(msg)
    for index, moment in enumerate(dates):
        _require_utc(moment, label=f"calendar[{index}]")
        if index and moment <= dates[index - 1]:
            msg = (
                f"calendar must strictly increase; calendar[{index}]={moment.isoformat()} "
                f"does not follow {dates[index - 1].isoformat()}"
            )
            raise CalendarError(msg)
    return dates


def _observation_or_raise(
    snapshot: MarketSnapshot,
    asset: str,
    date: dt.datetime,
) -> Observation:
    """Return the observation for ``asset`` on ``date``, or raise a pointed error."""
    observation = snapshot.observation_on(asset, date)
    if observation is None:
        msg = (
            f"no observation for {asset!r} dated {date.isoformat()} is knowable at "
            f"{snapshot.as_of.isoformat()}. The engine does not fill gaps: a missing "
            "period is a data problem (invariant I3), not a zero return."
        )
        raise LookaheadError(msg)
    return observation


def market_parameters(
    snapshot: MarketSnapshot,
    assets: Iterable[str],
) -> tuple[dict[str, float], dict[str, float | None]]:
    """Return ``(adv_usd, daily_volatility_bps)`` for ``assets`` as of ``snapshot``."""
    adv: dict[str, float] = {}
    volatility: dict[str, float | None] = {}
    for asset in assets:
        latest = snapshot.latest(asset)
        if latest is None:
            msg = (
                f"{asset!r} has no observation knowable at {snapshot.as_of.isoformat()}, so its "
                "order cannot be costed"
            )
            raise LookaheadError(msg)
        adv[asset] = latest.adv_usd
        volatility[asset] = latest.daily_volatility_bps
    return adv, volatility


async def run_backtest(
    *,
    data: PointInTimeMarketData,
    calendar: Sequence[dt.datetime],
    strategy: Strategy,
    config: BacktestConfig,
) -> BacktestRun:
    """Run the event-driven daily loop and return a complete, net-of-cost artifact.

    One iteration per period. At ``calendar[i]`` the strategy is handed the
    snapshot for that instant and returns target weights; the resulting trades
    are costed with the P9.3 model; borrow accrues on the post-trade short book
    over the coming period; and only then is the snapshot for ``calendar[i+1]``
    fetched and the realized return applied. The value that would constitute
    lookahead is not in scope while the decision is being made.

    Args:
        data: the point-in-time source. Every read the engine performs goes
            through its ``as_of``.
        calendar: ``T + 1`` strictly increasing timezone-aware UTC instants.
            Decisions are taken at the first ``T``.
        strategy: maps a snapshot to target weights as fractions of portfolio
            value.
        config: run parameters, including the mandatory benchmark and seed.

    Returns:
        A :class:`BacktestRun` whose ``artifact`` carries the I2 stamp, the I4
        cost provenance, the strategy and benchmark track records on identical
        dates, and an interval on every metric.

    Raises:
        CalendarError: if the calendar is unusable.
        LookaheadError: if a required observation is not knowable at the instant
            it is needed.
        WeightError: if the strategy returns unusable weights or breaches the
            gross-exposure cap.
        PortfolioWipeoutError: if a period wipes out the book.
        ReproducibilityError: if the git commit cannot be resolved and none was
            supplied.
    """
    dates = validated_calendar(calendar)
    n_periods = len(dates) - 1
    params = config.cost_params
    benchmark_assets = tuple(sorted(config.benchmark.weights))

    snapshot = await data.as_of(dates[0])
    entry_adv, entry_volatility = market_parameters(snapshot, benchmark_assets)

    weights: dict[str, float] = {}
    value = float(config.initial_capital_usd)
    equity: list[float] = [value]
    execution = ExecutionCosts()
    borrow_total = 0.0
    benchmark_returns: dict[str, list[float]] = {asset: [] for asset in benchmark_assets}

    for index in range(n_periods):
        decision_date = dates[index]
        targets = validate_weights(
            strategy(snapshot),
            known_assets=snapshot.assets,
            max_gross_exposure=config.max_gross_exposure,
            context=f"strategy at {decision_date.isoformat()}",
        )
        changes = {
            asset: targets.get(asset, 0.0) - weights.get(asset, 0.0)
            for asset in set(targets) | set(weights)
        }
        traded = [asset for asset, delta in changes.items() if delta != 0.0]
        adv, volatility = market_parameters(snapshot, traded)
        costs = cost_weight_changes(
            weight_changes=changes,
            portfolio_value_usd=value,
            adv_usd=adv,
            daily_volatility_bps=volatility,
            params=params,
        )
        execution = execution + costs
        value -= costs.total_usd
        if value <= 0.0:
            msg = (
                f"execution costs alone exhausted the portfolio at "
                f"{decision_date.isoformat()}: equity {value!r}"
            )
            raise PortfolioWipeoutError(msg)

        holding_days = (dates[index + 1] - decision_date).total_seconds() / SECONDS_PER_DAY
        borrow = accrue_borrow_usd(
            weights=targets,
            portfolio_value_usd=value,
            holding_period_days=holding_days,
            params=params,
        )
        borrow_total += borrow
        value -= borrow
        if value <= 0.0:
            msg = f"borrow exhausted the portfolio at {decision_date.isoformat()}: equity {value!r}"
            raise PortfolioWipeoutError(msg)

        # ---- The decision is made and paid for. Only now may the next period
        # ---- become knowable: this fetch is the engine's I1 boundary.
        snapshot = await data.as_of(dates[index + 1])
        period_returns: dict[str, float] = {}
        for asset in sorted(set(targets) | set(benchmark_assets)):
            observation = _observation_or_raise(snapshot, asset, dates[index + 1])
            period_returns[asset] = observation.total_return
        for asset in benchmark_assets:
            benchmark_returns[asset].append(period_returns[asset])

        weights, portfolio_return = drift_weights(targets, period_returns)
        value *= 1.0 + portfolio_return
        if value <= 0.0:
            # A backstop, not the main guard: `drift_weights` already refuses a
            # period return of -1 or worse, so reaching here means the equity
            # underflowed to zero after a long run of ruinous but survivable
            # periods. Either way there is no defined return afterwards.
            msg = f"strategy equity reached {value!r} at {dates[index + 1].isoformat()}"
            raise PortfolioWipeoutError(msg)
        equity.append(value)

    strategy_curve = EquityCurve(dates=dates, equity_usd=np.asarray(equity, dtype=np.float64))
    strategy_accounting = RealizedAccounting(
        initial_capital_usd=float(config.initial_capital_usd),
        final_equity_usd=equity[-1],
        half_spread_usd=execution.half_spread_usd,
        commission_usd=execution.commission_usd,
        impact_usd=execution.impact_usd,
        borrow_usd=borrow_total,
        total_cost_usd=execution.total_usd + borrow_total,
        traded_notional_usd=execution.traded_notional_usd,
        n_orders=execution.n_orders,
        n_rebalances=n_periods,
    )
    realized = {asset: tuple(series) for asset, series in benchmark_returns.items()}
    benchmark_curve, benchmark_accounting = buy_and_hold(
        spec=config.benchmark,
        dates=dates,
        asset_returns=realized,
        adv_usd=entry_adv,
        daily_volatility_bps=entry_volatility,
        initial_capital_usd=float(config.initial_capital_usd),
        params=params,
    )
    artifact = _build_artifact(
        config=config,
        data_version=data.data_version,
        dates=dates,
        strategy_curve=strategy_curve,
        strategy_accounting=strategy_accounting,
        benchmark_curve=benchmark_curve,
        benchmark_accounting=benchmark_accounting,
    )
    return BacktestRun(artifact=artifact, dates=dates, benchmark_asset_returns=realized)


def _build_artifact(
    *,
    config: BacktestConfig,
    data_version: str,
    dates: tuple[dt.datetime, ...],
    strategy_curve: EquityCurve,
    strategy_accounting: RealizedAccounting,
    benchmark_curve: EquityCurve,
    benchmark_accounting: RealizedAccounting,
) -> RunArtifact:
    """Assemble the reportable artifact from two finished track records."""
    intervals = config.intervals
    stamp = ReproducibilityStamp(
        git_commit=config.git_commit or current_git_commit(),
        data_version=data_version,
        config_hash=config_hash(
            config.hashable_configuration(data_version=data_version, calendar=dates)
        ),
        seed=config.seed,
    )
    strategy_summary = summarize_track_record(
        equity=strategy_curve,
        accounting=strategy_accounting,
        periods_per_year=config.periods_per_year,
        intervals=intervals,
    )
    benchmark_summary = summarize_track_record(
        equity=benchmark_curve,
        accounting=benchmark_accounting,
        periods_per_year=config.periods_per_year,
        intervals=intervals,
    )
    comparison = compare_to_benchmark(
        strategy=strategy_curve,
        benchmark=benchmark_curve,
        benchmark_description=config.benchmark.description,
        periods_per_year=config.periods_per_year,
        intervals=intervals,
    )
    return RunArtifact(
        stamp=stamp,
        costs=CostProvenance.from_params(config.cost_params),
        strategy=TrackRecord(equity=strategy_curve, summary=strategy_summary),
        benchmark=TrackRecord(equity=benchmark_curve, summary=benchmark_summary),
        comparison=comparison,
    )
