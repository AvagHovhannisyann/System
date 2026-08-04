"""Walk-forward analysis (P10.5): refit forward, never backward.

The scheme
----------

The calendar is cut into contiguous test windows that **tile** the usable
sample. Before each one, the strategy is refit on data that ends strictly
earlier, separated by an optional **purge** gap. Window ``w``'s test period is
``[test_start_w, test_stop_w)`` and ``test_stop_w == test_start_{w+1}``, so the
concatenated out-of-sample return series covers every tested period exactly
once. There is no ``step`` parameter: overlapping test windows would double
count observations in that concatenation, and a "walk-forward Sharpe" computed
over double-counted periods is not a statistic of anything.

Why the embargo is absent
-------------------------

:mod:`backend.backtest.cpcv` applies both a purge and a one-sided **embargo**,
and it needs both: with combinatorial splits, training data can sit *after* a
test block, so leakage flows forward through serial correlation. In a strictly
forward walk-forward it cannot — every training observation for window ``w``
precedes window ``w``'s test period. What remains is the *label overlap* leak: a
training observation whose outcome is only resolved during the test period.
That is what ``purge`` removes, in observations. Adding an embargo here would be
cargo-culting a control from a setting where it does something.

The guarantee that matters
--------------------------

"Never train on future data" is enforced by construction rather than by review.
The fit callback receives a :class:`~backend.backtest.engine.MarketSnapshot`
taken at the last *training* instant — the same object a strategy gets during
the simulation, with the same self-verifying point-in-time filter. It has no
handle on the data source, so there is no call it could make to see the test
period. The window arithmetic then only has to place that instant correctly,
and that is a property a test can check exhaustively.

The snapshot is additionally **narrowed from below** to the window's own
``[train_start, train_stop)`` span, so a rolling scheme fits on exactly the
``train_size`` positions it claims to and an anchored one on everything from
position 0. Without that narrowing, ``train_size`` and ``anchored`` would be
hashed into the run's configuration while changing nothing about the fit — a
parameter that documents a decision the code does not take. Narrowing is safe in
a way that widening never is: discarding old observations cannot manufacture
lookahead, so only the *upper* bound needs the snapshot's verification.

**Purge governs the fit, not the trading.** Observations in the purge gap are
withheld from the training snapshot, but the simulated strategy still sees them
when it trades — by then they are in its past, and hiding a knowable fact from a
decision would model a different system rather than a safer one.

Aggregation, and the one thing that would have flattered the strategy
---------------------------------------------------------------------

Per-window results are chained: window ``w + 1`` starts at the equity window
``w`` ended with, and starts **flat**, so it pays a fresh entry cost. That is
conservative and it is what a real retraining schedule does.

The aggregate **benchmark**, however, is a *single continuous buy-and-hold*
across the whole out-of-sample span, entered once. Chaining per-window
benchmarks instead would charge buy-and-hold an entry cost per window for
trading it never does, making the baseline worse and the strategy look better by
exactly that amount. The asymmetry is deliberate and runs against the strategy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from backend.backtest.artifact import (
    CostProvenance,
    EquityCurve,
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
from backend.backtest.benchmark import buy_and_hold
from backend.backtest.cpcv import PathDistribution
from backend.backtest.engine import (
    BacktestRun,
    market_parameters,
    run_backtest,
    validated_calendar,
)
from backend.backtest.metrics import FloatArray

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from backend.backtest.engine import (
        BacktestConfig,
        MarketSnapshot,
        PointInTimeMarketData,
        Strategy,
    )

__all__ = [
    "StrategyFactory",
    "WalkForwardAnalysis",
    "WalkForwardError",
    "WalkForwardScheme",
    "WalkForwardWindow",
    "WindowSpread",
    "run_walk_forward",
]


class WalkForwardError(ValueError):
    """Raised when a walk-forward scheme cannot be applied to a sample.

    Covers a scheme whose windows do not fit the calendar at all, and parameter
    combinations that would produce overlapping or degenerate windows. Fatal:
    silently shrinking a window to make it fit changes what is being measured.
    """


type StrategyFactory = Callable[[MarketSnapshot], Strategy]
"""Fit a strategy from a point-in-time snapshot of one training window.

The single argument is the whole contract, and it is what makes "never train on
future data" structural: the callback holds no reference to the data source and
therefore cannot request any other instant. The snapshot's ``as_of`` is the last
training instant, and its observations are narrowed to the window's own
``[train_start, train_stop)`` span.
"""


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One train/test window, in positional indices into the calendar.

    All bounds are half-open ``[start, stop)`` positions in the calendar the
    scheme was applied to.

    Attributes:
        index: 0-based position of this window in the walk.
        train_start: first training position.
        train_stop: one past the last training position.
        test_start: first tested position. ``train_stop + purge == test_start``.
        test_stop: one past the last tested position.
    """

    index: int
    train_start: int
    train_stop: int
    test_start: int
    test_stop: int

    @property
    def n_train(self) -> int:
        """Return the number of training observations."""
        return self.train_stop - self.train_start

    @property
    def n_test(self) -> int:
        """Return the number of tested observations."""
        return self.test_stop - self.test_start

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the window as a JSON-safe mapping."""
        return {
            "index": self.index,
            "train_start": self.train_start,
            "train_stop": self.train_stop,
            "test_start": self.test_start,
            "test_stop": self.test_stop,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardScheme:
    """How the calendar is cut into successive train/test windows.

    All sizes count **calendar positions**, not observations in the source. The
    two coincide for a daily bar feed dated on the simulation calendar, which is
    the case this engine is built for; a source carrying observations at instants
    that are not calendar positions would put more of them inside the same window
    span, so the fit would see more rows than ``train_size`` suggests. Stated
    because the alternative — counting rows — would make the window boundaries
    depend on which assets happen to have data, and two windows of the same
    nominal size would then cover different stretches of time.

    Attributes:
        train_size: training positions per window. For an anchored walk this
            is the size of the *first* window's training set; later windows
            expand back to position 0.
        test_size: tested positions per window, one return period each. Must be
            at least 2, because a window with one return period has no defined
            dispersion and therefore no interval.
        purge: positions dropped between the end of training and the start of
            testing. Set it to the label horizon: an observation whose label
            resolves over the next ``h`` periods must not be trained on if any
            part of that horizon falls in the test window. Default 0, which
            assumes point-in-time labels. Purged positions are withheld from the
            fit only — the simulated strategy still sees them when it trades,
            because by then they are in its past.
        anchored: ``True`` for an expanding training window starting at position
            0; ``False`` (default) for a rolling window of exactly
            ``train_size``. Anchored uses more history and adapts more slowly;
            rolling adapts faster and discards regimes. Neither is right in
            general, which is why it is a parameter and is hashed into the run's
            config hash.
    """

    train_size: int
    test_size: int
    purge: int = 0
    anchored: bool = False

    def __post_init__(self) -> None:
        """Validate the scheme.

        Raises:
            WalkForwardError: if the training size is below 1, the test size
                below 2, or the purge negative.
        """
        if self.train_size < 1:
            msg = f"train_size must be at least 1 position; got {self.train_size!r}"
            raise WalkForwardError(msg)
        if self.test_size < 2:
            msg = (
                f"test_size must be at least 2 positions so the window has a defined "
                f"dispersion and therefore an interval; got {self.test_size!r}"
            )
            raise WalkForwardError(msg)
        if self.purge < 0:
            msg = f"purge must be non-negative; got {self.purge!r}"
            raise WalkForwardError(msg)

    def windows(self, n_positions: int) -> tuple[WalkForwardWindow, ...]:
        """Return the tiling of ``n_positions`` into successive windows.

        Args:
            n_positions: number of calendar positions available, which for a
                backtest calendar of ``T + 1`` instants is ``T`` return periods
                plus the one instant they start from — pass ``len(calendar)``.

        Returns:
            The windows, in walk order. Test windows are contiguous and
            non-overlapping, and every training set ends at least ``purge``
            positions before its test set begins. A remainder too short for one
            more window is left untested and reported by
            :attr:`WalkForwardAnalysis.untested_tail`.

        Raises:
            WalkForwardError: if not even one window fits.
        """
        windows: list[WalkForwardWindow] = []
        test_start = self.train_size + self.purge
        while test_start + self.test_size <= n_positions:
            train_stop = test_start - self.purge
            train_start = 0 if self.anchored else max(0, train_stop - self.train_size)
            windows.append(
                WalkForwardWindow(
                    index=len(windows),
                    train_start=train_start,
                    train_stop=train_stop,
                    test_start=test_start,
                    test_stop=test_start + self.test_size,
                )
            )
            test_start += self.test_size
        if not windows:
            msg = (
                f"no walk-forward window fits: train_size={self.train_size} + "
                f"purge={self.purge} + test_size={self.test_size} exceeds the "
                f"{n_positions} available calendar positions"
            )
            raise WalkForwardError(msg)
        return tuple(windows)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the scheme as a JSON-safe mapping, for the run's config hash."""
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "purge": self.purge,
            "anchored": self.anchored,
        }


@dataclass(frozen=True, slots=True, eq=False)
class WindowSpread:
    """How a metric varied across windows — strategy and benchmark together.

    Both sides are required fields, for the same reason
    :class:`~backend.backtest.artifact.RunArtifact` requires a benchmark: "the
    strategy's Sharpe ranged from -0.2 to 1.4 across windows" is not a
    statement until the same sentence exists for buy-and-hold over the same
    windows. A per-window strategy spread is not reachable from this type
    without the benchmark's.

    Attributes:
        metric: the metric these distributions describe, named so a display
            cannot mislabel it.
        strategy: one value per window, in walk order, from the strategy's
            net-of-cost record.
        benchmark: one value per window, in walk order, from the benchmark's
            net-of-cost record over the same window dates.
    """

    metric: str
    strategy: PathDistribution
    benchmark: PathDistribution

    def to_dict(self) -> dict[str, JsonValue]:
        """Return both distributions as a JSON-safe mapping."""
        return {
            "metric": self.metric,
            "strategy": {key: float(value) for key, value in self.strategy.describe().items()},
            "benchmark": {key: float(value) for key, value in self.benchmark.describe().items()},
            "note": (
                "spread across disjoint walk-forward sub-periods of one path — NOT a CPCV "
                "path distribution (backend.backtest.cpcv) and not a correction for trial "
                "count (backend.backtest.dsr)"
            ),
        }


@dataclass(frozen=True, slots=True, eq=False)
class WalkForwardAnalysis:
    """The result of a walk-forward run: every window, and the chained whole.

    Attributes:
        scheme: the scheme that produced the windows.
        windows: the windows, in walk order.
        runs: one :class:`~backend.backtest.engine.BacktestRun` per window, each
            with its own artifact, its own benchmark over its own dates, and its
            own intervals.
        aggregate: the chained out-of-sample record — strategy returns
            concatenated across windows, benchmark entered once and held
            throughout. This is the number to report; a per-window figure is a
            diagnostic.
        n_calendar_points: length of the calendar the scheme was applied to.
            Carried so :attr:`untested_tail` can state how much of the sample
            the windows did not reach.
    """

    scheme: WalkForwardScheme
    windows: tuple[WalkForwardWindow, ...]
    runs: tuple[BacktestRun, ...]
    aggregate: RunArtifact
    n_calendar_points: int

    @property
    def untested_tail(self) -> int:
        """Return the calendar positions after the last window, in positions.

        A whole number of test windows rarely divides a sample exactly, so the
        last ``test_size - 1`` or fewer positions are usually untested. Reported
        rather than dropped in silence: an operator comparing a walk-forward
        result against a full-sample one needs to know the two do not cover the
        same span, and a growing tail is a sign the scheme no longer fits the
        data it is being applied to.
        """
        return self.n_calendar_points - self.windows[-1].test_stop

    @property
    def window_sharpe_spread(self) -> WindowSpread:
        """Return the per-window annualized Sharpe ratios of both sides.

        Reuses :class:`~backend.backtest.cpcv.PathDistribution` for its interval
        reporting, but the two are not the same object of study: CPCV paths are
        ``C(N-1, k-1)`` complete resamplings of the *same* sample, while these
        are ``W`` disjoint sub-periods of one path. This describes how the
        result varied across time, not how it would vary across resamplings, and
        it must never be presented as a CPCV distribution.
        """
        return WindowSpread(
            metric="annualized_sharpe_ratio_net_of_costs",
            strategy=PathDistribution(
                values=np.asarray(
                    [run.artifact.strategy.summary.sharpe.value for run in self.runs],
                    dtype=np.float64,
                )
            ),
            benchmark=PathDistribution(
                values=np.asarray(
                    [run.artifact.benchmark.summary.sharpe.value for run in self.runs],
                    dtype=np.float64,
                )
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the analysis as a JSON-safe mapping."""
        return {
            "scheme": self.scheme.to_dict(),
            "windows": [window.to_dict() for window in self.windows],
            "window_artifacts": [run.artifact.to_dict() for run in self.runs],
            "window_sharpe_spread": self.window_sharpe_spread.to_dict(),
            "n_calendar_points": self.n_calendar_points,
            "untested_tail_positions": self.untested_tail,
            "aggregate": self.aggregate.to_dict(),
        }


def _chain_equity(
    *,
    initial_capital_usd: float,
    net_returns: Sequence[float] | FloatArray,
) -> FloatArray:
    """Return the equity path obtained by compounding ``net_returns`` from capital."""
    growth = np.cumprod(1.0 + np.asarray(net_returns, dtype=np.float64))
    path = np.concatenate([np.ones(1, dtype=np.float64), growth]) * initial_capital_usd
    return np.asarray(path, dtype=np.float64)


def _sum_accounting(
    accountings: Sequence[RealizedAccounting],
    *,
    initial_capital_usd: float,
    final_equity_usd: float,
) -> RealizedAccounting:
    """Return the componentwise total of per-window accountings."""
    return RealizedAccounting(
        initial_capital_usd=initial_capital_usd,
        final_equity_usd=final_equity_usd,
        half_spread_usd=sum(item.half_spread_usd for item in accountings),
        commission_usd=sum(item.commission_usd for item in accountings),
        impact_usd=sum(item.impact_usd for item in accountings),
        borrow_usd=sum(item.borrow_usd for item in accountings),
        total_cost_usd=sum(item.total_cost_usd for item in accountings),
        traded_notional_usd=sum(item.traded_notional_usd for item in accountings),
        n_orders=sum(item.n_orders for item in accountings),
        n_rebalances=sum(item.n_rebalances for item in accountings),
    )


async def run_walk_forward(
    *,
    data: PointInTimeMarketData,
    calendar: Sequence[dt.datetime],
    fit: StrategyFactory,
    scheme: WalkForwardScheme,
    config: BacktestConfig,
) -> WalkForwardAnalysis:
    """Refit and test forward across the calendar, then chain the results.

    For each window: the strategy is fit on a snapshot taken at the last
    training instant and narrowed to the window's training span, then simulated
    over the window's test period with
    :func:`~backend.backtest.engine.run_backtest`. Each window starts flat and
    is capitalized at the equity the previous window ended with, so compounding
    and the size-dependence of market impact both carry through the walk.

    Args:
        data: the point-in-time source; every read goes through its ``as_of``.
        calendar: ``T + 1`` strictly increasing timezone-aware UTC instants.
        fit: builds a strategy from the training snapshot. Called once per
            window.
        scheme: how to cut the calendar.
        config: run parameters. ``initial_capital_usd`` capitalizes the first
            window; later windows inherit the running equity.

    Returns:
        A :class:`WalkForwardAnalysis` holding every window's artifact and the
        chained aggregate — the strategy's concatenated out-of-sample record
        against a single continuous buy-and-hold over the same span.

    Raises:
        CalendarError: if the calendar is unusable.
        WalkForwardError: if the scheme does not fit the calendar.
        LookaheadError: propagated from the engine when a required observation
            is not knowable when it is needed.
    """
    dates = validated_calendar(calendar)
    windows = scheme.windows(len(dates))

    runs: list[BacktestRun] = []
    capital = float(config.initial_capital_usd)
    for window in windows:
        # Upper bound: an as-of read at the last training instant, verified by
        # the snapshot itself. Lower bound: the window's own start, so a rolling
        # scheme fits on exactly `train_size` observations.
        training_snapshot = (await data.as_of(dates[window.train_stop - 1])).since(
            dates[window.train_start]
        )
        window_config = replace(config, initial_capital_usd=capital)
        run = await run_backtest(
            data=data,
            calendar=dates[window.test_start - 1 : window.test_stop],
            strategy=fit(training_snapshot),
            config=window_config,
        )
        runs.append(run)
        capital = float(run.artifact.strategy.equity.equity_usd[-1])

    aggregate = await _aggregate(
        data=data,
        dates=dates,
        windows=windows,
        runs=runs,
        scheme=scheme,
        config=config,
    )
    return WalkForwardAnalysis(
        scheme=scheme,
        windows=windows,
        runs=tuple(runs),
        aggregate=aggregate,
        n_calendar_points=len(dates),
    )


async def _aggregate(
    *,
    data: PointInTimeMarketData,
    dates: tuple[dt.datetime, ...],
    windows: tuple[WalkForwardWindow, ...],
    runs: Sequence[BacktestRun],
    scheme: WalkForwardScheme,
    config: BacktestConfig,
) -> RunArtifact:
    """Chain per-window results into one out-of-sample artifact."""
    span = dates[windows[0].test_start - 1 : windows[-1].test_stop]
    net_returns = np.concatenate([run.artifact.strategy.equity.net_returns for run in runs])
    strategy_curve = EquityCurve(
        dates=span,
        equity_usd=_chain_equity(
            initial_capital_usd=float(config.initial_capital_usd),
            net_returns=net_returns,
        ),
    )
    strategy_accounting = _sum_accounting(
        [run.artifact.strategy.summary.accounting for run in runs],
        initial_capital_usd=float(config.initial_capital_usd),
        final_equity_usd=float(strategy_curve.equity_usd[-1]),
    )

    benchmark_assets = tuple(sorted(config.benchmark.weights))
    asset_returns = {
        asset: tuple(value for run in runs for value in run.benchmark_asset_returns.get(asset, ()))
        for asset in benchmark_assets
    }
    entry_snapshot = await data.as_of(span[0])
    entry_adv, entry_volatility = market_parameters(entry_snapshot, benchmark_assets)
    benchmark_curve, benchmark_accounting = buy_and_hold(
        spec=config.benchmark,
        dates=span,
        asset_returns=asset_returns,
        adv_usd=entry_adv,
        daily_volatility_bps=entry_volatility,
        initial_capital_usd=float(config.initial_capital_usd),
        params=config.cost_params,
    )

    intervals = config.intervals
    hashable: dict[str, JsonValue] = dict(
        config.hashable_configuration(data_version=data.data_version, calendar=span)
    )
    hashable["walk_forward_scheme"] = scheme.to_dict()
    stamp = ReproducibilityStamp(
        git_commit=config.git_commit or current_git_commit(),
        data_version=data.data_version,
        config_hash=config_hash(hashable),
        seed=config.seed,
    )
    return RunArtifact(
        stamp=stamp,
        costs=CostProvenance.from_params(config.cost_params),
        strategy=TrackRecord(
            equity=strategy_curve,
            summary=summarize_track_record(
                equity=strategy_curve,
                accounting=strategy_accounting,
                periods_per_year=config.periods_per_year,
                intervals=intervals,
            ),
        ),
        benchmark=TrackRecord(
            equity=benchmark_curve,
            summary=summarize_track_record(
                equity=benchmark_curve,
                accounting=benchmark_accounting,
                periods_per_year=config.periods_per_year,
                intervals=intervals,
            ),
        ),
        comparison=compare_to_benchmark(
            strategy=strategy_curve,
            benchmark=benchmark_curve,
            benchmark_description=(
                f"{config.benchmark.description}; entered once at the start of the "
                "out-of-sample span and held across every walk-forward window"
            ),
            periods_per_year=config.periods_per_year,
            intervals=intervals,
        ),
    )
