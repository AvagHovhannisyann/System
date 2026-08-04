"""Tests for walk-forward analysis.

The one that matters is :func:`test_no_window_can_see_its_own_test_period`: it
records the snapshot every fit callback actually received and checks, window by
window, that nothing in it is dated at or after the first period that window is
scored on. A walk-forward that trains on its own test data reports an excellent
number and is worthless, and no other assertion in this file would catch it.

The rest fall into three groups: the window arithmetic (a tiling property that
can be checked exhaustively over many parameter combinations), the aggregation
(the chained record must equal the windows it was built from, and the benchmark
must be entered once rather than once per window), and the artifact contract
inherited from the engine (I2 stamp, I4 flags, an interval on every metric, a
benchmark on identical dates).
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.backtest.artifact import Estimate
from backend.backtest.benchmark import BuyAndHoldSpec
from backend.backtest.engine import (
    BacktestConfig,
    CalendarError,
    InjectedMarketData,
    MarketSnapshot,
    Observation,
    Strategy,
)
from backend.backtest.metrics import FloatArray
from backend.backtest.walkforward import (
    WalkForwardAnalysis,
    WalkForwardError,
    WalkForwardScheme,
    run_walk_forward,
)
from backend.costs.model import UNCALIBRATED_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Mapping

DATA_VERSION = "walkforward-fixture-v1"
CAPITAL = 1_000_000.0
ADV = 1_000_000_000.0


def calendar(count: int) -> tuple[dt.datetime, ...]:
    """Return ``count`` consecutive daily UTC instants."""
    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    return tuple(base + dt.timedelta(days=index) for index in range(count))


def _series(dates: tuple[dt.datetime, ...], returns: FloatArray) -> tuple[Observation, ...]:
    return tuple(
        Observation(date=date, knowledge_time=date, total_return=float(value), adv_usd=ADV)
        for date, value in zip(dates, returns, strict=True)
    )


def fixture_data(count: int) -> tuple[tuple[dt.datetime, ...], InjectedMarketData]:
    """Return a calendar and an injected source with two uncorrelated names.

    The returns come from a generator seeded here, in the test, so the series is
    a constant of this file. Nothing in the engine produced them.
    """
    dates = calendar(count)
    rng = np.random.default_rng(19700101)
    return dates, InjectedMarketData(
        observations={
            "AAA": _series(dates, rng.normal(0.0008, 0.012, count)),
            "BMK": _series(dates, rng.normal(0.0004, 0.009, count)),
        },
        data_version=DATA_VERSION,
    )


def make_config(**overrides: object) -> BacktestConfig:
    """Build a run configuration with a single-name buy-and-hold benchmark."""
    kwargs: dict[str, object] = {
        "benchmark": BuyAndHoldSpec(weights={"BMK": 1.0}),
        "seed": 21,
        "strategy_config": {"kind": "always-long"},
        "initial_capital_usd": CAPITAL,
    }
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)  # type: ignore[arg-type]


def always_long(_: MarketSnapshot) -> Mapping[str, float]:
    """Hold the whole book in AAA at every decision point."""
    return {"AAA": 1.0}


def fit_always_long(_: MarketSnapshot) -> Strategy:
    """Return the constant strategy, ignoring the training data entirely."""
    return always_long


async def run(
    *,
    n_dates: int = 40,
    scheme: WalkForwardScheme | None = None,
    config: BacktestConfig | None = None,
) -> WalkForwardAnalysis:
    """Run a walk-forward over the fixture data with the constant strategy."""
    dates, data = fixture_data(n_dates)
    return await run_walk_forward(
        data=data,
        calendar=dates,
        fit=fit_always_long,
        scheme=scheme if scheme is not None else WalkForwardScheme(train_size=10, test_size=5),
        config=config if config is not None else make_config(),
    )


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("train_size", [1, 3, 10, 25])
@pytest.mark.parametrize("test_size", [2, 5, 7])
@pytest.mark.parametrize("purge", [0, 1, 4])
@pytest.mark.parametrize("anchored", [False, True])
def test_windows_tile_the_sample_without_overlap_or_leakage(
    train_size: int, test_size: int, purge: int, *, anchored: bool
) -> None:
    n_observations = 60
    scheme = WalkForwardScheme(
        train_size=train_size, test_size=test_size, purge=purge, anchored=anchored
    )
    windows = scheme.windows(n_observations)
    for position, window in enumerate(windows):
        assert window.index == position
        # Training precedes testing by exactly the purge gap.
        assert window.train_stop + purge == window.test_start
        assert 0 <= window.train_start < window.train_stop
        assert window.n_test == test_size
        assert window.test_stop <= n_observations
        if anchored:
            assert window.train_start == 0
        else:
            assert window.n_train == min(train_size, window.train_stop)
        if position:
            # Test windows are contiguous and never overlap, so the aggregate
            # covers every tested period exactly once.
            assert windows[position - 1].test_stop == window.test_start
            assert windows[position - 1].train_stop < window.train_stop


def test_a_window_that_does_not_fit_is_an_error_rather_than_a_silent_shrink() -> None:
    with pytest.raises(WalkForwardError, match="no walk-forward window fits"):
        WalkForwardScheme(train_size=50, test_size=5).windows(20)


def test_a_scheme_refuses_a_test_window_too_short_for_a_dispersion_statistic() -> None:
    with pytest.raises(WalkForwardError, match="test_size must be at least 2"):
        WalkForwardScheme(train_size=10, test_size=1)


def test_a_scheme_refuses_an_empty_training_set() -> None:
    with pytest.raises(WalkForwardError, match="train_size must be at least 1"):
        WalkForwardScheme(train_size=0, test_size=5)


def test_a_scheme_refuses_a_negative_purge() -> None:
    with pytest.raises(WalkForwardError, match="purge must be non-negative"):
        WalkForwardScheme(train_size=10, test_size=5, purge=-1)


def test_the_scheme_is_part_of_the_runs_identity() -> None:
    assert WalkForwardScheme(train_size=10, test_size=5).to_dict() == {
        "train_size": 10,
        "test_size": 5,
        "purge": 0,
        "anchored": False,
    }


# ---------------------------------------------------------------------------
# The guarantee: no window trains on its own future
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("purge", [0, 2])
@pytest.mark.parametrize("anchored", [False, True])
async def test_no_window_can_see_its_own_test_period(purge: int, *, anchored: bool) -> None:
    dates, data = fixture_data(40)
    seen: list[MarketSnapshot] = []

    def recording_fit(snapshot: MarketSnapshot) -> Strategy:
        seen.append(snapshot)
        return always_long

    scheme = WalkForwardScheme(train_size=10, test_size=5, purge=purge, anchored=anchored)
    analysis = await run_walk_forward(
        data=data,
        calendar=dates,
        fit=recording_fit,
        scheme=scheme,
        config=make_config(),
    )

    assert len(seen) == len(analysis.windows)
    for snapshot, window in zip(seen, analysis.windows, strict=True):
        # The upper bound: the fit sees nothing dated at or after the first
        # instant of its own test period, and nothing purged.
        assert snapshot.as_of == dates[window.train_stop - 1]
        for asset in ("AAA", "BMK"):
            observed = [item.date for item in snapshot.history(asset)]
            assert observed, asset
            assert max(observed) <= dates[window.train_stop - 1]
            assert max(observed) < dates[window.test_start]
            # The lower bound: the window means what it says.
            assert min(observed) == dates[window.train_start]
            assert len(observed) == window.n_train


async def test_the_fit_callback_is_handed_no_route_back_to_the_data_source() -> None:
    # The contract is the argument list: one snapshot, and a snapshot exposes
    # only its own instant. This test pins that surface, because widening it is
    # how a future edit would quietly reintroduce lookahead.
    dates, data = fixture_data(30)
    captured: list[MarketSnapshot] = []

    def recording_fit(snapshot: MarketSnapshot) -> Strategy:
        captured.append(snapshot)
        return always_long

    await run_walk_forward(
        data=data,
        calendar=dates,
        fit=recording_fit,
        scheme=WalkForwardScheme(train_size=8, test_size=4),
        config=make_config(),
    )
    snapshot = captured[0]
    assert not hasattr(snapshot, "as_of_data")
    assert not [name for name in dir(snapshot) if "source" in name or "data_version" in name]


async def test_purged_observations_are_withheld_from_the_fit_but_not_from_trading() -> None:
    dates, data = fixture_data(30)
    training: list[MarketSnapshot] = []
    trading: list[MarketSnapshot] = []

    def recording_fit(snapshot: MarketSnapshot) -> Strategy:
        training.append(snapshot)

        def strategy(inner: MarketSnapshot) -> Mapping[str, float]:
            trading.append(inner)
            return {"AAA": 1.0}

        return strategy

    scheme = WalkForwardScheme(train_size=8, test_size=4, purge=3)
    analysis = await run_walk_forward(
        data=data,
        calendar=dates,
        fit=recording_fit,
        scheme=scheme,
        config=make_config(),
    )
    first = analysis.windows[0]
    purged = dates[first.train_stop : first.test_start]
    assert purged
    fit_dates = {item.date for item in training[0].history("AAA")}
    assert not fit_dates & set(purged)
    # The strategy trades at the instant just before the first tested period and
    # legitimately sees the purged bars: by then they are in its past.
    first_decision = trading[0]
    assert first_decision.as_of == dates[first.test_start - 1]
    assert set(purged) <= {item.date for item in first_decision.history("AAA")}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


async def test_the_aggregate_spans_exactly_the_tested_periods() -> None:
    dates, _ = fixture_data(40)
    analysis = await run()
    windows = analysis.windows
    span = analysis.aggregate.strategy.equity.dates
    assert span[0] == dates[windows[0].test_start - 1]
    assert span[-1] == dates[windows[-1].test_stop - 1]
    assert analysis.aggregate.strategy.equity.n_periods == sum(w.n_test for w in windows)


async def test_the_chained_record_compounds_the_windows_it_was_built_from() -> None:
    analysis = await run()
    per_window = np.concatenate(
        [run_.artifact.strategy.equity.net_returns for run_ in analysis.runs]
    )
    assert analysis.aggregate.strategy.equity.net_returns == pytest.approx(per_window, rel=1e-12)
    # Each window is capitalized at the equity the previous one ended with, so
    # the chained path and the per-window paths agree at the joins.
    assert analysis.aggregate.strategy.equity.equity_usd[-1] == pytest.approx(
        analysis.runs[-1].artifact.strategy.equity.equity_usd[-1], rel=1e-12
    )
    assert analysis.runs[1].artifact.strategy.equity.equity_usd[0] == pytest.approx(
        analysis.runs[0].artifact.strategy.equity.equity_usd[-1], rel=1e-12
    )


async def test_the_aggregate_benchmark_is_entered_once_not_once_per_window() -> None:
    # Charging buy-and-hold a fresh entry cost per window would make the
    # baseline worse and the strategy look better by exactly that amount.
    analysis = await run()
    aggregate = analysis.aggregate.benchmark.summary.accounting
    per_window = [run_.artifact.benchmark.summary.accounting for run_ in analysis.runs]
    assert aggregate.n_orders == 1
    assert aggregate.n_rebalances == 1
    assert aggregate.traded_notional_usd == pytest.approx(CAPITAL)
    # Every window re-enters its own diagnostic benchmark, capitalized at that
    # window's starting equity; the reportable aggregate does not.
    assert sum(item.n_orders for item in per_window) == len(analysis.runs)
    assert aggregate.traded_notional_usd < sum(item.traded_notional_usd for item in per_window)
    assert aggregate.total_cost_usd < sum(item.total_cost_usd for item in per_window)


async def test_the_strategy_pays_a_fresh_entry_cost_in_every_window() -> None:
    # The asymmetry with the benchmark is deliberate: a real retraining
    # schedule does re-enter, so this direction runs against the strategy.
    analysis = await run()
    accounting = analysis.aggregate.strategy.summary.accounting
    assert accounting.n_orders >= len(analysis.runs)
    assert accounting.total_cost_usd == pytest.approx(
        sum(run_.artifact.strategy.summary.accounting.total_cost_usd for run_ in analysis.runs)
    )
    assert accounting.n_rebalances == sum(w.n_test for w in analysis.windows)


async def test_the_window_sharpe_spread_reports_a_range_not_a_point() -> None:
    analysis = await run()
    spread = analysis.window_sharpe_spread
    assert len(spread.strategy) == len(analysis.runs)
    described = spread.strategy.describe()
    assert described["ci_lower"] <= described["median"] <= described["ci_upper"]
    assert set(described) >= {"n_paths", "mean", "median", "min", "max", "ci_lower", "ci_upper"}


async def test_the_window_spread_carries_the_benchmark_beside_the_strategy() -> None:
    # A per-window strategy spread is not reachable without the benchmark's:
    # both are required fields of the same object.
    analysis = await run()
    spread = analysis.window_sharpe_spread
    assert len(spread.benchmark) == len(spread.strategy) == len(analysis.runs)
    assert spread.benchmark.values.tolist() == [
        run_.artifact.benchmark.summary.sharpe.value for run_ in analysis.runs
    ]
    payload = spread.to_dict()
    assert set(payload) == {"metric", "strategy", "benchmark", "note"}
    assert "NOT a CPCV" in str(payload["note"])


# ---------------------------------------------------------------------------
# The artifact contract, inherited from the engine
# ---------------------------------------------------------------------------


async def test_every_window_and_the_aggregate_carry_a_benchmark_on_identical_dates() -> None:
    analysis = await run()
    for run_ in analysis.runs:
        assert run_.artifact.benchmark.equity.dates == run_.artifact.strategy.equity.dates
    assert analysis.aggregate.benchmark.equity.dates == analysis.aggregate.strategy.equity.dates
    assert analysis.aggregate.comparison.benchmark_description
    assert "held across every walk-forward window" in (
        analysis.aggregate.comparison.benchmark_description
    )


async def test_every_aggregate_metric_carries_an_interval() -> None:
    analysis = await run()
    for track in (analysis.aggregate.strategy, analysis.aggregate.benchmark):
        for name, metric in track.summary.metrics.items():
            assert isinstance(metric, Estimate), name
            assert metric.lower <= metric.upper, name
    assert isinstance(analysis.aggregate.comparison.information_ratio, Estimate)


async def test_the_aggregate_carries_the_uncalibrated_cost_flag() -> None:
    analysis = await run()
    payload = analysis.aggregate.to_dict()
    assert payload["uncalibrated"] is True
    assert any("UNCALIBRATED" in statement for statement in analysis.aggregate.disclosures)
    assert analysis.aggregate.costs.parameters["half_spread_bps"] == (
        UNCALIBRATED_DEFAULTS.half_spread_bps
    )


async def test_the_aggregate_stamp_pins_the_scheme_as_well_as_the_run() -> None:
    rolling = await run(scheme=WalkForwardScheme(train_size=10, test_size=5, anchored=False))
    anchored = await run(scheme=WalkForwardScheme(train_size=10, test_size=5, anchored=True))
    assert rolling.aggregate.stamp.config_hash != anchored.aggregate.stamp.config_hash
    assert rolling.aggregate.stamp.data_version == DATA_VERSION
    assert len(rolling.aggregate.stamp.git_commit) in {40, 64}


async def test_identical_inputs_reproduce_an_identical_walk_forward() -> None:
    first = await run()
    second = await run()
    assert first.aggregate.results_digest() == second.aggregate.results_digest()
    assert [r.artifact.results_digest() for r in first.runs] == [
        r.artifact.results_digest() for r in second.runs
    ]


async def test_changing_the_seed_changes_the_walk_forward_intervals_only() -> None:
    first = await run(config=make_config(seed=1))
    second = await run(config=make_config(seed=2))
    assert first.aggregate.results_digest() != second.aggregate.results_digest()
    assert first.aggregate.strategy.equity.equity_usd == pytest.approx(
        second.aggregate.strategy.equity.equity_usd, rel=1e-12
    )
    for name, estimate in first.aggregate.strategy.summary.metrics.items():
        other = second.aggregate.strategy.summary.metrics[name]
        assert estimate.value == other.value, name
        assert (estimate.lower, estimate.upper) != (other.lower, other.upper), name


async def test_the_untested_tail_is_reported_rather_than_dropped_in_silence() -> None:
    # 40 positions, first test starting at 10, windows of 5: 10..35 are tested
    # and positions 35..39 are not. An operator comparing this against a
    # full-sample result has to be told the spans differ.
    analysis = await run(n_dates=40, scheme=WalkForwardScheme(train_size=10, test_size=5))
    assert analysis.n_calendar_points == 40
    assert analysis.untested_tail == 40 - analysis.windows[-1].test_stop
    assert 0 <= analysis.untested_tail < 5
    assert analysis.to_dict()["untested_tail_positions"] == analysis.untested_tail


async def test_the_payload_carries_every_window_and_the_aggregate() -> None:
    analysis = await run()
    payload = analysis.to_dict()
    windows = payload["windows"]
    artifacts = payload["window_artifacts"]
    assert isinstance(windows, list)
    assert isinstance(artifacts, list)
    assert len(windows) == len(artifacts) == len(analysis.runs)
    assert payload["scheme"] == analysis.scheme.to_dict()
    assert isinstance(payload["aggregate"], dict)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


async def test_a_calendar_that_goes_backwards_is_refused_before_any_window_runs() -> None:
    dates, data = fixture_data(30)
    scrambled = (*dates[:5], dates[6], dates[5], *dates[7:])
    with pytest.raises(CalendarError, match="strictly increase"):
        await run_walk_forward(
            data=data,
            calendar=scrambled,
            fit=fit_always_long,
            scheme=WalkForwardScheme(train_size=8, test_size=4),
            config=make_config(),
        )


async def test_a_calendar_too_short_for_any_window_is_refused() -> None:
    dates, data = fixture_data(10)
    with pytest.raises(WalkForwardError, match="no walk-forward window fits"):
        await run_walk_forward(
            data=data,
            calendar=dates,
            fit=fit_always_long,
            scheme=WalkForwardScheme(train_size=20, test_size=5),
            config=make_config(),
        )
