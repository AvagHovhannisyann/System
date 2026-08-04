"""Tests for the event-driven backtest loop.

Three groups, in order of importance:

1. **Lookahead is unreachable.** Including the case that matters most — a data
   source that filters incorrectly is caught at the snapshot boundary rather
   than trusted.
2. **Costs strictly reduce returns.** Asserted against an otherwise identical
   run with a zeroed cost model, which is the only way to compare against a
   gross figure without ever reporting one.
3. **The arithmetic is hand-computable.** Every expected number below is
   derived in a comment from the cost model's documented formulas, not from a
   second implementation of the engine.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.backtest.artifact import Estimate, ReproducibilityError
from backend.backtest.benchmark import (
    BuyAndHoldSpec,
    PortfolioWipeoutError,
    WeightError,
    buy_and_hold,
)
from backend.backtest.engine import (
    BacktestConfig,
    CalendarError,
    InjectedMarketData,
    LookaheadError,
    MarketSnapshot,
    Observation,
    run_backtest,
)
from backend.costs.model import UNCALIBRATED_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DATA_VERSION = "test-injected-v1"
CAPITAL = 1_000_000.0
ADV = 1_000_000_000.0

# One entry trade of the full book into a name at 0.1% of ADV:
#   participation = 1e6 / 1e9                       = 1e-3
#   impact        = 1.0 * 200 bps * sqrt(1e-3)      = 6.324555320336759 bps
#   total         = 5 (half spread) + 1 (commission) + impact
ENTRY_COST_BPS = 5.0 + 1.0 + 200.0 * np.sqrt(1e-3)
ENTRY_COST_USD = ENTRY_COST_BPS / 10_000.0 * CAPITAL


def calendar(count: int) -> tuple[dt.datetime, ...]:
    """Return ``count`` consecutive daily UTC instants."""
    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    return tuple(base + dt.timedelta(days=index) for index in range(count))


def observations(
    dates: Sequence[dt.datetime],
    returns: Sequence[float],
    *,
    adv_usd: float = ADV,
    knowledge_lag_days: float = 0.0,
) -> tuple[Observation, ...]:
    """Build one observation per date, the first being pre-simulation history."""
    lag = dt.timedelta(days=knowledge_lag_days)
    return tuple(
        Observation(
            date=date,
            knowledge_time=date + lag,
            total_return=float(value),
            adv_usd=adv_usd,
        )
        for date, value in zip(dates, returns, strict=True)
    )


def injected(**series: tuple[Observation, ...]) -> InjectedMarketData:
    """Build an injected source from named observation series."""
    return InjectedMarketData(observations=dict(series), data_version=DATA_VERSION)


def config(**overrides: object) -> BacktestConfig:
    """Build a run configuration with a single-name buy-and-hold benchmark."""
    kwargs: dict[str, object] = {
        "benchmark": BuyAndHoldSpec(weights={"BMK": 1.0}),
        "seed": 42,
        "strategy_config": {"name": "always-long"},
        "initial_capital_usd": CAPITAL,
    }
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)  # type: ignore[arg-type]


def always_long(_: MarketSnapshot) -> Mapping[str, float]:
    """Hold the whole book in AAA at every decision point."""
    return {"AAA": 1.0}


def _standard_data() -> InjectedMarketData:
    dates = calendar(4)
    return injected(
        # The leading value is history preceding the simulation: it establishes
        # ADV at the first decision instant and is never applied to the book.
        AAA=observations(dates, [0.004, 0.10, -0.20, 0.25]),
        BMK=observations(dates, [0.001, 0.01, 0.02, -0.015]),
    )


def _long_run_data(count: int) -> tuple[tuple[dt.datetime, ...], InjectedMarketData]:
    """Return a calendar and a source with enough periods for a real bootstrap.

    The returns are drawn from a fixed generator seeded here in the test, so the
    series is a constant of this file rather than anything the engine produced.
    """
    dates = calendar(count)
    rng = np.random.default_rng(20240401)
    return dates, injected(
        AAA=observations(dates, rng.normal(0.0006, 0.011, count).tolist()),
        BMK=observations(dates, rng.normal(0.0003, 0.008, count).tolist()),
    )


# ---------------------------------------------------------------------------
# Lookahead is structurally unreachable
# ---------------------------------------------------------------------------


async def test_a_snapshot_never_contains_anything_dated_after_its_instant() -> None:
    dates = calendar(4)
    data = _standard_data()
    for index, moment in enumerate(dates):
        snapshot = await data.as_of(moment)
        assert len(snapshot.history("AAA")) == index + 1
        assert all(item.date <= moment for item in snapshot.history("AAA"))


async def test_a_late_publication_is_invisible_until_it_is_published() -> None:
    dates = calendar(4)
    data = injected(AAA=observations(dates, [0.0, 0.1, 0.1, 0.1], knowledge_lag_days=2.0))
    # The bar dated day 1 only becomes knowable on day 3.
    assert await_history_length(await data.as_of(dates[1]), "AAA") == 0
    assert await_history_length(await data.as_of(dates[2]), "AAA") == 1
    assert await_history_length(await data.as_of(dates[3]), "AAA") == 2


def await_history_length(snapshot: MarketSnapshot, asset: str) -> int:
    """Return how many observations for ``asset`` the snapshot holds."""
    return len(snapshot.history(asset))


def test_a_snapshot_refuses_a_bar_dated_after_its_own_instant() -> None:
    dates = calendar(3)
    future = Observation(date=dates[2], knowledge_time=dates[2], total_return=0.01, adv_usd=ADV)
    with pytest.raises(LookaheadError, match="in its future"):
        MarketSnapshot(as_of=dates[0], observations={"AAA": (future,)})


def test_a_snapshot_refuses_a_past_bar_that_was_published_later() -> None:
    # The other lookahead branch, and the one a real feed produces: the period
    # is over, but the vendor had not published it at the simulated instant.
    dates = calendar(3)
    late = Observation(date=dates[0], knowledge_time=dates[2], total_return=0.01, adv_usd=ADV)
    with pytest.raises(LookaheadError, match="only became knowable"):
        MarketSnapshot(as_of=dates[1], observations={"AAA": (late,)})


def test_an_observation_cannot_claim_to_be_known_before_its_period_ended() -> None:
    dates = calendar(3)
    with pytest.raises(LookaheadError, match="precedes the period end"):
        Observation(date=dates[2], knowledge_time=dates[0], total_return=0.01, adv_usd=ADV)


@dataclass(frozen=True, slots=True)
class _LeakySource:
    """A deliberately broken source that ignores the as-of filter entirely.

    This is the failure the snapshot's self-verification exists for: the source
    looks like a valid implementation of the protocol and returns the future.
    """

    everything: Mapping[str, tuple[Observation, ...]]

    @property
    def data_version(self) -> str:
        return "leaky-v1"

    async def as_of(self, as_of_ts: dt.datetime) -> MarketSnapshot:
        return MarketSnapshot(as_of=as_of_ts, observations=self.everything)


async def test_a_source_that_leaks_the_future_is_caught_at_the_boundary() -> None:
    dates = calendar(4)
    leaky = _LeakySource(
        everything={
            "AAA": observations(dates, [0.0, 0.1, -0.1, 0.2]),
            "BMK": observations(dates, [0.0, 0.01, 0.01, 0.01]),
        }
    )
    with pytest.raises(LookaheadError, match="in its future"):
        await run_backtest(data=leaky, calendar=dates, strategy=always_long, config=config())


async def test_a_strategy_cannot_position_in_an_asset_it_cannot_see() -> None:
    dates = calendar(4)
    data = _standard_data()

    def reaches_for_the_unknown(_: MarketSnapshot) -> Mapping[str, float]:
        return {"NOT-IN-UNIVERSE": 1.0}

    with pytest.raises(WeightError, match="lookahead"):
        await run_backtest(
            data=data, calendar=dates, strategy=reaches_for_the_unknown, config=config()
        )


async def test_a_missing_period_is_an_error_rather_than_a_zero_return() -> None:
    dates = calendar(4)
    # AAA stops reporting after the second date; the engine must not infer 0.
    data = injected(
        AAA=observations(dates[:2], [0.004, 0.10]),
        BMK=observations(dates, [0.001, 0.01, 0.02, -0.015]),
    )
    with pytest.raises(LookaheadError, match="does not fill gaps"):
        await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())


# ---------------------------------------------------------------------------
# Hand-computable arithmetic
# ---------------------------------------------------------------------------


async def test_equity_and_drawdown_match_hand_computation() -> None:
    dates = calendar(4)
    run = await run_backtest(
        data=_standard_data(), calendar=dates, strategy=always_long, config=config()
    )
    equity = run.artifact.strategy.equity.equity_usd

    # Entry: the whole book crosses into AAA once.
    after_entry = CAPITAL - ENTRY_COST_USD
    # A fully invested single-name book needs no further trade: the position's
    # weight after a return is w(1+r)/(1+w*r) = 1 for w = 1.
    expected = [
        CAPITAL,
        after_entry * 1.10,
        after_entry * 1.10 * 0.80,
        after_entry * 1.10 * 0.80 * 1.25,
    ]
    assert equity == pytest.approx(expected, rel=1e-12)

    # Peak 1.10, trough 1.10 * 0.80 -> drawdown exactly 20%.
    assert run.artifact.strategy.summary.max_drawdown.value == pytest.approx(0.20, rel=1e-12)


async def test_a_fully_invested_single_name_book_places_exactly_one_order() -> None:
    run = await run_backtest(
        data=_standard_data(),
        calendar=calendar(4),
        strategy=always_long,
        config=config(),
    )
    accounting = run.artifact.strategy.summary.accounting
    assert accounting.n_orders == 1
    assert accounting.n_rebalances == 3
    assert accounting.traded_notional_usd == pytest.approx(CAPITAL)


async def test_the_cost_waterfall_matches_the_documented_component_formulas() -> None:
    run = await run_backtest(
        data=_standard_data(),
        calendar=calendar(4),
        strategy=always_long,
        config=config(),
    )
    accounting = run.artifact.strategy.summary.accounting
    assert accounting.half_spread_usd == pytest.approx(5.0 / 10_000.0 * CAPITAL)
    assert accounting.commission_usd == pytest.approx(1.0 / 10_000.0 * CAPITAL)
    assert accounting.impact_usd == pytest.approx(200.0 * np.sqrt(1e-3) / 10_000.0 * CAPITAL)
    assert accounting.borrow_usd == 0.0
    assert accounting.total_cost_usd == pytest.approx(ENTRY_COST_USD)


async def test_a_short_book_accrues_borrow_while_a_long_book_does_not() -> None:
    dates = calendar(4)
    data = _standard_data()

    def short_aaa(_: MarketSnapshot) -> Mapping[str, float]:
        return {"AAA": -0.5}

    shorted = await run_backtest(data=data, calendar=dates, strategy=short_aaa, config=config())
    longed = await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())
    assert shorted.artifact.strategy.summary.accounting.borrow_usd > 0.0
    assert longed.artifact.strategy.summary.accounting.borrow_usd == 0.0


# ---------------------------------------------------------------------------
# I4: costs strictly reduce returns
# ---------------------------------------------------------------------------


def _free_params() -> object:
    """Return a cost parameter set with every component zeroed.

    Used only to establish the *upper bound* a costed run must fall below. The
    engine never reports the result of such a run as a strategy result; the
    comparison lives entirely inside this test.
    """
    return UNCALIBRATED_DEFAULTS.with_parameters(
        half_spread_bps=0.0,
        commission_bps=0.0,
        impact_coefficient=0.0,
        default_daily_volatility_bps=0.0,
        borrow_rate_bps_per_year=0.0,
    )


async def test_costs_strictly_reduce_returns_whenever_anything_is_traded() -> None:
    dates = calendar(4)
    data = _standard_data()
    costed = await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())
    free = await run_backtest(
        data=data,
        calendar=dates,
        strategy=always_long,
        config=config(cost_params=_free_params()),
    )
    costed_returns = costed.artifact.strategy.equity.net_returns
    free_returns = free.artifact.strategy.equity.net_returns

    assert costed.artifact.strategy.summary.accounting.traded_notional_usd > 0.0
    # The traded period is strictly worse; no period is ever better.
    assert costed_returns[0] < free_returns[0]
    assert np.all(costed_returns <= free_returns + 1e-15)
    assert (
        costed.artifact.strategy.equity.equity_usd[-1]
        < free.artifact.strategy.equity.equity_usd[-1]
    )


async def test_a_higher_turnover_strategy_pays_strictly_more() -> None:
    dates = calendar(4)
    data = _standard_data()

    def flip_flop(snapshot: MarketSnapshot) -> Mapping[str, float]:
        return {"AAA": 1.0} if len(snapshot.history("AAA")) % 2 else {"AAA": 0.0}

    churner = await run_backtest(data=data, calendar=dates, strategy=flip_flop, config=config())
    holder = await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())
    churn = churner.artifact.strategy.summary.accounting
    hold = holder.artifact.strategy.summary.accounting
    assert churn.traded_notional_usd > hold.traded_notional_usd
    assert churn.total_cost_usd > hold.total_cost_usd


async def test_the_benchmark_is_charged_its_own_entry_cost() -> None:
    run = await run_backtest(
        data=_standard_data(),
        calendar=calendar(4),
        strategy=always_long,
        config=config(),
    )
    benchmark = run.artifact.benchmark.summary.accounting
    assert benchmark.n_orders == 1
    assert benchmark.total_cost_usd == pytest.approx(ENTRY_COST_USD)
    assert run.artifact.benchmark.equity.equity_usd[0] == CAPITAL


# ---------------------------------------------------------------------------
# I2: reproducibility
# ---------------------------------------------------------------------------


async def test_identical_inputs_reproduce_an_identical_result() -> None:
    dates = calendar(4)
    first = await run_backtest(
        data=_standard_data(), calendar=dates, strategy=always_long, config=config()
    )
    second = await run_backtest(
        data=_standard_data(), calendar=dates, strategy=always_long, config=config()
    )
    assert first.artifact.results_digest() == second.artifact.results_digest()
    assert first.artifact.stamp.to_dict() == second.artifact.stamp.to_dict()


async def test_changing_the_seed_changes_the_result() -> None:
    # A long sample, deliberately: over three return periods every resample of
    # a block bootstrap draws from the same three numbers, so the 2.5%/97.5%
    # quantiles land on the same order statistics whatever the seed. That is a
    # true property of a tiny sample, and asserting the opposite on one would be
    # a test that passes for the wrong reason or not at all.
    dates, data = _long_run_data(120)
    first = await run_backtest(
        data=data, calendar=dates, strategy=always_long, config=config(seed=1)
    )
    second = await run_backtest(
        data=data, calendar=dates, strategy=always_long, config=config(seed=2)
    )
    assert first.artifact.results_digest() != second.artifact.results_digest()
    assert first.artifact.stamp.config_hash != second.artifact.stamp.config_hash
    # The simulation is deterministic, so only the intervals move. Recording
    # which half of the artifact the seed governs is the point of the test.
    for name, estimate in first.artifact.strategy.summary.metrics.items():
        other = second.artifact.strategy.summary.metrics[name]
        assert estimate.value == other.value, name
        assert (estimate.lower, estimate.upper) != (other.lower, other.upper), name
    assert (
        first.artifact.strategy.equity.equity_usd.tolist()
        == second.artifact.strategy.equity.equity_usd.tolist()
    )


async def test_identical_inputs_reproduce_an_identical_result_over_a_long_sample() -> None:
    dates, data = _long_run_data(120)
    first = await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())
    second = await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())
    assert first.artifact.results_digest() == second.artifact.results_digest()
    assert first.artifact.to_dict() | {"created_at": ""} == second.artifact.to_dict() | {
        "created_at": ""
    }


async def test_the_data_version_is_taken_from_the_source_not_the_caller() -> None:
    run = await run_backtest(
        data=_standard_data(),
        calendar=calendar(4),
        strategy=always_long,
        config=config(),
    )
    assert run.artifact.stamp.data_version == DATA_VERSION


async def test_changing_the_strategy_configuration_changes_the_config_hash() -> None:
    dates = calendar(4)
    first = await run_backtest(
        data=_standard_data(),
        calendar=dates,
        strategy=always_long,
        config=config(strategy_config={"lookback": 20}),
    )
    second = await run_backtest(
        data=_standard_data(),
        calendar=dates,
        strategy=always_long,
        config=config(strategy_config={"lookback": 60}),
    )
    assert first.artifact.stamp.config_hash != second.artifact.stamp.config_hash


async def test_an_explicit_commit_is_recorded_verbatim() -> None:
    commit = "b" * 40
    run = await run_backtest(
        data=_standard_data(),
        calendar=calendar(4),
        strategy=always_long,
        config=config(git_commit=commit),
    )
    assert run.artifact.stamp.git_commit == commit


async def test_a_malformed_explicit_commit_is_refused() -> None:
    with pytest.raises(ReproducibilityError, match="lowercase hex object"):
        await run_backtest(
            data=_standard_data(),
            calendar=calendar(4),
            strategy=always_long,
            config=config(git_commit="dirty-worktree"),
        )


# ---------------------------------------------------------------------------
# Reporting shape
# ---------------------------------------------------------------------------


async def test_the_benchmark_is_measured_on_exactly_the_strategy_dates() -> None:
    dates = calendar(4)
    run = await run_backtest(
        data=_standard_data(), calendar=dates, strategy=always_long, config=config()
    )
    assert run.artifact.strategy.equity.dates == dates
    assert run.artifact.benchmark.equity.dates == dates
    assert run.artifact.benchmark.equity.n_periods == run.artifact.strategy.equity.n_periods


async def test_every_reported_metric_carries_an_interval() -> None:
    run = await run_backtest(
        data=_standard_data(),
        calendar=calendar(4),
        strategy=always_long,
        config=config(),
    )
    for track in (run.artifact.strategy, run.artifact.benchmark):
        for name, metric in track.summary.metrics.items():
            assert isinstance(metric, Estimate), name
            assert metric.lower <= metric.upper, name
            assert metric.method.strip(), name
    assert isinstance(run.artifact.comparison.active_return, Estimate)
    assert isinstance(run.artifact.comparison.information_ratio, Estimate)


async def test_the_payload_carries_the_uncalibrated_flag_to_its_consumer() -> None:
    run = await run_backtest(
        data=_standard_data(),
        calendar=calendar(4),
        strategy=always_long,
        config=config(),
    )
    payload = run.artifact.to_dict()
    assert payload["uncalibrated"] is True
    basis = payload["calibration_basis"]
    assert isinstance(basis, str)
    assert "UNCALIBRATED" in basis
    costs = payload["costs"]
    assert isinstance(costs, dict)
    parameters = costs["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["half_spread_bps"] == UNCALIBRATED_DEFAULTS.half_spread_bps
    disclosures = payload["disclosures"]
    assert isinstance(disclosures, list)
    assert any("UNCALIBRATED" in str(item) for item in disclosures)


def _keys_of(payload: object) -> set[str]:
    """Return every mapping key appearing anywhere in a nested JSON payload."""
    if isinstance(payload, dict):
        found = set(payload)
        for value in payload.values():
            found |= _keys_of(value)
        return found
    if isinstance(payload, list):
        return set().union(*(_keys_of(item) for item in payload)) if payload else set()
    return set()


async def test_no_reported_field_anywhere_in_the_payload_is_gross() -> None:
    # I4 as a regression guard: adding a gross series to the artifact later
    # should break a test rather than reach a dashboard.
    run = await run_backtest(
        data=_standard_data(), calendar=calendar(4), strategy=always_long, config=config()
    )
    keys = _keys_of(run.artifact.to_dict())
    assert not [key for key in keys if "gross" in key.lower()]
    assert "net_returns" in keys


# ---------------------------------------------------------------------------
# Guards on the calendar and the weights
# ---------------------------------------------------------------------------


async def test_a_calendar_with_fewer_than_three_instants_is_refused() -> None:
    with pytest.raises(CalendarError, match="at least 3 instants"):
        await run_backtest(
            data=_standard_data(),
            calendar=calendar(2),
            strategy=always_long,
            config=config(),
        )


async def test_a_naive_calendar_is_refused_rather_than_localized() -> None:
    naive = (
        dt.datetime(2024, 1, 1),  # noqa: DTZ001
        dt.datetime(2024, 1, 2),  # noqa: DTZ001
        dt.datetime(2024, 1, 3),  # noqa: DTZ001
    )
    with pytest.raises(CalendarError, match="timezone-aware UTC"):
        await run_backtest(
            data=_standard_data(), calendar=naive, strategy=always_long, config=config()
        )


async def test_a_calendar_that_goes_backwards_is_refused() -> None:
    dates = calendar(4)
    scrambled = (dates[0], dates[2], dates[1], dates[3])
    with pytest.raises(CalendarError, match="strictly increase"):
        await run_backtest(
            data=_standard_data(), calendar=scrambled, strategy=always_long, config=config()
        )


async def test_leverage_is_refused_by_default() -> None:
    def levered(_: MarketSnapshot) -> Mapping[str, float]:
        return {"AAA": 1.5}

    with pytest.raises(WeightError, match="gross exposure"):
        await run_backtest(
            data=_standard_data(), calendar=calendar(4), strategy=levered, config=config()
        )


async def test_a_long_short_book_within_the_cap_is_allowed() -> None:
    dates = calendar(4)
    data = _standard_data()

    def market_neutral(_: MarketSnapshot) -> Mapping[str, float]:
        return {"AAA": 0.5, "BMK": -0.5}

    run = await run_backtest(data=data, calendar=dates, strategy=market_neutral, config=config())
    accounting = run.artifact.strategy.summary.accounting
    # Two names entered on the first decision, then re-levelled at each of the
    # remaining two: the legs drift apart with the market, so holding a *fixed*
    # weight pair costs turnover every period. Six orders, not two.
    assert accounting.n_rebalances == 3
    assert accounting.n_orders == 6
    assert accounting.borrow_usd > 0.0


async def test_a_levered_benchmark_is_refused_by_the_same_rule_as_a_levered_strategy() -> None:
    with pytest.raises(WeightError, match="benchmark gross exposure"):
        config(benchmark=BuyAndHoldSpec(weights={"BMK": 1.5}))


def test_an_injected_source_refuses_a_decreasing_knowledge_time() -> None:
    dates = calendar(3)
    late_then_early = (
        Observation(
            date=dates[0],
            knowledge_time=dates[2],
            total_return=0.0,
            adv_usd=ADV,
        ),
        Observation(
            date=dates[1],
            knowledge_time=dates[1],
            total_return=0.0,
            adv_usd=ADV,
        ),
    )
    with pytest.raises(CalendarError, match="non-decreasing knowledge_time"):
        InjectedMarketData(observations={"AAA": late_then_early}, data_version="v")


def test_an_injected_source_requires_a_data_version() -> None:
    with pytest.raises(ValueError, match="data_version must name"):
        InjectedMarketData(observations={}, data_version="  ")


def test_an_observation_requires_a_positive_average_daily_volume() -> None:
    dates = calendar(2)
    with pytest.raises(ValueError, match="strictly positive"):
        Observation(date=dates[0], knowledge_time=dates[0], total_return=0.0, adv_usd=0.0)


# ---------------------------------------------------------------------------
# Narrowing a snapshot
# ---------------------------------------------------------------------------


async def test_narrowing_a_snapshot_drops_old_observations_and_keeps_the_instant() -> None:
    dates = calendar(4)
    snapshot = await _standard_data().as_of(dates[3])
    assert len(snapshot.history("AAA")) == 4
    narrowed = snapshot.since(dates[2])
    assert narrowed.as_of == dates[3]
    assert [item.date for item in narrowed.history("AAA")] == [dates[2], dates[3]]


async def test_narrowing_a_snapshot_cannot_widen_it() -> None:
    dates = calendar(6)
    data = injected(AAA=observations(dates, [0.0] * 6))
    snapshot = await data.as_of(dates[2])
    # Asking for history from before the snapshot's own earliest observation
    # returns what it already had, never more.
    assert len(snapshot.since(dates[0]).history("AAA")) == 3


async def test_narrowing_a_snapshot_refuses_a_naive_instant() -> None:
    snapshot = await _standard_data().as_of(calendar(4)[3])
    with pytest.raises(CalendarError, match="timezone-aware UTC"):
        snapshot.since(dt.datetime(2024, 1, 2))  # noqa: DTZ001


# ---------------------------------------------------------------------------
# Unit and sign validation
# ---------------------------------------------------------------------------


def test_a_non_utc_instant_is_refused_even_though_it_is_timezone_aware() -> None:
    # The dangerous case: aware but offset, which every naive-datetime guard
    # lets through and which shifts a knowledge time by hours.
    eastern = dt.timezone(dt.timedelta(hours=-5))
    with pytest.raises(CalendarError, match=r"must be UTC \(offset 0\)"):
        Observation(
            date=dt.datetime(2024, 1, 1, tzinfo=eastern),
            knowledge_time=dt.datetime(2024, 1, 1, tzinfo=eastern),
            total_return=0.0,
            adv_usd=ADV,
        )


@pytest.mark.parametrize("bad_return", [-1.5, float("nan"), float("inf")])
def test_an_unusable_total_return_is_refused(bad_return: float) -> None:
    dates = calendar(2)
    with pytest.raises(ValueError, match="total_return must be finite"):
        Observation(date=dates[0], knowledge_time=dates[0], total_return=bad_return, adv_usd=ADV)


def test_a_negative_daily_volatility_is_refused() -> None:
    dates = calendar(2)
    with pytest.raises(ValueError, match="daily_volatility_bps"):
        Observation(
            date=dates[0],
            knowledge_time=dates[0],
            total_return=0.0,
            adv_usd=ADV,
            daily_volatility_bps=-1.0,
        )


def test_a_snapshot_refuses_observations_out_of_date_order() -> None:
    dates = calendar(3)
    series = observations(dates[:2], [0.0, 0.0])
    with pytest.raises(CalendarError, match="strictly ascending date order"):
        MarketSnapshot(as_of=dates[2], observations={"AAA": (series[1], series[0])})


def test_an_injected_source_refuses_observations_out_of_date_order() -> None:
    dates = calendar(3)
    series = observations(dates[:2], [0.0, 0.0])
    with pytest.raises(CalendarError, match="strictly ascending date order"):
        InjectedMarketData(observations={"AAA": (series[1], series[0])}, data_version="v")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initial_capital_usd": 0.0}, "initial_capital_usd must be positive"),
        ({"periods_per_year": 0.0}, "periods_per_year must be positive"),
        ({"max_gross_exposure": -0.5}, "max_gross_exposure must be non-negative"),
    ],
)
def test_the_run_configuration_validates_its_parameters(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config(**overrides)


async def test_costing_an_asset_with_no_knowable_observation_is_refused() -> None:
    # The benchmark asset must be priceable at the entry instant; a universe
    # that cannot cost its own baseline has no comparison to report.
    dates = calendar(4)
    data = injected(AAA=observations(dates, [0.0, 0.1, 0.1, 0.1]))
    with pytest.raises(LookaheadError, match="cannot be costed"):
        await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())


async def test_execution_costs_that_exhaust_the_book_stop_the_run() -> None:
    ruinous = UNCALIBRATED_DEFAULTS.with_parameters(half_spread_bps=20_000.0)
    with pytest.raises(PortfolioWipeoutError, match="execution costs alone"):
        await run_backtest(
            data=_standard_data(),
            calendar=calendar(4),
            strategy=always_long,
            config=config(cost_params=ruinous),
        )


async def test_a_period_that_wipes_out_the_book_stops_the_run() -> None:
    dates = calendar(4)
    data = injected(
        AAA=observations(dates, [0.0, -1.0, 0.0, 0.0]),
        BMK=observations(dates, [0.0, 0.01, 0.01, 0.01]),
    )
    with pytest.raises(PortfolioWipeoutError, match="wipes out the book"):
        await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())


def test_a_snapshot_reports_only_the_assets_it_actually_holds() -> None:
    dates = calendar(3)
    snapshot = MarketSnapshot(
        as_of=dates[2],
        observations={"AAA": observations(dates[:2], [0.0, 0.0]), "EMPTY": ()},
    )
    assert snapshot.assets == frozenset({"AAA"})
    assert snapshot.latest("EMPTY") is None
    assert snapshot.history("MISSING") == ()
    assert snapshot.returns_history("AAA") == (0.0, 0.0)
    assert snapshot.observation_on("AAA", dates[1]) is not None
    assert snapshot.observation_on("AAA", dates[2]) is None
    assert snapshot.observation_on("MISSING", dates[0]) is None


# ---------------------------------------------------------------------------
# The two sides share one accounting kernel
# ---------------------------------------------------------------------------


async def test_a_never_trading_strategy_reproduces_the_benchmark_helper_exactly() -> None:
    # benchmark.py claims the engine's loop and buy_and_hold share one costing
    # and drift kernel. If they ever diverge, the difference shows up as alpha,
    # so it is asserted rather than trusted: a strategy that buys AAA once and
    # holds must trace the same path as a buy-and-hold of AAA.
    dates = calendar(4)
    data = _standard_data()
    run = await run_backtest(data=data, calendar=dates, strategy=always_long, config=config())
    held, accounting = buy_and_hold(
        spec=BuyAndHoldSpec(weights={"AAA": 1.0}),
        dates=dates,
        asset_returns={"AAA": [0.10, -0.20, 0.25]},
        adv_usd={"AAA": ADV},
        daily_volatility_bps={"AAA": None},
        initial_capital_usd=CAPITAL,
        params=UNCALIBRATED_DEFAULTS,
    )
    assert run.artifact.strategy.equity.equity_usd == pytest.approx(held.equity_usd, rel=1e-12)
    assert run.artifact.strategy.summary.accounting.total_cost_usd == pytest.approx(
        accounting.total_cost_usd
    )


async def test_a_strategy_that_is_its_own_benchmark_refuses_an_information_ratio() -> None:
    # The degenerate comparison: zero-variance active returns. Reporting 0.0
    # would read as "no edge" when the truth is that the number does not exist.
    dates = calendar(4)
    data = _standard_data()
    with pytest.raises(ValueError, match="tracks its benchmark exactly"):
        await run_backtest(
            data=data,
            calendar=dates,
            strategy=always_long,
            config=config(benchmark=BuyAndHoldSpec(weights={"AAA": 1.0})),
        )


async def test_churning_on_noise_loses_to_buy_and_hold_after_costs() -> None:
    # The engine must not manufacture edge. A strategy that flips its book on a
    # coin-flip schedule over noise pays turnover for nothing, so its net active
    # return against buy-and-hold is negative.
    dates, data = _long_run_data(120)

    def flip_flop(snapshot: MarketSnapshot) -> Mapping[str, float]:
        return {"AAA": 1.0} if len(snapshot.history("AAA")) % 2 else {"BMK": 1.0}

    run = await run_backtest(data=data, calendar=dates, strategy=flip_flop, config=config())
    assert run.artifact.strategy.summary.accounting.traded_notional_usd > CAPITAL
    assert run.artifact.comparison.active_return.value < 0.0
