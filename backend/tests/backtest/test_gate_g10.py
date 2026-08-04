"""Gate G10 — the framework, end to end, against a known answer (P10.8).

Directive §5, Phase 10:

    *"Framework reproduces known results on synthetic data with injected signal
    of known strength — and correctly reports near-zero signal on pure noise.
    This second test is the critical one. If your framework finds alpha in
    random data, it is broken and everything downstream is fiction."*

``test_synthetic_truth.py`` gates the *statistics* — CPCV, DSR and PBO applied
directly to return matrices. This module gates the *framework*: every number
below comes out of :func:`backend.backtest.engine.run_backtest` driven over
seeded synthetic markets by :mod:`backend.backtest.synthetic`, and is then handed
to those same statistics. Nothing shortcuts the engine, the cost model or the
point-in-time boundary.

How the assertions are built, and why they are shaped this way
-------------------------------------------------------------

**One seed is not evidence.** At ``T = 252`` periods the sampling standard
deviation of a Sharpe ratio is about ``1/sqrt(T) = 0.063`` per period —
``1.2`` annualized — whatever the truth is. A single pure-noise run landing at
an annualized Sharpe of ``+1.2`` is an ordinary draw, and
``test_a_single_pure_noise_seed_can_look_like_a_real_edge`` asserts that it
happens. Every claim here is therefore about a distribution over independent
seeds, and every threshold is quoted in standard errors of that distribution.

**A weak assertion is not a test.** "PBO is not exactly zero" and "the signal is
positive" both pass on a broken framework. Each claim below is two-sided and
quantified: a bound above *and* below, plus a non-vacuity check that the
distribution being bounded is not degenerate.

**Costs, both ways (I4).** No figure here is reported gross — the engine has no
gross path and every artifact carries its cost provenance. But the noise
direction is checked before costs as well as after, using
:attr:`~backend.backtest.synthetic.CostMode.ZERO_COST_DIAGNOSTIC`, because a
framework that finds alpha in random data and then loses it to the spread has
still found alpha in random data. The gross diagnostic is where the critical
claim is made; the net figures are what the directive permits reporting.

**The known answer is known independently.** The injected strength is ``tau``,
the cross-sectional standard deviation of true expected return in **basis points
per period**, and the per-period Sharpe ratio it implies is a closed form derived
from the data-generating process alone.
``test_the_predicted_strength_is_derived_from_the_process_not_from_the_framework``
re-derives it by Monte Carlo in this file, importing nothing from
:mod:`backend.backtest`, so "the framework recovers the prediction" is not the
framework agreeing with itself.

**The gate is proved non-vacuous.**
:data:`backend.backtest.synthetic.SyntheticSpec.lookahead_leak_fraction` leaks a
fraction of the next period's return into the strategy's own signal — an
off-by-one in a rolling window, the most ordinary lookahead bug there is. The
leaked runs use the **same seeds and the same samples** as the honest noise
sweep, so the only difference is the defect, and
``test_the_gate_fails_on_a_framework_that_leaks_the_future`` asserts that every
one of the noise criteria is violated. A gate that cannot fail proves nothing.

One of the criteria — :func:`exceeds_the_oracle` — cannot be checked through an
output at all, and that is
``test_the_cross_validation_the_gate_relies_on_really_purges_and_embargoes``'s
subject: with point-in-time labels, deleting CPCV's purge changes the reported
numbers by almost nothing while leaving a leak that grows with the label horizon.
It is therefore asserted structurally.

Runtime
-------

About 100 seconds: roughly 850 engine runs, each a 252-period simulation over 8
or 12 instruments. The sweeps are module-scoped fixtures, so each is computed
once and asserted on from several tests.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from backend.backtest.cpcv import CombinatorialPurgedCV
from backend.backtest.engine import InjectedMarketData, Observation
from backend.backtest.synthetic import (
    SYNTHETIC_DATA_VERSION_PREFIX,
    ZERO_COST_PARAMS,
    ConfigurationSweep,
    CostMode,
    SharpeSweep,
    StrategyCandidate,
    SyntheticDataError,
    SyntheticMarket,
    SyntheticSpec,
    build_synthetic_market,
    disjoint_asset_blocks,
    format_report,
    require_synthetic_source,
    run_synthetic_backtest,
    sweep_configurations,
    sweep_seeds,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# The experiment, sized so that the claims below have the resolution they need
# ---------------------------------------------------------------------------

GATE_N_ASSETS: Final = 8
GATE_N_PERIODS: Final = 252
GATE_LOOKBACK_PERIODS: Final = 63
GATE_VOLATILITY_BPS: Final = 100.0
"""The shared data-generating process. 252 periods of 1%-per-period instruments
with a 63-period estimator window."""


def gate_spec(
    *,
    seed: int,
    alpha_dispersion_bps_per_day: float = 0.0,
    lookahead_leak_fraction: float = 0.0,
) -> SyntheticSpec:
    """Return the gate's process at one seed and one injected strength."""
    return SyntheticSpec(
        seed=seed,
        alpha_dispersion_bps_per_day=alpha_dispersion_bps_per_day,
        lookahead_leak_fraction=lookahead_leak_fraction,
        n_assets=GATE_N_ASSETS,
        n_periods=GATE_N_PERIODS,
        lookback_periods=GATE_LOOKBACK_PERIODS,
        idiosyncratic_volatility_bps_per_day=GATE_VOLATILITY_BPS,
    )


NOISE_SEEDS: Final = tuple(range(9_000, 9_128))
"""128 independent pure-noise experiments.

The per-seed standard deviation of the realized per-period Sharpe ratio is about
``1/sqrt(252) = 0.063``, so the mean over 128 seeds has a standard error near
``0.0056`` per period — about ``0.107`` annualized. That is the resolution every
noise threshold below is quoted against."""

LEAK_SEEDS: Final = NOISE_SEEDS[:48]
"""The leaked control reuses the first 48 noise samples verbatim: same seeds,
same returns, same sample length. The only difference is the injected defect."""

LADDER_SEEDS: Final = tuple(range(7_000, 7_032))
"""32 seeds per injected strength. Paired against each seed's own closed-form
prediction the residual standard deviation is about ``0.078`` per period, so the
mean residual has a standard error near ``0.014`` — under 5% of the smallest
prediction on the ladder."""

SEARCH_SEEDS: Final = tuple(range(5_000, 5_016))
"""16 seeds for the six-configuration search that PBO and DSR are computed over."""

SIGNAL_SEARCH_SEEDS: Final = tuple(range(5_100, 5_116))
"""16 seeds for the same search with a real edge present."""

INJECTED_STRENGTHS_BPS: Final = (16.0, 32.0, 48.0)
"""The signal ladder, in basis points of expected return per period.

Chosen where the closed form is empirically unbiased and where 32 seeds resolve
it to a few per cent. The predicted per-period Sharpe ratios are not proportional
to these numbers — estimator fidelity and return variance both move with ``tau``
— so matching them is a test of the functional form, which a framework with a
scale error fails."""

LEAK_FRACTION: Final = 0.25
"""A quarter of one period's return, leaked into a 63-period rolling mean.

Deliberately small. A strategy that simply reads tomorrow's return would fail
any test; the question is whether the gate catches the ordinary version of the
bug, where one bar of a long window is off by one and only partially so."""

SEARCH_N_ASSETS: Final = 12
"""12 instruments so that the search has six candidates of two assets each."""

SEARCH_SIGNAL_BPS: Final = 24.0
"""Injected strength for the search-with-an-edge case."""


def search_spec(*, seed: int, alpha_dispersion_bps_per_day: float) -> SyntheticSpec:
    """Return the search process: the gate's process widened to 12 instruments."""
    return SyntheticSpec(
        seed=seed,
        alpha_dispersion_bps_per_day=alpha_dispersion_bps_per_day,
        n_assets=SEARCH_N_ASSETS,
        n_periods=GATE_N_PERIODS,
        lookback_periods=GATE_LOOKBACK_PERIODS,
        idiosyncratic_volatility_bps_per_day=GATE_VOLATILITY_BPS,
    )


# ---------------------------------------------------------------------------
# The gate's criteria, named so that the mutation test can violate them
# ---------------------------------------------------------------------------

NOISE_MAX_ABS_T_STATISTIC: Final = 2.5
"""How many standard errors from zero the mean noise Sharpe ratio may sit.

A pure statistical statement: with the mean's own standard error estimated from
the seed-to-seed spread, 2.5 is a two-sided test at roughly the 1% level."""

NOISE_MAX_ABS_ANNUALIZED_SHARPE: Final = 0.35
"""How far from zero the mean noise Sharpe ratio may sit in absolute terms.

Roughly 3.3 standard errors at 128 seeds, and far below anything anyone would
act on. The t-statistic alone is not enough — a framework could have a small
bias and a huge variance — and the magnitude alone is not enough either, since a
framework with a tiny variance and a persistent tilt would slip under it. Both
are required."""

RECOVERY_RELATIVE_TOLERANCE: Final = 0.20
"""How far the recovered Sharpe ratio may sit from its closed-form prediction.

At the weakest rung this is about 4 standard errors of the paired residual; at
the strongest, about 14."""

RECOVERY_MAX_ABS_RESIDUAL: Final = 0.06
"""Absolute ceiling on the mean paired residual, per period.

The relative tolerance alone would let a large absolute error through at a large
injected strength; this catches that."""

ORACLE_EXCEEDANCE_STANDARD_ERRORS: Final = 2.5
"""How far above the oracle a recovered mean may sit before it is a leak.

No estimator is more aligned with the outcome than the true alphas, so the
oracle's Sharpe ratio is a hard ceiling on what any honest framework can recover
out of sample. On pure noise the ceiling is exactly zero, which is why the same
criterion serves both directions of G10 — and why the leaked control violates
it."""


def exceeds_the_oracle(sweep: SharpeSweep) -> bool:
    """Return whether a recovered mean sits significantly above the process's ceiling.

    ``True`` means the framework reported more out-of-sample skill than a book
    holding the *true* expected returns could have earned, which no correct
    framework can do. On a pure-noise process the ceiling is zero, so this is the
    directive's critical claim in its sharpest form.

    Args:
        sweep: a distribution of realized Sharpe ratios over seeds.

    Returns:
        Whether the mean exceeds the oracle by more than
        :data:`ORACLE_EXCEEDANCE_STANDARD_ERRORS` standard errors.
    """
    ceiling = sweep.spec.oracle_gross_sharpe_per_period()
    return sweep.mean > ceiling + ORACLE_EXCEEDANCE_STANDARD_ERRORS * sweep.standard_error


def noise_criteria(sweep: SharpeSweep) -> dict[str, bool]:
    """Return the gate's pure-noise criteria evaluated on one distribution.

    All three must hold for a framework to pass G10's critical direction. Exposed
    as a function so that the leaked-framework mutation can assert they *fail* on
    exactly the criteria the honest run passes, rather than on some other
    quantity chosen after the fact.

    Args:
        sweep: a distribution of realized Sharpe ratios over seeds, produced on
            data with no signal in it whatsoever.

    Returns:
        Mapping from criterion name to whether it holds.
    """
    return {
        "t_statistic": abs(sweep.t_statistic) < NOISE_MAX_ABS_T_STATISTIC,
        "magnitude": abs(sweep.annualized_mean) < NOISE_MAX_ABS_ANNUALIZED_SHARPE,
        "oracle_ceiling": not exceeds_the_oracle(sweep),
    }


# ---------------------------------------------------------------------------
# Fixtures — every engine run in this module happens exactly once
# ---------------------------------------------------------------------------


def _noise_spec(*, leak: float = 0.0) -> SyntheticSpec:
    """Return the pure-noise process, optionally with lookahead injected."""
    return gate_spec(seed=NOISE_SEEDS[0], lookahead_leak_fraction=leak)


@pytest.fixture(scope="module")
def noise_gross() -> SharpeSweep:
    """Pure noise, zero-cost diagnostic. The critical direction of G10."""
    return asyncio.run(
        sweep_seeds(_noise_spec(), seeds=NOISE_SEEDS, cost_mode=CostMode.ZERO_COST_DIAGNOSTIC)
    )


@pytest.fixture(scope="module")
def noise_net() -> SharpeSweep:
    """Pure noise, charged the shipped uncalibrated cost parameters."""
    return asyncio.run(
        sweep_seeds(_noise_spec(), seeds=NOISE_SEEDS, cost_mode=CostMode.NET_OF_MODELLED_COSTS)
    )


@pytest.fixture(scope="module")
def leak_gross() -> SharpeSweep:
    """The same noise samples, run by a framework that leaks the future."""
    return asyncio.run(
        sweep_seeds(
            _noise_spec(leak=LEAK_FRACTION),
            seeds=LEAK_SEEDS,
            cost_mode=CostMode.ZERO_COST_DIAGNOSTIC,
        )
    )


@pytest.fixture(scope="module")
def leak_net() -> SharpeSweep:
    """The leaked framework, charged full costs."""
    return asyncio.run(
        sweep_seeds(
            _noise_spec(leak=LEAK_FRACTION),
            seeds=LEAK_SEEDS,
            cost_mode=CostMode.NET_OF_MODELLED_COSTS,
        )
    )


def _ladder(cost_mode: CostMode) -> tuple[SharpeSweep, ...]:
    """Run the injected-strength ladder under one cost basis."""

    async def _run() -> tuple[SharpeSweep, ...]:
        return tuple(
            [
                await sweep_seeds(
                    gate_spec(seed=LADDER_SEEDS[0], alpha_dispersion_bps_per_day=strength),
                    seeds=LADDER_SEEDS,
                    cost_mode=cost_mode,
                )
                for strength in INJECTED_STRENGTHS_BPS
            ]
        )

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def ladder_gross() -> tuple[SharpeSweep, ...]:
    """One distribution per injected strength, zero-cost diagnostic."""
    return _ladder(CostMode.ZERO_COST_DIAGNOSTIC)


@pytest.fixture(scope="module")
def ladder_net() -> tuple[SharpeSweep, ...]:
    """One distribution per injected strength, net of modelled costs."""
    return _ladder(CostMode.NET_OF_MODELLED_COSTS)


def _searches(
    *, strength: float, seeds: Sequence[int], cost_mode: CostMode
) -> tuple[ConfigurationSweep, ...]:
    """Run the six-candidate search once per seed."""

    async def _run() -> tuple[ConfigurationSweep, ...]:
        sweeps: list[ConfigurationSweep] = []
        for seed in seeds:
            spec = search_spec(seed=seed, alpha_dispersion_bps_per_day=strength)
            market = build_synthetic_market(spec)
            sweeps.append(
                await sweep_configurations(
                    spec,
                    candidates=disjoint_asset_blocks(market, block_size=2),
                    cost_mode=cost_mode,
                    market=market,
                )
            )
        return tuple(sweeps)

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def search_noise_gross() -> tuple[ConfigurationSweep, ...]:
    """A six-configuration search over pure noise, zero-cost diagnostic."""
    return _searches(strength=0.0, seeds=SEARCH_SEEDS, cost_mode=CostMode.ZERO_COST_DIAGNOSTIC)


@pytest.fixture(scope="module")
def search_noise_net() -> tuple[ConfigurationSweep, ...]:
    """The same search, net of modelled costs."""
    return _searches(strength=0.0, seeds=SEARCH_SEEDS, cost_mode=CostMode.NET_OF_MODELLED_COSTS)


@pytest.fixture(scope="module")
def search_signal_net() -> tuple[ConfigurationSweep, ...]:
    """The same search with a real edge present, net of modelled costs."""
    return _searches(
        strength=SEARCH_SIGNAL_BPS,
        seeds=SIGNAL_SEARCH_SEEDS,
        cost_mode=CostMode.NET_OF_MODELLED_COSTS,
    )


def _pbos(searches: Sequence[ConfigurationSweep]) -> np.ndarray:
    """Return one PBO per search."""
    return np.array(
        [search.probability_of_backtest_overfitting().pbo for search in searches],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 0. The harness is what it claims to be (I3, I2)
# ---------------------------------------------------------------------------


def test_the_harness_refuses_to_be_pointed_at_data_it_did_not_generate() -> None:
    moment = build_synthetic_market(_noise_spec()).calendar[0]
    real_looking = InjectedMarketData(
        observations={
            "AAPL": (
                Observation(date=moment, knowledge_time=moment, total_return=0.01, adv_usd=1e9),
            )
        },
        data_version="sharadar-sep-2026-08-03",
    )
    with pytest.raises(SyntheticDataError, match="only runs on data it generated"):
        require_synthetic_source(real_looking)


def test_every_generated_dataset_announces_that_it_is_synthetic() -> None:
    spec = _noise_spec()
    market = build_synthetic_market(spec)
    assert market.data.data_version.startswith(SYNTHETIC_DATA_VERSION_PREFIX)
    assert f"seed={spec.seed}" in market.data.data_version
    # The prefix survives into the artifact's own I2 stamp, so a result that
    # escaped into a report would still be carrying its refusal.
    run = asyncio.run(run_synthetic_backtest(spec))
    assert run.artifact.stamp.data_version.startswith(SYNTHETIC_DATA_VERSION_PREFIX)
    assert run.is_research_finding is False
    assert "SYNTHETIC DATA" in run.disclosure
    assert "not a research finding" in run.disclosure


def test_every_run_carries_the_full_reproducibility_stamp_including_its_seed() -> None:
    spec = _noise_spec()
    run = asyncio.run(run_synthetic_backtest(spec))
    stamp = run.stamp
    assert stamp.seed == spec.seed
    assert len(stamp.git_commit) in {40, 64}
    assert stamp.data_version == spec.data_version
    assert len(stamp.config_hash) == 64
    assert isinstance(stamp.git_dirty, bool)
    # Two different seeds are two different experiments and must not collide.
    other = asyncio.run(run_synthetic_backtest(gate_spec(seed=spec.seed + 1)))
    assert other.stamp.config_hash != stamp.config_hash
    assert other.stamp.seed != stamp.seed


def test_a_market_cannot_be_reused_under_a_different_process() -> None:
    spec = _noise_spec()
    market = build_synthetic_market(spec)
    other = gate_spec(seed=spec.seed, alpha_dispersion_bps_per_day=25.0)
    with pytest.raises(SyntheticDataError, match="different spec"):
        asyncio.run(run_synthetic_backtest(other, market=market))


def test_the_zero_cost_diagnostic_is_labelled_as_one_wherever_it_appears() -> None:
    """Invariant I4: the gross path exists only as a diagnostic, and says so."""
    run = asyncio.run(
        run_synthetic_backtest(_noise_spec(), cost_mode=CostMode.ZERO_COST_DIAGNOSTIC)
    )
    provenance = run.artifact.costs
    assert provenance.parameters["half_spread_bps"] == 0.0
    assert provenance.parameters["commission_bps"] == 0.0
    assert provenance.parameters["borrow_rate_bps_per_year"] == 0.0
    assert provenance.uncalibrated is True
    assert "ZERO-COST DIAGNOSTIC" in provenance.calibration_basis
    assert "ZERO-COST DIAGNOSTIC" in run.disclosure
    assert run.cost_mode.is_reportable_basis is False
    assert ZERO_COST_PARAMS.half_spread_bps == 0.0
    # The reportable basis is a different object and is not zeroed.
    reportable = asyncio.run(run_synthetic_backtest(_noise_spec()))
    assert reportable.cost_mode.is_reportable_basis is True
    assert reportable.artifact.costs.parameters["half_spread_bps"] > 0.0
    assert reportable.total_cost_bps_of_capital > 0.0


def test_the_predicted_strength_is_derived_from_the_process_not_from_the_framework() -> None:
    """The "known answer" must be known without asking the thing under test.

    The closed form in :mod:`backend.backtest.synthetic` is re-derived here by
    Monte Carlo over the data-generating process, using nothing but numpy. If the
    two disagree, "the framework recovers the prediction" would only mean the
    framework agrees with itself.
    """
    for strength in INJECTED_STRENGTHS_BPS:
        spec = gate_spec(seed=1, alpha_dispersion_bps_per_day=strength)
        sigma = spec.idiosyncratic_volatility_bps_per_day / 1e4
        tau = strength / 1e4
        error = sigma / math.sqrt(spec.lookback_periods)

        generator = np.random.default_rng(20_260_803)
        draws = 400_000
        alpha = generator.normal(0.0, tau, draws)
        estimate = alpha + generator.normal(0.0, error, draws)
        outcome = alpha + generator.normal(0.0, sigma, draws)
        position = np.where(estimate >= 0.0, 1.0, -1.0)
        per_asset = position * outcome
        # n independent assets, each contributing 1/n of the book.
        expected = float(np.mean(per_asset))
        variance = float(np.var(per_asset, ddof=1)) / spec.n_assets
        monte_carlo = expected / math.sqrt(variance)

        assert spec.expected_gross_sharpe_per_period() == pytest.approx(monte_carlo, rel=0.03)
        # And the oracle really is a ceiling on it.
        assert spec.oracle_gross_sharpe_per_period() > spec.expected_gross_sharpe_per_period()
        assert 0.0 < spec.signal_fidelity < 1.0


def test_a_process_with_no_injected_signal_predicts_exactly_zero() -> None:
    spec = _noise_spec()
    assert spec.is_pure_noise is True
    assert spec.signal_fidelity == 0.0
    assert spec.expected_gross_sharpe_per_period() == 0.0
    assert spec.oracle_gross_sharpe_per_period() == 0.0
    market = build_synthetic_market(spec)
    assert market.conditional_gross_sharpe_per_period() == 0.0
    assert set(market.true_alpha_bps_per_day.values()) == {0.0}


# ---------------------------------------------------------------------------
# 1. Pure noise — the direction the directive calls critical
# ---------------------------------------------------------------------------


def test_the_framework_finds_no_alpha_in_pure_noise_before_costs(
    noise_gross: SharpeSweep,
) -> None:
    """G10's critical claim, made where costs cannot be doing the work.

    128 independent samples with an expected return of exactly zero on every
    instrument. The framework must report nothing, and must report it as a
    distribution centred on zero rather than as one lucky path.
    """
    assert noise_gross.n_seeds == len(NOISE_SEEDS)
    assert noise_gross.spec.is_pure_noise is True
    assert noise_gross.cost_mode is CostMode.ZERO_COST_DIAGNOSTIC

    criteria = noise_criteria(noise_gross)
    assert criteria["t_statistic"], (
        f"mean per-period Sharpe {noise_gross.mean:+.5f} is "
        f"{noise_gross.t_statistic:+.2f} standard errors from zero on data with no "
        "signal in it"
    )
    assert criteria["magnitude"], (
        f"mean annualized Sharpe on pure noise is {noise_gross.annualized_mean:+.3f}"
    )
    assert criteria["oracle_ceiling"], (
        "the framework recovered more than a book holding the true expected returns "
        f"could have earned, on a process whose oracle Sharpe ratio is exactly "
        f"{noise_gross.spec.oracle_gross_sharpe_per_period():.1f}"
    )
    assert abs(noise_gross.mean) < 0.02
    # Two-sided: neither systematically optimistic nor systematically pessimistic.
    assert 0.35 <= float(np.mean(noise_gross.values > 0.0)) <= 0.65


def test_the_pure_noise_distribution_is_not_degenerate(noise_gross: SharpeSweep) -> None:
    """The bound above is only meaningful if there is something to bound.

    A framework that returned zero for every seed would satisfy every claim in
    the previous test while measuring nothing at all.
    """
    assert noise_gross.std > 0.04
    assert noise_gross.standard_error > 0.0
    assert float(np.min(noise_gross.values)) < -0.08
    assert float(np.max(noise_gross.values)) > 0.08
    assert len(set(noise_gross.values.tolist())) == noise_gross.n_seeds


def test_a_single_pure_noise_seed_can_look_like_a_real_edge(
    noise_gross: SharpeSweep,
) -> None:
    """Why every claim here is about a distribution and never about one run.

    These are annualized Sharpe ratios on data with no signal whatsoever. Several
    of them are numbers a person would act on.
    """
    annualized = np.array([noise_gross.spec.annualize(value) for value in noise_gross.values])
    assert float(np.max(annualized)) > 1.0
    assert float(np.min(annualized)) < -1.0
    assert float(np.mean(np.abs(annualized) > 1.0)) > 0.05


def test_trading_pure_noise_loses_money_once_costs_are_charged(
    noise_gross: SharpeSweep, noise_net: SharpeSweep
) -> None:
    """Invariant I4, in the direction that matters: costs subtract, they do not add.

    The reportable figure on noise is not "about zero" — it is reliably negative,
    because the strategy pays a spread to rearrange a book with no edge in it. A
    framework reporting a *better* net figure than gross would be mis-signing its
    own cost model.
    """
    assert noise_net.cost_mode is CostMode.NET_OF_MODELLED_COSTS
    assert noise_net.mean < 0.0
    assert noise_net.t_statistic < -2.0
    drag = noise_gross.mean - noise_net.mean
    assert drag > 0.0
    assert noise_gross.spec.annualize(drag) == pytest.approx(0.53, abs=0.25)
    assert -1.2 < noise_net.annualized_mean < -0.15
    assert noise_net.mean_turnover_per_period > 0.05
    assert noise_net.mean_total_cost_bps_of_capital > 0.0
    assert noise_gross.mean_total_cost_bps_of_capital == 0.0


def test_the_benchmark_comparison_also_reports_nothing_on_pure_noise(
    noise_gross: SharpeSweep,
) -> None:
    """Directive §5-P10: the benchmark is shown with every result, so it is gated too.

    The active return is a *difference of two* noisy series, and on this process
    the benchmark is the noisier of them by a factor of two — an
    80-basis-point-per-period index has a 252-period sample mean with a standard
    deviation of about 5 basis points, against the strategy's 2.2. Bounding the
    active return tightly would therefore be asserting that the synthetic index's
    own draw came out near zero, which is luck rather than a property of the
    framework.

    So the claim is decomposed. The strategy's own contribution must be nil, with
    its own standard error; the comparison must be exactly the difference of the
    two series it says it is; and the information ratio must be small in absolute
    terms. What is left over belongs to the index, and is not the framework's to
    answer for.
    """
    strategy = np.array([float(np.mean(run.returns)) for run in noise_gross.runs])
    benchmark = np.array(
        [float(np.mean(run.artifact.benchmark.equity.net_returns)) for run in noise_gross.runs]
    )
    active = np.array([run.artifact.comparison.active_return.value for run in noise_gross.runs])
    ratios = np.array([run.artifact.comparison.information_ratio.value for run in noise_gross.runs])

    # The comparison is the difference it claims to be, run by run.
    assert active == pytest.approx(strategy - benchmark, abs=1e-15)

    # The strategy side of it carries no alpha, and this is the framework's part.
    strategy_error = float(np.std(strategy, ddof=1)) / math.sqrt(strategy.size)
    assert strategy_error > 0.0
    assert abs(float(np.mean(strategy))) < 3.0 * strategy_error
    assert abs(float(np.mean(strategy))) < 5.0e-5

    # And the reported ratio is nowhere near anything anyone would act on.
    assert abs(float(np.mean(ratios))) < 0.35
    assert float(np.std(ratios, ddof=1)) > 0.5
    for run in noise_gross.runs[:5]:
        assert "SYNTHETIC" in run.artifact.comparison.benchmark_description


# ---------------------------------------------------------------------------
# 2. An injected signal of known strength must come back at that strength
# ---------------------------------------------------------------------------


def test_an_injected_signal_is_recovered_at_its_closed_form_strength(
    ladder_gross: tuple[SharpeSweep, ...],
) -> None:
    """Recovery is compared against a number, not against "positive".

    Each seed's measurement is paired with the closed-form prediction for the
    alphas *that seed drew*, which removes the dominant source of scatter and
    leaves a residual whose standard error is a few per cent of the prediction.
    """
    assert len(ladder_gross) == len(INJECTED_STRENGTHS_BPS)
    for sweep, strength in zip(ladder_gross, INJECTED_STRENGTHS_BPS, strict=True):
        assert sweep.spec.alpha_dispersion_bps_per_day == strength
        assert sweep.mean == pytest.approx(sweep.mean_prediction, rel=RECOVERY_RELATIVE_TOLERANCE)
        assert abs(sweep.residual_mean) < RECOVERY_MAX_ABS_RESIDUAL, (
            f"tau={strength}bps: recovered {sweep.mean:+.4f} against a predicted "
            f"{sweep.mean_prediction:+.4f} per period"
        )
        assert abs(sweep.residual_mean) < 4.0 * sweep.residual_standard_error
        # The population closed form and the per-seed one must agree with each
        # other too, or "the prediction" would be ambiguous.
        assert sweep.mean_prediction == pytest.approx(
            sweep.spec.expected_gross_sharpe_per_period(), rel=0.15
        )


def test_recovery_never_exceeds_the_oracle_that_knows_the_answer(
    ladder_gross: tuple[SharpeSweep, ...],
) -> None:
    """A leak detector with a hard ceiling behind it.

    No estimator is better aligned with the outcome than the true alphas, so a
    mean out-of-sample recovery above the oracle is not a better strategy — it is
    a framework that has seen the answer.

    The claim is made about the mean rather than about each seed, because an
    individual seed can clear the ceiling by luck, and it is made in standard
    errors rather than as a bare inequality, because at high estimator fidelity
    the honest recovery approaches the ceiling and a bare inequality would then
    be testing a coin flip. It is the same criterion the pure-noise direction
    uses, where the ceiling is exactly zero — which is what makes it a single
    statement about the framework rather than two unrelated ones.
    """
    for sweep, strength in zip(ladder_gross, INJECTED_STRENGTHS_BPS, strict=True):
        ceiling = sweep.spec.oracle_gross_sharpe_per_period()
        assert ceiling > 0.0
        assert not exceeds_the_oracle(sweep), (
            f"tau={strength}bps: recovered {sweep.mean:+.4f} out of sample against an "
            f"oracle ceiling of {ceiling:+.4f} (standard error {sweep.standard_error:.4f})"
        )
        assert sweep.mean < 1.2 * ceiling
        assert float(np.max(sweep.values)) > 0.0


def test_recovery_tracks_the_injected_strength_rather_than_merely_existing(
    ladder_gross: tuple[SharpeSweep, ...],
) -> None:
    """A framework with a scale error recovers "some" signal at every strength.

    The predicted Sharpe ratios are deliberately **not** proportional to the
    injected ``tau``: estimator fidelity rises with it and so does return
    variance. Matching the predicted step sizes is therefore a test of the
    functional form, not of a constant.
    """
    measured = [sweep.mean for sweep in ladder_gross]
    predicted = [sweep.mean_prediction for sweep in ladder_gross]
    assert measured == sorted(measured)
    assert predicted == sorted(predicted)

    steps_measured = [measured[1] / measured[0], measured[2] / measured[1]]
    steps_predicted = [predicted[1] / predicted[0], predicted[2] / predicted[1]]
    for observed, expected in zip(steps_measured, steps_predicted, strict=True):
        assert observed == pytest.approx(expected, rel=0.20)
    # Doubling tau does not double the recoverable Sharpe ratio, and the ladder
    # would be a much weaker test if it did.
    assert steps_predicted[0] > steps_predicted[1]
    assert steps_predicted[1] < 2.0

    ratios = [sweep.recovery_ratio for sweep in ladder_gross]
    for ratio in ratios:
        assert 1.0 - RECOVERY_RELATIVE_TOLERANCE < ratio < 1.0 + RECOVERY_RELATIVE_TOLERANCE
    assert max(ratios) - min(ratios) < 0.25


def test_the_recovered_signal_is_reduced_by_costs_and_survives_them(
    ladder_gross: tuple[SharpeSweep, ...], ladder_net: tuple[SharpeSweep, ...]
) -> None:
    """Invariant I4 on the signal side: the reportable number is the smaller one."""
    for gross, net, strength in zip(ladder_gross, ladder_net, INJECTED_STRENGTHS_BPS, strict=True):
        assert net.mean < gross.mean, f"tau={strength}bps: costs did not reduce the result"
        drag = gross.mean - net.mean
        assert 0.0 < drag < 0.05
        # A real edge is still there after paying for it.
        assert net.t_statistic > 5.0
        assert net.mean > 0.7 * net.mean_prediction
        assert net.mean_total_cost_bps_of_capital > 0.0
    # Turnover falls as the signal strengthens, so the drag shrinks along the
    # ladder: a stronger estimate changes its mind less often.
    drags = [gross.mean - net.mean for gross, net in zip(ladder_gross, ladder_net, strict=True)]
    assert drags[0] > drags[-1]


# ---------------------------------------------------------------------------
# 3. The overfitting statistics, computed on what the engine produced
# ---------------------------------------------------------------------------


def test_the_search_is_a_real_search_over_real_engine_runs(
    search_noise_gross: tuple[ConfigurationSweep, ...],
) -> None:
    """The trial count PBO and DSR use is how many simulations happened."""
    search = search_noise_gross[0]
    assert search.trials == 6
    assert len(search.candidates) == 6
    assert search.return_matrix.shape == (GATE_N_PERIODS, 6)
    held = [asset for candidate in search.candidates for asset in candidate.assets]
    assert len(held) == len(set(held)) == SEARCH_N_ASSETS
    assert all(run.artifact.stamp.seed == search.spec.seed for run in search.runs)
    assert float(np.std(search.trial_sharpes, ddof=1)) > 0.0


def test_picking_the_in_sample_winner_among_noise_is_a_coin_flip(
    search_noise_gross: tuple[ConfigurationSweep, ...],
    search_noise_net: tuple[ConfigurationSweep, ...],
) -> None:
    """PBO on engine output must land where the theory says: one half.

    The six candidates hold disjoint pairs of instruments, so with no signal
    injected they are exchangeable and nothing but luck separates them. Both cost
    bases are checked: costs are symmetric across candidates here, so they must
    not move PBO.
    """
    for label, searches in (("gross", search_noise_gross), ("net", search_noise_net)):
        values = _pbos(searches)
        standard_error = float(np.std(values, ddof=1)) / math.sqrt(values.size)
        assert 0.30 <= float(np.mean(values)) <= 0.70, f"{label}: mean PBO {values.mean():.3f}"
        assert 0.25 <= float(np.median(values)) <= 0.75
        assert abs(float(np.mean(values)) - 0.5) < 3.0 * standard_error
        # Not degenerate: a framework reporting 0.5 for every seed would pass the
        # bounds above and be measuring nothing.
        assert float(np.std(values, ddof=1)) > 0.10
        assert float(np.min(values)) < 0.25
        assert float(np.max(values)) > 0.75


def test_probability_of_overfitting_collapses_when_the_edge_is_real(
    search_noise_net: tuple[ConfigurationSweep, ...],
    search_signal_net: tuple[ConfigurationSweep, ...],
) -> None:
    """The other side of the same statistic: it must discriminate, not just sit at a half.

    PBO's per-seed spread is wide by construction — about 0.28 — so the sharpest
    discrimination available here is in the medians and in the share of seeds
    falling below a quarter, not in a difference of means divided by a standard
    error the spread makes large. Both are asserted in both directions, so a
    statistic that reported "overfitted" for everything would fail as surely as
    one that reported "genuine" for everything.
    """
    noise = _pbos(search_noise_net)
    signal = _pbos(search_signal_net)
    assert float(np.mean(signal)) < 0.30
    assert float(np.median(signal)) < 0.20
    assert float(np.mean(noise)) - float(np.mean(signal)) > 0.20
    assert float(np.median(noise)) - float(np.median(signal)) > 0.30
    # Two-sided on the share of searches PBO calls generalizable.
    assert float(np.mean(signal < 0.25)) >= 0.70
    assert float(np.mean(noise < 0.25)) <= 0.40


def test_the_deflated_sharpe_ratio_certifies_nothing_found_in_noise(
    search_noise_gross: tuple[ConfigurationSweep, ...],
) -> None:
    """The winner of a search over six worthless candidates must not survive deflation.

    Its raw Sharpe ratio is positive — that is what searching does — and the
    Deflated Sharpe Ratio is the statistic that has to see through it.
    """
    raw = np.array([search.winner.sharpe_per_period for search in search_noise_gross])
    deflated = np.array([search.deflated_sharpe_of_winner().value for search in search_noise_gross])
    assert float(np.mean(raw)) > 0.0
    assert 0.25 <= float(np.mean(deflated)) <= 0.75
    assert float(np.mean(deflated >= 0.95)) <= 0.20
    assert float(np.std(deflated, ddof=1)) > 0.05


def test_falsifying_the_trial_count_manufactures_certainty_from_the_same_noise(
    search_noise_gross: tuple[ConfigurationSweep, ...],
) -> None:
    """Identical data, identical returns, identical formula — only the search is denied.

    This is why the trial count is a required argument with no default anywhere
    in :mod:`backend.backtest.dsr`, and it is demonstrated here on engine output
    rather than on a constructed matrix.
    """
    honest = np.array([search.deflated_sharpe_of_winner().value for search in search_noise_gross])
    denied = np.array([search.undeflated_sharpe_of_winner().value for search in search_noise_gross])
    assert bool(np.all(denied >= honest - 1e-12))
    assert float(np.mean(denied)) - float(np.mean(honest)) > 0.25
    assert float(np.mean(denied)) > 0.75
    for search in search_noise_gross[:3]:
        assert search.deflated_sharpe_of_winner().trials == 6
        assert search.undeflated_sharpe_of_winner().trials == 1


def test_cpcv_paths_of_the_selection_rule_report_nothing_on_noise(
    search_noise_gross: tuple[ConfigurationSweep, ...],
    search_signal_net: tuple[ConfigurationSweep, ...],
) -> None:
    """Combinatorially purged paths of "pick the in-sample winner", on engine output.

    Five complete out-of-sample paths per seed, each covering the whole sample.
    On noise the distribution must sit on zero; with an edge present it must move
    and stay moved.

    The signal claim is made about the distribution, not about every member of
    it: a seed whose search happens to pick badly can land near zero even with a
    real edge present, and a per-seed floor would be asserting that luck never
    goes the wrong way.
    """
    noise = np.array([search.selection_paths().mean for search in search_noise_gross])
    signal = np.array([search.selection_paths().mean for search in search_signal_net])
    noise_error = float(np.std(noise, ddof=1)) / math.sqrt(noise.size)
    signal_error = float(np.std(signal, ddof=1)) / math.sqrt(signal.size)
    assert abs(float(np.mean(noise))) < 3.0 * noise_error
    assert abs(float(np.mean(noise))) < 0.06
    assert float(np.mean(signal)) > 0.20
    assert float(np.mean(signal)) > 5.0 * signal_error
    assert float(np.mean(signal > 0.0)) >= 0.8
    assert float(np.median(signal)) > 0.20

    distribution = search_noise_gross[0].selection_paths()
    assert len(distribution) == 5
    lower, upper = distribution.confidence_interval(0.95)
    assert lower <= distribution.median <= upper


def test_the_cross_validation_the_gate_relies_on_really_purges_and_embargoes() -> None:
    """The splits behind the paths above must actually withhold the neighbours.

    Asserted structurally rather than through a result, because it *cannot* be
    asserted through one here: these labels are point-in-time, so purging removes
    only the observations straddling a test block's edge and dropping it moves
    the reported path Sharpe ratios by almost nothing. A gate that only looked at
    outputs would therefore accept a CPCV with the purge deleted, and the leak it
    lets through grows with the label horizon rather than announcing itself.

    With a one-period label and a one-period embargo, the row immediately before
    a test row carries a label that resolves inside the test block, and the row
    immediately after begins inside the embargo. Neither may be trained on.
    """
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=1.0)
    starts = np.arange(GATE_N_PERIODS, dtype=np.float64)
    splits = cv.split(starts, starts + 1.0)
    assert len(splits) == 15
    assert splits.n_paths == 5

    for split in splits:
        training = set(split.train_indices.tolist())
        tested = set(split.test_indices.tolist())
        assert not (training & tested)
        assert training
        assert split.n_purged + split.n_embargoed > 0
        for index in tested:
            assert index - 1 not in training
            assert index + 1 not in training


# ---------------------------------------------------------------------------
# 4. The mutation — a gate that cannot fail proves nothing
# ---------------------------------------------------------------------------


def test_the_gate_fails_on_a_framework_that_leaks_the_future(
    leak_gross: SharpeSweep, noise_gross: SharpeSweep
) -> None:
    """Break the framework so that it *should* find alpha in noise, and check the gate catches it.

    The defect is an ordinary one: a quarter of the next period's return leaking
    into a 63-period rolling mean, which is what an off-by-one in a rolling window
    does. The data is the same pure noise the honest sweep used — same seeds,
    same samples — so nothing has changed but the bug.

    Every one of the gate's pure-noise criteria must be violated. Asserting on
    the *same* criteria the honest run passes is the point: a mutation that
    failed some other quantity would prove nothing about this gate.
    """
    assert leak_gross.spec.is_pure_noise is True
    assert leak_gross.spec.injects_lookahead is True
    assert leak_gross.seeds == NOISE_SEEDS[: len(LEAK_SEEDS)]
    assert all(run.is_honest is False for run in leak_gross.runs)
    assert "LOOKAHEAD DELIBERATELY INJECTED" in leak_gross.runs[0].disclosure

    failed = noise_criteria(leak_gross)
    assert not failed["t_statistic"], (
        f"a leaking framework produced a mean per-period Sharpe of {leak_gross.mean:+.5f} "
        f"({leak_gross.t_statistic:+.2f} standard errors) and the gate did not notice"
    )
    assert not failed["magnitude"], (
        f"a leaking framework produced an annualized Sharpe of "
        f"{leak_gross.annualized_mean:+.3f} on pure noise and the gate did not notice"
    )
    assert not failed["oracle_ceiling"], (
        f"a leaking framework recovered {leak_gross.mean:+.5f} per period out of sample "
        "from a process whose oracle Sharpe ratio is exactly 0.0, and the gate did not "
        "notice"
    )
    assert exceeds_the_oracle(leak_gross)
    # And it is unambiguously worse than the honest run on the same samples.
    assert leak_gross.mean > noise_gross.mean + 0.03


def test_costs_do_not_rescue_the_gate_from_a_leak(leak_net: SharpeSweep) -> None:
    """The reportable, net-of-cost figure is caught too.

    A leak that survives the spread is the dangerous kind, and this is why the
    noise direction is checked on both cost bases: catching it only in the gross
    diagnostic would leave the number an operator actually sees unguarded.
    """
    failed = noise_criteria(leak_net)
    assert not failed["t_statistic"]
    assert not failed["magnitude"]
    assert not failed["oracle_ceiling"]
    assert leak_net.mean > 0.0
    assert leak_net.annualized_mean > 0.35


def test_the_size_of_the_manufactured_edge_is_recorded(
    leak_gross: SharpeSweep, leak_net: SharpeSweep
) -> None:
    """What a quarter of one bar in sixty-three is worth, in Sharpe ratios.

    Recorded as an assertion rather than a comment because it is the number that
    justifies the whole gate: a leak far too small to notice by reading the code
    produces a result far too good to be true, out of data with nothing in it.
    """
    assert leak_gross.annualized_mean > 0.7
    assert leak_net.annualized_mean > 0.35
    assert leak_gross.mean > leak_net.mean
    # Still nowhere near the "reads tomorrow's price" case, which is the point:
    # the gate does not need the bug to be blatant.
    assert leak_gross.annualized_mean < 3.0


# ---------------------------------------------------------------------------
# 5. Reproducibility of the harness itself (I2)
# ---------------------------------------------------------------------------


def test_the_same_seed_and_configuration_reproduce_the_same_result_exactly() -> None:
    spec = gate_spec(seed=424_242, alpha_dispersion_bps_per_day=20.0)
    first = asyncio.run(run_synthetic_backtest(spec))
    second = asyncio.run(run_synthetic_backtest(spec))
    assert first.artifact.results_digest() == second.artifact.results_digest()
    assert np.array_equal(first.returns, second.returns)
    assert first.stamp == second.stamp

    changed = asyncio.run(run_synthetic_backtest(gate_spec(seed=424_243)))
    assert changed.artifact.results_digest() != first.artifact.results_digest()
    assert changed.stamp.data_version != first.stamp.data_version


def test_the_report_leads_with_its_disclosure_and_shows_prediction_beside_measurement(
    ladder_gross: tuple[SharpeSweep, ...], noise_gross: SharpeSweep
) -> None:
    report = format_report([noise_gross, *ladder_gross])
    first_line = report.splitlines()[0]
    assert first_line.startswith("SYNTHETIC DATA")
    assert "not a research finding" in first_line
    assert "predicted SR" in report
    assert "measured SR" in report
    assert report.count("\n") > len(ladder_gross)
    with pytest.raises(ValueError, match="at least one sweep"):
        format_report([])


def test_a_candidate_family_must_be_exchangeable_to_be_usable() -> None:
    market = build_synthetic_market(search_spec(seed=7, alpha_dispersion_bps_per_day=0.0))
    with pytest.raises(ValueError, match="equal blocks"):
        disjoint_asset_blocks(market, block_size=5)
    with pytest.raises(ValueError, match="equal blocks"):
        disjoint_asset_blocks(market, block_size=12)
    blocks = disjoint_asset_blocks(market, block_size=3)
    assert len(blocks) == 4
    assert all(len(block.assets) == 3 for block in blocks)
    with pytest.raises(ValueError, match="must be named"):
        StrategyCandidate(label="  ", assets=("SYNTH-00",))
    with pytest.raises(ValueError, match="holds no assets"):
        StrategyCandidate(label="empty", assets=())


# ---------------------------------------------------------------------------
# 6. The harness's own contracts
#
# A gate is only as trustworthy as the instrument behind it, so the instrument's
# refusals are asserted rather than assumed. Every case below is a way of
# producing a number that would look like a gate result and would not be one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": -1}, "non-negative int"),
        ({"seed": True}, "non-negative int"),
        ({"n_assets": 1}, "at least 2 for a cross-sectional book"),
        ({"n_periods": 31}, "at least 32"),
        ({"lookback_periods": 1}, "lookback_periods must be at least 2"),
        ({"idiosyncratic_volatility_bps_per_day": 0.0}, "strictly positive"),
        ({"benchmark_volatility_bps_per_day": -1.0}, "strictly positive"),
        ({"adv_usd": 0.0}, "strictly positive"),
        ({"initial_capital_usd": -1.0}, "strictly positive"),
        ({"periods_per_year": 0.0}, "strictly positive"),
        ({"alpha_dispersion_bps_per_day": -1.0}, "injected signal strength"),
        ({"alpha_dispersion_bps_per_day": math.nan}, "injected signal strength"),
        ({"benchmark_drift_bps_per_day": math.inf}, "must be finite"),
        ({"lookahead_leak_fraction": 1.5}, "defect injector"),
        ({"lookahead_leak_fraction": -0.1}, "defect injector"),
        ({"bootstrap_resamples": 1}, "at least 2"),
    ],
)
def test_an_unusable_process_is_refused_rather_than_silently_adjusted(
    overrides: dict[str, object], message: str
) -> None:
    parameters: dict[str, object] = {
        "seed": 1,
        "n_assets": GATE_N_ASSETS,
        "n_periods": GATE_N_PERIODS,
        "lookback_periods": GATE_LOOKBACK_PERIODS,
        "idiosyncratic_volatility_bps_per_day": GATE_VOLATILITY_BPS,
    }
    parameters.update(overrides)
    with pytest.raises(ValueError, match=message):
        SyntheticSpec(**parameters)  # type: ignore[arg-type]


def test_a_strategy_cannot_be_built_over_instruments_that_do_not_exist() -> None:
    market = build_synthetic_market(_noise_spec())
    with pytest.raises(ValueError, match="at least one asset"):
        market.strategy(assets=())
    with pytest.raises(ValueError, match="no assets named"):
        market.strategy(assets=("AAPL",))
    with pytest.raises(ValueError, match="lookback_periods must be at least 2"):
        market.strategy(lookback_periods=1)
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        market.strategy(lookahead_leak_fraction=2.0)


def test_a_market_whose_calendar_disagrees_with_its_process_is_refused() -> None:
    market = build_synthetic_market(_noise_spec())
    with pytest.raises(ValueError, match=r"calendar holds \d+ instants"):
        SyntheticMarket(
            spec=market.spec,
            calendar=market.calendar[:-1],
            data=market.data,
            assets=market.assets,
            benchmark_asset=market.benchmark_asset,
            true_alpha_bps_per_day=market.true_alpha_bps_per_day,
            forward_returns=market.forward_returns,
        )


def test_a_sweep_refuses_to_average_over_things_that_are_not_comparable() -> None:
    spec = gate_spec(seed=1)
    one = asyncio.run(run_synthetic_backtest(spec))
    same = asyncio.run(run_synthetic_backtest(spec))
    other_mode = asyncio.run(
        run_synthetic_backtest(gate_spec(seed=2), cost_mode=CostMode.ZERO_COST_DIAGNOSTIC)
    )
    other_process = asyncio.run(
        run_synthetic_backtest(gate_spec(seed=2, alpha_dispersion_bps_per_day=20.0))
    )
    with pytest.raises(ValueError, match="at least 2 seeds"):
        SharpeSweep(runs=(one,))
    with pytest.raises(ValueError, match="mixes cost modes"):
        SharpeSweep(runs=(one, other_mode))
    with pytest.raises(ValueError, match="more than their seed"):
        SharpeSweep(runs=(one, other_process))
    with pytest.raises(ValueError, match="seeds must be distinct"):
        SharpeSweep(runs=(one, same))


def test_a_configuration_sweep_refuses_a_search_it_cannot_rank() -> None:
    spec = search_spec(seed=11, alpha_dispersion_bps_per_day=0.0)
    market = build_synthetic_market(spec)
    candidates = disjoint_asset_blocks(market, block_size=2)
    sweep = asyncio.run(sweep_configurations(spec, candidates=candidates, market=market))
    with pytest.raises(ValueError, match="at least 2 candidates"):
        ConfigurationSweep(
            spec=spec,
            cost_mode=sweep.cost_mode,
            runs=sweep.runs[:1],
            candidates=sweep.candidates[:1],
        )
    with pytest.raises(ValueError, match=r"runs for \d+ candidates"):
        ConfigurationSweep(
            spec=spec,
            cost_mode=sweep.cost_mode,
            runs=sweep.runs,
            candidates=sweep.candidates[:2],
        )


def test_every_serialized_payload_carries_its_refusal_to_be_read_as_a_result() -> None:
    """Whatever reaches a UI, a log or a file says what it is (invariant I3)."""
    spec = search_spec(seed=13, alpha_dispersion_bps_per_day=0.0)
    market = build_synthetic_market(spec)
    search = asyncio.run(
        sweep_configurations(
            spec, candidates=disjoint_asset_blocks(market, block_size=2), market=market
        )
    )
    run_payload = search.winner.to_dict()
    assert run_payload["is_research_finding"] is False
    assert "SYNTHETIC DATA" in str(run_payload["disclosure"])
    assert run_payload["is_honest"] is True
    reproducibility = run_payload["reproducibility"]
    assert isinstance(reproducibility, dict)
    assert reproducibility["seed"] == spec.seed
    assert str(reproducibility["data_version"]).startswith(SYNTHETIC_DATA_VERSION_PREFIX)
    predicted = run_payload["predicted"]
    assert isinstance(predicted, dict)
    assert predicted["conditional_gross_sharpe_per_period"] == 0.0

    search_payload = search.to_dict()
    assert search_payload["is_research_finding"] is False
    assert search_payload["trials"] == 6
    assert "SYNTHETIC DATA" in str(search_payload["disclosure"])

    sweep = asyncio.run(sweep_seeds(gate_spec(seed=17), seeds=(17, 18, 19)))
    sweep_payload = sweep.to_dict()
    assert sweep_payload["is_research_finding"] is False
    assert sweep_payload["n_seeds"] == 3
    assert sweep_payload["predicted_gross_sharpe_per_period"] == 0.0
    assert math.isnan(float(str(sweep_payload["recovery_ratio"])))
