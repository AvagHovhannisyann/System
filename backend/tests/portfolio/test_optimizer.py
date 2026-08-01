"""cvxpy portfolio optimizer: objective, constraints, infeasibility, reporting (P9.2).

Directive §5 Phase 9 asks for a specific problem — maximize expected return
minus a risk penalty minus an **explicit turnover penalty**, subject to sector
neutrality, beta neutrality, a 2% position cap, a 20% sector cap and full
investment — and its gate (G9) asks for two things beyond "it solves": that the
turnover penalty *demonstrably* reduces realized turnover, and that the
optimizer does not fall over on real universes.

The tests here are organized around the failure that would be invisible if it
happened. An optimizer that returns weights violating its constraints does not
crash; it produces a backtest. So:

- every constraint is re-checked **in NumPy against the returned array**, never
  read back from the solver;
- the max-return corner at zero risk aversion is checked against a **closed-form
  knapsack**, not against the optimizer's own opinion;
- infeasibility is checked to *refuse* — and, when the ladder is enabled, to
  report the rung it landed on and what it loosened, so that a relaxed portfolio
  is never indistinguishable from a requested one;
- the constraint-binding indicators (dashboard §6.8) are checked against
  independent predictions — at zero risk aversion exactly ``net / cap`` names sit
  on the position cap, and a sector target equal to the sector cap binds every
  sector — rather than only against a recomputation of the formula that produced
  them.

Universes are built from seeded factor models and passed through the real
Ledoit-Wolf estimator (P9.1), so the risk model these tests optimize against is
the one production uses, shrinkage intensity and all.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.portfolio.constraints import (
    DEFAULT_CONSTRAINTS,
    RELAXATION_LADDER,
    PortfolioConstraints,
    RelaxableConstraint,
    RelaxationLadder,
)
from backend.portfolio.covariance import (
    MINIMUM_OBSERVATIONS_FOR_SHRINKAGE,
    ShrinkageCovariance,
    ledoit_wolf_covariance,
)
from backend.portfolio.errors import NotPositiveSemiDefiniteError
from backend.portfolio.optimizer import (
    _INFEASIBLE_STATUSES,
    _RETRYABLE_STATUSES,
    _SOLVED_STATUSES,
    _SOLVER_OPTIONS,
    BINDING_TOLERANCE,
    CONSTRAINT_TOLERANCE,
    DEFAULT_SOLVERS,
    MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION,
    OptimizationResult,
    optimize_portfolio,
    turnover_penalty_from_round_trip_cost_bps,
)
from backend.portfolio.optimizer_errors import (
    InfeasibleProblemError,
    InsufficientRiskHistoryError,
    OptimizerInputError,
)

if TYPE_CHECKING:
    import numpy.typing as npt

# A daily-return scale expressed as a fraction — the unit the whole package
# documents. 2% daily idiosyncratic vol, 1% daily factor vol.
_DAILY_VOL = 0.02
_FACTOR_VOL = 0.01
_ALPHA_SCALE = 0.01


@dataclass(frozen=True)
class _Universe:
    """A universe to optimize over: risk model, sectors, betas, alphas."""

    covariance: ShrinkageCovariance
    sectors: tuple[str, ...]
    betas: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    assets: tuple[str, ...]


@functools.cache
def _universe(
    *,
    n_assets: int = 100,
    n_sectors: int = 8,
    seed: int = 0,
    n_observations: int = MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION,
    n_factors: int = 3,
    equal_betas: bool = False,
) -> _Universe:
    """Build a universe from a seeded factor model, through the real estimator.

    Sectors are assigned round-robin, which keeps every sector at ``n / k`` names
    — the balanced case in which the equal-weighted portfolio is feasible under
    the shipped constraints whenever ``n_assets >= 50`` and ``n_sectors >= 5``.
    """
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, _FACTOR_VOL, size=(n_observations, n_factors))
    loadings = rng.normal(0.0, 1.0, size=(n_factors, n_assets))
    returns = factors @ loadings + rng.normal(0.0, _DAILY_VOL, size=(n_observations, n_assets))
    assets = tuple(f"A{index:03d}" for index in range(n_assets))
    return _Universe(
        covariance=ledoit_wolf_covariance(returns, assets=assets),
        sectors=tuple(f"SECTOR_{index % n_sectors}" for index in range(n_assets)),
        betas=(
            np.ones(n_assets, dtype=np.float64)
            if equal_betas
            else rng.normal(1.0, 0.3, size=n_assets)
        ),
        alpha=rng.normal(0.0, _ALPHA_SCALE, size=n_assets),
        assets=assets,
    )


def _optimize(universe: _Universe, **overrides: object) -> OptimizationResult:
    """Run the optimizer over a universe with sensible test defaults."""
    kwargs: dict[str, object] = {
        "expected_returns": universe.alpha,
        "covariance": universe.covariance,
        "sectors": universe.sectors,
        "betas": universe.betas,
        "risk_aversion": 5.0,
        "turnover_penalty": 0.0,
    }
    kwargs.update(overrides)
    return optimize_portfolio(**kwargs)  # type: ignore[arg-type]


def _sector_matrix(sectors: tuple[str, ...]) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    """Return the sorted sector labels and their 0/1 membership matrix."""
    labels = tuple(sorted(set(sectors)))
    indicator = np.zeros((len(labels), len(sectors)), dtype=np.float64)
    for column, label in enumerate(sectors):
        indicator[labels.index(label), column] = 1.0
    return labels, indicator


def _assert_constraints_hold(
    result: OptimizationResult, universe: _Universe, constraints: PortfolioConstraints
) -> None:
    """Re-check every constraint in NumPy against the weights actually returned.

    This deliberately does not consult the solver, the result's own exposure
    fields, or the optimizer's internal verifier: it recomputes from
    ``result.weights``, which is the array a backtest would receive.
    """
    weights = np.asarray(result.weights, dtype=np.float64)
    labels, indicator = _sector_matrix(universe.sectors)

    assert np.isfinite(weights).all()
    assert float(np.max(np.abs(weights))) <= constraints.position_cap + CONSTRAINT_TOLERANCE
    assert abs(float(weights.sum()) - constraints.net_exposure) <= CONSTRAINT_TOLERANCE
    assert float(np.abs(weights).sum()) <= constraints.gross_exposure + CONSTRAINT_TOLERANCE

    sector_gross = indicator @ np.abs(weights)
    assert float(np.max(sector_gross)) <= constraints.sector_cap + CONSTRAINT_TOLERANCE

    targets = np.asarray([result.sector_targets[label] for label in labels], dtype=np.float64)
    sector_net = indicator @ weights
    deviation = float(np.max(np.abs(sector_net - targets)))
    assert deviation <= constraints.sector_neutrality_band + CONSTRAINT_TOLERANCE

    beta_deviation = abs(float(universe.betas @ weights) - result.target_beta)
    assert beta_deviation <= constraints.beta_neutrality_band + CONSTRAINT_TOLERANCE


# --------------------------------------------------------------------------
# It solves the directive's problem
# --------------------------------------------------------------------------


def test_solves_the_directive_constraint_set_unrelaxed() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe)

    assert result.status == "optimal"
    assert result.rung == 0
    assert result.relaxations == ()
    assert result.is_relaxed is False
    assert result.constraints == DEFAULT_CONSTRAINTS
    assert result.requested_constraints == DEFAULT_CONSTRAINTS
    assert result.full_investment_satisfied is True
    _assert_constraints_hold(result, universe, DEFAULT_CONSTRAINTS)


def test_default_book_is_long_only_as_a_consequence_of_the_no_leverage_rule() -> None:
    """sum(w) == 1 with sum(|w|) <= 1 forces w >= 0; that is arithmetic, not a flag."""
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe)
    assert float(np.min(np.asarray(result.weights))) >= -CONSTRAINT_TOLERANCE
    assert result.gross_exposure == pytest.approx(1.0, abs=CONSTRAINT_TOLERANCE)


def test_weights_are_read_only() -> None:
    """A mutable portfolio would leave the reported exposures describing something else."""
    result = _optimize(_universe(n_assets=100, n_sectors=8, seed=1))
    with pytest.raises(ValueError, match="read-only"):
        result.weights[0] = 0.5


def test_reported_exposures_are_recomputed_from_the_returned_weights() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe, risk_aversion=8.0, turnover_penalty=0.001)
    weights = np.asarray(result.weights)
    covariance = np.asarray(universe.covariance.covariance)

    assert result.expected_return == pytest.approx(float(universe.alpha @ weights), abs=1e-12)
    assert result.variance == pytest.approx(float(weights @ covariance @ weights), abs=1e-18)
    assert result.volatility == pytest.approx(math.sqrt(result.variance), abs=1e-15)
    assert result.net_exposure == pytest.approx(float(weights.sum()), abs=1e-15)
    assert result.gross_exposure == pytest.approx(float(np.abs(weights).sum()), abs=1e-15)
    assert result.portfolio_beta == pytest.approx(float(universe.betas @ weights), abs=1e-12)


def test_objective_decomposition_adds_up() -> None:
    """The three terms the directive names are reported separately and reconcile."""
    result = _optimize(
        _universe(n_assets=100, n_sectors=8, seed=1), risk_aversion=8.0, turnover_penalty=0.002
    )
    assert result.risk_penalty == pytest.approx(8.0 * result.variance, abs=1e-15)
    assert result.turnover_cost == pytest.approx(0.002 * result.turnover, abs=1e-15)
    assert result.objective_value == pytest.approx(
        result.expected_return - result.risk_penalty - result.turnover_cost, abs=1e-15
    )


def test_turnover_uses_the_two_sided_convention() -> None:
    """An initial rebalance out of cash into a fully invested book costs 1.0, not 0.5.

    The halved "one-way" convention is equally common in the literature, which
    is exactly why the choice is pinned rather than assumed: a backtest reading
    turnover under the other convention understates trading by half.
    """
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe)
    assert result.turnover == pytest.approx(1.0, abs=1e-6)
    assert result.summary()["turnover_convention"] == (
        "turnover = sum(|w - w_prev|), total traded notional as a fraction of capital (not halved)"
    )


# --------------------------------------------------------------------------
# Units on the objective's coefficients
# --------------------------------------------------------------------------


def test_turnover_penalty_from_round_trip_cost_bps_converts_basis_points() -> None:
    assert turnover_penalty_from_round_trip_cost_bps(20.0) == pytest.approx(0.0020)
    assert turnover_penalty_from_round_trip_cost_bps(0.0) == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_turnover_penalty_conversion_refuses_impossible_costs(value: float) -> None:
    with pytest.raises(OptimizerInputError, match="basis points"):
        turnover_penalty_from_round_trip_cost_bps(value)


def test_objective_coefficients_have_no_defaults() -> None:
    """A defaulted-to-zero turnover penalty would silently delete a term §5 requires.

    A default risk aversion would be a portfolio-policy number invented by a
    library. Both are required arguments, and this pins that they stay so.
    """
    parameters = inspect.signature(optimize_portfolio).parameters
    for name in ("risk_aversion", "turnover_penalty"):
        assert parameters[name].default is inspect.Parameter.empty
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


# --------------------------------------------------------------------------
# The turnover penalty demonstrably reduces turnover (G9)
# --------------------------------------------------------------------------


def _turnover_sweep(penalties: tuple[float, ...]) -> list[OptimizationResult]:
    """Optimize the same rebalance at a rising sequence of turnover penalties."""
    universe = _universe(n_assets=120, n_sectors=8, seed=2)
    held = _optimize(universe, risk_aversion=5.0, turnover_penalty=0.0)
    # A genuinely different alpha, so there is something to trade toward.
    revised = _universe(n_assets=120, n_sectors=8, seed=3).alpha
    return [
        _optimize(
            universe,
            expected_returns=revised,
            risk_aversion=5.0,
            turnover_penalty=penalty,
            previous_weights=np.asarray(held.weights),
        )
        for penalty in penalties
    ]


def test_raising_the_turnover_penalty_never_raises_realized_turnover() -> None:
    """G9's requirement, stated as the monotonicity it actually is.

    Convexity makes this weak monotonicity, not strict: over a range where the
    penalty does not yet change the optimum the turnover is flat. Asserting
    strict decrease everywhere would be asserting something false and would be
    "fixed" later by weakening the test — so the strict claim is made once,
    between the ends of the sweep, where it is true and load-bearing.
    """
    penalties = (0.0, 0.0005, 0.002, 0.01, 0.05)
    results = _turnover_sweep(penalties)
    turnovers = [result.turnover for result in results]

    for earlier, later in itertools.pairwise(turnovers):
        assert later <= earlier + CONSTRAINT_TOLERANCE, turnovers
    assert turnovers[-1] < turnovers[0] - 0.1, turnovers


def test_a_large_turnover_penalty_holds_the_existing_book() -> None:
    """The book it already holds is feasible, so a heavy enough penalty stays put."""
    results = _turnover_sweep((0.0, 1.0))
    assert results[-1].turnover == pytest.approx(0.0, abs=1e-5)
    # And it is still a legal portfolio, not merely an unchanged one.
    universe = _universe(n_assets=120, n_sectors=8, seed=2)
    _assert_constraints_hold(results[-1], universe, DEFAULT_CONSTRAINTS)


def test_the_turnover_penalty_is_paid_for_in_expected_return() -> None:
    """Trading less is not free; if it were, the penalty would not be shaping anything."""
    results = _turnover_sweep((0.0, 0.05))
    assert results[-1].expected_return < results[0].expected_return


def test_zero_turnover_penalty_is_permitted_as_the_control_arm() -> None:
    """G9's A/B needs an arm with the term switched off, so 0.0 is legal, not refused."""
    result = _optimize(_universe(n_assets=100, n_sectors=8, seed=1), turnover_penalty=0.0)
    assert result.turnover_cost == 0.0
    assert result.turnover_penalty == 0.0


# --------------------------------------------------------------------------
# Zero risk aversion reaches the capped maximum-return corner
# --------------------------------------------------------------------------


def _unconstrained_corner_universe() -> _Universe:
    """A universe whose sector and beta constraints are arithmetically vacuous.

    One sector with a cap of 100% and identical betas make sector neutrality and
    beta neutrality restatements of ``sum(w) == 1``. What is left is exactly the
    fractional knapsack ``max alpha'w`` subject to ``sum(w) = 1``,
    ``|w_i| <= 2%`` — which has a closed form to check against.
    """
    base = _universe(n_assets=100, n_sectors=8, seed=4, equal_betas=True)
    return _Universe(
        covariance=base.covariance,
        sectors=("ONLY",) * len(base.sectors),
        betas=base.betas,
        alpha=base.alpha,
        assets=base.assets,
    )


def test_zero_risk_aversion_reaches_the_closed_form_maximum_return_corner() -> None:
    universe = _unconstrained_corner_universe()
    constraints = PortfolioConstraints(sector_cap=1.0)
    result = _optimize(universe, risk_aversion=0.0, turnover_penalty=0.0, constraints=constraints)

    cap = constraints.position_cap
    held = round(constraints.net_exposure / cap)
    expected = np.zeros(universe.alpha.size, dtype=np.float64)
    expected[np.argsort(universe.alpha)[::-1][:held]] = cap

    np.testing.assert_allclose(np.asarray(result.weights), expected, atol=1e-6)
    assert result.expected_return == pytest.approx(float(universe.alpha @ expected), abs=1e-9)
    _assert_constraints_hold(result, universe, constraints)


def test_zero_risk_aversion_dominates_a_risk_averse_solution_on_expected_return() -> None:
    """The corner is the *maximum* return point; risk aversion trades return for variance."""
    universe = _unconstrained_corner_universe()
    constraints = PortfolioConstraints(sector_cap=1.0)
    corner = _optimize(universe, risk_aversion=0.0, constraints=constraints)
    cautious = _optimize(universe, risk_aversion=50.0, constraints=constraints)

    assert corner.expected_return > cautious.expected_return
    assert corner.variance > cautious.variance


def test_zero_risk_aversion_puts_exactly_the_knapsack_count_on_the_position_cap() -> None:
    """An independent prediction for the binding indicators: 1.0 / 2% = 50 names."""
    universe = _unconstrained_corner_universe()
    constraints = PortfolioConstraints(sector_cap=1.0)
    result = _optimize(universe, risk_aversion=0.0, constraints=constraints)
    assert len(result.binding_position_caps) == 50


# --------------------------------------------------------------------------
# Constraint-binding indicators (dashboard §6.8)
# --------------------------------------------------------------------------


def test_binding_indicators_match_the_weights_they_describe() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe)
    weights = np.asarray(result.weights)
    labels, indicator = _sector_matrix(universe.sectors)
    cap = result.constraints.position_cap

    expected_positions = {
        universe.assets[index]
        for index in range(weights.size)
        if abs(float(weights[index])) >= cap - BINDING_TOLERANCE
    }
    assert set(result.binding_position_caps) == expected_positions
    assert expected_positions, "a 2% cap over 100 names must bind somewhere"

    # And the complement really is strictly inside the cap.
    for index in range(weights.size):
        if universe.assets[index] not in expected_positions:
            assert abs(float(weights[index])) < cap - BINDING_TOLERANCE

    sector_gross = indicator @ np.abs(weights)
    expected_sectors = {
        labels[index]
        for index in range(len(labels))
        if float(sector_gross[index]) >= result.constraints.sector_cap - BINDING_TOLERANCE
    }
    assert set(result.binding_sector_caps) == expected_sectors


def test_sector_caps_bind_exactly_when_the_target_reaches_the_cap() -> None:
    """Five equal sectors put 20% in each — on the cap. Ten put 10% — clear of it.

    This is an independent prediction rather than a recomputation: the indicator
    has to agree with arithmetic done outside the optimizer.
    """
    on_the_cap = _universe(n_assets=100, n_sectors=5, seed=5)
    clear_of_it = _universe(n_assets=100, n_sectors=10, seed=5)

    tight = _optimize(on_the_cap)
    assert set(tight.binding_sector_caps) == set(on_the_cap.sectors)

    loose = _optimize(clear_of_it)
    assert loose.binding_sector_caps == ()


def test_gross_exposure_binds_at_the_fully_invested_default() -> None:
    """It is why the default book is long-only, so §6.8 should be able to show it."""
    result = _optimize(_universe(n_assets=100, n_sectors=8, seed=1))
    assert result.binding_gross_exposure is True
    assert result.summary()["binding_gross_exposure"] is True


def test_binding_indicators_are_reported_against_the_constraints_actually_used() -> None:
    """After a relaxation the indicator must describe the widened cap, not the asked-for one."""
    universe = _universe(n_assets=80, n_sectors=4, seed=6)
    result = _optimize(universe)

    assert result.is_relaxed is True
    assert result.constraints.sector_cap > result.requested_constraints.sector_cap
    _, indicator = _sector_matrix(universe.sectors)
    sector_gross = indicator @ np.abs(np.asarray(result.weights))
    assert float(np.max(sector_gross)) > result.requested_constraints.sector_cap
    assert float(np.max(sector_gross)) <= result.constraints.sector_cap + CONSTRAINT_TOLERANCE


# --------------------------------------------------------------------------
# Infeasibility: diagnosed, never silently relaxed
# --------------------------------------------------------------------------


def test_infeasible_problem_refuses_when_relaxation_is_disabled() -> None:
    """No ladder means no approximation: the answer is an error, not a portfolio."""
    universe = _universe(n_assets=80, n_sectors=4, seed=6)
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(universe, relaxation_ladder=None)

    error = caught.value
    assert error.relaxation_enabled is False
    assert len(error.attempts) == 1
    assert "Relaxation was disabled by the caller" in str(error)


def test_infeasibility_diagnosis_names_the_conflicting_constraints() -> None:
    """Four equal sectors need 25% each against a 20% sector cap. Say so, in numbers."""
    universe = _universe(n_assets=80, n_sectors=4, seed=6)
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(universe, relaxation_ladder=None)

    diagnosis = " ".join(caught.value.diagnosis)
    assert "sector cap" in diagnosis
    assert "25.00%" in diagnosis
    assert "20.00%" in diagnosis
    assert "full-investment target" in diagnosis


def test_a_universe_too_small_to_fill_the_book_is_diagnosed_by_capacity() -> None:
    universe = _universe(n_assets=30, n_sectors=6, seed=7)
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(universe, relaxation_ladder=None)

    diagnosis = " ".join(caught.value.diagnosis)
    assert "cannot fill the book" in diagnosis
    assert "30 name(s)" in diagnosis
    assert "60.00%" in diagnosis


def test_sector_targets_that_do_not_sum_to_the_net_exposure_are_refused_up_front() -> None:
    """The most common way a sector-neutral optimizer becomes mysteriously infeasible.

    Summing the per-sector equalities gives sum(w); so does full investment.
    Two statements about one number, refused before a solver ever sees them.
    """
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    targets = dict.fromkeys(sorted(set(universe.sectors)), 0.5 / 8)
    with pytest.raises(OptimizerInputError, match=r"0\.5") as caught:
        _optimize(universe, sector_targets=targets)
    assert "full-investment" in str(caught.value)


def test_sector_targets_must_cover_exactly_the_sectors_present() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    labels = sorted(set(universe.sectors))
    targets = dict.fromkeys(labels[:-1], 1.0 / 7)
    with pytest.raises(OptimizerInputError, match="Missing"):
        _optimize(universe, sector_targets=targets)


def test_the_ladder_is_climbed_one_rung_at_a_time_in_the_documented_order() -> None:
    universe = _universe(n_assets=80, n_sectors=4, seed=6)
    result = _optimize(universe)

    assert [attempt.rung for attempt in result.attempts] == list(range(result.rung + 1))
    for attempt in result.attempts:
        assert attempt.relaxations == RELAXATION_LADDER[: attempt.rung]
    for attempt in result.attempts[:-1]:
        assert attempt.status in _INFEASIBLE_STATUSES
    assert result.attempts[-1].status == "optimal"


def test_a_relaxed_result_cannot_be_mistaken_for_the_requested_one() -> None:
    """The entire point: a portfolio the policy forbids must announce itself."""
    universe = _universe(n_assets=80, n_sectors=4, seed=6)
    result = _optimize(universe)

    assert result.is_relaxed is True
    assert result.rung > 0
    assert result.relaxations != ()
    assert result.constraints != result.requested_constraints
    assert result.summary()["is_relaxed"] is True
    assert result.summary()["rung"] == result.rung
    described = result.summary()["relaxations"]
    assert isinstance(described, list)
    assert len(described) == len(result.relaxations)
    # Whatever was relaxed, the solution still satisfies the set it was solved
    # under — a relaxation is a different problem, not a licence to miss.
    _assert_constraints_hold(result, universe, result.constraints)


def test_relaxations_are_the_ladder_prefix_and_carry_their_units() -> None:
    universe = _universe(n_assets=80, n_sectors=4, seed=6)
    result = _optimize(universe)

    assert tuple(item.constraint for item in result.relaxations) == RELAXATION_LADDER[: result.rung]
    for item in result.relaxations:
        assert item.units
        assert item.rationale
        assert item.realized is not None


def test_the_position_cap_survives_every_rung() -> None:
    """It bounds single-name loss; no rung may buy feasibility with it."""
    for n_assets, n_sectors, seed in ((80, 4, 6), (30, 6, 7)):
        universe = _universe(n_assets=n_assets, n_sectors=n_sectors, seed=seed)
        result = _optimize(universe)
        assert result.constraints.position_cap == DEFAULT_CONSTRAINTS.position_cap
        assert result.constraints.gross_exposure == DEFAULT_CONSTRAINTS.gross_exposure
        for attempt in result.attempts:
            assert attempt.constraints.position_cap == DEFAULT_CONSTRAINTS.position_cap
        assert RelaxableConstraint.SECTOR_CAP in RELAXATION_LADDER
        weights = np.asarray(result.weights)
        assert float(np.max(np.abs(weights))) <= DEFAULT_CONSTRAINTS.position_cap + 1e-6


def test_the_full_investment_rung_deploys_what_the_universe_allows() -> None:
    """30 names at a 2% cap reach 60% of capital, and that is what comes back."""
    universe = _universe(n_assets=30, n_sectors=6, seed=7)
    result = _optimize(universe)

    assert result.rung == len(RELAXATION_LADDER)
    assert result.net_exposure == pytest.approx(0.60, abs=1e-4)
    assert result.full_investment_satisfied is False
    reduction = result.relaxations[-1]
    assert reduction.constraint is RelaxableConstraint.FULL_INVESTMENT
    assert reduction.original == 1.0
    assert reduction.was_consumed is True
    _assert_constraints_hold(result, universe, result.constraints)


def test_the_reduced_book_stays_neutral_in_proportion() -> None:
    """Scaling the book without scaling its targets would tilt it by the reduction."""
    universe = _universe(n_assets=30, n_sectors=6, seed=7)
    result = _optimize(universe)
    scale = result.net_exposure / DEFAULT_CONSTRAINTS.net_exposure
    for label, target in result.sector_targets.items():
        assert target == pytest.approx(scale / 6.0, rel=1e-6), label
    assert result.target_beta == pytest.approx(scale * float(universe.betas.mean()), rel=1e-6)


def test_a_book_below_the_ladder_floor_is_refused_and_the_floor_is_named() -> None:
    """The last rung declines rather than handing back a mostly-cash portfolio.

    A book that can deploy 60% against a floor of 90% is a universe problem for
    a human. The refusal says which number to look at — that sentence is the
    whole value of the ladder's last rung reporting itself.
    """
    universe = _universe(n_assets=30, n_sectors=6, seed=7)
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(universe, relaxation_ladder=RelaxationLadder(minimum_net_exposure=0.90))

    final = caught.value.attempts[-1]
    assert final.rung == len(RELAXATION_LADDER)
    assert final.status.startswith("refused")
    assert "60.00%" in final.status
    assert "90.00%" in final.status


def test_an_exhausted_ladder_returns_no_weights_at_all() -> None:
    universe = _universe(n_assets=30, n_sectors=6, seed=7)
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(universe, relaxation_ladder=RelaxationLadder(minimum_net_exposure=0.90))

    error = caught.value
    assert error.relaxation_enabled is True
    assert len(error.attempts) == len(RELAXATION_LADDER) + 1
    assert "No weights are returned" in str(error)
    assert not hasattr(error, "weights")


def test_infeasibility_without_an_arithmetic_conflict_says_so() -> None:
    """Silence from the diagnosis is itself a finding, and it is stated as one."""
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    # Feasible on every necessary condition taken alone; the conflict is that a
    # beta of exactly 3.0 is unreachable jointly with sector neutrality.
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(universe, target_beta=3.0, relaxation_ladder=None)
    message = str(caught.value)
    assert ("Necessary conditions violated" in message) or (
        "the conflict is in their interaction" in message
    )


# --------------------------------------------------------------------------
# The risk-model policy D-020 left to this layer
# --------------------------------------------------------------------------


def test_a_short_risk_history_is_refused_with_its_shrinkage_intensity() -> None:
    universe = _universe(n_assets=40, n_sectors=8, seed=8, n_observations=100)
    with pytest.raises(InsufficientRiskHistoryError) as caught:
        _optimize(universe)

    error = caught.value
    assert error.n_observations == 100
    assert error.minimum_observations == MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION
    assert error.shrinkage_intensity == universe.covariance.shrinkage_intensity
    assert "D-020" in str(error)


def test_the_observation_policy_is_a_default_not_a_law() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=8, n_observations=100)
    result = _optimize(universe, minimum_observations=100)
    assert result.minimum_observations == 100
    assert result.n_observations == 100
    # And it is recorded on the result, so an artifact says what policy it ran under.
    assert result.summary()["minimum_observations"] == 100


def test_a_policy_below_the_estimators_own_floor_is_refused() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    with pytest.raises(OptimizerInputError, match="Ledoit-Wolf"):
        _optimize(universe, minimum_observations=MINIMUM_OBSERVATIONS_FOR_SHRINKAGE - 1)


def test_the_shrinkage_intensity_is_carried_onto_the_result() -> None:
    """It is the direct measure of how much of the risk model was assumption."""
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe)
    assert result.shrinkage_intensity == universe.covariance.shrinkage_intensity
    assert 0.0 <= result.shrinkage_intensity <= 1.0


def test_a_covariance_that_is_not_psd_is_refused() -> None:
    """A self-check: a risk model mutated or built outside the estimator is not optimized."""
    indefinite = np.array([[1e-4, 0.0], [0.0, -1e-2]], dtype=np.float64)
    covariance = ShrinkageCovariance(
        covariance=indefinite,
        sample_covariance=indefinite,
        target=indefinite,
        shrinkage_intensity=0.1,
        target_variance=1e-4,
        n_observations=MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION,
        n_assets=2,
        min_eigenvalue=-1e-2,
        max_eigenvalue=1e-4,
        condition_number=-1.0,
        assets=None,
    )
    with pytest.raises(NotPositiveSemiDefiniteError):
        optimize_portfolio(
            expected_returns=np.array([0.01, 0.01]),
            covariance=covariance,
            sectors=["A", "B"],
            betas=np.array([1.0, 1.0]),
            risk_aversion=1.0,
            turnover_penalty=0.0,
        )


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_non_finite_expected_returns_are_refused_not_imputed() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    alpha = universe.alpha.copy()
    alpha[3] = np.nan
    with pytest.raises(OptimizerInputError, match="non-finite"):
        _optimize(universe, expected_returns=alpha)


@pytest.mark.parametrize(
    ("name", "value"),
    [("risk_aversion", -1.0), ("turnover_penalty", -1e-9), ("risk_aversion", float("nan"))],
)
def test_negative_or_non_finite_coefficients_are_refused(name: str, value: float) -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    with pytest.raises(OptimizerInputError, match=name):
        _optimize(universe, **{name: value})


def test_shape_disagreements_are_refused() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    with pytest.raises(OptimizerInputError, match="sectors has"):
        _optimize(universe, sectors=universe.sectors[:-1])
    with pytest.raises(OptimizerInputError, match="betas has length"):
        _optimize(universe, betas=universe.betas[:-1])


def test_asset_labels_that_disagree_with_the_covariance_are_refused() -> None:
    """Two orderings in play silently permutes the risk model against the alphas."""
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    relabelled = tuple(f"X{index}" for index in range(len(universe.assets)))
    with pytest.raises(OptimizerInputError, match="disagree"):
        _optimize(universe, assets=relabelled)


def test_an_unlabelled_sector_is_refused_rather_than_defaulted() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    sectors = ("  ", *universe.sectors[1:])
    with pytest.raises(OptimizerInputError, match="non-empty sector label"):
        _optimize(universe, sectors=sectors)


def test_an_unavailable_solver_is_a_configuration_error() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    with pytest.raises(OptimizerInputError, match="not installed"):
        _optimize(universe, solvers=("NO_SUCH_SOLVER",))
    with pytest.raises(OptimizerInputError, match="at least one solver"):
        _optimize(universe, solvers=())


# --------------------------------------------------------------------------
# Solver policy
# --------------------------------------------------------------------------


def test_only_optimal_counts_as_solved() -> None:
    """optimal_inaccurate means the residuals are bounded by the solver's tolerance."""
    assert set(_SOLVED_STATUSES) == {"optimal"}
    assert "optimal_inaccurate" in _RETRYABLE_STATUSES
    assert "optimal_inaccurate" not in _SOLVED_STATUSES
    # An uncertified infeasibility is a solver failing to decide, not a proof,
    # so it must not advance the ladder.
    assert set(_INFEASIBLE_STATUSES) == {"infeasible"}
    assert "infeasible_inaccurate" in _RETRYABLE_STATUSES


def test_the_fallback_solver_is_configured_tighter_than_the_verification_tolerance() -> None:
    """A fallback whose tolerance is looser than this module's cannot ever rescue anything.

    SCS at its shipped defaults returns ``optimal`` on this problem with
    residuals around 1e-5, which the post-solve re-check would reject — turning
    a solver failure into a constraint-violation error that blames problem
    construction. The options exist so the fallback is real.
    """
    assert "SCS" in DEFAULT_SOLVERS
    assert _SOLVER_OPTIONS["SCS"]["eps_abs"] < CONSTRAINT_TOLERANCE
    assert _SOLVER_OPTIONS["SCS"]["eps_rel"] < CONSTRAINT_TOLERANCE


def test_the_fallback_solver_produces_a_portfolio_that_passes_verification() -> None:
    """Not a claim about the options — the whole optimization, run on SCS alone."""
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe, solvers=("SCS",))
    assert result.solver == "SCS"
    assert result.status == "optimal"
    _assert_constraints_hold(result, universe, DEFAULT_CONSTRAINTS)


def test_solver_choice_does_not_change_the_portfolio_materially() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    reference = _optimize(universe, solvers=("CLARABEL",))
    fallback = _optimize(universe, solvers=("SCS",))
    np.testing.assert_allclose(
        np.asarray(reference.weights), np.asarray(fallback.weights), atol=1e-4
    )


# --------------------------------------------------------------------------
# Neutrality references, and the dollar-neutral configuration
# --------------------------------------------------------------------------


def test_default_targets_are_the_equal_weighted_universe() -> None:
    """Sector neutrality means no active sector bet, against a stated reference."""
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe)
    for label, target in result.sector_targets.items():
        count = sum(1 for sector in universe.sectors if sector == label)
        assert target == pytest.approx(count / 100.0)
    assert result.target_beta == pytest.approx(float(universe.betas.mean()))


def test_a_supplied_benchmark_reference_is_used_instead() -> None:
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    labels = sorted(set(universe.sectors))
    # Four sectors at 15% and four at 10%: a real tilt away from the
    # equal-weighted default, and still inside the 20% sector cap.
    tilted = {label: (0.15 if index < 4 else 0.10) for index, label in enumerate(labels)}
    tilted_beta = float(universe.betas.mean()) + 0.03

    result = _optimize(universe, sector_targets=tilted, target_beta=tilted_beta)

    assert result.is_relaxed is False
    assert result.target_beta == tilted_beta
    assert result.sector_targets == tilted
    for label in labels:
        assert result.sector_net_exposures[label] == pytest.approx(
            tilted[label], abs=CONSTRAINT_TOLERANCE
        )
    assert result.portfolio_beta == pytest.approx(tilted_beta, abs=CONSTRAINT_TOLERANCE)


def test_at_zero_net_exposure_full_investment_stops_pinning_the_scale() -> None:
    """A documented limitation, reported rather than papered over.

    ``sum(|w|) == gross`` is not a convex constraint, so at zero net exposure
    the deployed capital is decided by the objective. The result says so, and
    reports the gross exposure that resulted.
    """
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    constraints = PortfolioConstraints(net_exposure=0.0)
    labels = sorted(set(universe.sectors))
    result = _optimize(
        universe,
        constraints=constraints,
        sector_targets=dict.fromkeys(labels, 0.0),
        target_beta=0.0,
        risk_aversion=1.0,
    )

    assert constraints.full_investment_is_binding is False
    assert result.net_exposure == pytest.approx(0.0, abs=CONSTRAINT_TOLERANCE)
    assert result.gross_exposure <= 1.0 + CONSTRAINT_TOLERANCE
    assert result.portfolio_beta == pytest.approx(0.0, abs=CONSTRAINT_TOLERANCE)
    _assert_constraints_hold(result, universe, constraints)


def test_the_full_investment_rung_does_not_apply_to_a_dollar_neutral_book() -> None:
    """There is no net exposure left to reduce, and the attempt log says exactly that."""
    universe = _universe(n_assets=30, n_sectors=6, seed=7)
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(
            universe,
            constraints=PortfolioConstraints(net_exposure=0.0, position_cap=1e-4),
            sector_targets={"SECTOR_0": 0.0},
            # A beta target no 30-name book capped at 1 bp per name can reach:
            # the conflict is real, and it is the *full-investment* rung that
            # cannot help, which is what this test is about.
            target_beta=2.0,
            sectors=("SECTOR_0",) * 30,
        )
    final = caught.value.attempts[-1]
    assert final.status.startswith("not_applicable")
    assert "already 0" in final.status


def test_an_unreachable_beta_target_is_diagnosed_on_a_dollar_neutral_book() -> None:
    """The configuration with no long-only fallback still gets a diagnosis.

    A beta target is one of the likelier ways a dollar-neutral book becomes
    infeasible, and "no single constraint is arithmetically impossible" would be
    the wrong answer when one demonstrably is.
    """
    universe = _universe(n_assets=30, n_sectors=6, seed=7)
    with pytest.raises(InfeasibleProblemError) as caught:
        _optimize(
            universe,
            constraints=PortfolioConstraints(net_exposure=0.0, position_cap=1e-4),
            sector_targets={"SECTOR_0": 0.0},
            target_beta=2.0,
            sectors=("SECTOR_0",) * 30,
            relaxation_ladder=None,
        )
    diagnosis = " ".join(caught.value.diagnosis)
    assert "beta target" in diagnosis
    assert "2.0000" in diagnosis


# --------------------------------------------------------------------------
# What the operator and the artifact see
# --------------------------------------------------------------------------


def test_summary_is_json_serializable_and_states_its_units() -> None:
    """P9.5 and any stored run record carry this without re-running the optimizer."""
    result = _optimize(
        _universe(n_assets=100, n_sectors=8, seed=1), risk_aversion=8.0, turnover_penalty=0.002
    )
    summary = result.summary()
    round_tripped = json.loads(json.dumps(summary))

    assert round_tripped["status"] == "optimal"
    assert round_tripped["n_assets"] == 100
    assert "fractions of capital" in round_tripped["units"]
    assert "not halved" in round_tripped["turnover_convention"]
    # I4 belongs to backend.costs; the summary says so rather than implying the
    # shaping penalty is a cost estimate.
    assert "not a transaction cost" in round_tripped["cost_note"]
    assert round_tripped["max_absolute_weight"] <= DEFAULT_CONSTRAINTS.position_cap + 1e-6


def test_attempt_descriptions_read_as_an_audit_trail() -> None:
    universe = _universe(n_assets=80, n_sectors=4, seed=6)
    result = _optimize(universe)
    described = [attempt.describe() for attempt in result.attempts]

    assert described[0].startswith("rung 0 (relaxed: none)")
    assert "infeasible" in described[0]
    assert described[-1].endswith("[CLARABEL]")
    assert "sector_cap" in described[-1]


def test_the_result_records_the_coefficients_it_used() -> None:
    """I2: a result has to be regenerable, so it carries what produced it."""
    universe = _universe(n_assets=100, n_sectors=8, seed=1)
    result = _optimize(universe, risk_aversion=7.5, turnover_penalty=0.0031)
    assert result.risk_aversion == 7.5
    assert result.turnover_penalty == 0.0031
    assert result.n_assets == 100
    assert result.assets == universe.assets
    assert result.sectors == universe.sectors
