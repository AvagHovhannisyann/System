"""Property-based tests for the optimizer's constraints and refusals (P9.2, P9.4).

Directive §8 requires Hypothesis property suites on "optimizer constraints", and
the reason is specific to this component: a constraint violation does not raise,
it returns. A portfolio 3% into a single name looks exactly like a portfolio 2%
into it until someone reads the weights, and by then it is a backtest result.
So the properties asserted here are the ones a downstream consumer relies on and
cannot check for itself.

Universes are generated *structurally* — asset count, sector count, sector
balance, beta dispersion, risk aversion, turnover penalty — rather than
element-by-element, because the properties are about the geometry of the
constraint set rather than about particular floating-point values. Two families
are drawn:

- **feasible by construction**: at least 50 names (so a 2% cap can fill the
  book) with at least six balanced sectors (so no sector's equal-weight target
  reaches the 20% cap). The equal-weighted portfolio satisfies every constraint
  in these, so the optimizer must return an *unrelaxed* solution — and that
  solution must satisfy every constraint when re-checked in NumPy.
- **arbitrary**: no feasibility guarantee at all. Here the property is the one
  that matters most for this component — whatever comes back, it is never a
  quietly relaxed portfolio. Either it is the requested constraint set, or it
  says what it loosened, or it refuses.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pytest
from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.atoms.norm1 import norm1
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from backend.portfolio.constraints import (
    DEFAULT_CONSTRAINTS,
    RELAXATION_LADDER,
    PortfolioConstraints,
)
from backend.portfolio.covariance import ShrinkageCovariance, ledoit_wolf_covariance
from backend.portfolio.optimizer import (
    BINDING_TOLERANCE,
    CONSTRAINT_TOLERANCE,
    MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION,
    _reachable_beta_range,
    _Spec,
    optimize_portfolio,
)
from backend.portfolio.optimizer_errors import InfeasibleProblemError, OptimizerInputError

if TYPE_CHECKING:
    import numpy.typing as npt

    from backend.portfolio.optimizer import OptimizationResult

_DAILY_VOL = 0.02
_FACTOR_VOL = 0.01
_SOLVER_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@functools.cache
def _covariance(n_assets: int, seed: int) -> ShrinkageCovariance:
    """Estimate a risk model from a seeded factor panel, through the real estimator.

    Cached: the shrinkage estimate is the expensive part of an example and its
    *structure* is what these properties depend on, so a handful of distinct
    panels is explored while the cheap parts (alphas, betas, sector layout,
    coefficients) vary freely.
    """
    rng = np.random.default_rng(seed)
    observations = MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION
    factors = rng.normal(0.0, _FACTOR_VOL, size=(observations, 3))
    loadings = rng.normal(0.0, 1.0, size=(3, n_assets))
    returns = factors @ loadings + rng.normal(0.0, _DAILY_VOL, size=(observations, n_assets))
    return ledoit_wolf_covariance(returns)


def _sector_matrix(sectors: tuple[str, ...]) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    labels = tuple(sorted(set(sectors)))
    indicator = np.zeros((len(labels), len(sectors)), dtype=np.float64)
    for column, label in enumerate(sectors):
        indicator[labels.index(label), column] = 1.0
    return labels, indicator


@st.composite
def _problems(
    draw: st.DrawFn, *, feasible: bool
) -> tuple[ShrinkageCovariance, tuple[str, ...], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Draw a universe: risk model, sector labels, betas, expected returns.

    With ``feasible=True`` the draw is restricted to shapes in which the
    equal-weighted portfolio satisfies every shipped constraint — at least 50
    names (``1/n <= 2%``) spread round-robin over at least six sectors (so every
    sector's ``n_s / n`` target stays under both the 20% sector cap and the
    ``n_s * 2%`` its own names can reach).
    """
    if feasible:
        n_assets = draw(st.integers(min_value=55, max_value=90))
        n_sectors = draw(st.integers(min_value=6, max_value=11))
    else:
        n_assets = draw(st.integers(min_value=20, max_value=90))
        n_sectors = draw(st.integers(min_value=3, max_value=11))

    covariance = _covariance(n_assets, draw(st.integers(min_value=0, max_value=5)))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10_000)))
    beta_dispersion = draw(st.floats(min_value=0.05, max_value=0.6))
    alpha_scale = draw(st.floats(min_value=1e-4, max_value=5e-2))
    return (
        covariance,
        tuple(f"SECTOR_{index % n_sectors}" for index in range(n_assets)),
        rng.normal(1.0, beta_dispersion, size=n_assets),
        rng.normal(0.0, alpha_scale, size=n_assets),
    )


def _check_constraints(
    result: OptimizationResult,
    sectors: tuple[str, ...],
    betas: npt.NDArray[np.float64],
    constraints: PortfolioConstraints,
) -> None:
    """Re-check every constraint in NumPy against the array a backtest would get."""
    weights = np.asarray(result.weights, dtype=np.float64)
    labels, indicator = _sector_matrix(sectors)

    assert np.isfinite(weights).all()
    assert float(np.max(np.abs(weights))) <= constraints.position_cap + CONSTRAINT_TOLERANCE
    assert abs(float(weights.sum()) - constraints.net_exposure) <= CONSTRAINT_TOLERANCE
    assert float(np.abs(weights).sum()) <= constraints.gross_exposure + CONSTRAINT_TOLERANCE
    assert (
        float(np.max(indicator @ np.abs(weights))) <= constraints.sector_cap + CONSTRAINT_TOLERANCE
    )

    targets = np.asarray([result.sector_targets[label] for label in labels], dtype=np.float64)
    assert (
        float(np.max(np.abs(indicator @ weights - targets)))
        <= constraints.sector_neutrality_band + CONSTRAINT_TOLERANCE
    )
    assert (
        abs(float(betas @ weights) - result.target_beta)
        <= constraints.beta_neutrality_band + CONSTRAINT_TOLERANCE
    )


@given(problem=_problems(feasible=True), risk_aversion=st.floats(min_value=0.0, max_value=50.0))
@_SOLVER_SETTINGS
def test_every_constraint_holds_at_the_solution(
    problem: tuple[
        ShrinkageCovariance, tuple[str, ...], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ],
    risk_aversion: float,
) -> None:
    """Position cap, sector cap, sector neutrality, beta neutrality, full investment.

    The universe is feasible by construction, so a relaxation here would not be
    a rescue — it would be the optimizer conceding a constraint it did not have
    to. Both facts are asserted.
    """
    covariance, sectors, betas, alpha = problem
    result = optimize_portfolio(
        expected_returns=alpha,
        covariance=covariance,
        sectors=sectors,
        betas=betas,
        risk_aversion=risk_aversion,
        turnover_penalty=0.0,
    )

    assert result.status == "optimal"
    assert result.rung == 0
    assert result.relaxations == ()
    assert result.is_relaxed is False
    assert result.full_investment_satisfied is True
    _check_constraints(result, sectors, betas, DEFAULT_CONSTRAINTS)


@given(
    problem=_problems(feasible=True),
    penalties=st.tuples(
        st.floats(min_value=0.0, max_value=0.005),
        st.floats(min_value=0.01, max_value=1.0),
    ),
)
@_SOLVER_SETTINGS
def test_a_higher_turnover_penalty_never_trades_more(
    problem: tuple[
        ShrinkageCovariance, tuple[str, ...], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ],
    penalties: tuple[float, float],
) -> None:
    """G9 in property form, over random universes and random previous books.

    Weak monotonicity is the true statement — convexity makes the optimum
    locally flat in the coefficient — so that is what is asserted, on the
    turnover the solution *realizes* rather than on the parameter being
    accepted.
    """
    covariance, sectors, betas, alpha = problem
    low, high = penalties
    previous = np.full(len(sectors), 1.0 / len(sectors), dtype=np.float64)

    results = [
        optimize_portfolio(
            expected_returns=alpha,
            covariance=covariance,
            sectors=sectors,
            betas=betas,
            risk_aversion=5.0,
            turnover_penalty=penalty,
            previous_weights=previous,
        )
        for penalty in (low, high)
    ]
    assert results[1].turnover <= results[0].turnover + CONSTRAINT_TOLERANCE


@given(problem=_problems(feasible=False))
@_SOLVER_SETTINGS
def test_a_relaxed_portfolio_is_never_returned_silently(
    problem: tuple[
        ShrinkageCovariance, tuple[str, ...], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ],
) -> None:
    """The property this whole component exists to guarantee.

    Over universes with no feasibility guarantee, exactly three outcomes are
    permitted: the requested portfolio; a portfolio that says which constraints
    were loosened, by how much, and where it landed; or a refusal. There is no
    fourth branch in which a portfolio the stated policy forbids reaches a
    backtest looking like one it allows.
    """
    covariance, sectors, betas, alpha = problem
    result: OptimizationResult | None = None
    refusal: InfeasibleProblemError | None = None
    try:
        result = optimize_portfolio(
            expected_returns=alpha,
            covariance=covariance,
            sectors=sectors,
            betas=betas,
            risk_aversion=5.0,
            turnover_penalty=0.001,
        )
    except InfeasibleProblemError as error:
        refusal = error

    if refusal is not None:
        # Outcome three: a refusal that shows every rung it tried, in order.
        attempts = list(refusal.attempts)
        assert attempts
        assert [attempt.rung for attempt in attempts] == list(range(len(attempts)))
        assert "No weights are returned" in str(refusal)
        return

    assert result is not None
    # Whatever rung it came from, the solution obeys the set it was solved under.
    _check_constraints(result, sectors, betas, result.constraints)
    # The unrelaxable limits are unrelaxed, at every rung, always.
    assert result.constraints.position_cap == DEFAULT_CONSTRAINTS.position_cap
    assert result.constraints.gross_exposure == DEFAULT_CONSTRAINTS.gross_exposure

    if result.is_relaxed:
        assert result.rung > 0
        assert (
            tuple(item.constraint for item in result.relaxations)
            == RELAXATION_LADDER[: result.rung]
        )
        assert result.constraints != result.requested_constraints
        assert all(item.realized is not None for item in result.relaxations)
        assert result.summary()["is_relaxed"] is True
    else:
        assert result.rung == 0
        assert result.constraints == result.requested_constraints
        _check_constraints(result, sectors, betas, result.requested_constraints)


@given(problem=_problems(feasible=True), risk_aversion=st.floats(min_value=0.0, max_value=30.0))
@_SOLVER_SETTINGS
def test_binding_indicators_describe_the_portfolio_returned(
    problem: tuple[
        ShrinkageCovariance, tuple[str, ...], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ],
    risk_aversion: float,
) -> None:
    """Dashboard §6.8 shows these next to the weights; they have to be the same portfolio."""
    covariance, sectors, betas, alpha = problem
    result = optimize_portfolio(
        expected_returns=alpha,
        covariance=covariance,
        sectors=sectors,
        betas=betas,
        risk_aversion=risk_aversion,
        turnover_penalty=0.0,
    )
    weights = np.asarray(result.weights)
    labels, indicator = _sector_matrix(sectors)
    cap = result.constraints.position_cap

    expected_positions = {
        f"#{index}"
        for index in range(weights.size)
        if abs(float(weights[index])) >= cap - BINDING_TOLERANCE
    }
    assert set(result.binding_position_caps) == expected_positions

    sector_gross = indicator @ np.abs(weights)
    expected_sectors = {
        labels[index]
        for index in range(len(labels))
        if float(sector_gross[index]) >= result.constraints.sector_cap - BINDING_TOLERANCE
    }
    assert set(result.binding_sector_caps) == expected_sectors

    gross = float(np.abs(weights).sum())
    assert result.binding_gross_exposure == (
        gross >= result.constraints.gross_exposure - BINDING_TOLERANCE
    )
    # Every name reported as binding really is at the cap, in the array itself.
    for index in range(weights.size):
        at_cap = abs(float(weights[index])) >= cap - BINDING_TOLERANCE
        assert (f"#{index}" in result.binding_position_caps) == at_cap


# --------------------------------------------------------------------------
# The infeasibility diagnosis must never claim more than it can prove
# --------------------------------------------------------------------------


def _minimal_spec(betas: npt.NDArray[np.float64], constraints: PortfolioConstraints) -> _Spec:
    """Build the smallest ``_Spec`` the beta-range bound reads: betas and limits."""
    n_assets = betas.size
    return _Spec(
        alpha=np.zeros(n_assets),
        risk_factor=np.zeros((n_assets, n_assets)),
        covariance=np.zeros((n_assets, n_assets)),
        betas=betas,
        previous_weights=np.zeros(n_assets),
        sector_indicator=np.ones((1, n_assets)),
        sector_labels=("ONLY",),
        sector_targets=np.array([constraints.net_exposure]),
        target_beta=0.0,
        constraints=constraints,
        risk_aversion=1.0,
        turnover_penalty=0.0,
    )


def _true_beta_range(
    betas: npt.NDArray[np.float64], constraints: PortfolioConstraints
) -> tuple[float, float] | None:
    """Compute the exactly reachable portfolio-beta range with an LP."""
    weights = Variable(betas.size)
    limits = [
        cvxpy_sum(weights) == constraints.net_exposure,
        weights <= constraints.position_cap,
        weights >= -constraints.position_cap,
        norm1(weights) <= constraints.gross_exposure,
    ]
    lowest = Problem(Minimize(matmul(betas, weights)), limits)
    highest = Problem(Maximize(matmul(betas, weights)), limits)
    lowest.solve(solver="CLARABEL")  # type: ignore[no-untyped-call]
    highest.solve(solver="CLARABEL")  # type: ignore[no-untyped-call]
    if lowest.status != "optimal" or highest.status != "optimal":
        return None
    return float(lowest.value), float(highest.value)


@given(
    n_assets=st.integers(min_value=5, max_value=40),
    seed=st.integers(min_value=0, max_value=10_000),
    dispersion=st.floats(min_value=0.05, max_value=1.5),
    net_exposure=st.floats(min_value=0.0, max_value=1.0),
    position_cap=st.floats(min_value=0.02, max_value=0.5),
)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_the_diagnosed_beta_range_is_a_superset_of_the_reachable_one(
    n_assets: int, seed: int, dispersion: float, net_exposure: float, position_cap: float
) -> None:
    """A *necessary* condition may be loose; it may never be wrong.

    The infeasibility diagnosis uses this range to say "the beta target is
    unreachable". If the range were ever narrower than the truth, that sentence
    would be a fabricated proof — it would name beta neutrality as the culprit
    for a conflict living somewhere else, and send an operator to fix the wrong
    number. The first draft of this bound assumed a long-only book and was
    narrower than the truth for any book with ``gross > net``.
    """
    rng = np.random.default_rng(seed)
    betas = rng.normal(1.0, dispersion, size=n_assets)
    try:
        constraints = PortfolioConstraints(
            position_cap=position_cap,
            sector_cap=max(position_cap, 1.0),
            net_exposure=net_exposure,
        )
    except OptimizerInputError:
        return

    claimed = _reachable_beta_range(_minimal_spec(betas, constraints))
    truth = _true_beta_range(betas, constraints)
    if claimed is None or truth is None:
        return

    low, high = claimed
    true_low, true_high = truth
    assert low <= true_low + 1e-6, (claimed, truth)
    assert high >= true_high - 1e-6, (claimed, truth)


@given(
    n_assets=st.integers(min_value=5, max_value=40),
    seed=st.integers(min_value=0, max_value=10_000),
    dispersion=st.floats(min_value=0.05, max_value=1.5),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_the_beta_range_is_sharp_for_the_long_only_default(
    n_assets: int, seed: int, dispersion: float
) -> None:
    """Where the arithmetic permits sharpness, the bound is sharp.

    At ``net == gross`` the book is long-only and the greedy fill is exactly the
    reachable extreme, so the looser centred bound used for long-short books
    must not be applied there — a diagnosis that is valid but never tight would
    stop naming the beta target even when the beta target really is the problem.
    """
    rng = np.random.default_rng(seed)
    betas = rng.normal(1.0, dispersion, size=n_assets)
    constraints = PortfolioConstraints(position_cap=1.0 / n_assets + 0.01, sector_cap=1.0)
    if constraints.position_cap * n_assets < constraints.net_exposure:
        return

    claimed = _reachable_beta_range(_minimal_spec(betas, constraints))
    truth = _true_beta_range(betas, constraints)
    assert claimed is not None
    assert truth is not None
    assert claimed[0] == pytest.approx(truth[0], abs=1e-6)
    assert claimed[1] == pytest.approx(truth[1], abs=1e-6)
