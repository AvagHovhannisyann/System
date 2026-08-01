"""Convex portfolio optimizer with an explicit turnover penalty (P9.2).

Directive §5 Phase 9 specifies the problem: *maximize expected return minus a
risk penalty minus an explicit turnover penalty, subject to sector neutrality,
beta neutrality, a 2% position cap, a 20% sector cap, and a full-investment
constraint.* This module is that, in cvxpy, plus the two things a specification
does not say and an operator cannot do without: what happens when the
constraints cannot all hold at once, and how the caller finds out.

Objective
---------

::

    maximize   alpha' w  -  risk_aversion * w' Sigma w  -  turnover_penalty * ||w - w_prev||_1

All three terms are concave in ``w`` (the risk term is a positive multiple of a
positive semi-definite quadratic form, negated; the turnover term is a positive
multiple of a norm, negated), so the problem is convex and the solver's
``optimal`` status means a global optimum.

**Units, stated because directive §8 requires it and because this is where
factor-of-10,000 errors live.**

- ``alpha`` (``expected_returns``): simple **period return fractions**.
  ``0.01`` is a 1% expected return over the covariance's period. Not percent,
  not basis points.
- ``Sigma`` (``covariance``): squared period return fractions, i.e. exactly what
  :func:`~backend.portfolio.covariance.ledoit_wolf_covariance` returns, over
  the same period as ``alpha``. Nothing here annualizes anything.
- ``risk_aversion``: units of ``1 / return fraction`` — it multiplies a variance
  (squared return fraction) to produce something comparable with a return
  fraction. Concretely: a portfolio at 1% period volatility has
  ``w' Sigma w = 1e-4``, so ``risk_aversion = 25`` charges it 25 bp of expected
  period return. The ``1/2`` that some textbooks put in front of the quadratic
  term is **not** applied here; a value tuned against the half convention is
  twice as risk-averse as intended.
- ``turnover_penalty``: return fraction charged per unit of traded notional, so
  it is directly comparable with a round-trip transaction cost. ``0.0020``
  charges 20 bp of capital for turning the entire book over once. See
  :func:`turnover_penalty_from_round_trip_cost_bps`.
- ``w``, ``w_prev``, every cap and every exposure: fractions of portfolio
  capital. ``0.02`` is 2% of the book.
- ``betas``: dimensionless.

**Turnover convention.** ``turnover = sum(|w_i - w_prev_i|)``: total traded
notional as a fraction of capital. Selling 3% of the book and buying 3%
elsewhere is 6% of turnover under this convention, not 3%. The halved
("one-way") convention is equally common in the literature, which is why this
one is written down rather than assumed. ``previous_weights`` defaults to zeros
— an initial rebalance out of cash — so the first optimization is charged for
building the book, which is correct and is worth knowing before reading the
first turnover number a backtest prints.

**The turnover penalty is a shaping term, not a cost accrual.** It belongs to
the objective, where its job is to make the optimizer prefer a portfolio it can
hold. It is *not* the cost of trading, and subtracting it from a return series
would not make that series net of costs. Invariant I4 is satisfied by
:mod:`backend.costs`, which prices actual orders and stamps every estimate with
``uncalibrated`` and a ``calibration_basis``. Setting this coefficient from that
model is sensible (see :func:`turnover_penalty_from_round_trip_cost_bps`), and a
coefficient derived from the shipped, uncalibrated cost defaults inherits their
uncalibrated status — the backtest must say so, from the cost model's own flag.

Constraints, and the arithmetic that binds them together
--------------------------------------------------------

Neutrality is always neutrality *relative to something*. This module makes the
reference explicit rather than assuming one:

- **Sector neutrality**: each sector's net weight equals a target,
  ``sum_{i in s} w_i == sector_target_s``, optionally within a band.
- **Beta neutrality**: the portfolio beta equals a target,
  ``beta' w == target_beta``, optionally within a band.
- **Full investment**: ``sum(w) == net_exposure``, default ``1.0``.
- **Position cap**: ``|w_i| <= 0.02``.
- **Sector cap**: ``sum_{i in s} |w_i| <= 0.20`` — gross, so that a long and a
  short inside one sector cannot net into apparent compliance.
- **No leverage**: ``sum(|w_i|) <= 1.0``, directive §1.1, never relaxed.

Summing the sector-neutrality equalities gives ``sum(w) == sum_s target_s``.
That is the same quantity the full-investment constraint fixes, so **the sector
targets must sum to the net-exposure target** or no portfolio exists. This is
not a subtlety the caller should have to rediscover from an unexplained
``infeasible``: it is checked up front and raises
:class:`~backend.portfolio.optimizer_errors.OptimizerInputError` naming both
numbers.

When targets are not supplied they default to the **equal-weighted universe**
scaled to the net-exposure target: ``sector_target_s = net * n_s / n`` and
``target_beta = net * mean(beta)``. That reference is derived from the caller's
own inputs rather than invented here, it satisfies the summation identity by
construction, and it gives the phrases their usual meaning — "sector neutral"
becomes *no active sector bet*, "beta neutral" becomes *no active beta bet*. A
caller neutralizing against a cap-weighted benchmark passes that benchmark's
sector weights and beta instead.

**What "full investment" can and cannot mean.** At the default ``net_exposure
= 1.0`` it is an equality on ``sum(w)`` and is enforced exactly; combined with
the no-leverage limit ``sum(|w|) <= 1`` it also forces ``w >= 0``, so the
default book is long-only as a *consequence* of directive §1.1 rather than as a
separate switch. A dollar-neutral long-short book is configured by asking for
``net_exposure = 0.0`` with zero sector targets and a zero beta target — and
there full investment stops being enforceable, because at zero net exposure the
constraint that would express it is ``sum(|w|) == gross``, the boundary of a
convex set and therefore not a convex constraint. No cvxpy formulation fixes
that; the honest response is to report the realized gross exposure on every
result and to say, via
:attr:`~backend.portfolio.constraints.PortfolioConstraints.full_investment_is_binding`,
whether the constraint set or the objective's risk aversion decided how much
capital was deployed.

Infeasibility
-------------

Constraints conflict. A borrow filter thins a sector to two names; a 2% cap
over forty names cannot add to a fully invested book; a sector holding a
quarter of the universe cannot be both neutral-weighted and under a 20% cap.
The optimizer answers with :data:`~backend.portfolio.constraints.RELAXATION_LADDER`
— a fixed, documented order of loosenings, tried one rung at a time — and every
result records which rung it came from and what was relaxed
(:attr:`OptimizationResult.relaxations`). When the ladder is exhausted it
raises. It never returns the previous weights, the equal-weighted universe, or
zeros: a portfolio that quietly violates its constraints is indistinguishable
downstream from one that does not, and directive §0.3 names this component as
one where exactly that kind of silence invalidates everything built on it.

Every returned solution is re-checked in NumPy against the constraints, at this
module's tolerance rather than the solver's, and only the ``optimal`` status is
accepted as a solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

# cvxpy's atoms are imported from the modules that define them rather than from
# the top-level ``cvxpy`` namespace. cvxpy ships ``py.typed``, but
# ``cvxpy/atoms/__init__.py`` re-exports through ``import *`` without an
# ``__all__``, which mypy's strict ``no_implicit_reexport`` does not follow — so
# ``cp.sum`` and friends type-check as "module has no attribute". These paths are
# the same objects, and they keep `mypy --strict` (directive §8) honest without a
# blanket ignore over the whole module.
import cvxpy.settings as cvxpy_settings
import numpy as np
from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.atoms.elementwise.abs import abs as cvxpy_abs
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.sum_squares import sum_squares
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import SolverError
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.defines import installed_solvers

from backend.portfolio.constraints import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_RELAXATION_LADDER,
    RELAXATION_LADDER,
    PortfolioConstraints,
    RelaxableConstraint,
    Relaxation,
    RelaxationLadder,
)
from backend.portfolio.covariance import (
    DEFAULT_PSD_RELATIVE_TOLERANCE,
    MINIMUM_OBSERVATIONS_FOR_SHRINKAGE,
    ShrinkageCovariance,
)
from backend.portfolio.errors import NotPositiveSemiDefiniteError
from backend.portfolio.optimizer_errors import (
    ConstraintViolationError,
    InfeasibleProblemError,
    InsufficientRiskHistoryError,
    OptimizerInputError,
    SolverFailureError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

__all__ = [
    "BINDING_TOLERANCE",
    "CONSTRAINT_TOLERANCE",
    "DEFAULT_SOLVERS",
    "MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION",
    "OptimizationResult",
    "SolveAttempt",
    "optimize_portfolio",
    "turnover_penalty_from_round_trip_cost_bps",
]

MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION: Final = 252
"""Observations the risk model must rest on before it may be optimized against (count).

**This is the policy DECISIONS.md D-020 left open, decided here.** The
covariance estimator refuses below three observations because three is where
the Ledoit-Wolf intensity stops being algebraically degenerate; it recorded
explicitly that this is "where the *formula* stops being valid, not where a
risk model becomes trustworthy", and that the real minimum — "60? 252?" —
belongs to the consumer, where it is visible. This module is that consumer.

**252 daily observations, one trading year.** The reasoning, in the order it
matters:

1. *Earnings periodicity.* Every name in the universe reports four times a
   year. A window shorter than a year contains a number of earnings jumps per
   name that depends on where the window happens to fall, so the estimated
   variance of a name becomes a function of window alignment rather than of the
   name. At 60 observations one stock has two reports inside the window and its
   neighbour has none, and the optimizer reads the difference as a risk
   difference. A year gives every name the same four.
2. *Label horizons.* Phase 6 labels run to 63 days. A covariance estimated over
   fewer than four non-overlapping horizons is describing a shorter-lived
   process than the one being traded.
3. *Shrinkage intensity is the honest gauge.* The Ledoit-Wolf intensity reports
   what fraction of the risk model is imposed spherical structure rather than
   estimated covariance. On a realistic cross-section a 60-day window drives it
   high enough that the "risk model" is mostly the assumption that all names
   have equal variance and no correlation — an assumption under which a
   mean-variance optimizer is just an alpha-ranker with caps. The intensity is
   reported on every result (:attr:`OptimizationResult.shrinkage_intensity`) so
   this can be checked rather than trusted.

**Rejected — 60 observations (a quarter).** One quarter is one earnings season
and, in practice, one regime. It is the number that makes a backtest start
earlier, which is the wrong reason to choose it.

**Rejected — requiring more observations than assets (``n > p``).** It is the
condition under which the *sample* covariance is invertible, but shrinkage
exists precisely so that it is not needed, and at 500 names it would force a
two-year window whose oldest data describes a market that no longer exists.

**Rejected — no policy at all, deferring to the estimator's 3.** That is the
option D-020 explicitly refused, because it hands the optimizer a risk model
that is 99% assumption without anyone having to say so.

This is a *default*, not a law: ``minimum_observations=`` overrides it, the
effective value is recorded on every result, and values below
:data:`~backend.portfolio.covariance.MINIMUM_OBSERVATIONS_FOR_SHRINKAGE` are
refused because the estimator itself cannot honour them.
"""

DEFAULT_SOLVERS: Final = ("CLARABEL", "SCS")
"""Solvers tried in order (names as cvxpy spells them).

CLARABEL is an interior-point conic solver, is cvxpy's default for problems of
this shape, and returns a proper infeasibility certificate — which matters,
because "infeasible" is a load-bearing answer here rather than an error. SCS is
a first-order fallback for the rare problem CLARABEL cannot factor; it is
consulted only when the first solver fails to reach a definitive status, never
to obtain a second opinion on an infeasibility CLARABEL has certified.

Measured on a 120-name, 8-sector problem of this exact shape: CLARABEL leaves a
worst constraint residual of 3.5e-10, SCS **at its default tolerances leaves
1.2e-5** — an order of magnitude *outside* :data:`CONSTRAINT_TOLERANCE`. See
:data:`_SOLVER_OPTIONS` for why that means the fallback has to be configured
rather than merely listed.
"""

_SOLVER_OPTIONS: Final = MappingProxyType(
    {
        "SCS": MappingProxyType({"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 100_000}),
    }
)
"""Per-solver options, applied whenever that solver is used.

**A fallback solver whose tolerance is looser than this module's is not a
fallback.** SCS at its shipped tolerances returns ``optimal`` on this problem
with constraint residuals around 1e-5, ten times :data:`CONSTRAINT_TOLERANCE`.
The post-solve re-check would then reject those weights — correctly — and raise
:class:`~backend.portfolio.optimizer_errors.ConstraintViolationError`, whose
message says the solver's own tolerance is not the last word on whether a
position cap held. That would be true but useless: the fallback could never
rescue a problem CLARABEL failed on, it would only convert a solver failure into
a differently-named error, and the error would point at problem construction
rather than at the loose tolerance actually responsible.

Tightening SCS to 1e-9 brings its residual to 1.8e-10 — inside this module's
tolerance, with the same order of magnitude of runtime (0.09s against 0.07s on
the 120-name problem measured above). The iteration limit is raised to match, so
the tighter tolerance is pursued rather than abandoned at ``user_limit``.

CLARABEL is deliberately absent: it already solves to 3.5e-10 at its defaults,
and pinning a third-party solver's parameters that do not need pinning is how a
future version's better defaults get silently overridden.
"""

CONSTRAINT_TOLERANCE: Final = 1e-6
"""Tolerance for the post-solve constraint re-check, in fractions of capital.

1e-6 of capital is 0.0001% of the book: four orders of magnitude below the 2%
position cap, and far above the ~1e-9 residuals an interior-point solver leaves
on a well-scaled problem of this size. Tight enough that a real violation
cannot hide inside it, loose enough that a correct solve is never rejected for
arithmetic noise.
"""

BINDING_TOLERANCE: Final = 1e-6
"""How close to a limit a position must sit to be reported as binding it.

Same units and same magnitude as :data:`CONSTRAINT_TOLERANCE`. Feeds
:attr:`OptimizationResult.binding_position_caps` and
:attr:`OptimizationResult.binding_sector_caps`, which are what dashboard §6.8's
"constraint-binding indicators" display.
"""

_NET_EXPOSURE_BACKOFF_RELATIVE: Final = 1e-9
"""Relative step back from the maximum deployable net exposure (dimensionless)."""

_NET_EXPOSURE_BACKOFF_ABSOLUTE: Final = 1e-7
"""Absolute step back from the maximum deployable net exposure (fraction of capital).

The full-investment rung finds the largest net exposure the remaining
constraints admit by solving an LP, then asks the QP for exactly that. Sitting
on the exact boundary of a feasible region is where interior-point solvers
report ``infeasible`` or ``optimal_inaccurate``, so the target is stepped back
before the QP is asked for it. The step is
``max(relative * deployable, absolute)``.

**The absolute term is not decoration.** A purely relative backoff was measured
against the case it exists for — 30 names at a 2% cap, whose true maximum net
exposure is exactly 0.60 — and found to be an order of magnitude too small: the
LP returns ``0.6000000050`` (about +5e-9 of solver residual *above* the true
maximum) while a 1e-9 relative step back removes only 6e-10 of it. The QP was
therefore being handed a target the constraints cannot actually reach, and
survived only because the residual happened to land inside
:data:`CONSTRAINT_TOLERANCE` — precisely the "the cap held to within the
solver's convenience" arrangement this module refuses everywhere else. A
slightly harder problem returns ``infeasible`` instead, and the optimizer would
refuse a portfolio that exists.

1e-7 of capital is ten cents on a million-dollar book: five orders of magnitude
below the position cap and one below the tolerance every constraint is verified
at, so it cannot be a portfolio decision, and it is twenty times the residual
measured above. If a solver were ever loose enough to overshoot by more than
this, the rung's QP reports ``infeasible`` and the optimizer refuses — the safe
direction, and a refusal that says which rung it happened on.
"""

_SOLVED_STATUSES: Final = frozenset({cvxpy_settings.OPTIMAL})
"""The only statuses accepted as a solution. Deliberately a set of one."""

_INFEASIBLE_STATUSES: Final = frozenset({cvxpy_settings.INFEASIBLE})
"""Statuses that certify no portfolio exists, and so advance the ladder.

``infeasible_inaccurate`` is deliberately absent: an uncertified infeasibility
is a solver failing to decide, not a proof that the constraint set is empty, so
it falls through to the next solver instead of skipping a rung.
"""

_RETRYABLE_STATUSES: Final = frozenset(
    {
        cvxpy_settings.OPTIMAL_INACCURATE,
        cvxpy_settings.INFEASIBLE_INACCURATE,
        cvxpy_settings.UNBOUNDED,
        cvxpy_settings.UNBOUNDED_INACCURATE,
        cvxpy_settings.SOLVER_ERROR,
        cvxpy_settings.USER_LIMIT,
        cvxpy_settings.INFEASIBLE_OR_UNBOUNDED,
    }
)
"""Statuses that mean "ask the next solver", and failing that, raise."""


@dataclass(frozen=True, slots=True)
class SolveAttempt:
    """One rung of the relaxation ladder, and what the solver said about it.

    The full sequence of these is carried on every result and on every
    infeasibility error, which is what lets a test — and an operator — assert
    that the ladder was climbed in the documented order rather than trusting
    that it was.

    Attributes:
        rung: index into :data:`~backend.portfolio.constraints.RELAXATION_LADDER`
            plus one; ``0`` is the constraint set exactly as requested.
        relaxations: the constraints loosened at this rung, in ladder order.
            Empty at rung 0.
        status: the cvxpy status string, or — when the rung could not be formed
            at all — a sentence saying why, beginning ``"not_applicable"`` or
            ``"refused"``. The two are different facts and are kept apart on
            purpose: a rung that does not apply (the full-investment rung
            against a book whose net-exposure target is already zero) says
            nothing about the universe, while a rung *refused* because the most
            capital the constraints admit falls below the ladder's floor is the
            single most useful sentence in an infeasibility report — it names
            the number the operator has to change.
        solver: the solver that produced ``status``, or ``None``.
        constraints: the constraint set used at this rung.
    """

    rung: int
    relaxations: tuple[RelaxableConstraint, ...]
    status: str
    solver: str | None
    constraints: PortfolioConstraints

    def describe(self) -> str:
        """Return a one-line operator-readable description of the attempt.

        Returns:
            A sentence naming the rung, what it relaxed, and how it ended.
        """
        relaxed = ", ".join(item.value for item in self.relaxations) or "none"
        solver = self.solver or "n/a"
        return f"rung {self.rung} (relaxed: {relaxed}) -> {self.status} [{solver}]"


@dataclass(frozen=True, slots=True, eq=False)
class OptimizationResult:
    """A solved portfolio, its exposures, and the truth about how it was obtained.

    The weights array is read-only, so a consumer cannot mutate the portfolio in
    place and leave the recorded exposures and relaxations describing something
    else — the same discipline
    :class:`~backend.portfolio.covariance.ShrinkageCovariance` applies to the
    risk model.

    **Reading this object honestly.** :attr:`relaxations` empty means the
    portfolio requested is the portfolio returned. Anything else means the
    constraint set was loosened, and :meth:`Relaxation.describe` on each entry
    says which one, from what to what, in what units, and whether the solution
    actually consumed the loosening.

    Attributes:
        weights: optimal weights as fractions of capital, in the order the
            assets were supplied. Read-only.
        assets: asset identifiers in the same order, or ``None`` if the caller
            supplied none and the covariance carried none.
        sectors: each asset's sector label, in the same order.
        status: the cvxpy status of the accepted solve. Always ``"optimal"`` —
            no other status is treated as a solution.
        solver: the solver that produced it.
        objective_value: the optimized objective, in period return fractions.
        expected_return: ``alpha' w`` (period return fraction).
        variance: ``w' Sigma w`` (squared period return fraction).
        volatility: ``sqrt(variance)`` (period return fraction), reported
            because it is the number an operator can compare against
            intuition — a variance of 1e-4 means nothing at a glance and 1%
            period volatility means everything.
        risk_penalty: ``risk_aversion * variance`` (period return fraction).
        turnover: ``sum(|w - w_prev|)``, total traded notional as a fraction of
            capital. See the module docstring on the halved convention.
        turnover_cost: ``turnover_penalty * turnover`` (period return
            fraction) — the objective's charge for the trade, **not** a
            transaction-cost estimate. Costs come from :mod:`backend.costs`.
        net_exposure: ``sum(w)`` (fraction of capital).
        gross_exposure: ``sum(|w|)`` (fraction of capital).
        portfolio_beta: ``beta' w`` (dimensionless).
        sector_net_exposures: per-sector ``sum(w_i)`` (fraction of capital).
        sector_gross_exposures: per-sector ``sum(|w_i|)`` (fraction of
            capital).
        sector_targets: the per-sector net-weight targets actually applied
            (fraction of capital) — scaled ones if the full-investment rung
            reduced the book.
        target_beta: the portfolio-beta target actually applied.
        binding_position_caps: assets sitting on the position cap, by label or
            by ``"#index"`` when unlabelled.
        binding_sector_caps: sectors sitting on the sector cap.
        binding_gross_exposure: whether ``sum(|w|)`` sits on the no-leverage
            limit. Reported alongside the other two because it is the third
            *inequality* in the set, and at the shipped defaults it is the one
            that explains the portfolio's shape: ``sum(w) == 1`` against
            ``sum(|w|) <= 1`` binds the gross limit and forces the book
            long-only. The equalities — full investment, sector neutrality,
            beta neutrality — are tight by construction and so carry no
            information as indicators; :attr:`relaxations` is what says whether
            they held at the values that were *asked* for.
        constraints: the constraint set actually used.
        requested_constraints: the constraint set originally asked for.
            Identical to :attr:`constraints` when nothing was relaxed.
        relaxations: what was loosened to get here, in ladder order. Empty
            means nothing was.
        rung: which rung of the ladder produced this result; ``0`` is the
            requested constraint set.
        attempts: every rung tried, in order, including the ones that failed.
        n_observations: observations behind the risk model (count).
        n_assets: assets in the universe (count).
        minimum_observations: the observation policy applied (count).
        shrinkage_intensity: the risk model's Ledoit-Wolf intensity
            (dimensionless, ``[0, 1]``) — how much of it was assumption.
        risk_aversion: the coefficient used (``1 / return fraction``).
        turnover_penalty: the coefficient used (return fraction per unit of
            traded notional).
    """

    weights: npt.NDArray[np.float64]
    assets: tuple[str, ...] | None
    sectors: tuple[str, ...]
    status: str
    solver: str
    objective_value: float
    expected_return: float
    variance: float
    volatility: float
    risk_penalty: float
    turnover: float
    turnover_cost: float
    net_exposure: float
    gross_exposure: float
    portfolio_beta: float
    sector_net_exposures: Mapping[str, float]
    sector_gross_exposures: Mapping[str, float]
    sector_targets: Mapping[str, float]
    target_beta: float
    binding_position_caps: tuple[str, ...]
    binding_sector_caps: tuple[str, ...]
    binding_gross_exposure: bool
    constraints: PortfolioConstraints
    requested_constraints: PortfolioConstraints
    relaxations: tuple[Relaxation, ...]
    rung: int
    attempts: tuple[SolveAttempt, ...]
    n_observations: int
    n_assets: int
    minimum_observations: int
    shrinkage_intensity: float
    risk_aversion: float
    turnover_penalty: float

    @property
    def is_relaxed(self) -> bool:
        """Whether this is the requested portfolio or a relaxed one.

        Returns:
            ``True`` if any constraint was loosened to obtain this result. A
            caller that ignores this is treating "I could not do what you
            asked, so here is something near it" as "here is what you asked
            for".
        """
        return bool(self.relaxations)

    @property
    def full_investment_satisfied(self) -> bool:
        """Whether the book is deployed to its net-exposure target.

        Returns:
            ``True`` if the realized net exposure matches the *requested*
            net-exposure target within :data:`CONSTRAINT_TOLERANCE`. ``False``
            means the full-investment rung reduced the book — the amount is on
            the corresponding :class:`~backend.portfolio.constraints.Relaxation`.
        """
        return (
            abs(self.net_exposure - self.requested_constraints.net_exposure) <= CONSTRAINT_TOLERANCE
        )

    def summary(self) -> dict[str, str | float | int | bool | list[str]]:
        """Return a JSON-safe summary for operator display and run artifacts.

        Every value is a scalar or a list of strings, so a dashboard (P9.5) or
        a stored backtest record can carry the portfolio's exposures, its
        relaxation history and the provenance of its risk model without
        re-running the optimizer.

        Returns:
            Mapping of exposures (fractions of capital), the objective
            decomposition (period return fractions), the risk model's shape and
            shrinkage intensity, the relaxation records as descriptive strings,
            and the ladder attempts in the order they were made.
        """
        return {
            "n_assets": self.n_assets,
            "status": self.status,
            "solver": self.solver,
            "objective_value": self.objective_value,
            "expected_return": self.expected_return,
            "variance": self.variance,
            "volatility": self.volatility,
            "risk_penalty": self.risk_penalty,
            "turnover": self.turnover,
            "turnover_cost": self.turnover_cost,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "portfolio_beta": self.portfolio_beta,
            "max_absolute_weight": float(np.max(np.abs(self.weights))) if self.n_assets else 0.0,
            "is_relaxed": self.is_relaxed,
            "full_investment_satisfied": self.full_investment_satisfied,
            "rung": self.rung,
            "relaxations": [item.describe() for item in self.relaxations],
            "attempts": [item.describe() for item in self.attempts],
            "binding_position_caps": list(self.binding_position_caps),
            "binding_sector_caps": list(self.binding_sector_caps),
            "binding_gross_exposure": self.binding_gross_exposure,
            "n_observations": self.n_observations,
            "minimum_observations": self.minimum_observations,
            "shrinkage_intensity": self.shrinkage_intensity,
            "risk_aversion": self.risk_aversion,
            "turnover_penalty": self.turnover_penalty,
            "units": (
                "weights and exposures are fractions of capital; objective terms are "
                "period simple-return fractions; beta is dimensionless"
            ),
            "turnover_convention": (
                "turnover = sum(|w - w_prev|), total traded notional as a fraction of "
                "capital (not halved)"
            ),
            "cost_note": (
                "turnover_cost is the objective's shaping charge, not a transaction "
                "cost; net-of-cost reporting (I4) comes from backend.costs"
            ),
        }


@dataclass(frozen=True, slots=True)
class _Spec:
    """Fully resolved, validated problem data for one rung. Internal."""

    alpha: npt.NDArray[np.float64]
    risk_factor: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    betas: npt.NDArray[np.float64]
    previous_weights: npt.NDArray[np.float64]
    sector_indicator: npt.NDArray[np.float64]
    sector_labels: tuple[str, ...]
    sector_targets: npt.NDArray[np.float64]
    target_beta: float
    constraints: PortfolioConstraints
    risk_aversion: float
    turnover_penalty: float


def turnover_penalty_from_round_trip_cost_bps(round_trip_cost_bps: float) -> float:
    """Convert a round-trip trading cost in basis points to a turnover coefficient.

    The objective charges ``turnover_penalty * sum(|w - w_prev|)``, and
    ``sum(|w - w_prev|)`` is traded notional as a fraction of capital, so a
    coefficient of ``cost_bps * 1e-4`` charges exactly that cost on the notional
    actually traded. A caller wanting the optimizer to trade only when the alpha
    improvement pays for the trade sets the coefficient this way.

    **This does not make anything net of costs.** It shapes the portfolio; it
    does not price it. Invariant I4 is satisfied by :mod:`backend.costs`, whose
    estimates carry ``uncalibrated`` and ``calibration_basis``. A coefficient
    derived from those uncalibrated defaults is itself uncalibrated, and the
    backtest reporting on it must say so using the cost model's own flag rather
    than a claim made here.

    Args:
        round_trip_cost_bps: expected all-in cost of trading one unit of
            notional, in **basis points of that notional** (1 bp = 1e-4). Must
            be finite and non-negative.

    Returns:
        The turnover coefficient, in period return fractions per unit of traded
        notional fraction.

    Raises:
        OptimizerInputError: if the input is negative or non-finite.
    """
    if not math.isfinite(round_trip_cost_bps) or round_trip_cost_bps < 0.0:
        msg = (
            f"round_trip_cost_bps must be finite and >= 0; got {round_trip_cost_bps!r}. "
            f"The unit is basis points of traded notional: 20 bp is 20.0, not 0.0020."
        )
        raise OptimizerInputError(msg)
    return float(round_trip_cost_bps) * 1e-4


def optimize_portfolio(
    *,
    expected_returns: npt.ArrayLike,
    covariance: ShrinkageCovariance,
    sectors: Sequence[str],
    betas: npt.ArrayLike,
    risk_aversion: float,
    turnover_penalty: float,
    previous_weights: npt.ArrayLike | None = None,
    sector_targets: Mapping[str, float] | None = None,
    target_beta: float | None = None,
    constraints: PortfolioConstraints = DEFAULT_CONSTRAINTS,
    relaxation_ladder: RelaxationLadder | None = DEFAULT_RELAXATION_LADDER,
    minimum_observations: int = MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION,
    assets: Sequence[str] | None = None,
    solvers: Sequence[str] = DEFAULT_SOLVERS,
) -> OptimizationResult:
    """Solve for the optimal portfolio, or say precisely why it could not.

    Maximizes ``alpha' w - risk_aversion * w' Sigma w - turnover_penalty *
    ||w - w_prev||_1`` subject to the constraint set in ``constraints`` and the
    neutrality targets resolved from the universe. On infeasibility, climbs
    :data:`~backend.portfolio.constraints.RELAXATION_LADDER` one rung at a time
    and reports what it loosened; when the ladder runs out, raises.

    See the module docstring for units, the turnover convention, the meaning of
    the neutrality targets, and what full investment can and cannot mean at zero
    net exposure. Nothing about the objective's coefficients is defaulted:
    ``risk_aversion`` and ``turnover_penalty`` are required, because a default
    risk aversion would be a portfolio-policy number invented by a library, and
    a defaulted-to-zero turnover penalty would silently delete the term
    directive §5 Phase 9 makes explicit.

    Args:
        expected_returns: length-``n`` expected returns as **period simple
            return fractions**, in the same period as the covariance and in the
            same asset order.
        covariance: the risk model, from
            :func:`~backend.portfolio.covariance.ledoit_wolf_covariance`. Taken
            as the estimate object rather than a bare matrix on purpose: it
            carries the observation count and shrinkage intensity, without which
            the observation policy below cannot be enforced and the result could
            not report how much of its risk model was assumption.
        sectors: length-``n`` sector label per asset. Any hashable-as-string
            labels; GICS sectors in production.
        betas: length-``n`` asset betas against the market (dimensionless).
        risk_aversion: coefficient on ``w' Sigma w``, in ``1 / return
            fraction``. Must be finite and ``>= 0``. No ``1/2`` is applied.
        turnover_penalty: coefficient on ``sum(|w - w_prev|)``, in return
            fraction per unit of traded notional. Must be finite and ``>= 0``.
            Zero is permitted and disables the term — which is exactly the
            control arm the Phase 9 gate's turnover A/B needs.
        previous_weights: length-``n`` current weights as fractions of capital.
            Defaults to zeros, i.e. an initial rebalance out of cash.
        sector_targets: per-sector net-weight targets (fractions of capital).
            Must cover exactly the sectors present and sum to
            ``constraints.net_exposure``. Defaults to the equal-weighted
            universe scaled to the net-exposure target.
        target_beta: portfolio-beta target (dimensionless). Defaults to the
            same reference portfolio's beta, ``net_exposure * mean(betas)``.
        constraints: the caps, exposures and bands to impose. Defaults to
            directive §5 Phase 9's set.
        relaxation_ladder: how far each rung may loosen. ``None`` disables
            relaxation entirely, so an infeasible problem raises immediately
            rather than being approximated.
        minimum_observations: the observation-count policy, in rows behind the
            covariance estimate. Defaults to
            :data:`MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION`; see it for the
            reasoning and for what DECISIONS.md D-020 asked of this layer.
        assets: length-``n`` asset identifiers. Defaults to the labels the
            covariance carries. If both are present they must agree — a
            disagreement means two different asset orderings are in play, which
            silently permutes a portfolio.
        solvers: solver names to try in order. Defaults to
            :data:`DEFAULT_SOLVERS`.

    Returns:
        An :class:`OptimizationResult` carrying the weights, every exposure the
        constraints speak about, the binding limits, the risk model's
        provenance, and the relaxation history.

    Raises:
        OptimizerInputError: if shapes disagree, values are non-finite, policy
            parameters are negative, asset labels conflict, sector targets do
            not cover the sectors present, or the sector targets do not sum to
            the net-exposure target.
        InsufficientRiskHistoryError: if the covariance rests on fewer than
            ``minimum_observations`` observations.
        NotPositiveSemiDefiniteError: if the supplied covariance fails its
            eigenvalue check here — a self-check against a risk model that was
            mutated or built outside the estimator.
        InfeasibleProblemError: if no rung of the ladder admits a portfolio, or
            if relaxation is disabled and the requested set does not.
        SolverFailureError: if no solver reached a definitive status, or the
            only statuses reached were inaccurate ones.
        ConstraintViolationError: if a solution reported ``optimal`` violates a
            constraint when re-checked in NumPy.
    """
    spec, asset_labels = _resolve_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        sectors=sectors,
        betas=betas,
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        previous_weights=previous_weights,
        sector_targets=sector_targets,
        target_beta=target_beta,
        constraints=constraints,
        minimum_observations=minimum_observations,
        assets=assets,
        solvers=solvers,
    )

    attempts: list[SolveAttempt] = []
    rung_count = len(RELAXATION_LADDER) if relaxation_ladder is not None else 0

    for rung in range(rung_count + 1):
        rung_spec, relaxations, unformed_reason = _spec_for_rung(
            spec, relaxation_ladder, rung, solvers
        )
        applied = tuple(RELAXATION_LADDER[:rung])
        if rung_spec is None:
            attempts.append(
                SolveAttempt(
                    rung=rung,
                    relaxations=applied,
                    status=unformed_reason,
                    solver=None,
                    constraints=spec.constraints,
                )
            )
            continue

        status, solver, weights = _solve(rung_spec, solvers)
        attempts.append(
            SolveAttempt(
                rung=rung,
                relaxations=applied,
                status=status,
                solver=solver,
                constraints=rung_spec.constraints,
            )
        )
        if weights is None:
            continue

        violations = _verify(weights, rung_spec)
        if violations:
            raise ConstraintViolationError(
                violations=violations, status=status, solver=solver or "unknown"
            )
        return _build_result(
            weights=weights,
            spec=rung_spec,
            requested=spec,
            relaxations=relaxations,
            rung=rung,
            attempts=tuple(attempts),
            status=status,
            solver=solver or "unknown",
            assets=asset_labels,
            covariance=covariance,
            minimum_observations=minimum_observations,
        )

    raise InfeasibleProblemError(
        attempts=tuple(attempts),
        diagnosis=_diagnose(spec),
        relaxation_enabled=relaxation_ladder is not None,
    )


def _resolve_inputs(
    *,
    expected_returns: npt.ArrayLike,
    covariance: ShrinkageCovariance,
    sectors: Sequence[str],
    betas: npt.ArrayLike,
    risk_aversion: float,
    turnover_penalty: float,
    previous_weights: npt.ArrayLike | None,
    sector_targets: Mapping[str, float] | None,
    target_beta: float | None,
    constraints: PortfolioConstraints,
    minimum_observations: int,
    assets: Sequence[str] | None,
    solvers: Sequence[str],
) -> tuple[_Spec, tuple[str, ...] | None]:
    """Validate every input and resolve defaults into a solvable specification.

    Args:
        expected_returns: see :func:`optimize_portfolio`.
        covariance: see :func:`optimize_portfolio`.
        sectors: see :func:`optimize_portfolio`.
        betas: see :func:`optimize_portfolio`.
        risk_aversion: see :func:`optimize_portfolio`.
        turnover_penalty: see :func:`optimize_portfolio`.
        previous_weights: see :func:`optimize_portfolio`.
        sector_targets: see :func:`optimize_portfolio`.
        target_beta: see :func:`optimize_portfolio`.
        constraints: see :func:`optimize_portfolio`.
        minimum_observations: see :func:`optimize_portfolio`.
        assets: see :func:`optimize_portfolio`.
        solvers: see :func:`optimize_portfolio`.

    Returns:
        The resolved specification and the asset labels, if any.

    Raises:
        OptimizerInputError: on any shape, unit, coverage or arithmetic
            inconsistency in the inputs.
        InsufficientRiskHistoryError: if the risk model is too short.
        NotPositiveSemiDefiniteError: if the risk model fails its eigenvalue
            self-check.
    """
    if minimum_observations < MINIMUM_OBSERVATIONS_FOR_SHRINKAGE:
        msg = (
            f"minimum_observations must be at least "
            f"{MINIMUM_OBSERVATIONS_FOR_SHRINKAGE}, the point below which the "
            f"Ledoit-Wolf intensity is algebraically degenerate and the estimator "
            f"itself refuses; got {minimum_observations}. Lowering the policy below "
            f"the formula's own floor does not buy a risk model, it buys a matrix."
        )
        raise OptimizerInputError(msg)
    if covariance.n_observations < minimum_observations:
        raise InsufficientRiskHistoryError(
            n_observations=covariance.n_observations,
            n_assets=covariance.n_assets,
            minimum_observations=minimum_observations,
            shrinkage_intensity=covariance.shrinkage_intensity,
        )

    if not solvers:
        msg = "solvers must name at least one solver"
        raise OptimizerInputError(msg)
    installed = set(installed_solvers())  # type: ignore[no-untyped-call]
    unknown = [name for name in solvers if name not in installed]
    if unknown:
        msg = (
            f"solver(s) {unknown} are not installed; available: {sorted(installed)}. "
            f"An unavailable solver is a configuration error, not something to route "
            f"around silently."
        )
        raise OptimizerInputError(msg)

    alpha = _as_vector(expected_returns, name="expected_returns")
    n_assets = int(alpha.shape[0])
    if n_assets < 1:
        msg = "expected_returns must contain at least one asset"
        raise OptimizerInputError(msg)
    beta_vector = _as_vector(betas, name="betas", expected_length=n_assets)
    previous = (
        np.zeros(n_assets, dtype=np.float64)
        if previous_weights is None
        else _as_vector(previous_weights, name="previous_weights", expected_length=n_assets)
    )

    if covariance.n_assets != n_assets:
        msg = (
            f"covariance covers {covariance.n_assets} asset(s) but expected_returns has "
            f"{n_assets}. These must be the same universe in the same order; a "
            f"mismatch permutes the risk model against the alphas."
        )
        raise OptimizerInputError(msg)
    if len(sectors) != n_assets:
        msg = f"sectors has {len(sectors)} label(s) but the universe has {n_assets} asset(s)"
        raise OptimizerInputError(msg)
    sector_labels_per_asset = tuple(str(label) for label in sectors)
    if any(not label.strip() for label in sector_labels_per_asset):
        msg = (
            "every asset needs a non-empty sector label; an unlabelled asset cannot be "
            "made sector-neutral and must be excluded upstream, not defaulted here"
        )
        raise OptimizerInputError(msg)

    asset_labels = _resolve_assets(assets=assets, covariance=covariance, n_assets=n_assets)

    for name, value in (
        ("risk_aversion", risk_aversion),
        ("turnover_penalty", turnover_penalty),
    ):
        if not math.isfinite(value):
            msg = f"{name} must be finite; got {value!r}"
            raise OptimizerInputError(msg)
        if value < 0.0:
            msg = (
                f"{name} must be >= 0; got {value!r}. A negative coefficient turns a "
                f"penalty into a reward — for risk, or for trading — and makes the "
                f"objective non-concave or perverse."
            )
            raise OptimizerInputError(msg)

    unique_sectors = tuple(sorted(set(sector_labels_per_asset)))
    indicator = np.zeros((len(unique_sectors), n_assets), dtype=np.float64)
    for column, label in enumerate(sector_labels_per_asset):
        indicator[unique_sectors.index(label), column] = 1.0

    targets = _resolve_sector_targets(
        sector_targets=sector_targets,
        unique_sectors=unique_sectors,
        indicator=indicator,
        n_assets=n_assets,
        net_exposure=constraints.net_exposure,
    )
    resolved_beta_target = (
        constraints.net_exposure * float(np.mean(beta_vector))
        if target_beta is None
        else float(target_beta)
    )
    if not math.isfinite(resolved_beta_target):
        msg = f"target_beta must be finite; got {resolved_beta_target!r}"
        raise OptimizerInputError(msg)

    return (
        _Spec(
            alpha=alpha,
            risk_factor=_risk_factor(covariance),
            covariance=np.asarray(covariance.covariance, dtype=np.float64),
            betas=beta_vector,
            previous_weights=previous,
            sector_indicator=indicator,
            sector_labels=unique_sectors,
            sector_targets=targets,
            target_beta=resolved_beta_target,
            constraints=constraints,
            risk_aversion=float(risk_aversion),
            turnover_penalty=float(turnover_penalty),
        ),
        asset_labels,
    )


def _resolve_assets(
    *, assets: Sequence[str] | None, covariance: ShrinkageCovariance, n_assets: int
) -> tuple[str, ...] | None:
    """Reconcile caller-supplied asset labels with the covariance's own.

    Args:
        assets: labels supplied to the optimizer, or ``None``.
        covariance: the risk model, which may carry labels of its own.
        n_assets: the universe size the labels must match.

    Returns:
        The agreed labels, or ``None`` if neither source supplied any.

    Raises:
        OptimizerInputError: if the label count is wrong, or if both sources
            supplied labels and they disagree.
    """
    supplied = None if assets is None else tuple(str(asset) for asset in assets)
    if supplied is not None and len(supplied) != n_assets:
        msg = f"assets has {len(supplied)} label(s) but the universe has {n_assets} asset(s)"
        raise OptimizerInputError(msg)
    if supplied is not None and covariance.assets is not None and supplied != covariance.assets:
        msg = (
            "the asset labels supplied disagree with the labels carried by the "
            "covariance estimate. Two orderings are in play, and one of them will "
            "silently permute the risk model against the alphas; reconcile them "
            "upstream rather than choosing one here."
        )
        raise OptimizerInputError(msg)
    return supplied if supplied is not None else covariance.assets


def _resolve_sector_targets(
    *,
    sector_targets: Mapping[str, float] | None,
    unique_sectors: tuple[str, ...],
    indicator: npt.NDArray[np.float64],
    n_assets: int,
    net_exposure: float,
) -> npt.NDArray[np.float64]:
    """Resolve per-sector net-weight targets and check the summation identity.

    Args:
        sector_targets: caller-supplied targets, or ``None`` for the default
            equal-weighted-universe reference.
        unique_sectors: sector labels in the order the indicator uses.
        indicator: ``(n_sectors, n_assets)`` 0/1 membership matrix.
        n_assets: universe size.
        net_exposure: the full-investment target the sector targets must sum to.

    Returns:
        Length-``n_sectors`` targets as fractions of capital.

    Raises:
        OptimizerInputError: if the targets do not cover exactly the sectors
            present, contain a non-finite value, or do not sum to
            ``net_exposure``.
    """
    if sector_targets is None:
        counts = indicator.sum(axis=1)
        return np.asarray(net_exposure * counts / float(n_assets), dtype=np.float64)

    supplied = {str(key): float(value) for key, value in sector_targets.items()}
    missing = [label for label in unique_sectors if label not in supplied]
    extra = [label for label in supplied if label not in unique_sectors]
    if missing or extra:
        msg = (
            f"sector_targets must cover exactly the sectors present in the universe. "
            f"Missing: {missing}; unexpected: {extra}. A sector without a target has "
            f"no neutrality requirement at all, which is a silent change of strategy."
        )
        raise OptimizerInputError(msg)
    targets = np.asarray([supplied[label] for label in unique_sectors], dtype=np.float64)
    if not bool(np.isfinite(targets).all()):
        msg = f"sector_targets must all be finite; got {supplied}"
        raise OptimizerInputError(msg)

    total = float(targets.sum())
    if abs(total - net_exposure) > CONSTRAINT_TOLERANCE:
        msg = (
            f"sector targets sum to {total!r} but the full-investment (net exposure) "
            f"target is {net_exposure!r}. Summing the per-sector equalities gives "
            f"sum(w) = sum of sector targets, so these are two statements about the "
            f"same number and no portfolio satisfies both. This is the most common "
            f"way a sector-neutral optimizer becomes inexplicably infeasible, so it "
            f"is refused here with both numbers named rather than left to the solver."
        )
        raise OptimizerInputError(msg)
    return targets


def _as_vector(
    values: npt.ArrayLike, *, name: str, expected_length: int | None = None
) -> npt.NDArray[np.float64]:
    """Coerce an input to a finite one-dimensional float64 vector.

    Args:
        values: the array-like to coerce.
        name: the parameter name, for error messages.
        expected_length: the length required, if it is already known.

    Returns:
        A one-dimensional float64 array.

    Raises:
        OptimizerInputError: if the input is not numeric, not one-dimensional,
            not of the expected length, or contains a non-finite value.
    """
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} could not be read as a numeric vector: {exc}"
        raise OptimizerInputError(msg) from exc
    if vector.ndim != 1:
        msg = f"{name} must be one-dimensional; got shape {vector.shape}"
        raise OptimizerInputError(msg)
    if expected_length is not None and vector.shape[0] != expected_length:
        msg = f"{name} has length {vector.shape[0]}; expected {expected_length}"
        raise OptimizerInputError(msg)
    if not bool(np.isfinite(vector).all()):
        n_bad = int(np.count_nonzero(~np.isfinite(vector)))
        msg = (
            f"{name} contains {n_bad} non-finite value(s). They are refused, not "
            f"imputed: a NaN alpha filled with zero is a position decision made by "
            f"accident."
        )
        raise OptimizerInputError(msg)
    return vector


def _risk_factor(covariance: ShrinkageCovariance) -> npt.NDArray[np.float64]:
    """Return ``L`` with ``L @ L.T == Sigma``, so that ``w' Sigma w == ||L' w||^2``.

    Expressing the risk term as a squared norm rather than as
    ``cvxpy.quad_form`` keeps the problem provably convex from the parse tree
    alone: cvxpy has to be *told* that a matrix is positive semi-definite, and
    telling it so about a matrix that is not is how a "convex" problem quietly
    becomes something whose optimum means nothing.

    The factorization is by symmetric eigendecomposition. Eigenvalues within the
    estimator's own tolerance of zero are clipped to zero — floating-point
    hygiene on a matrix that is positive definite by construction, not a ridge:
    anything more negative than that tolerance raises instead.

    Args:
        covariance: the risk model to factor.

    Returns:
        An ``(n_assets, n_assets)`` factor ``L``.

    Raises:
        NotPositiveSemiDefiniteError: if the matrix has a materially negative
            eigenvalue, meaning it was not produced by (or was mutated after)
            the estimator.
    """
    matrix = np.asarray(covariance.covariance, dtype=np.float64)
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    max_eigenvalue = float(eigenvalues[-1])
    tolerance = DEFAULT_PSD_RELATIVE_TOLERANCE * max(max_eigenvalue, 1.0)
    min_eigenvalue = float(eigenvalues[0])
    if min_eigenvalue < -tolerance:
        raise NotPositiveSemiDefiniteError(min_eigenvalue=min_eigenvalue, tolerance=tolerance)
    clipped = np.clip(eigenvalues, 0.0, None)
    return np.asarray(eigenvectors * np.sqrt(clipped), dtype=np.float64)


def _spec_for_rung(
    spec: _Spec,
    ladder: RelaxationLadder | None,
    rung: int,
    solvers: Sequence[str],
) -> tuple[_Spec | None, tuple[Relaxation, ...], str]:
    """Build the constraint set for one rung of the ladder.

    Relaxations are cumulative: rung ``k`` applies ``RELAXATION_LADDER[:k]``.
    See :data:`~backend.portfolio.constraints.RELAXATION_LADDER` for why the
    order is what it is and why cumulative rather than combinatorial.

    Args:
        spec: the requested specification (rung 0).
        ladder: the relaxation amounts, or ``None`` when relaxation is off.
        rung: which rung to build; ``0`` is the requested set.
        solvers: solver names, needed by the full-investment rung's LP.

    Returns:
        ``(spec, relaxations, reason)``. ``spec`` is ``None`` when the rung
        cannot be formed, and ``reason`` then says which of the two reasons it
        was — the rung does not apply, or the rung was refused because the
        capital it could deploy falls below the ladder's floor. ``reason`` is
        empty when a specification was built.

    Raises:
        SolverFailureError: if the full-investment rung's sizing LP does not
            solve. That LP is feasible by inspection (the empty portfolio), so
            failing to solve it is a solver failure and not evidence about the
            universe; reporting it as "this rung does not apply" would end the
            ladder in an infeasibility claim nothing established.
    """
    if rung == 0 or ladder is None:
        return spec, (), ""

    applied = RELAXATION_LADDER[:rung]
    constraints = spec.constraints
    relaxations: list[Relaxation] = []

    if RelaxableConstraint.SECTOR_NEUTRALITY in applied:
        constraints = constraints.relaxed_with(sector_neutrality_band=ladder.sector_neutrality_band)
        relaxations.append(
            Relaxation(
                constraint=RelaxableConstraint.SECTOR_NEUTRALITY,
                original=spec.constraints.sector_neutrality_band,
                relaxed=ladder.sector_neutrality_band,
                units="fraction of capital, per-sector net-weight deviation",
                rationale=(
                    "relaxed first: the constraint most sensitive to universe "
                    "composition, and the one whose residual is smallest per unit of "
                    "feasibility bought, spread across sectors and reported"
                ),
            )
        )
    if RelaxableConstraint.BETA_NEUTRALITY in applied:
        constraints = constraints.relaxed_with(beta_neutrality_band=ladder.beta_neutrality_band)
        relaxations.append(
            Relaxation(
                constraint=RelaxableConstraint.BETA_NEUTRALITY,
                original=spec.constraints.beta_neutrality_band,
                relaxed=ladder.beta_neutrality_band,
                units="portfolio beta (dimensionless)",
                rationale=(
                    "relaxed second: same shape of concession as the sector band, but "
                    "market beta is the largest common factor in equity returns, so an "
                    "equal-sized residual carries more variance"
                ),
            )
        )
    if RelaxableConstraint.SECTOR_CAP in applied:
        ceiling = max(ladder.sector_cap_ceiling, spec.constraints.sector_cap)
        constraints = constraints.relaxed_with(sector_cap=ceiling)
        relaxations.append(
            Relaxation(
                constraint=RelaxableConstraint.SECTOR_CAP,
                original=spec.constraints.sector_cap,
                relaxed=ceiling,
                units="fraction of capital, gross weight per sector",
                rationale=(
                    "relaxed third: a concentration limit, so widening it admits "
                    "clustered idiosyncratic risk rather than a bounded factor "
                    "residual — a larger and less diversifiable concession than either "
                    "band"
                ),
            )
        )

    spec = replace(spec, constraints=constraints)

    if RelaxableConstraint.FULL_INVESTMENT in applied:
        if spec.constraints.net_exposure <= 0.0:
            # Nothing to reduce: a dollar-neutral book's net exposure is already
            # zero, and the amount of capital it deploys is not governed by this
            # constraint at all (see the module docstring).
            return (
                None,
                (),
                (
                    "not_applicable: the net-exposure target is already 0, so there is "
                    "no full investment left to relax (at zero net exposure the amount "
                    "of capital deployed is decided by the objective, not by this "
                    "constraint)"
                ),
            )
        deployable = _max_net_exposure(spec, solvers)
        if deployable < ladder.minimum_net_exposure:
            return (
                None,
                (),
                (
                    f"refused: the constraints admit at most {deployable:.2%} of "
                    f"capital, below the ladder's floor of "
                    f"{ladder.minimum_net_exposure:.2%}. A book that can deploy this "
                    f"little is a universe problem for a human to look at, and its "
                    f"backtested returns would be a statement about cash more than "
                    f"about the model — raise minimum_net_exposure only having decided "
                    f"that is acceptable"
                ),
            )
        scale = deployable / spec.constraints.net_exposure
        spec = replace(
            spec,
            constraints=spec.constraints.relaxed_with(net_exposure=deployable),
            sector_targets=spec.sector_targets * scale,
            target_beta=spec.target_beta * scale,
        )
        relaxations.append(
            Relaxation(
                constraint=RelaxableConstraint.FULL_INVESTMENT,
                original=float(constraints.net_exposure),
                relaxed=deployable,
                units="fraction of capital, sum of weights",
                rationale=(
                    "relaxed last: the only rung that abandons part of the mandate "
                    "rather than approximating it, and the only one that can rescue a "
                    "universe with too few names to fill the book. Sector and beta "
                    "targets are scaled by the same factor, so the reduced book stays "
                    "neutral in proportion rather than becoming tilted by the "
                    "reduction"
                ),
            )
        )

    return spec, tuple(relaxations), ""


def _max_net_exposure(spec: _Spec, solvers: Sequence[str]) -> float:
    """Find the largest net exposure the non-full-investment constraints admit.

    Solves the linear program ``max t`` over portfolios satisfying every
    constraint except full investment, with the neutrality targets scaled
    proportionally to ``t`` so that the reduced book stays neutral rather than
    inheriting a tilt from the reduction.

    The LP is feasible by inspection — the empty portfolio gives ``t = 0``, and
    every constraint is homogeneous in ``(w, t)`` — so its feasible set of
    exposures is the whole interval ``[0, t*]`` and no solver may report
    ``infeasible`` on it. A solver that fails here has failed, and that is
    raised rather than converted into a claim about the universe.

    Args:
        spec: the specification whose other constraints are already relaxed.
        solvers: solver names to try in order.

    Returns:
        The maximum deployable net exposure as a fraction of capital, stepped
        back by :data:`_NET_EXPOSURE_BACKOFF_ABSOLUTE` /
        :data:`_NET_EXPOSURE_BACKOFF_RELATIVE` so the QP is not asked to sit
        exactly on the boundary of its feasible region, and never above the
        net exposure originally requested.

    Raises:
        SolverFailureError: if no solver reached ``optimal`` on an LP that
            cannot be infeasible.
    """
    constraints = spec.constraints
    n_assets = int(spec.alpha.shape[0])
    weights = Variable(n_assets)
    deployed = Variable(nonneg=True)
    unit_sector_targets = spec.sector_targets / constraints.net_exposure
    unit_beta_target = spec.target_beta / constraints.net_exposure

    problem = Problem(
        Maximize(deployed),
        [
            deployed <= constraints.net_exposure,
            cvxpy_sum(weights) == deployed,
            weights <= constraints.position_cap,
            weights >= -constraints.position_cap,
            norm1(weights) <= constraints.gross_exposure,
            matmul(spec.sector_indicator, cvxpy_abs(weights)) <= constraints.sector_cap,
            cvxpy_abs(matmul(spec.sector_indicator, weights) - unit_sector_targets * deployed)
            <= constraints.sector_neutrality_band,
            cvxpy_abs(matmul(spec.betas, weights) - unit_beta_target * deployed)
            <= constraints.beta_neutrality_band,
        ],
    )
    last_status = "no_solver_ran"
    detail: str | None = None
    for solver in solvers:
        try:
            _run_solver(problem, solver)
        except SolverError as exc:  # pragma: no cover - solver-specific
            last_status = cvxpy_settings.SOLVER_ERROR
            detail = str(exc)
            continue
        last_status = str(problem.status)
        if last_status in _SOLVED_STATUSES and deployed.value is not None:
            raw = min(float(deployed.value), constraints.net_exposure)
            backoff = max(_NET_EXPOSURE_BACKOFF_ABSOLUTE, _NET_EXPOSURE_BACKOFF_RELATIVE * raw)
            return max(raw - backoff, 0.0)
    raise SolverFailureError(  # pragma: no cover - solver-specific
        status=last_status,
        solvers_tried=solvers,
        detail=(
            "this was the linear program that sizes the full-investment rung, which is "
            "feasible by inspection (the empty portfolio), so its failure is a solver "
            f"failure and not evidence that no portfolio exists.{f' {detail}' if detail else ''}"
        ),
    )


def _solve(
    spec: _Spec, solvers: Sequence[str]
) -> tuple[str, str | None, npt.NDArray[np.float64] | None]:
    """Solve one rung's problem, trying each solver until one is definitive.

    Only ``optimal`` counts as solved. A certified ``infeasible`` is a definite
    answer and stops the search — it is what sends the caller to the next rung.
    Every other status (including ``optimal_inaccurate``, whose constraint
    residuals are bounded by the solver's tolerance rather than the problem's)
    falls through to the next solver, and if none is left the caller raises.

    Args:
        spec: the resolved problem data for this rung.
        solvers: solver names to try in order.

    Returns:
        ``(status, solver, weights)``. ``weights`` is ``None`` unless the status
        is ``optimal``.

    Raises:
        SolverFailureError: if no solver reached either ``optimal`` or a
            certified ``infeasible``.
    """
    n_assets = int(spec.alpha.shape[0])
    weights = Variable(n_assets)

    objective = Maximize(
        matmul(spec.alpha, weights)
        - spec.risk_aversion * sum_squares(matmul(spec.risk_factor.T, weights))
        - spec.turnover_penalty * norm1(weights - spec.previous_weights)
    )
    problem = Problem(objective, _cvxpy_constraints(weights, spec))

    last_status = "no_solver_ran"
    detail: str | None = None
    for solver in solvers:
        try:
            _run_solver(problem, solver)
        except SolverError as exc:  # pragma: no cover - solver-specific
            last_status = cvxpy_settings.SOLVER_ERROR
            detail = str(exc)
            continue
        last_status = str(problem.status)
        if last_status in _SOLVED_STATUSES:
            if weights.value is None:  # pragma: no cover - defensive
                continue
            # Returned exactly as the solver produced them. Rounding the
            # sub-tolerance dust to zero would look tidier and would silently
            # move sum(w) by up to one tolerance per asset, breaking the very
            # full-investment equality the tidying was meant to flatter.
            # Rounding to tradeable sizes belongs to execution (Phase 11),
            # where the lot size is known and the rounding is visible.
            return last_status, solver, np.asarray(weights.value, dtype=np.float64)
        if last_status in _INFEASIBLE_STATUSES:
            return last_status, solver, None
        if last_status not in _RETRYABLE_STATUSES:  # pragma: no cover - defensive
            raise SolverFailureError(status=last_status, solvers_tried=solvers, detail=detail)

    raise SolverFailureError(status=last_status, solvers_tried=solvers, detail=detail)


def _run_solver(problem: Problem, solver: str) -> None:
    """Solve a cvxpy problem with this module's options for that solver.

    The one place :meth:`cvxpy.Problem.solve` is called, so that no rung can
    accidentally run a solver at tolerances looser than
    :data:`CONSTRAINT_TOLERANCE` (see :data:`_SOLVER_OPTIONS`).

    Args:
        problem: the problem to solve, mutated in place with the result.
        solver: the cvxpy solver name.

    Raises:
        SolverError: propagated from cvxpy when the solver cannot run at all.
    """
    options: Mapping[str, float] = _SOLVER_OPTIONS.get(solver, {})
    problem.solve(solver=solver, **options)  # type: ignore[no-untyped-call]


def _cvxpy_constraints(weights: Variable, spec: _Spec) -> list[Constraint]:
    """Build the cvxpy constraint list for a rung.

    Args:
        weights: the decision variable.
        spec: the resolved problem data, including the rung's constraint set.

    Returns:
        The constraints, in the order: full investment, position cap (both
        sides), no leverage, sector cap, sector neutrality, beta neutrality.
    """
    constraints = spec.constraints
    sector_deviation = matmul(spec.sector_indicator, weights) - spec.sector_targets
    beta_deviation = matmul(spec.betas, weights) - spec.target_beta
    built: list[Constraint] = [
        cvxpy_sum(weights) == constraints.net_exposure,
        weights <= constraints.position_cap,
        weights >= -constraints.position_cap,
        norm1(weights) <= constraints.gross_exposure,
        matmul(spec.sector_indicator, cvxpy_abs(weights)) <= constraints.sector_cap,
    ]
    if constraints.sector_neutrality_band > 0.0:
        built.append(cvxpy_abs(sector_deviation) <= constraints.sector_neutrality_band)
    else:
        built.append(sector_deviation == 0)
    if constraints.beta_neutrality_band > 0.0:
        built.append(cvxpy_abs(beta_deviation) <= constraints.beta_neutrality_band)
    else:
        built.append(beta_deviation == 0)
    return built


def _verify(weights: npt.NDArray[np.float64], spec: _Spec) -> list[str]:
    """Re-check every constraint in NumPy against the weights actually returned.

    The solver's ``optimal`` is a statement about its own residuals. This is the
    statement about the portfolio: each constraint re-evaluated at
    :data:`CONSTRAINT_TOLERANCE`, in fractions of capital, on the array that
    would otherwise be handed to a backtest.

    Args:
        weights: the candidate portfolio.
        spec: the rung's problem data.

    Returns:
        One sentence per violated constraint; empty when the portfolio is
        sound.
    """
    constraints = spec.constraints
    violations: list[str] = []
    if not bool(np.isfinite(weights).all()):
        n_bad = int(np.count_nonzero(~np.isfinite(weights)))
        violations.append(f"{n_bad} weight(s) are not finite (NaN or inf)")
        return violations

    max_absolute = float(np.max(np.abs(weights)))
    if max_absolute > constraints.position_cap + CONSTRAINT_TOLERANCE:
        violations.append(
            f"position cap: max |w| = {max_absolute!r} exceeds "
            f"{constraints.position_cap!r} (tolerance {CONSTRAINT_TOLERANCE!r})"
        )

    net = float(weights.sum())
    if abs(net - constraints.net_exposure) > CONSTRAINT_TOLERANCE:
        violations.append(
            f"full investment: sum(w) = {net!r} differs from the net-exposure target "
            f"{constraints.net_exposure!r}"
        )

    gross = float(np.abs(weights).sum())
    if gross > constraints.gross_exposure + CONSTRAINT_TOLERANCE:
        violations.append(
            f"no leverage: sum(|w|) = {gross!r} exceeds {constraints.gross_exposure!r}"
        )

    sector_gross = spec.sector_indicator @ np.abs(weights)
    worst_gross = int(np.argmax(sector_gross))
    if float(sector_gross[worst_gross]) > constraints.sector_cap + CONSTRAINT_TOLERANCE:
        violations.append(
            f"sector cap: sector {spec.sector_labels[worst_gross]!r} holds gross "
            f"{float(sector_gross[worst_gross])!r}, above {constraints.sector_cap!r}"
        )

    sector_deviation = np.abs(spec.sector_indicator @ weights - spec.sector_targets)
    worst_net = int(np.argmax(sector_deviation))
    if (
        float(sector_deviation[worst_net])
        > constraints.sector_neutrality_band + CONSTRAINT_TOLERANCE
    ):
        violations.append(
            f"sector neutrality: sector {spec.sector_labels[worst_net]!r} deviates "
            f"{float(sector_deviation[worst_net])!r} from its target, above the band "
            f"{constraints.sector_neutrality_band!r}"
        )

    beta_deviation = abs(float(spec.betas @ weights) - spec.target_beta)
    if beta_deviation > constraints.beta_neutrality_band + CONSTRAINT_TOLERANCE:
        violations.append(
            f"beta neutrality: portfolio beta deviates {beta_deviation!r} from its "
            f"target {spec.target_beta!r}, above the band "
            f"{constraints.beta_neutrality_band!r}"
        )
    return violations


def _diagnose(spec: _Spec) -> list[str]:
    """Explain, from arithmetic alone, why the requested constraints cannot hold.

    Each check is a *necessary* condition, so a violation is a proof of
    infeasibility that names the numbers responsible. Silence here does not
    prove feasibility — it means no single constraint is impossible on its own
    and the conflict lives in their interaction, which is itself worth telling
    the operator.

    Args:
        spec: the requested (rung 0) specification.

    Returns:
        One sentence per violated necessary condition.
    """
    constraints = spec.constraints
    notes: list[str] = []
    n_assets = int(spec.alpha.shape[0])
    capacity = n_assets * constraints.position_cap
    if capacity < constraints.net_exposure - CONSTRAINT_TOLERANCE:
        notes.append(
            f"the universe cannot fill the book: {n_assets} name(s) at a "
            f"{constraints.position_cap:.2%} position cap reach {capacity:.2%} of "
            f"capital, short of the {constraints.net_exposure:.2%} full-investment "
            f"target"
        )

    counts = spec.sector_indicator.sum(axis=1)
    sector_capacity = 0.0
    for index, label in enumerate(spec.sector_labels):
        names_in_sector = int(counts[index])
        target = float(spec.sector_targets[index])
        reachable = names_in_sector * constraints.position_cap
        sector_capacity += min(reachable, constraints.sector_cap)
        if abs(target) > reachable + constraints.sector_neutrality_band + CONSTRAINT_TOLERANCE:
            notes.append(
                f"sector {label!r} needs a net weight of {target:.2%} but holds "
                f"{names_in_sector} name(s), which reach at most {reachable:.2%} at "
                f"the {constraints.position_cap:.2%} position cap"
            )
        if abs(target) > constraints.sector_cap + constraints.sector_neutrality_band:
            notes.append(
                f"sector {label!r} needs a net weight of {target:.2%}, above the "
                f"{constraints.sector_cap:.2%} sector cap"
            )
    if sector_capacity < constraints.net_exposure - CONSTRAINT_TOLERANCE:
        notes.append(
            f"summed over sectors, the position and sector caps admit at most "
            f"{sector_capacity:.2%} of capital, short of the "
            f"{constraints.net_exposure:.2%} full-investment target"
        )

    reachable_beta = _reachable_beta_range(spec)
    if reachable_beta is not None:
        low, high = reachable_beta
        band = constraints.beta_neutrality_band
        if spec.target_beta < low - band or spec.target_beta > high + band:
            notes.append(
                f"the beta target {spec.target_beta:.4f} lies outside the "
                f"[{low:.4f}, {high:.4f}] range reachable at the "
                f"{constraints.position_cap:.2%} position cap and "
                f"{constraints.net_exposure:.2%} net exposure"
            )
    return notes


def _reachable_beta_range(spec: _Spec) -> tuple[float, float] | None:
    """Bound the portfolio betas reachable under the position, net and gross limits.

    Ignores the sector constraints, so the interval it returns is a superset of
    what is truly reachable — which is exactly what a *necessary* condition
    needs, and what makes a violation of it a proof rather than a suspicion.

    Two regimes, because the arithmetic genuinely differs:

    - **``net_exposure == gross_exposure``** (the shipped default, and the only
      case the first draft of this function handled). ``sum(w) = net`` with
      ``sum(|w|) <= net`` forces ``w >= 0`` elementwise, so the problem is a
      fractional knapsack and filling the highest-beta names to the cap is
      *exactly* the reachable maximum. Sharp, and sharpness is worth having:
      this is the configuration the directive specifies.
    - **``net_exposure < gross_exposure``** (a long-short book). Shorts are
      available, and the greedy long-only fill is then **not** an upper bound —
      measured on a 60-name universe at ``net = 0.5``, ``gross = 1.0`` it
      claimed ``[0.263, 0.737]`` against a true ``[0.110, 0.890]``. Using it
      there would let this function *manufacture* a proof of infeasibility
      against a beta target that is perfectly reachable, and name beta
      neutrality as the culprit for someone else's conflict. The bound used
      instead is centred: with ``beta_ref = (max + min) / 2``,
      ``beta'w = beta_ref * net + (beta - beta_ref)'w``, and the second term is
      bounded by filling the largest ``|beta_i - beta_ref|`` to the position cap
      until the gross limit is used up. Looser, but valid for any sign pattern.

    Args:
        spec: the requested specification.

    The centred bound is also what makes the diagnosis work for a
    **dollar-neutral** book. At ``net_exposure == 0`` there is no long-only
    regime to fall back on, but an unreachable beta target is one of the more
    likely ways such a book becomes infeasible — so this must not decline to
    answer there, which an earlier long-only-only formulation had to.

    Returns:
        ``(minimum, maximum)`` reachable portfolio beta, or ``None`` when the
        net exposure cannot be reached at all under the position cap (in which
        case the capacity diagnosis already covers it).
    """
    constraints = spec.constraints
    cap = constraints.position_cap
    net = constraints.net_exposure
    gross = constraints.gross_exposure
    n_assets = int(spec.betas.shape[0])
    if cap <= 0.0 or net > n_assets * cap:
        return None

    if gross <= net + CONSTRAINT_TOLERANCE:
        # Long-only by arithmetic: the greedy fill is exact.
        ascending = np.sort(spec.betas)
        notional = _greedy_notional(n_assets=n_assets, cap=cap, budget=net)
        return float(ascending @ notional), float(ascending[::-1] @ notional)

    reference = (float(spec.betas.max()) + float(spec.betas.min())) / 2.0
    deviations = np.sort(np.abs(spec.betas - reference))[::-1]
    notional = _greedy_notional(n_assets=n_assets, cap=cap, budget=min(gross, n_assets * cap))
    swing = float(deviations @ notional)
    centre = reference * net
    return centre - swing, centre + swing


def _greedy_notional(*, n_assets: int, cap: float, budget: float) -> npt.NDArray[np.float64]:
    """Spread a notional budget over ranked names, filling each to the cap in turn.

    Args:
        n_assets: how many names are available (count).
        cap: the most any one name may carry (fraction of capital).
        budget: the total notional to place (fraction of capital). Assumed to be
            within ``n_assets * cap``; any excess is dropped.

    Returns:
        Length-``n_assets`` notional per rank, summing to ``budget``.
    """
    filled = min(int(budget // cap), n_assets)
    notional = np.zeros(n_assets, dtype=np.float64)
    notional[:filled] = cap
    if filled < n_assets:
        notional[filled] = budget - filled * cap
    return notional


def _build_result(
    *,
    weights: npt.NDArray[np.float64],
    spec: _Spec,
    requested: _Spec,
    relaxations: tuple[Relaxation, ...],
    rung: int,
    attempts: tuple[SolveAttempt, ...],
    status: str,
    solver: str,
    assets: tuple[str, ...] | None,
    covariance: ShrinkageCovariance,
    minimum_observations: int,
) -> OptimizationResult:
    """Assemble the result, computing every exposure from the returned weights.

    Nothing here is read back out of cvxpy: every reported number is recomputed
    in NumPy from the weights, so the exposures a caller sees are the exposures
    of the array they were handed.

    Args:
        weights: the verified portfolio.
        spec: the rung's problem data (relaxed constraints, scaled targets).
        requested: the rung-0 problem data, for the "as requested" fields.
        relaxations: what was loosened to get here.
        rung: which rung produced it.
        attempts: every rung tried.
        status: the accepted solver status.
        solver: the solver that produced it.
        assets: asset labels, if any.
        covariance: the risk model, for its provenance fields.
        minimum_observations: the observation policy applied.

    Returns:
        The finished :class:`OptimizationResult`.
    """
    constraints = spec.constraints
    sector_net = spec.sector_indicator @ weights
    sector_gross = spec.sector_indicator @ np.abs(weights)
    variance = float(weights @ spec.covariance @ weights)
    expected_return = float(spec.alpha @ weights)
    turnover = float(np.abs(weights - spec.previous_weights).sum())
    portfolio_beta = float(spec.betas @ weights)

    realized: dict[RelaxableConstraint, float] = {
        RelaxableConstraint.SECTOR_NEUTRALITY: float(
            np.max(np.abs(sector_net - spec.sector_targets))
        ),
        RelaxableConstraint.BETA_NEUTRALITY: abs(portfolio_beta - spec.target_beta),
        RelaxableConstraint.SECTOR_CAP: float(np.max(sector_gross)),
        RelaxableConstraint.FULL_INVESTMENT: float(weights.sum()),
    }
    recorded = tuple(replace(item, realized=realized[item.constraint]) for item in relaxations)

    labels = assets if assets is not None else tuple(f"#{index}" for index in range(weights.size))
    binding_positions = tuple(
        labels[index]
        for index in range(weights.size)
        if abs(float(weights[index])) >= constraints.position_cap - BINDING_TOLERANCE
    )
    binding_sectors = tuple(
        spec.sector_labels[index]
        for index in range(len(spec.sector_labels))
        if float(sector_gross[index]) >= constraints.sector_cap - BINDING_TOLERANCE
    )
    gross_exposure = float(np.abs(weights).sum())
    binding_gross = gross_exposure >= constraints.gross_exposure - BINDING_TOLERANCE

    frozen = np.array(weights, dtype=np.float64, copy=True)
    frozen.flags.writeable = False

    return OptimizationResult(
        weights=frozen,
        assets=assets,
        sectors=tuple(
            spec.sector_labels[int(np.argmax(spec.sector_indicator[:, column]))]
            for column in range(weights.size)
        ),
        status=status,
        solver=solver,
        objective_value=(
            expected_return - spec.risk_aversion * variance - spec.turnover_penalty * turnover
        ),
        expected_return=expected_return,
        variance=variance,
        volatility=math.sqrt(max(variance, 0.0)),
        risk_penalty=spec.risk_aversion * variance,
        turnover=turnover,
        turnover_cost=spec.turnover_penalty * turnover,
        net_exposure=float(weights.sum()),
        gross_exposure=gross_exposure,
        portfolio_beta=portfolio_beta,
        sector_net_exposures=MappingProxyType(
            {label: float(sector_net[index]) for index, label in enumerate(spec.sector_labels)}
        ),
        sector_gross_exposures=MappingProxyType(
            {label: float(sector_gross[index]) for index, label in enumerate(spec.sector_labels)}
        ),
        sector_targets=MappingProxyType(
            {
                label: float(spec.sector_targets[index])
                for index, label in enumerate(spec.sector_labels)
            }
        ),
        target_beta=spec.target_beta,
        binding_position_caps=binding_positions,
        binding_sector_caps=binding_sectors,
        binding_gross_exposure=binding_gross,
        constraints=constraints,
        requested_constraints=requested.constraints,
        relaxations=recorded,
        rung=rung,
        attempts=attempts,
        n_observations=covariance.n_observations,
        n_assets=int(weights.size),
        minimum_observations=minimum_observations,
        shrinkage_intensity=covariance.shrinkage_intensity,
        risk_aversion=spec.risk_aversion,
        turnover_penalty=spec.turnover_penalty,
    )
