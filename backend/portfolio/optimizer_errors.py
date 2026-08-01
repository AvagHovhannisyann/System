"""Failure taxonomy for the portfolio optimizer (P9.2).

Separate from :mod:`backend.portfolio.errors` — which belongs to the covariance
estimator (P9.1) — but rooted in the same
:class:`~backend.portfolio.errors.PortfolioError` base, so a caller can catch
"portfolio construction failed" without knowing which layer failed.

Every failure here is raised. The alternative an optimizer is always tempted
toward — return the previous weights, return zeros, return whatever the solver
last had in its buffer — is the exact failure mode directive §0.3 warns about
for this component: a silent error that does not surface as a test failure and
quietly invalidates every backtest built on it. A portfolio that violates its
sector cap and says nothing is worse than no portfolio at all, because the
backtest will happily report its returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.portfolio.errors import PortfolioError

if TYPE_CHECKING:
    from collections.abc import Sequence

    # Type-checking only, so the runtime dependency stays one-directional
    # (optimizer imports errors, never the reverse). Typing the attempt log
    # properly matters: a caller reading which rungs were tried — the dashboard
    # §6.8 constraint panel, or a test asserting the ladder order — should not
    # have to cast its way through a bag of `object`.
    from backend.portfolio.optimizer import SolveAttempt

__all__ = [
    "ConstraintViolationError",
    "InfeasibleProblemError",
    "InsufficientRiskHistoryError",
    "OptimizerError",
    "OptimizerInputError",
    "SolverFailureError",
]


class OptimizerError(PortfolioError):
    """Base class for portfolio-optimization failures."""


class OptimizerInputError(OptimizerError):
    """Raised when the optimizer's inputs are inconsistent or unusable.

    Covers shape disagreements between the expected returns, the covariance,
    the sector labels and the betas; non-finite inputs; negative or incoherent
    policy parameters; and the arithmetic identity that sector targets must sum
    to the net-exposure target.

    That last one deserves its own sentence, because it is the single most
    common way a sector-neutral optimizer becomes mysteriously infeasible: the
    per-sector target weights and the full-investment target are two statements
    about the same sum. If the sector targets add to 1.0 and the net-exposure
    target is 0.0, no portfolio in the universe satisfies both, and a solver
    will report ``infeasible`` without ever saying why. Catching it here turns
    an unexplained solver status into a sentence naming the two numbers that
    disagree.
    """


class InsufficientRiskHistoryError(OptimizerError):
    """Raised when the risk model rests on fewer observations than policy allows.

    DECISIONS.md **D-020** left this decision explicitly open: the covariance
    module refuses below three observations because that is where the
    Ledoit-Wolf *formula* degenerates, and it recorded that the *policy*
    minimum — the point below which a risk model should not be traded on at all
    — belongs to the consumer, "where it is visible". This is that consumer and
    this is that check. See
    :data:`~backend.portfolio.optimizer.MINIMUM_OBSERVATIONS_FOR_OPTIMIZATION`
    for the number chosen and the reasoning behind it.

    Attributes:
        n_observations: rows behind the supplied covariance estimate (count).
        n_assets: columns in that estimate (count).
        minimum_observations: the policy minimum that was applied (count).
        shrinkage_intensity: the estimate's Ledoit-Wolf intensity
            (dimensionless, in ``[0, 1]``) — reported because it is the direct
            measure of how much of the refused risk model was assumption.
    """

    def __init__(
        self,
        *,
        n_observations: int,
        n_assets: int,
        minimum_observations: int,
        shrinkage_intensity: float,
    ) -> None:
        """Build the error from the risk model's shape and its shrinkage intensity."""
        self.n_observations = n_observations
        self.n_assets = n_assets
        self.minimum_observations = minimum_observations
        self.shrinkage_intensity = shrinkage_intensity
        super().__init__(
            f"the risk model rests on {n_observations} observation(s) across {n_assets} "
            f"asset(s); this optimizer requires at least {minimum_observations}. "
            f"The estimate offered is {shrinkage_intensity:.1%} shrinkage target, i.e. "
            f"that fraction of it is imposed structure rather than measured data. "
            f"The covariance estimator's own minimum (3) is where the Ledoit-Wolf "
            f"formula stops degenerating, not where a risk model becomes trustworthy "
            f"(DECISIONS.md D-020); this is the policy minimum, and it is enforced "
            f"here because here it is visible at the call site. Pass "
            f"`minimum_observations=` explicitly to state a different policy — "
            f"the number then appears in the result and in every artifact built "
            f"from it."
        )


class InfeasibleProblemError(OptimizerError):
    """Raised when no portfolio satisfies the constraints, relaxed or not.

    This is the terminal rung of the relaxation ladder. Reaching it means the
    optimizer tried the requested constraint set, then each documented
    relaxation in order, and found no feasible portfolio at any of them — or
    that relaxation was disabled by the caller.

    It is a refusal, not a fallback. The alternatives an optimizer is tempted
    into here — returning the previous weights, returning the equal-weighted
    universe, returning zeros — all produce a portfolio the backtest cannot
    distinguish from a solved one, which is how a constraint violation becomes
    a performance number.

    Attributes:
        attempts: one entry per rung tried, in the order tried, each carrying
            the rung's relaxations and the solver status it produced.
        diagnosis: necessary-condition violations computed from the inputs
            (each a sentence naming the numbers that conflict). Empty when no
            simple arithmetic conflict explains the infeasibility, which is
            itself informative — it means the conflict is in the interaction of
            constraints rather than in any single one.
        relaxation_enabled: ``False`` if the caller passed no ladder, in which
            case only the requested constraint set was tried.
    """

    def __init__(
        self,
        *,
        attempts: Sequence[SolveAttempt],
        diagnosis: Sequence[str],
        relaxation_enabled: bool,
    ) -> None:
        """Build the error from the rungs attempted and the arithmetic diagnosis."""
        self.attempts = tuple(attempts)
        self.diagnosis = tuple(diagnosis)
        self.relaxation_enabled = relaxation_enabled
        ladder_note = (
            f" The relaxation ladder was exhausted: {len(self.attempts)} constraint "
            f"set(s) were tried, in the documented order."
            if relaxation_enabled
            else " Relaxation was disabled by the caller, so only the requested "
            "constraint set was tried."
        )
        diagnosis_note = (
            " Necessary conditions violated: " + "; ".join(self.diagnosis)
            if self.diagnosis
            else " No single constraint is arithmetically impossible on its own, so "
            "the conflict is in their interaction — inspect the sector composition "
            "of the universe and the beta dispersion within it."
        )
        super().__init__(
            f"no portfolio satisfies the constraints.{ladder_note}{diagnosis_note} "
            f"No weights are returned: a portfolio that violates its constraints and "
            f"does not say so becomes a backtest result that cannot be told apart "
            f"from a solved one."
        )


class SolverFailureError(OptimizerError):
    """Raised when the solver did not reach a status that means "solved".

    ``optimal`` is the only status treated as a solution. In particular
    ``optimal_inaccurate`` is refused: it means the solver stopped on its own
    tolerance rather than on the problem's, so the returned weights may violate
    the position or sector cap by an unknown amount. A cap that is respected
    "to within the solver's convenience" is not a cap.

    Attributes:
        status: the terminal cvxpy status string.
        solvers_tried: solver names attempted, in order.
        detail: any solver exception text, or ``None``.
    """

    def __init__(
        self,
        *,
        status: str,
        solvers_tried: Sequence[str],
        detail: str | None = None,
    ) -> None:
        """Build the error from the final status and the solvers attempted."""
        self.status = status
        self.solvers_tried = tuple(solvers_tried)
        self.detail = detail
        detail_note = f" Solver detail: {detail}" if detail else ""
        super().__init__(
            f"the optimizer did not solve: final status {status!r} after trying "
            f"{list(self.solvers_tried)}. Only 'optimal' is accepted as a solution; "
            f"'optimal_inaccurate' in particular is refused, because it means the "
            f"constraint residuals are bounded by the solver's tolerance rather than "
            f"by the problem's, and an unbounded violation of a 2% position cap is "
            f"not a 2% position cap.{detail_note}"
        )


class ConstraintViolationError(OptimizerError):
    """Raised when a solution reported ``optimal`` violates a constraint anyway.

    This is the last line of defence and it checks the solver's work rather
    than trusting it: every constraint is re-evaluated in NumPy against the
    weights actually returned, at a tolerance stated by this module rather than
    by the solver. A violation here means either a numerically hard problem or
    a bug in the problem construction; either way the weights must not escape.

    Attributes:
        violations: one sentence per violated constraint, naming the realized
            value, the limit, and the tolerance.
        status: the solver status that accompanied the bad solution (always a
            "solved" status — that is the point).
        solver: the solver that produced it.
    """

    def __init__(self, *, violations: Sequence[str], status: str, solver: str) -> None:
        """Build the error from the violated constraints and the solver that lied."""
        self.violations = tuple(violations)
        self.status = status
        self.solver = solver
        super().__init__(
            f"solver {solver!r} reported status {status!r} but the returned weights "
            f"violate {len(self.violations)} constraint(s): "
            f"{'; '.join(self.violations)}. The weights are discarded. Constraints "
            f"are re-checked in NumPy against this module's tolerance precisely so "
            f"that a solver's idea of 'optimal' is never the last word on whether a "
            f"position cap held."
        )
