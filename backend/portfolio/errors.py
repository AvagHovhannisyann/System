"""Failure taxonomy for portfolio construction (P9).

Every failure here is raised, never absorbed into a plausible-looking return
value (invariant I3, forbidden behavior #2). A covariance estimator that
quietly returns something usable when its input was degenerate is the exact
shape of bug the directive's cost/risk realism rules exist to prevent: the
optimizer downstream cannot tell the difference between "this is the risk
model" and "this is what the risk model degraded to".
"""

from __future__ import annotations

__all__ = [
    "CovarianceError",
    "DegenerateReturnsError",
    "InsufficientObservationsError",
    "NotPositiveSemiDefiniteError",
    "PortfolioError",
    "SingularCovarianceError",
]


class PortfolioError(Exception):
    """Base class for portfolio-construction failures."""


class CovarianceError(PortfolioError):
    """Base class for covariance-estimation failures."""


class DegenerateReturnsError(CovarianceError):
    """Raised when a return matrix cannot support a covariance estimate.

    Covers the cases where there is nothing honest to return: non-numeric data,
    wrong dimensionality, fewer than two observations, no assets, non-finite
    entries, or a panel with no usable variance — every column constant, so the
    shrinkage target is the zero matrix and shrinkage cannot regularize
    anything. "No usable variance" is judged against the floating-point noise
    floor of demeaning rather than against exact zero, because a constant panel
    demeans to residue of order ``eps * level`` rather than to nothing.

    A panel that is merely *short* is a different failure with a different
    remedy, and raises :class:`InsufficientObservationsError` instead.

    Non-finite entries are refused rather than imputed. Silently dropping or
    filling NaNs would make the estimate depend on a hidden imputation rule
    that no caller declared.
    """


class InsufficientObservationsError(CovarianceError):
    """Raised when a panel is too short for the shrinkage formula to be valid.

    The Ledoit-Wolf intensity is a ratio of two estimated quantities: the
    distance from the sample covariance ``S`` to the target, and ``beta``, the
    estimated error in ``S`` itself. ``beta`` is estimated by comparing each
    observation's outer product ``x_k x_k'`` against their average ``S``:

    ::

        beta ∝ (1 / n) * sum_k || x_k x_k' - S ||_F^2

    With exactly **two** observations, demeaning makes the rows exact negatives
    (``x_2 == -x_1``), so ``x_1 x_1' == x_2 x_2' == S`` and every term of that
    sum is identically zero. The estimator therefore concludes that ``S`` was
    measured *without error* and shrinks by exactly zero — the opposite of the
    truth, since a two-observation covariance has rank one and is essentially
    all noise. It is an algebraic degeneracy of the formula, reproduced exactly
    by scikit-learn's implementation, not a property of the data.

    **Why this raises rather than warning or flooring.** A zero intensity means
    the caller silently receives the raw sample covariance — near-singular,
    which is the precise condition shrinkage exists to prevent. Warning is
    useless because the consumer is a mean-variance optimizer, which does not
    read warnings and responds to a singular covariance by taking an unbounded
    position along its null space. Flooring the intensity at some positive
    number would return a matrix labelled "Ledoit-Wolf" whose shrinkage was
    invented here rather than derived from the data — a plausible value in
    place of the work, which invariant I3 and forbidden behavior #2 rule out,
    and which would also make the reported intensity (the operator's measure of
    how much of the risk model is assumption) a fiction.

    The minimum is the boundary of the formula's validity, not a recommendation:
    three observations make the estimator well-defined, not adequate. A caller
    wanting a serious risk model needs far more, and that policy belongs to the
    caller, where it is visible.

    Attributes:
        n_observations: rows in the offending return panel (count).
        n_assets: columns in the offending return panel (count).
        minimum_observations: the smallest row count the estimator accepts
            (count).
    """

    def __init__(self, *, n_observations: int, n_assets: int, minimum_observations: int) -> None:
        """Build the error from the panel shape and the required minimum."""
        self.n_observations = n_observations
        self.n_assets = n_assets
        self.minimum_observations = minimum_observations
        mechanism = (
            " With exactly 2 observations the demeaned rows are exact negatives, so "
            "every observation's outer product equals the sample covariance, so the "
            "estimated error in the sample covariance is identically zero and the "
            "analytic intensity is exactly 0. Zero shrinkage would hand back the raw "
            "sample covariance — rank 1, near-singular — which is exactly what "
            "shrinkage exists to prevent."
            if n_observations == 2
            else ""
        )
        super().__init__(
            f"Ledoit-Wolf shrinkage needs at least {minimum_observations} observations; "
            f"got {n_observations} (panel is {n_observations}x{n_assets})."
            f"{mechanism}"
            f" Supply more observations. The intensity is not floored to a positive "
            f"value here, because a floor would be a number this estimator invented "
            f"rather than derived, reported to the operator as if it had been measured."
        )


class NotPositiveSemiDefiniteError(CovarianceError):
    """Raised when a computed covariance matrix fails its PSD check.

    This is a self-check on the estimator, not on the caller's data: the
    Ledoit-Wolf estimate is PSD by construction (a convex combination of a
    Gram-matrix-derived sample covariance and a non-negative multiple of the
    identity), so reaching this error means numerical conditioning has gone
    wrong badly enough that the result must not be handed to an optimizer.

    Attributes:
        min_eigenvalue: the smallest eigenvalue found (units: squared
            per-period return fraction).
        tolerance: the negative-eigenvalue tolerance that was applied, scaled
            by the largest eigenvalue (same units).
    """

    def __init__(self, *, min_eigenvalue: float, tolerance: float) -> None:
        """Build the error from the failing eigenvalue and the tolerance used."""
        self.min_eigenvalue = min_eigenvalue
        self.tolerance = tolerance
        super().__init__(
            f"covariance estimate is not positive semi-definite: smallest eigenvalue "
            f"{min_eigenvalue!r} is below the tolerance {-tolerance!r}. "
            f"A near-singular or indefinite covariance handed to a mean-variance "
            f"optimizer produces enormous, meaningless positions; the estimate is "
            f"refused rather than returned."
        )


class SingularCovarianceError(CovarianceError):
    """Raised when the shrunk estimate is still numerically singular.

    Shrinkage regularizes because the intensity ``delta`` is strictly positive:
    the estimate's smallest eigenvalue is then at least ``delta * mu``, where
    ``mu`` is the average sample variance. When the analytic intensity comes
    out at (or numerically indistinguishable from) zero, the estimate *is* the
    sample covariance, and if assets outnumber observations that matrix is
    singular.

    The two-observation case that makes the intensity identically zero is
    refused earlier and separately, by
    :class:`InsufficientObservationsError`, because it has a specific cause and
    a specific remedy. This error is the general backstop for every other way
    the intensity can come out at zero on a rank-deficient panel: ``beta`` is
    an average of ``|| x_k x_k' - S ||_F^2`` over observations, so it vanishes
    whenever every demeaned observation is the same vector up to sign, at any
    ``n``. A panel that alternates between two opposite return vectors reaches
    this error at ``n = 50`` as readily as at ``n = 3``.

    A singular matrix is not returned with a warning, because the caller that
    would ignore the warning is a mean-variance optimizer, and what it does
    with a singular covariance is take an unbounded position along the null
    space. The remedy is more observations, not a fudge factor: adding a ridge
    here would invent structure the estimator did not choose and would hide the
    fact that the panel cannot support a risk model.

    Attributes:
        shrinkage_intensity: the intensity the estimator chose (dimensionless).
        min_eigenvalue: smallest eigenvalue of the estimate (squared per-period
            return fraction).
        max_eigenvalue: largest eigenvalue of the estimate (same units).
        threshold: the numerical-rank threshold it failed, ``n_assets *
            eps * max_eigenvalue`` (same units).
        n_observations: rows in the return panel (count).
        n_assets: columns in the return panel (count).
    """

    def __init__(
        self,
        *,
        shrinkage_intensity: float,
        min_eigenvalue: float,
        max_eigenvalue: float,
        threshold: float,
        n_observations: int,
        n_assets: int,
    ) -> None:
        """Build the error from the estimate's conditioning and the panel shape."""
        self.shrinkage_intensity = shrinkage_intensity
        self.min_eigenvalue = min_eigenvalue
        self.max_eigenvalue = max_eigenvalue
        self.threshold = threshold
        self.n_observations = n_observations
        self.n_assets = n_assets
        two_observation_note = (
            " With exactly 2 observations the analytic intensity is always zero "
            "(the two demeaned rows are negatives, so their squares are identical "
            "and the estimated error in the sample covariance vanishes), so no "
            "2-observation panel of more than one asset can produce an invertible "
            "estimate."
            if n_observations == 2
            else ""
        )
        super().__init__(
            f"shrinkage covariance is numerically singular: shrinkage intensity "
            f"{shrinkage_intensity!r} left a smallest eigenvalue of "
            f"{min_eigenvalue!r}, at or below the numerical-rank threshold "
            f"{threshold!r} for a {n_observations}x{n_assets} panel."
            f"{two_observation_note}"
            f" Supply more observations; a ridge added here would invent structure "
            f"the estimator did not choose."
        )
