"""Ledoit-Wolf shrinkage covariance for the asset return panel (P9.1).

**Why shrinkage at all.** The sample covariance matrix of ``p`` assets over
``n`` observations has rank at most ``min(n, p)`` (``min(n - 1, p)`` once the
mean is estimated). Cross-sectionally the platform always has more names than
it has clean observations for them, so the sample estimate is singular or
near-singular, its smallest eigenvalues are noise, and a mean-variance
optimizer — which inverts it — loads maximally onto exactly those noise
directions. The result is enormous offsetting positions that look optimal and
are meaningless. Shrinkage pulls the estimate toward a well-conditioned target
and buys a large variance reduction for a small, quantified bias.

**The estimator.** Ledoit & Wolf (2004), *"Honey, I Shrunk the Sample
Covariance Matrix"*: the target is the spherical matrix ``mu * I`` where
``mu = trace(S) / p`` is the average sample variance, and the estimate is the
convex combination

::

    Sigma = (1 - delta) * S + delta * mu * I

with ``delta`` the analytically optimal shrinkage intensity (the value
minimizing expected squared Frobenius distance to the true covariance,
estimated from the data). ``delta`` is computed here in NumPy rather than
imported, because scikit-learn ships no type information and this module is
type-checked under ``mypy --strict``; the implementation follows the same
formula scikit-learn uses, and
``backend/tests/portfolio/test_covariance.py`` pins it against
``sklearn.covariance.ledoit_wolf`` so a divergence is a test failure rather
than a silent modelling difference.

**Units.** Every function here takes *simple period returns expressed as
fractions* — ``0.01`` means a 1% return, never ``1.0`` and never ``100``. The
resulting covariance is therefore in squared return fractions per period
squared, and variances are in squared return fractions per period. Nothing in
this module annualizes: a caller who wants annual units multiplies the matrix
by the number of periods per year and says so at its own boundary. Mixing a
percent-valued panel into this function silently inflates every variance by
10,000, which is the failure mode directive §8 singles out.

**What the operator sees.** :class:`ShrinkageCovariance` carries the
shrinkage intensity, the target, the sample covariance it shrank, and the
conditioning of the result. The intensity is the honest measure of how much
structure was imposed rather than estimated: ``delta = 0.9`` means the risk
model is nine parts assumption to one part data, and an operator reading a
backtest built on it deserves to know that without re-deriving it.

**The two-observation degeneracy, and why this module refuses it.** The
analytic intensity is ``min(beta, delta_dist) / delta_dist`` where ``beta``
estimates the error in ``S`` itself, as an average of
``|| x_k x_k' - S ||_F^2`` across observations. At ``n_observations == 2``
demeaning makes the two rows exact negatives, so both outer products equal
``S``, so ``beta`` is *identically zero* and the intensity is *exactly zero* —
the estimator concludes ``S`` was measured without error, when in fact it has
rank one and is almost entirely noise. scikit-learn reproduces this exactly;
it is a degeneracy of the formula, not of this implementation or of the data.

Zero intensity means the caller silently gets the raw sample covariance, which
is the near-singular object shrinkage exists to prevent, so these functions
**raise** :class:`~backend.portfolio.errors.InsufficientObservationsError`
below :data:`MINIMUM_OBSERVATIONS_FOR_SHRINKAGE` rather than returning it,
warning about it, or flooring the intensity. Flooring was rejected: a floor is
a number invented here, and reporting it as the shrinkage intensity would
misstate to the operator how much of the risk model is assumption. Warning was
rejected: the consumer is an optimizer, which does not read warnings and
answers a singular covariance with an unbounded position along its null space.
The reasoning is recorded on
:class:`~backend.portfolio.errors.InsufficientObservationsError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from backend.portfolio.errors import (
    DegenerateReturnsError,
    InsufficientObservationsError,
    NotPositiveSemiDefiniteError,
    SingularCovarianceError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "DEFAULT_PSD_RELATIVE_TOLERANCE",
    "MINIMUM_OBSERVATIONS_FOR_COVARIANCE",
    "MINIMUM_OBSERVATIONS_FOR_SHRINKAGE",
    "ShrinkageCovariance",
    "ledoit_wolf_covariance",
    "ledoit_wolf_shrinkage_intensity",
    "sample_covariance",
]

MINIMUM_OBSERVATIONS_FOR_COVARIANCE: Final = 2
"""Rows required to form a sample covariance at all (count).

One observation has zero covariance after demeaning, which is not a risk
model. Two is enough for :func:`sample_covariance`, which makes no claim about
estimation error — but not for the shrinkage estimator, see
:data:`MINIMUM_OBSERVATIONS_FOR_SHRINKAGE`.
"""

MINIMUM_OBSERVATIONS_FOR_SHRINKAGE: Final = 3
"""Rows required for the Ledoit-Wolf intensity to be well defined (count).

At exactly two observations the analytic intensity is identically zero for any
number of assets — the module docstring derives why — so the estimator would
return the raw, near-singular sample covariance while reporting that it had
shrunk optimally. Three is the boundary of the formula's validity, **not** a
recommendation: an honest risk model needs far more, and that judgement belongs
to the caller, where it is visible.
"""

DEFAULT_PSD_RELATIVE_TOLERANCE: float = 1e-10
"""Negative-eigenvalue tolerance, as a fraction of the largest eigenvalue.

Dimensionless. The check is ``min_eigenvalue >= -tol * max(max_eigenvalue,
1.0)``: eigenvalues of a covariance matrix carry the scale of the data, so an
absolute tolerance would be meaningless across panels whose variances differ
by orders of magnitude. The ``max(..., 1.0)`` floor keeps the tolerance from
collapsing to zero for a panel whose largest eigenvalue is itself tiny.
"""


@dataclass(frozen=True, slots=True, eq=False)
class ShrinkageCovariance:
    """A Ledoit-Wolf shrinkage covariance estimate and its provenance.

    All matrices are read-only (``ndarray.flags.writeable`` is cleared) so a
    consumer cannot mutate a risk model in place and leave the recorded
    shrinkage intensity describing something else.

    Attributes:
        covariance: the shrunk estimate, shape ``(n_assets, n_assets)``,
            symmetric and positive semi-definite. Units: squared per-period
            return fraction.
        sample_covariance: the unshrunk sample covariance it was computed
            from, same shape and units. Maximum-likelihood convention —
            divisor ``n_observations``, not ``n_observations - 1`` — matching
            Ledoit & Wolf (2004) and scikit-learn. Kept so a caller can see
            what was shrunk, and so the "estimate lies between sample and
            target" relation is checkable rather than asserted in prose.
        target: the shrinkage target ``mu * I``, same shape and units.
        shrinkage_intensity: ``delta`` in ``(1 - delta) * S + delta * T``.
            Dimensionless, in ``[0, 1]``. ``0`` means the sample covariance was
            used unchanged; ``1`` means the data contributed nothing but its
            average variance.
        target_variance: ``mu = trace(S) / n_assets``, the average sample
            variance imposed on every diagonal entry by the target. Units:
            squared per-period return fraction.
        n_observations: rows in the return panel (count).
        n_assets: columns in the return panel (count).
        min_eigenvalue: smallest eigenvalue of ``covariance`` (units: squared
            per-period return fraction). Positive for any non-degenerate panel
            once ``delta > 0``, which is what makes the estimate invertible.
        max_eigenvalue: largest eigenvalue of ``covariance``, same units.
        condition_number: ``max_eigenvalue / min_eigenvalue``, dimensionless.
            ``inf`` if the estimate is singular. This is the number that
            predicts whether the optimizer will produce sane positions.
        assets: asset identifiers in column order, if the caller supplied them
            or passed a :class:`pandas.DataFrame`. ``None`` otherwise.
    """

    covariance: npt.NDArray[np.float64]
    sample_covariance: npt.NDArray[np.float64]
    target: npt.NDArray[np.float64]
    shrinkage_intensity: float
    target_variance: float
    n_observations: int
    n_assets: int
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    assets: tuple[str, ...] | None

    def summary(self) -> dict[str, float | int | str | None]:
        """Return a JSON-safe summary for operator display and run artifacts.

        Every value is a scalar, so the dashboard (P9.5) and any stored run
        record can carry the shrinkage intensity alongside the result it
        produced without re-running the estimator.

        Returns:
            Mapping with the shrinkage intensity (dimensionless), the target
            variance and eigenvalue extremes (squared per-period return
            fraction), the condition number (dimensionless), the panel shape
            (counts), and a one-line human-readable interpretation.
        """
        return {
            "shrinkage_intensity": self.shrinkage_intensity,
            "target_variance": self.target_variance,
            "n_observations": self.n_observations,
            "n_assets": self.n_assets,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "condition_number": self.condition_number,
            "units": "squared per-period simple-return fraction",
            "interpretation": (
                f"{self.shrinkage_intensity:.1%} of the estimate is the spherical "
                f"target (imposed structure); {1.0 - self.shrinkage_intensity:.1%} "
                f"is the sample covariance of {self.n_observations} observations "
                f"across {self.n_assets} assets"
            ),
        }


def _as_return_matrix(
    returns: npt.ArrayLike | pd.DataFrame,
    *,
    assets: Sequence[str] | None,
    minimum_observations: int = MINIMUM_OBSERVATIONS_FOR_COVARIANCE,
) -> tuple[npt.NDArray[np.float64], tuple[str, ...] | None]:
    """Coerce and validate a return panel, returning the matrix and asset names.

    Args:
        returns: ``(n_observations, n_assets)`` panel of simple period returns
            as fractions.
        assets: optional column labels; overrides a DataFrame's columns.
        minimum_observations: rows the caller's estimator requires. Two for a
            plain sample covariance; :data:`MINIMUM_OBSERVATIONS_FOR_SHRINKAGE`
            for anything using the Ledoit-Wolf intensity.

    Returns:
        The float64 matrix and the resolved asset labels (or ``None``).

    Raises:
        DegenerateReturnsError: if the panel is not numeric, not
            two-dimensional, has no assets, has fewer than two observations,
            contains non-finite values, or carries a label count that disagrees
            with its width.
        InsufficientObservationsError: if the panel has at least two
            observations but fewer than ``minimum_observations``.
    """
    labels: tuple[str, ...] | None = None
    try:
        if isinstance(returns, pd.DataFrame):
            labels = tuple(str(column) for column in returns.columns)
            matrix = returns.to_numpy(dtype=np.float64, copy=True)
        else:
            matrix = np.array(returns, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        msg = f"returns could not be read as a numeric 2-D panel: {exc}"
        raise DegenerateReturnsError(msg) from exc

    if assets is not None:
        labels = tuple(str(asset) for asset in assets)

    if matrix.ndim != 2:
        msg = (
            f"returns must be a 2-D (n_observations, n_assets) panel; "
            f"got an array with {matrix.ndim} dimension(s) and shape {matrix.shape}"
        )
        raise DegenerateReturnsError(msg)

    n_observations, n_assets = matrix.shape
    if n_assets < 1:
        msg = "returns must contain at least one asset column; got zero"
        raise DegenerateReturnsError(msg)
    if n_observations < MINIMUM_OBSERVATIONS_FOR_COVARIANCE:
        msg = (
            f"returns must contain at least {MINIMUM_OBSERVATIONS_FOR_COVARIANCE} "
            f"observations to estimate a covariance; got {n_observations}. A single "
            f"observation has zero sample covariance after demeaning, which is not a "
            f"risk model."
        )
        raise DegenerateReturnsError(msg)
    if n_observations < minimum_observations:
        raise InsufficientObservationsError(
            n_observations=int(n_observations),
            n_assets=int(n_assets),
            minimum_observations=minimum_observations,
        )
    if not bool(np.isfinite(matrix).all()):
        n_bad = int(np.count_nonzero(~np.isfinite(matrix)))
        msg = (
            f"returns contain {n_bad} non-finite value(s) (NaN or inf). "
            f"They are refused, not imputed: filling them would make the estimate "
            f"depend on an undeclared imputation rule. Drop or fill upstream, "
            f"where the choice is visible."
        )
        raise DegenerateReturnsError(msg)
    if labels is not None and len(labels) != n_assets:
        msg = f"assets has {len(labels)} label(s) but returns has {n_assets} column(s)"
        raise DegenerateReturnsError(msg)

    return matrix, labels


def sample_covariance(returns: npt.ArrayLike | pd.DataFrame) -> npt.NDArray[np.float64]:
    """Sample covariance of a return panel, maximum-likelihood convention.

    Columns are demeaned and the cross-product is divided by
    ``n_observations`` — **not** ``n_observations - 1``. That is the
    convention Ledoit & Wolf (2004) derive the shrinkage intensity under, and
    the one scikit-learn uses; mixing it with the unbiased ``n - 1`` form would
    break the exact identity ``Sigma = (1 - delta) * S + delta * mu * I`` that
    :func:`ledoit_wolf_covariance` reports.

    Two observations are enough here, unlike
    :func:`ledoit_wolf_covariance`: the sample covariance makes no claim about
    its own estimation error, so nothing degenerates. The matrix it returns at
    ``n_observations == 2`` has rank one and must not be inverted.

    Args:
        returns: ``(n_observations, n_assets)`` panel of simple period returns
            expressed as **fractions** (``0.01`` is a 1% return).

    Returns:
        Symmetric ``(n_assets, n_assets)`` matrix in squared per-period return
        fractions.

    Raises:
        DegenerateReturnsError: if the panel fails validation (see
            :class:`~backend.portfolio.errors.DegenerateReturnsError`).
    """
    matrix, _ = _as_return_matrix(returns, assets=None)
    return _sample_covariance(matrix)


def _sample_covariance(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the ML sample covariance of an already-validated panel."""
    centered = matrix - matrix.mean(axis=0)
    n_observations = matrix.shape[0]
    covariance = (centered.T @ centered) / float(n_observations)
    return _symmetrize(covariance)


def _symmetrize(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Average a matrix with its transpose to remove floating-point asymmetry.

    ``X.T @ X`` is symmetric in exact arithmetic but can differ in the last
    bits under floating-point summation order. ``eigvalsh`` and every
    downstream consumer assume exact symmetry, so it is imposed rather than
    hoped for.
    """
    return (matrix + matrix.T) / 2.0


def ledoit_wolf_shrinkage_intensity(returns: npt.ArrayLike | pd.DataFrame) -> float:
    """Analytic Ledoit-Wolf shrinkage intensity toward the spherical target.

    Implements the estimator of Ledoit & Wolf (2004) for the target
    ``mu * I``, ``mu = trace(S) / p``: with ``X`` the demeaned panel,

    - ``delta_dist = ||S - mu * I||_F^2 / p`` — how far the sample covariance
      sits from the target (the shrinkage has nothing to do if this is zero);
    - ``beta = (1 / (p * n)) * (mean_k ||x_k x_k'||_F^2 - ||S||_F^2)`` — the
      estimation error in ``S`` itself;
    - ``delta = min(beta, delta_dist) / delta_dist``, clipped to ``[0, 1]``.

    Requires at least :data:`MINIMUM_OBSERVATIONS_FOR_SHRINKAGE` observations.
    At exactly two, ``beta`` is identically zero and this function would return
    ``0.0`` for any panel whatsoever — a degeneracy of the formula, not a
    finding about the data. It raises there rather than returning a number that
    looks like a measurement.

    Args:
        returns: ``(n_observations, n_assets)`` panel of simple period returns
            expressed as **fractions**.

    Returns:
        The shrinkage intensity, dimensionless and in ``[0, 1]``. ``0.0`` when
        the sample covariance already equals the target (nothing to shrink
        toward) — which for a single-asset panel is always the case, since a
        ``1x1`` sample covariance *is* its own spherical target.

    Raises:
        DegenerateReturnsError: if the panel fails validation.
        InsufficientObservationsError: if the panel has fewer than
            :data:`MINIMUM_OBSERVATIONS_FOR_SHRINKAGE` observations.
    """
    matrix, _ = _as_return_matrix(
        returns, assets=None, minimum_observations=MINIMUM_OBSERVATIONS_FOR_SHRINKAGE
    )
    return _shrinkage_intensity(matrix)


def _shrinkage_intensity(matrix: npt.NDArray[np.float64]) -> float:
    """Compute the shrinkage intensity for an already-validated panel."""
    n_observations, n_assets = matrix.shape
    centered = matrix - matrix.mean(axis=0)
    n = float(n_observations)
    p = float(n_assets)

    covariance = (centered.T @ centered) / n
    mu = float(np.trace(covariance)) / p

    # ||S - mu I||_F^2 / p — distance from the sample covariance to the target.
    dispersion = float(np.sum(covariance**2)) - p * mu**2
    delta_dist = dispersion / p
    if delta_dist <= 0.0:
        # S is exactly the spherical target: there is no structure to shrink.
        return 0.0

    # beta: the average estimation error of S, from the fourth moments.
    squared = centered**2
    fourth_moment = float(np.sum(squared.T @ squared)) / n
    beta = (fourth_moment - float(np.sum(covariance**2))) / (p * n)

    intensity = min(max(beta, 0.0), delta_dist) / delta_dist
    return float(min(max(intensity, 0.0), 1.0))


def ledoit_wolf_covariance(
    returns: npt.ArrayLike | pd.DataFrame,
    *,
    assets: Sequence[str] | None = None,
    psd_relative_tolerance: float = DEFAULT_PSD_RELATIVE_TOLERANCE,
) -> ShrinkageCovariance:
    """Estimate the asset return covariance by Ledoit-Wolf shrinkage.

    Computes ``Sigma = (1 - delta) * S + delta * mu * I`` where ``S`` is the
    maximum-likelihood sample covariance, ``mu = trace(S) / n_assets``, and
    ``delta`` is the analytic optimal intensity
    (:func:`ledoit_wolf_shrinkage_intensity`). The result is symmetrized and
    its eigenvalues are checked before it is returned; a matrix that is
    indefinite, or still numerically singular after shrinkage, raises rather
    than reaching an optimizer.

    Works when ``n_assets > n_observations``, which is the normal
    cross-sectional case: the sample covariance is singular there, and the
    intensity rises toward ``1`` precisely because the data cannot support the
    off-diagonal structure.

    The one shape it cannot rescue is a panel of exactly two observations,
    where the analytic intensity is identically zero for every panel — see the
    module docstring for the derivation and
    :class:`~backend.portfolio.errors.InsufficientObservationsError` for why
    that raises rather than being floored or warned about.

    Args:
        returns: ``(n_observations, n_assets)`` panel of simple period returns
            expressed as **fractions** — ``0.01`` is a 1% return. Not percent,
            not basis points. Rows are time, columns are assets. A
            :class:`pandas.DataFrame` is accepted and its columns become the
            asset labels. Every entry must be finite.
        assets: optional asset identifiers in column order. Overrides a
            DataFrame's columns when both are present.
        psd_relative_tolerance: how negative the smallest eigenvalue may be
            before the estimate is refused, as a fraction of the largest
            eigenvalue (dimensionless). See
            :data:`DEFAULT_PSD_RELATIVE_TOLERANCE`.

    Returns:
        A :class:`ShrinkageCovariance` carrying the estimate, the sample
        covariance and target it interpolates between, the shrinkage intensity,
        and the conditioning of the result. Covariance units are squared
        per-period return fractions.

    Raises:
        DegenerateReturnsError: if the panel is not a valid two-dimensional
            finite numeric panel, or if every column is constant so that
            ``trace(S) == 0`` and the target is the zero matrix — shrinkage
            cannot regularize a panel with no variance in it, and returning the
            zero matrix would hand the optimizer a silently useless risk model.
        InsufficientObservationsError: if the panel has fewer than
            :data:`MINIMUM_OBSERVATIONS_FOR_SHRINKAGE` observations, where the
            analytic intensity is degenerate rather than merely small.
        NotPositiveSemiDefiniteError: if the computed estimate fails its
            eigenvalue check.
        SingularCovarianceError: if the estimate is numerically singular after
            shrinkage — the intensity came out at zero on a rank-deficient
            panel. Reachable at any ``n_observations`` when every demeaned
            observation is the same vector up to sign, which makes the
            estimated error in the sample covariance vanish.
        ValueError: if ``psd_relative_tolerance`` is negative or non-finite.
    """
    if not np.isfinite(psd_relative_tolerance) or psd_relative_tolerance < 0.0:
        msg = f"psd_relative_tolerance must be finite and >= 0; got {psd_relative_tolerance!r}"
        raise ValueError(msg)

    matrix, labels = _as_return_matrix(
        returns, assets=assets, minimum_observations=MINIMUM_OBSERVATIONS_FOR_SHRINKAGE
    )
    n_observations, n_assets = matrix.shape

    covariance_sample = _sample_covariance(matrix)
    target_variance = float(np.trace(covariance_sample)) / float(n_assets)
    # A constant panel is rarely *exactly* constant after demeaning: subtracting
    # a mean that is itself a rounded sum leaves residue of order eps * level,
    # so trace(S) comes out at ~1e-38 rather than 0 and a bare `<= 0` test
    # misses it. Compare against the noise floor of the demeaning instead: any
    # total variance at or below eps times the panel's mean square is rounding,
    # not risk. (eps rather than eps**2 — the strict noise floor — leaves margin
    # and only rejects panels whose relative variation is below ~1e-8, which is
    # not a return series.)
    variance_floor = float(np.finfo(np.float64).eps) * float(np.mean(matrix**2))
    if target_variance <= variance_floor:
        msg = (
            f"the panel has no usable variance: trace(S) / n_assets == "
            f"{target_variance!r}, at or below the floating-point noise floor "
            f"{variance_floor!r} of demeaning it. The shrinkage target is then the "
            f"zero matrix and no amount of shrinkage produces an invertible "
            f"covariance. Check that the returns are fractions and that the panel "
            f"is not constant."
        )
        raise DegenerateReturnsError(msg)

    intensity = _shrinkage_intensity(matrix)
    target = target_variance * np.eye(n_assets, dtype=np.float64)
    shrunk = _symmetrize((1.0 - intensity) * covariance_sample + intensity * target)

    eigenvalues = np.linalg.eigvalsh(shrunk)
    min_eigenvalue = float(eigenvalues[0])
    max_eigenvalue = float(eigenvalues[-1])
    tolerance = psd_relative_tolerance * max(max_eigenvalue, 1.0)
    if min_eigenvalue < -tolerance:
        raise NotPositiveSemiDefiniteError(min_eigenvalue=min_eigenvalue, tolerance=tolerance)

    # Numerical-rank threshold, the convention numpy.linalg.matrix_rank uses.
    # Shrinkage exists to make the estimate invertible; if it did not, the
    # optimizer must not be handed the result.
    singular_threshold = float(n_assets) * float(np.finfo(np.float64).eps) * max_eigenvalue
    if min_eigenvalue <= singular_threshold:
        raise SingularCovarianceError(
            shrinkage_intensity=intensity,
            min_eigenvalue=min_eigenvalue,
            max_eigenvalue=max_eigenvalue,
            threshold=singular_threshold,
            n_observations=int(n_observations),
            n_assets=int(n_assets),
        )

    condition_number = max_eigenvalue / min_eigenvalue

    for array in (shrunk, covariance_sample, target):
        array.flags.writeable = False

    return ShrinkageCovariance(
        covariance=shrunk,
        sample_covariance=covariance_sample,
        target=target,
        shrinkage_intensity=intensity,
        target_variance=target_variance,
        n_observations=int(n_observations),
        n_assets=int(n_assets),
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        condition_number=condition_number,
        assets=labels,
    )
