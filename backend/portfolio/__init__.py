"""Portfolio construction: risk model, and (P9.2) the optimizer above it.

Public surface of ``backend.portfolio``.

- :mod:`backend.portfolio.covariance` — Ledoit-Wolf shrinkage estimation of
  the asset return covariance (P9.1). Sample covariance is singular whenever
  assets outnumber observations, which cross-sectionally is always; an
  optimizer that inverts a near-singular matrix loads onto its noise
  eigenvectors and returns enormous offsetting positions. Shrinkage toward a
  spherical target fixes the conditioning, and the intensity it chose is
  reported so an operator can see how much of the risk model is imposed
  structure rather than estimated data.
- :mod:`backend.portfolio.errors` — the failure taxonomy. Degenerate inputs
  raise; nothing here degrades quietly into a plausible-looking matrix. In
  particular a panel of two observations is refused outright, because the
  Ledoit-Wolf intensity is identically zero there and the caller would
  otherwise receive the raw, near-singular sample covariance while being told
  it had been shrunk.

**Units.** Return panels are simple period returns as *fractions* (``0.01``
is 1%), so covariances are in squared per-period return fractions. Nothing in
this package annualizes (directive §8: state the units, never infer them).
"""

from __future__ import annotations

from backend.portfolio.covariance import (
    DEFAULT_PSD_RELATIVE_TOLERANCE,
    MINIMUM_OBSERVATIONS_FOR_COVARIANCE,
    MINIMUM_OBSERVATIONS_FOR_SHRINKAGE,
    ShrinkageCovariance,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage_intensity,
    sample_covariance,
)
from backend.portfolio.errors import (
    CovarianceError,
    DegenerateReturnsError,
    InsufficientObservationsError,
    NotPositiveSemiDefiniteError,
    PortfolioError,
    SingularCovarianceError,
)

__all__ = [
    "DEFAULT_PSD_RELATIVE_TOLERANCE",
    "MINIMUM_OBSERVATIONS_FOR_COVARIANCE",
    "MINIMUM_OBSERVATIONS_FOR_SHRINKAGE",
    "CovarianceError",
    "DegenerateReturnsError",
    "InsufficientObservationsError",
    "NotPositiveSemiDefiniteError",
    "PortfolioError",
    "ShrinkageCovariance",
    "SingularCovarianceError",
    "ledoit_wolf_covariance",
    "ledoit_wolf_shrinkage_intensity",
    "sample_covariance",
]
