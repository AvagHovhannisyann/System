"""Monitoring: the detectors that say when the model has stopped applying (Phase 12).

Every other package in this platform answers "what is the number?". This one
answers "is the number still meaningful?", and that difference changes the
failure policy completely.

A research function that cannot produce a value emits ``NaN`` and lets the
pipeline continue — :mod:`backend.features` does exactly that, and it is the
right behaviour there, because a missing factor for one name is an ordinary
fact about coverage. A *detector* that cannot produce a value must not do the
same, because the two values a broken detector emits most easily are the two
that read as good news: ``0.0`` ("nothing has moved") and ``NaN`` (a blank cell
in a dashboard, "nothing to report"). Both are silently reassuring and both are
wrong. So the rule for this package, enforced by
:mod:`backend.monitoring.errors` rather than by convention, is: **a monitor that
cannot measure raises, and says what it could not measure and against what**.

Modules:

- :mod:`backend.monitoring.psi` (P12.2) — the Population Stability Index over a
  feature's cross-section, measured against a *frozen, named*
  :class:`~backend.monitoring.psi.ReferenceDistribution` whose bins were placed
  once at reference time. The whole design of that module exists to make the
  classic silent failure of this metric — recomputing the bins from each
  period's own sample, which drives PSI to its noise floor no matter how far the
  data has moved — unreachable through its public surface.
- :mod:`backend.monitoring.drift` (P12.2) — the reporting layer: the
  conventional threshold ladder as a configurable, self-describing object rather
  than two magic numbers, and a report that carries the I2 reproducibility stamp
  of the run that measured it *and* the identity of every reference each number
  was measured against. A PSI without a named reference is not interpretable.
- :mod:`backend.monitoring.errors` — the failure taxonomy.

Nothing in this package reads the database or manufactures a distribution. A
reference distribution is built from a sample the caller supplies, so a report
can never contain a number that came from anywhere but real data the caller
already had (invariant I3).
"""

from __future__ import annotations

from backend.monitoring.drift import (
    CONVENTIONAL_BANDS_BASIS,
    DEFAULT_MAJOR_THRESHOLD,
    DEFAULT_MODERATE_THRESHOLD,
    DriftBand,
    DriftBands,
    DriftReport,
    FeatureDrift,
    UnmeasurableFeature,
    drift_report,
    measure_feature_drift,
)
from backend.monitoring.errors import (
    DriftBandError,
    InsufficientSampleError,
    MonitoringError,
    MonitoringInputError,
    ReferenceDistributionError,
)
from backend.monitoring.psi import (
    AVAILABILITY_CATEGORIES,
    DEFAULT_BINS,
    DEFAULT_EPSILON,
    MINIMUM_BINS,
    NULL_PSI_NOISE_BUDGET,
    BinContribution,
    NanRateShift,
    PSIResult,
    ReferenceDistribution,
    minimum_sample_size,
    population_stability_index,
)

__all__ = [
    "AVAILABILITY_CATEGORIES",
    "CONVENTIONAL_BANDS_BASIS",
    "DEFAULT_BINS",
    "DEFAULT_EPSILON",
    "DEFAULT_MAJOR_THRESHOLD",
    "DEFAULT_MODERATE_THRESHOLD",
    "MINIMUM_BINS",
    "NULL_PSI_NOISE_BUDGET",
    "BinContribution",
    "DriftBand",
    "DriftBandError",
    "DriftBands",
    "DriftReport",
    "FeatureDrift",
    "InsufficientSampleError",
    "MonitoringError",
    "MonitoringInputError",
    "NanRateShift",
    "PSIResult",
    "ReferenceDistribution",
    "ReferenceDistributionError",
    "UnmeasurableFeature",
    "drift_report",
    "measure_feature_drift",
    "minimum_sample_size",
    "population_stability_index",
]
