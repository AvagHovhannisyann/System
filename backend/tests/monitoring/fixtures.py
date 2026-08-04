"""Synthetic fixtures for the monitoring suite. Every number here is constructed.

**Invariant I3.** No live feature data exists in this repository (blocker B1),
so nothing in this package may look like a measurement of one. Every array
below comes from ``numpy.random.default_rng`` with an explicit seed or from
``numpy.arange``; every reference distribution is named ``FIXTURE_…``, is
denominated in :data:`FIXTURE_UNITS` — a units string that is not a unit — and
carries a stamp whose ``data_version`` is
:data:`FIXTURE_DATA_VERSION`, which states in the payload that it is not a data
version. A PSI computed against any of these is a statement about the
arithmetic, never about a factor.

The distributions are chosen so the right answer is derivable rather than
observed:

* a standard normal, whose quantile bins are equally populated by construction,
  so an emptied bin's contribution is exactly
  ``(epsilon - 1/B) * ln(epsilon * B)`` and can be asserted to the bit;
* a uniform ramp (:func:`equal_decile_reference`), where the reference bin
  counts are *exactly* equal rather than approximately, which is what makes the
  empty-bin arithmetic of :mod:`backend.monitoring.psi` §2 checkable against a
  closed form instead of against a previously observed number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from backend.monitoring.psi import DEFAULT_BINS, ReferenceDistribution
from backend.tracking.stamp import ReproducibilityStamp

if TYPE_CHECKING:
    from backend.monitoring.psi import FloatArray

FIXTURE_DATA_VERSION: Final = "FIXTURE-synthetic-rng-not-a-data-version-B1"
"""Stands where a real data version would, and says it is not one (I3)."""

FIXTURE_UNITS: Final = "fixture units (synthetic rng output; not a feature's units)"
"""Stands where a feature's declared units would, and says it is not one."""

FIXTURE_COMMIT: Final = "0" * 40
FIXTURE_CONFIG_HASH: Final = "0" * 64

REFERENCE_SEED: Final = 20240101
"""Seed for the training-period fixture sample. Fixed so bins are reproducible."""

REFERENCE_SIZE: Final = 20_000
"""Observations in the fixture training sample (count)."""

OBSERVED_SIZE: Final = 2_000
"""Observations in a fixture live cross-section (count).

Comfortably above ``minimum_sample_size(10) == 360`` so that a refusal in a
detection test means the detector refused, not that the fixture was thin.
"""


def fixture_stamp(*, seed: int = 0) -> ReproducibilityStamp:
    """Return an I2 stamp whose data version announces itself as a fixture.

    Args:
        seed: the seed recorded on the stamp.

    Returns:
        A valid :class:`~backend.tracking.stamp.ReproducibilityStamp`.
    """
    return ReproducibilityStamp(
        git_commit=FIXTURE_COMMIT,
        git_dirty=False,
        data_version=FIXTURE_DATA_VERSION,
        config_hash=FIXTURE_CONFIG_HASH,
        seed=seed,
    )


def gaussian_sample(
    *, seed: int, size: int = OBSERVED_SIZE, shift: float = 0.0, scale: float = 1.0
) -> FloatArray:
    """Draw ``size`` values from ``shift + scale * N(0, 1)``.

    Args:
        seed: generator seed.
        size: number of values (count).
        shift: location added to every draw, in fixture units.
        scale: multiplier applied before the shift (dimensionless).

    Returns:
        A one-dimensional ``float64`` array.
    """
    draws = np.random.default_rng(seed).standard_normal(size)
    return np.asarray(draws * scale + shift, dtype=np.float64)


def gaussian_reference(
    *,
    feature: str = "FIXTURE_gaussian_factor",
    reference_id: str = "FIXTURE_train_window",
    n_bins: int = DEFAULT_BINS,
    seed: int = REFERENCE_SEED,
    size: int = REFERENCE_SIZE,
    absent: int = 0,
) -> ReferenceDistribution:
    """Build a fixture reference from a standard-normal training sample.

    Args:
        feature: fixture feature name.
        reference_id: fixture reference identifier.
        n_bins: quantile bins to cut (count).
        seed: generator seed for the training sample.
        size: training observations, before ``absent`` are blanked (count).
        absent: how many of them to replace with ``NaN``, so the reference
            records a non-zero availability rate to compare against.

    Returns:
        A frozen :class:`~backend.monitoring.psi.ReferenceDistribution`.
    """
    values = gaussian_sample(seed=seed, size=size)
    if absent:
        values = values.copy()
        values[:absent] = np.nan
    return ReferenceDistribution.from_sample(
        values,
        feature=feature,
        units=FIXTURE_UNITS,
        reference_id=reference_id,
        stamp=fixture_stamp(),
        n_bins=n_bins,
    )


def equal_decile_reference(
    *, feature: str = "FIXTURE_uniform_ramp", reference_id: str = "FIXTURE_equal_deciles"
) -> ReferenceDistribution:
    """Build a reference whose ten bins hold *exactly* 2,000 observations each.

    ``arange(20_000)`` has no repeated value, so its deciles fall strictly
    between observations and every bin count is exactly ``2_000``. That makes
    every reference fraction exactly ``0.1`` and the empty-bin arithmetic of
    :mod:`backend.monitoring.psi` §2 checkable against a closed form.

    Returns:
        A frozen reference with ``bin_counts == (2000,) * 10``.
    """
    return ReferenceDistribution.from_sample(
        np.arange(REFERENCE_SIZE, dtype=np.float64),
        feature=feature,
        units=FIXTURE_UNITS,
        reference_id=reference_id,
        stamp=fixture_stamp(),
        n_bins=DEFAULT_BINS,
    )


def with_absent(values: FloatArray, *, absent: int) -> FloatArray:
    """Return a copy of ``values`` with its first ``absent`` entries set to ``NaN``.

    Args:
        values: the cross-section to blank into.
        absent: how many leading entries become "not available" (count).

    Returns:
        A fresh array; ``values`` is not modified.
    """
    blanked = values.copy()
    blanked[:absent] = np.nan
    return blanked
