"""Synthetic fixtures for the live-versus-expected suite. Every number is constructed.

**Invariant I3, restated for this module.** No live track record exists in this
repository — blocker B2, no IBKR paper credentials — and no CPCV run has been
executed against real data either. So nothing here may resemble one. Every path
matrix comes from ``numpy.random.default_rng`` with an explicit seed, every
artefact identifier begins ``FIXTURE_``, every live window declares a ``source``
that says it is synthetic, and the reproducibility stamps carry
``FIXTURE_DATA_VERSION``, which states in the payload that it is not a data
version.

A halt or a continue produced from these fixtures is a statement about the
arithmetic of :mod:`backend.monitoring.expectation`. It is never a statement
about a strategy.

The live-window helper builds a series with an **exactly** known statistic —
a return series is shifted and rescaled so its mean and sample standard
deviation are what the caller asked for — so a test can place a live value a
stated distance outside a band edge and assert on that distance, rather than
drawing until something happens to fall outside.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.monitoring.expectation import (
    CostTreatment,
    CPCVArtefactRef,
    ExpectationBand,
    HaltPolicy,
    LiveWindow,
    WindowStatistic,
    path_digest,
)
from backend.tests.monitoring.fixtures import FIXTURE_DATA_VERSION, fixture_stamp

if TYPE_CHECKING:
    from backend.backtest.metrics import FloatArray

FIXTURE_ARTEFACT_ID: Final = "FIXTURE_cpcv_run_not_a_real_backtest_B2"
"""Stands where a run artefact id would, and says it is not one (I3)."""

FIXTURE_LIVE_SOURCE: Final = (
    "FIXTURE synthetic rng series; no paper account exists in this repository (B2)"
)
"""Stands where the paper-account reconciliation run would, and says it is not one."""

FIXTURE_TRIALS: Final = 37
"""A stand-in trial count. Real ones come from TESTING_LEDGER.md."""

DEFAULT_N_PATHS: Final = 5
"""``C(N-1, k-1)`` for the fixture CPCV shape ``N=6, k=2``."""

DEFAULT_N_OBSERVATIONS: Final = 1260
"""Five years of daily periods — enough that the default policy's tail mass resolves."""

FIXTURE_DAILY_SD: Final = 0.01
"""Per-period standard deviation of the fixture return paths, as a fraction."""

FIXTURE_DAILY_MEAN: Final = 0.0006
"""Per-period mean of the fixture return paths, as a fraction."""

LIVE_AS_OF: Final = dt.date(2026, 8, 1)
"""Last observation date of the fixture live windows."""

DECISION_AT: Final = dt.datetime(2026, 8, 3, 12, 0, tzinfo=dt.UTC)
"""Decision instant used by the fixture decisions: two days after ``LIVE_AS_OF``."""


def synthetic_paths(
    *,
    seed: int = 4242,
    n_paths: int = DEFAULT_N_PATHS,
    n_observations: int = DEFAULT_N_OBSERVATIONS,
    mean: float = FIXTURE_DAILY_MEAN,
    sd: float = FIXTURE_DAILY_SD,
) -> FloatArray:
    """Draw a fixture CPCV path matrix.

    Args:
        seed: generator seed.
        n_paths: rows (count).
        n_observations: columns, in observation periods (count).
        mean: per-period mean return, as a fraction.
        sd: per-period standard deviation, as a fraction.

    Returns:
        A ``(n_paths, n_observations)`` ``float64`` array of synthetic
        net-of-cost per-period returns.
    """
    draws = np.random.default_rng(seed).standard_normal((n_paths, n_observations))
    return np.asarray(draws * sd + mean, dtype=np.float64)


def fixture_artefact(
    paths: FloatArray,
    *,
    artefact_id: str = FIXTURE_ARTEFACT_ID,
    trials: int = FIXTURE_TRIALS,
    cost_treatment: CostTreatment = CostTreatment.NET_MODELLED,
    periods_per_year: float = 252.0,
) -> CPCVArtefactRef:
    """Build the I2 identity of a fixture CPCV artefact for ``paths``.

    Args:
        paths: the path matrix the band will be cut from.
        artefact_id: fixture artefact identifier.
        trials: selection search size to record on the band.
        cost_treatment: how the paths handle costs.
        periods_per_year: annualisation factor of the paths' periodicity.

    Returns:
        A :class:`~backend.monitoring.expectation.CPCVArtefactRef`.
    """
    n_paths, n_observations = paths.shape
    return CPCVArtefactRef(
        artefact_id=artefact_id,
        stamp=fixture_stamp(),
        path_digest=path_digest(paths),
        n_paths=int(n_paths),
        n_observations=int(n_observations),
        n_groups=6,
        n_test_groups=2,
        trials=trials,
        cost_treatment=cost_treatment,
        periods_per_year=periods_per_year,
    )


def fixture_band(
    *,
    paths: FloatArray | None = None,
    policy: HaltPolicy | None = None,
    statistic: WindowStatistic = WindowStatistic.SHARPE,
    trials: int = FIXTURE_TRIALS,
    cost_treatment: CostTreatment = CostTreatment.NET_MODELLED,
) -> ExpectationBand:
    """Cut a fixture expectation band.

    Args:
        paths: the path matrix. Defaults to :func:`synthetic_paths`.
        policy: the halt policy. Defaults to the module default.
        statistic: which statistic to band.
        trials: selection search size recorded on the artefact.
        cost_treatment: cost handling declared by the paths.

    Returns:
        An :class:`~backend.monitoring.expectation.ExpectationBand`.
    """
    matrix = synthetic_paths() if paths is None else paths
    return ExpectationBand.from_paths(
        matrix,
        artefact=fixture_artefact(matrix, trials=trials, cost_treatment=cost_treatment),
        policy=policy,
        statistic=statistic,
    )


def series_with_sharpe(
    target_sharpe: float,
    *,
    n_periods: int,
    seed: int = 909,
    scale: float = FIXTURE_DAILY_SD,
) -> FloatArray:
    """Build a return series whose per-period Sharpe ratio is exactly ``target_sharpe``.

    The base draws are centred, rescaled to sample standard deviation ``scale``
    (``ddof=1``, matching :func:`backend.backtest.metrics.sharpe_ratio`) and
    shifted to mean ``target_sharpe * scale``. The resulting Sharpe is exact to
    floating point, which is what lets a test place a live value a *stated*
    distance outside a band edge instead of drawing until something falls out.

    Args:
        target_sharpe: the per-period Sharpe ratio the series must have,
            dimensionless. Assumes a zero risk-free rate.
        n_periods: length of the series (count). At least 2.
        seed: generator seed for the base draws.
        scale: the series' sample standard deviation, as a fraction.

    Returns:
        A one-dimensional ``float64`` array of per-period returns.
    """
    base = np.random.default_rng(seed).standard_normal(n_periods)
    centred = base - base.mean()
    unit = centred / centred.std(ddof=1)
    return np.asarray(unit * scale + target_sharpe * scale, dtype=np.float64)


def live_window(
    returns: FloatArray,
    *,
    as_of: dt.date = LIVE_AS_OF,
    cost_treatment: CostTreatment = CostTreatment.NET_REALISED,
    source: str = FIXTURE_LIVE_SOURCE,
) -> LiveWindow:
    """Wrap a synthetic return series as a live evaluation window.

    Args:
        returns: the per-period returns.
        as_of: date of the last observation.
        cost_treatment: how costs were handled.
        source: where the returns came from — says "fixture" by default (I3).

    Returns:
        A :class:`~backend.monitoring.expectation.LiveWindow`.
    """
    return LiveWindow(returns=returns, as_of=as_of, cost_treatment=cost_treatment, source=source)


def pooled_window_sharpes(paths: FloatArray, *, window: int) -> FloatArray:
    """Recompute the pooled window Sharpe ratios independently of the module under test.

    A deliberate second transcription: ``mean / std(ddof=1)`` over every
    contiguous window of every path, written with numpy here rather than by
    calling :mod:`backend.monitoring.expectation`. A band asserted against its
    own implementation asserts nothing; asserted against this it asserts that the
    edges really are quantiles of the window distribution.

    Args:
        paths: the ``(n_paths, n_observations)`` path matrix.
        window: window length in observation periods.

    Returns:
        A one-dimensional array of every window's per-period Sharpe ratio.
    """
    values: list[float] = []
    for row in paths:
        for start in range(row.size - window + 1):
            piece = row[start : start + window]
            values.append(float(piece.mean() / piece.std(ddof=1)))
    return np.asarray(values, dtype=np.float64)


assert FIXTURE_DATA_VERSION.startswith("FIXTURE"), "fixtures must announce themselves (I3)"
