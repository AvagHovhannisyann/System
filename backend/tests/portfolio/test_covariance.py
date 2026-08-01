"""Ledoit-Wolf shrinkage covariance: properties, edge cases, units (P9.1).

The properties pinned here are the ones an optimizer downstream depends on and
cannot check for itself:

- the estimate is **symmetric and positive semi-definite**, including when
  assets outnumber observations (the normal cross-sectional case, where the
  sample covariance is singular);
- the **shrinkage intensity is in [0, 1]** and is reported, so an operator can
  see how much of the risk model is imposed structure;
- the estimate **lies between the sample covariance and the target**, which is
  what makes "shrinkage" a description of the arithmetic rather than a label;
- **perfectly correlated assets do not produce a singular result** — the case
  that breaks a raw sample covariance and silently produces enormous positions;
- a **two-observation panel is refused**, because the analytic intensity is
  identically zero there and the caller would otherwise be handed the raw,
  near-singular sample covariance under the name of a shrunk one.

Panels are generated from seeded ``numpy`` factor models rather than drawn
element-by-element by Hypothesis: the properties are about the *structure* of
the return panel (rank, correlation, aspect ratio), and Hypothesis is used to
explore that structure space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from backend.portfolio import (
    MINIMUM_OBSERVATIONS_FOR_SHRINKAGE,
    DegenerateReturnsError,
    InsufficientObservationsError,
    ShrinkageCovariance,
    SingularCovarianceError,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage_intensity,
    sample_covariance,
)
from backend.portfolio.covariance import _shrinkage_intensity
from backend.portfolio.errors import NotPositiveSemiDefiniteError

# A daily-return scale: 2% daily vol expressed as a fraction, the unit this
# module documents. Everything below stays in that unit deliberately.
_DAILY_VOL = 0.02


def _factor_returns(
    *,
    n_observations: int,
    n_assets: int,
    n_factors: int,
    seed: int,
    idiosyncratic_vol: float = _DAILY_VOL,
) -> np.ndarray:
    """Build a return panel from a low-rank factor model plus noise.

    Returns simple period returns as fractions, shape
    ``(n_observations, n_assets)``. ``n_factors=0`` gives pure idiosyncratic
    noise; a small factor count relative to ``n_assets`` gives the strongly
    correlated, low-rank structure real equity panels have.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_observations, n_assets)) * idiosyncratic_vol
    if n_factors == 0:
        return noise
    factors = rng.standard_normal((n_observations, n_factors)) * _DAILY_VOL
    loadings = rng.standard_normal((n_factors, n_assets))
    return factors @ loadings + noise


def _panel_builder(n_observations: int, n_assets: int, n_factors: int, seed: int) -> np.ndarray:
    """Adapt the factor-model generator to Hypothesis' keyword strategy form."""
    return _factor_returns(
        n_observations=n_observations,
        n_assets=n_assets,
        n_factors=min(n_factors, n_assets),
        seed=seed,
    )


_panels = st.builds(
    _panel_builder,
    n_observations=st.integers(min_value=MINIMUM_OBSERVATIONS_FOR_SHRINKAGE, max_value=120),
    n_assets=st.integers(min_value=1, max_value=60),
    n_factors=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
"""Panels of every admissible aspect ratio, from 3x1 up to 120x60."""

_wide_panels = st.builds(
    lambda n_observations, excess_assets, n_factors, seed: _panel_builder(
        n_observations=n_observations,
        n_assets=n_observations + excess_assets,
        n_factors=n_factors,
        seed=seed,
    ),
    n_observations=st.integers(min_value=MINIMUM_OBSERVATIONS_FOR_SHRINKAGE, max_value=40),
    excess_assets=st.integers(min_value=1, max_value=80),
    n_factors=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
"""Panels with strictly more assets than observations.

The cross-sectional case the estimator exists for, and the one where the
sample covariance is guaranteed singular. ``_panels`` reaches it by chance;
this strategy guarantees it on every example.
"""


# --------------------------------------------------------------------------
# Property: symmetry and positive semi-definiteness, including p > n
# --------------------------------------------------------------------------


@given(returns=_panels)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_estimate_is_symmetric_and_positive_semi_definite(returns: np.ndarray) -> None:
    """The estimate is exactly symmetric and has no meaningfully negative eigenvalue.

    ``ledoit_wolf_covariance`` raises rather than returning a non-PSD matrix,
    so reaching this assertion at all is most of the property; the explicit
    eigenvalue check guards against a tolerance that has been loosened until it
    admits anything.
    """
    estimate = ledoit_wolf_covariance(returns)
    assert np.array_equal(estimate.covariance, estimate.covariance.T)
    eigenvalues = np.linalg.eigvalsh(estimate.covariance)
    assert eigenvalues.min() >= -1e-10 * max(float(eigenvalues.max()), 1.0)
    assert estimate.min_eigenvalue == pytest.approx(float(eigenvalues.min()), abs=1e-18)


@pytest.mark.parametrize(("n_observations", "n_assets"), [(20, 50), (10, 200), (5, 6), (3, 300)])
def test_more_assets_than_observations_still_yields_an_invertible_estimate(
    n_observations: int, n_assets: int
) -> None:
    """The case that motivates shrinkage: p > n makes the sample covariance singular.

    The sample covariance has rank at most ``n_observations - 1`` here, so it
    is singular by construction. The shrunk estimate must not be.
    """
    returns = _factor_returns(
        n_observations=n_observations, n_assets=n_assets, n_factors=2, seed=11
    )
    estimate = ledoit_wolf_covariance(returns)

    sample_eigenvalues = np.linalg.eigvalsh(estimate.sample_covariance)
    assert np.linalg.matrix_rank(estimate.sample_covariance) <= n_observations - 1
    assert sample_eigenvalues.min() < 1e-12 * float(sample_eigenvalues.max())

    assert estimate.min_eigenvalue > 0.0
    assert np.isfinite(estimate.condition_number)
    # An invertible estimate is the whole point: this must not raise.
    np.linalg.inv(estimate.covariance)


@given(returns=_wide_panels)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p_greater_than_n_is_always_invertible_and_strictly_shrunk(
    returns: np.ndarray,
) -> None:
    """Property form of the same claim, over every wide aspect ratio.

    When assets outnumber observations the sample covariance is singular for
    every panel, so a zero intensity would be fatal every time. The estimator
    must choose a strictly positive intensity and the result must invert.
    """
    n_observations, n_assets = returns.shape
    assert n_assets > n_observations

    estimate = ledoit_wolf_covariance(returns)
    assert estimate.shrinkage_intensity > 0.0
    assert estimate.min_eigenvalue > 0.0
    assert np.isfinite(estimate.condition_number)
    np.linalg.inv(estimate.covariance)


# --------------------------------------------------------------------------
# Property: shrinkage intensity in [0, 1], and exposed
# --------------------------------------------------------------------------


@given(returns=_panels)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_shrinkage_intensity_is_a_valid_convex_weight(returns: np.ndarray) -> None:
    """Intensity is a convex-combination weight, so it lives in [0, 1]."""
    estimate = ledoit_wolf_covariance(returns)
    assert 0.0 <= estimate.shrinkage_intensity <= 1.0
    assert estimate.shrinkage_intensity == ledoit_wolf_shrinkage_intensity(returns)


def test_shrinkage_intensity_rises_as_observations_become_scarce() -> None:
    """Less data means more imposed structure — the operator-visible statement.

    Not a knife-edge assertion about a particular value: the point is the
    direction. A panel with a handful of observations across many names cannot
    support its own off-diagonal structure, and the estimator must say so.
    """
    long_panel = _factor_returns(n_observations=2000, n_assets=20, n_factors=3, seed=3)
    short_panel = _factor_returns(n_observations=25, n_assets=20, n_factors=3, seed=3)
    assert (
        ledoit_wolf_covariance(short_panel).shrinkage_intensity
        > ledoit_wolf_covariance(long_panel).shrinkage_intensity
    )


def test_summary_reports_the_intensity_for_an_operator() -> None:
    """The intensity must be readable from the artifact, not re-derived."""
    estimate = ledoit_wolf_covariance(
        _factor_returns(n_observations=40, n_assets=25, n_factors=2, seed=5)
    )
    summary = estimate.summary()
    assert summary["shrinkage_intensity"] == estimate.shrinkage_intensity
    assert summary["n_observations"] == 40
    assert summary["n_assets"] == 25
    assert summary["units"] == "squared per-period simple-return fraction"
    assert isinstance(summary["interpretation"], str)
    assert "imposed structure" in summary["interpretation"]


# --------------------------------------------------------------------------
# Property: the estimate lies between the sample covariance and the target
# --------------------------------------------------------------------------


@given(returns=_panels)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_estimate_lies_between_the_sample_covariance_and_the_target(
    returns: np.ndarray,
) -> None:
    """Every entry sits on the segment joining the sample entry and the target entry."""
    estimate = ledoit_wolf_covariance(returns)
    lower = np.minimum(estimate.sample_covariance, estimate.target)
    upper = np.maximum(estimate.sample_covariance, estimate.target)
    scale = max(float(np.abs(upper).max()), 1e-300)
    assert (estimate.covariance >= lower - 1e-12 * scale).all()
    assert (estimate.covariance <= upper + 1e-12 * scale).all()


@given(returns=_panels)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_the_convex_combination_identity_holds_exactly(returns: np.ndarray) -> None:
    """``Sigma == (1 - delta) * S + delta * T`` is the definition, not an approximation."""
    estimate = ledoit_wolf_covariance(returns)
    rebuilt = (
        1.0 - estimate.shrinkage_intensity
    ) * estimate.sample_covariance + estimate.shrinkage_intensity * estimate.target
    assert np.allclose(estimate.covariance, rebuilt, rtol=1e-12, atol=1e-300)


def test_the_target_is_the_average_sample_variance_on_the_diagonal() -> None:
    """The target is ``mu * I`` with ``mu = trace(S) / p`` — spherical, not diagonal-S."""
    returns = _factor_returns(n_observations=80, n_assets=12, n_factors=2, seed=17)
    estimate = ledoit_wolf_covariance(returns)
    expected_mu = float(np.trace(estimate.sample_covariance)) / 12
    assert estimate.target_variance == pytest.approx(expected_mu, rel=1e-12)
    assert np.allclose(estimate.target, expected_mu * np.eye(12))


# --------------------------------------------------------------------------
# Property: perfect correlation does not produce a singular result
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_assets", [2, 5, 30])
def test_perfectly_correlated_assets_do_not_produce_a_singular_estimate(n_assets: int) -> None:
    """Rank-1 returns are the classic singular sample covariance.

    Every asset is a scaled copy of one common factor, so the sample covariance
    has rank exactly 1 and is uninvertible for any ``n_assets > 1``. The
    shrunk estimate must be invertible, because that is what stops the
    optimizer from taking an unbounded position along the null space.
    """
    rng = np.random.default_rng(23)
    factor = rng.standard_normal(120) * _DAILY_VOL
    loadings = rng.uniform(0.5, 2.0, size=n_assets)
    returns = np.outer(factor, loadings)

    estimate = ledoit_wolf_covariance(returns)
    assert np.linalg.matrix_rank(estimate.sample_covariance) == 1
    assert estimate.shrinkage_intensity > 0.0
    assert estimate.min_eigenvalue > 0.0
    assert np.isfinite(estimate.condition_number)
    np.linalg.inv(estimate.covariance)


def test_identical_asset_columns_do_not_produce_a_singular_estimate() -> None:
    """The degenerate limit of perfect correlation: duplicated columns."""
    rng = np.random.default_rng(29)
    column = (rng.standard_normal(90) * _DAILY_VOL).reshape(-1, 1)
    returns = np.tile(column, (1, 8))

    estimate = ledoit_wolf_covariance(returns)
    assert estimate.min_eigenvalue > 0.0
    np.linalg.inv(estimate.covariance)


def test_one_constant_asset_among_others_is_regularized_not_rejected() -> None:
    """A name that did not move is real data, not a defect — only an all-zero panel is."""
    returns = _factor_returns(n_observations=60, n_assets=6, n_factors=2, seed=31)
    returns[:, 3] = 0.0
    estimate = ledoit_wolf_covariance(returns)
    assert estimate.sample_covariance[3, 3] == 0.0
    assert estimate.covariance[3, 3] > 0.0  # the target lifted it off zero
    assert estimate.min_eigenvalue > 0.0


# --------------------------------------------------------------------------
# Agreement with the reference implementation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_observations", "n_assets", "n_factors"),
    [(250, 30, 3), (60, 100, 2), (20, 5, 1), (3, 2, 1), (500, 200, 5)],
)
def test_matches_scikit_learns_ledoit_wolf(
    n_observations: int, n_assets: int, n_factors: int
) -> None:
    """The NumPy implementation must not drift from the reference formula.

    ``backend.portfolio.covariance`` re-implements the analytic intensity in
    NumPy because scikit-learn ships no type information and the backend is
    checked under ``mypy --strict``. That is a maintenance liability unless a
    divergence fails a test, which is what this is.
    """
    from sklearn.covariance import ledoit_wolf  # type: ignore[import-untyped]

    returns = _factor_returns(
        n_observations=n_observations, n_assets=n_assets, n_factors=n_factors, seed=41
    )
    reference_covariance, reference_intensity = ledoit_wolf(returns)
    estimate = ledoit_wolf_covariance(returns)

    assert estimate.shrinkage_intensity == pytest.approx(float(reference_intensity), rel=1e-10)
    assert np.allclose(estimate.covariance, reference_covariance, rtol=1e-10, atol=1e-300)


def test_sample_covariance_uses_the_maximum_likelihood_divisor() -> None:
    """Divisor is ``n``, not ``n - 1`` — the convention the shrinkage is derived under."""
    returns = _factor_returns(n_observations=50, n_assets=4, n_factors=1, seed=43)
    computed = sample_covariance(returns)
    expected = np.cov(returns, rowvar=False, bias=True)
    assert np.allclose(computed, expected, rtol=1e-12)
    unbiased = np.cov(returns, rowvar=False, bias=False)
    assert not np.allclose(computed, unbiased, rtol=1e-6)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_covariance_scales_with_the_square_of_the_return_unit() -> None:
    """Passing percent where fractions are documented inflates variance 10,000x.

    This is the directive §8 failure mode made explicit: the function cannot
    detect the mistake, so the test states the consequence in the record. It
    also pins the estimator's scale equivariance — the shrinkage intensity is
    unit-free and must not move when the panel is rescaled.
    """
    fractions = _factor_returns(n_observations=90, n_assets=10, n_factors=2, seed=47)
    percent = fractions * 100.0

    as_fractions = ledoit_wolf_covariance(fractions)
    as_percent = ledoit_wolf_covariance(percent)

    assert np.allclose(as_percent.covariance, as_fractions.covariance * 10_000.0, rtol=1e-10)
    assert as_percent.shrinkage_intensity == pytest.approx(
        as_fractions.shrinkage_intensity, rel=1e-10
    )


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def test_a_dataframe_supplies_asset_labels() -> None:
    frame = pd.DataFrame(
        _factor_returns(n_observations=40, n_assets=3, n_factors=1, seed=53),
        columns=["AAPL", "MSFT", "NVDA"],
    )
    estimate = ledoit_wolf_covariance(frame)
    assert estimate.assets == ("AAPL", "MSFT", "NVDA")


def test_explicit_assets_override_dataframe_columns() -> None:
    frame = pd.DataFrame(
        _factor_returns(n_observations=40, n_assets=2, n_factors=1, seed=59),
        columns=["a", "b"],
    )
    estimate = ledoit_wolf_covariance(frame, assets=["X", "Y"])
    assert estimate.assets == ("X", "Y")


def test_a_label_count_mismatch_is_refused() -> None:
    returns = _factor_returns(n_observations=30, n_assets=3, n_factors=1, seed=61)
    with pytest.raises(DegenerateReturnsError, match="label"):
        ledoit_wolf_covariance(returns, assets=["only", "two"])


def test_returned_matrices_are_read_only() -> None:
    """A risk model that can be mutated in place stops matching its own metadata."""
    estimate = ledoit_wolf_covariance(
        _factor_returns(n_observations=30, n_assets=4, n_factors=1, seed=67)
    )
    for matrix in (estimate.covariance, estimate.sample_covariance, estimate.target):
        with pytest.raises(ValueError, match="read-only"):
            matrix[0, 0] = 1.0


def test_the_input_panel_is_not_mutated() -> None:
    returns = _factor_returns(n_observations=30, n_assets=4, n_factors=1, seed=71)
    before = returns.copy()
    ledoit_wolf_covariance(returns)
    assert np.array_equal(returns, before)


# --------------------------------------------------------------------------
# Degenerate inputs raise rather than degrade
# --------------------------------------------------------------------------


def test_a_one_dimensional_panel_is_refused() -> None:
    with pytest.raises(DegenerateReturnsError, match="2-D"):
        ledoit_wolf_covariance(np.zeros(10))


def test_a_single_observation_is_refused() -> None:
    with pytest.raises(DegenerateReturnsError, match="at least 2 observations"):
        ledoit_wolf_covariance(np.array([[0.01, 0.02, -0.01]]))


@pytest.mark.parametrize(
    "bad",
    [
        [["a", "b"], ["c", "d"]],
        [[0.01, 0.02], [0.03]],
        [[{"AAPL": 0.01}]],
    ],
    ids=["strings", "ragged", "objects"],
)
def test_a_non_numeric_panel_is_refused_as_a_covariance_failure(bad: object) -> None:
    """NumPy's own TypeError/ValueError is translated into this package's taxonomy.

    A caller catching ``CovarianceError`` should not have to also catch whatever
    NumPy happens to raise when handed a ragged list or a column of strings.
    """
    with pytest.raises(DegenerateReturnsError, match="numeric 2-D panel"):
        ledoit_wolf_covariance(bad)  # type: ignore[arg-type]


def test_a_panel_with_no_assets_is_refused() -> None:
    with pytest.raises(DegenerateReturnsError, match="at least one asset"):
        ledoit_wolf_covariance(np.zeros((10, 0)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_returns_are_refused_not_imputed(bad: float) -> None:
    """I3-adjacent: filling a NaN would make the estimate depend on an unstated rule."""
    returns = _factor_returns(n_observations=20, n_assets=3, n_factors=1, seed=73)
    returns[4, 1] = bad
    with pytest.raises(DegenerateReturnsError, match="non-finite"):
        ledoit_wolf_covariance(returns)


@pytest.mark.parametrize("level", [0.0, 0.001, -0.02, 1.0])
def test_an_entirely_constant_panel_is_refused(level: float) -> None:
    """Zero total variance makes the target the zero matrix; shrinkage cannot help.

    Parametrized over the constant's level because a constant panel is not
    exactly constant after demeaning: subtracting a rounded mean leaves residue
    of order ``eps * level``, so ``trace(S)`` is a denormal-scale positive
    number rather than zero and a bare ``> 0`` test lets the panel through to
    the singularity check with a misleading message.
    """
    with pytest.raises(DegenerateReturnsError, match="no usable variance"):
        ledoit_wolf_covariance(np.full((30, 4), level))


@pytest.mark.parametrize("tolerance", [-1e-12, np.nan, np.inf])
def test_an_invalid_psd_tolerance_is_refused(tolerance: float) -> None:
    returns = _factor_returns(n_observations=30, n_assets=4, n_factors=1, seed=79)
    with pytest.raises(ValueError, match="psd_relative_tolerance"):
        ledoit_wolf_covariance(returns, psd_relative_tolerance=tolerance)


def test_the_psd_check_can_actually_fail() -> None:
    """The guard is exercised by a real estimate, not merely constructed.

    A rank-deficient panel produces a shrunk matrix whose smallest eigenvalue
    rounds a hair below zero. With the tolerance set to exactly zero the refusal
    fires on a genuinely computed estimate, so the guard is proven live rather
    than assumed. At the default tolerance the same matrix passes the PSD check
    and is caught one line later as singular, which is the intended division of
    labour between the two errors.
    """
    direction = np.array([0.01, -0.02, 0.005, 0.03])
    panel = np.array([direction if row % 2 == 0 else -direction for row in range(6)])

    with pytest.raises(NotPositiveSemiDefiniteError) as raised:
        ledoit_wolf_covariance(panel, psd_relative_tolerance=0.0)

    error = raised.value
    assert error.min_eigenvalue < 0.0
    assert error.tolerance == 0.0
    assert "not positive semi-definite" in str(error)

    # The same panel at the default tolerance: PSD passes, singularity catches it.
    with pytest.raises(SingularCovarianceError):
        ledoit_wolf_covariance(panel)


def test_shrinkage_intensity_is_zero_when_the_sample_covariance_is_already_spherical() -> None:
    """No structure to shrink toward: orthogonal columns of equal variance.

    The two demeaned columns are orthogonal and have equal norm, so ``S`` is
    exactly ``mu * I`` — it already *is* the target. The distance to the target
    is zero and the intensity is defined as zero rather than computed as
    ``0 / 0``. The estimate is still invertible, so nothing downstream breaks:
    a zero intensity is only dangerous when the sample covariance is
    rank-deficient, which is a separate condition and separately guarded.
    """
    root_three = np.sqrt(3.0)
    returns = np.array([[root_three, 1.0], [-root_three, 1.0], [0.0, -2.0]]) * 0.01

    assert np.allclose(sample_covariance(returns), 2e-4 * np.eye(2), atol=1e-18)
    assert ledoit_wolf_shrinkage_intensity(returns) == 0.0

    estimate = ledoit_wolf_covariance(returns)
    assert isinstance(estimate, ShrinkageCovariance)
    assert np.allclose(estimate.covariance, estimate.sample_covariance)
    assert estimate.min_eigenvalue > 0.0


# --------------------------------------------------------------------------
# The two-observation degeneracy
#
# At n == 2 the demeaned rows are exact negatives, so every observation's outer
# product equals the sample covariance, so Ledoit-Wolf's estimate of the error
# in that sample covariance is identically zero, so the analytic intensity is
# exactly zero — for every panel, whatever its data. The estimator would then
# hand back the raw rank-1 sample covariance while reporting that it had shrunk
# optimally. It raises instead. These tests pin both halves: the mechanism, so a
# future reader can see the guard is not superstition, and the refusal.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_assets", [1, 2, 5, 300])
def test_the_analytic_intensity_is_identically_zero_at_two_observations(n_assets: int) -> None:
    """The mechanism, measured on the unguarded internal function.

    This is the finding the guard exists for. If someone removes the guard
    believing two observations are merely "a bit thin", this test still states
    what actually happens: zero shrinkage, at every width, exactly.
    """
    rng = np.random.default_rng(101)
    panel = rng.standard_normal((2, n_assets)) * _DAILY_VOL
    assert _shrinkage_intensity(panel) == 0.0


def test_scikit_learn_has_the_same_two_observation_degeneracy() -> None:
    """It is the estimator, not this implementation.

    Pinned against the reference so the refusal cannot be mistaken for a local
    bug that a future rewrite might "fix" by matching sklearn more closely —
    sklearn returns zero here too.
    """
    # The `import-untyped` ignore lives on the first import of this module, in
    # `test_matches_scikit_learns_ledoit_wolf`; repeating it here is flagged as
    # a redundant ignore under `mypy --strict`.
    from sklearn.covariance import ledoit_wolf

    rng = np.random.default_rng(103)
    panel = rng.standard_normal((2, 8)) * _DAILY_VOL
    _, reference_intensity = ledoit_wolf(panel)
    # sklearn does not clamp, so its raw value lands on either side of zero by a
    # rounding error — "less than no shrinkage". Ours clamps to exactly 0.0.
    # Either way the shrinkage applied is nil, which is the point.
    assert abs(float(reference_intensity)) < 1e-12
    assert _shrinkage_intensity(panel) == 0.0


@pytest.mark.parametrize("n_assets", [1, 2, 300])
def test_two_observations_are_refused_by_the_covariance_estimator(n_assets: int) -> None:
    """The decision: raise. Not warn, not floor, not return the sample covariance.

    ``n_assets=1`` is included deliberately. A 1x1 sample covariance is
    invertible, so the downstream singularity check does *not* catch it, and
    the caller would receive an estimate whose reported shrinkage intensity of
    zero is an artifact of the formula rather than a fact about the data. The
    refusal is at the estimator's domain boundary for that reason.
    """
    rng = np.random.default_rng(107)
    panel = rng.standard_normal((2, n_assets)) * _DAILY_VOL

    with pytest.raises(InsufficientObservationsError) as raised:
        ledoit_wolf_covariance(panel)

    error = raised.value
    assert error.n_observations == 2
    assert error.n_assets == n_assets
    assert error.minimum_observations == MINIMUM_OBSERVATIONS_FOR_SHRINKAGE
    # The message must name the mechanism, not just the count.
    assert "identically zero" in str(error)


def test_two_observations_are_refused_by_the_intensity_function_too() -> None:
    """One rule, both entry points: the public intensity must not return the 0.0 either."""
    rng = np.random.default_rng(109)
    with pytest.raises(InsufficientObservationsError):
        ledoit_wolf_shrinkage_intensity(rng.standard_normal((2, 4)) * _DAILY_VOL)


def test_the_plain_sample_covariance_still_accepts_two_observations() -> None:
    """The minimum belongs to the shrinkage formula, not to covariance as such.

    ``sample_covariance`` claims nothing about its own estimation error, so
    nothing degenerates. It returns an honest rank-1 matrix, and the rank is
    asserted here so the caller's obligation not to invert it is on the record.
    """
    returns = np.array([[0.01, -0.02, 0.005], [-0.01, 0.03, 0.001]])
    covariance = sample_covariance(returns)
    assert covariance.shape == (3, 3)
    assert np.linalg.matrix_rank(covariance) == 1


def test_three_observations_produce_a_genuine_positive_intensity() -> None:
    """The boundary is exactly where the formula becomes well defined, not one past it."""
    rng = np.random.default_rng(113)
    panel = rng.standard_normal((MINIMUM_OBSERVATIONS_FOR_SHRINKAGE, 6)) * _DAILY_VOL
    estimate = ledoit_wolf_covariance(panel)
    assert estimate.shrinkage_intensity > 0.0
    assert estimate.min_eigenvalue > 0.0


# --------------------------------------------------------------------------
# The singularity backstop is still live
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_observations", [4, 6, 50])
def test_a_sign_alternating_panel_is_refused_as_singular(n_observations: int) -> None:
    """Zero intensity is reachable above the two-observation guard, and is caught.

    ``beta`` is an average of ``|| x_k x_k' - S ||_F^2``, so it vanishes
    whenever every demeaned observation is the same vector up to sign — at any
    ``n``, not only at 2. Such a panel has a rank-1 sample covariance and a zero
    intensity, and must not reach an optimizer. This proves the second guard is
    not dead code left over from the two-observation case.
    """
    direction = np.array([0.01, -0.02, 0.005, 0.03])
    panel = np.array([direction if row % 2 == 0 else -direction for row in range(n_observations)])

    with pytest.raises(SingularCovarianceError) as raised:
        ledoit_wolf_covariance(panel)
    assert raised.value.n_observations == n_observations
    assert raised.value.shrinkage_intensity == pytest.approx(0.0, abs=1e-12)
