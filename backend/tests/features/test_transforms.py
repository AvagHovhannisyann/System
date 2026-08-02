"""The cross-sectional transform pipeline: units, degeneracies, NaN policy (P5.2).

Companion to ``test_transforms_properties.py``, which states the same contract
as randomized properties. This file pins the cases where the *right answer is
known by hand*, and the cases where an implementation can be plausibly wrong in
a way no property would notice:

- **Winsorization cuts at order statistics**, not interpolated percentiles. The
  difference is one line in the implementation and it is the entire reason
  winsorization is exactly idempotent; the interpolated alternative is computed
  here and shown to keep eating into the distribution on every pass.
- **Neutralization returns residuals, not fitted values.** Both are the same
  length, both look like a factor, and a model trained on the fitted values
  would be trained on pure sector membership. So the residual is checked
  against hand-derived numbers and against the definition
  (``residual + fitted == input``).
- **NaN is excluded from every statistic and never imputed.** The tests state
  this the only way that distinguishes it from a filled value: an absent name
  must leave the *other* names' outputs exactly as they would be if it had
  never been in the array at all, and the fabricated alternatives (mean-fill,
  zero-fill) are computed alongside and shown to differ.
- **Every degeneracy in the module docstring's table** — zero dispersion,
  all-NaN, one observation, a sector of one, too few names for the beta
  regression — has its documented outcome asserted, including the ones whose
  outcome is "``NaN``, and specifically not ``0.0``".
- **Idempotence, exactly as far as it goes.** Winsorization bit-for-bit; the
  three statistical steps up to a floating-point term whose *size* is measured
  rather than assumed; and the pipeline as a whole demonstrated **not**
  idempotent, with a separate counterexample for each of the two mechanisms the
  module docstring names.

Anything involving ``1e155`` or ``1e308`` below is a regression test for a
defect this suite found in the committed code, not a hypothetical: see
``TestOverflowIsNotFabrication``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.features._stats import order_statistic_bounds
from backend.features.transforms import (
    MINIMUM_GROUP_MEMBERS_FOR_NEUTRALIZATION,
    MINIMUM_OBSERVATIONS_FOR_BETA_NEUTRALIZATION,
    MINIMUM_OBSERVATIONS_FOR_ZSCORE,
    ZERO_DISPERSION_RELATIVE_TOLERANCE,
    beta_neutralize,
    cross_sectional_zscore,
    neutralize,
    transform_cross_section,
    winsorize,
)

if TYPE_CHECKING:
    import numpy.typing as npt

EPS = float(np.finfo(np.float64).eps)
"""float64 machine epsilon, the unit every tolerance below is expressed in."""


def _floats(*values: float) -> npt.NDArray[np.float64]:
    """One date's cross-section, in whatever units the test says."""
    return np.asarray(values, dtype=np.float64)


def _sectors(*labels: int) -> npt.NDArray[np.int64]:
    """One label per security. Only equality between labels is meaningful."""
    return np.asarray(labels, dtype=np.int64)


# ==========================================================================
# winsorize
# ==========================================================================


class TestWinsorize:
    def test_clips_both_tails_to_hand_computed_order_statistics(self) -> None:
        # Eleven names, so the 1st percentile lands at sorted index
        # floor(10 * 0.01) == 0 and the 99th at ceil(10 * 0.99) == 10: with this
        # few names the default 1/99 cut is a no-op, which is worth pinning
        # because it is surprising and it is correct.
        values = _floats(-500.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 900.0)
        assert np.array_equal(winsorize(values), values)

        # At 25/75 the bounds are real interior order statistics: index
        # floor(10 * 0.25) == 2 -> 2.0 and ceil(10 * 0.75) == 8 -> 8.0.
        clipped = winsorize(values, lower_pct=25.0, upper_pct=75.0)
        assert np.array_equal(
            clipped, _floats(2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0)
        )

    def test_the_cut_points_are_members_of_the_sample(self) -> None:
        """Not interpolated between two names. This is what makes the map idempotent."""
        rng = np.random.default_rng(5)
        observed = rng.standard_normal(37) * 3.0
        lower, upper = order_statistic_bounds(observed, lower_pct=1.0, upper_pct=99.0)
        assert lower in set(observed.tolist())
        assert upper in set(observed.tolist())
        assert lower <= upper

    def test_interpolated_percentiles_would_not_be_idempotent(self) -> None:
        """The rejected alternative, computed here so the choice is not folklore.

        NumPy's default linear interpolation puts the 10th percentile of an
        already-clipped sample strictly *above* the previous bound, so each
        pass shaves more off the distribution. The order-statistic convention
        reaches a fixed point after one pass; this one does not reach one at all.
        """
        values = np.arange(20.0)
        interpolated = values.copy()
        bounds: list[float] = []
        for _ in range(3):
            low = float(np.percentile(interpolated, 10.0))
            bounds.append(low)
            interpolated = np.clip(interpolated, low, float(np.percentile(interpolated, 90.0)))
        assert bounds[0] < bounds[1] < bounds[2]

        once = winsorize(values, lower_pct=10.0, upper_pct=90.0)
        assert np.array_equal(winsorize(once, lower_pct=10.0, upper_pct=90.0), once)

    def test_nan_takes_no_part_in_the_bounds_and_survives_the_clip(self) -> None:
        with_absent = _floats(-100.0, 1.0, 2.0, 3.0, np.nan, 500.0)
        without = _floats(-100.0, 1.0, 2.0, 3.0, 500.0)
        clipped = winsorize(with_absent, lower_pct=25.0, upper_pct=75.0)
        assert np.isnan(clipped[4])
        assert np.array_equal(
            clipped[[0, 1, 2, 3, 5]], winsorize(without, lower_pct=25.0, upper_pct=75.0)
        )

    def test_a_single_observation_is_returned_unchanged(self) -> None:
        assert np.array_equal(winsorize(_floats(7.0)), _floats(7.0))
        assert np.array_equal(
            winsorize(_floats(np.nan, 7.0, np.nan)), _floats(np.nan, 7.0, np.nan), equal_nan=True
        )

    def test_preserves_rank_order(self) -> None:
        rng = np.random.default_rng(13)
        values = rng.standard_normal(200) * 10.0
        clipped = winsorize(values)
        before = np.argsort(np.argsort(values))
        after = np.argsort(np.argsort(clipped))
        # Clipping creates ties at the bounds, so ranks may compress but never
        # cross: a name above another before must not be below it after.
        assert np.all(np.diff(clipped[np.argsort(values)]) >= 0.0)
        assert before.size == after.size

    def test_the_input_array_is_not_mutated(self) -> None:
        values = _floats(-100.0, 1.0, 2.0, 3.0, 500.0)
        before = values.copy()
        winsorize(values)
        assert np.array_equal(values, before)

    def test_coerces_a_list_or_an_integer_array_to_float64(self) -> None:
        """The annotation states the contract; ``as_float_1d`` is the safety net.

        The ``type: ignore`` is the point of the test: a caller who bypasses the
        declared type still gets a ``float64`` result rather than integer
        division or a silently truncated array.
        """
        coerced = winsorize([1, 2, 3], lower_pct=0.0, upper_pct=100.0)  # type: ignore[arg-type]
        assert np.array_equal(coerced, _floats(1.0, 2.0, 3.0))
        assert winsorize(np.array([1, 2, 3])).dtype == np.float64


# ==========================================================================
# cross_sectional_zscore
# ==========================================================================


class TestCrossSectionalZscore:
    def test_produces_mean_zero_and_sample_standard_deviation_one(self) -> None:
        rng = np.random.default_rng(17)
        values = rng.standard_normal(50) * 4.0 + 11.0
        scores = cross_sectional_zscore(values)
        assert float(np.mean(scores)) == pytest.approx(0.0, abs=1e-13)
        assert float(np.std(scores, ddof=1)) == pytest.approx(1.0, rel=1e-13)

    def test_uses_the_sample_divisor_not_the_population_one(self) -> None:
        """ddof=1 is documented; ddof=0 would inflate every score by sqrt(n/(n-1))."""
        values = _floats(1.0, 2.0, 3.0, 4.0)
        scores = cross_sectional_zscore(values)
        population = (values - values.mean()) / np.std(values, ddof=0)
        assert scores == pytest.approx((values - 2.5) / np.std(values, ddof=1))
        assert not np.allclose(scores, population)

    def test_nan_is_excluded_from_the_mean_and_the_standard_deviation(self) -> None:
        with_absent = _floats(1.0, 2.0, np.nan, 3.0)
        without = _floats(1.0, 2.0, 3.0)
        scores = cross_sectional_zscore(with_absent)
        assert np.isnan(scores[2])
        assert scores[[0, 1, 3]] == pytest.approx(cross_sectional_zscore(without))
        assert scores[[0, 1, 3]] == pytest.approx(_floats(-1.0, 0.0, 1.0))

    def test_a_mean_filled_absent_name_would_change_everyone_else_and_does_not(self) -> None:
        """I3 in one assertion: imputation is not a private matter for the missing name.

        Filling a ``NaN`` with the cross-sectional mean leaves the mean alone
        but raises the observation count, which shrinks the sample standard
        deviation, which rescales *every other name's* z-score. The honest
        answer and the fabricated one are computed side by side.
        """
        values = _floats(1.0, 2.0, 100.0, np.nan)
        mean_filled = _floats(1.0, 2.0, 100.0, float(np.nanmean(values)))

        honest = cross_sectional_zscore(values)
        fabricated = cross_sectional_zscore(mean_filled)

        assert np.isnan(honest[3])
        assert not np.isnan(fabricated[3])
        assert not np.allclose(honest[:3], fabricated[:3])

    @pytest.mark.parametrize(
        ("values", "reason"),
        [
            (_floats(), "empty"),
            (_floats(np.nan, np.nan, np.nan), "all absent"),
            (_floats(4.0), "one observation"),
            (_floats(np.nan, 4.0, np.nan), "one present observation"),
        ],
    )
    def test_too_few_observations_give_all_nan(
        self, values: npt.NDArray[np.float64], reason: str
    ) -> None:
        assert reason
        scores = cross_sectional_zscore(values)
        assert scores.shape == values.shape
        assert bool(np.all(np.isnan(scores)))

    @pytest.mark.parametrize("level", [0.0, 1.0, -3.5, 1e-9, 1e9])
    def test_a_constant_cross_section_gives_nan_and_never_inf_or_zero(self, level: float) -> None:
        """The classic divide-by-zero. ``0.0`` here would be a fabricated measurement."""
        scores = cross_sectional_zscore(np.full(6, level))
        assert bool(np.all(np.isnan(scores)))
        assert not bool(np.any(np.isinf(scores)))

    def test_dispersion_below_the_relative_tolerance_is_treated_as_none(self) -> None:
        """A "constant" cross-section rarely has exactly zero standard deviation."""
        below = 1.0 + np.arange(4.0) * (ZERO_DISPERSION_RELATIVE_TOLERANCE / 100.0)
        assert float(np.std(below, ddof=1)) > 0.0
        assert bool(np.all(np.isnan(cross_sectional_zscore(below))))

        above = 1.0 + np.arange(4.0) * (ZERO_DISPERSION_RELATIVE_TOLERANCE * 100.0)
        assert not bool(np.any(np.isnan(cross_sectional_zscore(above))))

    def test_the_minimum_observation_constant_is_the_boundary_it_claims_to_be(self) -> None:
        assert MINIMUM_OBSERVATIONS_FOR_ZSCORE == 2
        one_short = _floats(*([np.nan] * 3), 5.0)
        assert bool(np.all(np.isnan(cross_sectional_zscore(one_short))))
        exactly_enough = _floats(*([np.nan] * 3), 5.0, 6.0)
        assert not bool(np.any(np.isnan(cross_sectional_zscore(exactly_enough)[[3, 4]])))


# ==========================================================================
# neutralize
# ==========================================================================


class TestNeutralize:
    def test_returns_the_residual_from_the_group_mean_not_the_fitted_value(self) -> None:
        """The defect this is guarding against produces a factor that *is* the sector.

        Fitted values and residuals have the same shape, the same units and the
        same plausible appearance in a dashboard. A model handed the fitted
        values learns sector membership and nothing else, and every check that
        does not compare against hand-derived numbers passes.
        """
        values = _floats(1.0, 3.0, 10.0, 20.0)
        sectors = _sectors(0, 0, 1, 1)
        residual = neutralize(values, groups=sectors)

        assert residual == pytest.approx(_floats(-1.0, 1.0, -5.0, 5.0))

        fitted = _floats(2.0, 2.0, 15.0, 15.0)  # the group means
        assert residual + fitted == pytest.approx(values)
        assert not np.allclose(residual, fitted)

    def test_every_usable_group_is_left_with_a_mean_of_zero(self) -> None:
        rng = np.random.default_rng(19)
        values = rng.standard_normal(60) * 2.0 + 5.0
        sectors = rng.integers(0, 7, 60).astype(np.int64)
        residual = neutralize(values, groups=sectors)
        for label in np.unique(sectors):
            members = residual[sectors == label]
            present = members[~np.isnan(members)]
            if present.size >= MINIMUM_GROUP_MEMBERS_FOR_NEUTRALIZATION:
                assert float(np.mean(present)) == pytest.approx(0.0, abs=1e-13)

    def test_only_label_equality_matters_not_label_order_or_magnitude(self) -> None:
        values = _floats(1.0, 3.0, 10.0, 20.0)
        assert np.array_equal(
            neutralize(values, groups=_sectors(0, 0, 1, 1)),
            neutralize(values, groups=_sectors(-77, -77, 4_000_000, 4_000_000)),
        )

    def test_a_group_of_one_gets_nan_rather_than_a_fabricated_zero(self) -> None:
        """Its residual would be identically zero for any input whatsoever."""
        values = _floats(1.0, 3.0, 999.0)
        residual = neutralize(values, groups=_sectors(0, 0, 1))
        assert residual[:2] == pytest.approx(_floats(-1.0, 1.0))
        assert np.isnan(residual[2])

        # The point: the lone member's "residual" does not depend on its value.
        other = neutralize(_floats(1.0, 3.0, -4.2), groups=_sectors(0, 0, 1))
        assert np.isnan(other[2])

    def test_a_group_whose_members_are_all_absent_is_all_nan(self) -> None:
        values = _floats(1.0, 3.0, np.nan, np.nan)
        residual = neutralize(values, groups=_sectors(0, 0, 1, 1))
        assert residual[:2] == pytest.approx(_floats(-1.0, 1.0))
        assert bool(np.all(np.isnan(residual[2:])))

    def test_an_absent_member_is_dropped_from_its_group_mean_not_filled(self) -> None:
        values = _floats(10.0, 20.0, np.nan)
        honest = neutralize(values, groups=_sectors(0, 0, 0))
        assert honest[:2] == pytest.approx(_floats(-5.0, 5.0))
        assert np.isnan(honest[2])

        zero_filled = neutralize(_floats(10.0, 20.0, 0.0), groups=_sectors(0, 0, 0))
        assert not np.allclose(honest[:2], zero_filled[:2])

    def test_a_group_of_exactly_the_minimum_size_is_usable(self) -> None:
        assert MINIMUM_GROUP_MEMBERS_FOR_NEUTRALIZATION == 2
        residual = neutralize(_floats(4.0, 6.0, 1.0, 2.0), groups=_sectors(0, 0, 1, 1))
        assert residual == pytest.approx(_floats(-1.0, 1.0, -0.5, 0.5))

    def test_absent_values_do_not_shrink_a_group_below_the_minimum_silently(self) -> None:
        values = _floats(4.0, 6.0, np.nan, 1.0, 2.0)
        residual = neutralize(values, groups=_sectors(0, 0, 0, 1, 1))
        assert residual[[0, 1]] == pytest.approx(_floats(-1.0, 1.0))
        values_thinner = _floats(4.0, np.nan, np.nan, 1.0, 2.0)
        assert bool(
            np.all(np.isnan(neutralize(values_thinner, groups=_sectors(0, 0, 0, 1, 1))[:3]))
        )


# ==========================================================================
# beta_neutralize
# ==========================================================================


class TestBetaNeutralize:
    def test_returns_the_regression_residual_against_hand_computed_numbers(self) -> None:
        values = _floats(1.0, 2.0, 3.0, 10.0)
        betas = _floats(0.5, 1.0, 1.5, 2.0)
        # beta_mean 1.25, value_mean 4.0; slope = 7.0 / 1.25 = 5.6.
        residual = beta_neutralize(values, betas=betas)
        assert residual == pytest.approx(_floats(1.2, -0.6, -2.4, 1.8))

        fitted = 4.0 + 5.6 * (betas - 1.25)
        assert residual + fitted == pytest.approx(values)
        assert not np.allclose(residual, fitted)

    def test_the_residual_is_uncorrelated_with_beta_and_sums_to_zero(self) -> None:
        rng = np.random.default_rng(23)
        betas = rng.standard_normal(80) * 0.3 + 1.0
        values = 2.0 + 1.7 * betas + rng.standard_normal(80) * 0.4
        residual = beta_neutralize(values, betas=betas)
        assert float(np.sum(residual)) == pytest.approx(0.0, abs=1e-11)
        assert float(np.dot(residual, betas - betas.mean())) == pytest.approx(0.0, abs=1e-11)

    def test_adding_a_multiple_of_beta_to_the_feature_leaves_the_residual_alone(self) -> None:
        """The defining property of "market exposure removed", stated as arithmetic."""
        rng = np.random.default_rng(29)
        betas = rng.standard_normal(40) * 0.4 + 1.0
        values = rng.standard_normal(40)
        assert beta_neutralize(values + 3.3 * betas, betas=betas) == pytest.approx(
            beta_neutralize(values, betas=betas), abs=1e-12
        )

    def test_a_name_missing_either_input_is_absent_from_the_fit_and_the_output(self) -> None:
        values = _floats(1.0, 2.0, 3.0, 10.0, np.nan, 4.0)
        betas = _floats(0.5, 1.0, 1.5, 2.0, 1.1, np.nan)
        residual = beta_neutralize(values, betas=betas)
        assert bool(np.all(np.isnan(residual[[4, 5]])))
        assert residual[:4] == pytest.approx(
            beta_neutralize(_floats(1.0, 2.0, 3.0, 10.0), betas=_floats(0.5, 1.0, 1.5, 2.0))
        )

    def test_two_complete_pairs_give_all_nan_rather_than_exact_zeros(self) -> None:
        """Two points determine the line, so the residuals are zero for any data."""
        assert MINIMUM_OBSERVATIONS_FOR_BETA_NEUTRALIZATION == 3
        residual = beta_neutralize(_floats(1.0, 9.0, np.nan), betas=_floats(0.5, 2.0, 1.0))
        assert bool(np.all(np.isnan(residual)))

        exactly_enough = beta_neutralize(_floats(1.0, 9.0, 4.0), betas=_floats(0.5, 2.0, 1.0))
        assert not bool(np.any(np.isnan(exactly_enough)))

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.7])
    def test_constant_betas_give_all_nan_because_the_slope_is_not_identified(
        self, level: float
    ) -> None:
        residual = beta_neutralize(_floats(1.0, 5.0, 9.0, 2.0), betas=np.full(4, level))
        assert bool(np.all(np.isnan(residual)))
        assert not bool(np.any(np.isinf(residual)))

    def test_an_absent_beta_is_not_replaced_by_the_average_beta(self) -> None:
        values = _floats(1.0, 2.0, 3.0, 10.0)
        betas = _floats(0.5, 1.0, 1.5, np.nan)
        honest = beta_neutralize(values, betas=betas)
        assert np.isnan(honest[3])
        mean_filled = beta_neutralize(values, betas=_floats(0.5, 1.0, 1.5, 1.0))
        assert not np.allclose(honest[:3], mean_filled[:3])


# ==========================================================================
# The documented degeneracy table
# ==========================================================================


class TestDegenerateCrossSections:
    """Every row of the table in ``transforms.__doc__``, asserted.

    The rule the table encodes is that caller errors raise and data conditions
    return ``NaN``. A twenty-year sweep must not abort because one date's
    smallest sector had one surviving member.
    """

    def test_every_transform_returns_an_empty_array_for_an_empty_cross_section(self) -> None:
        empty = _floats()
        labels = _sectors()
        for result in (
            winsorize(empty),
            cross_sectional_zscore(empty),
            neutralize(empty, groups=labels),
            beta_neutralize(empty, betas=empty),
            transform_cross_section(empty, groups=labels, betas=empty),
        ):
            assert result.shape == (0,)
            assert result.dtype == np.float64

    def test_every_transform_returns_all_nan_for_an_all_absent_cross_section(self) -> None:
        absent = np.full(5, np.nan)
        labels = _sectors(0, 0, 1, 1, 1)
        betas = _floats(0.8, 1.0, 1.2, 1.4, 0.9)
        for result in (
            winsorize(absent),
            cross_sectional_zscore(absent),
            neutralize(absent, groups=labels),
            beta_neutralize(absent, betas=betas),
            transform_cross_section(absent, groups=labels, betas=betas),
        ):
            assert result.shape == (5,)
            assert bool(np.all(np.isnan(result)))

    def test_no_transform_ever_returns_an_infinity(self) -> None:
        awkward = _floats(0.0, 0.0, 0.0, 1e-300, 1e300, np.nan)
        labels = _sectors(0, 0, 1, 1, 2, 2)
        betas = _floats(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        for result in (
            winsorize(awkward),
            cross_sectional_zscore(awkward),
            neutralize(awkward, groups=labels),
            beta_neutralize(awkward, betas=betas),
            transform_cross_section(awkward, groups=labels, betas=betas),
        ):
            assert not bool(np.any(np.isinf(result)))

    @pytest.mark.parametrize(
        ("bad", "expected"),
        [
            (np.array([[1.0, 2.0], [3.0, 4.0]]), "one-dimensional cross-section"),
            (np.array([1.0, np.inf, 2.0]), "infinite value"),
            (np.array([1.0, -np.inf, 2.0]), "infinite value"),
        ],
    )
    def test_caller_errors_raise_rather_than_producing_nan(
        self, bad: npt.NDArray[np.float64], expected: str
    ) -> None:
        with pytest.raises(ValueError, match=expected):
            winsorize(bad)
        with pytest.raises(ValueError, match=expected):
            cross_sectional_zscore(bad)

    def test_a_flattened_two_date_panel_is_refused_by_name(self) -> None:
        """The refusal message has to say *why*, because the symptom is invisible."""
        panel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="look-ahead leakage"):
            winsorize(panel)

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [(-1.0, 99.0), (1.0, 101.0), (99.0, 1.0), (float("nan"), 99.0), (1.0, float("inf"))],
    )
    def test_an_invalid_percentile_pair_is_a_caller_error(self, lower: float, upper: float) -> None:
        with pytest.raises(ValueError, match=r"lower_pct|upper_pct"):
            winsorize(_floats(1.0, 2.0, 3.0), lower_pct=lower, upper_pct=upper)

    def test_mismatched_lengths_are_refused_rather_than_broadcast(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            neutralize(_floats(1.0, 2.0, 3.0), groups=_sectors(0, 1))
        with pytest.raises(ValueError, match="equal length"):
            beta_neutralize(_floats(1.0, 2.0, 3.0), betas=_floats(1.0, 1.1))

    @pytest.mark.parametrize(
        ("groups", "expected"),
        [
            (np.array([0.0, 1.0, np.nan]), "whole-numbered group labels"),
            (np.array([0.0, 1.0, 4.999]), "whole-numbered group labels"),
            (np.array(["tech", "tech", "energy"]), "integer group labels"),
            (np.array([[0, 1], [1, 0]]), "one-dimensional"),
        ],
    )
    def test_unusable_group_labels_are_a_caller_error(self, groups: object, expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            neutralize(_floats(1.0, 2.0, 3.0), groups=groups)  # type: ignore[arg-type]

    def test_whole_numbered_float_labels_are_accepted(self) -> None:
        labels = np.array([2.0, 2.0], dtype=np.float64)
        residual = neutralize(_floats(1.0, 3.0), groups=labels)  # type: ignore[arg-type]
        assert residual == pytest.approx(_floats(-1.0, 1.0))


# ==========================================================================
# Overflow: the defect this suite found
# ==========================================================================


class TestOverflowIsNotFabrication:
    """Regression tests for a real defect in the committed implementation.

    ``dispersion_is_degenerate`` guarded the bottom of the ``float64`` range and
    nothing guarded the top. A cross-section of finite, admissible values whose
    spread exceeds ~``1.3e154`` overflows ``numpy``'s variance — it takes a mean
    of *squared* deviations — and the original code then divided by ``inf`` and
    returned exactly ``0.0`` for every name: "every security is precisely
    average", finite, plausible and completely fabricated. The sibling defects
    returned ``+inf``/``-inf`` from ``neutralize`` and ``beta_neutralize``,
    which the *next* transform in the pipeline would have rejected with a
    ``ValueError`` blaming a caller who passed nothing infinite.
    """

    @pytest.mark.parametrize("magnitude", [1e155, 1e200, 1e300])
    def test_a_z_score_whose_standard_deviation_overflows_is_nan_not_zero(
        self, magnitude: float
    ) -> None:
        values = _floats(1.0, 2.0, 3.0, 4.0) * magnitude
        assert bool(np.all(np.isfinite(values)))
        assert not np.isfinite(float(np.std(values, ddof=1)))

        scores = cross_sectional_zscore(values)
        assert bool(np.all(np.isnan(scores)))
        assert not bool(np.any(scores == 0.0))

    def test_a_z_score_whose_mean_overflows_is_nan(self) -> None:
        values = _floats(1.7e308, 1.6e308, 1.5e308, 1.4e308)
        assert bool(np.all(np.isfinite(values)))
        assert bool(np.all(np.isnan(cross_sectional_zscore(values))))

    def test_a_group_whose_total_overflows_gives_nan_not_inf(self) -> None:
        values = _floats(1.7e308, 1.6e308, 1.0, 2.0)
        residual = neutralize(values, groups=_sectors(0, 0, 1, 1))
        assert bool(np.all(np.isnan(residual[:2])))
        # The unaffected sector is untouched: one date's overflow is not
        # contagious across groups.
        assert residual[2:] == pytest.approx(_floats(-0.5, 0.5))

    def test_a_residual_that_overflows_gives_nan_for_that_entry_only(self) -> None:
        values = _floats(1.7e308, -1.7e308, -1.7e308, 1.0, 2.0)
        residual = neutralize(values, groups=_sectors(0, 0, 0, 1, 1))
        assert np.isnan(residual[0])
        assert bool(np.all(np.isfinite(residual[1:])))

    @pytest.mark.parametrize(
        ("values", "betas"),
        [
            (_floats(1.7e308, -1.7e308, 1e307, 5e307), _floats(0.5, 1.0, 1.5, 2.0)),
            (_floats(1.0, 2.0, 3.0, 4.0) * 1e200, _floats(1.0, 2.0, 3.5, 4.0) * 1e200),
            (_floats(1.0, 2.0, 3.0, 4.0), _floats(1e-200, 2e-200, 3.5e-200, 4e-200)),
        ],
        ids=["value-overflow", "both-huge", "betas-square-to-zero"],
    )
    def test_a_beta_fit_that_overflows_gives_all_nan_and_never_inf(
        self, values: npt.NDArray[np.float64], betas: npt.NDArray[np.float64]
    ) -> None:
        residual = beta_neutralize(values, betas=betas)
        assert bool(np.all(np.isnan(residual)))
        assert not bool(np.any(np.isinf(residual)))

    def test_the_pipeline_survives_a_date_it_cannot_standardize(self) -> None:
        """A twenty-year sweep must record "unavailable", not raise and not lie."""
        values = _floats(1.0, 2.0, 3.0, 4.0) * 1e200
        out = transform_cross_section(
            values, groups=_sectors(0, 0, 1, 1), betas=_floats(0.8, 1.2, 0.9, 1.4)
        )
        assert bool(np.all(np.isnan(out)))


# ==========================================================================
# Idempotence
# ==========================================================================


class TestIdempotence:
    """What is true, at what precision, and — for the pipeline — what is false."""

    @pytest.mark.parametrize("lower", [0.0, 1.0, 10.0, 25.0])
    def test_winsorize_is_idempotent_bit_for_bit(self, lower: float) -> None:
        rng = np.random.default_rng(31)
        values = rng.standard_normal(97) * 8.0
        values[rng.random(97) < 0.15] = np.nan

        once = winsorize(values, lower_pct=lower, upper_pct=100.0 - lower)
        current = once
        for _ in range(4):
            current = winsorize(current, lower_pct=lower, upper_pct=100.0 - lower)
            assert np.array_equal(current, once, equal_nan=True)

    def test_z_scoring_is_idempotent_up_to_a_measured_floating_point_term(self) -> None:
        rng = np.random.default_rng(37)
        values = rng.standard_normal(64) * 3.0 + 20.0
        once = cross_sectional_zscore(values)
        twice = cross_sectional_zscore(once)
        assert float(np.max(np.abs(twice - once))) < 1e-13
        assert not np.array_equal(twice, once)  # "up to floating point", not "exactly"

    def test_the_z_score_idempotence_gap_tracks_the_cross_sections_conditioning(self) -> None:
        """The docstring's claim measured, not asserted at one convenient tolerance.

        The gap is bounded by ``eps * kappa`` with ``kappa = max|x| / sigma``,
        because the first pass's own mean and standard deviation are only
        accurate to that. Quoting a bare ``eps`` would understate it by ten
        orders of magnitude at conditioning this module still accepts:
        :data:`ZERO_DISPERSION_RELATIVE_TOLERANCE` only refuses ``kappa`` above
        ``1e12``.
        """
        gaps: list[tuple[float, float]] = []
        for spread in (1e-1, 1e-4, 1e-7, 1e-11):
            values = 1.0 + _floats(0.0, 1.0, 2.0, 3.5, -1.0) * spread
            once = cross_sectional_zscore(values)
            twice = cross_sectional_zscore(once)
            kappa = float(np.max(np.abs(values))) / float(np.std(values, ddof=1))
            gap = float(np.max(np.abs(twice - once)))
            assert gap <= 32.0 * EPS * kappa
            gaps.append((kappa, gap))

        best_kappa, best_gap = gaps[0]
        worst_kappa, worst_gap = gaps[-1]
        # The dependence is real, not a conservative bound around a constant:
        # nine orders of magnitude of conditioning buy at least six orders of
        # magnitude of drift away from the fixed point. (The bound is one-sided
        # on purpose. Individual conditionings get lucky — the same sweep at a
        # spread of 1e-10 lands on an exact cancellation and drifts by one ulp —
        # so only the endpoints of a wide sweep are compared.)
        assert worst_kappa / best_kappa > 1e8
        assert worst_gap > best_gap * 1e6
        assert best_gap < 1e-13

    def test_neutralize_is_idempotent_up_to_floating_point(self) -> None:
        rng = np.random.default_rng(41)
        values = rng.standard_normal(80) * 5.0 + 2.0
        values[rng.random(80) < 0.2] = np.nan
        sectors = rng.integers(0, 6, 80).astype(np.int64)

        once = neutralize(values, groups=sectors)
        twice = neutralize(once, groups=sectors)
        assert np.array_equal(np.isnan(once), np.isnan(twice))
        present = ~np.isnan(once)
        assert float(np.max(np.abs(twice[present] - once[present]))) < 1e-12

    def test_a_group_that_lost_its_only_member_stays_nan_on_the_second_pass(self) -> None:
        """The fixed point includes the ``NaN`` pattern, or idempotence is a half-truth."""
        values = _floats(1.0, 3.0, 999.0, np.nan)
        sectors = _sectors(0, 0, 1, 1)
        once = neutralize(values, groups=sectors)
        twice = neutralize(once, groups=sectors)
        assert np.array_equal(np.isnan(once), np.isnan(twice))
        assert np.array_equal(once, twice, equal_nan=True)

    def test_beta_neutralize_is_idempotent_up_to_floating_point(self) -> None:
        rng = np.random.default_rng(43)
        betas = rng.standard_normal(70) * 0.35 + 1.0
        values = 1.0 + 2.5 * betas + rng.standard_normal(70)
        values[rng.random(70) < 0.15] = np.nan

        once = beta_neutralize(values, betas=betas)
        twice = beta_neutralize(once, betas=betas)
        present = ~np.isnan(once)
        assert np.array_equal(np.isnan(once), np.isnan(twice))
        assert float(np.max(np.abs(twice[present] - once[present]))) < 1e-12

    # ---------------------------------------------------------------- #
    # The pipeline is NOT idempotent. Both mechanisms, separately.
    # ---------------------------------------------------------------- #

    def test_the_pipeline_is_not_idempotent_because_it_re_standardizes(self) -> None:
        """Mechanism 1. Ranks survive a second pass; the values a model consumes do not.

        Neutralization shrinks the cross-sectional standard deviation below 1 —
        that shrinkage *is* the sector exposure being removed. Running the
        pipeline again z-scores the residual back up to unit dispersion,
        multiplying every value by ``1 / sigma_residual``. Nothing in the module
        can detect that this happened, which is why the docstring says
        "transform raw features, once".
        """
        rng = np.random.default_rng(47)
        values = rng.standard_normal(120) * 4.0 + 30.0
        sectors = rng.integers(0, 5, 120).astype(np.int64)

        # 0/100 makes winsorization a no-op on both passes, isolating the
        # rescaling from the (separate, benign) fact that a second winsorization
        # would also clip the residual's tails.
        once = transform_cross_section(values, groups=sectors, lower_pct=0.0, upper_pct=100.0)
        twice = transform_cross_section(once, groups=sectors, lower_pct=0.0, upper_pct=100.0)

        sigma_once = float(np.std(once, ddof=1))
        assert sigma_once < 1.0  # the sector bet has been removed
        assert not np.allclose(twice, once)
        # The second pass is the first pass rescaled by exactly 1 / sigma.
        assert twice == pytest.approx(once / sigma_once, rel=1e-9)
        assert float(np.std(twice, ddof=1)) == pytest.approx(1.0, rel=1e-9)
        # Ranks are untouched, which is precisely why this is easy to miss.
        assert np.array_equal(np.argsort(twice), np.argsort(once))

    def test_the_pipeline_is_not_idempotent_because_the_two_projections_do_not_commute(
        self,
    ) -> None:
        """Mechanism 2, isolated from rescaling by using the two steps directly.

        Sector demeaning and beta regression are both orthogonal projections,
        but onto subspaces that are not orthogonal to each other. Making a
        vector orthogonal to ``[1, beta]`` re-introduces a sector component, so
        the composition is not itself a projection and applying it twice is not
        the same as applying it once. Composing projections yields a projection
        only when they commute, and sector membership and market beta do not.
        """
        values = _floats(1.0, 4.0, 2.0, 9.0, 3.0, 7.0)
        sectors = _sectors(0, 0, 1, 1, 2, 2)
        betas = _floats(0.7, 1.6, 1.1, 0.8, 1.9, 1.2)

        sector_neutral = neutralize(values, groups=sectors)
        for label in np.unique(sectors):
            assert float(np.mean(sector_neutral[sectors == label])) == pytest.approx(0.0, abs=1e-13)

        both = beta_neutralize(sector_neutral, betas=betas)
        sector_means_after = [
            float(np.mean(both[sectors == label])) for label in np.unique(sectors)
        ]
        # Beta neutralization put a sector bet back in.
        assert max(abs(mean) for mean in sector_means_after) > 1e-3

        # So re-applying the composition is not a no-op, even though each step
        # on its own is idempotent.
        again = beta_neutralize(neutralize(both, groups=sectors), betas=betas)
        assert not np.allclose(again, both, atol=1e-9)

    def test_the_pipeline_docstring_states_the_non_idempotence(self) -> None:
        """Directive §8: the caller's obligation has to be written down, not just true."""
        doc = transform_cross_section.__doc__ or ""
        assert "Not idempotent" in doc


# ==========================================================================
# Units
# ==========================================================================


class TestUnits:
    """Directive §8: units are stated, and the arithmetic agrees with the statement."""

    @pytest.mark.parametrize(
        "function",
        [winsorize, cross_sectional_zscore, neutralize, beta_neutralize, transform_cross_section],
    )
    def test_every_public_transform_states_its_units(self, function: object) -> None:
        doc = (getattr(function, "__doc__", "") or "").lower()
        assert "unit" in doc or "dimensionless" in doc

    def test_winsorize_returns_the_inputs_units_it_clips_and_never_rescales(self) -> None:
        values = _floats(-100.0, 1.0, 2.0, 3.0, 500.0)
        in_percent = values * 100.0
        assert winsorize(in_percent, lower_pct=25.0, upper_pct=75.0) == pytest.approx(
            winsorize(values, lower_pct=25.0, upper_pct=75.0) * 100.0
        )
        # Every output value is one of the input values: no new numbers appear.
        assert set(winsorize(values).tolist()) <= set(values.tolist())

    def test_the_z_score_is_dimensionless(self) -> None:
        """Affine rescaling of the input leaves the output identical, which is the claim."""
        values = _floats(1.0, 2.0, 3.0, 7.0, -4.0)
        assert cross_sectional_zscore(values * 10_000.0 + 5.0) == pytest.approx(
            cross_sectional_zscore(values), abs=1e-12
        )

    def test_neutralize_returns_residuals_in_the_inputs_units(self) -> None:
        values = _floats(1.0, 3.0, 10.0, 20.0)
        sectors = _sectors(0, 0, 1, 1)
        assert neutralize(values * 100.0, groups=sectors) == pytest.approx(
            neutralize(values, groups=sectors) * 100.0
        )
        # A shift is absorbed by the group mean; a scale is not. That is what
        # "residuals in the input's units" means operationally.
        assert neutralize(values + 7.0, groups=sectors) == pytest.approx(
            neutralize(values, groups=sectors)
        )

    def test_beta_neutralize_returns_residuals_in_the_values_units(self) -> None:
        values = _floats(1.0, 2.0, 3.0, 10.0)
        betas = _floats(0.5, 1.0, 1.5, 2.0)
        assert beta_neutralize(values * 100.0, betas=betas) == pytest.approx(
            beta_neutralize(values, betas=betas) * 100.0
        )
        # Betas are dimensionless and only their spread matters: rescaling them
        # rescales the slope and leaves the residual, in values' units, alone.
        assert beta_neutralize(values, betas=betas * 3.0 + 1.0) == pytest.approx(
            beta_neutralize(values, betas=betas), abs=1e-12
        )

    def test_the_pipeline_output_is_dimensionless(self) -> None:
        rng = np.random.default_rng(53)
        values = rng.standard_normal(50) * 2.0 + 1.0
        sectors = rng.integers(0, 4, 50).astype(np.int64)
        assert transform_cross_section(values * 1e6, groups=sectors) == pytest.approx(
            transform_cross_section(values, groups=sectors), abs=1e-9
        )

    def test_the_output_dispersion_is_at_most_one_when_no_name_is_dropped(self) -> None:
        rng = np.random.default_rng(59)
        values = rng.standard_normal(150) * 3.0
        sectors = rng.integers(0, 5, 150).astype(np.int64)
        out = transform_cross_section(values, groups=sectors)
        assert not bool(np.any(np.isnan(out)))
        assert float(np.std(out, ddof=1)) <= 1.0

    def test_the_output_dispersion_can_exceed_one_when_a_name_is_dropped(self) -> None:
        """The convenient claim is false, so it is pinned as false rather than omitted.

        Z-scoring gives the *whole* cross-section unit dispersion. A singleton
        sector then removes one name, and the survivors' z-scores need not have
        unit dispersion among themselves. Here the dropped name carried most of
        the spread, so what is left is stretched, not shrunk. The inequality
        that does survive is the projection one, over the retained names.
        """
        values = _floats(1.0, 30.0, 2.0)
        sectors = _sectors(0, 0, 1)
        out = transform_cross_section(values, groups=sectors, lower_pct=0.0, upper_pct=100.0)

        assert np.isnan(out[2])
        assert float(np.nanstd(out, ddof=1)) > 1.0

        # The claim that does hold, on the names that survived.
        scores = cross_sectional_zscore(winsorize(values, lower_pct=0.0, upper_pct=100.0))
        retained = ~np.isnan(out)
        assert float(np.sum(out[retained] ** 2)) <= float(np.sum(scores[retained] ** 2)) + 1e-12


# ==========================================================================
# The pipeline as a whole
# ==========================================================================


class TestPipeline:
    def test_is_exactly_the_documented_composition_of_its_steps(self) -> None:
        rng = np.random.default_rng(61)
        values = rng.standard_normal(90) * 6.0 + 3.0
        values[rng.random(90) < 0.1] = np.nan
        sectors = rng.integers(0, 5, 90).astype(np.int64)
        betas = rng.standard_normal(90) * 0.3 + 1.0

        manual = beta_neutralize(
            neutralize(
                cross_sectional_zscore(winsorize(values, lower_pct=1.0, upper_pct=99.0)),
                groups=sectors,
            ),
            betas=betas,
        )
        assert np.array_equal(
            transform_cross_section(values, groups=sectors, betas=betas), manual, equal_nan=True
        )

    def test_omitting_betas_skips_beta_neutralization_entirely(self) -> None:
        rng = np.random.default_rng(67)
        values = rng.standard_normal(40)
        sectors = rng.integers(0, 3, 40).astype(np.int64)
        assert np.array_equal(
            transform_cross_section(values, groups=sectors),
            neutralize(cross_sectional_zscore(winsorize(values)), groups=sectors),
            equal_nan=True,
        )

    def test_winsorizing_before_standardizing_is_not_the_same_as_after(self) -> None:
        """The directive fixes the order; this is what the order buys.

        One outlier sets the scale for everyone if it is still present when the
        standard deviation is computed. Winsorizing first bounds its influence;
        winsorizing the z-scores afterwards bounds only the outlier itself,
        leaving every other name compressed toward zero.
        """
        # 41 names: at 2.5/97.5 the cut points are sorted[1] and sorted[39], so
        # the outlier is genuinely clipped. (At the default 1/99 they are
        # sorted[0] and sorted[40] — the min and the max — and winsorization is
        # a no-op on a cross-section this small, which is itself worth knowing.)
        values = _floats(*([1.0, 2.0, 3.0, 4.0, 5.0] * 8), 5_000.0)
        assert np.array_equal(winsorize(values), values)

        directive_order = cross_sectional_zscore(winsorize(values, lower_pct=2.5, upper_pct=97.5))
        reversed_order = winsorize(cross_sectional_zscore(values), lower_pct=2.5, upper_pct=97.5)

        assert float(np.std(directive_order, ddof=1)) == pytest.approx(1.0, rel=1e-9)
        # The outlier has been brought back into the pack rather than left to
        # define the scale...
        assert float(np.max(np.abs(directive_order))) < 2.0
        # ...whereas standardizing first compressed every ordinary name into a
        # sliver, and clipping afterwards cannot undo that.
        assert float(np.std(reversed_order, ddof=1)) < 0.01
        assert not np.allclose(directive_order, reversed_order)

    def test_absent_values_stay_absent_all_the_way_through(self) -> None:
        rng = np.random.default_rng(71)
        values = rng.standard_normal(60) * 2.0
        absent = rng.random(60) < 0.25
        values[absent] = np.nan
        sectors = rng.integers(0, 4, 60).astype(np.int64)
        out = transform_cross_section(values, groups=sectors)
        assert bool(np.all(np.isnan(out[absent])))

    def test_the_element_order_is_the_callers_security_order(self) -> None:
        values = _floats(5.0, 1.0, 4.0, 2.0, 3.0, 6.0)
        sectors = _sectors(0, 0, 0, 1, 1, 1)
        out = transform_cross_section(values, groups=sectors, lower_pct=0.0, upper_pct=100.0)
        permutation = np.array([3, 0, 5, 1, 4, 2])
        permuted = transform_cross_section(
            values[permutation], groups=sectors[permutation], lower_pct=0.0, upper_pct=100.0
        )
        assert permuted == pytest.approx(out[permutation])
