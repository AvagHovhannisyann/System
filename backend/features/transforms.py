"""The cross-sectional feature transform pipeline (P5.2).

Directive §5 Phase 5 fixes the order:

::

    winsorize 1/99 → cross-sectional z-score → sector neutralize → optional beta neutralize

Each step answers a specific way a raw factor lies to a model.

**Winsorization** bounds the influence of the tails. A raw book-to-price of 400
— a real number produced by a real company with almost no market value — is not
four hundred times as informative as a book-to-price of 1. Left alone it
dominates the standardization that follows and, downstream, the split points of
a tree. Clipping to the 1st and 99th percentiles keeps the name and its rank
while removing its leverage over everything else.

**Cross-sectional z-scoring** makes features comparable to each other and
comparable to themselves across dates. Momentum in log-return units and
book-to-price as a ratio cannot be combined until both are expressed in
cross-sectional standard deviations; and a factor whose raw dispersion doubles
in a volatile month would otherwise silently double its weight in the model.

**Sector neutralization** removes the part of a factor that is a bet on
industries. Un-neutralized value in 2000 is a short technology position wearing
a factor's name, and it will be scored as stock selection when it was sector
timing. The residual from a within-sector demeaning is what is left after the
sector call is taken out.

**Beta neutralization** does the same for market exposure: low-volatility and
quality factors carry a persistent negative beta, and a backtest of them is
partly a backtest of being short the market. Optional, because for several
factors it removes more signal than exposure, and that is a modelling judgement
that belongs to the caller where it is visible.

--------------------------------------------------------------------------
Units
--------------------------------------------------------------------------

- :func:`winsorize` returns values in the **input's units**. It clips; it never
  rescales.
- :func:`cross_sectional_zscore` returns a **dimensionless** value: the input's
  units cancel in ``(x - mean) / std``. The number is in cross-sectional
  standard deviations of that date.
- :func:`neutralize` and :func:`beta_neutralize` return **regression residuals,
  in the units of their input**. Applied after standardization — the pipeline
  order — that means dimensionless z-score units; applied to a raw feature it
  means the raw feature's units. Neither function rescales, so the residual of
  a unit-variance input has variance *below* one, by exactly the fraction of
  cross-sectional variance the sectors or the market explained.
- :func:`transform_cross_section` therefore returns a **dimensionless** array
  whose cross-sectional standard deviation is at most 1 and generally less. It
  is deliberately not re-standardized at the end: the shrinkage *is* the
  information about how much of the factor was a sector or market bet, and
  re-scaling it to unit variance would erase that and re-inflate the residual
  noise of the names in small sectors.

--------------------------------------------------------------------------
The cross-sectional contract: no cross-date leakage
--------------------------------------------------------------------------

Every function here operates on **one date's cross-section**: one element per
security, all elements from the same date. This is enforced structurally rather
than by convention — the functions take one-dimensional arrays and no date,
timestamp, index, session, or configuration; they hold no module state, read no
files, and call nothing that could reach another date's data. There is no
expression inside this module that could refer to a value from a different date,
because no such value is reachable from the arguments.

That is the whole defence, and it is why the transforms take arrays rather than
a panel: a percentile or a mean computed over a panel that happens to span two
dates uses the second date's data to transform the first, which is look-ahead
bias with no visible symptom — the backtest simply improves. Two-dimensional
input is refused for this reason (see
:func:`backend.features._stats.as_float_1d`). Callers holding a panel iterate
over dates and call these functions per date.

``backend/tests/features/test_transforms_properties.py`` asserts the property
behaviourally as well: perturbing one date's values arbitrarily leaves every
other date's outputs bit-for-bit identical.

--------------------------------------------------------------------------
Missing values
--------------------------------------------------------------------------

``NaN`` means **not available**. It is excluded from every statistic — every
percentile, mean, standard deviation and regression fit here is computed over
the present values only — and it propagates: a name with no input value has no
output value. It is **never imputed**, not with the cross-sectional mean, not
with zero, not with the sector average. Under directive invariant I3 a filled
``NaN`` is fabricated data: downstream, in the feature matrix, it is
indistinguishable from a measurement, and "this stock's book-to-price is exactly
average" is a claim no one made.

``+inf`` and ``-inf`` are refused with :class:`ValueError` rather than clipped;
:func:`backend.features._stats.reject_infinities` explains why.

--------------------------------------------------------------------------
Degenerate cross-sections
--------------------------------------------------------------------------

Each degeneracy has a defined outcome. The rule separating them is the one in
:mod:`backend.features._stats`: **caller errors raise, data conditions return
NaN.** A twenty-year backtest must not abort because one date's smallest sector
had a single surviving member; it must record that the feature is unavailable
for that name on that date.

======================================  ===================================
Condition                               Result
======================================  ===================================
All values ``NaN``                      All ``NaN`` (every transform)
Empty cross-section                     Empty array (every transform)
Single observation, winsorize           Returned unchanged (bounds coincide)
Fewer than 2 observations, z-score      All ``NaN``
Zero cross-sectional dispersion         All ``NaN`` (never ``inf``)
Sector with < 2 present members         ``NaN`` for that sector's members
Fewer than 3 observations, beta         All ``NaN``
Zero dispersion in ``betas``            All ``NaN``
Mismatched lengths, 2-D input, ``inf``  :class:`ValueError`
Percentile outside ``[0, 100]``         :class:`ValueError`
======================================  ===================================

The minimum-observation rules are not arbitrary. A least-squares fit with as
many free parameters as observations reproduces its input exactly and returns
residuals that are identically zero: one name alone in its sector is *always* at
its sector mean, and two names regressed on ``[1, beta]`` are *always* on the
fitted line. Those zeros are artefacts of the arithmetic, not measurements of
neutrality, and emitting them would place fabricated exact-zero exposures into
the feature matrix. So the affected entries are ``NaN`` — not available, which
is the truth.

--------------------------------------------------------------------------
Idempotence: what is claimed, and what is not
--------------------------------------------------------------------------

Applied to its own output:

- :func:`winsorize` is **exactly idempotent** — bit-for-bit. Its cut points are
  order statistics of the sample rather than interpolated values, and clipping
  to an order statistic leaves it at the same rank, so the second pass
  recomputes the same bounds and clips nothing. See
  :func:`backend.features._stats.order_statistic_bounds`.
- :func:`cross_sectional_zscore` is **idempotent up to floating point**. The
  output has mean 0 and sample standard deviation 1 in exact arithmetic, so the
  second pass subtracts ~0 and divides by ~1; the residual difference is
  rounding, of order ``eps`` relative to the values.
- :func:`neutralize` is **idempotent up to floating point**. Within-group
  demeaning is an orthogonal projection, and groups that lost their only member
  to ``NaN`` stay ``NaN`` — a fixed point — so the second pass has the same
  usable set and subtracts ~0 from each group.
- :func:`beta_neutralize` is **idempotent up to floating point**, for the same
  reason: the residual is orthogonal to ``[1, betas]`` on the fitted subset, so
  the refit slope and intercept are ~0.

**The pipeline as a whole is not idempotent, and this is not a floating-point
caveat.** ``transform_cross_section(transform_cross_section(x)) !=
transform_cross_section(x)`` for two independent reasons:

1. **Re-standardization rescales.** Neutralization shrinks the cross-sectional
   standard deviation below 1 — that shrinkage is the sector and market exposure
   being removed. Running the pipeline again z-scores the residual back up to
   unit variance, multiplying every value by ``1 / sigma_residual > 1``. The
   ranks survive; the values do not, and it is the values that a model consumes.
2. **Sequential neutralization is not a projection.** Removing sectors and then
   removing beta applies two orthogonal projections whose subspaces are not
   orthogonal to each other. A vector made orthogonal to ``[1, betas]`` is not
   generally orthogonal to the sector dummies, so the composition
   ``M_beta M_sector`` re-introduces a sector component and is not idempotent
   even before any rescaling. Composing projections only yields a projection
   when they commute, which sector membership and market beta do not.

Both claims are tested by construction with explicit counterexamples in
``backend/tests/features/test_transforms.py``; the per-step claims are property
tested in ``test_transforms_properties.py``. The consequence for callers is
concrete: **transform raw features, once.** Feeding an already-transformed
feature back through the pipeline is not a harmless no-op, and nothing in this
module can detect that it happened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from backend.features._stats import (
    as_float_1d,
    as_group_1d,
    dispersion_is_degenerate,
    observed_mask,
    order_statistic_bounds,
    reject_infinities,
    require_matching_length,
    require_percentile_pair,
)

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "MINIMUM_GROUP_MEMBERS_FOR_NEUTRALIZATION",
    "MINIMUM_OBSERVATIONS_FOR_BETA_NEUTRALIZATION",
    "MINIMUM_OBSERVATIONS_FOR_ZSCORE",
    "ZERO_DISPERSION_RELATIVE_TOLERANCE",
    "beta_neutralize",
    "cross_sectional_zscore",
    "neutralize",
    "transform_cross_section",
    "winsorize",
]

MINIMUM_OBSERVATIONS_FOR_ZSCORE: Final = 2
"""Present values a cross-section needs before it can be standardized (count).

The sample standard deviation uses ``ddof=1``: with one observation it is
``0 / 0``, undefined rather than zero. One name is also not a cross-section —
"one standard deviation above the average of itself" is not a statement about
anything — so the date returns ``NaN`` instead of a number.
"""

MINIMUM_GROUP_MEMBERS_FOR_NEUTRALIZATION: Final = 2
"""Present values a group needs before its members can be neutralized (count).

A group of one is perfectly fitted by its own mean: the residual is identically
zero for any input whatsoever. That zero says "this name carries no
sector-relative signal", which is a fabricated claim rather than a measurement,
so the lone member's output is ``NaN``. In practice a singleton sector on a
liquid universe means the sector map is wrong, and ``NaN`` surfaces that in
coverage statistics instead of hiding it behind a plausible zero.
"""

MINIMUM_OBSERVATIONS_FOR_BETA_NEUTRALIZATION: Final = 3
"""Present pairs required to regress a cross-section on ``[1, betas]`` (count).

Two points determine the line through them exactly, leaving residuals that are
identically zero regardless of the data — the same fabricated neutrality as a
singleton sector. Three observations leave one residual degree of freedom, which
is the arithmetic minimum, **not** a recommendation: a beta-neutralization
estimated from three names is nearly all noise, and how many names are enough is
a judgement for the caller, where it is visible.
"""

ZERO_DISPERSION_RELATIVE_TOLERANCE: Final = 1e-12
"""Relative floor below which a cross-section counts as having no dispersion.

Dimensionless: the standard deviation and the magnitude it is compared against
are both in the input's units (see
:func:`backend.features._stats.dispersion_is_degenerate`). Roughly four orders
of magnitude above the ``float64`` demeaning noise floor (``eps ~ 2.2e-16``), so
a constant cross-section is caught even when rounding leaves it with a residual
standard deviation, and roughly four orders below the relative dispersion of any
real cross-sectional feature, so nothing genuine is discarded.
"""


def winsorize(
    values: npt.NDArray[np.float64], *, lower_pct: float = 1.0, upper_pct: float = 99.0
) -> npt.NDArray[np.float64]:
    """Clip a cross-section to its own percentile bounds.

    The bounds are order statistics of the **present** values on this date:
    ``NaN`` entries take no part in computing them and remain ``NaN`` in the
    output. Clipping preserves the units of the input and the caller's element
    order, and preserves rank order (it is a monotone map).

    Idempotent bit-for-bit — ``winsorize(winsorize(x)) == winsorize(x)``,
    exactly, not to a tolerance. See
    :func:`backend.features._stats.order_statistic_bounds` for why order
    statistics rather than interpolated percentiles are used.

    Args:
        values: one date's cross-section of a feature, one element per security,
            in any units. ``NaN`` means "not available".
        lower_pct: lower cut point in percent of the observed distribution;
            ``1.0`` is the 1st percentile. Default follows directive §5.
        upper_pct: upper cut point in percent; ``99.0`` is the 99th percentile.

    Returns:
        A new array in the input's units and element order, with present values
        clipped into ``[p_lower, p_upper]`` and ``NaN`` preserved. An empty
        input returns an empty array; an all-``NaN`` input returns all ``NaN``
        (there is no distribution to compute bounds from); a single present
        value is returned unchanged, since both bounds equal it.

    Raises:
        ValueError: if ``values`` is not a one-dimensional numeric array,
            contains ``+inf`` or ``-inf``, or if the percentile pair is
            non-finite, outside ``[0, 100]``, or inverted.

    Example:
        >>> import numpy as np
        >>> x = np.array([-100.0, 1.0, 2.0, 3.0, np.nan, 500.0])
        >>> winsorize(x, lower_pct=25.0, upper_pct=75.0)
        array([ 1.,  1.,  2.,  3., nan,  3.])
    """
    require_percentile_pair(lower_pct, upper_pct)
    array = as_float_1d(values, name="values")
    reject_infinities(array, name="values")

    observed = array[observed_mask(array)]
    if observed.size == 0:
        return array

    lower, upper = order_statistic_bounds(observed, lower_pct=lower_pct, upper_pct=upper_pct)
    # np.clip propagates NaN: min/max with NaN is NaN, so absent values stay absent.
    return np.asarray(np.clip(array, lower, upper), dtype=np.float64)


def cross_sectional_zscore(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standardize a cross-section to mean 0 and sample standard deviation 1.

    Computes ``(x - mean) / std`` with both statistics taken over the present
    values on this date only, and ``std`` the **sample** standard deviation
    (``ddof=1``). Absent values stay absent.

    Idempotent up to floating point: the output has mean 0 and sample standard
    deviation 1 in exact arithmetic, so a second application changes nothing
    beyond rounding.

    Args:
        values: one date's cross-section of a feature, one element per security,
            in any units. ``NaN`` means "not available".

    Returns:
        A new **dimensionless** array in the caller's element order — the
        input's units cancel — measured in cross-sectional standard deviations
        of this date. ``NaN`` is preserved. The whole array is ``NaN`` when
        fewer than :data:`MINIMUM_OBSERVATIONS_FOR_ZSCORE` values are present,
        or when the cross-section has no dispersion (see
        :data:`ZERO_DISPERSION_RELATIVE_TOLERANCE`): a constant cross-section
        carries no cross-sectional information, and dividing by its ~zero
        standard deviation would return ``inf`` or amplify rounding noise into
        full-scale scores. Zero is deliberately *not* returned in that case —
        "every name is exactly average" would be a fabricated measurement, where
        ``NaN`` correctly says the feature is unavailable that day.

    Raises:
        ValueError: if ``values`` is not a one-dimensional numeric array or
            contains ``+inf`` or ``-inf``.

    Example:
        >>> import numpy as np
        >>> cross_sectional_zscore(np.array([1.0, 2.0, 3.0, np.nan]))
        array([-1.,  0.,  1., nan])
    """
    array = as_float_1d(values, name="values")
    reject_infinities(array, name="values")

    present = observed_mask(array)
    observed = array[present]
    if observed.size < MINIMUM_OBSERVATIONS_FOR_ZSCORE:
        return np.full(array.shape, np.nan, dtype=np.float64)

    mean = float(np.mean(observed))
    standard_deviation = float(np.std(observed, ddof=1))
    scale = float(np.max(np.abs(observed)))
    if dispersion_is_degenerate(
        standard_deviation, scale=scale, relative_tolerance=ZERO_DISPERSION_RELATIVE_TOLERANCE
    ):
        return np.full(array.shape, np.nan, dtype=np.float64)

    return np.asarray((array - mean) / standard_deviation, dtype=np.float64)


def neutralize(
    values: npt.NDArray[np.float64], *, groups: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Remove group (sector) exposure by within-group demeaning.

    The residual of a least-squares regression on a full set of group indicator
    columns *is* the value minus its group mean, so this is that regression,
    computed directly. Every group mean is taken over the present values in that
    group on this date only; groups do not borrow from one another and no group
    can see another date.

    The result has a cross-sectional mean of zero within every usable group, so
    a portfolio built from it takes no deliberate sector position. Its
    cross-sectional standard deviation is *below* that of the input by the
    fraction of variance the sectors explained — that shrinkage is the sector
    bet being removed, and it is not scaled back out.

    Idempotent up to floating point.

    Args:
        values: one date's cross-section of a feature, one element per security,
            in any units. ``NaN`` means "not available".
        groups: group label per security, same length and order as ``values``.
            Sector codes, industry codes, or any exhaustive integer partition.
            Only equality between labels is used; their order and magnitude are
            ignored. A security whose group is unknown must be excluded by
            passing ``NaN`` for its value rather than given a placeholder label,
            which would neutralize unrelated names against each other.

    Returns:
        A new array of residuals **in the input's units** and the caller's
        element order. ``NaN`` where the input was absent, and ``NaN`` for every
        member of a group with fewer than
        :data:`MINIMUM_GROUP_MEMBERS_FOR_NEUTRALIZATION` present values — a
        lone member is exactly its own group mean, so its residual would be zero
        by construction rather than by measurement.

    Raises:
        ValueError: if the two arrays differ in length, if either is not
            one-dimensional, if ``values`` contains ``+inf`` or ``-inf``, or if
            ``groups`` is not integral.

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 3.0, 10.0, 20.0])
        >>> sectors = np.array([0, 0, 1, 1], dtype=np.int64)
        >>> neutralize(x, groups=sectors)
        array([-1.,  1., -5.,  5.])
    """
    array = as_float_1d(values, name="values")
    reject_infinities(array, name="values")
    labels = as_group_1d(groups, name="groups")
    require_matching_length({"values": array, "groups": labels})

    if array.size == 0:
        return array

    present = observed_mask(array)
    _, codes = np.unique(labels, return_inverse=True)
    n_groups = int(codes.max()) + 1 if codes.size else 0
    present_codes = codes[present]

    counts = np.bincount(present_codes, minlength=n_groups)
    totals = np.bincount(present_codes, weights=array[present], minlength=n_groups)

    usable = counts >= MINIMUM_GROUP_MEMBERS_FOR_NEUTRALIZATION
    means = np.full(n_groups, np.nan, dtype=np.float64)
    means[usable] = totals[usable] / counts[usable]

    # NaN in `means` propagates to every member of an unusable group; NaN in
    # `array` propagates for every absent value. Neither is filled.
    return np.asarray(array - means[codes], dtype=np.float64)


def beta_neutralize(
    values: npt.NDArray[np.float64], *, betas: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Remove market-beta exposure by cross-sectional regression.

    Fits ``value = alpha + slope * beta`` by ordinary least squares across the
    securities present on this date and returns the residual
    ``value - alpha - slope * beta``. The residual is uncorrelated with beta
    across the cross-section, so a portfolio built from it takes no deliberate
    market position through the factor.

    Only securities with **both** a value and a beta take part in the fit; a
    security missing either is absent from the output. Betas are inputs, not
    outputs: this function does not estimate them, and their estimation window
    is the caller's point-in-time responsibility.

    Idempotent up to floating point: the residual is orthogonal to ``[1, beta]``
    on the fitted subset, so a refit finds slope and intercept of ~0.

    Args:
        values: one date's cross-section of a feature, one element per security,
            in any units. ``NaN`` means "not available".
        betas: market beta per security (dimensionless), same length and order
            as ``values``. ``NaN`` means "not available" and excludes that
            security from both the fit and the output.

    Returns:
        A new array of residuals **in the units of** ``values``, in the caller's
        element order. ``NaN`` wherever either input was absent. The whole array
        is ``NaN`` when fewer than
        :data:`MINIMUM_OBSERVATIONS_FOR_BETA_NEUTRALIZATION` complete pairs are
        present — a two-point fit is exact and its residuals are identically
        zero — or when the present betas have no dispersion (see
        :data:`ZERO_DISPERSION_RELATIVE_TOLERANCE`): the design matrix
        ``[1, beta]`` is then rank deficient, the slope is not identified, and
        "beta neutral" is not a claim that can be made from a cross-section in
        which every name has the same beta.

    Raises:
        ValueError: if the two arrays differ in length, if either is not a
            one-dimensional numeric array, or if either contains ``+inf`` or
            ``-inf``.

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0, 3.0, 10.0])
        >>> b = np.array([0.5, 1.0, 1.5, 2.0])
        >>> np.round(beta_neutralize(x, betas=b), 6)
        array([ 1.2, -0.6, -2.4,  1.8])
    """
    array = as_float_1d(values, name="values")
    exposures = as_float_1d(betas, name="betas")
    reject_infinities(array, name="values")
    reject_infinities(exposures, name="betas")
    require_matching_length({"values": array, "betas": exposures})

    present = observed_mask(array) & observed_mask(exposures)
    n_present = int(np.count_nonzero(present))
    if n_present < MINIMUM_OBSERVATIONS_FOR_BETA_NEUTRALIZATION:
        return np.full(array.shape, np.nan, dtype=np.float64)

    fitted_values = array[present]
    fitted_betas = exposures[present]
    beta_mean = float(np.mean(fitted_betas))
    centered_betas = fitted_betas - beta_mean
    beta_dispersion = float(np.std(fitted_betas, ddof=1))
    beta_scale = float(np.max(np.abs(fitted_betas)))
    if dispersion_is_degenerate(
        beta_dispersion, scale=beta_scale, relative_tolerance=ZERO_DISPERSION_RELATIVE_TOLERANCE
    ):
        return np.full(array.shape, np.nan, dtype=np.float64)

    value_mean = float(np.mean(fitted_values))
    slope = float(
        np.dot(centered_betas, fitted_values - value_mean) / np.dot(centered_betas, centered_betas)
    )

    result = np.full(array.shape, np.nan, dtype=np.float64)
    result[present] = fitted_values - value_mean - slope * centered_betas
    return result


def transform_cross_section(
    values: npt.NDArray[np.float64],
    *,
    groups: npt.NDArray[np.int64],
    betas: npt.NDArray[np.float64] | None = None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> npt.NDArray[np.float64]:
    """Run the full Phase 5 transform pipeline over one date's cross-section.

    Applies, in the order directive §5 fixes:
    :func:`winsorize` → :func:`cross_sectional_zscore` → :func:`neutralize` →
    :func:`beta_neutralize` (the last only when ``betas`` is supplied). The
    order matters: winsorizing before standardizing keeps one outlier from
    setting the scale for everyone, and standardizing before neutralizing means
    the residual is expressed in cross-sectional standard deviations rather than
    in whatever the raw feature was denominated in.

    **Not idempotent.** Re-running the pipeline on its own output rescales the
    residual back to unit variance and re-composes two non-commuting
    projections; the module docstring derives both. Transform raw features
    exactly once.

    Args:
        values: one date's cross-section of a raw feature, one element per
            security, in any units. ``NaN`` means "not available".
        groups: sector (or other group) label per security, same length and
            order as ``values``.
        betas: optional market beta per security (dimensionless), same length
            and order. ``None`` skips beta neutralization entirely, which is the
            directive's default; passing an array applies it.
        lower_pct: winsorization lower cut point in percent. Default ``1.0``.
        upper_pct: winsorization upper cut point in percent. Default ``99.0``.

    Returns:
        A new **dimensionless** array in the caller's element order: the feature
        in cross-sectional standard-deviation units, with sector — and
        optionally market — exposure removed. Its cross-sectional standard
        deviation is at most 1 and generally below it; the output is not
        re-standardized, because that shrinkage measures how much of the factor
        was a sector or market bet. ``NaN`` wherever the value was absent or a
        step could not be computed (see the degenerate-cross-section table in
        the module docstring); ``NaN`` propagates, never filled.

    Raises:
        ValueError: if the arrays differ in length, are not one-dimensional
            numeric arrays, contain ``+inf`` or ``-inf``, if ``groups`` is not
            integral, or if the percentile pair is invalid.
    """
    result = winsorize(values, lower_pct=lower_pct, upper_pct=upper_pct)
    result = cross_sectional_zscore(result)
    result = neutralize(result, groups=groups)
    if betas is not None:
        result = beta_neutralize(result, betas=betas)
    return result
