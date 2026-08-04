"""Population Stability Index over feature distributions (P12.2).

The PSI asks one question: *does this feature look, today, like it looked when
the model was trained?* It answers it by binning both distributions the same way
and summing a divergence over the bins,

::

    PSI = sum_i (q_i - p_i) * ln(q_i / p_i)

where ``p_i`` is the fraction of the **reference** (training-period)
distribution that fell in bin ``i`` and ``q_i`` the fraction of the **live**
distribution that falls in it now. That expression is Jeffreys divergence — the
symmetrized Kullback-Leibler divergence, ``KL(q||p) + KL(p||q)`` — which is why
it is non-negative, zero only when the two histograms agree exactly, and
unchanged if the two samples are swapped. The one-sided form ``sum q ln(q/p)``
is also called PSI in some references and gives materially smaller numbers; this
module implements the symmetric form, which is the one the 0.1/0.25 convention
was calibrated against.

The formula is four lines of arithmetic. Every way it goes wrong is a decision
made outside the formula, so each is made explicitly here.

--------------------------------------------------------------------------
1. Binning is fixed at reference time. This is the whole design.
--------------------------------------------------------------------------

Bins are quantile bins of the **reference** sample, computed once when
:meth:`ReferenceDistribution.from_sample` is called and frozen into the
resulting object along with the reference's own bin counts. Every later
comparison re-uses those edges.

The tempting alternative — recompute the quantile edges from each period's
sample — destroys the detector, and does so silently. Quantile edges computed
from a sample put, by construction, about ``1/B`` of *that* sample in each bin.
So ``q_i ≈ p_i ≈ 1/B`` for every ``i``, every period, and PSI collapses to the
noise floor no matter how far the distribution has moved. A monitor built that
way reports "stable" through an upstream unit change, a vendor switching from
percent to fraction, or a factor's dispersion halving. It is the classic silent
failure of this metric and it is why this module has **no** entry point that
takes two raw samples: :func:`population_stability_index` accepts a
:class:`ReferenceDistribution` and one live array, and the reference carries its
edges with it. ``backend/tests/monitoring/test_psi.py`` pins the failure mode as
a test in its own right.

The consequence is that a reference is an *artifact*, not a parameter. It
carries an identifier, a :class:`~backend.tracking.stamp.ReproducibilityStamp`
naming the data version it was built from, and a fingerprint over its own edges
and counts — because "PSI = 0.31" without "measured against reference
``train_2015_2020@a3f1…``" is not interpretable (invariant I2).

--------------------------------------------------------------------------
2. Empty bins: an epsilon floor, applied to the live side, made visible
--------------------------------------------------------------------------

``(q - p) ln(q/p)`` is undefined when either fraction is zero, and a bin that is
empty on one side is the *normal* consequence of real drift — when a
distribution shifts, its old tail bin empties. Three policies exist and all
three change the number:

* **Refuse to report when a bin is empty.** Rejected: the detector would go
  silent at exactly the moment it has the most to say.
* **Merge the empty bin into its neighbour.** Rejected, and this is the
  important rejection: merging at comparison time changes the binning *per
  period*, which re-introduces the failure of §1 through the back door and makes
  two periods' PSI values incomparable — they would be sums over different
  partitions.
* **Floor both fractions at a small epsilon.** Chosen. ``p`` and ``q`` are
  replaced by ``max(p, epsilon)`` and ``max(q, epsilon)`` in both the difference
  and the logarithm, so a bin that empties contributes a large but finite
  amount.

"Large but finite" is doing real work and must not be buried: with the default
``epsilon = 1e-6`` and ten equal reference bins, **one** emptied decile
contributes ``(1e-6 - 0.1) * ln(1e-6 / 0.1) ≈ 1.15`` — on its own, more than
four times the conventional "major drift" threshold. That number is a
consequence of the floor, not a measurement of the data. So
:class:`PSIResult` reports :attr:`~PSIResult.epsilon`, which bins were floored,
and :attr:`~PSIResult.floor_contribution` — how much of the total the floored
bins supplied — and the epsilon is part of the reference's fingerprint, so two
PSI values quoted against the same reference were computed with the same floor.

The **reference** side can never be empty: a reference whose own bins are not
all occupied is refused at construction
(:class:`~backend.monitoring.errors.ReferenceDistributionError`), because bins
that do not describe the reference give the comparison an arbitrary baseline.
The floor therefore only ever applies to the live sample — with one exception,
the availability split of §5, where a training period with no missing values at
all legitimately has ``p_absent = 0``.

--------------------------------------------------------------------------
3. Minimum sample size, and where it comes from
--------------------------------------------------------------------------

PSI on a small cross-section is noise. To second order, ``(q-p) ln(q/p) ≈
(q-p)²/p``, so ``n * PSI`` is approximately Pearson's ``X²``, which under the
null of no drift is ``chi-squared`` with ``B - 1`` degrees of freedom. Hence,
with no drift at all,

::

    E[PSI] ≈ (B - 1) / n            sd[PSI] ≈ sqrt(2 * (B - 1)) / n

Ten bins and one hundred names give ``E[PSI] ≈ 0.09``, which any operator
reading the conventional ladder would call a moderate drift. The minimum is
therefore derived rather than chosen: :func:`minimum_sample_size` requires
enough observations that this null expectation sits at or below
:data:`NULL_PSI_NOISE_BUDGET` — a quarter of the conventional 0.10 band — which
gives ``n >= (B - 1) / 0.025``: **360** observations for ten bins, 120 for four.
Below that, :func:`population_stability_index` raises
:class:`~backend.monitoring.errors.InsufficientSampleError` rather than
returning a number that will be read as a signal.

Two honest caveats, stated because the arithmetic above is cleaner than reality:

* the chi-squared approximation assumes **independent** observations. A
  cross-section of equity features is not independent — common factors correlate
  the names — so the true sampling variation is *larger* than ``(B-1)/n``, and
  the minimum is a floor rather than a guarantee;
* the same expectation is reported on every result as
  :attr:`PSIResult.null_expected_value`, so an operator can compare a measured
  PSI against the noise scale of the sample it came from instead of only against
  the folklore ladder of §4.

--------------------------------------------------------------------------
4. Thresholds are convention, and they live elsewhere
--------------------------------------------------------------------------

The familiar bands — below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 major
— come from credit-scorecard practice (they are quoted, without derivation, in
Siddiqi, *Credit Risk Scorecards*, 2006, and repeated in most vendor
documentation since). They are not derived from any distributional result and
they carry no significance level. Nothing in this module knows about them:
classification lives in :mod:`backend.monitoring.drift`, where
:class:`~backend.monitoring.drift.DriftBands` holds the numbers, states in its
own ``basis`` field that they are convention, and lets a caller replace them.

--------------------------------------------------------------------------
5. NaN is "not available", and a change in the NaN rate is drift
--------------------------------------------------------------------------

:mod:`backend.features` treats ``NaN`` as "not available" and never imputes it.
Dropping those entries and computing PSI over the survivors would therefore
discard the single most operationally important drift signal there is: a feature
whose availability collapses from 98% to 60% usually means an upstream source
broke, and the distribution of the names that *survive* can look perfectly
stable while that happens.

So the NaN rate is measured explicitly, as its own two-category PSI over
``{present, absent}`` (:class:`NanRateShift`), reported alongside the
distribution PSI and never added into it — they are different findings and
summing them would let a large availability break hide inside a moderate
distributional number, or the reverse. If the live sample has enough
observations to talk about availability but too few *present* values to bin,
the availability shift is attached to the raised
:class:`~backend.monitoring.errors.InsufficientSampleError` rather than lost.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

* PSI is **dimensionless**. It is a divergence between two histograms, so it has
  no units regardless of what the feature is denominated in. Its magnitude does
  depend on the bin count, which is why the bin count is part of the reference's
  identity.
* Feature values are in **the units the feature declares**
  (:class:`backend.features.spec.FeatureSpec`); the reference records that
  string so a report cannot be read in the wrong unit. Values are never
  rescaled here.
* Fractions (``p``, ``q``, NaN rates) are **fractions in [0, 1]**, never
  percent.
* Inputs are **one cross-section**: one value per security, from one date, in
  any order — PSI is order-invariant. Pooling several dates into one array is
  not refusable by this module and is a caller error: the resulting histogram is
  a time-average, and drift within the window cancels against itself.
* ``+inf`` / ``-inf`` are refused, matching the features package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

import numpy as np

from backend.monitoring.errors import (
    InsufficientSampleError,
    MonitoringInputError,
    ReferenceDistributionError,
)
from backend.tracking.stamp import ReproducibilityStamp, canonical_config_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "AVAILABILITY_CATEGORIES",
    "DEFAULT_BINS",
    "DEFAULT_EPSILON",
    "MINIMUM_BINS",
    "NULL_PSI_NOISE_BUDGET",
    "BinContribution",
    "FloatArray",
    "JsonValue",
    "NanRateShift",
    "PSIResult",
    "ReferenceDistribution",
    "minimum_sample_size",
    "population_stability_index",
]

type FloatArray = npt.NDArray[np.float64]
"""One-dimensional ``float64`` cross-section. Units are the feature's own."""

type IntArray = npt.NDArray[np.int64]
"""One-dimensional ``int64`` array of bin indices or counts (dimensionless)."""

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None

DEFAULT_BINS: Final = 10
"""Quantile bins used when a caller does not say otherwise (count).

Deciles are the convention the 0.1/0.25 ladder was calibrated against, and PSI
is not comparable across bin counts — a finer partition finds more divergence
for the same shift — so the count travels with the reference distribution rather
than being a per-call argument.
"""

MINIMUM_BINS: Final = 4
"""Fewest bins a reference distribution may have (count).

With three bins or fewer, a distribution can move substantially while leaving
every bin fraction almost unchanged (a symmetric widening, for instance, is
nearly invisible to a coarse partition), so the detector's power collapses
before its arithmetic does. Four is the floor at which quantile bins still
describe a shape rather than a location.
"""

DEFAULT_EPSILON: Final = 1e-6
"""Floor applied to a bin fraction before the logarithm (a fraction).

Small enough to leave a populated bin's contribution untouched to eleven
significant figures, large enough that an emptied bin contributes a bounded
amount (``≈ 1.15`` for an emptied decile) rather than an infinity. It is a
policy constant, not a measurement — see §2 of the module docstring — and it is
part of the reference fingerprint so that two PSI values quoted against one
reference share it.
"""

NULL_PSI_NOISE_BUDGET: Final = 0.025
"""Largest null-hypothesis PSI a sample may carry to be usable (dimensionless).

A quarter of the conventional 0.10 "moderate drift" band. The sample minimum is
derived from it (:func:`minimum_sample_size`), so the relationship between "how
much noise am I willing to mistake for drift" and "how many names do I need" is
one number rather than a table of magic constants.
"""

AVAILABILITY_CATEGORIES: Final = 2
"""Categories in the NaN-rate comparison: present and absent (count)."""

_MAX_ABSENT_FRACTION: Final = 1.0
_LEDGER_UNSAFE: Final = ("|", "\n", "\r")


def minimum_sample_size(n_bins: int) -> int:
    """Return the fewest observations PSI may be computed from, for ``n_bins``.

    Derived, not chosen. Under the null of no drift, ``n * PSI`` is
    approximately ``chi-squared(n_bins - 1)`` (module docstring §3), so pure
    multinomial noise produces ``E[PSI] ≈ (n_bins - 1) / n``. Requiring that
    expectation to sit at or below :data:`NULL_PSI_NOISE_BUDGET` gives

    ::

        n >= (n_bins - 1) / NULL_PSI_NOISE_BUDGET

    which is 360 observations for ten bins and 120 for four.

    Args:
        n_bins: number of bins the PSI will be summed over (count), at least 2.
            The two-category NaN-rate comparison passes
            :data:`AVAILABILITY_CATEGORIES`.

    Returns:
        The minimum number of observations (count).

    Raises:
        MonitoringInputError: if ``n_bins`` is not an integer of at least 2.
            One bin has no divergence to measure.

    Example:
        >>> minimum_sample_size(10)
        360
        >>> minimum_sample_size(2)
        40
    """
    supplied: object = n_bins
    if isinstance(supplied, bool) or not isinstance(supplied, int):
        msg = f"n_bins must be an int, got {type(supplied).__name__}"
        raise MonitoringInputError(msg)
    if n_bins < AVAILABILITY_CATEGORIES:
        msg = (
            f"n_bins must be at least {AVAILABILITY_CATEGORIES}; got {n_bins}. A single "
            f"bin holds the whole distribution on both sides, so its PSI is identically "
            f"zero whatever the data does."
        )
        raise MonitoringInputError(msg)
    return math.ceil((n_bins - 1) / NULL_PSI_NOISE_BUDGET)


def _as_float_1d(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Coerce a cross-section to a one-dimensional ``float64`` array.

    Args:
        values: array-like of numbers, one element per security. ``NaN`` is
            allowed and meaningful ("not available"); infinities are not.
        name: parameter name, used verbatim in error messages.

    Returns:
        A fresh one-dimensional ``float64`` array (a copy, so a caller may
        mutate its input afterwards without disturbing a computed result).

    Raises:
        MonitoringInputError: if the values are not numeric, are not
            one-dimensional, or contain ``+inf`` / ``-inf``.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric; could not convert to float64 ({exc})"
        raise MonitoringInputError(msg) from exc
    if array.ndim != 1:
        msg = (
            f"{name} must be a one-dimensional cross-section (one value per security, "
            f"one date); got shape {array.shape}. Pooling dates into one histogram "
            f"time-averages the distribution and lets drift inside the window cancel "
            f"against itself."
        )
        raise MonitoringInputError(msg)
    infinite = np.flatnonzero(np.isinf(array))
    if infinite.size:
        first = int(infinite[0])
        msg = (
            f"{name} contains {infinite.size} infinite value(s), first at position "
            f"{first}. An infinity is not an extreme observation, it is an unhandled "
            f"division by zero upstream; binning it would launder it into a plausible "
            f"tail count. Emit NaN where the value is genuinely unavailable."
        )
        raise MonitoringInputError(msg)
    return np.array(array, dtype=np.float64, copy=True)


def _require_named(value: str, *, field: str) -> str:
    """Return a stripped identifier, refusing blanks and separator characters.

    Args:
        value: the candidate string.
        field: field name, used verbatim in error messages.

    Returns:
        ``value`` stripped of surrounding whitespace.

    Raises:
        ReferenceDistributionError: if the value is not a string, is blank, or
            contains a pipe or newline. A reference distribution's identity is
            quoted in reports, exceptions and ledger rows; a blank one makes
            every number measured against it uninterpretable (I2).
    """
    supplied: object = value
    if not isinstance(supplied, str):
        msg = f"{field} must be a string, got {type(supplied).__name__}"
        raise ReferenceDistributionError(msg)
    stripped = value.strip()
    if not stripped:
        msg = (
            f"{field} is empty. A PSI is a comparison against a named reference; "
            f"without the name the number cannot be interpreted or reproduced (I2)."
        )
        raise ReferenceDistributionError(msg)
    for character in _LEDGER_UNSAFE:
        if character in stripped:
            msg = f"{field}={value!r} contains {character!r}, which would corrupt a report row"
            raise ReferenceDistributionError(msg)
    return stripped


def _psi_term(
    reference_fraction: float, observed_fraction: float, *, epsilon: float
) -> tuple[float, bool, bool]:
    """Return one bin's PSI contribution and which sides were floored.

    Computes ``(q' - p') * ln(q' / p')`` with ``p' = max(p, epsilon)`` and
    ``q' = max(q, epsilon)``. The floor is applied inside *both* the difference
    and the logarithm, so the contribution is exactly the one a distribution
    holding ``epsilon`` of its mass in that bin would produce — rather than a
    hybrid of a floored logarithm and an unfloored weight, which is not the PSI
    of any distribution.

    Args:
        reference_fraction: ``p``, the reference share of this bin, in ``[0, 1]``.
        observed_fraction: ``q``, the live share of this bin, in ``[0, 1]``.
        epsilon: the floor, a fraction strictly inside ``(0, 1)``.

    Returns:
        ``(contribution, reference_floored, observed_floored)``. The contribution
        is dimensionless and non-negative (both factors share a sign).
    """
    reference_floored = reference_fraction < epsilon
    observed_floored = observed_fraction < epsilon
    p = max(reference_fraction, epsilon)
    q = max(observed_fraction, epsilon)
    return (q - p) * math.log(q / p), reference_floored, observed_floored


@dataclass(frozen=True, slots=True)
class BinContribution:
    """One bin's share of a PSI, with everything needed to read it.

    Attributes:
        index: bin position, ``0`` being the lowest.
        lower: inclusive lower edge, in the feature's units. ``-inf`` for the
            first bin.
        upper: exclusive upper edge, in the feature's units. ``+inf`` for the
            last bin.
        reference_count: reference observations in this bin (count).
        reference_fraction: reference share of this bin, a fraction.
        observed_count: live observations in this bin (count).
        observed_fraction: live share of this bin, a fraction.
        contribution: this bin's term of the PSI sum, dimensionless and
            non-negative.
        floored: whether the epsilon floor was applied to either fraction. When
            true the contribution is a consequence of the floor as much as of
            the data (module docstring §2).
    """

    index: int
    lower: float
    upper: float
    reference_count: int
    reference_fraction: float
    observed_count: int
    observed_fraction: float
    contribution: float
    floored: bool

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the contribution as a JSON-safe mapping."""
        return {
            "index": self.index,
            "lower": self.lower if math.isfinite(self.lower) else str(self.lower),
            "upper": self.upper if math.isfinite(self.upper) else str(self.upper),
            "reference_count": self.reference_count,
            "reference_fraction": self.reference_fraction,
            "observed_count": self.observed_count,
            "observed_fraction": self.observed_fraction,
            "contribution": self.contribution,
            "floored": self.floored,
        }


@dataclass(frozen=True, slots=True)
class NanRateShift:
    """Drift in *availability*: the two-category PSI over {present, absent}.

    Reported separately from the distribution PSI and never folded into it. A
    feature can be perfectly stable among the names that still have a value
    while a third of the universe silently stops having one, and those are
    different findings with different causes — the second is almost always an
    upstream source that broke rather than a market that moved.

    The epsilon floor of the module docstring §2 applies here on the *reference*
    side too: a training period with no missing values at all has
    ``reference_absent_fraction == 0``, and that is a legitimate reference rather
    than a malformed one.

    Attributes:
        reference_absent_fraction: share of the reference sample that was
            ``NaN``, a fraction in ``[0, 1]``.
        observed_absent_fraction: share of the live sample that is ``NaN``, a
            fraction in ``[0, 1]``.
        n_reference: reference observations, present and absent (count).
        n_observed: live observations, present and absent (count).
        value: the two-category PSI, dimensionless and non-negative.
        floored: whether the epsilon floor was applied to either side.
        null_expected_value: ``1 / n_observed``, the PSI this comparison would
            show on average from sampling noise alone with no change in the
            underlying rate (module docstring §3). A lower bound on noise, since
            it assumes independent observations.
    """

    reference_absent_fraction: float
    observed_absent_fraction: float
    n_reference: int
    n_observed: int
    value: float
    floored: bool
    null_expected_value: float

    @property
    def rate_change(self) -> float:
        """Return ``observed - reference`` absent fraction (a signed fraction)."""
        return self.observed_absent_fraction - self.reference_absent_fraction

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the availability shift as a JSON-safe mapping."""
        return {
            "reference_absent_fraction": self.reference_absent_fraction,
            "observed_absent_fraction": self.observed_absent_fraction,
            "rate_change": self.rate_change,
            "n_reference": self.n_reference,
            "n_observed": self.n_observed,
            "value": self.value,
            "floored": self.floored,
            "null_expected_value": self.null_expected_value,
            "units": "all fractions dimensionless in [0, 1]; value is a dimensionless PSI",
        }


@dataclass(frozen=True, slots=True)
class ReferenceDistribution:
    """A training-period distribution, binned once and frozen (module §1).

    This is the artifact every PSI in the system is measured against. It holds
    the bin edges, the reference counts that produced them, the epsilon floor,
    and its own identity: a name, the feature and units it describes, and the
    :class:`~backend.tracking.stamp.ReproducibilityStamp` of the run that built
    it. Construction goes through :meth:`from_sample`; the raw constructor
    validates but does not compute, so a reference can be rebuilt from stored
    counts without the original sample.

    Every bin is non-empty by construction. A reference with an empty bin is
    refused, so ``p > 0`` always and the epsilon floor of the module docstring
    only ever applies to the live side.

    Attributes:
        feature: the monitored feature's name, as declared in
            :class:`backend.features.spec.FeatureSpec`.
        units: the feature's units, carried verbatim from its declaration so a
            report cannot be read in the wrong unit (directive §8).
        reference_id: stable identifier of *this* reference — the training
            window it was cut from, for example ``"train_2015_2020"``. Quoted in
            every report and every refusal.
        stamp: the I2 stamp of the run that built the reference. Its
            ``data_version`` is what makes "the training distribution" a
            specific set of numbers rather than a description.
        edges: ``n_bins + 1`` bin edges in the feature's units, strictly
            increasing, with ``edges[0] == -inf`` and ``edges[-1] == +inf``.
            The outer edges are infinite on purpose: a live value beyond
            anything seen in training belongs in the extreme bin, where it
            registers as drift. Dropping it — which finite outer edges would
            do — would hide exactly the observations that matter most.
            Bins are half-open ``[lower, upper)``.
        bin_counts: reference observations per bin (counts), all at least 1.
        n_absent: reference observations that were ``NaN`` (count).
        epsilon: the floor applied to a fraction before the logarithm.
    """

    feature: str
    units: str
    reference_id: str
    stamp: ReproducibilityStamp
    edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    n_absent: int
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        """Validate the reference and freeze its sequences.

        Raises:
            ReferenceDistributionError: if any part of the reference cannot
                describe a usable comparison basis — a blank identifier, edges
                that are not strictly increasing or not infinite at the ends,
                fewer than :data:`MINIMUM_BINS` bins, an empty bin, a sample
                below :func:`minimum_sample_size`, a negative absent count, or
                an epsilon outside ``(0, 1)``.
        """
        object.__setattr__(self, "feature", _require_named(self.feature, field="feature"))
        object.__setattr__(self, "units", _require_named(self.units, field="units"))
        object.__setattr__(
            self, "reference_id", _require_named(self.reference_id, field="reference_id")
        )
        supplied_stamp: object = self.stamp
        if not isinstance(supplied_stamp, ReproducibilityStamp):
            msg = (
                f"stamp must be a ReproducibilityStamp, got "
                f"{type(supplied_stamp).__name__}. A reference distribution that cannot "
                f"say which data version it was cut from makes every PSI measured "
                f"against it unreproducible (I2)."
            )
            raise ReferenceDistributionError(msg)

        try:
            edges = tuple(float(edge) for edge in self.edges)
            counts = tuple(int(count) for count in self.bin_counts)
        except (TypeError, ValueError) as exc:
            msg = f"edges and bin_counts must be sequences of numbers ({exc})"
            raise ReferenceDistributionError(msg) from exc
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "bin_counts", counts)

        if len(edges) != len(counts) + 1:
            msg = (
                f"{len(edges)} edges cannot delimit {len(counts)} bins; a partition into "
                f"n bins has exactly n + 1 edges"
            )
            raise ReferenceDistributionError(msg)
        if len(counts) < MINIMUM_BINS:
            msg = (
                f"a reference distribution needs at least {MINIMUM_BINS} bins to describe "
                f"a shape rather than a location; got {len(counts)}"
            )
            raise ReferenceDistributionError(msg)
        if edges[0] != -math.inf or edges[-1] != math.inf:
            msg = (
                f"the outer edges must be -inf and +inf; got {edges[0]!r} and {edges[-1]!r}. "
                f"Finite outer edges would drop live values beyond the training range, "
                f"which are the observations most likely to be the drift."
            )
            raise ReferenceDistributionError(msg)
        for position in range(len(edges) - 1):
            if not edges[position] < edges[position + 1]:
                msg = (
                    f"edges must strictly increase; edges[{position}]={edges[position]!r} is "
                    f"not below edges[{position + 1}]={edges[position + 1]!r}. Equal edges "
                    f"describe an empty bin, which no observation can ever fall in."
                )
                raise ReferenceDistributionError(msg)
        if any(not math.isfinite(edge) for edge in edges[1:-1]):
            msg = f"interior edges must be finite; got {edges[1:-1]!r}"
            raise ReferenceDistributionError(msg)
        empty = [index for index, count in enumerate(counts) if count < 1]
        if empty:
            msg = (
                f"reference bin(s) {empty} hold no observations. A bin the reference "
                f"never occupied gives the comparison an arbitrary baseline: any live "
                f"observation landing there would be divided by a fabricated floor "
                f"rather than by a measured share. Re-cut the bins (fewer of them, or "
                f"explicit edges) so that every bin describes the reference."
            )
            raise ReferenceDistributionError(msg)
        if self.n_absent < 0:
            msg = f"n_absent must be a non-negative count; got {self.n_absent}"
            raise ReferenceDistributionError(msg)
        minimum = minimum_sample_size(len(counts))
        n_present = sum(counts)
        if n_present < minimum:
            msg = (
                f"reference {self.reference_id!r} for feature {self.feature!r} holds "
                f"{n_present} present observations across {len(counts)} bins, below the "
                f"minimum of {minimum}. Quantile edges placed from a sample this small "
                f"are themselves noise, and every later comparison would inherit it."
            )
            raise ReferenceDistributionError(msg)
        if not 0.0 < self.epsilon < 1.0:
            msg = (
                f"epsilon must be a fraction strictly inside (0, 1); got {self.epsilon!r}. "
                f"Zero re-admits the undefined logarithm the floor exists to remove."
            )
            raise ReferenceDistributionError(msg)

    @property
    def n_bins(self) -> int:
        """Number of bins (count)."""
        return len(self.bin_counts)

    @property
    def n_present(self) -> int:
        """Reference observations that carried a value (count)."""
        return sum(self.bin_counts)

    @property
    def n_total(self) -> int:
        """Reference observations, present and absent (count)."""
        return self.n_present + self.n_absent

    @property
    def absent_fraction(self) -> float:
        """Share of the reference sample that was ``NaN`` (a fraction in [0, 1])."""
        total = self.n_total
        return self.n_absent / total if total else 0.0

    @property
    def fractions(self) -> tuple[float, ...]:
        """Reference share of each bin (fractions summing to 1 over present values)."""
        present = self.n_present
        return tuple(count / present for count in self.bin_counts)

    @property
    def fingerprint(self) -> str:
        """Return a SHA-256 digest over everything that defines the comparison.

        Covers the feature, units, identifier, edges, reference counts, absent
        count and epsilon — every input to a PSI other than the live sample.
        Two reports quoting the same fingerprint were measured on the same
        partition with the same floor and are therefore comparable; two quoting
        different fingerprints are not, however similar their reference
        identifiers look.

        Edges are hashed as ``float.hex()`` strings so that ``±inf`` survives
        canonical JSON (which refuses non-finite numbers) and finite edges are
        captured exactly rather than through a decimal rendering.

        Returns:
            A 64-character lowercase hex digest.
        """
        return canonical_config_hash(
            {
                "feature": self.feature,
                "units": self.units,
                "reference_id": self.reference_id,
                "edges": [edge.hex() for edge in self.edges],
                "bin_counts": list(self.bin_counts),
                "n_absent": self.n_absent,
                "epsilon": self.epsilon.hex(),
            }
        )

    @classmethod
    def from_sample(
        cls,
        values: npt.ArrayLike,
        *,
        feature: str,
        units: str,
        reference_id: str,
        stamp: ReproducibilityStamp,
        n_bins: int = DEFAULT_BINS,
        edges: Sequence[float] | None = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> Self:
        """Build a reference distribution from a training-period sample.

        Bins are the sample's own quantiles — ``n_bins`` equal-probability
        intervals — computed **here, once**, and frozen into the returned
        object. Nothing recomputes them later; that is the whole point of the
        type (module docstring §1).

        ``NaN`` entries are counted as absent rather than dropped, so the
        reference records the availability rate the live sample will be compared
        against (module docstring §5).

        Args:
            values: the training-period cross-section (or a pooled training
                window), one element per observation, in the feature's units.
                ``NaN`` means "not available"; infinities are refused.
            feature: the monitored feature's name.
            units: the feature's units, from its declaration.
            reference_id: stable identifier for this reference.
            stamp: the I2 stamp of the run building it.
            n_bins: number of quantile bins (count). Defaults to
                :data:`DEFAULT_BINS`. PSI is not comparable across bin counts.
            edges: optional explicit **interior** cut points in the feature's
                units, strictly increasing and finite. Supply these when the
                feature has mass points that make quantile bins collapse (a
                factor that is 60% zeros, say). The reference *fractions* are
                still measured from ``values`` — only the partition is the
                caller's.
            epsilon: floor applied to a fraction before the logarithm.

        Returns:
            A frozen :class:`ReferenceDistribution`.

        Raises:
            MonitoringInputError: if ``values`` is not a one-dimensional numeric
                array, contains an infinity, or ``n_bins`` is not an integer of
                at least 2.
            ReferenceDistributionError: if the sample is empty, holds fewer than
                :func:`minimum_sample_size` present values, produces fewer than
                :data:`MINIMUM_BINS` distinct quantile edges (a distribution
                concentrated on mass points — pass ``edges`` explicitly), or
                leaves a bin empty.
        """
        array = _as_float_1d(values, name="values")
        present = array[~np.isnan(array)]
        n_absent = int(array.size - present.size)
        if present.size == 0:
            msg = (
                f"reference {reference_id!r} for feature {feature!r} holds no present "
                f"observations ({array.size} values, all NaN or none at all). There is no "
                f"distribution to be a reference, and an empty reference would make every "
                f"later PSI a comparison against nothing (I3)."
            )
            raise ReferenceDistributionError(msg)

        if edges is None:
            requested = minimum_sample_size(n_bins)
            if n_bins < MINIMUM_BINS:
                # Checked here rather than only in __post_init__, where the same
                # partition would be refused with a message about mass points.
                # Asking for three quantile bins is a caller's choice, not a
                # property of the data, and a diagnostic that blames the data
                # sends the reader looking in the wrong place.
                msg = (
                    f"n_bins={n_bins} is below the minimum of {MINIMUM_BINS}. Fewer than "
                    f"{MINIMUM_BINS} bins describe a location rather than a shape, and a "
                    f"distribution can move substantially while leaving every bin fraction "
                    f"almost unchanged."
                )
                raise ReferenceDistributionError(msg)
            if present.size < requested:
                msg = (
                    f"reference {reference_id!r} for feature {feature!r} holds "
                    f"{present.size} present observations, below the minimum of "
                    f"{requested} for {n_bins} bins. Quantile edges placed from a sample "
                    f"this small are themselves noise, and every later comparison would "
                    f"inherit it. Use fewer bins or a longer training window."
                )
                raise ReferenceDistributionError(msg)
            probabilities = [index / n_bins for index in range(1, n_bins)]
            interior = np.unique(np.quantile(present, probabilities, method="linear"))
            if interior.size + 1 < MINIMUM_BINS:
                msg = (
                    f"the {n_bins}-quantile edges of feature {feature!r} collapse to "
                    f"{interior.size + 1} distinct bin(s), below the minimum of "
                    f"{MINIMUM_BINS}. The distribution is concentrated on mass points, so "
                    f"equal-probability bins do not exist for it. Pass explicit `edges` "
                    f"chosen for those mass points instead — a partition that does not "
                    f"describe the reference cannot measure drift away from it."
                )
                raise ReferenceDistributionError(msg)
            cut_points = tuple(float(edge) for edge in interior)
        else:
            cut_points = tuple(float(edge) for edge in edges)
            if any(not math.isfinite(edge) for edge in cut_points):
                msg = f"explicit interior edges must all be finite; got {cut_points!r}"
                raise ReferenceDistributionError(msg)

        full_edges = (-math.inf, *cut_points, math.inf)
        counts = _bin_counts(full_edges, present)
        return cls(
            feature=feature,
            units=units,
            reference_id=reference_id,
            stamp=stamp,
            edges=full_edges,
            bin_counts=tuple(int(count) for count in counts),
            n_absent=n_absent,
            epsilon=epsilon,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the reference as a JSON-safe mapping, fingerprint included."""
        return {
            "feature": self.feature,
            "units": self.units,
            "reference_id": self.reference_id,
            "fingerprint": self.fingerprint,
            "data_version": self.stamp.data_version,
            "git_reference": self.stamp.git_reference,
            "config_hash": self.stamp.config_hash,
            "seed": self.stamp.seed,
            "edges": [edge if math.isfinite(edge) else str(edge) for edge in self.edges],
            "bin_counts": list(self.bin_counts),
            "n_present": self.n_present,
            "n_absent": self.n_absent,
            "absent_fraction": self.absent_fraction,
            "epsilon": self.epsilon,
            "n_bins": self.n_bins,
            "binning": "quantile bins of the reference sample, fixed at reference time",
        }


def _bin_counts(edges: tuple[float, ...], values: FloatArray) -> IntArray:
    """Return the count of ``values`` falling in each bin of ``edges``.

    Bins are half-open ``[lower, upper)``: a value exactly on an interior edge
    belongs to the bin above it. The outer edges are infinite, so nothing is
    ever dropped for being outside the reference range.

    Args:
        edges: ``n_bins + 1`` strictly increasing edges, outermost infinite.
        values: present (non-``NaN``) values, in the feature's units.

    Returns:
        An ``int64`` array of length ``n_bins`` summing to ``values.size``.
    """
    interior = np.asarray(edges[1:-1], dtype=np.float64)
    indices = np.searchsorted(interior, values, side="right")
    return np.asarray(np.bincount(indices, minlength=len(edges) - 1), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class PSIResult:
    """One measured PSI, with the arithmetic that produced it left visible.

    Attributes:
        reference: the frozen reference the comparison was made against. Carries
            its own identifier, fingerprint and I2 stamp, so the result is
            interpretable on its own (module docstring §1).
        value: the Population Stability Index over the **present** values,
            dimensionless and non-negative. Zero only when the two histograms
            agree exactly.
        bins: per-bin contributions, in bin order, summing to :attr:`value`.
        availability: the separate NaN-rate comparison. Never added into
            :attr:`value` — see the module docstring §5.
        n_observed_total: live observations supplied, present and absent (count).
        n_observed_present: live observations carrying a value (count).
        null_expected_value: ``(n_bins - 1) / n_observed_present``, the PSI this
            comparison would show on average under **no drift at all**, from
            multinomial sampling noise. Reported so a measured value can be read
            against the noise scale of its own sample rather than only against
            the conventional ladder. It assumes independent observations, which
            a cross-section of equities is not, so it is a lower bound on noise.
        null_standard_deviation: ``sqrt(2 * (n_bins - 1)) / n_observed_present``,
            the matching noise scale, same caveat.
    """

    reference: ReferenceDistribution
    value: float
    bins: tuple[BinContribution, ...]
    availability: NanRateShift
    n_observed_total: int
    n_observed_present: int
    null_expected_value: float
    null_standard_deviation: float

    @property
    def feature(self) -> str:
        """The monitored feature's name."""
        return self.reference.feature

    @property
    def reference_id(self) -> str:
        """Identifier of the reference distribution the value was measured against."""
        return self.reference.reference_id

    @property
    def floored_bins(self) -> tuple[int, ...]:
        """Indices of bins where the epsilon floor was applied.

        Non-empty means part of :attr:`value` is a consequence of
        :attr:`~ReferenceDistribution.epsilon` rather than of the data — an
        emptied decile contributes about 1.15 at the default floor. Read
        alongside :attr:`floor_contribution`.
        """
        return tuple(contribution.index for contribution in self.bins if contribution.floored)

    @property
    def floor_contribution(self) -> float:
        """PSI supplied by floored bins (dimensionless, part of :attr:`value`).

        Always a ``float``, including when no bin was floored. A bare ``sum()``
        over an empty selection returns the integer ``0``, which serialises to
        ``0`` rather than ``0.0`` in :meth:`to_dict` and would make a dashboard
        column change type between dates — the sort of difference a JSON
        consumer notices and a reader does not.
        """
        return math.fsum(
            contribution.contribution for contribution in self.bins if contribution.floored
        )

    @property
    def observed_fractions(self) -> tuple[float, ...]:
        """Live share of each bin (fractions summing to 1 over present values)."""
        return tuple(contribution.observed_fraction for contribution in self.bins)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the result as a JSON-safe mapping, reference and floor included."""
        return {
            "feature": self.feature,
            "value": self.value,
            "units": "dimensionless (Jeffreys divergence between two histograms)",
            "reference": self.reference.to_dict(),
            "bins": [contribution.to_dict() for contribution in self.bins],
            "availability": self.availability.to_dict(),
            "n_observed_total": self.n_observed_total,
            "n_observed_present": self.n_observed_present,
            "epsilon": self.reference.epsilon,
            "floored_bins": list(self.floored_bins),
            "floor_contribution": self.floor_contribution,
            "null_expected_value": self.null_expected_value,
            "null_standard_deviation": self.null_standard_deviation,
        }


def _availability_shift(
    *, reference: ReferenceDistribution, n_observed_total: int, n_observed_absent: int
) -> NanRateShift:
    """Compare the live NaN rate against the reference's (module docstring §5).

    Args:
        reference: the frozen reference, which records its own absent count.
        n_observed_total: live observations, present and absent (count).
        n_observed_absent: live observations that were ``NaN`` (count).

    Returns:
        A :class:`NanRateShift` carrying the two rates and their two-category
        PSI.
    """
    reference_absent = reference.absent_fraction
    observed_absent = n_observed_absent / n_observed_total
    absent_term, absent_reference_floored, absent_observed_floored = _psi_term(
        reference_absent, observed_absent, epsilon=reference.epsilon
    )
    present_term, present_reference_floored, present_observed_floored = _psi_term(
        _MAX_ABSENT_FRACTION - reference_absent,
        _MAX_ABSENT_FRACTION - observed_absent,
        epsilon=reference.epsilon,
    )
    return NanRateShift(
        reference_absent_fraction=reference_absent,
        observed_absent_fraction=observed_absent,
        n_reference=reference.n_total,
        n_observed=n_observed_total,
        value=absent_term + present_term,
        floored=any(
            (
                absent_reference_floored,
                absent_observed_floored,
                present_reference_floored,
                present_observed_floored,
            )
        ),
        null_expected_value=(AVAILABILITY_CATEGORIES - 1) / n_observed_total,
    )


def population_stability_index(
    *, reference: ReferenceDistribution, observed: npt.ArrayLike
) -> PSIResult:
    """Measure how far a live cross-section has drifted from its reference.

    The comparison uses the reference's **own** bin edges and **own** bin
    fractions, both fixed when the reference was built. There is deliberately no
    overload that takes two raw samples: recomputing the bins from the live
    sample would put ``1/B`` of it in each bin by construction and collapse the
    statistic to its noise floor whatever the data did (module docstring §1).

    ``NaN`` is not dropped silently. It is counted, compared against the
    reference's own availability rate as a separate two-category PSI, and
    returned on :attr:`PSIResult.availability`. The distribution PSI itself is
    computed over the present values only, because a bin index cannot be
    assigned to a value that does not exist.

    Args:
        reference: the frozen training-period distribution, from
            :meth:`ReferenceDistribution.from_sample`.
        observed: the live cross-section — one value per security, one date, in
            the feature's declared units. ``NaN`` means "not available";
            infinities are refused. Order is irrelevant.

    Returns:
        A :class:`PSIResult` carrying the value, the per-bin contributions, the
        availability shift, the epsilon floor and which bins it touched, and the
        null-hypothesis noise scale for this sample size.

    Raises:
        MonitoringInputError: if ``observed`` is not a one-dimensional numeric
            array or contains an infinity.
        InsufficientSampleError: if there are too few observations for the
            statistic to be a measurement rather than noise — including the
            no-data case, which raises rather than returning ``0.0``, since a
            PSI of zero reads as "no drift", the most dangerous wrong answer a
            detector can give (I3). When the sample is large enough to compare
            availability but too small after removing ``NaN`` values, the
            availability shift is attached to the exception.

    Example:
        >>> import numpy as np
        >>> from backend.tracking.stamp import ReproducibilityStamp
        >>> stamp = ReproducibilityStamp(
        ...     git_commit="0" * 40,
        ...     git_dirty=False,
        ...     data_version="fixture",
        ...     config_hash="0" * 64,
        ...     seed=0,
        ... )
        >>> training = np.linspace(-3.0, 3.0, 500)
        >>> reference = ReferenceDistribution.from_sample(
        ...     training,
        ...     feature="example",
        ...     units="dimensionless z-score",
        ...     reference_id="fixture_reference",
        ...     stamp=stamp,
        ...     n_bins=4,
        ... )
        >>> result = population_stability_index(reference=reference, observed=training)
        >>> result.value
        0.0
    """
    array = _as_float_1d(observed, name="observed")
    n_total = int(array.size)
    present = array[~np.isnan(array)]
    n_present = int(present.size)

    availability_minimum = minimum_sample_size(AVAILABILITY_CATEGORIES)
    if n_total < availability_minimum:
        raise InsufficientSampleError(
            feature=reference.feature,
            reference_id=reference.reference_id,
            quantity="observations",
            n_observations=n_total,
            minimum=availability_minimum,
            n_bins=AVAILABILITY_CATEGORIES,
        )
    availability = _availability_shift(
        reference=reference, n_observed_total=n_total, n_observed_absent=n_total - n_present
    )

    distribution_minimum = minimum_sample_size(reference.n_bins)
    if n_present < distribution_minimum:
        raise InsufficientSampleError(
            feature=reference.feature,
            reference_id=reference.reference_id,
            quantity="present (non-NaN) observations",
            n_observations=n_present,
            minimum=distribution_minimum,
            n_bins=reference.n_bins,
            availability=availability,
        )

    counts = _bin_counts(reference.edges, present)
    reference_fractions = reference.fractions
    contributions: list[BinContribution] = []
    total = 0.0
    for index, count in enumerate(counts):
        observed_fraction = int(count) / n_present
        term, reference_floored, observed_floored = _psi_term(
            reference_fractions[index], observed_fraction, epsilon=reference.epsilon
        )
        total += term
        contributions.append(
            BinContribution(
                index=index,
                lower=reference.edges[index],
                upper=reference.edges[index + 1],
                reference_count=reference.bin_counts[index],
                reference_fraction=reference_fractions[index],
                observed_count=int(count),
                observed_fraction=observed_fraction,
                contribution=term,
                floored=reference_floored or observed_floored,
            )
        )
    degrees_of_freedom = reference.n_bins - 1
    return PSIResult(
        reference=reference,
        value=total,
        bins=tuple(contributions),
        availability=availability,
        n_observed_total=n_total,
        n_observed_present=n_present,
        null_expected_value=degrees_of_freedom / n_present,
        null_standard_deviation=math.sqrt(2.0 * degrees_of_freedom) / n_present,
    )
