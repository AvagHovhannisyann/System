"""Drift reporting: thresholds, classification, and a report that names itself (P12.2).

:mod:`backend.monitoring.psi` measures. This module *reports*, and the two are
separated because they have different epistemic status. The PSI is arithmetic
over two histograms and is as true as the data. The ladder that turns it into
"stable / moderate / major" is convention, and pretending otherwise is how a
folklore constant acquires the authority of a measurement.

--------------------------------------------------------------------------
1. The thresholds are convention and say so in their own payload
--------------------------------------------------------------------------

The familiar cut points — below 0.10 stable, 0.10 to 0.25 moderate, above 0.25
major — come from credit-scorecard practice (quoted without derivation in
Siddiqi, *Credit Risk Scorecards*, 2006, and repeated in vendor documentation
since). They are not derived from any distributional result, they carry no
significance level, and they are not invariant to the bin count: the same shift
scored over twenty bins produces a larger PSI than over ten, so a ladder
calibrated on deciles does not transfer.

So :class:`DriftBands` holds them as data rather than as literals in a
comparison, requires a ``basis`` string that states where they came from, ships
that string in :meth:`DriftBands.to_dict`, and lets a caller replace the numbers
with ones derived for their own bin count and universe size. What it will not
accept is a ladder that classifies nothing — a "major" threshold at or below
"moderate", or a non-positive or non-finite one — because such a ladder marks
every measurement the same band and the report still looks populated
(:class:`~backend.monitoring.errors.DriftBandError`).

Alongside the band, every measured feature carries
:attr:`~backend.monitoring.psi.PSIResult.null_expected_value` — the PSI this
same sample would have shown with **no drift at all**, from multinomial sampling
noise. That number is derived rather than conventional, and where it approaches
the moderate threshold it is the honest reading of the band: the ladder is
saying more than the sample can support.

--------------------------------------------------------------------------
2. A report names its references, or it is not interpretable (I2)
--------------------------------------------------------------------------

"``momentum_12_1`` PSI 0.31" is not a finding. "``momentum_12_1`` PSI 0.31,
measured against reference ``train_2015_2020``, fingerprint ``a3f1…``, data
version ``…``" is. Two PSI values are comparable only when they were measured on
the same partition with the same floor, and the fingerprint is the only thing
that establishes that — reference *identifiers* can be reused across re-cuts,
and two references with the same name and different edges produce numbers that
look like a time series and are not one.

:class:`DriftReport` therefore carries:

* the :class:`~backend.tracking.stamp.ReproducibilityStamp` of the run that
  performed the measurement — git commit, data version, config hash, seed, the
  four components of I2 — with no default and no "unknown" fallback; and
* for every feature, the full reference identity, through
  :meth:`~backend.monitoring.psi.PSIResult.to_dict`.

--------------------------------------------------------------------------
3. A feature that could not be measured is reported as unmeasured, not as zero
--------------------------------------------------------------------------

A panel report over thirty features should not be destroyed because one feature
had forty names on one date. But the alternative — recording ``0.0`` for it —
is the exact failure the whole package exists to prevent, because zero reads as
"this feature has not moved".

So a refusal from :func:`~backend.monitoring.psi.population_stability_index`
becomes an :class:`UnmeasurableFeature` entry: it carries the feature, the
reference it *would* have been measured against, the reason in the detector's
own words, and — when the sample was large enough to compare availability but
too small to bin — the availability shift that was still measurable, which is
usually the actual finding. It carries **no PSI value and no band**, there is no
attribute on it a template could render as a number, and
:attr:`DriftReport.complete` is false while any exist.

:class:`~backend.monitoring.errors.MonitoringInputError` is deliberately *not*
caught. A two-dimensional array or an infinity is a fact about the calling code
rather than about one date's data — wrong on every date or none — so it
propagates rather than becoming thirty identical refusal rows.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from backend.monitoring.errors import DriftBandError, InsufficientSampleError
from backend.monitoring.psi import (
    JsonValue,
    NanRateShift,
    PSIResult,
    ReferenceDistribution,
    population_stability_index,
)
from backend.tracking.stamp import ReproducibilityStamp

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "CONVENTIONAL_BANDS_BASIS",
    "DEFAULT_MAJOR_THRESHOLD",
    "DEFAULT_MODERATE_THRESHOLD",
    "DriftBand",
    "DriftBands",
    "DriftReport",
    "FeatureDrift",
    "UnmeasurableFeature",
    "drift_report",
    "measure_feature_drift",
]

DEFAULT_MODERATE_THRESHOLD: Final = 0.10
"""Conventional lower cut point of the drift ladder (dimensionless PSI).

Convention, not a derived quantity — see the module docstring §1 and
:data:`CONVENTIONAL_BANDS_BASIS`. Calibrated against ten bins; it does not
transfer to a different bin count unchanged.
"""

DEFAULT_MAJOR_THRESHOLD: Final = 0.25
"""Conventional upper cut point of the drift ladder (dimensionless PSI).

Convention, not a derived quantity. See :data:`DEFAULT_MODERATE_THRESHOLD`.
"""

CONVENTIONAL_BANDS_BASIS: Final = (
    "credit-scorecard convention (Siddiqi, Credit Risk Scorecards, 2006); not derived from "
    "any distributional result, carries no significance level, and was calibrated against "
    "ten bins — a finer partition finds more divergence for the same shift, so this ladder "
    "does not transfer to a different bin count unchanged. Compare against the sample's own "
    "null_expected_value before treating a band as a finding."
)
"""What the default thresholds are, stated in the payload that carries them."""


class DriftBand(StrEnum):
    """Where a PSI falls on the configured ladder.

    ``StrEnum`` so a member is usable directly as the text stored in a report
    row or rendered in the dashboard, without a mapping that could disagree with
    the enum.

    Attributes:
        STABLE: below the moderate threshold.
        MODERATE: at or above the moderate threshold, below the major one.
        MAJOR: at or above the major threshold.
    """

    STABLE = "stable"
    MODERATE = "moderate"
    MAJOR = "major"


_BAND_ORDER: Final = (DriftBand.STABLE, DriftBand.MODERATE, DriftBand.MAJOR)


def _worst(bands: Sequence[DriftBand]) -> DriftBand:
    """Return the most severe band in ``bands``, or ``STABLE`` if empty.

    Args:
        bands: bands to reduce.

    Returns:
        The band with the highest severity.
    """
    return max(bands, key=_BAND_ORDER.index, default=DriftBand.STABLE)


@dataclass(frozen=True, slots=True)
class DriftBands:
    """The threshold ladder, as data that states where it came from.

    Attributes:
        moderate: PSI at or above which a feature is called ``MODERATE``
            (dimensionless). Must be finite and strictly positive.
        major: PSI at or above which a feature is called ``MAJOR``
            (dimensionless). Must be finite and strictly greater than
            :attr:`moderate`.
        basis: where these numbers came from, in prose. Travels with the report
            so a reader is never shown a band without being told the ladder is a
            convention. Must not be blank.
    """

    moderate: float = DEFAULT_MODERATE_THRESHOLD
    major: float = DEFAULT_MAJOR_THRESHOLD
    basis: str = CONVENTIONAL_BANDS_BASIS

    def __post_init__(self) -> None:
        """Refuse a ladder that cannot classify.

        Raises:
            DriftBandError: if either threshold is non-numeric, non-finite or
                non-positive, if ``major`` is not strictly above ``moderate``,
                or if ``basis`` is blank. Each of these leaves every measurement
                in one band while the report still looks populated.
        """
        for field_name, value in (("moderate", self.moderate), ("major", self.major)):
            supplied: object = value
            if isinstance(supplied, bool) or not isinstance(supplied, (int, float)):
                msg = f"{field_name} threshold must be a real number, got {type(supplied).__name__}"
                raise DriftBandError(msg)
            if not math.isfinite(value):
                msg = (
                    f"{field_name} threshold must be finite; got {value!r}. An infinite or "
                    f"NaN threshold classifies every measurement into one band."
                )
                raise DriftBandError(msg)
            if value <= 0.0:
                msg = (
                    f"{field_name} threshold must be strictly positive; got {value!r}. PSI is "
                    f"non-negative, so a threshold at or below zero fires on every sample "
                    f"including an identical one."
                )
                raise DriftBandError(msg)
        if not self.major > self.moderate:
            msg = (
                f"major threshold {self.major!r} must be strictly above moderate "
                f"{self.moderate!r}; a ladder whose rungs are out of order or equal never "
                f"reports the middle band and would silently classify a moderate shift as "
                f"major or the reverse."
            )
            raise DriftBandError(msg)
        supplied_basis: object = self.basis
        if not isinstance(supplied_basis, str) or not supplied_basis.strip():
            msg = (
                "basis is blank. These thresholds are a convention, not a derived quantity, "
                "and a report that shows a band without saying so lends folklore the "
                "authority of a measurement."
            )
            raise DriftBandError(msg)

    def classify(self, value: float) -> DriftBand:
        """Place a PSI on the ladder.

        Boundaries are inclusive on the upper band: a value exactly equal to
        :attr:`moderate` is ``MODERATE``, exactly :attr:`major` is ``MAJOR``.
        The choice is arbitrary but must be made somewhere, and this direction
        errs toward reporting drift rather than away from it.

        Args:
            value: a dimensionless PSI, finite and non-negative.

        Returns:
            The band ``value`` falls in.

        Raises:
            DriftBandError: if ``value`` is not a finite, non-negative real
                number. ``NaN`` compares false against every threshold and would
                otherwise be classified ``STABLE`` — a blank cell reading as
                "nothing to report", which is the failure this package exists to
                prevent.
        """
        supplied: object = value
        if isinstance(supplied, bool) or not isinstance(supplied, (int, float)):
            msg = f"value must be a real number, got {type(supplied).__name__}"
            raise DriftBandError(msg)
        if not math.isfinite(value) or value < 0.0:
            msg = (
                f"cannot classify PSI={value!r}: a band is only meaningful for a finite, "
                f"non-negative divergence. NaN in particular compares false against every "
                f"threshold and would be reported as 'stable'."
            )
            raise DriftBandError(msg)
        if value >= self.major:
            return DriftBand.MAJOR
        if value >= self.moderate:
            return DriftBand.MODERATE
        return DriftBand.STABLE

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the ladder as a JSON-safe mapping, basis included."""
        return {
            "moderate": float(self.moderate),
            "major": float(self.major),
            "basis": self.basis,
            "units": "dimensionless PSI",
            "boundary": "inclusive on the upper band (value >= threshold enters it)",
        }


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """One feature's measured drift, banded.

    Attributes:
        psi: the measurement, carrying its own reference identity, per-bin
            contributions, epsilon floor and availability shift.
        bands: the ladder the classification used, carried here rather than
            looked up, so a stored finding can be re-read without the config
            that produced it.
    """

    psi: PSIResult
    bands: DriftBands

    @property
    def feature(self) -> str:
        """The monitored feature's name."""
        return self.psi.feature

    @property
    def reference_id(self) -> str:
        """Identifier of the reference this feature was measured against (I2)."""
        return self.psi.reference_id

    @property
    def reference_fingerprint(self) -> str:
        """Digest of the reference's edges, counts and floor (I2)."""
        return self.psi.reference.fingerprint

    @property
    def distribution_band(self) -> DriftBand:
        """Band of the distributional PSI over the present values."""
        return self.bands.classify(self.psi.value)

    @property
    def availability_band(self) -> DriftBand:
        """Band of the separate NaN-rate PSI.

        Classified on the same ladder but never summed with
        :attr:`distribution_band`'s value: a large availability break must not
        be able to hide inside a moderate distributional number, nor the
        reverse (:mod:`backend.monitoring.psi` §5).
        """
        return self.bands.classify(self.psi.availability.value)

    @property
    def band(self) -> DriftBand:
        """The more severe of the distribution and availability bands."""
        return _worst((self.distribution_band, self.availability_band))

    @property
    def exceeds_sampling_noise(self) -> bool:
        """Whether the measured PSI is above this sample's own null expectation.

        ``False`` means the value is within what multinomial noise alone would
        produce from a distribution that has not moved (:mod:`backend.monitoring.psi`
        §3) — a band assigned to such a value is the ladder speaking, not the
        data. The comparison assumes independent observations, which a
        cross-section of equities is not, so it is a weak test in the
        permissive direction.
        """
        return self.psi.value > self.psi.null_expected_value

    @property
    def floor_driven(self) -> bool:
        """Whether the epsilon floor supplied most of the PSI.

        True when floored (empty) bins contributed more than half the total. The
        value is still a real finding — a bin emptying *is* drift — but its
        magnitude is set by :attr:`~backend.monitoring.psi.ReferenceDistribution.epsilon`
        rather than measured, so it must not be compared against a value from a
        reference with a different floor.
        """
        return self.psi.floor_contribution > 0.5 * self.psi.value

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the finding as a JSON-safe mapping, reference identity included."""
        return {
            "measured": True,
            "feature": self.feature,
            "reference_id": self.reference_id,
            "reference_fingerprint": self.reference_fingerprint,
            "band": str(self.band),
            "distribution_band": str(self.distribution_band),
            "availability_band": str(self.availability_band),
            "exceeds_sampling_noise": self.exceeds_sampling_noise,
            "floor_driven": self.floor_driven,
            "bands": self.bands.to_dict(),
            "psi": self.psi.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class UnmeasurableFeature:
    """A feature the detector refused to put a number on.

    Deliberately carries no value and no band. There is no attribute here a
    dashboard could render as a PSI, because the whole point of the entry is
    that no PSI exists for this feature on this date and ``0.0`` would read as
    "no drift" (module docstring §3).

    Attributes:
        feature: the feature that could not be measured.
        reference_id: the reference it would have been measured against, so the
            refusal is as interpretable as a measurement would have been (I2).
        reference_fingerprint: that reference's digest.
        reason: the detector's own message, kept verbatim rather than
            summarised — it carries the counts that produced the refusal.
        quantity: *what* was too few, in the detector's words — ``"observations"``
            when the whole cross-section was too small to say anything at all,
            ``"present (non-NaN) observations"`` when it was large enough to
            compare availability but too thin after removing absent values. The
            two are different findings and the count below is meaningless
            without this word.
        n_observed: how many of :attr:`quantity` were supplied (count).
        minimum_required: how many the statistic needed (count).
        availability: the NaN-rate comparison when one could still be made.
            Present exactly when the sample was large enough to talk about
            availability but too small, after removing absent values, to talk
            about the distribution — which is the signature of an upstream
            source that broke rather than a market that moved, and is therefore
            usually the finding rather than a consolation for its absence.
    """

    feature: str
    reference_id: str
    reference_fingerprint: str
    reason: str
    quantity: str
    n_observed: int
    minimum_required: int
    availability: NanRateShift | None

    @classmethod
    def from_error(
        cls, error: InsufficientSampleError, *, reference: ReferenceDistribution
    ) -> UnmeasurableFeature:
        """Build a refusal entry from the exception the detector raised.

        Args:
            error: the refusal raised by
                :func:`~backend.monitoring.psi.population_stability_index`.
            reference: the reference the measurement would have used.

        Returns:
            An :class:`UnmeasurableFeature` carrying the refusal verbatim.
        """
        return cls(
            feature=error.feature,
            reference_id=error.reference_id,
            reference_fingerprint=reference.fingerprint,
            reason=str(error),
            quantity=error.quantity,
            n_observed=error.n_observations,
            minimum_required=error.minimum,
            availability=error.availability,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the refusal as a JSON-safe mapping, with no numeric stand-in."""
        return {
            "measured": False,
            "feature": self.feature,
            "reference_id": self.reference_id,
            "reference_fingerprint": self.reference_fingerprint,
            "reason": self.reason,
            "quantity": self.quantity,
            "n_observed": self.n_observed,
            "minimum_required": self.minimum_required,
            "availability": None if self.availability is None else self.availability.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DriftReport:
    """One date's feature-drift report, stamped and referenced (I2).

    Attributes:
        as_of: the date the observed cross-sections were drawn from.
        stamp: the I2 stamp of the run that performed the measurement — git
            commit, data version, config hash, seed. No default: a report that
            cannot say how to regenerate itself cannot be constructed.
        bands: the ladder every band in the report was assigned on.
        measured: findings for the features a PSI could be computed for, in the
            order they were supplied.
        unmeasurable: entries for the features the detector refused, in the
            order they were supplied. Non-empty means the report is incomplete
            and says so; it never means those features were stable.
    """

    as_of: dt.date
    stamp: ReproducibilityStamp
    bands: DriftBands
    measured: tuple[FeatureDrift, ...]
    unmeasurable: tuple[UnmeasurableFeature, ...]

    def __post_init__(self) -> None:
        """Validate the report's identity and feature uniqueness.

        Raises:
            DriftBandError: if ``stamp`` is not a
                :class:`~backend.tracking.stamp.ReproducibilityStamp`, ``as_of``
                is not a date, or one feature appears more than once. Two rows
                for one feature on one date are two different references quoted
                under one name, and a reader cannot tell which is which.
        """
        supplied_stamp: object = self.stamp
        if not isinstance(supplied_stamp, ReproducibilityStamp):
            msg = (
                f"stamp must be a ReproducibilityStamp, got {type(supplied_stamp).__name__}. "
                f"A drift report that cannot say which commit, data version, config and seed "
                f"produced it is not reproducible and its numbers cannot be checked (I2)."
            )
            raise DriftBandError(msg)
        supplied_date: object = self.as_of
        if not isinstance(supplied_date, dt.date) or isinstance(supplied_date, dt.datetime):
            msg = (
                f"as_of must be a datetime.date (not a datetime), got "
                f"{type(supplied_date).__name__}. A cross-section belongs to a date."
            )
            raise DriftBandError(msg)
        names = [finding.feature for finding in self.measured]
        names += [entry.feature for entry in self.unmeasurable]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = (
                f"feature(s) {duplicates} appear more than once in one report. Two rows for "
                f"one feature on one date are two references quoted under one name, and "
                f"nothing downstream can tell which number belongs to which."
            )
            raise DriftBandError(msg)

    @property
    def complete(self) -> bool:
        """Whether every requested feature was measured."""
        return not self.unmeasurable

    @property
    def worst_band(self) -> DriftBand:
        """The most severe band among the measured features.

        Reads ``STABLE`` when nothing was measured at all, which is why it must
        never be shown without :attr:`complete` beside it — and why
        :meth:`to_dict` emits the two together.
        """
        return _worst([finding.band for finding in self.measured])

    def by_band(self, band: DriftBand) -> tuple[FeatureDrift, ...]:
        """Return the measured findings in ``band``, in report order.

        Args:
            band: the band to select.

        Returns:
            The matching findings.
        """
        return tuple(finding for finding in self.measured if finding.band is band)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the whole report as a JSON-safe mapping.

        Carries the I2 stamp, the ladder and its basis, every finding with its
        reference identity, and every refusal. ``worst_band`` never travels
        without ``complete`` and ``n_unmeasurable`` next to it.
        """
        return {
            "as_of": self.as_of.isoformat(),
            "git_reference": self.stamp.git_reference,
            "data_version": self.stamp.data_version,
            "config_hash": self.stamp.config_hash,
            "seed": self.stamp.seed,
            "reproducible": self.stamp.reproducible,
            "bands": self.bands.to_dict(),
            "complete": self.complete,
            "n_measured": len(self.measured),
            "n_unmeasurable": len(self.unmeasurable),
            "worst_band": str(self.worst_band),
            "measured": [finding.to_dict() for finding in self.measured],
            "unmeasurable": [entry.to_dict() for entry in self.unmeasurable],
        }


def measure_feature_drift(
    *,
    reference: ReferenceDistribution,
    observed: npt.ArrayLike,
    bands: DriftBands | None = None,
) -> FeatureDrift:
    """Measure one feature's drift and place it on the ladder.

    A thin, deliberately transparent composition of
    :func:`~backend.monitoring.psi.population_stability_index` and
    :meth:`DriftBands.classify`. It does not catch anything: a caller measuring
    one feature wants the refusal, not a report row about it.

    Args:
        reference: the frozen, named training-period distribution.
        observed: the live cross-section — one value per security, one date, in
            the feature's declared units. ``NaN`` means "not available".
        bands: the threshold ladder. Defaults to the conventional one, whose
            ``basis`` field says it is convention.

    Returns:
        A :class:`FeatureDrift` carrying the measurement and its bands.

    Raises:
        MonitoringInputError: if ``observed`` is not a one-dimensional numeric
            cross-section, or contains an infinity.
        InsufficientSampleError: if the sample is too small for the statistic to
            be a measurement — including the no-data case, which raises rather
            than reporting ``0.0``.
    """
    result = population_stability_index(reference=reference, observed=observed)
    return FeatureDrift(psi=result, bands=DriftBands() if bands is None else bands)


def drift_report(
    *,
    as_of: dt.date,
    stamp: ReproducibilityStamp,
    observations: Sequence[tuple[ReferenceDistribution, npt.ArrayLike]],
    bands: DriftBands | None = None,
) -> DriftReport:
    """Measure a panel of features against their references and report the result.

    Each ``(reference, observed)`` pair is measured independently. A feature the
    detector refuses becomes an :class:`UnmeasurableFeature` entry rather than a
    zero, and the report's :attr:`~DriftReport.complete` flag goes false; one
    thin cross-section therefore does not destroy the other twenty-nine features'
    findings, and does not masquerade as one of them either (module docstring §3).

    :class:`~backend.monitoring.errors.MonitoringInputError` is **not** caught.
    A malformed array is a fact about the calling code, wrong on every date or
    none, so it propagates on the first occurrence instead of becoming a row.

    Args:
        as_of: the date the observed cross-sections were drawn from.
        stamp: the I2 stamp of this measurement run. Required — see
            :class:`DriftReport`.
        observations: one ``(reference, observed cross-section)`` pair per
            feature. The reference names the feature, so no separate key is
            taken and the two cannot disagree.
        bands: the threshold ladder. Defaults to the conventional one.

    Returns:
        A :class:`DriftReport`.

    Raises:
        DriftBandError: if the stamp or ``as_of`` is malformed, or one feature
            appears twice.
        MonitoringInputError: propagated from the detector for a malformed
            cross-section.
    """
    ladder = DriftBands() if bands is None else bands
    measured: list[FeatureDrift] = []
    unmeasurable: list[UnmeasurableFeature] = []
    for reference, observed in observations:
        try:
            measured.append(
                measure_feature_drift(reference=reference, observed=observed, bands=ladder)
            )
        except InsufficientSampleError as refusal:
            unmeasurable.append(UnmeasurableFeature.from_error(refusal, reference=reference))
    return DriftReport(
        as_of=as_of,
        stamp=stamp,
        bands=ladder,
        measured=tuple(measured),
        unmeasurable=tuple(unmeasurable),
    )
