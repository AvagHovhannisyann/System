"""Failure taxonomy for the monitoring package (directive §5 Phase 12).

A monitor is not an ordinary computation, and its failure policy is not the
ordinary one. Everywhere else in this platform an unavailable number is
reported as ``NaN`` — "not available" — and the pipeline continues. A drift
detector may not do that, because the two values a broken detector is most
likely to emit are the two values that read as *good news*:

* ``0.0`` reads as "this feature has not moved";
* ``NaN`` renders as a blank cell in a dashboard and reads as "nothing to
  report".

Both are the most dangerous possible wrong answers (directive §2 I3, §9.1-9.2).
So every condition under which a Population Stability Index cannot be *measured*
raises, carrying the numbers that produced the refusal:

* :class:`MonitoringInputError` — the array is not a cross-section, or contains
  an infinity, which is the residue of a division by zero upstream rather than
  a value in a distribution;
* :class:`ReferenceDistributionError` — a reference distribution cannot be
  built, or is malformed. Without a reference there is no comparison and no PSI;
* :class:`InsufficientSampleError` — there is data, but not enough of it for the
  statistic to mean anything. This is the refusal that matters most: PSI on a
  small cross-section is dominated by multinomial sampling noise, and a number
  produced there would be read as a signal;
* :class:`DriftBandError` — the alerting thresholds are not a usable ladder.

Every error names the feature and the reference distribution it was measured
against, because a drift refusal that does not say *which* reference is as
uninterpretable as a drift number that does not (invariant I2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.monitoring.psi import NanRateShift

__all__ = [
    "DriftBandError",
    "InsufficientSampleError",
    "MonitoringError",
    "MonitoringInputError",
    "ReferenceDistributionError",
]


class MonitoringError(Exception):
    """Base class for every failure raised by :mod:`backend.monitoring`."""


class MonitoringInputError(MonitoringError, ValueError):
    """Raised when an argument cannot describe a distribution at all.

    Covers a two-dimensional panel where a cross-section was expected, a
    non-numeric array, an infinity among the values, and a bin count or epsilon
    outside its admissible range. These are facts about the calling code rather
    than about one date's data, so they are wrong on every date or none and
    failing loudly on the first is the only useful behaviour.

    ``+inf`` is refused rather than binned into the top bucket, matching
    :func:`backend.features._stats.reject_infinities`: an infinity is not an
    extreme observation, it is a division by zero that has not been dealt with,
    and binning it would launder it into a plausible tail count.
    """


class ReferenceDistributionError(MonitoringError, ValueError):
    """Raised when a reference distribution cannot be built or is malformed.

    A PSI is a comparison against a *named, frozen* reference. If that reference
    cannot be constructed — too few observations to place quantile edges, a
    feature so concentrated on mass points that its quantiles collapse, a bin
    that no reference observation falls in, edges that do not increase — then
    there is nothing to compare against and no number to report.

    The refusal is deliberately not softened into "use whatever bins we managed
    to compute". Bins that do not describe the reference produce a PSI whose
    baseline is arbitrary, and that number is indistinguishable, downstream,
    from a measured one.
    """


class InsufficientSampleError(MonitoringError):
    """Raised when a sample is too small for PSI to be a measurement.

    Under the null hypothesis of no drift, the PSI of a sample of ``n``
    observations across ``B`` bins has expectation approximately ``(B - 1) / n``
    purely from multinomial sampling noise (see
    :func:`backend.monitoring.psi.minimum_sample_size` for the derivation). With
    ten bins and one hundred names that is ``0.09`` — within rounding of the
    conventional ``0.10`` "moderate drift" threshold — from a distribution that
    has not moved at all.

    So below the minimum this raises rather than returning a number. A caller
    that wants a monitor on a small universe must reduce the bin count, not
    lower the sample threshold: the trade is visible in the reference
    distribution's own definition either way.

    Attributes:
        feature: name of the feature being monitored.
        reference_id: identifier of the reference distribution.
        quantity: what was too small, in words ("observations",
            "present (non-NaN) observations").
        n_observations: how many were supplied (count).
        minimum: how many are required (count).
        n_bins: the bin count the minimum was derived from.
        availability: the NaN-rate comparison, when it could still be made.
            Present when the sample is large enough to talk about *availability*
            but too small — after removing absent values — to talk about the
            *distribution*. That is the signature of an upstream source that has
            broken rather than drifted, and it is carried on the exception so
            the caller receives the finding instead of only the refusal.
    """

    def __init__(
        self,
        *,
        feature: str,
        reference_id: str,
        quantity: str,
        n_observations: int,
        minimum: int,
        n_bins: int,
        availability: NanRateShift | None = None,
    ) -> None:
        """Build the error from the sample that was too small.

        Args:
            feature: name of the feature being monitored.
            reference_id: identifier of the reference distribution.
            quantity: what was counted, in words.
            n_observations: how many were supplied (count).
            minimum: how many are required (count).
            n_bins: bin count the minimum was derived from.
            availability: NaN-rate comparison, if one could be made.
        """
        self.feature = feature
        self.reference_id = reference_id
        self.quantity = quantity
        self.n_observations = n_observations
        self.minimum = minimum
        self.n_bins = n_bins
        self.availability = availability
        expected_noise = (n_bins - 1) / n_observations if n_observations > 0 else float("inf")
        detail = ""
        if availability is not None:
            detail = (
                f" The NaN rate moved from {availability.reference_absent_fraction:.4f} to "
                f"{availability.observed_absent_fraction:.4f} (availability PSI "
                f"{availability.value:.4f}); that shift is itself the finding, and it is "
                f"attached to this exception as `.availability`."
            )
        super().__init__(
            f"refusing to compute PSI for feature {feature!r} against reference "
            f"{reference_id!r}: {n_observations} {quantity} is below the minimum of "
            f"{minimum} for {n_bins} bins. Multinomial noise alone would give a PSI of "
            f"about {expected_noise:.4f} here with no drift whatsoever, so a number "
            f"computed from this sample would be noise presented as a signal. Zero is "
            f"not returned either: for a detector, 'no drift' is the most dangerous "
            f"wrong answer (I3).{detail}"
        )


class DriftBandError(MonitoringError, ValueError):
    """Raised when the alerting thresholds do not form a usable ladder.

    The bands are configurable precisely because the conventional 0.10 / 0.25
    cut points are credit-scoring folklore rather than derived quantities (see
    :class:`backend.monitoring.drift.DriftBands`). Configurable does not mean
    unconstrained: a ladder whose "major" threshold sits below its "moderate"
    one, or whose thresholds are non-positive or non-finite, classifies nothing
    and would silently mark every measurement as the same band.
    """
