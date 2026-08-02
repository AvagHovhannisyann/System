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

The same policy, one step harder, governs P12.1
-----------------------------------------------

The live-versus-expected comparison has a *third* dangerous answer on top of the
two above, and it is the worst of them: **"within the expected band"** produced
by a comparison that never happened. A drift detector that silently reports zero
mislabels a dashboard tile; an auto-halt that silently reports "in band" leaves
capital deployed against a strategy nobody is checking. So
:class:`ComparisonUnavailableError` exists for every condition under which the
comparison cannot be made — no band, no live sample, a live window of the wrong
length, stale data — and :func:`backend.monitoring.expectation.decide` converts
it into a **halt**, never into a pass (directive §2 I3, §5 Phase 12).

:class:`ComparisonUnavailableError` is deliberately **not** a ``ValueError``, for
the same reason :class:`InsufficientSampleError` is not: an ``except ValueError``
somewhere up the call stack must not be able to swallow the one refusal whose
suppression means trading continues unmonitored.

The alerting failures (P12.4) follow from one sentence: *an alert nobody
acknowledges is a log line*. :class:`AlertUndeliverableError` is what a
dispatcher raises when every channel refused an alert — because the alternative,
returning normally after a failed delivery, is the "silent drop" the alerting
task exists to prevent — and :class:`UnacknowledgedHaltError` is what refuses to
resume trading from a halt whose alert nobody has signed for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.monitoring.psi import NanRateShift

__all__ = [
    "AlertDeliveryError",
    "AlertError",
    "AlertUndeliverableError",
    "AlreadyAcknowledgedError",
    "ComparisonUnavailableError",
    "CostBasisError",
    "DriftBandError",
    "ExpectationBandError",
    "HaltHistoryError",
    "InsufficientSampleError",
    "MonitoringError",
    "MonitoringInputError",
    "ReferenceDistributionError",
    "SystemHaltedError",
    "UnacknowledgedHaltError",
    "UnknownAlertError",
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


class ExpectationBandError(MonitoringError, ValueError):
    """Raised when an expectation band cannot be built from a CPCV distribution.

    Covers a malformed path matrix, an evaluation window longer than the paths
    or shorter than :data:`~backend.monitoring.expectation.MINIMUM_WINDOW_PERIODS`,
    a window statistic that is undefined on some window, an artefact reference
    missing the identity invariant I2 requires, and — the refusal that matters —
    a requested tail mass **finer than the path distribution can resolve**.

    That last one deserves its own sentence. An empirical quantile at ``0.005``
    taken from 40 effectively independent windows is not a 0.5% quantile; it is
    the smallest of 40 numbers wearing one. Halting on it would be halting on an
    order statistic whose sampling error is larger than the band it defines, so
    the band is refused and the caller is told to lengthen the sample, shorten
    the window, or accept a larger false-halt budget. Widening the band silently
    to whatever the data supports would be the same error one level down.
    """


class CostBasisError(MonitoringError, ValueError):
    """Raised when live and expected performance are not both net of costs (I4).

    Directive §2 I4 and §9.6 make this non-negotiable: no result is reported
    gross. For a *comparison* the requirement is sharper than for a report,
    because the failure is silent and directional — a gross live series compared
    against a net-of-cost backtest band will sit at or above the band for as
    long as costs are positive, which reads as "the strategy is doing fine" and
    is the precise reading that keeps a losing strategy running.

    Both sides therefore declare a
    :class:`~backend.monitoring.expectation.CostTreatment`, and anything other
    than net on both sides raises. ``UNKNOWN`` raises too: an undeclared cost
    basis is not evidence of a net one.
    """


class ComparisonUnavailableError(MonitoringError):
    """Raised when live-versus-expected cannot be evaluated at all.

    Not a ``ValueError`` — see the module docstring. The conditions are: no band
    for the running strategy, no live return sample, a live window whose length
    does not match the horizon the band was built for, live data staler than the
    policy permits, or any failure inside the comparison itself.

    The class exists so :func:`backend.monitoring.expectation.decide` has exactly
    one thing to catch and exactly one thing to do with it — **halt**. It never
    carries a numeric result and there is no attribute on it that reads as a
    comparison outcome, because "unavailable" must not be renderable as
    "in band" (I3).

    Attributes:
        reason: what could not be done, in the checker's own words.
    """

    def __init__(self, reason: str) -> None:
        """Build the refusal from the condition that produced it.

        Args:
            reason: what could not be done, including the numbers behind it.
        """
        self.reason = reason
        super().__init__(reason)


class AlertError(MonitoringError):
    """Base class for alert-pipeline failures (P12.4)."""


class AlertDeliveryError(AlertError):
    """Raised by an :class:`~backend.monitoring.alerts.AlertChannel` that failed.

    A channel raises this instead of returning: a delivery function that can
    return ``False`` is a delivery function whose caller can ignore the result,
    and the ignored path is the one that loses the alert.

    Attributes:
        channel_id: the channel that failed.
        detail: why, verbatim.
    """

    def __init__(self, *, channel_id: str, detail: str) -> None:
        """Build the failure from the channel that raised it.

        Args:
            channel_id: identifier of the channel.
            detail: why delivery failed.
        """
        self.channel_id = channel_id
        self.detail = detail
        super().__init__(f"channel {channel_id!r} failed to deliver: {detail}")


class AlertUndeliverableError(AlertError):
    """Raised when every configured channel refused an alert.

    The alert is **already persisted** when this is raised — dispatch writes
    before it delivers — so the exception means "nobody was told", never "the
    alert is gone". The escalation state is on the stored row and the failed
    attempts are rows of their own, so an operator can ask the database which
    alerts nobody received. Raising as well as recording is deliberate: a caller
    that ignores the return value of a dispatch call would otherwise treat total
    delivery failure as success.

    Attributes:
        dedup_key: the alert's condition key.
        failures: ``(channel_id, detail)`` for every channel that refused.
    """

    def __init__(self, *, dedup_key: str, failures: tuple[tuple[str, str], ...]) -> None:
        """Build the escalation from the channel failures that caused it.

        Args:
            dedup_key: condition key of the undelivered alert.
            failures: one ``(channel_id, detail)`` pair per failed channel.
        """
        self.dedup_key = dedup_key
        self.failures = failures
        rendered = (
            "; ".join(f"{channel}: {detail}" for channel, detail in failures) or "no channels"
        )
        super().__init__(
            f"alert {dedup_key} was persisted and escalated but reached nobody ({rendered}). "
            f"It is recorded as undelivered rather than dropped; an operator query on "
            f"undelivered alerts will return it."
        )


class AlreadyAcknowledgedError(AlertError):
    """Raised when an alert that already carries an acknowledgement is acknowledged again.

    Acknowledgement is a *fact about a person*, not a flag. Overwriting it would
    silently rewrite who took responsibility for an alert, so the second
    acknowledgement is refused and the first is carried on the exception.

    Attributes:
        dedup_key: the alert's condition key.
        acknowledged_by: who acknowledged it first.
    """

    def __init__(self, *, dedup_key: str, acknowledged_by: str) -> None:
        """Build the refusal from the incumbent acknowledgement.

        Args:
            dedup_key: condition key of the alert.
            acknowledged_by: who already acknowledged it.
        """
        self.dedup_key = dedup_key
        self.acknowledged_by = acknowledged_by
        super().__init__(
            f"alert {dedup_key} was already acknowledged by {acknowledged_by!r}; an "
            f"acknowledgement records who took responsibility and is never overwritten"
        )


class UnknownAlertError(AlertError, LookupError):
    """Raised when an alert referenced by key or id is not in the store."""


class HaltHistoryError(MonitoringError, ValueError):
    """Raised when a halt-history write would produce an inconsistent history.

    Covers resuming a halt that is already resumed, resuming an event that is
    not a halt, and recording a resume with no actor or no reason. An
    auto-halt's *cause* is a machine decision; the decision to resume is a human
    one and the history says which human, or it is not recorded.
    """


class UnacknowledgedHaltError(HaltHistoryError):
    """Raised when trading would resume from a halt whose alert nobody signed for.

    This is what makes acknowledgement load-bearing rather than decorative
    (directive §6.10). A halt raised an alert; if that alert has no
    acknowledgement row, nobody has yet stated they looked at it, and resuming
    would turn the entire alerting pipeline into a log nobody reads.

    Attributes:
        halt_event_id: the halt that cannot be resumed yet.
        dedup_key: the unacknowledged alert's condition key.
    """

    def __init__(self, *, halt_event_id: int, dedup_key: str) -> None:
        """Build the refusal from the halt and its unacknowledged alert.

        Args:
            halt_event_id: id of the halt event.
            dedup_key: condition key of the unacknowledged alert.
        """
        self.halt_event_id = halt_event_id
        self.dedup_key = dedup_key
        super().__init__(
            f"halt {halt_event_id} cannot be resumed: its alert {dedup_key} has no "
            f"acknowledgement. Someone must sign for the alert before trading resumes, or "
            f"the alert was never read by anyone."
        )


class SystemHaltedError(MonitoringError):
    """Raised by the monitoring-side halt gate when a halt is in force.

    This is the seam the execution-side kill switch reads (see
    :mod:`backend.monitoring.history`). It is an exception rather than a boolean
    on purpose: a boolean has a value a caller can forget to check, and the
    forgotten check leaves orders flowing during a halt.

    Attributes:
        halt_event_id: the halt in force.
        cause: why it was raised.
        occurred_at_iso: when, ISO-8601 UTC.
    """

    def __init__(self, *, halt_event_id: int, cause: str, occurred_at_iso: str) -> None:
        """Build the gate refusal from the active halt.

        Args:
            halt_event_id: id of the active halt event.
            cause: the halt cause.
            occurred_at_iso: when the halt was recorded, ISO-8601 UTC.
        """
        self.halt_event_id = halt_event_id
        self.cause = cause
        self.occurred_at_iso = occurred_at_iso
        super().__init__(
            f"trading is halted: halt event {halt_event_id}, cause {cause!r}, raised at "
            f"{occurred_at_iso}. A halt is cleared by an operator resume, never by a retry."
        )
