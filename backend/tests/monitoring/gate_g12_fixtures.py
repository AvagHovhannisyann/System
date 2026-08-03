"""The injected conditions gate G12 is verified against, and the cycle that carries them.

**Invariant I3, restated because a gate is exactly where it would be violated.**
Every array here is drawn from ``numpy.random.default_rng`` with an explicit
seed and then *deliberately corrupted* by a named injection. Nothing here was
measured. No live feature panel exists in this repository (blocker B1) and no
live track record exists either (blocker B2), so a G12 result is a statement
about whether the monitoring stack reacts to a condition **we put there on
purpose** — never a statement about a factor, a strategy, or a market.

The three drift injections are orthogonal on purpose
-----------------------------------------------------

A detector that only notices a mean shift would pass a one-injection gate and
still be blind to the two failures that actually happen:

* :data:`INJECTED_LOCATION_SHIFT` — the whole cross-section moves by one
  reference standard deviation. The textbook case.
* :data:`INJECTED_SCALE_FACTOR` — the mean is unchanged and the dispersion
  grows. A detector comparing means reports nothing.
* :data:`INJECTED_ABSENT_FRACTION` — the values that are still present are drawn
  from the reference distribution itself, so the *shape* has genuinely not
  moved; what changed is that two fifths of the panel stopped arriving. That is
  an upstream source breaking, and D-031 is explicit that it must surface on its
  own axis rather than being averaged into the distributional number.

Each injection is applied to exactly one feature of a three-feature panel, and
the other two are drawn from their own references' distributions with different
seeds. Those two are the gate's control: they are what a false-positive detector
would light up on, and they are asserted stable in the same report as the
injected one.

The performance deviation is placed, not drawn
----------------------------------------------

:func:`deviated_live_window` builds a return series whose per-period Sharpe
ratio is *exactly* a stated number of band standard deviations below the band's
lower edge (via
:func:`~backend.tests.monitoring.expectation_fixtures.series_with_sharpe`). The
gate therefore asserts on a distance it chose, rather than drawing series until
one of them happens to fall out of the band — which would be the gate marking
its own homework.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final

from backend.monitoring.alerts import (
    MonitoringSnapshot,
    default_rules,
    dispatch,
    evaluate_rules,
)
from backend.monitoring.drift import DriftReport, drift_report
from backend.tests.monitoring.expectation_fixtures import (
    LIVE_AS_OF,
    fixture_band,
    live_window,
    series_with_sharpe,
)
from backend.tests.monitoring.fixtures import (
    OBSERVED_SIZE,
    fixture_stamp,
    gaussian_reference,
    gaussian_sample,
    with_absent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.monitoring.alerts import AlertChannel, AlertStore, DispatchResult
    from backend.monitoring.expectation import ExpectationBand, LiveWindow
    from backend.monitoring.psi import FloatArray, ReferenceDistribution

GATE_FEATURES: Final = (
    "FIXTURE_G12_control_a",
    "FIXTURE_G12_injected",
    "FIXTURE_G12_control_b",
)
"""The panel. The middle name is the only one anything is ever injected into."""

INJECTED_INDEX: Final = 1
"""Position of :data:`GATE_FEATURES`' injected feature in the panel."""

_REFERENCE_SEEDS: Final = (901, 902, 903)
"""Seeds for the three fixture training samples."""

_OBSERVED_SEEDS: Final = (9101, 9102, 9103)
"""Seeds for the three live cross-sections.

Deliberately different from :data:`_REFERENCE_SEEDS`. Re-observing the reference
sample itself would give a PSI of exactly zero and would prove nothing about the
detector's noise floor; a fresh draw from the same distribution is the honest
negative case, and it is what the gate's unshifted arm measures.
"""

INJECTED_LOCATION_SHIFT: Final = 1.0
"""Location injection, in reference standard deviations (fixture units)."""

INJECTED_SCALE_FACTOR: Final = 1.75
"""Dispersion injection, dimensionless. Applied with the location left alone."""

INJECTED_ABSENT_FRACTION: Final = 0.40
"""Availability injection: the fraction of the cross-section that stops arriving."""

INJECTED_ABSENT_COUNT: Final = int(OBSERVED_SIZE * INJECTED_ABSENT_FRACTION)
"""The availability injection as a count of observations."""

GATE_AS_OF: Final = LIVE_AS_OF
"""The date every gate report and every gate live window belongs to."""

DEVIATION_SIGMAS_BELOW_EDGE: Final = 0.5
"""How far below the band's lower edge the injected deviation is placed.

In units of the band's own dispersion. Chosen rather than derived: the point is
that the gate states the distance it injected instead of searching for one that
happens to work.
"""


def gate_reference(index: int) -> ReferenceDistribution:
    """Build the frozen reference for one panel feature.

    Args:
        index: position in :data:`GATE_FEATURES`.

    Returns:
        A :class:`~backend.monitoring.psi.ReferenceDistribution` whose feature
        name and identifier both announce themselves as fixtures.
    """
    feature = GATE_FEATURES[index]
    return gaussian_reference(
        feature=feature,
        reference_id=f"FIXTURE_G12_train_window_{feature}",
        seed=_REFERENCE_SEEDS[index],
    )


def gate_panel(
    *,
    location_shift: float = 0.0,
    scale_factor: float = 1.0,
    absent: int = 0,
) -> tuple[tuple[ReferenceDistribution, FloatArray], ...]:
    """Build one date's panel, injecting into :data:`INJECTED_INDEX` only.

    With every argument left at its default nothing is injected at all, which is
    the gate's negative case: three cross-sections freshly drawn from the same
    distributions their references were cut from.

    Args:
        location_shift: added to every value of the injected feature, in
            reference standard deviations.
        scale_factor: multiplies the injected feature's deviations before the
            shift, dimensionless.
        absent: how many of the injected feature's observations become ``NaN``
            (count). The survivors are still drawn from the reference
            distribution, so this moves availability without moving shape.

    Returns:
        Three ``(reference, observed cross-section)`` pairs, in panel order.
    """
    panel: list[tuple[ReferenceDistribution, FloatArray]] = []
    for index in range(len(GATE_FEATURES)):
        reference = gate_reference(index)
        if index == INJECTED_INDEX:
            observed = gaussian_sample(
                seed=_OBSERVED_SEEDS[index], shift=location_shift, scale=scale_factor
            )
            if absent:
                observed = with_absent(observed, absent=absent)
        else:
            observed = gaussian_sample(seed=_OBSERVED_SEEDS[index])
        panel.append((reference, observed))
    return tuple(panel)


def gate_drift_report(
    *,
    location_shift: float = 0.0,
    scale_factor: float = 1.0,
    absent: int = 0,
    as_of: dt.date = GATE_AS_OF,
) -> DriftReport:
    """Measure :func:`gate_panel` and report it, stamped (I2).

    Args:
        location_shift: location injection, in reference standard deviations.
        scale_factor: dispersion injection, dimensionless.
        absent: availability injection, in observations.
        as_of: the date the cross-sections belong to.

    Returns:
        A :class:`~backend.monitoring.drift.DriftReport` on the conventional
        ladder.
    """
    return drift_report(
        as_of=as_of,
        stamp=fixture_stamp(),
        observations=gate_panel(
            location_shift=location_shift, scale_factor=scale_factor, absent=absent
        ),
    )


def gate_band() -> ExpectationBand:
    """Return the expectation band the performance clause is judged against.

    Returns:
        The default fixture band: a per-period Sharpe band cut from the
        synthetic CPCV path matrix at the 63-period horizon.
    """
    return fixture_band()


def deviated_live_window(
    band: ExpectationBand,
    *,
    sigmas_below_edge: float = DEVIATION_SIGMAS_BELOW_EDGE,
    as_of: dt.date = GATE_AS_OF,
) -> LiveWindow:
    """Place a live window a stated distance below the band's lower edge.

    Args:
        band: the band to place the window relative to.
        sigmas_below_edge: distance below :attr:`ExpectationBand.lower`, in
            units of :attr:`ExpectationBand.dispersion`. Non-negative.
        as_of: date of the window's last observation.

    Returns:
        A :class:`~backend.monitoring.expectation.LiveWindow` whose per-period
        Sharpe ratio is exactly ``band.lower - sigmas_below_edge *
        band.dispersion``.
    """
    target = band.lower - sigmas_below_edge * band.dispersion
    return live_window(series_with_sharpe(target, n_periods=band.window_periods), as_of=as_of)


def in_band_live_window(band: ExpectationBand, *, as_of: dt.date = GATE_AS_OF) -> LiveWindow:
    """Place a live window exactly at the band's median.

    Args:
        band: the band to place the window relative to.
        as_of: date of the window's last observation.

    Returns:
        A :class:`~backend.monitoring.expectation.LiveWindow` whose per-period
        Sharpe ratio is exactly :attr:`ExpectationBand.median`.
    """
    return live_window(series_with_sharpe(band.median, n_periods=band.window_periods), as_of=as_of)


async def run_monitoring_cycle(
    snapshot: MonitoringSnapshot,
    *,
    store: AlertStore,
    channels: Sequence[AlertChannel],
    now: dt.datetime,
) -> tuple[DispatchResult, ...]:
    """Run the standard rule set over a snapshot and dispatch everything it raises.

    This is the whole reporting layer in one call — the shape a scheduled
    monitoring job has — so a gate assertion is about what an operator would
    actually receive rather than about an intermediate value.

    Args:
        snapshot: what the monitoring run observed.
        store: where alerts, attempts and acknowledgements are persisted.
        channels: delivery channels in priority order.
        now: the cycle instant (UTC).

    Returns:
        One :class:`~backend.monitoring.alerts.DispatchResult` per alert raised,
        in rule order. Empty when the cycle found nothing to say.
    """
    results: list[DispatchResult] = []
    for alert in evaluate_rules(default_rules(), snapshot, now=now):
        results.append(await dispatch(alert, store=store, channels=channels, now=now))
    return tuple(results)
