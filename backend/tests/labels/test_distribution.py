"""Label-distribution behaviour across volatility regimes, and the ESS report.

The Phase 6 gate asks for a *sane* label distribution across regimes and for the
overlap's effect on effective sample size to be reported. What can honestly be
established without real market data is the part that is a property of the
construction rather than of the market:

**Regime invariance.** Barriers are sized in units of trailing volatility, so a
quiet regime and a violent one asked the same question must answer with the same
class proportions. If the barrier were an absolute percentage, the quiet regime
would be all vertical-barrier outcomes and the violent one all touches — and the
model would be learning the regime rather than the security. Here the two halves
of a path that differ *only* in scale are asserted to label identically.

**What this does not establish.** Whether the distribution is sane on real
equities — the class balance actually observed at 5, 21 and 63 days, and its
stability through 2008, 2020 and quiet years — is a measurement, and it needs
market data that blocker **B1** has not yet supplied. That measurement is the
remaining half of gate G6 and is not claimed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.labels.barriers import (
    BarrierSpec,
    TripleBarrierOutcome,
    triple_barrier_labels,
    usable_event_indices,
)
from backend.labels.uniqueness import sample_uniqueness_from_labels
from backend.tests.labels.paths import prices_from_log_returns, wave_log_returns

REGIME_BARS = 300
QUIET_SCALE = 1.0
VIOLENT_SCALE = 6.0
SPEC = BarrierSpec(horizon=5, volatility_window=20)


def _two_regime_prices() -> np.ndarray:
    """A path whose second half is its first half scaled up sixfold.

    Bar ``i`` in the quiet regime and bar ``REGIME_BARS + i`` in the violent one
    have per-bar returns in exact proportion, so any volatility-relative label
    must agree on them and any absolute-percentage label cannot.
    """
    quiet = wave_log_returns(REGIME_BARS, scale=QUIET_SCALE)
    violent = wave_log_returns(REGIME_BARS, scale=VIOLENT_SCALE)
    return prices_from_log_returns(np.concatenate((quiet, violent)))


class TestRegimeInvariance:
    def test_the_violent_regime_is_genuinely_more_volatile(self) -> None:
        # The premise of every assertion below.
        prices = _two_regime_prices()
        labels = triple_barrier_labels(prices, [150, 450], SPEC)
        quiet, violent = labels.trailing_volatility
        assert violent == pytest.approx(VIOLENT_SCALE * quiet, rel=1e-9)

    def test_matched_bars_in_the_two_regimes_get_the_same_label(self) -> None:
        prices = _two_regime_prices()
        # Events far enough inside each regime that both the trailing window and
        # the labelling window sit entirely within one scale.
        quiet_events = np.arange(40, REGIME_BARS - 10)
        violent_events = quiet_events + REGIME_BARS

        quiet = triple_barrier_labels(prices, quiet_events, SPEC)
        violent = triple_barrier_labels(prices, violent_events, SPEC)

        assert quiet.outcome.tolist() == violent.outcome.tolist()
        assert (violent.resolution_index - REGIME_BARS).tolist() == (
            quiet.resolution_index.tolist()
        )

    def test_the_distribution_uses_all_three_classes(self) -> None:
        # Non-vacuity: matching two all-vertical distributions would prove
        # nothing. Each regime must actually exercise every outcome.
        prices = _two_regime_prices()
        counts = triple_barrier_labels(prices, np.arange(40, REGIME_BARS - 10), SPEC)
        distribution = counts.outcome_counts()
        assert distribution[TripleBarrierOutcome.UPPER_FIRST] > 0
        assert distribution[TripleBarrierOutcome.LOWER_FIRST] > 0
        assert distribution[TripleBarrierOutcome.VERTICAL] > 0

    @pytest.mark.parametrize("horizon", [5, 21, 63])
    def test_every_project_horizon_labels_a_long_path_without_refusing(self, horizon: int) -> None:
        prices = _two_regime_prices()
        spec = BarrierSpec(horizon=horizon, volatility_window=20)
        events = usable_event_indices(prices.size, spec)
        labels = triple_barrier_labels(prices, events, spec)
        assert labels.n_labels == events.size
        assert labels.n_ambiguous == 0


class TestEffectiveSampleSizeReport:
    def test_labelling_every_bar_costs_most_of_the_nominal_sample(self) -> None:
        # The end-to-end form of P6.3: label every usable bar at the 21-day
        # horizon, then ask what the sample is actually worth. Consecutive
        # 21-bar labels overlap by 20 bars, so the answer is far below the
        # nominal count — the number the directive requires to be reported.
        prices = _two_regime_prices()
        spec = BarrierSpec(horizon=21, volatility_window=20)
        events = usable_event_indices(prices.size, spec)
        labels = triple_barrier_labels(prices, events, spec)

        uniqueness = sample_uniqueness_from_labels(labels)
        assert uniqueness.nominal_count == labels.n_labels
        assert uniqueness.effective_sample_size < 0.10 * uniqueness.nominal_count
        # At most `horizon` labels can claim one bar when events are one bar
        # apart, and fewer in practice because a label that touches a barrier
        # early has a span shorter than the full horizon.
        assert 1 < uniqueness.max_concurrency <= spec.horizon
        assert np.all(uniqueness.average_uniqueness > 0.0)
        assert np.all(uniqueness.average_uniqueness <= 1.0)

        report = uniqueness.report()
        assert str(uniqueness.nominal_count) in report
        assert "effective sample size" in report

    def test_labelling_every_horizon_th_bar_costs_nothing(self) -> None:
        # The complement: space the events a full horizon apart and the labels
        # stop overlapping, so the effective sample size is the nominal one
        # exactly. This is what makes the test above a measurement of overlap
        # rather than of the weighting code being pessimistic.
        prices = _two_regime_prices()
        spec = BarrierSpec(horizon=21, volatility_window=20)
        events = usable_event_indices(prices.size, spec)[:: spec.horizon]
        labels = triple_barrier_labels(prices, events, spec)

        uniqueness = sample_uniqueness_from_labels(labels)
        assert uniqueness.max_concurrency == 1
        assert uniqueness.effective_sample_size == pytest.approx(float(labels.n_labels))
        assert uniqueness.uniqueness_ratio == pytest.approx(1.0)
