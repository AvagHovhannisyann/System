"""The lookahead suite: a label at ``t`` must not depend on data after ``t + H``.

This is the test that catches the class of bug that would otherwise never
surface. A barrier sized with full-sample volatility, a beta fitted over the
whole history, a scan window off by one to the right — none of these produce a
wrong-looking label distribution, a failing type check, or an exception. They
produce labels that are very slightly prescient, a model that learns from them,
and a backtest that looks excellent and is worthless (directive §2 I1, §0.3).

Two independent formulations, because they fail differently:

**Truncation.** Recompute the label for event ``t`` on a series that has been
cut off at ``t + horizon``. If the label is a function of that window only, the
answer is bit-identical. If anything reaches further right, the truncated series
either produces a different number or raises — and either is a failure here.

**Mutation.** Recompute after replacing every bar beyond the last event's
horizon with wildly different prices. Same requirement. This catches an
implementation that reads future data but only in a way that survives
truncation (a full-series mean, a global normalization).

Each is paired with a **non-vacuity** check that alters data *inside* the window
and asserts the labels do change. Without those, an implementation that returned
a constant would pass every test in this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.labels.barriers import (
    BarrierSpec,
    triple_barrier_labels,
    usable_event_indices,
)
from backend.labels.residualize import (
    ResidualSpec,
    residualized_triple_barrier_labels,
    usable_residual_event_indices,
)
from backend.tests.labels.paths import (
    factor_series,
    prices_from_log_returns,
    wave_log_returns,
)

N_BARS = 160
HORIZON = 21
SPEC = BarrierSpec(horizon=HORIZON, volatility_window=20)
RESIDUAL_SPEC = ResidualSpec(estimation_window=60)


def _price_series() -> np.ndarray:
    """A deterministic 161-bar close series."""
    return prices_from_log_returns(wave_log_returns(N_BARS))


def _factor_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asset, market and sector closes with a genuine factor structure."""
    return factor_series(N_BARS)


def _assert_same_label(actual: object, expected: object, position: int) -> None:
    """Assert two label sets agree on every field of one observation."""
    for field in (
        "outcome",
        "resolution_index",
        "realized_log_return",
        "upper_barrier_log_return",
        "lower_barrier_log_return",
        "trailing_volatility",
    ):
        left = getattr(actual, field)
        right = getattr(expected, field)
        assert left[0] == pytest.approx(right[position]), (
            f"{field} changed for the event at position {position}"
        )


class TestPricePathLookahead:
    def test_truncating_at_the_horizon_reproduces_every_label_exactly(self) -> None:
        close = _price_series()
        events = usable_event_indices(close.size, SPEC)
        full = triple_barrier_labels(close, events, SPEC)

        for position, event in enumerate(events):
            truncated = close[: int(event) + HORIZON + 1]
            recomputed = triple_barrier_labels(truncated, [int(event)], SPEC)
            _assert_same_label(recomputed, full, position)

    def test_altering_data_beyond_the_last_horizon_changes_nothing(self) -> None:
        close = _price_series()
        events = usable_event_indices(close.size, SPEC)
        baseline = triple_barrier_labels(close, events, SPEC)

        beyond = int(events[-1]) + HORIZON + 1
        altered = close.copy()
        altered[beyond:] = close[beyond:] * np.linspace(3.0, 0.2, close.size - beyond)

        recomputed = triple_barrier_labels(altered, events, SPEC)
        assert recomputed.outcome.tolist() == baseline.outcome.tolist()
        assert recomputed.resolution_index.tolist() == baseline.resolution_index.tolist()
        assert recomputed.realized_log_return == pytest.approx(baseline.realized_log_return)
        assert recomputed.trailing_volatility == pytest.approx(baseline.trailing_volatility)

    def test_the_alteration_would_have_been_visible_had_it_landed_earlier(self) -> None:
        # Non-vacuity: the same violent rescaling applied one bar *inside* the
        # last event's window must change that event's label. Without this, an
        # implementation returning a constant would pass the test above.
        close = _price_series()
        events = usable_event_indices(close.size, SPEC)
        baseline = triple_barrier_labels(close, events, SPEC)

        inside = int(events[-1]) + 1
        altered = close.copy()
        altered[inside:] = close[inside:] * np.linspace(3.0, 0.2, close.size - inside)

        recomputed = triple_barrier_labels(altered, events, SPEC)
        assert recomputed.outcome.tolist() != baseline.outcome.tolist()

    def test_volatility_is_not_estimated_over_the_labelling_window(self) -> None:
        # The sharpest form of the barrier-sizing bug: replace the labelling
        # window with a violent path and check the *barrier* is untouched. A
        # barrier sized on realized future volatility fails here even though the
        # outcome might, by luck, be unchanged.
        close = _price_series()
        event = int(usable_event_indices(close.size, SPEC)[0])
        baseline = triple_barrier_labels(close, [event], SPEC)

        altered = close.copy()
        window = slice(event + 1, event + HORIZON + 1)
        altered[window] = close[window] * np.linspace(1.0, 4.0, HORIZON)
        altered[event + HORIZON + 1 :] = close[event + HORIZON + 1 :] * 4.0

        recomputed = triple_barrier_labels(altered, [event], SPEC)
        assert recomputed.trailing_volatility[0] == pytest.approx(baseline.trailing_volatility[0])
        assert recomputed.upper_barrier_log_return[0] == pytest.approx(
            baseline.upper_barrier_log_return[0]
        )

    def test_intrabar_extremes_beyond_the_horizon_are_also_ignored(self) -> None:
        close = _price_series()
        events = usable_event_indices(close.size, SPEC)
        high = close * 1.002
        low = close * 0.998
        baseline = triple_barrier_labels(close, events, SPEC, high=high, low=low)

        beyond = int(events[-1]) + HORIZON + 1
        altered_high = high.copy()
        altered_low = low.copy()
        altered_high[beyond:] = high[beyond:] * 5.0
        altered_low[beyond:] = low[beyond:] * 0.2

        recomputed = triple_barrier_labels(close, events, SPEC, high=altered_high, low=altered_low)
        assert recomputed.outcome.tolist() == baseline.outcome.tolist()
        assert recomputed.resolution_index.tolist() == baseline.resolution_index.tolist()


class TestResidualLookahead:
    def test_truncating_at_the_horizon_reproduces_every_residual_label(self) -> None:
        asset, market, sector = _factor_series()
        events = usable_residual_event_indices(asset.size, SPEC, RESIDUAL_SPEC)
        full = residualized_triple_barrier_labels(
            asset, market, sector, events, SPEC, RESIDUAL_SPEC
        )

        for position, event in enumerate(events):
            cut = int(event) + HORIZON + 1
            recomputed = residualized_triple_barrier_labels(
                asset[:cut], market[:cut], sector[:cut], [int(event)], SPEC, RESIDUAL_SPEC
            )
            _assert_same_label(recomputed.labels, full.labels, position)
            assert recomputed.beta_market[0] == pytest.approx(full.beta_market[position])
            assert recomputed.beta_sector[0] == pytest.approx(full.beta_sector[position])

    def test_altering_factors_beyond_the_last_horizon_changes_nothing(self) -> None:
        asset, market, sector = _factor_series()
        events = usable_residual_event_indices(asset.size, SPEC, RESIDUAL_SPEC)
        baseline = residualized_triple_barrier_labels(
            asset, market, sector, events, SPEC, RESIDUAL_SPEC
        )

        beyond = int(events[-1]) + HORIZON + 1
        ramp = np.linspace(2.5, 0.4, asset.size - beyond)
        altered_market = market.copy()
        altered_sector = sector.copy()
        altered_asset = asset.copy()
        altered_market[beyond:] = market[beyond:] * ramp
        altered_sector[beyond:] = sector[beyond:] * ramp
        altered_asset[beyond:] = asset[beyond:] * ramp

        recomputed = residualized_triple_barrier_labels(
            altered_asset, altered_market, altered_sector, events, SPEC, RESIDUAL_SPEC
        )
        assert recomputed.labels.outcome.tolist() == baseline.labels.outcome.tolist()
        assert recomputed.beta_market == pytest.approx(baseline.beta_market)
        assert recomputed.beta_sector == pytest.approx(baseline.beta_sector)
        assert recomputed.labels.trailing_volatility == pytest.approx(
            baseline.labels.trailing_volatility
        )

    def test_betas_are_not_fitted_over_the_labelling_window(self) -> None:
        # A beta fitted on the full sample (or on a window that reaches past the
        # event) moves when the *forward* factor path changes. A trailing one
        # does not. This isolates that single failure, which no distributional
        # check on the labels would ever reveal.
        asset, market, sector = _factor_series()
        event = int(usable_residual_event_indices(asset.size, SPEC, RESIDUAL_SPEC)[0])
        baseline = residualized_triple_barrier_labels(
            asset, market, sector, [event], SPEC, RESIDUAL_SPEC
        )

        altered_market = market.copy()
        altered_market[event + 1 :] = market[event + 1 :] * np.linspace(
            1.0, 6.0, market.size - event - 1
        )
        recomputed = residualized_triple_barrier_labels(
            asset, altered_market, sector, [event], SPEC, RESIDUAL_SPEC
        )
        assert recomputed.beta_market[0] == pytest.approx(baseline.beta_market[0])
        assert recomputed.beta_sector[0] == pytest.approx(baseline.beta_sector[0])
        assert recomputed.labels.trailing_volatility[0] == pytest.approx(
            baseline.labels.trailing_volatility[0]
        )

    def test_the_forward_factor_path_does_still_reach_the_label(self) -> None:
        # Non-vacuity for the test above: the forward factors must matter to the
        # *outcome* even though they must not matter to the betas.
        asset, market, sector = _factor_series()
        events = usable_residual_event_indices(asset.size, SPEC, RESIDUAL_SPEC)
        baseline = residualized_triple_barrier_labels(
            asset, market, sector, events, SPEC, RESIDUAL_SPEC
        )

        first = int(events[0])
        altered_market = market.copy()
        altered_market[first + 1 :] = market[first + 1 :] * np.linspace(
            1.0, 6.0, market.size - first - 1
        )
        recomputed = residualized_triple_barrier_labels(
            asset, altered_market, sector, events, SPEC, RESIDUAL_SPEC
        )
        assert recomputed.labels.outcome.tolist() != baseline.labels.outcome.tolist()
