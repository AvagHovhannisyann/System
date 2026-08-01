"""Property tests for label construction (P6.4).

DECISIONS.md **D-018** closes with a standing lesson for exactly this suite:

    A property suite that randomizes *data* while holding *query shape* fixed
    measures breadth, not correctness. […] Any future property suite (labels
    P6.4, optimizer P9.4, cost model, backtest engine) must randomize the
    *structure* of what it exercises, not only the values fed through one fixed
    structure.

So the generators below randomize structure first and values second. What
varies from example to example is: the *shape* of the price path (six
qualitatively different generators — drift, jumps, regime changes, sawtooths,
step functions, free random walks), whether intrabar extremes exist at all and
how wide the wicks are, the horizon, the volatility window, whether the barriers
are symmetric, which intrabar policy is in force, and the *shape* of the event
set (one event, every usable event, a sparse stride, a dense block, duplicates).
For uniqueness, the *layout* of the label spans varies: disjoint, identical,
nested, staggered, clustered with gaps, single-bar.

The primary property is agreement with a reference implementation written as a
plain bar-by-bar Python scan with :func:`statistics.stdev` — no numpy, no
vectorization, no shared code with the implementation. Two implementations
agreeing on tens of thousands of structurally different problems is evidence;
one implementation agreeing with itself is not.

**On the tie guard.** Both implementations compare a cumulative return against a
barrier with ``>=``. They compute that barrier through different arithmetic
(``statistics.stdev`` on Fractions versus a two-pass ``numpy`` reduction, and
``math.log`` versus ``numpy.log``), so they can disagree in the last bit or two.
Where a path passes within ``1e-12`` of a barrier — a measure-zero coincidence
in continuous terms and a floating-point artefact in practice — the example is
discarded via :func:`hypothesis.assume`. This suppresses a comparison that is
genuinely undefined at that precision; it does not suppress any disagreement
about *which* barrier was touched, and the boundary rule itself is pinned
exactly by the hand-derived cases in
:mod:`backend.tests.labels.test_barriers`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.labels.barriers import (
    BarrierSpec,
    IntrabarPolicy,
    LabelSet,
    TripleBarrierOutcome,
    triple_barrier_labels,
    usable_event_indices,
)
from backend.labels.errors import (
    DegenerateVolatilityError,
    LabelError,
)
from backend.labels.residualize import (
    ResidualSpec,
    residualized_triple_barrier_labels,
    usable_residual_event_indices,
)
from backend.labels.uniqueness import sample_uniqueness
from backend.tests.labels.paths import prices_from_log_returns, wave_log_returns

if TYPE_CHECKING:
    from backend.labels._arrays import FloatArray, IntArray

TIE_TOLERANCE = 1.0e-12
"""Below this distance from a barrier the two implementations' arithmetic differs."""

# A dither with period 3 guarantees that no two consecutive returns are equal,
# so a trailing window is never exactly flat by construction of the generator
# rather than by luck. Degenerate windows are still reachable (and still
# asserted on) when a generator produces cancelling values.
_DITHER = (1e-6, 0.0, -1e-6)


@dataclass(frozen=True)
class _Problem:
    """One randomly *shaped* labelling problem."""

    close: FloatArray
    high: FloatArray | None
    low: FloatArray | None
    spec: BarrierSpec
    events: IntArray
    shape: str


@st.composite
def _log_return_paths(draw: st.DrawFn, n_bars: int) -> list[float]:
    """Draw a path shape first, then its parameters."""
    shape = draw(st.sampled_from(["walk", "drift", "jump", "regime", "sawtooth", "steps"]))
    if shape == "walk":
        returns = draw(
            st.lists(
                st.floats(min_value=-0.06, max_value=0.06, allow_nan=False, allow_infinity=False),
                min_size=n_bars,
                max_size=n_bars,
            )
        )
    elif shape == "drift":
        slope = draw(st.floats(min_value=-0.03, max_value=0.03, allow_nan=False))
        wobble = draw(st.floats(min_value=0.0, max_value=0.01, allow_nan=False))
        returns = [slope + wobble * math.sin(0.9 * index) for index in range(n_bars)]
    elif shape == "jump":
        base = draw(st.floats(min_value=0.0, max_value=0.004, allow_nan=False))
        size = draw(st.floats(min_value=-0.4, max_value=0.4, allow_nan=False))
        where = draw(st.integers(min_value=0, max_value=n_bars - 1))
        returns = [base * ((index % 5) - 2) for index in range(n_bars)]
        returns[where] += size
    elif shape == "regime":
        quiet = draw(st.floats(min_value=0.0005, max_value=0.006, allow_nan=False))
        loud = draw(st.floats(min_value=0.02, max_value=0.12, allow_nan=False))
        switch = draw(st.integers(min_value=1, max_value=n_bars - 1))
        returns = [
            (quiet if index < switch else loud) * math.sin(1.3 * index) for index in range(n_bars)
        ]
    elif shape == "sawtooth":
        up = draw(st.floats(min_value=0.001, max_value=0.08, allow_nan=False))
        down = draw(st.floats(min_value=0.001, max_value=0.08, allow_nan=False))
        returns = [up if index % 2 else -down for index in range(n_bars)]
    else:  # steps: piecewise-constant returns, so trailing windows go flat
        block = draw(st.integers(min_value=1, max_value=8))
        levels = draw(
            st.lists(
                st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
                min_size=1,
                max_size=6,
            )
        )
        returns = [levels[(index // block) % len(levels)] for index in range(n_bars)]
    return [value + _DITHER[index % 3] for index, value in enumerate(returns)]


@st.composite
def _problems(draw: st.DrawFn) -> _Problem:
    """Draw a complete labelling problem, structure first."""
    n_bars = draw(st.integers(min_value=12, max_value=70))
    horizon = draw(st.integers(min_value=1, max_value=12))
    volatility_window = draw(st.integers(min_value=2, max_value=10))
    spec = BarrierSpec(
        horizon=horizon,
        upper_multiple=draw(st.floats(min_value=0.05, max_value=4.0, allow_nan=False)),
        lower_multiple=draw(st.floats(min_value=0.05, max_value=4.0, allow_nan=False)),
        volatility_window=volatility_window,
        intrabar_policy=draw(st.sampled_from(list(IntrabarPolicy))),
    )
    close = prices_from_log_returns(draw(_log_return_paths(n_bars)))

    with_extremes = draw(st.booleans())
    high: FloatArray | None = None
    low: FloatArray | None = None
    if with_extremes:
        upper_wick = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=0.08, allow_nan=False),
                min_size=close.size,
                max_size=close.size,
            )
        )
        lower_wick = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=0.08, allow_nan=False),
                min_size=close.size,
                max_size=close.size,
            )
        )
        high = close * np.exp(np.asarray(upper_wick))
        low = close * np.exp(-np.asarray(lower_wick))

    usable = usable_event_indices(close.size, spec)
    assume(usable.size > 0)
    event_shape = draw(st.sampled_from(["all", "single", "stride", "block", "duplicated"]))
    if event_shape == "all":
        events = usable
    elif event_shape == "single":
        events = np.array([draw(st.sampled_from(usable.tolist()))], dtype=np.intp)
    elif event_shape == "stride":
        step = draw(st.integers(min_value=2, max_value=5))
        events = usable[::step]
    elif event_shape == "block":
        start = draw(st.integers(min_value=0, max_value=max(usable.size - 1, 0)))
        events = usable[start : start + 4]
    else:
        chosen = draw(st.sampled_from(usable.tolist()))
        events = np.array([chosen, chosen, chosen], dtype=np.intp)
    assume(events.size > 0)

    return _Problem(close=close, high=high, low=low, spec=spec, events=events, shape=event_shape)


@dataclass(frozen=True)
class _ReferenceLabel:
    """One event resolved by the reference scan."""

    outcome: TripleBarrierOutcome
    resolution_index: int
    volatility: float
    margin: float


def _reference(problem: _Problem) -> list[_ReferenceLabel] | None:
    """Resolve every event by a plain Python bar-by-bar scan.

    Independent of the implementation in every respect that could hide a shared
    bug: ``math.log`` rather than ``numpy.log``, :func:`statistics.stdev` rather
    than a ``numpy`` reduction, an early-exit loop rather than ``argmax`` over a
    boolean mask.

    Returns:
        One :class:`_ReferenceLabel` per event, or ``None`` if some event's
        trailing window is exactly flat — in which case the implementation is
        required to refuse the whole batch instead.
    """
    log_close = [math.log(float(price)) for price in problem.close]
    log_high = (
        [math.log(float(price)) for price in problem.high]
        if problem.high is not None
        else log_close
    )
    log_low = (
        [math.log(float(price)) for price in problem.low] if problem.low is not None else log_close
    )
    returns = [log_close[bar] - log_close[bar - 1] for bar in range(1, len(log_close))]

    spec = problem.spec
    labels: list[_ReferenceLabel] = []
    for raw_event in problem.events:
        event = int(raw_event)
        window = [returns[bar - 1] for bar in range(event - spec.volatility_window + 1, event + 1)]
        volatility = statistics.stdev(window)
        if volatility <= 0.0:
            return None
        upper = spec.upper_multiple * volatility * math.sqrt(spec.horizon)
        lower = -spec.lower_multiple * volatility * math.sqrt(spec.horizon)

        anchor = log_close[event]
        outcome = TripleBarrierOutcome.VERTICAL
        resolution = event + spec.horizon
        margin = math.inf
        for bar in range(event + 1, event + spec.horizon + 1):
            reach_up = log_high[bar] - anchor
            reach_down = log_low[bar] - anchor
            margin = min(margin, abs(reach_up - upper), abs(reach_down - lower))
            touched_upper = reach_up >= upper
            touched_lower = reach_down <= lower
            if touched_upper and touched_lower:
                outcome = (
                    TripleBarrierOutcome.LOWER_FIRST
                    if spec.intrabar_policy is IntrabarPolicy.LOWER_FIRST
                    else TripleBarrierOutcome.AMBIGUOUS
                )
                resolution = bar
                break
            if touched_upper:
                outcome = TripleBarrierOutcome.UPPER_FIRST
                resolution = bar
                break
            if touched_lower:
                outcome = TripleBarrierOutcome.LOWER_FIRST
                resolution = bar
                break
        labels.append(
            _ReferenceLabel(
                outcome=outcome,
                resolution_index=resolution,
                volatility=volatility,
                margin=margin,
            )
        )
    return labels


def _label(problem: _Problem) -> LabelSet:
    """Run the implementation on a problem."""
    return triple_barrier_labels(
        problem.close,
        problem.events,
        problem.spec,
        high=problem.high,
        low=problem.low,
    )


class TestAgreementWithAReferenceImplementation:
    @given(_problems())
    @hypothesis_settings(max_examples=600, deadline=None)
    def test_the_two_implementations_resolve_every_event_identically(
        self, problem: _Problem
    ) -> None:
        reference = _reference(problem)
        if reference is None:
            # A flat trailing window: the implementation must refuse it rather
            # than size a zero-width barrier. The refusal is part of the
            # contract, so it is asserted rather than skipped.
            with pytest.raises(DegenerateVolatilityError):
                _label(problem)
            return
        assume(all(entry.margin > TIE_TOLERANCE for entry in reference))

        labels = _label(problem)
        assert labels.outcome.tolist() == [int(entry.outcome) for entry in reference]
        assert labels.resolution_index.tolist() == [entry.resolution_index for entry in reference]
        assert labels.trailing_volatility == pytest.approx(
            [entry.volatility for entry in reference]
        )


class TestStructuralInvariants:
    @given(_problems())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_resolution_always_lands_inside_the_scan_window(self, problem: _Problem) -> None:
        try:
            labels = _label(problem)
        except DegenerateVolatilityError:
            return
        events = np.asarray(labels.event_index)
        resolution = np.asarray(labels.resolution_index)
        assert np.all(resolution > events)
        assert np.all(resolution <= events + problem.spec.horizon)

    @given(_problems())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_a_vertical_outcome_always_resolves_at_the_deadline(self, problem: _Problem) -> None:
        try:
            labels = _label(problem)
        except DegenerateVolatilityError:
            return
        vertical = np.asarray(labels.outcome) == int(TripleBarrierOutcome.VERTICAL)
        deadline = np.asarray(labels.event_index) + problem.spec.horizon
        assert np.all(np.asarray(labels.resolution_index)[vertical] == deadline[vertical])

    @given(_problems())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_barriers_are_the_documented_multiple_of_trailing_volatility(
        self, problem: _Problem
    ) -> None:
        try:
            labels = _label(problem)
        except DegenerateVolatilityError:
            return
        scale = np.asarray(labels.trailing_volatility) * math.sqrt(problem.spec.horizon)
        assert labels.upper_barrier_log_return == pytest.approx(problem.spec.upper_multiple * scale)
        assert labels.lower_barrier_log_return == pytest.approx(
            -problem.spec.lower_multiple * scale
        )

    @given(_problems())
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_close_only_labelling_is_never_ambiguous(self, problem: _Problem) -> None:
        close_only = _Problem(
            close=problem.close,
            high=None,
            low=None,
            spec=problem.spec,
            events=problem.events,
            shape=problem.shape,
        )
        try:
            labels = _label(close_only)
        except DegenerateVolatilityError:
            return
        assert labels.n_ambiguous == 0

    @given(_problems())
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_events_labelled_together_match_events_labelled_alone(self, problem: _Problem) -> None:
        try:
            together = _label(problem)
        except DegenerateVolatilityError:
            return
        for position, event in enumerate(problem.events):
            single = triple_barrier_labels(
                problem.close,
                [int(event)],
                problem.spec,
                high=problem.high,
                low=problem.low,
            )
            assert int(single.outcome[0]) == int(together.outcome[position])
            assert int(single.resolution_index[0]) == int(together.resolution_index[position])


class TestMonotonicityAndInvariance:
    @given(_problems(), st.floats(min_value=1.0, max_value=6.0, allow_nan=False))
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_widening_the_barriers_can_only_delay_resolution(
        self, problem: _Problem, widening: float
    ) -> None:
        try:
            narrow = _label(problem)
        except DegenerateVolatilityError:
            return
        wider_spec = BarrierSpec(
            horizon=problem.spec.horizon,
            upper_multiple=problem.spec.upper_multiple * widening,
            lower_multiple=problem.spec.lower_multiple * widening,
            volatility_window=problem.spec.volatility_window,
            intrabar_policy=problem.spec.intrabar_policy,
        )
        wide = triple_barrier_labels(
            problem.close, problem.events, wider_spec, high=problem.high, low=problem.low
        )
        assert np.all(np.asarray(wide.resolution_index) >= np.asarray(narrow.resolution_index))

    @given(_problems(), st.floats(min_value=0.2, max_value=5.0, allow_nan=False))
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_the_label_is_invariant_to_rescaling_the_whole_path(
        self, problem: _Problem, scale: float
    ) -> None:
        # Multiplying every log return by c scales the cumulative path and the
        # trailing volatility by exactly c, so the barriers move with the path
        # and the label cannot change. A barrier sized in absolute percent
        # instead of volatility units would fail this.
        reference = _reference(problem)
        if reference is None:
            return
        assume(all(entry.margin > TIE_TOLERANCE for entry in reference))

        anchor = math.log(float(problem.close[0]))

        def rescale(series: FloatArray) -> FloatArray:
            return np.asarray(np.exp(anchor + scale * (np.log(series) - anchor)))

        scaled = _Problem(
            close=rescale(problem.close),
            high=None if problem.high is None else rescale(problem.high),
            low=None if problem.low is None else rescale(problem.low),
            spec=problem.spec,
            events=problem.events,
            shape=problem.shape,
        )
        baseline = _label(problem)
        try:
            rescaled = _label(scaled)
        except LabelError:
            # Rescaling can push a path's volatility below representable range;
            # a refusal is the honest outcome, not a disagreement.
            return
        assert rescaled.outcome.tolist() == baseline.outcome.tolist()


class TestLookaheadProperty:
    @given(_problems())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_truncating_at_the_horizon_reproduces_the_label(self, problem: _Problem) -> None:
        # The lookahead invariant, over randomly *shaped* problems rather than
        # the single fixed series of test_lookahead.py.
        try:
            full = _label(problem)
        except DegenerateVolatilityError:
            return
        for position, event in enumerate(problem.events):
            cut = int(event) + problem.spec.horizon + 1
            truncated = triple_barrier_labels(
                problem.close[:cut],
                [int(event)],
                problem.spec,
                high=None if problem.high is None else problem.high[:cut],
                low=None if problem.low is None else problem.low[:cut],
            )
            assert int(truncated.outcome[0]) == int(full.outcome[position])
            assert int(truncated.resolution_index[0]) == int(full.resolution_index[position])
            assert float(truncated.trailing_volatility[0]) == pytest.approx(
                float(full.trailing_volatility[position])
            )


@st.composite
def _span_layouts(draw: st.DrawFn) -> tuple[IntArray, IntArray]:
    """Draw a span *layout* — the structural dimension for uniqueness."""
    layout = draw(
        st.sampled_from(
            ["disjoint", "identical", "nested", "staggered", "clustered", "single_bar", "mixed"]
        )
    )
    n_labels = draw(st.integers(min_value=1, max_value=25))
    length = draw(st.integers(min_value=1, max_value=15))
    origin = draw(st.integers(min_value=0, max_value=50))

    if layout == "disjoint":
        starts = [origin + index * (length + draw(st.integers(0, 3))) for index in range(n_labels)]
        ends = [start + length - 1 for start in starts]
    elif layout == "identical":
        starts = [origin] * n_labels
        ends = [origin + length - 1] * n_labels
    elif layout == "nested":
        # Each span strictly inside the previous one.
        starts = [origin + index for index in range(n_labels)]
        ends = [origin + 2 * n_labels + length - index - 1 for index in range(n_labels)]
    elif layout == "staggered":
        step = draw(st.integers(min_value=1, max_value=max(length, 1)))
        starts = [origin + index * step for index in range(n_labels)]
        ends = [start + length - 1 for start in starts]
    elif layout == "clustered":
        gap = draw(st.integers(min_value=5, max_value=40))
        starts = [origin + (index % 3) + gap * (index // 3) for index in range(n_labels)]
        ends = [start + length - 1 for start in starts]
    elif layout == "single_bar":
        starts = [
            draw(st.integers(min_value=origin, max_value=origin + 20)) for _ in range(n_labels)
        ]
        ends = list(starts)
    else:
        starts = [
            draw(st.integers(min_value=origin, max_value=origin + 30)) for _ in range(n_labels)
        ]
        ends = [start + draw(st.integers(min_value=0, max_value=length)) for start in starts]

    return (
        np.asarray(starts, dtype=np.intp),
        np.asarray(ends, dtype=np.intp),
    )


class TestUniquenessProperties:
    @given(_span_layouts())
    @hypothesis_settings(max_examples=500, deadline=None)
    def test_every_weight_lies_in_the_unit_interval(self, spans: tuple[IntArray, IntArray]) -> None:
        weights = sample_uniqueness(*spans).average_uniqueness
        assert np.all(weights > 0.0)
        assert np.all(weights <= 1.0 + 1e-12)

    @given(_span_layouts())
    @hypothesis_settings(max_examples=500, deadline=None)
    def test_effective_sample_size_never_exceeds_the_nominal_count(
        self, spans: tuple[IntArray, IntArray]
    ) -> None:
        result = sample_uniqueness(*spans)
        assert 0.0 < result.effective_sample_size <= result.nominal_count + 1e-12

    @given(_span_layouts())
    @hypothesis_settings(max_examples=500, deadline=None)
    def test_the_sample_is_full_size_exactly_when_nothing_overlaps(
        self, spans: tuple[IntArray, IntArray]
    ) -> None:
        result = sample_uniqueness(*spans)
        no_overlap = result.max_concurrency <= 1
        full_size = result.effective_sample_size == pytest.approx(float(result.nominal_count))
        assert no_overlap == full_size

    @given(_span_layouts())
    @hypothesis_settings(max_examples=500, deadline=None)
    def test_weighted_span_lengths_count_the_covered_bars(
        self, spans: tuple[IntArray, IntArray]
    ) -> None:
        # sum_i sum_{t in span_i} 1/c_t = sum_{t: c_t >= 1} 1. An exact identity
        # that pins the concurrency array and the per-label averaging together.
        first, last = spans
        result = sample_uniqueness(first, last)
        lengths = (last - first + 1).astype(np.float64)
        covered = int(np.count_nonzero(result.concurrency > 0))
        assert float(np.sum(result.average_uniqueness * lengths)) == pytest.approx(covered)

    @given(_span_layouts(), st.integers(min_value=0, max_value=60), st.integers(0, 15))
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_adding_a_label_never_raises_another_label_s_weight(
        self, spans: tuple[IntArray, IntArray], extra_start: int, extra_length: int
    ) -> None:
        first, last = spans
        before = sample_uniqueness(first, last).average_uniqueness
        after = sample_uniqueness(
            np.append(first, extra_start), np.append(last, extra_start + extra_length)
        ).average_uniqueness
        assert np.all(after[: before.size] <= before + 1e-12)

    @given(st.integers(min_value=1, max_value=30), st.integers(min_value=1, max_value=20))
    @hypothesis_settings(max_examples=200, deadline=None)
    def test_n_identical_spans_each_weigh_one_over_n(self, n_labels: int, length: int) -> None:
        result = sample_uniqueness([7] * n_labels, [7 + length - 1] * n_labels)
        assert result.average_uniqueness == pytest.approx([1.0 / n_labels] * n_labels)
        assert result.effective_sample_size == pytest.approx(1.0)


@st.composite
def _factor_problems(draw: st.DrawFn) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """Draw an asset/market/sector triple with an independent factor structure."""
    n_bars = draw(st.integers(min_value=60, max_value=110))
    market = wave_log_returns(
        n_bars,
        scale=draw(st.floats(min_value=0.4, max_value=1.5, allow_nan=False)),
        frequency=0.70,
        second_frequency=0.31,
    )
    sector = wave_log_returns(
        n_bars,
        scale=draw(st.floats(min_value=0.4, max_value=1.5, allow_nan=False)),
        phase=draw(st.floats(min_value=0.0, max_value=3.0, allow_nan=False)),
        frequency=0.23,
        second_frequency=1.10,
    )
    idiosyncratic = wave_log_returns(
        n_bars,
        scale=draw(st.floats(min_value=0.2, max_value=1.0, allow_nan=False)),
        phase=draw(st.floats(min_value=0.0, max_value=3.0, allow_nan=False)),
        frequency=0.41,
        second_frequency=1.90,
    )
    asset = (
        draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)) * market
        + draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)) * sector
        + idiosyncratic
    )
    return (
        prices_from_log_returns(asset),
        prices_from_log_returns(market),
        prices_from_log_returns(sector),
        np.asarray(market, dtype=np.float64),
    )


class TestResidualProperties:
    @given(_factor_problems(), st.floats(min_value=-3.0, max_value=3.0, allow_nan=False))
    @hypothesis_settings(max_examples=150, deadline=None)
    def test_the_residual_label_ignores_the_stock_s_factor_loading(
        self,
        problem: tuple[FloatArray, FloatArray, FloatArray, IntArray],
        extra_loading: float,
    ) -> None:
        # Adding c * market to the asset's returns shifts the fitted market beta
        # by exactly c and leaves the residual path — and so the label — alone.
        # This is what "the label reflects idiosyncratic movement rather than
        # beta" means, stated as a property rather than an intention.
        asset_close, market_close, sector_close, market_returns = problem
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        events = usable_residual_event_indices(asset_close.size, spec, residual_spec)
        assume(events.size > 0)

        # Both arrays are the per-bar returns for bars 1..n, in the same order
        # prices_from_log_returns consumed them.
        tilted_returns = np.diff(np.log(asset_close)) + extra_loading * market_returns
        tilted_close = prices_from_log_returns(tilted_returns.tolist())

        try:
            base = residualized_triple_barrier_labels(
                asset_close, market_close, sector_close, events, spec, residual_spec
            )
        except LabelError as base_error:
            with pytest.raises(type(base_error)):
                residualized_triple_barrier_labels(
                    tilted_close, market_close, sector_close, events, spec, residual_spec
                )
            return

        tilted = residualized_triple_barrier_labels(
            tilted_close, market_close, sector_close, events, spec, residual_spec
        )
        assert tilted.beta_market == pytest.approx(base.beta_market + extra_loading, abs=1e-6)
        assert tilted.labels.trailing_volatility == pytest.approx(
            base.labels.trailing_volatility, rel=1e-9
        )
        assert tilted.labels.realized_log_return == pytest.approx(
            base.labels.realized_log_return, abs=1e-10
        )

    @given(_factor_problems())
    @hypothesis_settings(max_examples=150, deadline=None)
    def test_residual_labels_resolve_inside_their_window_and_are_never_ambiguous(
        self, problem: tuple[FloatArray, FloatArray, FloatArray, IntArray]
    ) -> None:
        asset_close, market_close, sector_close, _ = problem
        spec = BarrierSpec(horizon=7, volatility_window=12)
        residual_spec = ResidualSpec(estimation_window=45)
        events = usable_residual_event_indices(asset_close.size, spec, residual_spec)
        assume(events.size > 0)
        try:
            result = residualized_triple_barrier_labels(
                asset_close, market_close, sector_close, events, spec, residual_spec
            )
        except LabelError:
            return
        resolution = np.asarray(result.labels.resolution_index)
        assert np.all(resolution > events)
        assert np.all(resolution <= events + spec.horizon)
        assert result.labels.n_ambiguous == 0
