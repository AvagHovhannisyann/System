"""Arithmetic and availability of the three price factors (P5.3).

Each factor is checked against a hand-worked example: a price path chosen so
that the right answer can be written down in closed form and compared exactly,
rather than compared against whatever the implementation happens to produce.
The paths are obviously fixtures — flat series with one step, geometric ramps,
alternating moves — and no number in this file is presented as a market
observation.

The second half checks the conditions under which a factor must decline to
answer. Those matter as much as the arithmetic: a factor that returns a number
for a halted stock, a name with three months of history, or a corrupt print is
not wrong in a way anyone will notice, and the resulting column would be partly
a measurement of data coverage.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from backend.features.compute import FeatureComputeRequest
from backend.features.factors._prices import (
    MAX_PRICE_STALENESS,
    PriceSeries,
    PriceSeriesError,
    load_adjusted_closes,
)
from backend.features.factors.momentum import (
    MOMENTUM_12_1,
    SHORT_TERM_REVERSAL,
    momentum_12_1,
    short_term_reversal,
)
from backend.features.factors.risk import LOW_VOLATILITY, low_volatility
from backend.tests.features.factors.store import (
    FixtureAsOfSession,
    FixtureBar,
    FixturePriceStore,
    FixtureResult,
    bars_from_closes,
    midnight_utc,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputation, FloatArray
    from backend.features.spec import FeatureSpec

COMPUTE_DATE = dt.date(2026, 3, 2)
"""A Monday. The last usable trading day is therefore Friday 2026-02-27."""

LAST_TRADING_DAY = dt.date(2026, 2, 27)
SECURITY = 101

MOMENTUM_BARS = 253
"""Closes needed for momentum 12-1: 231 formation + 21 skipped + 1 endpoint."""

VOLATILITY_BARS = 253
"""Closes needed for a 252-return volatility estimate."""

REVERSAL_BARS = 22
"""Closes needed for a 21-bar reversal return."""

PRICE_FACTORS: list[tuple[FeatureSpec, FeatureComputation, int]] = [
    (MOMENTUM_12_1, momentum_12_1, MOMENTUM_BARS),
    (SHORT_TERM_REVERSAL, short_term_reversal, REVERSAL_BARS),
    (LOW_VOLATILITY, low_volatility, VOLATILITY_BARS),
]
"""Each price factor with the number of closes it needs, for the shared cases."""

PRICE_FACTOR_IDS = ["momentum_12_1", "short_term_reversal", "low_volatility"]


def store_with(closes: list[float], *, last_day: dt.date = LAST_TRADING_DAY) -> FixturePriceStore:
    """Build a one-security store from a list of closes ending at ``last_day``."""
    return FixturePriceStore(bars_from_closes(SECURITY, closes, last_trading_day=last_day))


async def compute(
    spec: FeatureSpec,
    computation: FeatureComputation,
    store: FixturePriceStore,
    *,
    compute_date: dt.date = COMPUTE_DATE,
    securities: tuple[int, ...] = (SECURITY,),
) -> FloatArray:
    """Run a registered computation against a fixture store at the declared cutoff.

    Mirrors what :func:`backend.features.compute.compute_feature` does — resolve
    the as-of from the declaration, then hand the computation a session pinned
    at exactly that instant — without needing a database for the session.
    """
    as_of = spec.knowledge_cutoff(compute_date)
    session = FixtureAsOfSession(store, as_of)
    request = FeatureComputeRequest(
        feature=spec.name,
        compute_date=compute_date,
        as_of=as_of,
        security_ids=securities,
    )
    return await computation(cast("AsyncSession", session), request)


# ---------------------------------------------------------------------------
# Momentum 12-1: hand-worked
# ---------------------------------------------------------------------------


async def test_momentum_is_the_log_return_from_252_bars_back_to_21_bars_back() -> None:
    """A path that moves only inside the formation window, by a factor of exactly 2.

    253 closes. The oldest (index 0, 252 bars before the newest) is 50; the close
    21 bars before the newest (index 231) is 100. Momentum must be ``ln(2)`` and
    must ignore the 21 most recent bars entirely, which are set to a third value
    so that including them would change the answer.
    """
    closes = [50.0] * 232 + [7.0] * 21
    closes[231] = 100.0
    values = await compute(MOMENTUM_12_1, momentum_12_1, store_with(closes))
    assert values.shape == (1,)
    assert values[0] == pytest.approx(math.log(2.0), abs=1e-12)


async def test_momentum_ignores_prints_older_than_the_formation_window() -> None:
    """Bars before the 252-bar endpoint are not part of the return, however extreme."""
    inside = [50.0] * 232 + [7.0] * 21
    inside[231] = 100.0
    with_history = [1.0, 900.0, 3.0, *inside]
    values = await compute(MOMENTUM_12_1, momentum_12_1, store_with(with_history))
    assert values[0] == pytest.approx(math.log(2.0), abs=1e-12)


async def test_momentum_is_negative_for_a_faller() -> None:
    """Sign convention: the factor is the raw formation return, not its negation."""
    closes = [100.0] * 232 + [1.0] * 21
    closes[231] = 80.0
    values = await compute(MOMENTUM_12_1, momentum_12_1, store_with(closes))
    assert values[0] == pytest.approx(math.log(0.8), abs=1e-12)


async def test_momentum_needs_253_prints() -> None:
    """252 closes is one short of a formation window; the answer is NaN, not a shorter window."""
    values = await compute(MOMENTUM_12_1, momentum_12_1, store_with([100.0] * (MOMENTUM_BARS - 1)))
    assert math.isnan(values[0])


async def test_momentum_refuses_a_history_stretched_beyond_the_calendar_bound() -> None:
    """253 prints spread over four years are not a twelve-month return.

    The bars are placed one per week, so the security has more than enough
    prints but they span far more than the 400-day bound the loader applies.
    Everything outside the bound is dropped and the count falls short.
    """
    days = [LAST_TRADING_DAY - dt.timedelta(weeks=i) for i in range(MOMENTUM_BARS)]
    store = FixturePriceStore(
        tuple(
            FixtureBar(
                security_id=SECURITY,
                trading_date=day,
                close_usd=Decimal("100"),
                knowledge_time=midnight_utc(day) + dt.timedelta(days=1),
            )
            for day in days
        )
    )
    values = await compute(MOMENTUM_12_1, momentum_12_1, store)
    assert math.isnan(values[0])


# ---------------------------------------------------------------------------
# Short-term reversal: hand-worked
# ---------------------------------------------------------------------------


async def test_reversal_is_the_negated_log_return_over_the_last_21_bars() -> None:
    """A stock that rose 10% over the prior month scores ``-ln(1.1)``: a recent winner."""
    closes = [100.0] * REVERSAL_BARS
    closes[-1] = 110.0
    values = await compute(SHORT_TERM_REVERSAL, short_term_reversal, store_with(closes))
    assert values[0] == pytest.approx(-math.log(1.1), abs=1e-12)


async def test_reversal_scores_a_recent_loser_positively() -> None:
    """The name carries the direction: a high score is a stock that fell."""
    closes = [100.0] * REVERSAL_BARS
    closes[-1] = 90.0
    values = await compute(SHORT_TERM_REVERSAL, short_term_reversal, store_with(closes))
    assert values[0] == pytest.approx(-math.log(0.9), abs=1e-12)
    assert values[0] > 0.0


async def test_reversal_measures_exactly_the_window_momentum_skips() -> None:
    """21 bars, so a move 22 bars back belongs to momentum's window and not to this one."""
    closes = [100.0] * (REVERSAL_BARS + 1)
    closes[0] = 5.0
    values = await compute(SHORT_TERM_REVERSAL, short_term_reversal, store_with(closes))
    assert values[0] == pytest.approx(0.0, abs=1e-12)


async def test_reversal_needs_22_prints() -> None:
    """21 closes span only 20 bars; NaN rather than a shorter window."""
    values = await compute(
        SHORT_TERM_REVERSAL, short_term_reversal, store_with([100.0] * (REVERSAL_BARS - 1))
    )
    assert math.isnan(values[0])


# ---------------------------------------------------------------------------
# Low volatility: hand-worked
# ---------------------------------------------------------------------------


async def test_low_volatility_is_the_negated_annualized_sample_deviation() -> None:
    """Alternating ±1% steps give a deviation computable in closed form.

    252 returns alternating between ``+ln(1.01)`` and ``-ln(1.01)`` have mean 0
    exactly (126 of each) and sample deviation ``ln(1.01) * sqrt(252/251)``.
    """
    step = math.log(1.01)
    closes = [100.0]
    for i in range(252):
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    values = await compute(LOW_VOLATILITY, low_volatility, store_with(closes))
    expected = -step * math.sqrt(252 / 251) * math.sqrt(252)
    assert values[0] == pytest.approx(expected, rel=1e-9)


async def test_low_volatility_is_zero_for_a_flat_price_and_never_positive() -> None:
    """A stock that never moves has zero volatility; the negation keeps it at the top."""
    values = await compute(LOW_VOLATILITY, low_volatility, store_with([100.0] * VOLATILITY_BARS))
    assert values[0] == pytest.approx(0.0, abs=1e-15)


async def test_a_calmer_stock_scores_higher_than_a_wilder_one() -> None:
    """The whole point of the sign convention, asserted rather than assumed."""
    calm = [100.0 * (1.001 if i % 2 else 1 / 1.001) for i in range(VOLATILITY_BARS)]
    wild = [100.0 * (1.05 if i % 2 else 1 / 1.05) for i in range(VOLATILITY_BARS)]
    calm_value = await compute(LOW_VOLATILITY, low_volatility, store_with(calm))
    wild_value = await compute(LOW_VOLATILITY, low_volatility, store_with(wild))
    assert calm_value[0] > wild_value[0]


async def test_low_volatility_uses_only_the_most_recent_252_returns() -> None:
    """Extra loaded history must not lengthen the estimation window.

    The 400-day calendar bound admits comfortably more than the 253 closes the
    estimate needs, so "everything the loader returned" and "the window the
    factor declares" are different sets and the factor has to pick the second.
    Here the oldest twenty closes swing by a factor of five and the newest 253
    are flat: an estimate over everything would report enormous volatility where
    the declared window contains none.
    """
    noisy = [100.0 * (5.0 if i % 2 else 1.0) for i in range(20)]
    flat = [100.0] * VOLATILITY_BARS
    values = await compute(LOW_VOLATILITY, low_volatility, store_with(noisy + flat))
    assert values[0] == pytest.approx(0.0, abs=1e-15)


async def test_low_volatility_needs_253_prints() -> None:
    """252 closes give 251 returns; NaN rather than an estimate from a shorter window."""
    values = await compute(
        LOW_VOLATILITY, low_volatility, store_with([100.0] * (VOLATILITY_BARS - 1))
    )
    assert math.isnan(values[0])


# ---------------------------------------------------------------------------
# Availability: when a factor must decline to answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "computation", "bar_count"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_a_security_with_no_bars_at_all_is_nan_not_missing(
    spec: FeatureSpec, computation: FeatureComputation, bar_count: int
) -> None:
    """An empty ``price_bar`` is the state today (P3.4 is blocked on B1): NaN, not KeyError.

    The second security has no bars, and its value must arrive as NaN in the
    right position rather than shortening the array — a misaligned feature
    vector attaches one company's numbers to another and looks entirely healthy.
    """
    store = store_with([100.0 + i for i in range(bar_count)])
    values = await compute(spec, computation, store, securities=(SECURITY, SECURITY + 1))
    assert values.shape == (2,)
    assert not math.isnan(values[0])
    assert math.isnan(values[1])
    empty = await compute(spec, computation, FixturePriceStore())
    assert math.isnan(empty[0])


@pytest.mark.parametrize(("spec", "computation", "bar_count"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_a_stale_series_is_nan_rather_than_a_stale_score(
    spec: FeatureSpec, computation: FeatureComputation, bar_count: int
) -> None:
    """A name that stopped printing keeps no score.

    The series is complete and every bar is inside the factor's calendar
    lookback; the only thing wrong with it is that the newest print is older
    than :data:`~backend.features.factors._prices.MAX_PRICE_STALENESS`, which is
    what a halt or a suspension looks like. Without the staleness rule the
    factor would happily report the momentum of a stock that is not trading.

    The gap is kept small deliberately (a fortnight, four days past the
    tolerance). A large gap would push the oldest bars out of the calendar
    lookback and produce ``NaN`` for insufficiency instead, which would leave
    the staleness rule itself untested — so the same series is also computed at
    a date it *is* fresh for, and must produce a number there.
    """
    stale_last_day = COMPUTE_DATE - dt.timedelta(days=14)
    assert stale_last_day.weekday() < 5  # a Monday; the fixture calendar needs a weekday
    assert COMPUTE_DATE - stale_last_day > MAX_PRICE_STALENESS
    closes = [100.0 * 1.002**i * (1.0 + 0.003 * (i % 5)) for i in range(bar_count)]
    store = store_with(closes, last_day=stale_last_day)

    values = await compute(spec, computation, store)
    assert math.isnan(values[0])

    fresh_enough = stale_last_day + dt.timedelta(days=1)
    assert not math.isnan((await compute(spec, computation, store, compute_date=fresh_enough))[0])


@pytest.mark.parametrize(("spec", "computation", "bar_count"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_a_corrupt_non_positive_close_gives_nan_not_an_infinity(
    spec: FeatureSpec, computation: FeatureComputation, bar_count: int
) -> None:
    """A zero print makes a log return undefined; NaN says so, ``inf`` would not.

    The transform pipeline refuses ``±inf`` outright
    (``backend.features._stats.reject_infinities``), so an infinity here would
    abort a whole cross-section for one bad tick instead of dropping one name.
    """
    closes = [100.0] * bar_count
    closes[0] = 0.0  # the oldest bar, which every factor's window starts from
    values = await compute(spec, computation, store_with(closes))
    assert math.isnan(values[0])


@pytest.mark.parametrize(("spec", "computation", "bar_count"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_values_are_returned_in_the_requested_order(
    spec: FeatureSpec, computation: FeatureComputation, bar_count: int
) -> None:
    """Positional alignment: reversing the request reverses the values, nothing else."""
    other = SECURITY + 7
    store = FixturePriceStore(
        (
            *bars_from_closes(SECURITY, [100.0] * bar_count, last_trading_day=LAST_TRADING_DAY),
            *bars_from_closes(
                other,
                [100.0 * 1.01**i * (1.03 if i % 2 else 1.0) for i in range(bar_count)],
                last_trading_day=LAST_TRADING_DAY,
            ),
        )
    )
    forward = await compute(spec, computation, store, securities=(SECURITY, other))
    backward = await compute(spec, computation, store, securities=(other, SECURITY))
    assert forward[0] == pytest.approx(backward[1])
    assert forward[1] == pytest.approx(backward[0])
    assert forward[0] != pytest.approx(forward[1])


# ---------------------------------------------------------------------------
# The loader's own contract
# ---------------------------------------------------------------------------


async def test_the_loader_issues_no_knowledge_time_predicate_of_its_own() -> None:
    """The knowledge bound belongs to the as-of session, and is not restated here.

    A second, hand-written ``knowledge_time`` filter would either duplicate
    :func:`backend.db.as_of` (misleading about where I1 is enforced) or
    contradict it (a weaker filter that quietly wins). Neither is acceptable, so
    the emitted SQL must not mention the column at all.
    """
    store = store_with([100.0] * REVERSAL_BARS)
    as_of = SHORT_TERM_REVERSAL.knowledge_cutoff(COMPUTE_DATE)
    session = FixtureAsOfSession(store, as_of)
    await load_adjusted_closes(
        cast("AsyncSession", session),
        security_ids=(SECURITY,),
        compute_date=COMPUTE_DATE,
        as_of=as_of,
        lookback_days=60,
    )
    assert len(session.statements) == 1
    sql = str(session.statements[0]).lower()
    assert "price_bar" in sql
    assert "knowledge_time" not in sql


async def test_the_loader_returns_an_ascending_series_despite_descending_rows() -> None:
    """The fixture session hands rows back newest-first; the arithmetic is positional."""
    closes = [100.0, 101.0, 102.0, 103.0]
    store = store_with(closes)
    as_of = SHORT_TERM_REVERSAL.knowledge_cutoff(COMPUTE_DATE)
    history = await load_adjusted_closes(
        cast("AsyncSession", FixtureAsOfSession(store, as_of)),
        security_ids=(SECURITY,),
        compute_date=COMPUTE_DATE,
        as_of=as_of,
        lookback_days=60,
    )
    series = history[SECURITY]
    assert list(series.trading_dates) == sorted(series.trading_dates)
    assert np.array_equal(series.adjusted_close, np.array(closes))


class DuplicatingSession(FixtureAsOfSession):
    """A session that returns two versions of one bar, as a broken store would.

    The fixture store resolves versions by ``knowledge_time`` and so cannot
    produce a duplicate; this subclass does, to exercise the loader's own
    duplicate check — the net under a store that failed to resolve.
    """

    async def execute(self, statement: object, *args: object, **kwargs: object) -> FixtureResult:
        """Return the same (security, trading day) twice at different prices."""
        await super().execute(statement, *args, **kwargs)
        day = midnight_utc(LAST_TRADING_DAY)
        return FixtureResult(
            [(SECURITY, day, Decimal("100")), (SECURITY, day, Decimal("200"))],
        )


async def test_two_visible_versions_of_one_bar_are_refused_rather_than_picked_between() -> None:
    """A duplicate means version resolution did not happen; choosing one would hide that."""
    as_of = SHORT_TERM_REVERSAL.knowledge_cutoff(COMPUTE_DATE)
    session = DuplicatingSession(FixturePriceStore(), as_of)
    with pytest.raises(PriceSeriesError, match="two visible price bars"):
        await load_adjusted_closes(
            cast("AsyncSession", session),
            security_ids=(SECURITY,),
            compute_date=COMPUTE_DATE,
            as_of=as_of,
            lookback_days=60,
        )


def test_a_series_reports_staleness_from_its_last_print() -> None:
    """The staleness rule, checked directly on the boundary it draws."""
    fresh = PriceSeries(SECURITY, (LAST_TRADING_DAY,), np.array([100.0]))
    assert not fresh.is_stale(COMPUTE_DATE, tolerance=MAX_PRICE_STALENESS)
    exactly_at_tolerance = PriceSeries(
        SECURITY, (COMPUTE_DATE - MAX_PRICE_STALENESS,), np.array([100.0])
    )
    assert not exactly_at_tolerance.is_stale(COMPUTE_DATE, tolerance=MAX_PRICE_STALENESS)
    just_past = PriceSeries(
        SECURITY,
        (COMPUTE_DATE - MAX_PRICE_STALENESS - dt.timedelta(days=1),),
        np.array([100.0]),
    )
    assert just_past.is_stale(COMPUTE_DATE, tolerance=MAX_PRICE_STALENESS)
    assert PriceSeries(SECURITY, (), np.zeros(0)).is_stale(
        COMPUTE_DATE, tolerance=MAX_PRICE_STALENESS
    )
