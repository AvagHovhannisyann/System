"""The lookahead suite: one test per factor, per gate G5's third clause (P5.3).

The property under test, stated once:

    A factor computed for rebalance date ``D`` must not change when facts whose
    ``knowledge_time`` is later than ``D``'s cutoff are added to the store.

This is the bug class that produces no symptom. A factor that reads one day too
far right returns a plausible number, of the right sign, with a healthy
distribution, and fails no type check. The only thing that changes is the
backtest, which improves — which is exactly why the directive calls invariant I1
*"the single most important property in the system"* and why this file exists
rather than being folded into the arithmetic tests.

--------------------------------------------------------------------------
Three formulations, because they fail differently
--------------------------------------------------------------------------

**1. Addition.** Compute the factor, add facts that were not knowable at the
cutoff, recompute. The value must be bit-identical. Two kinds of late fact are
added, and the second is the one that catches a subtle implementation:

- *new trading days* after the last visible bar — caught by any implementation
  that respects the visible set at all;
- *restatements of bars already inside the window* — a re-adjusted close for a
  day the factor is using, published later. An implementation that took the
  newest version of each bar regardless of knowledge time, or that resolved
  duplicates by result order, passes the first check and fails this one.

**2. Non-vacuity.** The same restatement, published *before* the cutoff, must
change the value. Without this the addition test would pass trivially for a
factor that ignored its inputs, returned a constant, or returned ``NaN``
throughout — and a suite whose central test is satisfied by a broken
implementation is worse than no suite.

**3. Detection.** A bar dated on or after ``D`` that is nonetheless visible must
raise. That state means the price connector stamped a ``knowledge_time`` earlier
than the bar's own close, which no honest policy does; consuming it would make
every value a day prescient. It is refused rather than filtered out, because a
filter would leave the defect in the store for every other consumer.

--------------------------------------------------------------------------
The six B1-blocked factors
--------------------------------------------------------------------------

Book-to-price, earnings yield, gross profitability, ROIC, accruals and asset
growth read a fundamentals table that does not exist (P3.5, blocked on B1), so
they raise. The lookahead property still has content for them and is asserted in
the only form available: no addition to the store, however late its knowledge
time, can turn the refusal into a number. A factor that started answering
because some data arrived would be answering from something other than its
declared source.

Their *declared* half of the property — that a caller cannot request a
knowledge cutoff fresher than the seven-day lag permits — is checked per factor
in ``test_declarations.py``, which is where the lag itself is asserted.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from backend.features.compute import FeatureComputeRequest
from backend.features.factors import BASELINE_FACTORS
from backend.features.factors._fundamentals import FundamentalsSourceUnavailableError
from backend.features.factors._prices import PriceTemporalIntegrityError
from backend.features.factors.growth import accruals, asset_growth
from backend.features.factors.momentum import (
    MOMENTUM_12_1,
    SHORT_TERM_REVERSAL,
    momentum_12_1,
    short_term_reversal,
)
from backend.features.factors.quality import gross_profitability, roic
from backend.features.factors.risk import LOW_VOLATILITY, low_volatility
from backend.features.factors.value import book_to_price, earnings_yield
from backend.tests.features.factors.store import (
    FixtureAsOfSession,
    FixtureBar,
    FixturePriceStore,
    bars_from_closes,
    business_days_ending,
    midnight_utc,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputation, FloatArray
    from backend.features.spec import FeatureSpec

COMPUTE_DATE = dt.date(2026, 3, 2)
"""A Monday; the last knowable trading day is Friday 2026-02-27."""

LAST_TRADING_DAY = dt.date(2026, 2, 27)
SECURITY = 101
HISTORY_BARS = 263
"""Ten closes more than the longest window (253), so the compute date can move back."""

PRICE_FACTORS: list[tuple[FeatureSpec, FeatureComputation]] = [
    (MOMENTUM_12_1, momentum_12_1),
    (SHORT_TERM_REVERSAL, short_term_reversal),
    (LOW_VOLATILITY, low_volatility),
]
PRICE_FACTOR_IDS = ["momentum_12_1", "short_term_reversal", "low_volatility"]

FUNDAMENTAL_FACTORS: list[FeatureComputation] = [
    book_to_price,
    earnings_yield,
    gross_profitability,
    roic,
    accruals,
    asset_growth,
]
FUNDAMENTAL_FACTOR_IDS = [
    "book_to_price",
    "earnings_yield",
    "gross_profitability",
    "roic",
    "accruals",
    "asset_growth",
]

SPEC_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in BASELINE_FACTORS}


def baseline_closes() -> list[float]:
    """A price path that gives every factor a distinct, non-degenerate value.

    A gentle upward drift times a five-bar sawtooth, so momentum is positive,
    the prior month's return is non-zero, and realized volatility is non-zero.
    The sawtooth period is deliberately coprime with every window length used
    here (21 and 252 bars, and the 6-bar shift in the moving-window test): a
    period that divided one of them would make the path *look* varied while
    leaving the ratio of the two window endpoints identical, and a test built on
    it would assert nothing. Obviously constructed; no claim is made that it
    resembles a real stock.
    """
    return [100.0 * 1.002**i * (1.0 + 0.003 * (i % 5)) for i in range(HISTORY_BARS)]


def baseline_store() -> FixturePriceStore:
    """The store every lookahead test starts from: one security, one clean history."""
    return FixturePriceStore(
        bars_from_closes(SECURITY, baseline_closes(), last_trading_day=LAST_TRADING_DAY)
    )


async def compute(
    spec: FeatureSpec,
    computation: FeatureComputation,
    store: FixturePriceStore,
    *,
    compute_date: dt.date = COMPUTE_DATE,
) -> FloatArray:
    """Run a computation at exactly the cutoff its own declaration permits."""
    as_of = spec.knowledge_cutoff(compute_date)
    session = FixtureAsOfSession(store, as_of)
    request = FeatureComputeRequest(
        feature=spec.name,
        compute_date=compute_date,
        as_of=as_of,
        security_ids=(SECURITY,),
    )
    return await computation(cast("AsyncSession", session), request)


def late_knowledge() -> dt.datetime:
    """An instant after every price factor's cutoff: noon on the compute date."""
    return midnight_utc(COMPUTE_DATE) + dt.timedelta(hours=12)


def restate(store: FixturePriceStore, *, knowledge_time: dt.datetime | None) -> FixturePriceStore:
    """Republish every bar at a re-scaled close, as a vendor re-adjustment would.

    The rescaling is a **ramp** — the ``i``-th bar is multiplied by
    ``1 + i/1000`` — rather than a constant factor. A constant factor would be
    invisible to momentum and reversal, which are ratios of two closes and so
    are unchanged by any uniform rescaling; a test built on one would pass
    whether or not the restatement leaked in. The ramp changes the ratio of
    every pair of closes and therefore changes all three factors.

    Args:
        store: the store whose bars are being republished.
        knowledge_time: when the corrections become knowable. ``None`` means
            "in time" — one second after each bar's own knowledge time, which is
            comfortably before every price factor's cutoff.

    Returns:
        A new store carrying the originals and the corrections.
    """
    return store.with_bars(
        FixtureBar(
            security_id=bar.security_id,
            trading_date=bar.trading_date,
            close_usd=bar.close_usd * (Decimal(1000 + index) / Decimal(1000)),
            knowledge_time=(
                bar.knowledge_time + dt.timedelta(seconds=1)
                if knowledge_time is None
                else knowledge_time
            ),
        )
        for index, bar in enumerate(store.bars)
    )


# ---------------------------------------------------------------------------
# 1. Addition: facts published after the cutoff change nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_future_trading_days_published_late_do_not_move_the_value(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """Bars for days after the compute date, knowable only later, are invisible.

    Five further trading days are appended at wildly different prices. If any of
    them reached the arithmetic the value would move by tens of percent, so an
    exact-equality assertion is the right strength here.
    """
    store = baseline_store()
    before = await compute(spec, computation, store)
    assert not math.isnan(before[0])

    future_days = [COMPUTE_DATE + dt.timedelta(days=offset) for offset in range(5)]
    later = store.with_bars(
        FixtureBar(
            security_id=SECURITY,
            trading_date=day,
            close_usd=Decimal("999"),
            knowledge_time=midnight_utc(day) + dt.timedelta(days=1),
        )
        for day in future_days
    )
    after = await compute(spec, computation, later)
    assert np.array_equal(before, after)


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_a_restatement_published_after_the_cutoff_does_not_move_the_value(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """The harder case: a *correction* to a bar the factor is already using.

    Every bar in the window is re-published at a different price with a
    knowledge time after the cutoff — a vendor re-adjustment, exactly the shape
    D-011 says a correction takes. An implementation that took the newest
    version of each bar, or that resolved duplicates by result order rather than
    by knowledge time, would silently switch to the corrected series here.
    """
    store = baseline_store()
    before = await compute(spec, computation, store)
    assert not math.isnan(before[0])

    after = await compute(spec, computation, restate(store, knowledge_time=late_knowledge()))
    assert np.array_equal(before, after)


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_a_retraction_published_after_the_cutoff_does_not_move_the_value(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """A fact withdrawn after the cutoff was still a fact at the cutoff.

    The point-in-time claim runs in both directions: a backtest dated before a
    withdrawal must see what the operator saw, not what was later admitted.
    """
    store = baseline_store()
    before = await compute(spec, computation, store)
    assert not math.isnan(before[0])

    retracted = store.with_bars(
        FixtureBar(
            security_id=bar.security_id,
            trading_date=bar.trading_date,
            close_usd=bar.close_usd,
            knowledge_time=late_knowledge(),
            is_retraction=True,
        )
        for bar in store.bars
    )
    after = await compute(spec, computation, retracted)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# 2. Non-vacuity: the same facts, published in time, do move the value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_the_same_restatement_published_in_time_does_move_the_value(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """Without this, the addition tests would pass for a factor that ignored its inputs.

    The restatement is the one from the test above, published a day *before* the
    cutoff instead of after it. Every close in the window is scaled by 3 except
    the newest, which is halved — so the level change cancels out of a ratio and
    the shape change is what must show through, for all three factors.
    """
    store = baseline_store()
    before = await compute(spec, computation, store)

    after = await compute(spec, computation, restate(store, knowledge_time=None))
    assert not math.isnan(after[0])
    assert not np.array_equal(before, after)


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_a_retraction_published_in_time_does_change_the_answer(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """Retracting the whole history before the cutoff leaves nothing to compute from."""
    store = baseline_store()
    before = await compute(spec, computation, store)
    assert not math.isnan(before[0])

    retracted = store.with_bars(
        FixtureBar(
            security_id=bar.security_id,
            trading_date=bar.trading_date,
            close_usd=bar.close_usd,
            knowledge_time=bar.knowledge_time + dt.timedelta(seconds=1),
            is_retraction=True,
        )
        for bar in store.bars
    )
    after = await compute(spec, computation, retracted)
    assert math.isnan(after[0])


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_moving_the_compute_date_forward_moves_the_window(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """A factor pinned to a date must actually depend on that date.

    A factor that always used "the newest bars in the store" would pass every
    addition test above — the store's newest bars are the same ones — while
    being entirely wrong about which window it measured. Computing the same
    store at an earlier compute date must therefore give a different answer.
    """
    store = baseline_store()
    at_date = await compute(spec, computation, store)
    earlier_day = business_days_ending(LAST_TRADING_DAY, 6)[0]
    earlier = await compute(spec, computation, store, compute_date=earlier_day)
    assert not math.isnan(at_date[0])
    assert not math.isnan(earlier[0])
    assert at_date[0] != earlier[0]


# ---------------------------------------------------------------------------
# 3. Detection: a bar that becomes visible too early is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_a_bar_dated_on_the_compute_date_and_visible_is_refused(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """The store defect a lag could hide but not fix.

    The bar for the compute date is stamped knowable at ``D 00:00Z`` — the
    trading day's *open*, which is what a connector reusing ``valid_from`` as
    its knowledge time would produce. It is therefore visible to a zero-lag
    feature, and consuming it would use the compute date's own close to trade on
    the compute date.
    """
    tainted = baseline_store().with_bars(
        [
            FixtureBar(
                security_id=SECURITY,
                trading_date=COMPUTE_DATE,
                close_usd=Decimal("500"),
                knowledge_time=midnight_utc(COMPUTE_DATE),
            )
        ]
    )
    with pytest.raises(PriceTemporalIntegrityError) as raised:
        await compute(spec, computation, tainted)
    assert raised.value.bar_date == COMPUTE_DATE
    assert raised.value.compute_date == COMPUTE_DATE
    assert raised.value.security_id == SECURITY


@pytest.mark.parametrize(("spec", "computation"), PRICE_FACTORS, ids=PRICE_FACTOR_IDS)
async def test_the_same_bar_stamped_honestly_is_simply_invisible(
    spec: FeatureSpec, computation: FeatureComputation
) -> None:
    """Non-vacuity for the detector: it fires on the defect, not on the bar's existence.

    The identical bar, stamped with an honest knowledge time (after its own
    close), does not raise and does not move the value — it is filtered by the
    knowledge bound, which is where that job belongs.
    """
    store = baseline_store()
    before = await compute(spec, computation, store)
    honest = store.with_bars(
        [
            FixtureBar(
                security_id=SECURITY,
                trading_date=COMPUTE_DATE,
                close_usd=Decimal("500"),
                knowledge_time=midnight_utc(COMPUTE_DATE) + dt.timedelta(hours=21),
            )
        ]
    )
    after = await compute(spec, computation, honest)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# The six B1-blocked factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("computation", FUNDAMENTAL_FACTORS, ids=FUNDAMENTAL_FACTOR_IDS)
async def test_no_late_arriving_data_can_turn_a_blocked_refusal_into_a_number(
    computation: FeatureComputation,
) -> None:
    """A factor whose source does not exist answers from nothing, before and after.

    The store is filled with price bars — including bars published long after
    the cutoff — and the refusal is identical. A fundamental factor that started
    returning values because *price* data arrived would be computing from
    something other than the fundamentals it declares.
    """
    name = FUNDAMENTAL_FACTOR_IDS[FUNDAMENTAL_FACTORS.index(computation)]
    spec = SPEC_BY_NAME[name]
    empty = FixturePriceStore()
    populated = baseline_store().with_bars(
        FixtureBar(
            security_id=SECURITY,
            trading_date=COMPUTE_DATE + dt.timedelta(days=offset),
            close_usd=Decimal("999"),
            knowledge_time=late_knowledge() + dt.timedelta(days=offset),
        )
        for offset in range(5)
    )
    for store in (empty, populated):
        with pytest.raises(FundamentalsSourceUnavailableError) as raised:
            await compute(spec, computation, store)
        assert raised.value.feature == name
        assert raised.value.blocker == "B1"
