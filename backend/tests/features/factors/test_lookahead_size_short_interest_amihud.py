"""The lookahead suite for P5.3's remaining three factors, per gate G5 (P5.3).

The property under test, stated once and identically to
``test_lookahead.py``:

    A factor computed for rebalance date ``D`` must not change when facts whose
    ``knowledge_time`` is later than ``D``'s cutoff are added to the store.

This file covers ``amihud_illiquidity``, ``size`` and ``short_interest``. It is
separate from ``test_lookahead.py`` for the same reason that file is separate
from the arithmetic tests — I1 is the property the directive calls *"the single
most important in the system"*, and a bug in it produces a plausible number, of
the right sign, with a healthy distribution, that fails no type check and
improves the backtest.

--------------------------------------------------------------------------
``amihud_illiquidity``: the same three formulations, plus a volume leg
--------------------------------------------------------------------------

**1. Addition.** Compute, add facts that were not knowable at the cutoff,
recompute, require bit-identical values. Four kinds of late fact, and the last
two are the ones a careless implementation survives the first two on: new trading
days; *price* restatements of bars already inside the window; *volume*
restatements of those same bars; and retractions.

The volume leg is this factor's own exposure and is not covered anywhere else in
the package — Amihud is the only factor that reads ``volume_shares``, so a loader
that resolved price versions correctly and volume versions by result order would
pass every existing lookahead test in the repository.

**2. Non-vacuity.** The same restatements, published *before* the cutoff, must
change the value — and the volume one must change it by an exactly predictable
factor, since tripling every day's shares traded triples every denominator.
Without this the addition tests would pass for an implementation that ignored its
inputs or returned ``NaN`` throughout.

**3. Detection.** A bar dated on or after ``D`` that is nonetheless visible must
raise, because that state means the price connector stamped a ``knowledge_time``
earlier than the bar's own close. It is refused rather than filtered: a filter
would leave the defect in the store for every other consumer.

--------------------------------------------------------------------------
``size`` and ``short_interest``: the property still has content
--------------------------------------------------------------------------

Both refuse — ``size`` because its share count comes from the B1-blocked
fundamentals feed, ``short_interest`` because nothing anywhere produces its
numerator. The lookahead property is asserted in the only form available: **no
addition to the store, however late its knowledge time, can turn the refusal into
a number.** A factor that started answering because some data arrived would be
answering from something other than its declared source. The store used is
deliberately full — including bars dated after the compute date and published
long after the cutoff — because that is the state most likely to tempt a
half-written implementation into reading something.

Their *declared* half of the property — that a caller cannot request a cutoff
fresher than the lag permits — is asserted per factor in
``test_size_liquidity.py`` and ``test_short_interest.py``, alongside the lags
themselves.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from backend.features.compute import FeatureComputeRequest
from backend.features.factors._fundamentals import FundamentalsSourceUnavailableError
from backend.features.factors._prices import PriceTemporalIntegrityError
from backend.features.factors.liquidity import AMIHUD_ILLIQUIDITY, amihud_illiquidity
from backend.features.factors.short_interest import (
    SHORT_INTEREST,
    ShortInterestSourceUnavailableError,
    short_interest,
)
from backend.features.factors.size import SIZE, size
from backend.tests.features.factors.store import business_days_ending, midnight_utc
from backend.tests.features.factors.volume_store import (
    FixtureVolumeBar,
    FixtureVolumeSession,
    FixtureVolumeStore,
    volume_bars,
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
"""Ten closes more than Amihud's window (253), so the compute date can move back."""

VOLUME_MULTIPLE = 3
"""Factor a whole-history volume restatement multiplies every day's shares by.

Chosen so the effect on the answer is exact and writable down: every dollar
volume triples, every ratio's denominator triples, so the mean of the ratios is
divided by three. An approximate "the value moved" assertion would pass for an
implementation that moved it for the wrong reason.
"""

BLOCKED_FACTORS: list[tuple[FeatureSpec, FeatureComputation, type[Exception]]] = [
    (SIZE, size, FundamentalsSourceUnavailableError),
    (SHORT_INTEREST, short_interest, ShortInterestSourceUnavailableError),
]
BLOCKED_IDS = ["size", "short_interest"]


def baseline_closes() -> list[float]:
    """A price path giving Amihud a distinct, non-degenerate value.

    A gentle upward drift times a **seven-bar** sawtooth. The period is chosen
    coprime with the six-business-day shift the moving-window test applies and
    with the 252-observation window: a period dividing either would leave the
    sequence of absolute returns literally unchanged under the shift, and the
    moving-window test would then be asserting nothing about the price leg.
    Obviously constructed; no claim is made that it resembles a real stock.
    """
    return [100.0 * 1.002**index * (1.0 + 0.004 * (index % 7)) for index in range(HISTORY_BARS)]


def baseline_volumes() -> list[int]:
    """Share counts with an eleven-bar cycle, coprime with the price path's seven.

    Two coprime periods mean a shifted window re-pairs returns with different
    volumes, so the moving-window test exercises both legs rather than one.
    """
    return [1_000_000 + 30_000 * (index % 11) for index in range(HISTORY_BARS)]


def baseline_store() -> FixtureVolumeStore:
    """The store every lookahead test starts from: one security, one clean history."""
    return FixtureVolumeStore(
        volume_bars(
            SECURITY,
            baseline_closes(),
            baseline_volumes(),
            last_trading_day=LAST_TRADING_DAY,
        )
    )


async def compute(
    spec: FeatureSpec,
    computation: FeatureComputation,
    store: FixtureVolumeStore,
    *,
    compute_date: dt.date = COMPUTE_DATE,
) -> FloatArray:
    """Run a computation at exactly the cutoff its own declaration permits."""
    as_of = spec.knowledge_cutoff(compute_date)
    session = FixtureVolumeSession(store, as_of)
    request = FeatureComputeRequest(
        feature=spec.name,
        compute_date=compute_date,
        as_of=as_of,
        security_ids=(SECURITY,),
    )
    return await computation(cast("AsyncSession", session), request)


def late_knowledge() -> dt.datetime:
    """An instant after the zero-lag cutoff: noon on the compute date."""
    return midnight_utc(COMPUTE_DATE) + dt.timedelta(hours=12)


def in_time(bar: FixtureVolumeBar) -> dt.datetime:
    """One second after a bar's own knowledge time — comfortably before the cutoff."""
    return bar.knowledge_time + dt.timedelta(seconds=1)


def restate_prices(
    store: FixtureVolumeStore, *, knowledge_time: dt.datetime | None
) -> FixtureVolumeStore:
    """Republish every bar at a re-scaled close, as a vendor re-adjustment would.

    The rescaling is a **ramp** — the ``i``-th bar is multiplied by
    ``1 + i/1000`` — rather than a constant factor. A constant factor would leave
    every daily log return unchanged, so Amihud's numerator would not move and the
    test would pass whether or not the restatement leaked in. The ramp changes the
    ratio of every consecutive pair, and it changes the unadjusted close too, so
    both legs move.

    Args:
        store: the store whose bars are being republished.
        knowledge_time: when the corrections become knowable. ``None`` means "in
            time" — one second after each bar's own knowledge time.

    Returns:
        A new store carrying the originals and the corrections.
    """
    return store.with_bars(
        FixtureVolumeBar(
            security_id=bar.security_id,
            trading_date=bar.trading_date,
            close_usd=bar.close_usd * (Decimal(1000 + index) / Decimal(1000)),
            close_raw_usd=bar.close_raw_usd * (Decimal(1000 + index) / Decimal(1000)),
            volume_shares=bar.volume_shares,
            knowledge_time=in_time(bar) if knowledge_time is None else knowledge_time,
        )
        for index, bar in enumerate(store.bars)
    )


def restate_volumes(
    store: FixtureVolumeStore, *, knowledge_time: dt.datetime | None
) -> FixtureVolumeStore:
    """Republish every bar with the same prices and ``VOLUME_MULTIPLE`` times the shares.

    The leg no other factor in the package reads. A loader that resolved price
    versions by ``knowledge_time`` but volume versions by result order — or that
    took the newest row's volume regardless — passes every price-only lookahead
    test in the repository and fails this one.

    Args:
        store: the store whose bars are being republished.
        knowledge_time: when the corrections become knowable. ``None`` means "in
            time" — one second after each bar's own knowledge time.

    Returns:
        A new store carrying the originals and the corrections.
    """
    return store.with_bars(
        FixtureVolumeBar(
            security_id=bar.security_id,
            trading_date=bar.trading_date,
            close_usd=bar.close_usd,
            close_raw_usd=bar.close_raw_usd,
            volume_shares=bar.volume_shares * VOLUME_MULTIPLE,
            knowledge_time=in_time(bar) if knowledge_time is None else knowledge_time,
        )
        for bar in store.bars
    )


# ---------------------------------------------------------------------------
# 1. Addition: facts published after the cutoff change nothing
# ---------------------------------------------------------------------------


async def test_future_trading_days_published_late_do_not_move_the_value() -> None:
    """Bars for days after the compute date, knowable only later, are invisible.

    Five further trading days are appended at a wildly different price and a
    hundredth of the volume. If any of them reached the arithmetic the value would
    move by orders of magnitude, so exact equality is the right strength here.
    """
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert not math.isnan(before[0])

    later = store.with_bars(
        FixtureVolumeBar(
            security_id=SECURITY,
            trading_date=COMPUTE_DATE + dt.timedelta(days=offset),
            close_usd=Decimal("999"),
            close_raw_usd=Decimal("999"),
            volume_shares=10_000,
            knowledge_time=midnight_utc(COMPUTE_DATE + dt.timedelta(days=offset))
            + dt.timedelta(days=1),
        )
        for offset in range(5)
    )
    after = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, later)
    assert np.array_equal(before, after)


async def test_a_price_restatement_published_after_the_cutoff_does_not_move_the_value() -> None:
    """A *correction* to bars the factor is already using, published too late.

    Exactly the shape D-011 says a re-adjustment takes. An implementation that
    took the newest version of each bar, or resolved duplicates by result order
    rather than by knowledge time, would silently switch to the corrected series.
    """
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert not math.isnan(before[0])

    after = await compute(
        AMIHUD_ILLIQUIDITY,
        amihud_illiquidity,
        restate_prices(store, knowledge_time=late_knowledge()),
    )
    assert np.array_equal(before, after)


async def test_a_volume_restatement_published_after_the_cutoff_does_not_move_the_value() -> None:
    """This factor's own leg: volumes are corrected too, and corrections are versioned.

    Nothing else in the package reads ``volume_shares``, so this is the only test
    in the repository that can see a loader which respected knowledge time for
    prices and ignored it for volume.
    """
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert not math.isnan(before[0])

    after = await compute(
        AMIHUD_ILLIQUIDITY,
        amihud_illiquidity,
        restate_volumes(store, knowledge_time=late_knowledge()),
    )
    assert np.array_equal(before, after)


async def test_a_retraction_published_after_the_cutoff_does_not_move_the_value() -> None:
    """A fact withdrawn after the cutoff was still a fact at the cutoff.

    The point-in-time claim runs in both directions: a backtest dated before a
    withdrawal must see what the operator saw, not what was later admitted.
    """
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert not math.isnan(before[0])

    retracted = store.with_bars(
        FixtureVolumeBar(
            security_id=bar.security_id,
            trading_date=bar.trading_date,
            close_usd=bar.close_usd,
            close_raw_usd=bar.close_raw_usd,
            volume_shares=bar.volume_shares,
            knowledge_time=late_knowledge(),
            is_retraction=True,
        )
        for bar in store.bars
    )
    after = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, retracted)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# 2. Non-vacuity: the same facts, published in time, do move the value
# ---------------------------------------------------------------------------


async def test_the_same_price_restatement_published_in_time_does_move_the_value() -> None:
    """Without this, the addition tests would pass for a factor that ignored its inputs."""
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)

    after = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, restate_prices(store, knowledge_time=None)
    )
    assert not math.isnan(after[0])
    assert not np.array_equal(before, after)


async def test_the_same_volume_restatement_published_in_time_divides_the_value_exactly() -> None:
    """Non-vacuity with an exact expected magnitude, not merely "it changed".

    Tripling every day's shares traded triples every day's dollar volume, so every
    ratio's denominator triples and the mean of the ratios is divided by three. A
    test asserting only inequality would pass for an implementation that moved the
    value for some entirely different reason.
    """
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert not math.isnan(before[0])

    after = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, restate_volumes(store, knowledge_time=None)
    )
    assert after[0] == pytest.approx(before[0] / VOLUME_MULTIPLE, rel=1e-12)


async def test_a_retraction_published_in_time_does_change_the_answer() -> None:
    """Retracting the whole history before the cutoff leaves nothing to compute from."""
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert not math.isnan(before[0])

    retracted = store.with_bars(
        FixtureVolumeBar(
            security_id=bar.security_id,
            trading_date=bar.trading_date,
            close_usd=bar.close_usd,
            close_raw_usd=bar.close_raw_usd,
            volume_shares=bar.volume_shares,
            knowledge_time=in_time(bar),
            is_retraction=True,
        )
        for bar in store.bars
    )
    after = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, retracted)
    assert math.isnan(after[0])


async def test_moving_the_compute_date_forward_moves_the_window() -> None:
    """A factor pinned to a date must actually depend on that date.

    A factor that always used "the newest bars in the store" would pass every
    addition test above — the store's newest bars are the same ones — while being
    entirely wrong about which window it measured. Computing the same store at an
    earlier compute date must therefore give a different answer.
    """
    store = baseline_store()
    at_date = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    earlier_day = business_days_ending(LAST_TRADING_DAY, 6)[0]
    earlier = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store, compute_date=earlier_day)
    assert not math.isnan(at_date[0])
    assert not math.isnan(earlier[0])
    assert at_date[0] != earlier[0]


# ---------------------------------------------------------------------------
# 3. Detection: a bar that becomes visible too early is refused
# ---------------------------------------------------------------------------


async def test_a_bar_dated_on_the_compute_date_and_visible_is_refused() -> None:
    """The store defect a lag could hide but not fix.

    The bar for the compute date is stamped knowable at ``D 00:00Z`` — the trading
    day's *open*, which is what a connector reusing ``valid_from`` as its
    knowledge time would produce. It is therefore visible to a zero-lag feature,
    and consuming it would use the compute date's own bar to trade on the compute
    date. Amihud is more exposed to this than a 252-bar standard deviation is: it
    averages ``1/volume``, which is convex, so one extra thin day can move the
    estimate materially.
    """
    tainted = baseline_store().with_bars(
        [
            FixtureVolumeBar(
                security_id=SECURITY,
                trading_date=COMPUTE_DATE,
                close_usd=Decimal("500"),
                close_raw_usd=Decimal("500"),
                volume_shares=1_000,
                knowledge_time=midnight_utc(COMPUTE_DATE),
            )
        ]
    )
    with pytest.raises(PriceTemporalIntegrityError) as raised:
        await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, tainted)
    assert raised.value.bar_date == COMPUTE_DATE
    assert raised.value.compute_date == COMPUTE_DATE
    assert raised.value.security_id == SECURITY


async def test_the_same_bar_stamped_honestly_is_simply_invisible() -> None:
    """Non-vacuity for the detector: it fires on the defect, not on the bar's existence.

    The identical bar, stamped with an honest knowledge time (after its own
    close), does not raise and does not move the value — it is filtered by the
    knowledge bound, which is where that job belongs.
    """
    store = baseline_store()
    before = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    honest = store.with_bars(
        [
            FixtureVolumeBar(
                security_id=SECURITY,
                trading_date=COMPUTE_DATE,
                close_usd=Decimal("500"),
                close_raw_usd=Decimal("500"),
                volume_shares=1_000,
                knowledge_time=midnight_utc(COMPUTE_DATE) + dt.timedelta(hours=21),
            )
        ]
    )
    after = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, honest)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# The two blocked factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "computation", "expected"), BLOCKED_FACTORS, ids=BLOCKED_IDS)
async def test_no_late_arriving_data_can_turn_a_blocked_refusal_into_a_number(
    spec: FeatureSpec, computation: FeatureComputation, expected: type[Exception]
) -> None:
    """A factor whose source does not exist answers from nothing, before and after.

    The store is filled with price bars — including bars dated after the compute
    date and published long after the cutoff — and the refusal is identical, and
    is the *same* refusal in both cases. A blocked factor that started returning
    values because price data arrived would be computing from something other than
    the source it declares; one that started raising the price loader's
    temporal-integrity error instead would be reading a table it never declared.
    """
    empty = FixtureVolumeStore()
    populated = baseline_store().with_bars(
        FixtureVolumeBar(
            security_id=SECURITY,
            trading_date=COMPUTE_DATE + dt.timedelta(days=offset),
            close_usd=Decimal("999"),
            close_raw_usd=Decimal("999"),
            volume_shares=10_000,
            knowledge_time=late_knowledge() + dt.timedelta(days=offset),
        )
        for offset in range(5)
    )
    for store in (empty, populated):
        with pytest.raises(expected) as raised:
            await compute(spec, computation, store)
        assert getattr(raised.value, "feature", None) == spec.name


@pytest.mark.parametrize(("spec", "computation", "expected"), BLOCKED_FACTORS, ids=BLOCKED_IDS)
async def test_a_blocked_factor_refuses_before_touching_the_session_it_was_given(
    spec: FeatureSpec, computation: FeatureComputation, expected: type[Exception]
) -> None:
    """No I/O means no half-done refusal, and no chance to read something late.

    The session records every statement executed against it. A blocked factor must
    leave it untouched: a computation that queried and *then* refused would have
    read at its pinned instant for no reason, and the next careless edit would
    make it return what it read.
    """
    as_of = spec.knowledge_cutoff(COMPUTE_DATE)
    session = FixtureVolumeSession(baseline_store(), as_of)
    request = FeatureComputeRequest(
        feature=spec.name,
        compute_date=COMPUTE_DATE,
        as_of=as_of,
        security_ids=(SECURITY,),
    )
    with pytest.raises(expected):
        await computation(cast("AsyncSession", session), request)
    assert session.statements == []
