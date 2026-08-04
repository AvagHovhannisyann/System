"""Size and Amihud illiquidity: declarations, arithmetic, and one refusal (P5.3).

Two of the three factors that complete P5.3's baseline library, and they sit at
opposite ends of the package's honesty spectrum.

``amihud_illiquidity`` computes. Its arithmetic is checked against hand-worked
examples — price paths whose absolute log return is constant by construction and
volume paths whose dollar volume is constant by construction, so the mean of the
ratio can be written down in closed form and compared exactly rather than
compared against whatever the implementation happens to produce. The awkward part
of this factor is its units and its alignment, so those get their own tests: the
denominator must be built from the *unadjusted* close and volume while the
numerator comes from the *adjusted* close, and each return must be divided by the
volume of the day it ended on rather than the day before.

``size`` does not compute, and could not honestly. Its numerator is a share count
and no table in this store holds one. What is tested is that it refuses, that the
refusal names what has to be unblocked, and — the test that matters most — that a
full year of price history does not tempt it into answering, because price per
share is not market value and a factor that quietly ranked on share price would
look entirely healthy.

The lags, units and source tables of both are asserted against literals written
out by hand from each module's stated justification, following
``test_declarations.py``: a test that recomputes the formula it is checking
cannot see an error in that formula (D-027).
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from backend.features.compute import FeatureComputeRequest, resolve_as_of
from backend.features.errors import (
    AvailabilityLagViolationError,
    FeatureComputeError,
    FeatureError,
)
from backend.features.factors import BASELINE_FACTORS, FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE
from backend.features.factors._fundamentals import (
    FUNDAMENTALS_BLOCKER,
    FUNDAMENTALS_CONNECTOR_TASK,
    FundamentalsSourceUnavailableError,
)
from backend.features.factors._prices import MAX_PRICE_STALENESS, PriceSeries, PriceSeriesError
from backend.features.factors.liquidity import (
    AMIHUD_ILLIQUIDITY,
    AMIHUD_LOOKBACK_DAYS,
    AMIHUD_OBSERVATIONS,
    DollarVolumeSeries,
    amihud_illiquidity,
    load_dollar_volume_bars,
)
from backend.features.factors.size import SIZE, size
from backend.features.registry import default_registry
from backend.features.spec import MAX_FEATURES, compute_instant
from backend.tests.features.factors.store import midnight_utc
from backend.tests.features.factors.volume_store import (
    FixtureVolumeBar,
    FixtureVolumeResult,
    FixtureVolumeSession,
    FixtureVolumeStore,
    volume_bars,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputation, FloatArray
    from backend.features.spec import FeatureSpec

COMPUTE_DATE = dt.date(2026, 3, 2)
"""A Monday. The last usable trading day is therefore Friday 2026-02-27."""

LAST_TRADING_DAY = dt.date(2026, 2, 27)
SECURITY = 101

AMIHUD_BARS = 253
"""Closes needed for a 252-observation Amihud estimate."""

FLAT_VOLUME = 1_000_000
"""A round share count, so a dollar volume is checkable by eye."""


def alternating_closes(count: int = AMIHUD_BARS) -> list[float]:
    """A path whose every daily absolute log return is exactly ``ln(1.01)``.

    Closes alternate between 100 and 101, so each consecutive ratio is 101/100 or
    100/101 and the absolute log return is the same number every day. That is
    what makes the mean of ``|r| / dollar_volume`` writable in closed form.
    """
    return [100.0 if index % 2 == 0 else 101.0 for index in range(count)]


def matching_volumes(closes: Sequence[float], *, traded_usd: float) -> list[int]:
    """Share counts that make every day's dollar volume exactly ``traded_usd``.

    The prices above are 100 and 101 and ``traded_usd`` is chosen as a multiple of
    both, so the share counts come out as exact integers and nothing is rounded.
    """
    return [round(traded_usd / close) for close in closes]


def store_with(
    closes: Sequence[float],
    volumes: Sequence[int],
    *,
    raw_closes: Sequence[float] | None = None,
    last_day: dt.date = LAST_TRADING_DAY,
) -> FixtureVolumeStore:
    """Build a one-security store from closes and volumes ending at ``last_day``."""
    return FixtureVolumeStore(
        volume_bars(
            SECURITY,
            closes,
            volumes,
            last_trading_day=last_day,
            raw_closes=raw_closes,
        )
    )


async def compute(
    spec: FeatureSpec,
    computation: FeatureComputation,
    store: FixtureVolumeStore,
    *,
    compute_date: dt.date = COMPUTE_DATE,
    securities: tuple[int, ...] = (SECURITY,),
) -> FloatArray:
    """Run a registered computation against a fixture store at the declared cutoff.

    Mirrors what :func:`backend.features.compute.compute_feature` does — resolve
    the as-of from the declaration, then hand the computation a session pinned at
    exactly that instant — without needing a database for the session.
    """
    as_of = spec.knowledge_cutoff(compute_date)
    session = FixtureVolumeSession(store, as_of)
    request = FeatureComputeRequest(
        feature=spec.name,
        compute_date=compute_date,
        as_of=as_of,
        security_ids=securities,
    )
    return await computation(cast("AsyncSession", session), request)


# ---------------------------------------------------------------------------
# Declarations: lags, units and source tables, written out by hand
# ---------------------------------------------------------------------------


def test_both_declarations_are_registered_in_the_process_wide_catalog() -> None:
    """Importing the package populates the registry the 30-feature cap is about."""
    registry = default_registry()
    for spec in (SIZE, AMIHUD_ILLIQUIDITY):
        assert spec.name in registry
        assert registry.spec(spec.name) is spec


def test_amihud_declares_a_zero_availability_lag() -> None:
    """D-027's price regime: a daily bar is knowable at its own close."""
    assert AMIHUD_ILLIQUIDITY.availability_lag == dt.timedelta(0)


def test_size_declares_the_seven_day_fundamentals_margin() -> None:
    """The share count is the slower leg, so size sits in D-027's other regime."""
    assert SIZE.availability_lag == dt.timedelta(days=7)


def test_the_declared_source_tables_are_exactly_what_each_factor_reads() -> None:
    """Impact analysis depends on this: which features a connector change touches."""
    assert AMIHUD_ILLIQUIDITY.source_tables == frozenset({PRICE_SOURCE_TABLE})
    assert SIZE.source_tables == frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE})


def test_the_knowledge_cutoffs_are_the_stated_instants() -> None:
    """The lag's entire operational content, against literals rather than a formula.

    D-027's standing lesson: a test that recomputes ``compute_instant(d) - lag``
    agrees with a sign error in that expression. These instants are written out.
    """
    assert AMIHUD_ILLIQUIDITY.knowledge_cutoff(COMPUTE_DATE).isoformat() == (
        "2026-03-02T00:00:00+00:00"
    )
    assert SIZE.knowledge_cutoff(COMPUTE_DATE).isoformat() == "2026-02-23T00:00:00+00:00"


@pytest.mark.parametrize("spec", [AMIHUD_ILLIQUIDITY, SIZE], ids=["amihud_illiquidity", "size"])
def test_a_caller_may_not_ask_for_an_instant_fresher_than_the_lag_permits(
    spec: FeatureSpec,
) -> None:
    """One microsecond past the cutoff is refused, before any session is opened."""
    permitted = spec.knowledge_cutoff(COMPUTE_DATE)
    assert resolve_as_of(spec, COMPUTE_DATE) == permitted
    with pytest.raises(AvailabilityLagViolationError):
        resolve_as_of(spec, COMPUTE_DATE, requested_as_of=permitted + dt.timedelta(microseconds=1))


@pytest.mark.parametrize("spec", [AMIHUD_ILLIQUIDITY, SIZE], ids=["amihud_illiquidity", "size"])
def test_a_caller_may_reconstruct_either_factor_at_an_older_instant(spec: FeatureSpec) -> None:
    """Reading older data is always I1-safe: it is what a historical replay does."""
    older = spec.knowledge_cutoff(COMPUTE_DATE) - dt.timedelta(days=30)
    assert resolve_as_of(spec, COMPUTE_DATE, requested_as_of=older) == older


def test_the_units_state_the_family_and_the_scale_of_each_number() -> None:
    """Directive §8: a float64 column carries no unit, so the declaration must."""
    assert "1/USD" in AMIHUD_ILLIQUIDITY.units
    assert "252" in AMIHUD_ILLIQUIDITY.units
    assert "not scaled by 1e6" in AMIHUD_ILLIQUIDITY.units
    assert "natural log of market capitalization measured in USD" in SIZE.units


@pytest.mark.parametrize("spec", [AMIHUD_ILLIQUIDITY, SIZE], ids=["amihud_illiquidity", "size"])
def test_the_units_are_not_percent_or_basis_points(spec: FeatureSpec) -> None:
    """The two forms that are the silent bug class this project spends effort on."""
    units = spec.units.lower()
    assert "percent" not in units
    assert "basis point" not in units


def test_each_definition_states_the_expected_premium_sign() -> None:
    """P5.4 tests a premium against a stated expectation, not a remembered one.

    Opposite signs, and both follow the package's rule that a factor named after a
    measured quantity is not negated: illiquid names earn a premium, large ones
    do not.
    """
    assert "Expected premium sign: POSITIVE" in AMIHUD_ILLIQUIDITY.definition
    assert "Expected premium sign: NEGATIVE" in SIZE.definition


def test_the_size_definition_names_the_shortcut_it_refuses_to_take() -> None:
    """A reviewer must be able to see that "just use price" was considered and rejected."""
    assert "price per share is not market value" in SIZE.definition


# ---------------------------------------------------------------------------
# Amihud illiquidity: hand-worked
# ---------------------------------------------------------------------------


async def test_amihud_is_the_mean_of_absolute_return_over_dollar_volume() -> None:
    """Constant ``|r|`` over constant dollar volume: the mean is one exact ratio.

    Closes alternate 100/101 so every daily absolute log return is ``ln(1.01)``;
    share counts are chosen so every day's dollar volume is exactly 101,000,000
    USD. The mean of 252 identical ratios is that ratio.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    assert set(volumes) == {1_010_000, 1_000_000}  # exact, nothing rounded away

    values = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, volumes))
    assert values.shape == (1,)
    assert values[0] == pytest.approx(math.log(1.01) / 101_000_000.0, rel=1e-12)


async def test_a_thinner_stock_scores_proportionally_higher() -> None:
    """Ten times less trading for the same price path is ten times the illiquidity.

    Also the sign convention, asserted rather than assumed: the factor is named
    after the quantity it measures, so a HIGH score is the ILLIQUID name.
    """
    closes = alternating_closes()
    liquid = matching_volumes(closes, traded_usd=101_000_000.0)
    thin = [volume // 10 for volume in liquid]

    liquid_value = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, liquid))
    thin_value = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, thin))
    assert thin_value[0] == pytest.approx(10.0 * liquid_value[0], rel=1e-12)
    assert thin_value[0] > liquid_value[0]


async def test_dollar_volume_uses_the_unadjusted_close_not_the_adjusted_one() -> None:
    """The units point, isolated so nothing else can explain a pass.

    Two stores with identical adjusted closes and identical share counts, so the
    return leg is identical in both. They differ only in ``close_raw_usd``, which
    is doubled in the second — a security whose adjustment factor is 1/2, as it
    would be after a 2-for-1 split. Real traded dollars therefore double and the
    illiquidity must halve. An implementation that built dollar volume from
    ``close_usd`` would return the same number twice.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    doubled_raw = [close * 2.0 for close in closes]

    on_adjusted_basis = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, volumes)
    )
    on_raw_basis = await compute(
        AMIHUD_ILLIQUIDITY,
        amihud_illiquidity,
        store_with(closes, volumes, raw_closes=doubled_raw),
    )
    assert on_raw_basis[0] == pytest.approx(on_adjusted_basis[0] / 2.0, rel=1e-12)


async def test_each_return_is_divided_by_the_volume_of_the_day_it_ended_on() -> None:
    """Alignment, tested at the one index where an off-by-one is visible.

    A 253-bar window has 252 observations, and the oldest bar contributes only the
    price the first return is measured *from* — its volume is never a denominator.
    So zeroing the oldest bar's volume must leave the value untouched, while
    zeroing the next bar's volume must void the estimate. An implementation that
    paired return ``t`` with volume ``t-1`` gets both of these backwards.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    expected = math.log(1.01) / 101_000_000.0

    oldest_untraded = list(volumes)
    oldest_untraded[0] = 0
    unaffected = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, oldest_untraded)
    )
    assert unaffected[0] == pytest.approx(expected, rel=1e-12)

    second_untraded = list(volumes)
    second_untraded[1] = 0
    voided = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, second_untraded)
    )
    assert math.isnan(voided[0])


async def test_amihud_uses_only_the_most_recent_252_observations() -> None:
    """Extra loaded history must not lengthen the estimation window.

    The 400-day calendar bound admits comfortably more than the 253 closes the
    estimate needs, so "everything the loader returned" and "the window the factor
    declares" are different sets and the factor has to pick the second. Here the
    oldest twenty bars swing by a factor of five on a hundredth of the volume —
    an estimate over everything would report illiquidity orders of magnitude
    larger than the declared window contains.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    noisy_closes = [100.0 * (5.0 if index % 2 else 1.0) for index in range(20)]
    noisy_volumes = [10_000] * 20

    values = await compute(
        AMIHUD_ILLIQUIDITY,
        amihud_illiquidity,
        store_with(noisy_closes + closes, noisy_volumes + volumes),
    )
    assert values[0] == pytest.approx(math.log(1.01) / 101_000_000.0, rel=1e-12)


async def test_amihud_needs_253_prints() -> None:
    """252 closes give 251 observations; NaN rather than a shorter window."""
    closes = alternating_closes(AMIHUD_BARS - 1)
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    values = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, volumes))
    assert math.isnan(values[0])


async def test_amihud_refuses_a_history_stretched_beyond_the_calendar_bound() -> None:
    """253 prints spread over five years are not a one-year liquidity estimate.

    The bars are placed one per week, so the security has enough prints but they
    span far more than the 400-day bound the loader applies. Everything outside
    the bound is dropped and the count falls short.
    """
    days = [LAST_TRADING_DAY - dt.timedelta(weeks=index) for index in range(AMIHUD_BARS)]
    store = FixtureVolumeStore(
        tuple(
            FixtureVolumeBar(
                security_id=SECURITY,
                trading_date=day,
                close_usd=Decimal("100"),
                close_raw_usd=Decimal("100"),
                volume_shares=FLAT_VOLUME,
                knowledge_time=midnight_utc(day) + dt.timedelta(days=1),
            )
            for day in days
        )
    )
    assert AMIHUD_LOOKBACK_DAYS < 7 * AMIHUD_BARS  # the fixture really does overflow the bound
    values = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert math.isnan(values[0])


@pytest.mark.parametrize("untraded", [0, -5_000], ids=["zero-volume", "negative-volume"])
async def test_an_untraded_or_corrupt_day_voids_the_estimate_rather_than_being_dropped(
    untraded: int,
) -> None:
    """Dropping the day would select on exactly the least liquid observations.

    Silently skipping an untraded day biases the security's score toward looking
    more liquid than it was, and nothing downstream could see it happen. The whole
    estimate is voided instead — and the non-vacuity half matters as much: with
    that one day traded, the same series produces a number.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    assert not math.isnan(
        (await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, volumes)))[0]
    )

    broken = list(volumes)
    broken[100] = untraded
    values = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, broken))
    assert math.isnan(values[0])
    assert not math.isinf(values[0])


async def test_a_zero_denominator_never_produces_an_infinity() -> None:
    """``x / 0.0`` is ``inf``, and the transform pipeline rejects infinities outright.

    An infinity escaping this factor would abort a whole cross-section for one
    untraded day at one security, instead of dropping that one name.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    broken = list(volumes)
    broken[200] = 0
    values = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, broken))
    assert not np.isinf(values).any()
    assert math.isnan(values[0])


async def test_a_corrupt_non_positive_close_gives_nan_not_an_infinity() -> None:
    """A zero print makes a log return undefined; NaN says so, ``inf`` would not."""
    closes = alternating_closes()
    closes[0] = 0.0  # the oldest bar, which the window's first return starts from
    volumes = matching_volumes(alternating_closes(), traded_usd=101_000_000.0)
    values = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store_with(closes, volumes))
    assert math.isnan(values[0])
    assert not np.isinf(values).any()


async def test_a_stale_series_is_nan_rather_than_a_stale_liquidity_score() -> None:
    """A name that stopped printing keeps no score.

    The series is complete and every bar is inside the calendar lookback; the only
    thing wrong with it is that the newest print is older than
    :data:`~backend.features.factors._prices.MAX_PRICE_STALENESS`, which is what a
    halt looks like. The gap is kept small on purpose — a large one would push the
    oldest bars out of the lookback and produce NaN for insufficiency instead,
    leaving the staleness rule itself untested — so the same series is also
    computed at a date it *is* fresh for and must produce a number there.
    """
    stale_last_day = COMPUTE_DATE - dt.timedelta(days=14)
    assert stale_last_day.weekday() < 5
    assert COMPUTE_DATE - stale_last_day > MAX_PRICE_STALENESS
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    store = store_with(closes, volumes, last_day=stale_last_day)

    values = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, store)
    assert math.isnan(values[0])

    fresh_enough = stale_last_day + dt.timedelta(days=1)
    revalued = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, store, compute_date=fresh_enough
    )
    assert not math.isnan(revalued[0])


async def test_a_security_with_no_bars_at_all_is_nan_not_missing() -> None:
    """An empty ``price_bar`` is the state today (P3.4 is blocked on B1).

    The second security has no bars, and its value must arrive as NaN in the right
    position rather than shortening the array — a misaligned feature vector
    attaches one company's numbers to another and looks entirely healthy.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    values = await compute(
        AMIHUD_ILLIQUIDITY,
        amihud_illiquidity,
        store_with(closes, volumes),
        securities=(SECURITY, SECURITY + 1),
    )
    assert values.shape == (2,)
    assert not math.isnan(values[0])
    assert math.isnan(values[1])

    empty = await compute(AMIHUD_ILLIQUIDITY, amihud_illiquidity, FixtureVolumeStore())
    assert math.isnan(empty[0])


async def test_values_are_returned_in_the_requested_order() -> None:
    """Positional alignment: reversing the request reverses the values, nothing else."""
    other = SECURITY + 7
    closes = alternating_closes()
    liquid = matching_volumes(closes, traded_usd=101_000_000.0)
    thin = [volume // 4 for volume in liquid]
    store = FixtureVolumeStore(
        (
            *volume_bars(SECURITY, closes, liquid, last_trading_day=LAST_TRADING_DAY),
            *volume_bars(other, closes, thin, last_trading_day=LAST_TRADING_DAY),
        )
    )
    forward = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, store, securities=(SECURITY, other)
    )
    backward = await compute(
        AMIHUD_ILLIQUIDITY, amihud_illiquidity, store, securities=(other, SECURITY)
    )
    assert forward[0] == pytest.approx(backward[1], rel=1e-12)
    assert forward[1] == pytest.approx(backward[0], rel=1e-12)
    assert forward[0] != pytest.approx(forward[1], rel=1e-12)


# ---------------------------------------------------------------------------
# The dollar-volume loader's own contract
# ---------------------------------------------------------------------------


async def test_the_loader_issues_no_knowledge_time_predicate_of_its_own() -> None:
    """The knowledge bound belongs to the as-of session, and is not restated here.

    A second, hand-written ``knowledge_time`` filter would either duplicate
    :func:`backend.db.as_of` (misleading about where I1 is enforced) or contradict
    it (a weaker filter that quietly wins). The emitted SQL must not mention the
    column at all — while it must mention the two columns this loader exists for.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    as_of = AMIHUD_ILLIQUIDITY.knowledge_cutoff(COMPUTE_DATE)
    session = FixtureVolumeSession(store_with(closes, volumes), as_of)
    await load_dollar_volume_bars(
        cast("AsyncSession", session),
        security_ids=(SECURITY,),
        compute_date=COMPUTE_DATE,
        as_of=as_of,
        lookback_days=AMIHUD_LOOKBACK_DAYS,
    )
    assert len(session.statements) == 1
    sql = str(session.statements[0]).lower()
    assert "price_bar" in sql
    assert "close_raw_usd" in sql
    assert "volume_shares" in sql
    assert "knowledge_time" not in sql


async def test_the_loader_returns_an_ascending_series_despite_descending_rows() -> None:
    """The fixture session hands rows back newest-first; the arithmetic is positional."""
    closes = [100.0, 101.0, 102.0, 103.0]
    volumes = [10, 20, 30, 40]
    as_of = AMIHUD_ILLIQUIDITY.knowledge_cutoff(COMPUTE_DATE)
    history = await load_dollar_volume_bars(
        cast("AsyncSession", FixtureVolumeSession(store_with(closes, volumes), as_of)),
        security_ids=(SECURITY,),
        compute_date=COMPUTE_DATE,
        as_of=as_of,
        lookback_days=AMIHUD_LOOKBACK_DAYS,
    )
    series = history[SECURITY]
    assert list(series.prices.trading_dates) == sorted(series.prices.trading_dates)
    assert np.array_equal(series.prices.adjusted_close, np.array(closes))
    assert np.array_equal(
        series.dollar_volume,
        np.array([close * volume for close, volume in zip(closes, volumes, strict=True)]),
    )
    assert series.bar_count == len(closes)


class DuplicatingVolumeSession(FixtureVolumeSession):
    """A session that returns two versions of one bar, as a broken store would.

    The fixture store resolves versions by ``knowledge_time`` and so cannot
    produce a duplicate; this subclass does, to exercise the loader's own
    duplicate check — the net under a store that failed to resolve.
    """

    async def execute(
        self, statement: object, *args: object, **kwargs: object
    ) -> FixtureVolumeResult:
        """Return the same (security, trading day) twice at different volumes."""
        await super().execute(statement, *args, **kwargs)
        day = midnight_utc(LAST_TRADING_DAY)
        return FixtureVolumeResult(
            [
                (SECURITY, day, Decimal("100"), Decimal("100"), 1_000),
                (SECURITY, day, Decimal("100"), Decimal("100"), 2_000),
            ]
        )


async def test_two_visible_versions_of_one_bar_are_refused_rather_than_picked_between() -> None:
    """A duplicate means version resolution did not happen; choosing one would hide that."""
    as_of = AMIHUD_ILLIQUIDITY.knowledge_cutoff(COMPUTE_DATE)
    session = DuplicatingVolumeSession(FixtureVolumeStore(), as_of)
    with pytest.raises(PriceSeriesError, match="two visible price bars"):
        await load_dollar_volume_bars(
            cast("AsyncSession", session),
            security_ids=(SECURITY,),
            compute_date=COMPUTE_DATE,
            as_of=as_of,
            lookback_days=AMIHUD_LOOKBACK_DAYS,
        )


class NaiveInstantSession(FixtureVolumeSession):
    """A session whose bars carry a naive ``valid_from``, as a broken connector would."""

    async def execute(
        self, statement: object, *args: object, **kwargs: object
    ) -> FixtureVolumeResult:
        """Return one bar whose event time has no timezone."""
        await super().execute(statement, *args, **kwargs)
        naive = dt.datetime(2026, 2, 27, 0, 0)  # noqa: DTZ001 — the defect under test
        return FixtureVolumeResult([(SECURITY, naive, Decimal("100"), Decimal("100"), 1_000)])


async def test_a_naive_event_time_is_refused_rather_than_assigned_a_zone() -> None:
    """Guessing a zone would put bars on the wrong side of a date boundary."""
    as_of = AMIHUD_ILLIQUIDITY.knowledge_cutoff(COMPUTE_DATE)
    session = NaiveInstantSession(FixtureVolumeStore(), as_of)
    with pytest.raises(FeatureComputeError, match="naive valid_from"):
        await load_dollar_volume_bars(
            cast("AsyncSession", session),
            security_ids=(SECURITY,),
            compute_date=COMPUTE_DATE,
            as_of=as_of,
            lookback_days=AMIHUD_LOOKBACK_DAYS,
        )


async def test_a_window_with_no_width_is_refused() -> None:
    """A non-positive lookback cannot contain the trading days the factor needs."""
    as_of = AMIHUD_ILLIQUIDITY.knowledge_cutoff(COMPUTE_DATE)
    with pytest.raises(FeatureComputeError, match="lookback_days must be positive"):
        await load_dollar_volume_bars(
            cast("AsyncSession", FixtureVolumeSession(FixtureVolumeStore(), as_of)),
            security_ids=(SECURITY,),
            compute_date=COMPUTE_DATE,
            as_of=as_of,
            lookback_days=0,
        )


async def test_no_securities_requested_means_no_query_and_an_empty_mapping() -> None:
    """An empty universe on a date is a fact, not an error, and costs no I/O."""
    as_of = AMIHUD_ILLIQUIDITY.knowledge_cutoff(COMPUTE_DATE)
    session = FixtureVolumeSession(FixtureVolumeStore(), as_of)
    history = await load_dollar_volume_bars(
        cast("AsyncSession", session),
        security_ids=(),
        compute_date=COMPUTE_DATE,
        as_of=as_of,
        lookback_days=AMIHUD_LOOKBACK_DAYS,
    )
    assert history == {}
    assert session.statements == []


def test_a_series_whose_columns_are_misaligned_is_refused_at_construction() -> None:
    """Pairing each return with another day's volume is the one silent defect here.

    Every value would be plausible, correctly typed and wrong. The container
    refuses rather than trusting its caller, because nothing downstream — not a
    distribution check, not the type system — could detect the swap.
    """
    prices = PriceSeries(SECURITY, (LAST_TRADING_DAY,), np.array([100.0]))
    with pytest.raises(FeatureComputeError, match="paired positionally"):
        DollarVolumeSeries(prices=prices, dollar_volume=np.array([1.0, 2.0]))
    aligned = DollarVolumeSeries(prices=prices, dollar_volume=np.array([1.0]))
    assert aligned.bar_count == 1


def test_the_declared_window_constants_are_the_ones_the_arithmetic_uses() -> None:
    """252 observations need 253 closes, and the calendar bound must admit them."""
    assert AMIHUD_OBSERVATIONS == 252
    assert AMIHUD_LOOKBACK_DAYS == 400
    # ~367 calendar days at 252 trading days a year, plus slack for holidays.
    assert AMIHUD_LOOKBACK_DAYS > AMIHUD_OBSERVATIONS * 7 / 5


# ---------------------------------------------------------------------------
# Size: the refusal, and the shortcut it does not take
# ---------------------------------------------------------------------------


class NoSession:
    """A stand-in that fails loudly if ``size`` tries to read anything.

    The refusal must happen before any I/O. If the computation reached for the
    database, this object's missing ``execute`` would raise ``AttributeError``
    rather than the named error, and the assertion on the exception type would
    fail.
    """


def size_request(securities: tuple[int, ...] = (1, 2, 3)) -> FeatureComputeRequest:
    """Build a request for ``size`` at its declared cutoff."""
    return FeatureComputeRequest(
        feature=SIZE.name,
        compute_date=COMPUTE_DATE,
        as_of=SIZE.knowledge_cutoff(COMPUTE_DATE),
        security_ids=securities,
    )


async def test_size_raises_instead_of_returning_a_plausible_market_capitalization() -> None:
    """The one behaviour I3 permits for a feature whose numerator has no source."""
    with pytest.raises(FundamentalsSourceUnavailableError) as raised:
        await size(cast("AsyncSession", NoSession()), size_request())
    assert raised.value.feature == "size"
    assert raised.value.table == FUNDAMENTALS_TABLE
    assert raised.value.blocker == FUNDAMENTALS_BLOCKER
    assert raised.value.task == FUNDAMENTALS_CONNECTOR_TASK


async def test_the_size_refusal_names_the_table_the_blocker_and_the_connector_task() -> None:
    """An operator reading the message must know what to unblock and where to look."""
    with pytest.raises(FundamentalsSourceUnavailableError) as raised:
        await size(cast("AsyncSession", NoSession()), size_request())
    message = str(raised.value)
    assert FUNDAMENTALS_TABLE in message
    assert FUNDAMENTALS_BLOCKER in message
    assert FUNDAMENTALS_CONNECTOR_TASK in message
    assert "BLOCKERS.md" in message


async def test_the_size_refusal_is_part_of_the_feature_error_taxonomy() -> None:
    """A caller catching ``FeatureComputeError`` catches this without knowing about B1."""
    with pytest.raises(FeatureComputeError):
        await size(cast("AsyncSession", NoSession()), size_request())
    assert issubclass(FundamentalsSourceUnavailableError, FeatureError)


async def test_size_refuses_for_an_empty_universe_too() -> None:
    """Returning an empty array for zero securities would be a well-typed lie.

    It says "computed, nothing to report" for a feature that computed nothing.
    """
    with pytest.raises(FundamentalsSourceUnavailableError):
        await size(cast("AsyncSession", NoSession()), size_request(securities=()))


async def test_a_full_year_of_prices_does_not_tempt_size_into_answering() -> None:
    """The shortcut, refused against a store that would make it easy to take.

    ``price_bar`` exists and here it is full. A "size" built from prices alone
    would return a plausible, correctly-shaped column that ranked on share price
    rather than on company value — a measurement of share-splitting policy — and
    nothing downstream could tell. The refusal must be identical to the one issued
    against an empty store.
    """
    closes = alternating_closes()
    volumes = matching_volumes(closes, traded_usd=101_000_000.0)
    populated = store_with(closes, volumes)
    with pytest.raises(FundamentalsSourceUnavailableError) as raised:
        await compute(SIZE, size, populated)
    assert raised.value.feature == "size"

    with pytest.raises(FundamentalsSourceUnavailableError):
        await compute(SIZE, size, FixtureVolumeStore())


def test_the_price_leg_alone_is_declared_insufficient() -> None:
    """Both source tables are named, and the fundamentals one is the missing half."""
    assert PRICE_SOURCE_TABLE in SIZE.source_tables
    assert FUNDAMENTALS_TABLE in SIZE.source_tables
    assert SIZE.source_tables != frozenset({PRICE_SOURCE_TABLE})


def test_the_compute_instant_convention_is_unchanged_by_these_two_factors() -> None:
    """Sanity anchor: the cutoffs above are midnight opening the compute date, less the lag."""
    assert compute_instant(COMPUTE_DATE).isoformat() == "2026-03-02T00:00:00+00:00"


# ---------------------------------------------------------------------------
# The catalog, now that P5.3's library is complete
# ---------------------------------------------------------------------------

PLANNED_FACTORS = [
    "accruals",
    "amihud_illiquidity",
    "asset_growth",
    "book_to_price",
    "earnings_yield",
    "gross_profitability",
    "low_volatility",
    "momentum_12_1",
    "roic",
    "short_interest",
    "short_term_reversal",
    "size",
]
"""Every factor the library registers, written out by hand.

``PLAN.md`` P5.3 and directive §5 Phase 5 name eleven; the twelfth is
``short_term_reversal``, which is the complement of the month ``momentum_12_1``
skips and was added alongside it (:mod:`backend.features.factors.momentum`
explains why the two belong together). The list is spelled out here rather than
derived from the package, so that registering a factor nobody decided to add is a
test failure rather than an import side effect.
"""


def test_the_catalog_holds_exactly_the_planned_factors() -> None:
    """P5.3's baseline library is complete: no more and no fewer."""
    assert [spec.name for spec in BASELINE_FACTORS] == PLANNED_FACTORS


def test_the_catalog_tuple_is_sorted_by_name_like_the_registry() -> None:
    """``BASELINE_FACTORS`` enumerates the way ``FeatureRegistry.specs()`` does.

    A catalog that reads differently from the registry it describes is a catalog
    that will drift from it.
    """
    assert [spec.name for spec in BASELINE_FACTORS] == sorted(PLANNED_FACTORS)


def test_the_twelve_factors_fit_inside_the_thirty_feature_cap() -> None:
    """Directive §5: the cap includes Phase 7's LLM features, so headroom is the finding."""
    registry = default_registry()
    assert len(BASELINE_FACTORS) == 12
    assert set(PLANNED_FACTORS) <= set(registry.names())
    assert len(registry) <= MAX_FEATURES
    assert registry.remaining_capacity() == MAX_FEATURES - len(registry)
    # Every baseline factor is now registered, so what is left is Phase 7's
    # budget. Eighteen slots for LLM-derived features against the same cap.
    assert registry.remaining_capacity() >= 18
