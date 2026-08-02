"""Point-in-time universe construction: the reads, the assembly, the screen (P4.1).

:func:`build_universe` is the Phase 4 entry point. It turns *(rebalance date,
criteria)* into a :class:`~backend.universe.snapshot.UniverseSnapshot` recording
what every candidate was screened on and what each screen decided — using only
what was knowable at the ``as_of`` instant bound on the session it is handed.

--------------------------------------------------------------------------
The one thing this module exists to get right
--------------------------------------------------------------------------

A universe reconstructed for a past date must contain the names that **later**
stopped existing. If it does not, every backtest run on it is survivorship
biased: the sample is silently restricted to the companies that survived, which
is a condition unknowable at the time and correlated with returns in the most
flattering possible direction. Nothing downstream reports it — the Sharpe ratio
simply goes up.

Two mechanisms together prevent it, and both are needed:

1. **Every read goes through the ``as_of()``-scoped session the caller supplies**
   (invariant I1). This module never opens a session, never imports the private
   engine, and refuses a session with no as-of bound
   (:class:`~backend.universe.errors.UniverseSessionError`). The as-of layer
   then guarantees no row with ``knowledge_time > as_of`` can be returned, so a
   fact learned after the rebalance date cannot enter the screen.
2. **Listing status is evaluated against the rebalance date, not against
   "now"** (:meth:`ListedSecurity.is_listed_on`). Point 1 alone is not enough:
   a research read is normally made at *today's* as-of over a *past* rebalance
   date, so the identity version the store returns is the current one — the one
   that says the name was delisted in 2020. Asking "is ``delisted_on`` NULL" of
   that row drops the name from the 2019 universe it belonged in. Asking
   "was it listed on 2019-03-29" keeps it. That comparison is the whole of gate
   G4 and it lives in one small pure function so it can be tested directly.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**Dollar volume is** ``close_raw_usd * volume_shares`` **— both unadjusted.**
This is the units bug §8 warns about and it is silent. ``volume_shares`` is the
share count actually printed on the trade date and ``close_raw_usd`` is the
price actually printed, so their product is the dollars that actually changed
hands. Multiplying the *adjusted* close by the *unadjusted* volume would scale
every pre-split day's liquidity by the cumulative split factor — a 4-for-1 split
would make the year before it look four times as liquid, and an ADV floor
calibrated on that is not the floor anyone thinks it is. (Adjusted price times
adjusted volume would also be correct, the factor cancelling; adjusted volume is
not stored, so the raw pair is the pair to use.)

**ADV is the median over the last** ``criteria.adv_lookback_days`` **bars** in
the window, in USD/day. Median, not mean: one earnings-day volume spike should
not qualify an otherwise untradeable name.

**The ADV window is a calendar span sized to contain the requested number of
trading days** (:func:`adv_window_calendar_days`), because trading days are not
knowable without an exchange calendar and this package has no business owning
one. If **fewer** than ``adv_lookback_days`` bars are found in that span, the
ADV is ``None`` and the name fails the ADV screen. That is the conservative
direction and it is also the *correct* one: the span is sized generously enough
(1.5x plus ten days) that a continuously trading name always has enough bars, so
a name short of bars in it is a name that did not trade — precisely what an ADV
floor is for.

**The price screened is the unadjusted close of the most recent bar within that
same window.** Bounding it by the window rather than searching backwards without
limit means a name whose last print is months old has no price at all and fails,
rather than being screened on a stale quote that would pass.

**The identity version used is the one in force at the start of the rebalance
date** (``valid_from <= D 00:00Z < valid_to``). Daily bars are the finest
resolution in this system, so an intraday identity change is below the
resolution of everything that consumes the universe; picking the start of the
day states which side of that ambiguity we are on rather than leaving it to
whichever row the database returned first.

**Rebalance dates are exchange-calendar dates with no time component.** A
``datetime`` is refused rather than truncated (it is a ``date`` subclass, so
truncation would be silent).

--------------------------------------------------------------------------
Two of the five screens cannot run today, and the build refuses
--------------------------------------------------------------------------

``market_cap`` needs shares outstanding, and ``borrow`` needs a locate feed.
**Neither source exists in this repository** — there is no fundamentals table
and no borrow table, blocked on ``BLOCKERS.md`` B1 and B2 respectively. So
:func:`require_available_inputs` refuses the whole build, before any read, any
time the criteria request one of them. Since
:class:`~backend.universe.criteria.UniverseCriteria` makes a strictly positive
market-cap floor mandatory, that is **every** build today.

This is deliberate and it is the point of the package. The alternatives are all
worse: fabricating a market cap from a price and an invented share count is I3
fabrication; skipping the screen produces a universe that looks correct, is a
different universe than the criteria describe, and silently admits microcaps the
strategy could never have traded; and excluding every name produces an empty
universe that looks like a bug in the criteria rather than a missing feed.
Refusing names the source and the blocker, and cannot be mistaken for a result.

Everything below the refusal — the reads, the listing predicate, the ADV
arithmetic, the screen, the snapshot — is complete and tested, so resolving B1
and B2 is a matter of adding the inputs to :func:`assemble_candidates`, not of
writing Phase 4 again.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa

from backend.db.asof import AS_OF_INFO_KEY
from backend.db.models import PriceBar, SecurityMaster
from backend.universe.criteria import (
    FILTER_ORDER,
    UniverseCandidate,
    UniverseCriteria,
    evaluate_candidate,
)
from backend.universe.errors import (
    UniverseConsistencyError,
    UniverseInputUnavailableError,
    UniverseSessionError,
)
from backend.universe.snapshot import UniverseSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "ADV_WINDOW_PADDING_DAYS",
    "FILTERS_WITH_AVAILABLE_INPUTS",
    "MEDIAN_PRECISION_DIGITS",
    "UNAVAILABLE_FILTER_INPUTS",
    "DailyBar",
    "ListedSecurity",
    "adv_window_calendar_days",
    "adv_window_start",
    "assemble_candidates",
    "bound_as_of",
    "build_universe",
    "median_dollar_volume_usd",
    "read_listings",
    "read_price_window",
    "require_available_inputs",
    "screen_candidates",
]

_ADV_WINDOW_CALENDAR_NUMERATOR: Final = 3
_ADV_WINDOW_CALENDAR_DENOMINATOR: Final = 2
"""Calendar days per trading day, as an exact ratio (1.5).

The true long-run ratio for a US exchange is about 365.25/252 ≈ 1.45. Rounding
*up* to 1.5 is the safe direction: an over-long window costs a few extra rows,
while an under-long one silently reports "not enough bars" for names that traded
every day, which the ADV screen would then exclude as illiquid.
"""

ADV_WINDOW_PADDING_DAYS: Final = 10
"""Extra calendar days added to the ADV window on top of the 1.5x ratio.

Absorbs holiday clustering — the ratio above is a long-run average, and a window
landing on Thanksgiving week or the days around New Year sees a locally lower
trading-day density than the average predicts.
"""

MEDIAN_PRECISION_DIGITS: Final = 50
"""Decimal working precision for the even-count median (significant digits).

The median of an even number of dollar volumes is ``(a + b) / 2``. Dollar
volumes reach ~1e13 with six decimal places — around twenty significant digits —
so the default 28-digit context is *probably* enough and "probably" is not a
property worth relying on for a value that decides membership. Fifty digits
makes the halving exact for any value the price and volume columns can hold, so
the screen's boundary cases do not move with the size of the numbers.
"""

UNAVAILABLE_FILTER_INPUTS: Final[dict[str, tuple[str, str]]] = {
    "market_cap": (
        (
            "a shares-outstanding history in the securities master (no fundamentals table "
            "exists in this repository; the point-in-time feed is still being procured)"
        ),
        "B1",
    ),
    "borrow": (
        (
            "a borrow-availability feed (deliberately not bought from a data vendor — the "
            "locate data is to be taken from the broker in Phase 11, which needs the paper "
            "account)"
        ),
        "B2",
    ),
}
"""Screens whose input data source does not exist here, with the blocker that owns it.

Keyed by filter name from :data:`~backend.universe.criteria.FILTER_ORDER`. A
filter absent from this mapping is one this module can actually evaluate today:
``exchange`` and ``price`` come from ``security_master`` and ``price_bar``, and
``adv`` from ``price_bar``. When a source lands, the entry is deleted and the
input is assembled in :func:`assemble_candidates` — nothing else moves.
"""

FILTERS_WITH_AVAILABLE_INPUTS: Final = tuple(
    name for name in FILTER_ORDER if name not in UNAVAILABLE_FILTER_INPUTS
)
"""Screens this module can actually evaluate today: ``exchange``, ``price``, ``adv``.

Derived from :data:`UNAVAILABLE_FILTER_INPUTS` rather than restated, so a source
landing for ``market_cap`` or ``borrow`` moves this tuple automatically instead
of leaving a stale claim behind for a reader to trust.
"""


@dataclass(frozen=True, slots=True)
class ListedSecurity:
    """One security's identity as it stood at a rebalance date.

    The read model of :class:`backend.db.models.SecurityMaster`, holding only
    the columns the screen consults. A plain value object rather than the ORM
    row so the listing arithmetic can be exercised on securities written out by
    hand, without a database.

    Attributes:
        security_id: identity-anchor key (dimensionless).
        exchange: MIC code of the primary listing at this version, verbatim from
            the master.
        first_listed_on: exchange-calendar date of first listing, or ``None``
            when the master does not record one.
        delisted_on: exchange-calendar date of delisting, or ``None`` while
            listed. **The last day the name trades**, not the day after — see
            :meth:`is_listed_on`.
    """

    security_id: int
    exchange: str
    first_listed_on: dt.date | None
    delisted_on: dt.date | None

    def is_listed_on(self, on_date: dt.date) -> bool:
        """Return whether this security was listed and tradeable on ``on_date``.

        **This is gate G4 in one comparison.** The identity version handed to
        this method is the one the store returns at the session's as-of, which
        for a research read over a past rebalance date is normally the *current*
        one — the version that already records the delisting. Testing
        ``delisted_on is None`` against that row would answer "is this name
        listed today", drop every name that has since disappeared, and produce a
        survivorship-biased universe that looks entirely ordinary. Comparing
        against ``on_date`` answers the question actually asked.

        Boundaries, both chosen towards inclusion:

        - ``delisted_on == on_date`` is **listed**. A delisting date is the last
          day the name trades, so the name was tradeable that day; excluding it
          would be a small survivorship bias of exactly the kind this method
          exists to prevent.
        - ``first_listed_on == on_date`` is **listed** — an IPO trades on its
          first day.
        - ``first_listed_on is None`` is treated as listed. The version is in
          force at this date, so the entity existed; an unrecorded listing date
          is missing metadata, not evidence of absence. The price and ADV
          screens still require real bars, so a name that never traded cannot
          reach the universe through this branch.

        Args:
            on_date: the rebalance date, an exchange-calendar date.

        Returns:
            ``True`` when the name was listed on that date.
        """
        if self.first_listed_on is not None and on_date < self.first_listed_on:
            return False
        return not (self.delisted_on is not None and on_date > self.delisted_on)


@dataclass(frozen=True, slots=True)
class DailyBar:
    """One security's daily bar, reduced to the two columns the screen uses.

    The read model of :class:`backend.db.models.PriceBar`. Both columns are
    **unadjusted**, which is what makes their product the dollars that actually
    changed hands — see the module docstring on why the adjusted close must not
    be paired with the unadjusted volume.

    Attributes:
        security_id: identity-anchor key (dimensionless).
        trade_date: the exchange-calendar day the bar describes.
        close_raw_usd: unadjusted closing price, **USD per share**.
        volume_shares: unadjusted volume, **shares**.
    """

    security_id: int
    trade_date: dt.date
    close_raw_usd: Decimal
    volume_shares: int

    @property
    def dollar_volume_usd(self) -> Decimal:
        """Dollars traded on this bar: ``close_raw_usd * volume_shares`` (**USD**)."""
        return self.close_raw_usd * self.volume_shares


def bound_as_of(session: AsyncSession) -> dt.datetime:
    """Return the as-of instant bound on ``session``, refusing an unscoped one.

    Invariant I1 requires every bitemporal read to be pinned to a knowledge
    instant, and :mod:`backend.db.asof` enforces that at the ORM and SQL
    boundaries — so an unscoped session would fail there anyway. Checking here
    instead means the failure names the universe build rather than a statement
    rewrite, and it happens before any work. The instant is also what the
    snapshot records as its data version (I2), so the builder needs it in hand
    regardless.

    Args:
        session: the session handed to :func:`build_universe`.

    Returns:
        The bound as-of instant, timezone-aware UTC.

    Raises:
        UniverseSessionError: if the session carries no as-of bound (it was not
            produced by ``as_of()`` — an ingestion writer session, or a
            hand-built one), or if the bound value is not a timezone-aware UTC
            datetime.
    """
    bound: object = session.info.get(AS_OF_INFO_KEY)
    if bound is None:
        msg = (
            "the session handed to the universe builder is not scoped by as_of(). Every "
            "input to a point-in-time universe is a bitemporal read and must be pinned to "
            "a knowledge instant (I1); an unscoped session would read the store as it "
            "stands now and reconstruct a universe out of facts that did not exist at the "
            "rebalance date. Open the session with `async with as_of(instant) as session:` "
            "and hand that one in"
        )
        raise UniverseSessionError(msg)
    if not isinstance(bound, dt.datetime):
        msg = f"session as-of bound is {bound!r}, which is not a datetime"
        raise UniverseSessionError(msg)
    if bound.tzinfo is None or bound.utcoffset() != dt.timedelta(0):
        msg = f"session as-of bound {bound!r} must be a timezone-aware UTC instant"
        raise UniverseSessionError(msg)
    return bound


def require_available_inputs(criteria: UniverseCriteria) -> None:
    """Refuse the build if any requested screen has no data source in this system.

    Checked in :data:`~backend.universe.criteria.FILTER_ORDER` so the error names
    the earliest unavailable screen, and called by :func:`build_universe`
    **before any read** — the caller is never left wondering whether a partial
    universe was computed.

    Today this refuses every build, because a market-cap floor is mandatory and
    nothing in this repository supplies shares outstanding. That is the honest
    state of the system, not a defect in this function; see the module docstring
    for why each alternative is worse.

    Args:
        criteria: the screens requested. Only the ones
            :meth:`~backend.universe.criteria.UniverseCriteria.applied_filters`
            reports are checked — a borrow screen that is switched off needs no
            borrow feed.

    Raises:
        UniverseInputUnavailableError: naming the filter, the missing source,
            and the ``BLOCKERS.md`` identifier tracking it.
    """
    for filter_name in criteria.applied_filters():
        unavailable = UNAVAILABLE_FILTER_INPUTS.get(filter_name)
        if unavailable is not None:
            source, blocker = unavailable
            raise UniverseInputUnavailableError(
                filter_name=filter_name, source=source, blocker=blocker
            )


def adv_window_calendar_days(adv_lookback_days: int) -> int:
    """Return the calendar span that reliably contains ``adv_lookback_days`` trading days.

    ``ceil(adv_lookback_days * 3 / 2) + ADV_WINDOW_PADDING_DAYS``. The ratio and
    the padding are justified on their own constants; the short version is that
    both round in the direction of reading a few extra rows rather than of
    silently reporting an unmeasurable ADV for a name that traded every day.

    Args:
        adv_lookback_days: the ADV window in **trading days**, as the criteria
            state it.

    Returns:
        The window in **calendar days**, inclusive of the rebalance date itself.
    """
    numerator = adv_lookback_days * _ADV_WINDOW_CALENDAR_NUMERATOR
    scaled = -(-numerator // _ADV_WINDOW_CALENDAR_DENOMINATOR)  # ceiling division
    return scaled + ADV_WINDOW_PADDING_DAYS


def adv_window_start(rebalance_date: dt.date, adv_lookback_days: int) -> dt.date:
    """Return the first calendar date of the ADV window ending at ``rebalance_date``.

    Args:
        rebalance_date: the last date of the window, inclusive.
        adv_lookback_days: the ADV window in trading days.

    Returns:
        The first date of the window, inclusive, so the window spans
        :func:`adv_window_calendar_days` calendar dates in total.
    """
    return rebalance_date - dt.timedelta(days=adv_window_calendar_days(adv_lookback_days) - 1)


def median_dollar_volume_usd(dollar_volumes: Sequence[Decimal]) -> Decimal:
    """Return the median of a sequence of daily dollar volumes.

    Exact decimal arithmetic throughout: the comparison against the ADV floor is
    ``>=``, so a name sitting on the floor is included, and binary floating point
    would turn that boundary into a coin flip that differs between runs. For an
    even count the median is ``(a + b) / 2`` computed at
    :data:`MEDIAN_PRECISION_DIGITS` significant digits, which is exact for any
    value the price and volume columns can hold.

    Args:
        dollar_volumes: one dollar volume per bar, in **USD**, in any order.

    Returns:
        The median, in **USD per day**.

    Raises:
        UniverseConsistencyError: if the sequence is empty. The median of no
            observations is undefined, and returning zero would make an
            unmeasured name look like a name that traded nothing.
    """
    if not dollar_volumes:
        msg = (
            "cannot take the median of zero dollar volumes; a name with no bars has an "
            "unmeasurable ADV, which the caller represents as None rather than as zero"
        )
        raise UniverseConsistencyError(msg)
    ordered = sorted(dollar_volumes)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    with localcontext() as context:
        context.prec = MEDIAN_PRECISION_DIGITS
        return +((ordered[middle - 1] + ordered[middle]) / 2)


def _require_calendar_date(rebalance_date: dt.date) -> None:
    """Refuse a ``datetime`` passed where an exchange-calendar date is required.

    Args:
        rebalance_date: the value to check.

    Raises:
        UniverseConsistencyError: if the value carries a time component.
            ``datetime`` subclasses ``date``, so silently truncating it would
            make ``2020-03-31T16:00`` and ``2020-03-31T00:00`` the same request
            while looking like two different ones.
    """
    if isinstance(rebalance_date, dt.datetime):
        msg = (
            f"rebalance_date={rebalance_date!r} is a datetime; a rebalance date is an "
            f"exchange-calendar date with no time component. Pass `.date()` explicitly so "
            f"the truncation is visible in the caller rather than assumed here"
        )
        raise UniverseConsistencyError(msg)


def _start_of_day(day: dt.date) -> dt.datetime:
    """Return midnight UTC at the start of ``day`` — the event-time instant D-011 uses."""
    return dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC)


async def read_listings(
    session: AsyncSession, *, rebalance_date: dt.date
) -> tuple[ListedSecurity, ...]:
    """Read every identity version in force at ``rebalance_date``, through the as-of session.

    The event-time predicate selects the version covering the **start** of the
    rebalance date (module docstring); the knowledge-time predicate is not
    written here at all — the as-of layer applies it to every bitemporal SELECT
    on this session, which is precisely why the builder must not open its own.

    This returns every listed *and* delisted-later identity the store knows
    about at that instant. Filtering to the ones actually tradeable on the date
    is :meth:`ListedSecurity.is_listed_on`'s job, deliberately kept in Python
    where it is directly testable rather than buried in a WHERE clause.

    Args:
        session: an ``as_of()``-scoped session.
        rebalance_date: the date to reconstruct identities at.

    Returns:
        One :class:`ListedSecurity` per security, ascending by ``security_id``.

    Raises:
        UniverseConsistencyError: if two identity versions of one security are
            in force at the same instant. That is a defect in the store's
            event-time intervals, and picking one of the two would make the
            universe depend on row order.
    """
    instant = _start_of_day(rebalance_date)
    statement = (
        sa.select(
            SecurityMaster.security_id,
            SecurityMaster.exchange,
            SecurityMaster.first_listed_on,
            SecurityMaster.delisted_on,
        )
        .where(SecurityMaster.valid_from <= instant, SecurityMaster.valid_to > instant)
        .order_by(SecurityMaster.security_id)
    )
    rows = (await session.execute(statement)).all()
    listings = tuple(
        ListedSecurity(
            security_id=int(row[0]),
            exchange=str(row[1]),
            first_listed_on=row[2],
            delisted_on=row[3],
        )
        for row in rows
    )
    seen: set[int] = set()
    duplicates: set[int] = set()
    for item in listings:
        if item.security_id in seen:
            duplicates.add(item.security_id)
        seen.add(item.security_id)
    if duplicates:
        first = min(duplicates)
        msg = (
            f"{len(duplicates)} security/securities have more than one identity version in "
            f"force at {rebalance_date.isoformat()} (first: security_id={first}). "
            f"Versions of one logical key must have disjoint [valid_from, valid_to) "
            f"intervals (D-011); choosing between them here would make the universe depend "
            f"on row order"
        )
        raise UniverseConsistencyError(msg)
    return listings


async def read_price_window(
    session: AsyncSession, *, first_date: dt.date, last_date: dt.date
) -> tuple[DailyBar, ...]:
    """Read every daily bar in a calendar window, through the as-of session.

    No ``security_id`` filter: the candidate set is essentially every security in
    the master, so an ``IN`` list of that size buys nothing and makes this read
    depend on the listing read. Bars belonging to securities that are not
    candidates are simply not looked up during assembly.

    Args:
        session: an ``as_of()``-scoped session.
        first_date: first calendar date of the window, inclusive.
        last_date: last calendar date of the window, inclusive — normally the
            rebalance date.

    Returns:
        Bars ascending by ``(security_id, trade_date)``.

    Raises:
        UniverseConsistencyError: if a bar's event-time interval does not start
            at midnight UTC. D-011 encodes trading day *D* as
            ``[D 00:00Z, D+1 00:00Z)``; a bar starting anywhere else is not a
            daily bar, and truncating its ``valid_from`` to a date would fold
            two bars onto one day and corrupt the median silently.
    """
    statement = (
        sa.select(
            PriceBar.security_id,
            PriceBar.valid_from,
            PriceBar.close_raw_usd,
            PriceBar.volume_shares,
        )
        .where(
            PriceBar.valid_from >= _start_of_day(first_date),
            PriceBar.valid_from <= _start_of_day(last_date),
        )
        .order_by(PriceBar.security_id, PriceBar.valid_from)
    )
    rows = (await session.execute(statement)).all()
    bars: list[DailyBar] = []
    for row in rows:
        valid_from: dt.datetime = row[1]
        if valid_from.timetz() != dt.time(tzinfo=dt.UTC):
            msg = (
                f"price_bar row for security_id={row[0]} has valid_from={valid_from!r}, "
                f"which is not midnight UTC. D-011 encodes trading day D as "
                f"[D 00:00Z, D+1 00:00Z); truncating this to a date would fold it onto a "
                f"neighbouring day's bar"
            )
            raise UniverseConsistencyError(msg)
        bars.append(
            DailyBar(
                security_id=int(row[0]),
                trade_date=valid_from.date(),
                close_raw_usd=Decimal(row[2]),
                volume_shares=int(row[3]),
            )
        )
    return tuple(bars)


def _bars_by_security(bars: Iterable[DailyBar]) -> dict[int, list[DailyBar]]:
    """Group bars by security, ascending by trade date, refusing a duplicated day.

    Args:
        bars: the window's bars, in any order.

    Returns:
        Mapping of ``security_id`` to that security's bars, ascending by date.

    Raises:
        UniverseConsistencyError: if one security has two bars for one trading
            day. The as-of read returns a single winning version per
            ``(security_id, valid_from)``, so a duplicate means the read was not
            versioned — and a duplicated day would quietly double that day's
            weight in the median.
    """
    grouped: dict[int, list[DailyBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.security_id].append(bar)
    for security_id, security_bars in grouped.items():
        security_bars.sort(key=lambda bar: bar.trade_date)
        dates = [bar.trade_date for bar in security_bars]
        if len(set(dates)) != len(dates):
            msg = (
                f"security_id={security_id} has more than one bar for the same trading day "
                f"in the ADV window. An as-of read returns one winning version per "
                f"(security_id, valid_from), so this is an unversioned read rather than a "
                f"data quirk, and it would double-count a day in the median"
            )
            raise UniverseConsistencyError(msg)
    return dict(grouped)


def assemble_candidates(
    listings: Iterable[ListedSecurity],
    bars: Iterable[DailyBar],
    *,
    rebalance_date: dt.date,
    criteria: UniverseCriteria,
) -> tuple[UniverseCandidate, ...]:
    """Turn point-in-time reads into one screening input record per candidate.

    Pure: no I/O, no clock, no session. Candidates are the securities
    :meth:`ListedSecurity.is_listed_on` admits at ``rebalance_date`` — including
    the ones that were delisted *later*, which is the whole of gate G4.

    Per candidate, from the bars in the window:

    - ``price_usd`` — ``close_raw_usd`` of the most recent bar in the window, or
      ``None`` when the window holds no bar for the name;
    - ``adv_usd`` — the median dollar volume over the **last**
      ``criteria.adv_lookback_days`` bars in the window, or ``None`` when the
      window holds fewer bars than that.

    ``market_cap_usd`` and ``borrow_available`` are left ``None`` because no
    source for them exists in this repository. They are **never reached in
    practice**: :func:`require_available_inputs` refuses any build whose criteria
    apply those screens, which today is every build. That refusal is what keeps
    these ``None``s from turning into a universe where every name silently fails
    the market-cap screen — an empty universe that reads as a criteria mistake
    rather than as a missing feed. When B1 and B2 resolve, the two values are
    assembled here from their sources and the entries are removed from
    :data:`UNAVAILABLE_FILTER_INPUTS`.

    Args:
        listings: identity versions in force at the rebalance date.
        bars: every bar in the ADV window, for any security.
        rebalance_date: the date being screened.
        criteria: the screens requested — only ``adv_lookback_days`` is read
            here; the thresholds are applied by :func:`screen_candidates`.

    Returns:
        One candidate per listed security, ascending by ``security_id``.

    Raises:
        UniverseConsistencyError: propagated from :func:`_bars_by_security` if
            the bars are not a versioned read, or from
            :class:`~backend.universe.criteria.UniverseCandidate` if a value is
            not a finite non-negative amount.
    """
    _require_calendar_date(rebalance_date)
    grouped = _bars_by_security(bars)
    candidates: list[UniverseCandidate] = []
    for listing in sorted(listings, key=lambda item: item.security_id):
        if not listing.is_listed_on(rebalance_date):
            continue
        window = grouped.get(listing.security_id, [])
        price_usd = window[-1].close_raw_usd if window else None
        adv_usd: Decimal | None = None
        if len(window) >= criteria.adv_lookback_days:
            lookback = window[-criteria.adv_lookback_days :]
            adv_usd = median_dollar_volume_usd([bar.dollar_volume_usd for bar in lookback])
        candidates.append(
            UniverseCandidate(
                security_id=listing.security_id,
                exchange=listing.exchange,
                price_usd=price_usd,
                adv_usd=adv_usd,
            )
        )
    return tuple(candidates)


def screen_candidates(
    candidates: Iterable[UniverseCandidate],
    *,
    rebalance_date: dt.date,
    criteria: UniverseCriteria,
    as_of: dt.datetime,
) -> UniverseSnapshot:
    """Apply the screens to every candidate and package the result as a snapshot.

    Pure: no I/O. Every candidate gets an outcome, members and exclusions alike,
    because the exclusions are what make §6.3's filter-impact waterfall
    reconstructible from the stored record.

    Args:
        candidates: the assembled screening inputs, one per name considered.
        rebalance_date: the date being screened.
        criteria: the screens to apply.
        as_of: the knowledge instant the inputs were read at, timezone-aware
            UTC. Recorded on the snapshot as its data version (I2).

    Returns:
        The snapshot, with ``outcomes`` covering every candidate and ``members``
        the ascending list of those that failed nothing.

    Raises:
        UniverseConsistencyError: if the same security appears twice among the
            candidates, or if ``as_of`` is not a timezone-aware UTC instant
            (propagated from :class:`~backend.universe.snapshot.UniverseSnapshot`).
    """
    _require_calendar_date(rebalance_date)
    ordered = sorted(candidates, key=lambda candidate: candidate.security_id)
    identifiers = [candidate.security_id for candidate in ordered]
    if len(set(identifiers)) != len(identifiers):
        msg = (
            f"{len(identifiers) - len(set(identifiers))} security/securities appear more "
            f"than once among the candidates for {rebalance_date.isoformat()}. A candidate "
            f"screened twice would be counted twice in every waterfall built from this "
            f"snapshot"
        )
        raise UniverseConsistencyError(msg)
    outcomes = tuple(evaluate_candidate(candidate, criteria) for candidate in ordered)
    return UniverseSnapshot(
        rebalance_date=rebalance_date,
        criteria=criteria,
        criteria_hash=criteria.criteria_hash(),
        as_of=as_of,
        members=tuple(outcome.security_id for outcome in outcomes if outcome.included),
        outcomes=outcomes,
    )


async def build_universe(
    session: AsyncSession,
    *,
    rebalance_date: dt.date,
    criteria: UniverseCriteria,
) -> UniverseSnapshot:
    """Build the point-in-time universe for one rebalance date (P4.1).

    The order of operations is the contract, not an implementation detail:

    1. the session's as-of bound is required (:func:`bound_as_of`) — I1;
    2. every requested screen is required to have a data source
       (:func:`require_available_inputs`) — I3, and **before any read**, so a
       refusal never leaves a partial universe behind;
    3. identities and bars are read through the supplied session and nothing
       else;
    4. candidates are assembled and screened, purely.

    Args:
        session: an ``as_of()``-scoped ``AsyncSession``. This function never
            opens a session of its own: the caller's as-of instant is the whole
            of the universe's point-in-time content, and a session opened here
            would silently read the store as it stands now.
        rebalance_date: the exchange-calendar date to build the universe for. A
            ``datetime`` is refused rather than truncated.
        criteria: the screens to apply.

    Returns:
        The :class:`~backend.universe.snapshot.UniverseSnapshot` for that date —
        members plus the screening outcome of every candidate considered.
        Persisting it is the caller's choice
        (:func:`~backend.universe.snapshot.persist_snapshot`), because the
        transaction boundary belongs to whoever is building the run.

    Raises:
        UniverseSessionError: if the session is not ``as_of()``-scoped.
        UniverseInputUnavailableError: if a requested screen's input source does
            not exist in this system. **Today this is every build**: a
            market-cap floor is mandatory and nothing supplies shares
            outstanding (BLOCKERS.md B1). See the module docstring.
        UniverseConsistencyError: if ``rebalance_date`` carries a time
            component, if it is later than the session's as-of date (the inputs
            for it cannot have been knowable), or if the store returns
            inconsistent rows.
    """
    _require_calendar_date(rebalance_date)
    as_of = bound_as_of(session)
    if rebalance_date > as_of.date():
        msg = (
            f"rebalance_date={rebalance_date.isoformat()} is after the session's as-of date "
            f"{as_of.date().isoformat()}. Nothing about that date was knowable at the as-of "
            f"instant, so the only universe this could produce is an empty one wearing a "
            f"date it was never screened at"
        )
        raise UniverseConsistencyError(msg)
    require_available_inputs(criteria)
    listings = await read_listings(session, rebalance_date=rebalance_date)
    bars = await read_price_window(
        session,
        first_date=adv_window_start(rebalance_date, criteria.adv_lookback_days),
        last_date=rebalance_date,
    )
    candidates = assemble_candidates(
        listings, bars, rebalance_date=rebalance_date, criteria=criteria
    )
    return screen_candidates(
        candidates, rebalance_date=rebalance_date, criteria=criteria, as_of=as_of
    )
