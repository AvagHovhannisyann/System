"""Short interest: a semi-monthly number with a publication lag (P5.3).

The fraction of a company's shares that have been sold short and not yet
covered. High short interest predicts low subsequent returns — Desai, Ramesh,
Thiagarajan & Balachandran (2002), Asquith, Pathak & Ritter (2005), Boehmer,
Jones & Zhang (2008) — and the usual reading is that short sellers are informed:
shorting is costly, capacity-constrained and asymmetrically risky, so a large
short position is a comparatively expensive opinion rather than noise.

**This factor does not compute today, and it is blocked differently from the
others.** The six value/quality/growth factors and ``size`` wait on a connector
that exists as a *task* (P3.5) against a *blocker* the operator has already
decided (B1). Short interest waits on neither: there is no short-interest table,
no Phase 3 connector task for one, and no ``BLOCKERS.md`` entry. The gap is
**unregistered**, which is itself the finding — see
:class:`ShortInterestSourceUnavailableError`, which says so rather than
borrowing B1's name for a feed B1's decision does not cover. (B1 selected
Sharadar SF1/SEP/SFP/ACTIONS; none of the four carries short interest.)

--------------------------------------------------------------------------
Availability lag: 17 days, and this is the whole factor
--------------------------------------------------------------------------

Every other factor in this package is a piece of arithmetic with a temporal
caveat attached. This one is a temporal problem with a piece of arithmetic
attached, because of how the number is produced.

**US short interest is published on a settlement-date basis with a reporting
lag.** Under FINRA Rule 4560 members report their short positions twice a month
— as of the settlement date on the 15th (or the preceding business day) and as
of the settlement date of the last business day of the month — reports are due a
couple of business days later, and the aggregated figures are *disseminated*
around the eighth business day after that reporting settlement date. So every
row carries a prominent date (the settlement date) which is **not** the date the
number became public, and the two are separated by more than a week.

That separation is the entire temporal risk here, and it has a specific failure
mode. Every vendor file of this data is indexed by the settlement date; it is the
obvious join key, the obvious ``valid_from``, and — for a connector author not
paying attention — the obvious ``knowledge_time``. A connector that made that
substitution would make each observation appear knowable **eight business days
before it existed**, twice a month, forever. Nothing about the resulting factor
would look wrong: the values would be real short-interest ratios, of the right
magnitude, correctly signed, with a healthy cross-sectional distribution. Only
the backtest would change, and it would improve. This is D-027's standing lesson
in its most concentrated form, and it is why the declared margin below is sized
to cover the *whole* publication gap rather than a residual uncertainty around
it.

Three components, each in the lookahead direction, each rounded up — the same
shape as D-027's fundamentals margin, with a much larger first term:

**1. Settlement date to dissemination (14 days).** Eight business days, expressed
in the wall-clock arithmetic ``availability_lag`` requires (:mod:`~backend.features.spec`:
"declare 45 days, not 30 sessions"). Eight business days always spans one weekend
and usually two (+4 days), and a two-week span can contain up to two market
holidays (+2), so the calendar-day bound is 14. **The precise dissemination
schedule is a rule detail that must be re-read from FINRA and the exchanges when
the connector is written**; the margin is deliberately sized so that a schedule
a few days longer than the one assumed here still does not produce lookahead.

**2. Dissemination is intraday; the compute instant is midnight (1 day).** A
feature computed for ``D`` reads the store as of ``midnight_utc(D)``, and a file
released during ``D``'s business hours is hours *later* than that instant. One
day rounds the release inside the previous compute date.

**3. Vendor redistribution after the exchange release (2 days, a labelled
guess).** A row appears in whatever feed is eventually bought some time after the
exchange publishes it. Undocumented, and the same component and the same
justification as the third term of
:data:`~backend.features.factors.value.FUNDAMENTAL_AVAILABILITY_LAG`.

Seventeen days total. **The margin is generous and that is close to free here.**
D-027 argues against padding a *daily* factor, where every extra day discards a
bar that genuinely was available. Short interest is observed twice a month, so
its value is a step function that changes about 24 times a year: a margin of a
few extra days changes *which* observation is read on only a handful of rebalance
dates and changes it to the previous one, never to a fabricated one. The
asymmetry that makes D-027 refuse a margin on prices is the asymmetry that makes
a margin cheap here.

**The lag is the maximum of the two regimes this factor touches, not their
sum.** The denominator (shares outstanding) comes from the B1-blocked
fundamentals feed, whose own margin is 7 days. A lag bounds how fresh the
*freshest* input may be, and both legs are read at one cutoff, so 17 days covers
the 7-day requirement outright. Adding them would be arithmetic on a bound rather
than reasoning about one.

**It is a floor on staleness, never a ceiling** (:mod:`backend.features.spec`).
Declaring 17 days does not claim the value is 17 days old — on most rebalance
dates the freshest visible observation refers to a settlement date one to three
weeks earlier, which is the data's nature and not this declaration's doing.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**The value is a dimensionless fraction: shares sold short per share of common
stock outstanding.** ``0.08`` means 8 of every 100 shares outstanding are held
short. Not a count of shares — a raw short-interest count ranks mega-caps above
micro-caps mechanically and would measure company size, which ``size`` already
measures. Not days-to-cover (shares short over average daily volume, in units of
days): that is a squeeze/crowding metric whose denominator is a volume average,
and it is a different factor from the one the return-predictability literature
above measures.

**Shares outstanding rather than free float.** Float is the economically better
denominator — insider and strategic blocks cannot be borrowed — but float is a
vendor *estimate* with no as-reported source and no filing that fixes it, so a
float-normalized factor would carry an unstated modelling choice inside a number
labelled as a measurement. Shares outstanding is as-reported, point-in-time and
already required by ``size``. The cost is a systematically understated ratio for
closely held companies, and it is stated here rather than smoothed over.

**Both legs are read at one cutoff**, as the single declared lag says: the short
position from its own feed, the share count from the fundamentals feed's most
recent knowable fiscal period. Neither may be a later restatement.

**Expected premium sign: NEGATIVE**, and the value is not negated. ``short_interest``
is named after the quantity it measures, so it follows the same rule as
``accruals`` and ``asset_growth`` rather than the rule ``low_volatility``
follows; the convention is stated once in :mod:`backend.features.factors`.

**A non-positive share count is ``NaN``**, as is a missing short-interest
observation or a missing filing. Zero shares short is a real observation and is
reported as ``0.0``. Nothing is carried forward from the previous settlement date
by this factor: the as-of session decides what is visible, and a value that
persists between publications does so because the store still holds it, not
because this code re-published it.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final, NoReturn

from backend.db.base import Base
from backend.features.errors import FeatureComputeError
from backend.features.factors._fundamentals import (
    FUNDAMENTALS_TABLE,
    FundamentalsSourceUnavailableError,
    fundamentals_source_present,
)
from backend.features.registry import feature
from backend.features.spec import FeatureSpec

if TYPE_CHECKING:
    from collections.abc import Container

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray

__all__ = [
    "SHORT_INTEREST",
    "SHORT_INTEREST_AVAILABILITY_LAG",
    "SHORT_INTEREST_SOURCE_OF_RECORD",
    "SHORT_INTEREST_TABLE",
    "ShortInterestComputationNotWrittenError",
    "ShortInterestSourceUnavailableError",
    "require_short_interest_source",
    "short_interest",
    "short_interest_source_present",
]

SHORT_INTEREST_TABLE: Final = "short_interest_report"
"""Physical table a semi-monthly short-interest feed would land in.

**Provisional, and nothing supplies it.** The name follows the store's
convention (singular snake_case, one row per security per reporting settlement
date: ``price_bar``, ``edgar_filing``, ``fundamental_report``) and is declared
once so that :data:`SHORT_INTEREST`'s ``source_tables`` and the schema probe in
:func:`short_interest_source_present` cannot drift apart. If a connector ever
lands under another name, this constant is the only edit — and the probe will
not silently start believing a feed arrived.
"""

SHORT_INTEREST_SOURCE_OF_RECORD: Final = (
    "FINRA Rule 4560 semi-monthly short interest (settlement dates on the 15th and "
    "the last business day of each month), as disseminated by FINRA and the exchanges"
)
"""Where the numbers would have to come from, named so the gap is actionable.

An error that says "no data" is only marginally better than a fabricated number.
This constant is what turns the refusal into something an operator can act on: it
names the source of record, which is what a Phase 3 connector task and a
``BLOCKERS.md`` entry would both have to start from.
"""

SHORT_INTEREST_AVAILABILITY_LAG: Final = dt.timedelta(days=17)
"""Availability-lag margin ``short_interest`` declares (wall clock).

Sized to cover the **whole** settlement-date-to-dissemination gap rather than a
residual uncertainty around it, because the plausible connector defect here is
stamping ``knowledge_time`` from the settlement date printed on every row. Three
components, each rounded up in the lookahead direction: eight business days from
reporting settlement date to dissemination, bounded at 14 calendar days by two
weekends and two possible market holidays; 1 day because dissemination is
intraday while the compute instant is midnight UTC; and 2 days for vendor
redistribution after the exchange release, a labelled guess matching the third
component of :data:`~backend.features.factors.value.FUNDAMENTAL_AVAILABILITY_LAG`.
The module docstring derives each.

Larger than the fundamentals margin, and deliberately so: this factor's
observation is semi-monthly, so days of margin cost a fraction of one observation
rather than a bar of signal. Reducible only on evidence — a real connector with a
documented dissemination-based knowledge-time policy, validated against a
published FINRA settlement/dissemination calendar — with the reduction logged in
``DECISIONS.md``. Never reducible on the grounds that it costs signal.
"""


class ShortInterestSourceUnavailableError(FeatureComputeError):
    """Raised when ``short_interest`` is asked to compute with no feed to read.

    Distinct from
    :class:`~backend.features.factors._fundamentals.FundamentalsSourceUnavailableError`
    because it describes a different kind of gap and needs a different action. That
    error means "the operator has chosen a vendor and the connector is waiting on
    credentials" (blocker B1, connector task P3.5). This one means **nothing is
    waiting on anything**: there is no short-interest table, no Phase 3 connector
    task for one, and no ``BLOCKERS.md`` entry, and B1's decision does not cover
    the feed (it selected Sharadar SF1, SEP, SFP and ACTIONS — none of which
    carries short interest).

    Reusing B1's name here would be the more comfortable option and it would be
    wrong twice over: it would claim a decision covers a feed it does not, and it
    would let an unregistered gap ride along invisibly behind a blocker that is
    already marked DECIDED. An unregistered gap is exactly the thing directive
    §9.8 says goes to a human.

    Attributes:
        feature: name of the feature that could not be computed.
        table: the missing source table.
        source_of_record: where such data would have to come from.
    """

    def __init__(self, feature: str) -> None:
        """Build the error from the feature that was asked for.

        Args:
            feature: name of the feature that could not be computed.
        """
        self.feature = feature
        self.table = SHORT_INTEREST_TABLE
        self.source_of_record = SHORT_INTEREST_SOURCE_OF_RECORD
        super().__init__(
            f"feature {feature!r} reads semi-monthly US short interest, which this "
            f"platform does not ingest: table {SHORT_INTEREST_TABLE!r} is not in the "
            f"schema, no Phase 3 connector task covers it, and no BLOCKERS.md entry "
            f"tracks it. The gap is UNREGISTERED and needs an operator decision on a "
            f"source before it can even be scheduled — B1 does not cover it (that "
            f"decision selected Sharadar SF1/SEP/SFP/ACTIONS, none of which carries "
            f"short interest). The source of record is: {SHORT_INTEREST_SOURCE_OF_RECORD}. "
            f"The declaration is complete and reviewable — in particular its 17-day "
            f"availability lag, which exists because this data is published on a "
            f"settlement-date basis roughly eight business days after that settlement "
            f"date, and a connector that stamped knowledge_time from the settlement "
            f"date would grant eight business days of foresight twice a month. "
            f"Refusing rather than returning a plausible number: a fabricated "
            f"short-interest ratio is indistinguishable from a measured one "
            f"everywhere downstream (I3, directive §9.1-9.2)."
        )


class ShortInterestComputationNotWrittenError(FeatureComputeError):
    """Raised when both feeds exist but this factor's arithmetic does not.

    The state after a short-interest connector *and* P3.5 have landed and before
    this module is finished. Separate from
    :class:`ShortInterestSourceUnavailableError` for the reason
    :mod:`backend.features.factors._fundamentals` gives for its own pair: the two
    name different outstanding work — there, an operator must register and clear a
    gap; here, an engineer must write a query against schemas that now exist. One
    error covering both would keep reporting "no connector exists" long after one
    did, which is how a task goes quietly missing.

    Attributes:
        feature: name of the feature whose computation is unwritten.
        table: the short-interest table, now present.
    """

    def __init__(self, feature: str) -> None:
        """Build the error from the feature that was asked for.

        Args:
            feature: name of the feature whose computation is unwritten.
        """
        self.feature = feature
        self.table = SHORT_INTEREST_TABLE
        super().__init__(
            f"feature {feature!r} has a complete declaration and no computation: "
            f"tables {SHORT_INTEREST_TABLE!r} and {FUNDAMENTALS_TABLE!r} are both "
            f"mapped, so a missing feed no longer explains the gap and the query "
            f"against them must be written (P5.3 completion). Refusing rather than "
            f"returning a plausible number (I3)."
        )


def short_interest_source_present(tables: Container[str] | None = None) -> bool:
    """Return whether a short-interest table is mapped.

    Reads the live SQLAlchemy metadata rather than a hand-maintained flag, so the
    answer changes when a connector actually lands and cannot be left stale by
    someone forgetting to update it.

    Args:
        tables: table names to probe. Defaults to the mapped schema
            (``Base.metadata.tables``). The parameter exists so the "feed has
            arrived" branches of :func:`require_short_interest_source` are
            testable without registering a fake table in process-wide metadata,
            which would corrupt every other test in the run.

    Returns:
        ``True`` when :data:`SHORT_INTEREST_TABLE` is among the names.
    """
    known: Container[str] = Base.metadata.tables if tables is None else tables
    return SHORT_INTEREST_TABLE in known


def require_short_interest_source(
    feature: str, *, tables: Container[str] | None = None
) -> NoReturn:
    """Refuse to compute ``short_interest``, naming which feed is actually missing.

    Always raises. There is no success path, because neither of the two feeds this
    factor needs exists and neither has a substitute; the return type says so, so
    a caller that tried to continue afterwards would not type-check.

    Three outcomes rather than one, ordered most-specific first, because they name
    three different pieces of outstanding work and an operator reading the message
    has to know which one is theirs. The fundamentals branch reuses
    :class:`~backend.features.factors._fundamentals.FundamentalsSourceUnavailableError`
    rather than restating B1 in a second error family — that gap is already
    registered and already has an owner.

    Args:
        feature: name of the feature being computed, for the message.
        tables: table names to probe; defaults to the mapped schema. See
            :func:`short_interest_source_present`.

    Raises:
        ShortInterestSourceUnavailableError: if no short-interest table is mapped
            — today's state, and an unregistered gap.
        FundamentalsSourceUnavailableError: if a short-interest feed has landed
            but the fundamentals feed supplying the share-count denominator has
            not. Blocked on B1, which is a registered blocker with a named task.
        ShortInterestComputationNotWrittenError: if both are mapped — the feeds
            arrived and the arithmetic is the missing piece.
    """
    if not short_interest_source_present(tables):
        raise ShortInterestSourceUnavailableError(feature)
    if not fundamentals_source_present(tables):
        raise FundamentalsSourceUnavailableError(feature)
    raise ShortInterestComputationNotWrittenError(feature)


SHORT_INTEREST = FeatureSpec(
    name="short_interest",
    definition=(
        "Short interest as a fraction of shares outstanding (Desai et al. 2002; "
        "Asquith, Pathak & Ritter 2005; Boehmer, Jones & Zhang 2008): shares sold "
        "short and not yet covered, from the most recent semi-monthly reporting "
        "settlement date whose figures had been DISSEMINATED by the cutoff, divided "
        "by common shares outstanding as reported for the most recent fiscal period "
        "knowable at the same cutoff. Dissemination, never the settlement date the "
        "row is indexed by: the two are separated by about eight business days, and "
        "keying on the settlement date would grant that much foresight twice a "
        "month. Shares outstanding rather than free float, because float is a "
        "vendor estimate with no as-reported source; the ratio is therefore "
        "understated for closely held companies. Not days-to-cover, which is a "
        "different (volume-normalized) statistic. Expected premium sign: NEGATIVE "
        "— heavily shorted stocks have historically underperformed, so the factor "
        "is named after the quantity it measures and is NOT negated. Zero shares "
        "short is a real observation and produces 0.0; a non-positive share count, "
        "a missing filing or a missing short-interest observation produce NaN. "
        "BLOCKED: no short-interest connector exists, no Phase 3 task covers one "
        "and no BLOCKERS.md entry tracks it, so this feature raises rather than "
        "returning a value. The share-count denominator is separately blocked on B1."
    ),
    units=(
        "dimensionless fraction: shares sold short per share of common stock "
        "outstanding (0.08 = 8 of every 100 shares outstanding are held short)"
    ),
    availability_lag=SHORT_INTEREST_AVAILABILITY_LAG,
    source_tables=frozenset({SHORT_INTEREST_TABLE, FUNDAMENTALS_TABLE}),
)
"""Declaration of ``short_interest``. Lag 17 days — see the module docstring.

Declares both source tables because the ratio needs both legs, and neither
exists. The 17 days is the maximum of the two regimes rather than their sum: a
lag bounds the freshest input, both legs are read at one cutoff, and 17 covers
the fundamentals feed's 7 outright.
"""


@feature(SHORT_INTEREST)
async def short_interest(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``short_interest``: no connector produces this data.

    Args:
        _session: the pinned session, unused — there is no table to read.
            Refusing before any I/O also means the refusal cannot be half-done.
        request: the securities and compute date, used only to name the feature
            in the error.

    Returns:
        Never returns.

    Raises:
        ShortInterestSourceUnavailableError: always, today. There is no
            short-interest table, no connector task and no blocker entry.
        FundamentalsSourceUnavailableError: instead of the above if a
            short-interest feed lands while P3.5 is still blocked on B1.
        ShortInterestComputationNotWrittenError: instead of either once both
            tables are mapped and this arithmetic is still unwritten.
    """
    require_short_interest_source(request.feature)
