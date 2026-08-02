"""Universe screening criteria and the per-name outcome of applying them (P4.1).

The directive's Phase 4 universe is five screens: an exchange whitelist, a price
floor, a market-capitalisation floor, an average-dollar-volume floor, and a
borrow-availability requirement. This module holds the criteria that parameterise
them, the per-name input record they are evaluated against, and the evaluation
itself. It performs **no I/O** — assembling the inputs point-in-time is
:mod:`backend.universe.builder`'s job — which is what makes the screening
arithmetic testable against candidates whose correct answer is derivable by hand.

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

**Every monetary quantity is US dollars, as** ``Decimal``. Not float: a
market-cap floor of 300 million and an ADV floor of one million are compared
against values assembled from prices and share counts, and binary floating point
turns an exact boundary case into a coin flip that differs between runs. The
comparison is ``value >= floor`` — a name sitting exactly on the floor is
**included**, which is stated here because "at least" versus "more than" silently
moves the boundary of the universe.

- ``min_adv_usd`` — dollars per day. Compared against the **median** daily dollar
  volume over the trailing ``adv_lookback_days`` *trading* days (not calendar
  days). Median rather than mean because a single earnings-day volume spike
  should not qualify an otherwise untradeable name.
- ``min_price_usd`` — dollars per share, compared against the **unadjusted**
  close (``price_bar.close_raw_usd``). Unadjusted is the whole point of a price
  floor: the floor is a proxy for tick-size and quoting frictions, which apply to
  the price actually printed. A split-adjusted history makes a $2 stock look like
  a $40 one in 2004 and the screen would then admit names that were penny stocks
  at the time.
- ``min_market_cap_usd`` — dollars.
- ``allowed_exchanges`` — ISO 10383 MIC codes, compared **verbatim** against
  ``security_master.exchange``. No case folding and no aliasing: a mismatch
  between ``"XNYS"`` and ``"xnys"`` is a symbology defect worth surfacing as an
  empty universe rather than papering over, because silently accepting both means
  the screen also silently accepts whatever else the master happens to spell
  differently.
- ``require_borrow`` — when ``False`` the borrow screen is **not applied at
  all**, and it disappears from :meth:`UniverseCriteria.applied_filters` and from
  the waterfall. That is different from applying it and passing everything, and
  the two must stay distinguishable: the first is a stated choice, the second is
  the silent-pass-through failure the whole package is built against.

**Every floor must be strictly positive.** A floor of zero is not a lenient
screen, it is an absent one — no non-negative value can fail ``value >= 0`` — yet
it would still be recorded in the criteria hash and still appear in the waterfall
with a removal count of zero, which reads exactly like a screen that ran and
found nothing to remove. Refusing it also closes the one route around the
market-cap refusal in :mod:`backend.universe.errors`: with a positive floor
mandatory, a build cannot obtain a market-cap-free universe by setting the floor
to zero and calling the screen satisfied.

**Filter order is declared, frozen, and part of the criteria hash**
(:data:`FILTER_ORDER`). Order does not change *which* names are members — a name
must pass every applied screen — but it entirely determines the attribution in
the filter-impact waterfall (§6.3), because a name failing three screens is
counted once, against the first. Reordering the tuple therefore changes every
published waterfall, so it changes the criteria hash too and old snapshots stop
comparing equal to new ones. That is the correct behaviour: they are no longer
describing the same measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Final

from backend.tracking.stamp import canonical_config_hash
from backend.universe.errors import UniverseConsistencyError, UniverseCriteriaError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "FILTER_ORDER",
    "MAX_ADV_LOOKBACK_DAYS",
    "MIN_ADV_LOOKBACK_DAYS",
    "FilterOutcome",
    "UniverseCandidate",
    "UniverseCriteria",
    "evaluate_candidate",
]

FILTER_ORDER: Final = ("exchange", "price", "market_cap", "adv", "borrow")
"""The screens, in the order the waterfall attributes removals to them.

Cheapest and most categorical first: the exchange whitelist is a set membership
test on data every candidate already carries, whereas borrow availability is the
one screen whose input is a separate feed. A name failing several screens is
attributed to the earliest here, so this tuple is the definition of the §6.3
waterfall's shape and is hashed into the criteria (see the module docstring).
"""

MIN_ADV_LOOKBACK_DAYS: Final = 2
"""Shortest permitted ADV window, in trading days.

One bar is not a median and not an *average* daily volume; it is that day's
volume wearing the name of a statistic.
"""

MAX_ADV_LOOKBACK_DAYS: Final = 252
"""Longest permitted ADV window, in trading days (about one calendar year).

A guard against a typo'd lookback rather than a claim about the right window:
the price query's calendar span scales with this number, and a request for
2520 days is far more likely a slipped digit than an intent to measure a decade
of liquidity.
"""


def _canonical_decimal(value: Decimal) -> str:
    """Render a ``Decimal`` so equal values always produce equal text.

    ``Decimal("1E+6")``, ``Decimal("1000000")`` and ``Decimal("1000000.00")`` are
    the same number written three ways, and hashing their ``str()`` forms would
    give one set of criteria three different identities. Normalising first
    collapses trailing zeros and the exponent form; ``format(..., "f")`` then
    renders without scientific notation.

    Args:
        value: the amount to render. Units are the caller's; this only fixes the
            spelling.

    Returns:
        Plain positional decimal text, e.g. ``"1000000"``, ``"1.5"``.
    """
    return format(value.normalize(), "f")


def _require_positive_amount(field: str, value: Decimal) -> None:
    """Refuse a floor that is not a finite, strictly positive amount.

    Args:
        field: field name, for the message.
        value: the floor, in US dollars.

    Raises:
        UniverseCriteriaError: if the value is NaN, infinite, or ``<= 0``. See
            the module docstring for why zero is refused rather than treated as
            "no screen".
    """
    if not value.is_finite():
        msg = f"{field}={value!r} is not a finite amount; a floor must be a real number of USD"
        raise UniverseCriteriaError(msg)
    if value <= 0:
        msg = (
            f"{field}={value!r} must be strictly positive. A floor of zero screens nothing "
            f"— no non-negative value fails `value >= 0` — yet it is recorded in the "
            f"criteria hash and appears in the waterfall with zero removals, which is "
            f"indistinguishable from a screen that ran. State the floor you mean, or drop "
            f"the criterion by changing the code that requests it"
        )
        raise UniverseCriteriaError(msg)


def _require_non_negative_input(field: str, security_id: int, value: Decimal) -> None:
    """Refuse a candidate input that cannot be a real measurement.

    Args:
        field: field name, for the message.
        security_id: the candidate the value belongs to.
        value: the measured amount, in US dollars.

    Raises:
        UniverseConsistencyError: if the value is NaN, infinite, or negative. A
            negative price or a NaN ADV is a corrupt input, not a screening
            decision; letting it flow through would produce a perfectly ordinary
            looking exclusion and bury the defect that caused it.
    """
    if not value.is_finite() or value < 0:
        msg = (
            f"security_id={security_id} has {field}={value!r}, which is not a finite "
            f"non-negative USD amount. This is a corrupt input rather than a name that "
            f"fails a screen, and screening it would hide the defect behind an ordinary "
            f"looking exclusion"
        )
        raise UniverseConsistencyError(msg)


@dataclass(frozen=True, slots=True)
class UniverseCriteria:
    """The five screens' parameters, validated together.

    Attributes:
        min_adv_usd: floor on median daily dollar volume over the trailing
            ``adv_lookback_days`` trading days, in **US dollars per day**.
            Strictly positive.
        min_price_usd: floor on the unadjusted closing price, in **US dollars per
            share**. Strictly positive; see the module docstring on why the
            comparison uses the unadjusted close.
        min_market_cap_usd: floor on market capitalisation, in **US dollars**.
            Strictly positive. Its input does not exist in this repository yet
            (BLOCKERS.md B1) and requesting a build therefore refuses — see
            :class:`~backend.universe.errors.UniverseInputUnavailableError`.
        allowed_exchanges: ISO 10383 MIC codes admitted, compared verbatim
            against ``security_master.exchange``. Non-empty.
        require_borrow: whether the borrow-availability screen is applied at all.
            ``False`` removes it from :meth:`applied_filters` and from the
            waterfall; it does **not** mean "applied and passed".
        adv_lookback_days: length of the ADV window in **trading days** (bars),
            not calendar days, so the window is calendar-independent and exact.
            Between :data:`MIN_ADV_LOOKBACK_DAYS` and
            :data:`MAX_ADV_LOOKBACK_DAYS`.
    """

    min_adv_usd: Decimal
    min_price_usd: Decimal
    min_market_cap_usd: Decimal
    allowed_exchanges: frozenset[str]
    require_borrow: bool
    adv_lookback_days: int = 20

    def __post_init__(self) -> None:
        """Validate floors, exchange codes, and the lookback window.

        Raises:
            UniverseCriteriaError: if any floor is not finite and strictly
                positive, if ``allowed_exchanges`` is empty or contains a blank
                code, if ``require_borrow`` is not a ``bool``, or if
                ``adv_lookback_days`` is outside
                ``[MIN_ADV_LOOKBACK_DAYS, MAX_ADV_LOOKBACK_DAYS]``.
        """
        _require_positive_amount("min_adv_usd", self.min_adv_usd)
        _require_positive_amount("min_price_usd", self.min_price_usd)
        _require_positive_amount("min_market_cap_usd", self.min_market_cap_usd)
        if not self.allowed_exchanges:
            msg = (
                "allowed_exchanges is empty, so no security could ever qualify. An empty "
                "whitelist is not 'every exchange' — state the MIC codes explicitly"
            )
            raise UniverseCriteriaError(msg)
        blank = [code for code in self.allowed_exchanges if not code.strip()]
        if blank:
            msg = f"allowed_exchanges contains {len(blank)} blank code(s); MIC codes are non-empty"
            raise UniverseCriteriaError(msg)
        require_borrow: object = self.require_borrow
        if not isinstance(require_borrow, bool):
            msg = (
                f"require_borrow must be a bool, got {type(require_borrow).__name__}; it is "
                "a screen on/off switch, and a truthy value would silently turn it on"
            )
            raise UniverseCriteriaError(msg)
        lookback: object = self.adv_lookback_days
        if isinstance(lookback, bool) or not isinstance(lookback, int):
            got = type(lookback).__name__
            msg = f"adv_lookback_days must be an int (bool is refused), got {got}"
            raise UniverseCriteriaError(msg)
        if not MIN_ADV_LOOKBACK_DAYS <= self.adv_lookback_days <= MAX_ADV_LOOKBACK_DAYS:
            msg = (
                f"adv_lookback_days={self.adv_lookback_days} is outside "
                f"[{MIN_ADV_LOOKBACK_DAYS}, {MAX_ADV_LOOKBACK_DAYS}] trading days"
            )
            raise UniverseCriteriaError(msg)

    def applied_filters(self) -> tuple[str, ...]:
        """Return the screens actually evaluated, in :data:`FILTER_ORDER`.

        Every screen but ``borrow`` is unconditional. ``borrow`` appears only
        when :attr:`require_borrow` is ``True``; when it is ``False`` the screen
        is absent from the waterfall entirely, which is how "not applied" stays
        distinguishable from "applied and removed nobody".

        Returns:
            The applied filter names, a subsequence of :data:`FILTER_ORDER`.
        """
        return tuple(name for name in FILTER_ORDER if name != "borrow" or self.require_borrow)

    def as_config(self) -> dict[str, object]:
        """Return the criteria as a JSON-canonical mapping — the hash preimage.

        Decimals render through :func:`_canonical_decimal` so the identity
        depends on the *value* rather than on how it was written; the exchange
        set renders sorted so a ``frozenset``'s arbitrary iteration order cannot
        change the hash; :data:`FILTER_ORDER` is included because it defines the
        waterfall attribution, so a reorder makes old and new snapshots
        incomparable and should say so by changing the hash.

        Returns:
            A JSON-serialisable mapping suitable for
            :func:`~backend.tracking.stamp.canonical_config_hash` and for the
            ``criteria`` column of ``universe_snapshot``.
        """
        return {
            "adv_lookback_days": self.adv_lookback_days,
            "allowed_exchanges": sorted(self.allowed_exchanges),
            "filter_order": list(FILTER_ORDER),
            "min_adv_usd": _canonical_decimal(self.min_adv_usd),
            "min_market_cap_usd": _canonical_decimal(self.min_market_cap_usd),
            "min_price_usd": _canonical_decimal(self.min_price_usd),
            "require_borrow": self.require_borrow,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> UniverseCriteria:
        """Rebuild criteria from the mapping :meth:`as_config` produced.

        The inverse used when a persisted snapshot is read back, so a stored
        universe can be re-screened, re-hashed, and compared against the one in
        force. Round-tripping is asserted by the test suite rather than assumed:
        criteria that do not survive the trip would make every persisted
        ``criteria_hash`` unverifiable.

        Args:
            config: a mapping in the shape :meth:`as_config` returns.

        Returns:
            The reconstructed criteria.

        Raises:
            UniverseCriteriaError: if a key is missing, has the wrong JSON type,
                or holds an unparseable decimal — and, through the constructor,
                if the reconstructed values do not describe a filter. A stored
                ``filter_order`` differing from the current :data:`FILTER_ORDER`
                is refused too: the snapshot's waterfall was attributed under an
                order this code no longer applies.
        """
        try:
            lookback = config["adv_lookback_days"]
            exchanges = config["allowed_exchanges"]
            filter_order = config["filter_order"]
            require_borrow = config["require_borrow"]
            amounts = {
                field: config[field]
                for field in ("min_adv_usd", "min_market_cap_usd", "min_price_usd")
            }
        except KeyError as exc:
            msg = f"stored universe criteria are missing key {exc.args[0]!r}"
            raise UniverseCriteriaError(msg) from exc
        if not isinstance(lookback, int) or isinstance(lookback, bool):
            msg = f"stored adv_lookback_days must be a JSON integer, got {lookback!r}"
            raise UniverseCriteriaError(msg)
        if not isinstance(require_borrow, bool):
            msg = f"stored require_borrow must be a JSON boolean, got {require_borrow!r}"
            raise UniverseCriteriaError(msg)
        if not isinstance(exchanges, list) or not all(isinstance(code, str) for code in exchanges):
            msg = f"stored allowed_exchanges must be a JSON array of strings, got {exchanges!r}"
            raise UniverseCriteriaError(msg)
        if list(FILTER_ORDER) != filter_order:
            msg = (
                f"stored filter_order {filter_order!r} differs from the current "
                f"FILTER_ORDER {list(FILTER_ORDER)!r}. The snapshot's waterfall attributed "
                f"removals under an order this code no longer applies, so its counts and a "
                f"freshly built one's are not the same measurement"
            )
            raise UniverseCriteriaError(msg)
        parsed: dict[str, Decimal] = {}
        for field, raw in amounts.items():
            if not isinstance(raw, str):
                msg = f"stored {field} must be a decimal string, got {raw!r}"
                raise UniverseCriteriaError(msg)
            try:
                parsed[field] = Decimal(raw)
            except ArithmeticError as exc:
                msg = f"stored {field}={raw!r} is not a decimal amount"
                raise UniverseCriteriaError(msg) from exc
        return cls(
            min_adv_usd=parsed["min_adv_usd"],
            min_price_usd=parsed["min_price_usd"],
            min_market_cap_usd=parsed["min_market_cap_usd"],
            allowed_exchanges=frozenset(exchanges),
            require_borrow=require_borrow,
            adv_lookback_days=lookback,
        )

    def criteria_hash(self) -> str:
        """Return the SHA-256 hex digest identifying these criteria (I2).

        Computed over the canonical JSON of :meth:`as_config` by
        :func:`~backend.tracking.stamp.canonical_config_hash` — the same
        canonicalisation the reproducibility stamp uses for a run's config, so a
        universe's identity is derived the same way every other artifact's is.

        Note that the full :class:`~backend.tracking.stamp.ReproducibilityStamp`
        is deliberately *not* used here: it requires a seed and a data version,
        and a universe build is deterministic given ``(criteria, rebalance_date,
        as_of)`` — recording ``seed=0`` would put a fabricated number in a field
        whose meaning is that it was the seed actually used. The snapshot records
        the as-of instant instead, which is the data version of a point-in-time
        read.

        Returns:
            64 lowercase hex characters.
        """
        return canonical_config_hash(self.as_config())


@dataclass(frozen=True, slots=True)
class UniverseCandidate:
    """One security's screening inputs, assembled point-in-time.

    Produced by :mod:`backend.universe.builder` from as-of reads; consumed by
    :func:`evaluate_candidate`. Every optional field is ``None`` for exactly one
    reason — **the value is not knowable at the as-of instant** — and ``None``
    fails its screen rather than passing it. That asymmetry is the package's
    central rule: excluding a name we cannot show to be eligible drops a trade,
    while including one we cannot show to be eligible invents a trade, and only
    the second corrupts a backtest.

    A ``None`` here is a *per-name* gap. An entire input source being absent is
    not representable in this class at all; it refuses the build
    (:class:`~backend.universe.errors.UniverseInputUnavailableError`) before any
    candidate is assembled, so a systemic outage can never be mistaken for
    several thousand individually ineligible names.

    Attributes:
        security_id: the identity anchor's surrogate key (dimensionless).
        exchange: MIC code from the ``security_master`` version in force at the
            rebalance date, verbatim.
        price_usd: unadjusted closing price on the most recent bar at or before
            the rebalance date, in **USD per share**; ``None`` when no bar was
            knowable in the lookback window.
        adv_usd: median daily dollar volume over the trailing window, in **USD
            per day**; ``None`` when fewer than ``adv_lookback_days`` bars were
            knowable.
        market_cap_usd: market capitalisation in **USD**; ``None`` when the
            source covered the universe but not this name.
        borrow_available: whether shares were locatable to borrow;
            ``None`` when the borrow feed covered the universe but not this name.
            Only consulted when the criteria require borrow.
    """

    security_id: int
    exchange: str
    price_usd: Decimal | None = None
    adv_usd: Decimal | None = None
    market_cap_usd: Decimal | None = None
    borrow_available: bool | None = None

    def __post_init__(self) -> None:
        """Validate the identity and refuse impossible measurements.

        Raises:
            UniverseConsistencyError: if ``security_id`` is not a positive
                integer, or if any supplied amount is NaN, infinite, or negative.
        """
        security_id: object = self.security_id
        if isinstance(security_id, bool) or not isinstance(security_id, int):
            msg = f"security_id must be an int, got {type(security_id).__name__}"
            raise UniverseConsistencyError(msg)
        if self.security_id <= 0:
            msg = f"security_id={self.security_id} is not a positive identity-anchor key"
            raise UniverseConsistencyError(msg)
        if self.price_usd is not None:
            _require_non_negative_input("price_usd", self.security_id, self.price_usd)
        if self.adv_usd is not None:
            _require_non_negative_input("adv_usd", self.security_id, self.adv_usd)
        if self.market_cap_usd is not None:
            _require_non_negative_input("market_cap_usd", self.security_id, self.market_cap_usd)


@dataclass(frozen=True, slots=True)
class FilterOutcome:
    """What the screens decided about one candidate.

    Attributes:
        security_id: the candidate's identity-anchor key.
        included: whether the name is a member of the universe — true exactly
            when ``failed_filters`` is empty.
        failed_filters: every applied screen the name failed, in
            :data:`FILTER_ORDER`. **All** failures are recorded, not just the
            first: the waterfall attributes the name to ``failed_filters[0]``
            and needs no more, but "this name failed price *and* ADV" is the
            answer to the operator's actual question — would loosening one screen
            bring it back — and it cannot be recovered later from a single
            attribution.
    """

    security_id: int
    included: bool
    failed_filters: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that inclusion and the failure list agree.

        Raises:
            UniverseConsistencyError: if ``included`` disagrees with whether
                ``failed_filters`` is empty, or if a name is not in
                :data:`FILTER_ORDER`, or if the names are out of order or
                repeated. Two independently settable fields would permit "an
                included name that failed the price screen", which no consumer
                could act on.
        """
        unknown = [name for name in self.failed_filters if name not in FILTER_ORDER]
        if unknown:
            msg = f"failed_filters names unknown screen(s) {unknown}; expected {list(FILTER_ORDER)}"
            raise UniverseConsistencyError(msg)
        positions = [FILTER_ORDER.index(name) for name in self.failed_filters]
        if positions != sorted(set(positions)):
            msg = (
                f"failed_filters={self.failed_filters!r} is not in FILTER_ORDER without "
                f"repeats; the waterfall attributes to the first entry and would misreport"
            )
            raise UniverseConsistencyError(msg)
        if self.included != (not self.failed_filters):
            msg = (
                f"security_id={self.security_id}: included={self.included} disagrees with "
                f"failed_filters={self.failed_filters!r}. A member is exactly a candidate "
                f"that failed nothing"
            )
            raise UniverseConsistencyError(msg)

    @property
    def attributed_filter(self) -> str | None:
        """Return the screen the waterfall counts this exclusion against.

        Returns:
            The first entry of :attr:`failed_filters` — the earliest screen in
            :data:`FILTER_ORDER` the name failed — or ``None`` for a member.
        """
        return self.failed_filters[0] if self.failed_filters else None


def evaluate_candidate(candidate: UniverseCandidate, criteria: UniverseCriteria) -> FilterOutcome:
    """Apply every screen the criteria request to one candidate.

    All applied screens are evaluated — there is no short-circuit — so the
    outcome records the complete failure set rather than the first failure. Each
    screen fails when its input is ``None``, i.e. when the value was not knowable
    at the as-of instant: see :class:`UniverseCandidate` on why the unknown
    direction is exclusion.

    The screens, in :data:`FILTER_ORDER`:

    ``exchange``
        ``candidate.exchange in criteria.allowed_exchanges``, compared verbatim.

    ``price``
        ``price_usd >= min_price_usd`` (USD per share, unadjusted close).

    ``market_cap``
        ``market_cap_usd >= min_market_cap_usd`` (USD).

    ``adv``
        ``adv_usd >= min_adv_usd`` (USD per day, median over the trailing
        window).

    ``borrow``
        ``borrow_available is True``. Applied only when
        ``criteria.require_borrow``; ``None`` (no locate information for this
        name) fails, because a short the platform cannot show was borrowable is
        a short it could not have placed.

    Args:
        candidate: the name's assembled point-in-time inputs.
        criteria: the screens to apply.

    Returns:
        A :class:`FilterOutcome` whose ``failed_filters`` is ordered by
        :data:`FILTER_ORDER`.
    """
    failed: list[str] = []
    if candidate.exchange not in criteria.allowed_exchanges:
        failed.append("exchange")
    if candidate.price_usd is None or candidate.price_usd < criteria.min_price_usd:
        failed.append("price")
    if candidate.market_cap_usd is None or candidate.market_cap_usd < criteria.min_market_cap_usd:
        failed.append("market_cap")
    if candidate.adv_usd is None or candidate.adv_usd < criteria.min_adv_usd:
        failed.append("adv")
    if criteria.require_borrow and candidate.borrow_available is not True:
        failed.append("borrow")
    return FilterOutcome(
        security_id=candidate.security_id,
        included=not failed,
        failed_filters=tuple(failed),
    )
