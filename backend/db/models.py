"""First fact tables of the bitemporal store (P2.2, DECISIONS.md D-011).

Three tables:

- ``security`` — a non-bitemporal *identity anchor*: one row per security
  entity, holding only the surrogate ``security_id``. It exists so both
  bitemporal tables can carry a real foreign key (the versioned tables
  cannot be FK targets because their logical key is not unique per row).
- ``security_master`` — bitemporal entity identity: ticker, name, exchange,
  listing dates. Identity changes (renames, ticker changes, delistings) are
  new versions, so the master reconstructs what the entity *looked like* at
  any (event time, knowledge time) pair.
- ``price_bar`` — bitemporal daily OHLCV bars. All prices in USD per share,
  volume in shares; raw close and the cumulative adjustment factor are
  stored separately (directive Phase 3 requirement).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import BigInteger, Date, ForeignKey, Identity, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base
from backend.db.bitemporal import BitemporalMixin


class Security(Base):
    """Identity anchor for one security entity.

    Carries only the database-generated surrogate ``security_id``
    (dimensionless integer). All descriptive attributes live in
    :class:`SecurityMaster` so they can be versioned bitemporally; this
    anchor exists purely as a stable foreign-key target. Rows are created
    once per entity by ingestion and never change.
    """

    __tablename__ = "security"

    security_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate entity key, database-generated. Dimensionless.",
    )


class SecurityMaster(BitemporalMixin, Base):
    """Bitemporal securities master: versioned entity identity (D-011).

    One row per *version* of an entity's identity. The logical key is
    ``security_id``; ``valid_from``/``valid_to`` bound the event-time
    interval during which the identity applied (open-ended ``valid_to =
    'infinity'`` until superseded), and ``knowledge_time`` is when that
    identity became knowable. A ticker change, rename, or delisting is a new
    row — never an update — so any historical as-of query reconstructs the
    identity as it was believed at that time.

    Units and assumptions: ``ticker`` is the primary listing ticker (upper
    case, exchange-local convention); ``first_listed_on``/``delisted_on``
    are exchange-calendar dates (no intraday component; nullable when
    unknown or, for ``delisted_on``, while still listed).
    """

    __tablename__ = "security_master"
    __bitemporal_key__ = ("security_id",)

    security_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("security.security_id"),
        doc="Logical entity key: FK to the security identity anchor.",
    )
    ticker: Mapped[str] = mapped_column(
        Text,
        doc="Primary listing ticker at this version (exchange-local symbology).",
    )
    name: Mapped[str] = mapped_column(
        Text,
        doc="Issuer/instrument name at this version.",
    )
    exchange: Mapped[str] = mapped_column(
        Text,
        doc="Primary listing exchange code at this version (e.g. XNYS, XNAS).",
    )
    first_listed_on: Mapped[dt.date | None] = mapped_column(
        Date,
        doc="First listing date (exchange calendar date); NULL when unknown.",
    )
    delisted_on: Mapped[dt.date | None] = mapped_column(
        Date,
        doc="Delisting date (exchange calendar date); NULL while listed or unknown.",
    )


class PriceBar(BitemporalMixin, Base):
    """Bitemporal daily OHLCV price bar (D-011; directive Phase 3 units).

    Logical key: ``security_id``. Event time: the trading day *D*, encoded
    half-open as ``valid_from = D 00:00Z``, ``valid_to = D+1 00:00Z``.
    ``knowledge_time`` is when this bar (or this re-adjusted version of it)
    became knowable; a vendor re-adjustment after a corporate action arrives
    as a new version with a later ``knowledge_time``, never as an update.

    Units:

    - ``open_usd`` / ``high_usd`` / ``low_usd`` / ``close_usd`` — USD per
      share, adjusted for corporate actions consistently with
      ``adjustment_factor`` as of this version's ``knowledge_time``.
    - ``close_raw_usd`` — USD per share, unadjusted (as printed on trade
      date). Stored separately per directive Phase 3.
    - ``adjustment_factor`` — dimensionless cumulative multiplicative
      factor: ``close_raw_usd * adjustment_factor == close_usd`` within this
      version. 1 means no adjustment.
    - ``volume_shares`` — unadjusted share count traded on day *D*.
    """

    __tablename__ = "price_bar"
    __bitemporal_key__ = ("security_id",)

    security_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("security.security_id"),
        doc="Logical entity key: FK to the security identity anchor.",
    )
    open_usd: Mapped[Decimal] = mapped_column(
        Numeric(20, 6), doc="Opening price, USD per share (adjusted)."
    )
    high_usd: Mapped[Decimal] = mapped_column(
        Numeric(20, 6), doc="High price, USD per share (adjusted)."
    )
    low_usd: Mapped[Decimal] = mapped_column(
        Numeric(20, 6), doc="Low price, USD per share (adjusted)."
    )
    close_usd: Mapped[Decimal] = mapped_column(
        Numeric(20, 6), doc="Closing price, USD per share (adjusted)."
    )
    volume_shares: Mapped[int] = mapped_column(
        BigInteger, doc="Unadjusted trading volume, in shares."
    )
    close_raw_usd: Mapped[Decimal] = mapped_column(
        Numeric(20, 6), doc="Unadjusted closing price, USD per share, as printed on trade date."
    )
    adjustment_factor: Mapped[Decimal] = mapped_column(
        Numeric(20, 10),
        doc="Dimensionless cumulative factor: close_raw_usd * adjustment_factor == close_usd.",
    )
