"""Fact tables of the bitemporal store (P2.2 and P3.2, DECISIONS.md D-011).

Phase 2 tables:

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

Phase 3 tables (P3.2, migration 0006):

- ``edgar_filing`` — one bitemporal row per (SEC EDGAR filing, filer CIK)
  pair, whose event time *and* knowledge time are the filing's **acceptance
  instant**.
- ``edgar_filing_document`` — the document manifest of each filing, keyed on
  the accession alone: a submission's documents do not vary by filer.

Neither EDGAR table carries a foreign key to ``security``: EDGAR identifies
filers by CIK, and no CIK-to-``security_id`` mapping exists yet (it arrives
with the corporate-actions and securities-master connectors, blocked on B1).
The CIK is stored as EDGAR states it; resolving it to an entity is a later
phase's job and inventing the link now would be a fabricated identity.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import BigInteger, Date, ForeignKey, Identity, Integer, Numeric, Text
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


class EdgarFiling(BitemporalMixin, Base):
    """One SEC EDGAR filing, versioned bitemporally (P3.2, D-011).

    Logical key: ``(accession_number, cik)``. EDGAR's daily index lists one
    accession once per associated filer — a joint Form 4 by six Deerfield
    entities and its issuer appears seven times (accession
    ``0001193805-24-000360``, 2024-03-12), and 1809 of the 2024-03-11 index's
    3394 accessions are listed under more than one CIK. The row therefore
    records a (filing, filer) association, which is what the source states.
    Keying on the accession alone would force a choice of one filer's CIK and
    name to stand for the filing, which is a fabricated choice. The submission
    itself — its acceptance instant, form type, period and document manifest —
    is identical across those rows, and its documents are stored once
    (:class:`EdgarFilingDocument` is keyed on the accession, not on the filer).

    **Event time and knowledge time are both the acceptance instant.** A
    filing's existence becomes a fact at the moment EDGAR accepts it and stays
    a fact indefinitely, so ``valid_from`` is the acceptance instant (in UTC)
    and ``valid_to`` is ``'infinity'``. ``knowledge_time`` is the same instant
    on a filing's first version, because acceptance is when the submission
    became publicly retrievable. The two are separate columns rather than one
    because a later correction to the *header* (a new version of this row)
    keeps ``valid_from`` at the original acceptance instant while carrying the
    later ``knowledge_time`` at which the correction became knowable — which
    is exactly the shape D-011 exists to express. Read
    :attr:`acceptance_datetime` rather than ``valid_from`` where the intent is
    "when was this filed".

    ``filing_date`` is stored **beside** the acceptance instant and is never
    used as a knowledge time. Under 17 CFR 232.13 a submission accepted after
    5:30 p.m. ET is deemed filed the next business day, while Forms 3/4/5,
    Schedules 13D/13G/14N, Form 144 and Rule 462(b) filings are deemed filed
    the same day until 10 p.m. ET. So the filing date can trail acceptance by
    a weekend (accession ``0000950172-24-000037``: accepted 2024-03-08
    17:59:09 ET, filed 2024-03-11) *and* can precede the acceptance instant in
    UTC terms (accession ``0001225208-24-004041``: filed 2024-03-11, accepted
    2024-03-12T00:13:08Z). The second direction is the lookahead: treating the
    filing date as knowledge time would claim a filing was knowable roughly a
    day before it existed.

    Units and assumptions: ``cik`` is a dimensionless EDGAR Central Index Key;
    ``filing_date``, ``index_date`` and ``period_of_report`` are calendar dates
    with no time component; ``document_count`` counts manifest entries actually
    parsed and ``declared_document_count`` is EDGAR's own
    ``<PUBLIC-DOCUMENT-COUNT>``, kept separately because a disagreement between
    them is a measurement worth reporting rather than something to reconcile.
    """

    __tablename__ = "edgar_filing"
    __bitemporal_key__ = ("accession_number", "cik")

    accession_number: Mapped[str] = mapped_column(
        Text,
        doc="EDGAR accession number in dashed form, e.g. '0001104659-24-032038'. Logical key.",
    )
    cik: Mapped[int] = mapped_column(
        BigInteger,
        doc="EDGAR Central Index Key this filing is listed under. Part of the logical key.",
    )
    company_name: Mapped[str] = mapped_column(
        Text,
        doc="Filer name exactly as the daily index spells it (not normalized).",
    )
    form_type: Mapped[str] = mapped_column(
        Text,
        doc="EDGAR form type from the submission header, e.g. '10-K', '8-K', '4'.",
    )
    filing_date: Mapped[dt.date] = mapped_column(
        Date,
        doc="EDGAR's assigned filing date (calendar date). Never used as a knowledge time.",
    )
    index_date: Mapped[dt.date] = mapped_column(
        Date,
        doc="Date of the daily index this filing was disseminated in; may exceed filing_date.",
    )
    period_of_report: Mapped[dt.date | None] = mapped_column(
        Date,
        doc="Fiscal period the filing reports on (calendar date); NULL when none is declared.",
    )
    declared_document_count: Mapped[int | None] = mapped_column(
        Integer,
        doc="The header's <PUBLIC-DOCUMENT-COUNT> (count); NULL when the header omits it.",
    )
    document_count: Mapped[int] = mapped_column(
        Integer,
        doc="Manifest entries actually parsed and written to edgar_filing_document (count).",
    )
    source_url: Mapped[str] = mapped_column(
        Text,
        doc="The -index-headers.html URL these fields were parsed from (reproducibility, I2).",
    )

    @property
    def acceptance_datetime(self) -> dt.datetime:
        """Return when EDGAR accepted this submission, timezone-aware UTC.

        This is ``valid_from`` under its domain name. It is the instant the
        filing became publicly retrievable, converted from the submission
        header's US/Eastern value by
        :func:`backend.ingest.edgar.parse.acceptance_datetime_to_utc`.
        """
        return self.valid_from


class EdgarFilingDocument(BitemporalMixin, Base):
    """One document inside an EDGAR filing's manifest (P3.2, D-011).

    Logical key: ``(accession_number, document_sequence)``. Event time and
    knowledge time are the parent filing's acceptance instant — a document is
    knowable exactly when the submission containing it is.

    There is deliberately **no foreign key to** ``edgar_filing``: a bitemporal
    table's logical key is not unique per row (that is the point of
    versioning), so it cannot be a foreign-key target. D-011 solves this for
    price data with a separate identity anchor; here EDGAR already supplies a
    stable external identifier — the accession number — so the join key is the
    accession, and referential integrity is a property of the writer (both
    tables are written in the same batch) rather than of the schema. Stated
    here rather than left to be discovered.

    Units: ``document_sequence`` is EDGAR's sequence number within the filing
    (dimensionless) and is **not contiguous** — accession
    ``0000950170-24-029225`` lists sequences 1, 2, 3, 5 — so no count may be
    derived from it.
    """

    __tablename__ = "edgar_filing_document"
    __bitemporal_key__ = ("accession_number", "document_sequence")

    accession_number: Mapped[str] = mapped_column(
        Text,
        doc="Accession number of the filing this document belongs to (join key, not an FK).",
    )
    document_sequence: Mapped[int] = mapped_column(
        Integer,
        doc="EDGAR sequence number within the filing (dimensionless, not contiguous).",
    )
    document_type: Mapped[str] = mapped_column(
        Text,
        doc="EDGAR document type, e.g. '10-K', 'EX-99.1', 'GRAPHIC', 'XML'.",
    )
    filename: Mapped[str] = mapped_column(
        Text,
        doc="File name within the filing's archive directory.",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        doc="Manifest description; NULL when EDGAR's manifest carries none.",
    )
    document_url: Mapped[str] = mapped_column(
        Text,
        doc="Absolute URL of the document within the EDGAR archive (reproducibility, I2).",
    )
