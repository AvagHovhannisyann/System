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

Phase 3 tables (P3.8, migration 0008):

- ``macro_series`` — bitemporal FRED series *definitions*: title, units,
  frequency, seasonal adjustment. Carries the units every macro value must
  be read with (directive §8).
- ``macro_observation`` — bitemporal macro readings keyed on (series,
  observation date), one row per FRED **vintage**, so a revision is a
  later-knowledge row rather than an update.

Neither EDGAR table carries a foreign key to ``security``: EDGAR identifies
filers by CIK, and no CIK-to-``security_id`` mapping exists yet (it arrives
with the corporate-actions and securities-master connectors, blocked on B1).
The CIK is stored as EDGAR states it; resolving it to an entity is a later
phase's job and inventing the link now would be a fabricated identity.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
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


class MacroSeries(BitemporalMixin, Base):
    """One version of a FRED macro series' *definition* (P3.8, D-011).

    Logical key: ``series_id``. This table answers "what does this series
    mean, and in what units" — :class:`MacroObservation` answers "what was
    its value". They are separate because the units of a macro series are a
    property of the series, not of each reading, and directive §8 requires
    units to be stated explicitly rather than implied by a column name.
    ``UNRATE`` is percent, ``PAYEMS`` is thousands of persons and ``GDPC1``
    is billions of chained dollars; a pipeline that treats them as
    interchangeable numbers is the silent unit bug §8 names.

    **Every version of one series shares a single ``valid_from``** — the
    constant :data:`~backend.ingest.fred.parse.SERIES_DEFINITION_VALID_FROM` —
    with ``valid_to = 'infinity'``. Only ``knowledge_time`` distinguishes
    versions, so a renamed or re-based series is a *correction* under D-011 and
    latest-knowledge-wins yields exactly one definition at any ``as_of``.

    This is the one place the :class:`EdgarFiling` analogy breaks, and the
    break matters. A filing is its own entity, keyed by accession, so giving
    each one its own ``valid_from`` is right. A series definition is a
    *superseded* fact keyed by ``series_id`` alone — the securities-master
    ticker shape — and giving each version its own ``valid_from`` would make
    them distinct facts rather than versions of one, so several would be
    visible at once and a join from an observation to its units would fan out
    and double every row. That is the exact defect
    :mod:`backend.ingest.supersession` documents.

    Units and assumptions: ``frequency_short`` is FRED's own code (``A``,
    ``Q``, ``M``, ``W``, ``D``, ...) and is stored verbatim rather than
    interpreted — this connector deliberately derives **no** period length
    from it (see :class:`MacroObservation` for why). ``observation_start`` and
    ``observation_end`` are calendar dates describing the series' coverage as
    FRED stated it in this vintage; they are not a coverage guarantee for any
    other vintage.
    """

    __tablename__ = "macro_series"
    __bitemporal_key__ = ("series_id",)

    series_id: Mapped[str] = mapped_column(
        Text,
        doc="FRED series identifier, e.g. 'GDPC1', 'UNRATE'. Logical key, stored verbatim.",
    )
    title: Mapped[str] = mapped_column(
        Text, doc="FRED's series title, e.g. 'Real Gross Domestic Product'."
    )
    frequency: Mapped[str] = mapped_column(
        Text, doc="FRED's long frequency label, e.g. 'Quarterly'. Verbatim, not interpreted."
    )
    frequency_short: Mapped[str] = mapped_column(
        Text, doc="FRED's short frequency code, e.g. 'Q'. Verbatim; no period length is derived."
    )
    units: Mapped[str] = mapped_column(
        Text,
        doc=(
            "FRED's units string, e.g. 'Billions of Chained 2017 Dollars', 'Percent'. "
            "The authoritative unit of every MacroObservation.value for this series "
            "at this vintage (directive §8)."
        ),
    )
    units_short: Mapped[str] = mapped_column(
        Text, doc="FRED's abbreviated units string, e.g. 'Bil. of Chn. 2017 $', '%'."
    )
    seasonal_adjustment_short: Mapped[str] = mapped_column(
        Text,
        doc=(
            "FRED's seasonal-adjustment code, e.g. 'SA', 'NSA', 'SAAR'. Stored because "
            "comparing a seasonally adjusted series against an unadjusted one is a "
            "silent analysis error."
        ),
    )
    observation_start: Mapped[dt.date] = mapped_column(
        Date, doc="Earliest observation date FRED reported for this series at this vintage."
    )
    observation_end: Mapped[dt.date] = mapped_column(
        Date, doc="Latest observation date FRED reported for this series at this vintage."
    )
    vintage_start_date: Mapped[dt.date] = mapped_column(
        Date,
        doc=(
            "FRED ``realtime_start`` of this metadata version (calendar date, US/Eastern "
            "calendar). The raw date from which knowledge_time was derived under the "
            "connector's documented lag — never used as a knowledge time itself."
        ),
    )
    vintage_end_date: Mapped[dt.date | None] = mapped_column(
        Date,
        doc=(
            "FRED ``realtime_end`` as stated at ingestion; NULL when FRED returned the "
            "open sentinel 9999-12-31. Point-in-time as of ingestion and deliberately "
            "never corrected — see MacroObservation.vintage_end_date."
        ),
    )
    source_url: Mapped[str] = mapped_column(
        Text,
        doc=(
            "The fred/series request this version was parsed from, with the API key "
            "removed (reproducibility I2; secret isolation I5)."
        ),
    )


class MacroObservation(BitemporalMixin, Base):
    """One value of one macro series at one observation date and one vintage (P3.8).

    Logical key: ``(series_id, observation_date)``. Every FRED **revision** of
    that pair is a separate row carrying a later ``knowledge_time`` — never an
    update — so a query at an ``as_of`` before a revision returns the number
    that was believed then, and the same query after it returns the revised
    number. Macro series are revised routinely (GDP two or more times, payrolls
    every month), and ingesting only the latest value would hand every backtest
    revisions that had not happened yet: the I1 failure this platform exists to
    prevent.

    Event time
    ----------

    ``valid_from`` is ``observation_date`` at 00:00 UTC and ``valid_to`` is
    ``'infinity'`` — **deliberately open-ended**, which differs from
    :class:`PriceBar`'s tiling of day *D* into ``[D, D+1)``. Two reasons:

    1. the reading is a fact *about* a period that stays true forever once
       measured; the period it describes is carried explicitly by
       ``observation_date``, so nothing is lost by leaving the interval open;
    2. bounding it would require a period length, and FRED does not publish
       one per observation. It could only be derived from the series
       frequency, and a derived boundary that is wrong for the irregular
       cases (frequency changes, discontinued series) would be an invented
       fact. Directive §9.4/§9.8: not guessed.

    Consecutive observations of one series therefore have **overlapping**
    open event-time intervals. That is correct here and is not the defect
    :mod:`backend.ingest.supersession` guards against: those are distinct
    logical keys (distinct ``observation_date``s), so the as-of read returns
    exactly one row per (series, observation date) and no join can fan out.
    The query a consumer wants is "greatest ``observation_date`` whose
    ``valid_from <= t``", not "interval containing *t*" — which would be wrong
    for macro anyway, since the reading for the quarter containing *t* is
    typically not published until well after *t*.

    Missing values
    --------------

    FRED marks a missing observation with the string ``"."``. It is stored as
    ``value = NULL`` with ``is_missing = true``, never as ``0`` and never as
    NaN, and a database CHECK keeps the two columns consistent. The boolean
    exists so "FRED stated there is no value here" is distinguishable from
    "we never ingested this row" — the first is knowledge, the second is
    absence of it. A missing value is deliberately **not** recorded as a
    retraction: FRED does not distinguish a deleted observation from an
    unavailable one, and inventing that distinction would be fabrication.

    Units
    -----

    ``value`` is **unitless in this table by design**. Its unit is
    :attr:`MacroSeries.units` for the same ``series_id`` at the same
    ``knowledge_time``, and any consumer that reads a value without reading
    that string is committing the §8 unit error. The column is arbitrary
    precision ``NUMERIC`` with no declared scale, so no source value is
    silently rounded on the way in.
    """

    __tablename__ = "macro_observation"
    __bitemporal_key__ = ("series_id", "observation_date")

    series_id: Mapped[str] = mapped_column(
        Text, doc="FRED series identifier this observation belongs to. Part of the logical key."
    )
    observation_date: Mapped[dt.date] = mapped_column(
        Date,
        doc=(
            "The period this reading measures, as FRED labels it (period start for "
            "aggregated frequencies: Q1 2024 is '2024-01-01'). Calendar date, no time "
            "component. Mirrored into valid_from at 00:00 UTC by a CHECK constraint."
        ),
    )
    value: Mapped[Decimal | None] = mapped_column(
        Numeric,
        doc=(
            "The observed value at this vintage. NULL exactly when is_missing is true. "
            "Unit is MacroSeries.units for this series — never assume one. Arbitrary "
            "precision: no declared scale, so nothing is rounded at write."
        ),
    )
    is_missing: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        doc="True when FRED reported the missing marker '.' for this observation at this vintage.",
    )
    vintage_start_date: Mapped[dt.date] = mapped_column(
        Date,
        doc=(
            "FRED ``realtime_start``: the first vintage date at which this value was the "
            "latest revision. The raw date knowledge_time was derived from under the "
            "connector's documented conservative lag — never itself a knowledge time."
        ),
    )
    vintage_end_date: Mapped[dt.date | None] = mapped_column(
        Date,
        doc=(
            "FRED ``realtime_end`` **as stated at ingestion**; NULL when FRED returned the "
            "open sentinel 9999-12-31. Never used by the as-of read path, which supersedes "
            "purely by knowledge_time (D-011). It is not corrected when a later revision "
            "closes the period: the superseding fact is the later-knowledge row, and "
            "rewriting this one would be an UPDATE the store does not permit."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 7 tables (P7.1, migration 0009) — the LLM provider registry.
#
# Neither table below is bitemporal, and that is deliberate. The bitemporal
# columns describe when a fact was true in the world and when it became
# knowable to the *market* (D-011). An operator's provider credential and an
# operator's model assignment are things **we** did to our own system: they
# have no market knowability, so any ``knowledge_time`` written for them would
# be a fabricated number in the one column whose meaning is that it is not
# fabricated (I3). This is the same reasoning that keeps ``ingestion_run`` and
# ``config_change_event`` out of the bitemporal store. Consequently neither
# table is in the bitemporal registry and neither is scoped by the Core-level
# read guard.
# ---------------------------------------------------------------------------

PROVIDER_NAMES_SQL = "'anthropic', 'openai'"
"""Vocabulary of LLM providers, as the database CHECK constraints spell it.

Duplicated from :class:`backend.extraction.providers.catalog.Provider` rather
than imported: ``backend.db`` must not depend on ``backend.extraction``, and
an import in that direction would invert the layering the repository layout
(§4) sets out. The duplication is not left to trust —
``backend/tests/extraction/test_providers_registry.py`` asserts that this
string and the enum name exactly the same set, so drift fails the suite rather
than reaching a database.

Adding a provider is therefore a **code plus migration** change. That is the
honest cost, not an oversight: a provider the platform cannot construct a
request for (:mod:`backend.extraction.providers.probe`) is a provider it
cannot use, so there is no state in which a bare database row would be
sufficient.
"""


class LlmProviderCredential(Base):
    """One provider's API key, encrypted at rest (§6.5, §7, I5).

    One row per provider — the credential *currently in force*. The plaintext
    key exists nowhere: :attr:`ciphertext` is a Fernet token produced by
    :mod:`backend.core.crypto` under the environment KEK, and
    :attr:`masked_display` is the only rendering any reader ever sees
    (``sk-...4f2a``). There is no column, view, or endpoint from which a full
    key can be recovered other than decryption inside the process that is
    about to call the provider.

    **Deliberately mutable, unlike every other table in this schema.** Rotation
    UPDATEs this row and deletion removes it; there is no append-only trigger.
    That is a security decision, not a lapse in the repository's
    append-only discipline:

    - a retired key kept as a historical row stays decryptable for as long as
      the KEK lives, so an append-only credential table would quietly widen the
      blast radius of a KEK compromise from "every key in use" to "every key
      ever used". Rotation is normally performed *because* the old key should
      stop existing;
    - the history §6.5 asks for is history of the **operator's actions**, and
      that lives in ``config_change_event`` (:mod:`backend.db.audit`), which is
      append-only and records who rotated what, when, and — as masked
      renderings — from which key to which. That history is safe to keep
      forever precisely because it contains no recoverable secret.

    So: this table is current state; the audit log is the record. A change that
    is not in the audit log did not happen.

    Units: :attr:`key_version` is a dimensionless counter starting at 1 and
    incremented by one on each rotation. All timestamps are ``TIMESTAMPTZ`` in
    UTC, from the database clock at transaction start.
    """

    __tablename__ = "llm_provider_credential"
    __table_args__ = (
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would double the prefix and diverge from the
        # name migration 0009 creates.
        CheckConstraint(f"provider IN ({PROVIDER_NAMES_SQL})", name="provider_known"),
        CheckConstraint("ciphertext <> ''", name="ciphertext_not_empty"),
        CheckConstraint("masked_display <> ''", name="masked_display_not_empty"),
        CheckConstraint("key_version >= 1", name="key_version_positive"),
    )

    provider: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
        doc="Provider this credential authenticates against, e.g. 'anthropic'. One row each.",
    )
    ciphertext: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Fernet token of the API key, urlsafe-base64 ASCII, produced under the "
            "environment KEK (backend.core.crypto). Never the plaintext; never returned "
            "by any endpoint (§7)."
        ),
    )
    masked_display: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Permanently masked rendering for the operator, e.g. 'sk-...4f2a' "
            "(backend.core.crypto.mask_secret). Stored rather than derived because "
            "deriving it would require decrypting on every read."
        ),
    )
    key_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Dimensionless rotation counter: 1 on first configuration, +1 per rotation.",
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When this provider was first configured, UTC, from the database clock.",
    )
    rotated_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the key in force was last replaced, UTC. Equals created_at until rotation.",
    )


class ExtractionModelAssignment(Base):
    """One *version* of one extraction task's model assignment (§6.5, P7.1).

    §6.5 requires that changing a task's assignment **creates a new
    configuration version rather than mutating the current one**, so the unit
    of this table is a version, not a task: the logical key is
    ``(task, version)``, the assignment in force is the row with the greatest
    ``version`` for that task, and every superseded version stays readable
    exactly as it was written. Migration 0009 installs a
    ``BEFORE UPDATE OR DELETE`` row trigger, so that is enforced by the
    database rather than by convention.

    A row records the assignment *and* the act of assigning it — ``actor``,
    ``recorded_at`` and ``correlation_id`` — which duplicates what
    :mod:`backend.db.audit` records for the same change. The duplication is
    intentional: the audit log answers "what did the operator change" across
    every subsystem, while this table answers "what configuration did this
    extraction run use", and an extraction result that cannot name its own
    configuration without joining a cross-subsystem log is a reproducibility
    hazard (I2).

    Assignments are **not** foreign-keyed to
    :class:`LlmProviderCredential`. A task may legitimately be assigned to a
    provider whose key is not configured yet (configure the pipeline, then add
    the credential); the refusal belongs at call time, where
    :mod:`backend.extraction.providers.probe` and the extraction client raise
    rather than invent a result (I3).

    Units and assumptions:

    - ``temperature`` — dimensionless sampling temperature. **Defaults to 0
      everywhere** (§5-P7 requires it); the column merely permits other values
      because §6.5 makes temperature a per-task setting. Range 0 to 2 inclusive,
      which is the union of the providers' accepted ranges — a value a specific
      provider rejects is that provider's error to state, not this schema's to
      guess.
    - ``max_tokens`` — maximum response length in tokens (count, ≥ 1).
    - ``timeout_s`` — per-request wall-clock timeout in **seconds** (> 0).
    - ``version`` — dimensionless, 1 for a task's first assignment, +1 each
      change. Ascending version order is the change order for that task
      (writes per task are serialized by an advisory lock).
    """

    __tablename__ = "extraction_model_assignment"
    __table_args__ = (
        UniqueConstraint("task", "version", name="uq_extraction_model_assignment_task_version"),
        CheckConstraint("task <> ''", name="task_not_empty"),
        CheckConstraint("model <> ''", name="model_not_empty"),
        CheckConstraint("actor <> ''", name="actor_not_empty"),
        CheckConstraint(f"provider IN ({PROVIDER_NAMES_SQL})", name="provider_known"),
        CheckConstraint("version >= 1", name="version_positive"),
        CheckConstraint("temperature >= 0 AND temperature <= 2", name="temperature_in_range"),
        CheckConstraint("max_tokens >= 1", name="max_tokens_positive"),
        CheckConstraint("timeout_s > 0", name="timeout_positive"),
    )

    assignment_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; not the version number.",
    )
    task: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Extraction task this assignment configures, e.g. 'risk_factor_delta'. Open text.",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Configuration version for this task: 1 for the first, +1 per change (§6.5).",
    )
    provider: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Provider the task calls, e.g. 'anthropic'. Not an FK — see the class docstring.",
    )
    model: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Provider-side model identifier, stored verbatim and never interpreted here.",
    )
    temperature: Mapped[Decimal] = mapped_column(
        Numeric(4, 3),
        nullable=False,
        doc="Dimensionless sampling temperature, 0 to 2. 0 is the default everywhere (§5-P7).",
    )
    max_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Maximum response length in tokens (count, >= 1).",
    )
    timeout_s: Mapped[Decimal] = mapped_column(
        Numeric(6, 3),
        nullable=False,
        doc="Per-request wall-clock timeout in seconds (> 0).",
    )
    actor: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Who made this assignment, as asserted by the caller — not an authenticated identity.",
    )
    correlation_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Request id (D-003) this assignment was made under; NULL outside a request.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc=(
            "When this version was recorded, UTC, from the database clock at transaction "
            "start (never the caller's). Ordering key for a task is `version`, not this."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 7 tables (P7.3/P7.6/P7.10, migration 0010): the extraction framework
#
# Four tables, all **append-only** by trigger, holding what an extraction run
# produced and the prompt versions it produced it under.
#
# None is bitemporal, for the reason recorded above the P7.1 tables and in
# revisions 0005/0007/0009: the bitemporal columns describe when a fact was
# true in the world and when it became knowable to the *market* (D-011). A
# prompt version, an activation, a golden-set score and an extraction result
# are all things **we** did to our own system. They have no market
# knowability, and a `knowledge_time` invented for them would be a fabricated
# value in the one column whose meaning is that it is not fabricated (I3).
#
# None is a hypertable either: a prompt library is a human-sized collection,
# and extraction results are keyed by document rather than by an event-time
# axis worth chunking on.
# ---------------------------------------------------------------------------


class ExtractionPromptVersion(Base):
    """One immutable, content-addressed version of one prompt (§6.5, P7.10).

    A prompt version has no version *number* and no mutable "current text"
    row: its identity **is** :attr:`version_hash`, a digest of its own content
    (:mod:`backend.extraction.prompts.versioning`). Three consequences shape
    this table:

    - two writings of the same prompt are the same row, so saving is
      idempotent on ``(name, version_hash)`` and that pair is unique;
    - editing a prompt cannot edit a version — it produces a different address
      — so nothing here is ever updated, enforced by migration 0010's
      ``BEFORE UPDATE OR DELETE`` trigger rather than by convention;
    - **rollback is selection, not mutation** (§6.5's "one-click rollback"):
      it is an activation naming an earlier hash, so the version returned to
      is byte-identical to the one that was measured and the golden-set score
      attached to that hash still describes the text now in force.

    :attr:`schema_digest` is part of the content address and therefore part of
    what makes two versions different. Leaving it out would let a response
    cached under an old output schema stay addressable by a prompt that now
    demands a different shape.

    Units: :attr:`sequence` is dimensionless, 1 for a prompt's first saved
    version and +1 per new version, and is the ordering key — a clock can tie
    and under concurrency can run backwards relative to the sequence of events
    (:mod:`backend.db.audit`), so ``recorded_at`` answers "when" and this
    answers "after what".
    """

    __tablename__ = "extraction_prompt_version"
    __table_args__ = (
        UniqueConstraint("name", "version_hash", name="uq_extraction_prompt_version_address"),
        UniqueConstraint("name", "sequence", name="uq_extraction_prompt_version_sequence"),
        CheckConstraint("name <> ''", name="name_not_empty"),
        CheckConstraint("version_hash <> ''", name="version_hash_not_empty"),
        CheckConstraint("template <> ''", name="template_not_empty"),
        CheckConstraint("schema_digest <> ''", name="schema_digest_not_empty"),
        CheckConstraint("actor <> ''", name="actor_not_empty"),
        CheckConstraint("sequence >= 1", name="sequence_positive"),
    )

    prompt_version_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; not the content address.",
    )
    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "The prompt's identity, conventionally the extraction task it serves. Part of "
            "the content address: two prompts sharing text are still two histories."
        ),
    )
    version_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Content address: 32 lowercase hex characters over name, system, template and "
            "schema_digest. Derived, never chosen; the cache's prompt_version component."
        ),
    )
    system: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="System instruction sent with every call, verbatim.",
    )
    template: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="User-message template, a string.Template source whose only variable is $document.",
    )
    schema_digest: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Digest of the Pydantic output model the response must satisfy. Part of the "
            "address, so a schema change is a new prompt version — which is what it is."
        ),
    )
    actor: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Who saved it, as asserted by the caller — not an authenticated identity.",
    )
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Free text explaining the change; NULL when none was given. Not in the address.",
    )
    correlation_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Request id (D-003) it was saved under; NULL outside a request. Not in the address.",
    )
    sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Dimensionless save order within one prompt: 1 for the first, +1 per version.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc=(
            "When this version was first saved, UTC, from the database clock at transaction "
            "start (never the caller's). Ordering key is `sequence`, not this."
        ),
    )


class ExtractionPromptActivation(Base):
    """One act of pointing a prompt at a version (§6.5, P7.10).

    "Which version is in force" is a *pointer*, and moving it is a
    configuration change, so it is an event — §6.11's rule that config changes
    are events and not mutations, applied to prompts. The version in force is
    the row with the greatest :attr:`sequence` for that prompt; there is no
    "current version" column anywhere, because a column would be a state that
    could disagree with its own history.

    §6.5's **one-click rollback is an activation naming an earlier hash** — the
    same call as moving forward. Nothing is restored because nothing was
    destroyed. :attr:`is_rollback` records that the pointer moved *backwards*
    and is derived from the activation history inside the writing transaction,
    never asserted by the caller, so it cannot disagree with the record.

    The composite foreign key to :class:`ExtractionPromptVersion` is the point
    of the constraint, not decoration: activating a hash nobody saved would
    leave the task with no resolvable prompt at its next call, and the failure
    would surface far from the mistake.
    """

    __tablename__ = "extraction_prompt_activation"
    __table_args__ = (
        ForeignKeyConstraint(
            ("name", "version_hash"),
            (
                "extraction_prompt_version.name",
                "extraction_prompt_version.version_hash",
            ),
            name="fk_extraction_prompt_activation_version",
        ),
        UniqueConstraint("name", "sequence", name="uq_extraction_prompt_activation_sequence"),
        CheckConstraint("name <> ''", name="name_not_empty"),
        CheckConstraint("version_hash <> ''", name="version_hash_not_empty"),
        CheckConstraint("actor <> ''", name="actor_not_empty"),
        CheckConstraint("sequence >= 1", name="sequence_positive"),
        # A prompt's first activation has no predecessor and every later one
        # does. Written as a constraint rather than trusted, because a NULL
        # predecessor on a later activation would silently break the chain a
        # reader walks to reconstruct which version was in force when.
        CheckConstraint(
            "(sequence = 1) = (previous_version_hash IS NULL)",
            name="first_activation_has_no_predecessor",
        ),
    )

    activation_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless.",
    )
    name: Mapped[str] = mapped_column(Text, nullable=False, doc="The prompt whose pointer moved.")
    version_hash: Mapped[str] = mapped_column(
        Text, nullable=False, doc="Content address of the version put in force."
    )
    previous_version_hash: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Address in force before this activation; NULL only on a prompt's first one.",
    )
    actor: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Who activated it, as asserted by the caller — not an authenticated identity.",
    )
    correlation_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Request id (D-003) it was activated under; NULL outside a request.",
    )
    sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Dimensionless activation order within one prompt: 1 for the first, +1 each.",
    )
    is_rollback: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc=(
            "True when this hash had already been in force earlier in this prompt's history "
            "— i.e. the operator went back. Derived from the history, never asserted."
        ),
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc=(
            "When the pointer moved, UTC, from the database clock at transaction start. "
            "Ordering key is `sequence`, not this."
        ),
    )


class ExtractionGoldenScore(Base):
    """One golden-set result, attached to one prompt version (§6.5, P7.10, D-014).

    §5-P7 requires that any prompt change re-runs the golden set and that the
    score is recorded. A score is bound to ``(name, version_hash)`` by a
    foreign key, so it describes text that cannot change underneath it — a
    score attached to a mutable prompt would silently start describing text it
    never saw.

    There is deliberately **no pass/fail column and no stored threshold.** Per
    D-014, the directive's "≥ 85% agreement" is the operator's *prior*: a model
    gate above the human labeller's own intra-rater agreement cannot be met by
    any model, and one near it is measuring label noise. The floor has not been
    measured yet (B3), so a threshold column would invite a fabricated number
    in the one place a reader would take as authoritative (I3). Judging a score
    is :func:`backend.extraction.prompts.store.golden_verdict`, which requires
    the threshold *and the statement of where it came from*.

    Units: :attr:`agreement` is a **fraction in [0, 1]**, never a percentage
    (§8) — "85" versus "0.85" is exactly the silent unit bug §8 exists for.
    :attr:`document_count` is a count of documents, recorded because agreement
    over 12 documents and over 400 are not the same number.
    """

    __tablename__ = "extraction_golden_score"
    __table_args__ = (
        ForeignKeyConstraint(
            ("name", "version_hash"),
            (
                "extraction_prompt_version.name",
                "extraction_prompt_version.version_hash",
            ),
            name="fk_extraction_golden_score_version",
        ),
        CheckConstraint("name <> ''", name="name_not_empty"),
        CheckConstraint("version_hash <> ''", name="version_hash_not_empty"),
        CheckConstraint("golden_set_id <> ''", name="golden_set_id_not_empty"),
        CheckConstraint("actor <> ''", name="actor_not_empty"),
        CheckConstraint("agreement >= 0 AND agreement <= 1", name="agreement_is_a_fraction"),
        CheckConstraint("document_count >= 1", name="document_count_positive"),
    )

    score_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; also the ordering key.",
    )
    name: Mapped[str] = mapped_column(Text, nullable=False, doc="The prompt scored.")
    version_hash: Mapped[str] = mapped_column(
        Text, nullable=False, doc="Content address of the exact version scored."
    )
    golden_set_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Identifier of the labelled set used, including its own version. Two scores are "
            "comparable only when this matches; without it a 300-document set and a "
            "500-document one would be mixed silently."
        ),
    )
    agreement: Mapped[Decimal] = mapped_column(
        Numeric(6, 5),
        nullable=False,
        doc="Agreement with the human labels: a fraction in [0, 1], never a percentage (§8).",
    )
    document_count: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Documents scored (count, >= 1)."
    )
    scored_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc=(
            "When the scoring run finished, UTC. Supplied by the caller, not defaulted: the "
            "run may have finished long before the row was written."
        ),
    )
    actor: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Who ran the scoring, as asserted by the caller — not an authenticated identity.",
    )
    notes: Mapped[str | None] = mapped_column(
        Text, nullable=True, doc="Free text about the run; NULL when none was given."
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Distinct from scored_at.",
    )


class ExtractionResult(Base):
    """One model call's extraction record: what was asked, and what came back (P7.3).

    §5-P7's pipeline ends *"store with the prompt version hash"* and requires
    raw responses to be stored. This is that row: one per model call — that is,
    one per (task, document, chunk, prompt version, model) — holding
    :attr:`raw_response` **verbatim**, the validated :attr:`output` when
    validation accepted it, and :attr:`validation_errors` when it did not.

    A rejected response is stored, not discarded. Re-asking the same model the
    same question at temperature 0 would produce the same malformed answer and
    cost money to rediscover, and the rejection is data about the prompt that
    Gate G7 needs to be able to count.

    **Append-only** (migration 0010's trigger). An extraction is an
    observation, and every consumer treats it as evidence: the golden set
    scores it (P7.8), the contamination probe differences it (P7.9), the
    document inspector shows it (§6.5). Re-running is a new row, never an
    overwrite — and if two rows for one address disagree, that disagreement is
    itself the observation worth keeping, because it falsifies the determinism
    the cache design rests on.

    What is deliberately absent: the anonymization mapping, and the document
    text in either form. The mapping is the one artifact that re-identifies a
    document, and storing it beside the payload would make anonymization
    decorative; :attr:`payload_digest` is enough to prove two extractions read
    the same text and not enough to reconstruct it.

    Units: :attr:`chunk_index` is 0-based and indexes the **anonymized**
    document (:mod:`backend.extraction.tasks.pipeline` explains why masking
    precedes chunking); token counts are counts **as reported by the
    provider**, never estimated (I3); :attr:`latency_ms` is wall-clock
    milliseconds.
    """

    __tablename__ = "extraction_result"
    __table_args__ = (
        CheckConstraint("task <> ''", name="task_not_empty"),
        CheckConstraint("document_id <> ''", name="document_id_not_empty"),
        CheckConstraint("prompt_version_hash <> ''", name="prompt_version_hash_not_empty"),
        CheckConstraint("payload_digest <> ''", name="payload_digest_not_empty"),
        CheckConstraint("model <> ''", name="model_not_empty"),
        CheckConstraint("chunk_count >= 1", name="chunk_count_positive"),
        CheckConstraint(
            "chunk_index >= 0 AND chunk_index < chunk_count", name="chunk_index_in_range"
        ),
        CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_counted"),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0", name="output_tokens_counted"
        ),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_non_negative"),
        # A response either validated or it did not, and the row must say which
        # without a reader having to interpret two nullable columns
        # independently: an output with errors beside it, or neither, would be a
        # record nobody could act on.
        CheckConstraint(
            "(output IS NOT NULL) <> (jsonb_array_length(validation_errors) > 0)",
            name="output_xor_validation_errors",
        ),
    )

    result_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; also the ordering key.",
    )
    task: Mapped[str] = mapped_column(
        Text, nullable=False, doc="Extraction task this record belongs to."
    )
    document_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Source document identifier — an EDGAR accession, a transcript id, or for a "
            "composed pair a derived id naming both sides. Provenance only: never sent to a "
            "model and never part of a cache address."
        ),
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="0-based position of this chunk within the anonymized document.",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Chunks the anonymized document produced (count, >= 1)."
    )
    prompt_version_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Content address of the prompt used (§5-P7). Not a foreign key to "
            "extraction_prompt_version: a prompt is addressable whether or not anyone chose "
            "to save it to the library, and refusing to record an extraction because its "
            "prompt was unsaved would discard the observation to protect a join."
        ),
    )
    payload_digest: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Digest of the anonymized text actually sent — the cache key's document "
            "component. Proves two extractions read the same text; cannot reconstruct it."
        ),
    )
    model: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Qualified model identifier, e.g. 'anthropic:claude-3-5-haiku-20241022'.",
    )
    raw_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The provider's response text, verbatim and unnormalized (§5-P7).",
    )
    output: Mapped[dict[str, object] | None] = mapped_column(
        # none_as_null: without it SQLAlchemy binds Python None as JSONB `null`,
        # which is NOT SQL NULL — `output IS NOT NULL` would then be true for a
        # rejected response and the xor CHECK would reject the row. The same
        # SQL-NULL-versus-JSON-null trap backend.db.audit warns about, arriving
        # from the write side. Here SQL NULL is the right storage: "there is no
        # output" is an absence, not a JSON value.
        JSONB(none_as_null=True),
        nullable=True,
        doc=(
            "The schema-validated output as JSON, or SQL NULL when validation rejected the "
            "response. NULL is a recorded outcome, not an omission."
        ),
    )
    validation_errors: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        doc="One line per schema problem, in Pydantic's order; empty array when output is set.",
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Whether the response came from the cache rather than from a paid call.",
    )
    input_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Prompt tokens as reported by the provider (count); NULL when it reported none.",
    )
    output_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Response tokens as reported by the provider (count); NULL when it reported none.",
    )
    latency_ms: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 3),
        nullable=True,
        doc="Measured wall-clock duration of the call in milliseconds; NULL on a cache hit.",
    )
    correlation_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Request id (D-003) the extraction ran under; NULL outside a request.",
    )
    extracted_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc=(
            "When the record was written, UTC, from the database clock at transaction start. "
            "Ordering key is `result_id`, not this."
        ),
    )
