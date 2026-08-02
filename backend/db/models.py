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
            name="first_activation_no_predecessor",
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


# ---------------------------------------------------------------------------
# Phase 4 tables (P4.1/P4.2, migration 0011): persisted universe snapshots
#
# Two tables recording what the point-in-time universe screen decided on one
# rebalance date under one set of criteria, **append-only** by the same
# ``BEFORE UPDATE OR DELETE`` trigger shape revisions 0003/0004/0007/0009/0010
# use.
#
# Neither is bitemporal, for the reason recorded above the Phase 7 tables: the
# bitemporal columns describe when a fact was true in the world and when it
# became knowable to the *market* (D-011). A universe snapshot is a computation
# **we** ran over facts that are already bitemporal. Its inputs carry the
# knowledge times; the snapshot carries the ``as_of`` instant it read them at,
# which is the whole of its point-in-time content. Inventing a separate
# ``knowledge_time`` for it would be a fabricated value in the one column whose
# meaning is that it is not fabricated (I3).
#
# Neither is a hypertable: rebalance dates are monthly-to-weekly, so a decade of
# history is hundreds of snapshot rows, not an event-time stream worth chunking.
# ---------------------------------------------------------------------------


class UniverseSnapshot(Base):
    """One point-in-time universe build: criteria, instant, and counts (P4.1).

    The header row for one execution of
    :func:`backend.universe.builder.build_universe`. Its members and the
    per-name screening outcomes live in :class:`UniverseMember`, one row per
    **candidate considered** — not per member — because §6.3 requires a
    filter-impact waterfall showing how many names each screen removed, and a
    table holding only survivors cannot answer that. See that class's docstring.

    **Name collision, stated rather than discovered:**
    :class:`backend.universe.snapshot.UniverseSnapshot` is the in-memory value
    object this table persists. They carry the same fields and the same meaning;
    this one is the row, that one is the result. Code that needs both imports
    this module qualified (``models.UniverseSnapshot``).

    **Identity is** ``(rebalance_date, criteria_hash, as_of)``, and it is
    unique. Those three values determine the snapshot completely given the
    contents of the bitemporal store, which is exactly the reproducibility claim
    I2 asks for: same criteria, same rebalance date, same knowledge instant,
    same universe. Re-running a build after more data has been ingested is a
    *different* ``as_of`` and therefore a new row rather than a correction —
    which is what makes "the universe as we knew it on date X" answerable after
    the fact.

    Deliberately **no reproducibility stamp columns.** A universe build is
    deterministic: it draws no random numbers, so
    :class:`~backend.tracking.stamp.ReproducibilityStamp` cannot be constructed
    for it without inventing a seed, and that module refuses partial stamps for
    precisely this reason. The criteria hash is produced by the same
    canonicalisation the stamp uses
    (:func:`~backend.tracking.stamp.canonical_config_hash`), and ``as_of`` is
    the data version of a point-in-time read.

    Units: ``rebalance_date`` is an exchange-calendar date with no time
    component; ``as_of`` is a timezone-aware UTC instant; both counts are
    dimensionless counts of securities.
    """

    __tablename__ = "universe_snapshot"
    __table_args__ = (
        UniqueConstraint(
            "rebalance_date",
            "criteria_hash",
            "as_of",
            name="uq_universe_snapshot_build",
        ),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would double the prefix and diverge from the
        # names migration 0011 creates.
        CheckConstraint("criteria_hash <> ''", name="criteria_hash_not_empty"),
        CheckConstraint("candidate_count >= 0", name="candidate_count_non_negative"),
        CheckConstraint("member_count >= 0", name="member_count_non_negative"),
        # A member is a candidate that failed nothing, so members can never
        # outnumber the names considered. A violation here means the screen and
        # the counts disagree, which is the one arithmetic error that would make
        # every waterfall built from this row wrong.
        CheckConstraint("member_count <= candidate_count", name="members_within_candidates"),
    )

    snapshot_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless.",
    )
    rebalance_date: Mapped[dt.date] = mapped_column(
        Date,
        nullable=False,
        doc="The rebalance date this universe was built for (calendar date, no time part).",
    )
    criteria_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "SHA-256 hex digest of the canonical JSON of `criteria` "
            "(backend.universe.criteria.UniverseCriteria.criteria_hash). 64 lowercase hex "
            "characters. Two snapshots are comparable only when this matches."
        ),
    )
    criteria: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        doc=(
            "The screening criteria as canonical JSON — the preimage of criteria_hash. "
            "Stored beside the digest so a snapshot states its own screens rather than "
            "referring to a hash nobody can invert, and so the digest is verifiable."
        ),
    )
    as_of: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc=(
            "The knowledge instant the inputs were read at, UTC — the as_of bound on the "
            "session that built this snapshot (I1). This is the snapshot's data version: "
            "the same criteria at the same rebalance date read at a later as_of may give a "
            "different universe, and that difference is information, not an inconsistency."
        ),
    )
    candidate_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Securities considered — listed at the rebalance date, before any screen (count).",
    )
    member_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Securities passing every applied screen (count). Never exceeds candidate_count.",
    )
    built_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc=(
            "When the row was written, UTC, from the database clock at transaction start. "
            "Wall-clock provenance only — never a knowledge time, and never used to order "
            "snapshots (rebalance_date does that)."
        ),
    )


class UniverseMember(Base):
    """One candidate's screening outcome within one snapshot (P4.1/P4.2).

    **One row per candidate considered, not per member.** The class is named for
    the question it usually answers — "who was in the universe on this date" —
    but storing only survivors would make §6.3's filter-impact waterfall
    unreconstructible from the record, and P4.2 requires exactly that
    reconstruction. So an excluded name gets a row too, carrying
    :attr:`failed_filters`: every screen it failed, ordered by
    :data:`~backend.universe.criteria.FILTER_ORDER`. Members are the rows with
    ``included = true``, which the database also guarantees is exactly the rows
    with an empty ``failed_filters`` (the CHECK below).

    Recording *every* failure rather than only the attributed one is deliberate.
    The waterfall needs only ``failed_filters[0]`` — a name failing three screens
    is counted once, against the earliest — but the operator's real question is
    whether loosening one screen would bring a name back, and that is
    unanswerable from a single attribution. The extra failures cost a few bytes
    and cannot be recovered later.

    The foreign key to ``security`` is the identity anchor, not the bitemporal
    ``security_master``: a versioned table's logical key is not unique per row,
    so it cannot be a foreign-key target (D-011). Which *version* of the identity
    was in force is a function of the snapshot's ``rebalance_date`` and ``as_of``
    and is re-derivable through :func:`backend.db.as_of`.
    """

    __tablename__ = "universe_member"
    __table_args__ = (
        CheckConstraint(
            "included = (jsonb_array_length(failed_filters) = 0)",
            name="included_iff_no_failures",
        ),
    )

    snapshot_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("universe_snapshot.snapshot_id"),
        primary_key=True,
        doc="The snapshot this outcome belongs to.",
    )
    security_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("security.security_id"),
        primary_key=True,
        doc="The candidate, by identity-anchor key. One row per security per snapshot.",
    )
    included: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="True when the name passed every applied screen — i.e. is a universe member.",
    )
    failed_filters: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        doc=(
            "Screens this name failed, as a JSON array of names ordered by "
            "backend.universe.criteria.FILTER_ORDER. Empty exactly when included is true. "
            "The waterfall attributes the exclusion to element 0."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 11 tables (P11.2, migration 0014): the order management system
#
# Two tables, **append-only** by the same ``BEFORE UPDATE OR DELETE`` row-trigger
# shape revisions 0003/0004/0007/0009/0010/0011 use:
#
# - ``execution_order`` — the immutable content of one order, plus the
#   content-derived idempotency key that makes a retried submission a no-op. It
#   carries **no state column**, deliberately: an append-only table cannot update
#   one, and a denormalized state that could drift from the history is the exact
#   condition the transition log exists to make impossible. Current state is
#   ``backend.execution.lifecycle.replay`` over the transitions.
# - ``execution_order_transition`` — the audit trail: one row per state change,
#   in a gapless per-order sequence, carrying the fill payload on fill events and
#   nothing on any other.
#
# Neither is bitemporal, for the reason recorded above the Phase 4 tables: the
# bitemporal columns describe when a fact was true in the world and when it
# became knowable to the *market* (D-011). An order is a decision **we** took and
# an event stream **we** were sent; its point-in-time content is the ``as_of``
# already baked into the reproducibility stamp of the plan that produced it.
# Inventing a ``knowledge_time`` for it would be a fabricated value in the one
# column whose meaning is that it is not fabricated (I3).
#
# Neither is a hypertable: a daily-rebalanced book of a few hundred names
# produces thousands of orders a year, not an event-time stream worth chunking.
#
# **Paper-only is enforced here, not merely observed.** ``venue`` carries a
# server default and a ``CHECK (venue = 'paper')`` constraint, and no writer —
# ORM, raw INSERT or COPY — supplies it; ``fill_source`` is constrained to two
# non-live literals so a live execution has no representation at all; and
# ``fill_cost_basis`` is constrained to ``'lower_bound'`` so no fill can claim to
# be a calibrated estimate of cost (D-013). See ``backend/execution/orders.py``
# for the structural argument these constraints restate.
# ---------------------------------------------------------------------------


class ExecutionOrder(Base):
    """One order: its content, its idempotency key, and the stamp that produced it (P11.2).

    Every column is immutable content. The row records *what was instructed*;
    what happened to it lives entirely in :class:`ExecutionOrderTransition`, and
    the current state is derived by replaying that log rather than stored here.
    That split is what makes the append-only trigger sufficient: there is no
    field whose value legitimately changes, so a table that refuses UPDATE loses
    nothing.

    **Identity is** ``idempotency_key``, and it is UNIQUE. The key is a SHA-256
    digest of ``idempotency_preimage`` — the canonical JSON of the order's own
    content, including the four I2 stamp components — computed by
    :mod:`backend.execution.idempotency`. Two workers racing to submit the same
    order both compute the same key, and the database lets exactly one of them
    insert it. The constraint is the enforcement point; the Python code around it
    only decides what to do with the loser.

    The preimage is stored beside the digest so the digest is *verifiable*: a
    stored key that does not hash from its stored preimage, or a preimage that
    disagrees with a resubmitted order, is
    :class:`~backend.execution.errors.IdempotencyCollisionError` rather than a
    silent substitution of one trade for another.

    Units: ``quantity_shares`` is whole shares; ``limit_price_usd`` is US dollars
    per share; ``rebalance_date`` is a calendar date with no time component.
    """

    __tablename__ = "execution_order"
    __table_args__ = (
        # The enforcement point for idempotency (I2, P11.2). Not an index for
        # lookups — the lookups are a side benefit — but the constraint that
        # makes a duplicate submission fail in the database rather than in a
        # check-then-insert window no application can close.
        UniqueConstraint("idempotency_key", name="uq_execution_order_idempotency_key"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would double the prefix and diverge from the
        # names migration 0014 creates.
        #
        # Paper-only, at the database. There is no writer that supplies this
        # column and no enum member other than 'paper', so this constraint is
        # the third independent statement of the same fact (directive §1.1,
        # §9.5) and the only one that binds a writer bypassing Python entirely.
        CheckConstraint("venue = 'paper'", name="venue_is_paper"),
        CheckConstraint("quantity_shares > 0", name="quantity_shares_positive"),
        CheckConstraint("security_id > 0", name="security_id_positive"),
        CheckConstraint("side IN ('buy', 'sell')", name="side_is_known"),
        CheckConstraint("order_type IN ('market', 'limit')", name="order_type_is_known"),
        CheckConstraint("time_in_force IN ('day', 'gtc')", name="time_in_force_is_known"),
        # The type and the price together are the instruction; either half alone
        # is ambiguous. Two-sided, in the D-030 shape: a limit order without a
        # price and a market order with one are both refused, because merely
        # forbidding one direction trades a fabrication bug for a silent-absence
        # bug pointing the other way.
        CheckConstraint(
            "(order_type = 'limit') = (limit_price_usd IS NOT NULL)",
            name="limit_price_iff_limit_order",
        ),
        CheckConstraint(
            "limit_price_usd IS NULL OR limit_price_usd > 0",
            name="limit_price_positive",
        ),
        CheckConstraint(
            "slice_count >= 1 AND slice_index >= 0 AND slice_index < slice_count",
            name="slice_within_count",
        ),
        # The key must be a SHA-256 digest, not any string a caller invented:
        # a key that is not a content digest is a counter wearing a digest's
        # clothes, and a counter cannot survive the restart it exists to.
        CheckConstraint("idempotency_key ~ '^[0-9a-f]{64}$'", name="idempotency_key_is_sha256"),
        CheckConstraint("idempotency_preimage <> ''", name="idempotency_preimage_present"),
        CheckConstraint("idempotency_schema <> ''", name="idempotency_schema_present"),
        # I2: all four stamp components present on every order, so every fill
        # traces to the commit and config that produced the decision.
        CheckConstraint("git_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'", name="git_commit_is_sha"),
        CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        CheckConstraint("data_version <> ''", name="data_version_present"),
        CheckConstraint("seed >= 0", name="seed_non_negative"),
    )

    order_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless.",
    )
    idempotency_key: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "SHA-256 hex digest of idempotency_preimage (64 lowercase hex characters). "
            "UNIQUE: this is where a duplicate submission is refused."
        ),
    )
    idempotency_schema: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Version of the preimage recipe "
            "(backend.execution.idempotency.IDEMPOTENCY_SCHEMA). Recorded per row so keys "
            "computed under two recipes are distinguishable after the fact."
        ),
    )
    idempotency_preimage: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "The canonical JSON the key was hashed from — stored so the digest is "
            "verifiable, and so a resubmission whose content differs is detected rather "
            "than absorbed as the same trade."
        ),
    )
    venue: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'paper'"),
        doc=(
            "Execution venue. Always 'paper', by CHECK constraint and by there being no "
            "other member of backend.execution.orders.ExecutionVenue. No writer supplies "
            "it; the server default does. Directive §1.1 and §9.5: live trading is not "
            "configurable, so a non-paper value is a corruption, not a setting."
        ),
    )
    security_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("security.security_id"),
        nullable=False,
        doc=(
            "The security to trade, by identity-anchor key. Not the bitemporal master: a "
            "versioned table's logical key is not unique per row and cannot be an FK target."
        ),
    )
    side: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'buy' or 'sell'. Direction lives here, never in the sign of the quantity.",
    )
    quantity_shares: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="Shares to trade (whole shares), strictly positive.",
    )
    order_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'market' or 'limit'.",
    )
    time_in_force: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'day' or 'gtc'. Intraday instruments (IOC/FOK) are out of scope (§1.1).",
    )
    limit_price_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
        doc=(
            "Limit price in US dollars per share, present exactly when order_type = 'limit' "
            "(CHECK, both directions). Numeric rather than float: an order's identity is "
            "hashed from its text, and a binary float has no exact decimal rendering."
        ),
    )
    rebalance_date: Mapped[dt.date] = mapped_column(
        Date,
        nullable=False,
        doc=(
            "The rebalance this order implements (calendar date, no time part). Part of the "
            "order's hashed identity: the same target position on two dates is two orders."
        ),
    )
    slice_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        doc=(
            "Zero-based slice index within a parent order; 0 when unsliced. Part of the "
            "hashed identity, so a TWAP's slices are distinct orders rather than one order "
            "submitted repeatedly (P11.4)."
        ),
    )
    slice_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
        doc="Number of slices the parent was divided into; 1 when unsliced. Dimensionless.",
    )
    git_commit: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: full commit SHA of the code that produced this order.",
    )
    git_dirty: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc=(
            "I2: whether the working tree differed from HEAD when the stamp was taken. True "
            "means the order is not regenerable from git_commit alone, and the row says so "
            "rather than recording the commit it was nearly produced from."
        ),
    )
    data_version: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: identifier of the data snapshot the decision read.",
    )
    config_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: SHA-256 hex digest of the canonical config (64 lowercase hex characters).",
    )
    seed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="I2: the random seed the producing run used, recorded verbatim. Dimensionless.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc=(
            "When the row was written, UTC, from the database clock at transaction start. "
            "Wall-clock provenance only — never a knowledge time, and never an ordering key "
            "(order_id is). Deliberately absent from the idempotency preimage: a key that "
            "moved with the clock would make every retry a new order."
        ),
    )


class ExecutionOrderTransition(Base):
    """One recorded state change of one order — the append-only audit trail (P11.2).

    The order's whole history, one row per event, in a gapless per-order sequence
    starting at 1. Current state is the fold of this log
    (:func:`backend.execution.lifecycle.replay`), never a stored field, so there
    is no denormalized value that can come to disagree with the history that
    justifies it.

    **The primary key is the concurrency control.** ``(order_id,
    sequence_number)`` is unique, and each writer computes ``last + 1`` from the
    tail it read. Two writers that read the same tail try to claim the same
    sequence number and the database rejects the loser
    (:class:`~backend.execution.errors.ConcurrentTransitionError`), which is a
    retryable condition rather than corruption: the loser re-reads, and its event
    may or may not still be legal from the state that now holds. Optimistic
    rather than a lock because the contended case is rare and a lock held across
    a broker round trip is how an OMS deadlocks itself.

    **Illegal transitions are refused by the database, not only by Python.** Two
    constraints restate
    :data:`backend.execution.lifecycle.TRANSITIONS`: ``from_state_not_terminal``
    bans any transition out of a terminal state — this is what makes ``FILLED ->
    PENDING_NEW`` unrepresentable — and ``legal_transition`` enumerates the 25
    legal ``(from_state, event, to_state)`` triples. A third, the
    ``execution_transition_chain_guard`` trigger (migration 0014), checks each
    row against its predecessor: the sequence is gapless, the chain connects, and
    the cumulative fill quantity is the running sum and never exceeds the order's
    quantity.

    **Fill payload columns follow D-030's two-sided rule.** They are present
    exactly on fill events and NULL on every other, enforced in both directions:
    a non-fill row carrying a quantity would be a trade nobody reported, and a
    fill row missing one would be a trade whose size is a gap. ``fill_source``
    admits only non-live values (I3) and ``fill_cost_basis`` only
    ``'lower_bound'`` (D-013).

    Units: quantities are whole shares; ``fill_price_usd`` is US dollars per
    share.
    """

    __tablename__ = "execution_order_transition"
    __table_args__ = (
        CheckConstraint("sequence_number >= 1", name="sequence_number_positive"),
        # The named property this whole table exists to guarantee. Implied by
        # legal_transition below, and stated separately anyway: it is the one
        # constraint whose violation produces a phantom position, and a reader
        # grepping for it should find it by name.
        CheckConstraint(
            "from_state NOT IN ('filled', 'cancelled', 'rejected', 'expired')",
            name="from_state_not_terminal",
        ),
        # The 25 legal triples of backend.execution.lifecycle.TRANSITIONS,
        # restated in SQL. A test compares the two so they cannot drift.
        CheckConstraint(
            "(from_state, event, to_state) IN ("
            "('draft', 'release', 'pending_new'), "
            "('draft', 'abandon', 'cancelled'), "
            "('pending_new', 'acknowledge', 'acknowledged'), "
            "('pending_new', 'reject', 'rejected'), "
            "('pending_new', 'expire', 'expired'), "
            "('acknowledged', 'partial_fill', 'partially_filled'), "
            "('acknowledged', 'fill_complete', 'filled'), "
            "('acknowledged', 'request_cancel', 'pending_cancel'), "
            "('acknowledged', 'expire', 'expired'), "
            "('acknowledged', 'venue_cancel', 'cancelled'), "
            "('partially_filled', 'partial_fill', 'partially_filled'), "
            "('partially_filled', 'fill_complete', 'filled'), "
            "('partially_filled', 'request_cancel', 'pending_cancel_partial'), "
            "('partially_filled', 'expire', 'expired'), "
            "('partially_filled', 'venue_cancel', 'cancelled'), "
            "('pending_cancel', 'cancel_confirmed', 'cancelled'), "
            "('pending_cancel', 'cancel_rejected', 'acknowledged'), "
            "('pending_cancel', 'partial_fill', 'pending_cancel_partial'), "
            "('pending_cancel', 'fill_complete', 'filled'), "
            "('pending_cancel', 'expire', 'expired'), "
            "('pending_cancel_partial', 'cancel_confirmed', 'cancelled'), "
            "('pending_cancel_partial', 'cancel_rejected', 'partially_filled'), "
            "('pending_cancel_partial', 'partial_fill', 'pending_cancel_partial'), "
            "('pending_cancel_partial', 'fill_complete', 'filled'), "
            "('pending_cancel_partial', 'expire', 'expired'))",
            name="legal_transition",
        ),
        CheckConstraint(
            "filled_quantity_after_shares >= 0",
            name="filled_after_non_negative",
        ),
        # D-030 shape, both directions. Present on a fill:
        CheckConstraint(
            "event NOT IN ('partial_fill', 'fill_complete') OR ("
            "fill_quantity_shares IS NOT NULL AND fill_price_usd IS NOT NULL "
            "AND fill_source IS NOT NULL AND fill_cost_basis IS NOT NULL)",
            name="fill_payload_present",
        ),
        # ...and absent on everything else, so a non-fill row cannot carry a
        # quantity or a price that no venue ever reported.
        CheckConstraint(
            "event IN ('partial_fill', 'fill_complete') OR ("
            "fill_quantity_shares IS NULL AND fill_price_usd IS NULL "
            "AND fill_source IS NULL AND fill_cost_basis IS NULL "
            "AND venue_fill_id IS NULL)",
            name="fill_payload_absent",
        ),
        CheckConstraint(
            "fill_quantity_shares IS NULL OR fill_quantity_shares > 0",
            name="fill_quantity_positive",
        ),
        CheckConstraint(
            "fill_price_usd IS NULL OR fill_price_usd > 0",
            name="fill_price_positive",
        ),
        # I3: a live fill has no representation. Not a disabled option — an
        # absent one. Adding a third value needs a migration, not a config edit.
        CheckConstraint(
            "fill_source IS NULL OR fill_source IN ('simulated', 'paper_broker')",
            name="fill_source_is_not_live",
        ),
        # D-013: paper fills bound slippage from below. A row claiming any other
        # basis would let a later calibration treat an optimistic fill as a
        # measured estimate, which is the single easiest way to turn a losing
        # strategy into a winning backtest.
        CheckConstraint(
            "fill_cost_basis IS NULL OR fill_cost_basis = 'lower_bound'",
            name="fill_cost_basis_is_lower_bound",
        ),
        CheckConstraint("venue_fill_id IS NULL OR venue_fill_id <> ''", name="venue_fill_id_set"),
    )

    order_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("execution_order.order_id"),
        primary_key=True,
        doc="The order whose history this row extends.",
    )
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        doc=(
            "Position in this order's history, 1-based and gapless. The ordering key — "
            "never a timestamp — and, with order_id, the uniqueness that turns two "
            "concurrent writers into one winner and one retryable loser."
        ),
    )
    from_state: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The state the order was in. Never a terminal state (CHECK).",
    )
    event: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The event applied (backend.execution.lifecycle.OrderEvent).",
    )
    to_state: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The resulting state. The (from_state, event, to_state) triple is CHECKed.",
    )
    filled_quantity_after_shares: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc=(
            "Cumulative shares filled after this event (whole shares). Stored rather than "
            "recomputed so a truncated history is detectable instead of merely shorter; the "
            "chain-guard trigger checks it against the previous row's value plus this row's "
            "fill quantity."
        ),
    )
    fill_quantity_shares: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        doc=(
            "Shares traded by this event (whole shares); NULL on every event that is not a "
            "fill. Absent rather than zero: a zero is a quantity nobody reported (D-030)."
        ),
    )
    fill_price_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
        doc="Price traded at, US dollars per share; NULL on every non-fill event.",
    )
    fill_source: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "'simulated' or 'paper_broker'; NULL on every non-fill event. There is no value "
            "denoting a live execution (I3): a simulated fill and a paper-broker fill are "
            "different values on the row, so neither can be mistaken for the other and "
            "neither can be mistaken for a real one."
        ),
    )
    fill_cost_basis: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "Always 'lower_bound' when present, NULL on every non-fill event (D-013). Paper "
            "and simulated fills are optimistic — they fill at the touch and model no queue "
            "position — so slippage measured from them bounds the true cost from below and "
            "is never an estimate of it. The label travels on the row so the qualification "
            "reaches every query, export and blotter."
        ),
    )
    venue_fill_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "The venue's identifier for this execution, when it gave one; NULL when it did "
            "not, and NULL on every non-fill event. Absent rather than invented (I3)."
        ),
    )
    occurred_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc=(
            "When the event happened at its origin, UTC: the venue's timestamp for a "
            "venue-reported event, ours for a locally-originated one. Never an ordering key "
            "— sequence_number is — because venue clocks and ours disagree and the "
            "disagreement is information, not noise to be sorted away."
        ),
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc=(
            "When the row was written, UTC, from the database clock at transaction start. "
            "Wall-clock provenance only; the gap between this and occurred_at is the "
            "system's own latency, which P11.3 reads and nothing else should."
        ),
    )
    note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "Free text from the caller — a venue reject reason, an operator's justification "
            "for a manual cancel. Never parsed; the structured meaning is in event."
        ),
    )


class LlmSpendLedger(Base):
    """One spend event against one provider's cap (P7.7, §6.5, I4).

    **An event log, not a running total.** Three kinds of row —
    ``reserved``, ``released``, ``settled`` — tied together by
    ``reservation_id``, each carrying a signed :attr:`delta_amount`, so a
    window's committed spend is ``SUM(delta_amount)`` over that window's rows.
    Nothing is ever updated, which is what makes this an audit trail rather
    than a number whose history has been overwritten (§6.11: configuration
    changes are events, not mutations — spend is no different). Migration 0013
    installs the same ``BEFORE UPDATE OR DELETE`` trigger the rest of this
    schema uses.

    Why an event log rather than a balance: the cap is enforced **before** the
    call, against an *upper bound* on what the call will cost, and reconciled
    afterwards against what it actually cost. Both figures are worth keeping.
    A balance would keep neither, and "how wrong was the estimate" — the
    question that says whether the bound is doing its job or quietly strangling
    a budget — would be unanswerable.

    Why :attr:`served_model` exists beside :attr:`requested_model`: §6.5 makes
    the ceiling behaviour configurable between halting and degrading to a
    cheaper model, and a degraded call is answered by a model the caller did
    not ask for. A result attributed to a model that never read the document is
    not reproducible from its recorded configuration (I2), so both models are
    recorded on every row and the substitution is visible as
    ``requested_model <> served_model`` — which the ``degraded_iff_substituted``
    CHECK ties to the flag rather than leaving the two free to disagree.

    Not bitemporal, for the same reason ``llm_provider_credential`` and
    ``config_change_event`` are not: the bitemporal columns describe when a fact
    was true in the world and when it became knowable to the *market* (D-011).
    Spending our own money on our own extraction has no market knowability, and
    a ``knowledge_time`` invented for it would be a fabricated value in the one
    column whose meaning is that it is not fabricated (I3).

    Units and assumptions:

    - every monetary column is ``NUMERIC(20, 10)`` in the **major unit** of the
      currency named by :attr:`currency` (dollars, not cents), at ten decimal
      places because cost-tier models are priced in fractions of a cent per
      call and a two-decimal column would round a real call to zero;
    - :attr:`currency` is an ISO-4217 alphabetic code. There are no exchange
      rates in this system, so rows in different currencies are never summed
      together — a cap and the prices checked against it must agree;
    - token columns are dimensionless counts. The ``_bound`` pair is the
      **upper bound** the estimate used; :attr:`input_tokens` and
      :attr:`output_tokens` are counts **as reported by the provider** and are
      NULL when it reported none, never estimated (I3);
    - :attr:`daily_window` is a UTC calendar day (``YYYY-MM-DD``) and
      :attr:`monthly_window` a UTC calendar month (``YYYY-MM``). UTC because it
      is the only clock this platform stores anything in — not because it is
      any vendor's billing day.
    """

    __tablename__ = "llm_spend_ledger"
    __table_args__ = (
        # One settlement and one release per reservation, enforced by the
        # database rather than by the application's memory: double-counting a
        # delta corrupts every later cap check, and the check that would catch
        # it in Python runs in a different transaction.
        UniqueConstraint("reservation_id", "event", name="uq_llm_spend_ledger_reservation_event"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names migration 0013 creates.
        CheckConstraint(f"provider IN ({PROVIDER_NAMES_SQL})", name="provider_known"),
        CheckConstraint("event IN ('reserved', 'released', 'settled')", name="event_known"),
        CheckConstraint("policy IN ('halt', 'degrade')", name="policy_known"),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('succeeded', 'failed')", name="outcome_known"
        ),
        CheckConstraint("reservation_id <> ''", name="reservation_id_not_empty"),
        CheckConstraint("requested_model <> ''", name="requested_model_not_empty"),
        CheckConstraint("served_model <> ''", name="served_model_not_empty"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="currency_is_iso4217_alpha"),
        CheckConstraint("daily_window ~ '^\\d{4}-\\d{2}-\\d{2}$'", name="daily_window_shape"),
        CheckConstraint("monthly_window ~ '^\\d{4}-\\d{2}$'", name="monthly_window_shape"),
        # A cost is never negative; the *delta* is signed, because a settlement
        # normally hands headroom back.
        CheckConstraint("estimated_cost >= 0", name="estimated_cost_non_negative"),
        CheckConstraint("actual_cost IS NULL OR actual_cost >= 0", name="actual_cost_non_negative"),
        CheckConstraint("daily_limit >= 0", name="daily_limit_non_negative"),
        CheckConstraint("monthly_limit >= 0", name="monthly_limit_non_negative"),
        CheckConstraint("input_tokens_bound >= 0", name="input_tokens_bound_non_negative"),
        CheckConstraint("output_tokens_bound >= 1", name="output_tokens_bound_positive"),
        CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_counted"),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0", name="output_tokens_counted"
        ),
        # The event algebra, asserted in the schema so a row cannot claim an
        # arithmetic the ledger does not do:
        #   reserved  -> delta = +estimate
        #   released  -> delta = -estimate
        #   settled   -> delta = coalesce(actual, estimate) - estimate
        CheckConstraint(
            "(event = 'reserved' AND delta_amount = estimated_cost) "
            "OR (event = 'released' AND delta_amount = -estimated_cost) "
            "OR (event = 'settled' "
            "    AND delta_amount = COALESCE(actual_cost, estimated_cost) - estimated_cost)",
            name="delta_matches_event",
        ),
        # An outcome belongs to a completed call and to nothing else.
        CheckConstraint("(outcome IS NOT NULL) = (event = 'settled')", name="outcome_iff_settled"),
        # "Reconciled" means the cost is a measurement rather than the bound.
        CheckConstraint("reconciled = (actual_cost IS NOT NULL)", name="reconciled_iff_measured"),
        # A reservation and a release carry no provider-reported counts: nothing
        # was reported yet, and inventing zeros would read as a free call.
        CheckConstraint(
            "event = 'settled' OR (input_tokens IS NULL AND output_tokens IS NULL)",
            name="tokens_only_on_settlement",
        ),
        CheckConstraint(
            "degraded = (requested_model <> served_model)", name="degraded_iff_substituted"
        ),
    )

    ledger_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; also the append order.",
    )
    reservation_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "32-character hex UUID4 tying one call's reserved row to its settled or released "
            "row. Generated in the process, before the row exists, because a reservation is "
            "referred to across the provider call that sits between the two writes."
        ),
    )
    event: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'reserved', 'released' or 'settled' — see the class docstring's event algebra.",
    )
    provider: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Provider whose cap this row moves, e.g. 'anthropic'. Not an FK — a spend record "
        "outlives the credential, and deleting a key must not erase what it spent.",
    )
    requested_model: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Qualified model identifier the caller asked for, e.g. 'anthropic:some-model'.",
    )
    served_model: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Qualified model that actually answered. Differs from requested_model exactly when "
            "the cap degraded the call. This is the column that makes a degraded result "
            "reproducible (I2)."
        ),
    )
    currency: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="ISO-4217 alphabetic code every monetary column on this row is denominated in.",
    )
    estimated_cost: Mapped[Decimal] = mapped_column(
        Numeric(20, 10),
        nullable=False,
        doc=(
            "The UPPER BOUND on this call's cost, in the currency's major unit — max output "
            "tokens at the model's output rate plus a bound on the prompt at its input rate. "
            "Not an expectation: the cap is checked before the response exists, and an "
            "expectation would leak by exactly the amount it was optimistic."
        ),
    )
    actual_cost: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 10),
        nullable=True,
        doc=(
            "What the call really cost, from provider-reported token counts, in the currency's "
            "major unit. NULL on reserved and released rows, and on a settlement where the "
            "provider reported no counts — that row settles at the bound rather than at a "
            "re-estimate, because an estimated actual is a fabricated measurement (I3)."
        ),
    )
    delta_amount: Mapped[Decimal] = mapped_column(
        Numeric(20, 10),
        nullable=False,
        doc=(
            "This row's SIGNED contribution to its windows' committed spend, in the currency's "
            "major unit. Committed spend is SUM(delta_amount) over the window; a settlement's "
            "delta is normally negative, handing back the difference between the bound and the "
            "truth."
        ),
    )
    input_tokens_bound: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Upper bound on prompt tokens used by the estimate (count). Never a measurement.",
    )
    output_tokens_bound: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc=(
            "Upper bound on response tokens used by the estimate (count) — the request's "
            "max_tokens, which the provider cannot exceed, so this bound is exact."
        ),
    )
    input_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Prompt tokens AS REPORTED by the provider (count), or NULL. Never estimated (I3).",
    )
    output_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Response tokens as reported (count), or NULL. Never estimated (I3).",
    )
    daily_window: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="UTC calendar day this row is booked to, 'YYYY-MM-DD'. Not any vendor's billing day.",
    )
    monthly_window: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="UTC calendar month this row is booked to, 'YYYY-MM'.",
    )
    daily_limit: Mapped[Decimal] = mapped_column(
        Numeric(20, 10),
        nullable=False,
        doc=(
            "The daily cap in force when this row was written, in the currency's major unit. "
            "Copied onto the row rather than joined at read time: a cap raised this afternoon "
            "must not rewrite this morning's audit trail."
        ),
    )
    monthly_limit: Mapped[Decimal] = mapped_column(
        Numeric(20, 10),
        nullable=False,
        doc="The monthly cap in force when this row was written, same units and same reasoning.",
    )
    policy: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'halt' or 'degrade' — the ceiling behaviour in force for this provider (§6.5).",
    )
    degraded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc=(
            "True when a cheaper model was substituted. Redundant with "
            "requested_model <> served_model and tied to it by CHECK, because this is the "
            "column an operator filters on and a derived predicate is easy to get wrong in a "
            "hand-written query."
        ),
    )
    outcome: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "'succeeded' or 'failed' on a settled row, NULL otherwise. A failed call still "
            "settles — at its bound — because nothing observable says whether the request left "
            "the host, and over-counting is the safe direction for a spend control."
        ),
    )
    reconciled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc=(
            "True exactly when actual_cost is a measurement. False means this row settled at "
            "its upper bound, which over-counts deliberately."
        ),
    )
    correlation_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "Request id (D-003) the call was made under, or NULL outside a request (a Celery "
            "worker, a backfill script). What joins a spend row to the extraction it paid for."
        ),
    )
    occurred_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc=(
            "The UTC instant the window keys were computed from. The reserved row's value is "
            "the authorization instant; a settlement carries its own. Never an ordering key — "
            "ledger_id is."
        ),
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Wall-clock provenance only.",
    )


# ---------------------------------------------------------------------------
# Phase 11 tables (P11.3 and P11.5, migration 0016): reconciliation and the halt log
#
# Both are append-only for the same reason ``execution_order`` is: a
# reconciliation records what two snapshots said at a moment, and a halt records
# something that happened. Neither stops being true later. A correction is a new
# reconciliation and a re-halt is a new engagement; erasing either would destroy
# the trail an operator has to reason backwards through.
# ---------------------------------------------------------------------------


class ExecutionReconciliation(Base):
    """One cycle's comparison of our book against a statement (P11.3).

    The row is not a summary of a comparison — it is the comparison's **inputs and
    its output together**. Both snapshots are stored in full beside their digests,
    which is what makes a break investigable after the fact:
    :func:`backend.execution.reconciliation.rerun` rebuilds the snapshots from
    these payloads, re-derives the verdict, and refuses if the digest has moved.
    A row holding only "3 breaks found" would be a claim nobody could check.

    Each stored payload is exactly the preimage of its stored digest (canonical
    JSON, keys sorted at every level), so the two cannot drift apart: a digest
    that does not hash from the payload beside it is detectable rather than
    assumed.

    ``result_digest`` deliberately does **not** cover the I2 stamp. A stored
    verdict must re-derive identically when re-run at a later commit; if the stamp
    were in the digest, every re-run would differ by construction and
    re-runnability could not be tested. The stamp lives in its own columns.

    ``cash_tolerance_usd`` is copied onto the row rather than read from code at
    query time, for the same reason ``llm_spend_ledger`` copies its caps: a
    tolerance widened next month must not rewrite last month's verdict.

    Units: ``cash_tolerance_usd`` is US dollars; share counts inside the snapshot
    payloads are signed whole shares; the two ``*_observed_at`` columns are
    timezone-aware instants.
    """

    __tablename__ = "execution_reconciliation"
    __table_args__ = (
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would double the prefix and diverge from the
        # names migration 0016 creates.
        CheckConstraint("cycle_id <> ''", name="cycle_id_present"),
        # Our own ledger cannot stand in for the statement it is checked against:
        # a reconciliation of a snapshot with itself always passes and proves
        # nothing, which is the most comfortable way for this control to become
        # decorative.
        CheckConstraint("internal_origin = 'internal_ledger'", name="internal_origin_is_ledger"),
        # I3, restated in SQL: a statement from a real-money account has no
        # representation here. A third value needs a migration, not a config edit.
        CheckConstraint(
            "reported_origin IN ('paper_broker', 'simulated')",
            name="reported_origin_is_not_live",
        ),
        CheckConstraint("internal_digest ~ '^[0-9a-f]{64}$'", name="internal_digest_is_sha256"),
        CheckConstraint("reported_digest ~ '^[0-9a-f]{64}$'", name="reported_digest_is_sha256"),
        CheckConstraint("result_digest ~ '^[0-9a-f]{64}$'", name="result_digest_is_sha256"),
        # The ceiling from backend.execution.reconciliation.MAX_CASH_TOLERANCE_USD,
        # restated where no Python can waive it. Above five cents a "tolerance"
        # absorbs an event rather than a quantisation step between a two-decimal
        # statement and a six-decimal ledger, and no magnitude of event is
        # rounding. Zero is allowed: tightening is always permitted.
        CheckConstraint(
            "cash_tolerance_usd >= 0 AND cash_tolerance_usd <= 0.05",
            name="tolerance_within_ceiling",
        ),
        CheckConstraint(
            "finding_count >= 0 AND break_count >= 0 AND break_count <= finding_count",
            name="counts_consistent",
        ),
        # The verdict is a function of the findings, not an independent opinion.
        CheckConstraint("matched = (break_count = 0)", name="matched_iff_no_breaks"),
        # I2: all four stamp components on every verdict, so a break traces to the
        # commit and config that produced it.
        CheckConstraint("git_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'", name="git_commit_is_sha"),
        CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        CheckConstraint("data_version <> ''", name="data_version_present"),
        CheckConstraint("seed >= 0", name="seed_non_negative"),
    )

    reconciliation_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; also the append order.",
    )
    cycle_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "The execution cycle this verdict judges. A verdict about another cycle is not "
            "evidence about this one, and the kill switch refuses to accept one."
        ),
    )
    internal_origin: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Always 'internal_ledger' (CHECK). The side derived from our own records.",
    )
    reported_origin: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "'paper_broker' or 'simulated' (CHECK) — see "
            "backend.execution.reconciliation.SnapshotOrigin. There is no member for a "
            "real-money account, so a fabricated statement is a distinct value on the row "
            "rather than something a reader has to infer (I3)."
        ),
    )
    internal_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        doc=(
            "Our snapshot as canonical JSON — the preimage of internal_digest. Stored in full "
            "so the verdict can be re-derived from the row alone rather than from a hash "
            "nobody can invert."
        ),
    )
    internal_digest: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="SHA-256 hex digest of internal_snapshot (64 lowercase hex characters).",
    )
    reported_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        doc="The statement snapshot as canonical JSON — the preimage of reported_digest.",
    )
    reported_digest: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="SHA-256 hex digest of reported_snapshot (64 lowercase hex characters).",
    )
    internal_observed_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc="When our book looked like internal_snapshot, UTC. Never an ordering key.",
    )
    reported_observed_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc=(
            "When the statement was observed, UTC. Reconciliation refuses two snapshots more "
            "than MAX_SNAPSHOT_SKEW_SECONDS apart: they describe two different books, and any "
            "difference between them is explained by the activity in the gap."
        ),
    )
    cash_tolerance_usd: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        doc=(
            "The allowed absolute cash divergence in US dollars, as it stood when this verdict "
            "was taken. One cent by default: it bounds one quantisation step between a "
            "two-decimal statement and a six-decimal ledger, and bounds nothing else — the "
            "cheapest real break is larger by orders of magnitude."
        ),
    )
    findings: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB,
        nullable=False,
        doc=(
            "Every finding in the verdict's fixed order, as canonical JSON. Includes "
            "observations that are not breaks — notably a position both sides call flat where "
            "one said so and the other was silent, which is recorded precisely because a "
            "missing position and a zero position are different facts."
        ),
    )
    finding_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Number of findings (count). Stored so a truncated payload is detectable.",
    )
    break_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Number of findings whose severity is 'break' (count). Never exceeds finding_count.",
    )
    matched: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="True exactly when break_count is zero (CHECK). Observations do not prevent a match.",
    )
    result_digest: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "SHA-256 hex digest over the cycle, the tolerance, both snapshot digests and every "
            "finding — and deliberately NOT over the I2 stamp, so a re-run at a later commit "
            "must reproduce it exactly. That equality is the re-runnability guarantee."
        ),
    )
    git_commit: Mapped[str] = mapped_column(
        Text, nullable=False, doc="I2: full commit SHA of the run that produced the verdict."
    )
    git_dirty: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="I2: whether the working tree differed from HEAD, counting untracked files.",
    )
    data_version: Mapped[str] = mapped_column(
        Text, nullable=False, doc="I2: identifier of the data snapshot used."
    )
    config_hash: Mapped[str] = mapped_column(
        Text, nullable=False, doc="I2: SHA-256 canonical config hash (64 lowercase hex)."
    )
    seed: Mapped[int] = mapped_column(
        BigInteger, nullable=False, doc="I2: the random seed, recorded verbatim. Dimensionless."
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Wall-clock provenance only.",
    )


class ExecutionHalt(Base):
    """The halt log: engagements and their clearances, append-only (P11.5).

    **A halt is a row, not a flag.** A halt held in memory evaporates on restart,
    and it evaporates at exactly the moment it matters — the process that halted
    crashes, the supervisor restarts it, and the new worker begins its next cycle
    believing nothing is wrong while the condition is still true and now
    unobserved. Nothing in :mod:`backend.execution` caches this state; the current
    state is a fold over these rows, computed on every read.

    An ``engaged`` row is open until a ``cleared`` row names it in
    ``clears_halt_id``. There is no timeout, no automatic re-arm, and no path by
    which the condition going away clears the halt: a halt outlives its cause,
    because the point of halting is that a human looks. The clearance carries who
    and why, both non-blank by CHECK, so the log answers "who turned this back on".

    Two refusals live here rather than in Python — ``UNIQUE (clears_halt_id)`` so a
    halt is cleared at most once, and the ``execution_halt_clearance_guard``
    trigger so a clearance can only name an engagement. **Neither constrains an
    engagement**, deliberately: a constraint that can reject a halt-engage row is
    a constraint that can stop the kill switch from firing, and a redundant halt
    row costs nothing while a refused one costs everything.

    ``cycle_id`` is what makes the directive's "halts within one cycle" auditable
    after the fact: the halt written by the cycle that observed the condition
    carries that cycle's own id, so the log itself shows that no cycle elapsed in
    between.
    """

    __tablename__ = "execution_halt"
    __table_args__ = (
        # A halt is cleared at most once. NULLs are distinct in Postgres, so this
        # leaves engagements — which all carry NULL here — completely unconstrained.
        UniqueConstraint("clears_halt_id", name="uq_execution_halt_clears_halt_id"),
        CheckConstraint("event IN ('engaged', 'cleared')", name="event_is_known"),
        # D-030 shape, both directions: the trigger names the condition on an
        # engagement and is absent on a clearance. A clearance carrying a trigger
        # would read as a second halt.
        CheckConstraint(
            "(event = 'engaged') = (halt_trigger IS NOT NULL)", name="trigger_iff_engaged"
        ),
        CheckConstraint(
            "halt_trigger IS NULL OR halt_trigger IN ("
            "'drawdown_breach', 'stale_data', 'reconciliation_mismatch', 'manual', "
            "'unknown_condition')",
            name="trigger_is_known",
        ),
        # The three clearance columns are present exactly on a clearance —
        # stated **per column**, which is what the first version got wrong.
        #
        # It originally read `(event = 'cleared') = (a IS NOT NULL AND b IS NOT
        # NULL AND c IS NOT NULL)`, which only forbids an engagement carrying all
        # three. Any proper subset was accepted, and one of those subsets is
        # dangerous: `uq_execution_halt_clears_halt_id` is on the column
        # unconditionally, so an engagement carrying a stray clears_halt_id
        # consumes the unique slot for that halt. The genuine clearance is then
        # refused with SQLSTATE 23505 and surfaces as HaltAlreadyClearedError —
        # telling an operator the halt "has already been cleared" when it has
        # not, while open_halts (which counts clearances only where
        # event = 'cleared') keeps reporting it open. The halt becomes
        # permanently un-clearable and the error actively misdescribes why.
        #
        # The clearance guard trigger cannot catch it either: it returns
        # immediately for a non-clearance, deliberately, because nothing may
        # refuse an engagement.
        #
        # Same D-030 shape `trigger_iff_engaged` already uses for halt_trigger.
        CheckConstraint(
            "(event = 'cleared') = (clears_halt_id IS NOT NULL) "
            "AND (event = 'cleared') = (cleared_by IS NOT NULL) "
            "AND (event = 'cleared') = (clearance_reason IS NOT NULL)",
            name="clearance_fields_iff_cleared",
        ),
        CheckConstraint("cleared_by IS NULL OR cleared_by <> ''", name="cleared_by_present"),
        CheckConstraint(
            "clearance_reason IS NULL OR clearance_reason <> ''", name="clearance_reason_present"
        ),
        CheckConstraint("cycle_id <> ''", name="cycle_id_present"),
        CheckConstraint("detail <> ''", name="detail_present"),
        # I2 on every halt: which commit, config, data version and seed observed
        # the condition, and which observed the clearance.
        CheckConstraint("git_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'", name="git_commit_is_sha"),
        CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        CheckConstraint("data_version <> ''", name="data_version_present"),
        CheckConstraint("seed >= 0", name="seed_non_negative"),
    )

    halt_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. The id a clearance must name.",
    )
    event: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'engaged' or 'cleared'. There is no third event: a halt is never amended in place.",
    )
    halt_trigger: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "Which condition fired, on an engagement; NULL on a clearance (CHECK, both "
            "directions). One of the four the directive names plus 'unknown_condition', which "
            "exists because a kill switch whose trigger list is exhaustive fails open on "
            "everything not on the list."
        ),
    )
    cycle_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "The execution cycle this row belongs to. On an engagement it is the cycle that "
            "observed the condition — the column that makes 'halts within one cycle' "
            "checkable from the log rather than only at the moment it happened."
        ),
    )
    detail: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Prose naming the condition or the clearance. Non-blank by CHECK.",
    )
    evidence: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        doc=(
            "The measurements behind the decision, as JSON: the drawdown and the limit it "
            "breached, the data age and the age allowed, the digests of the reconciliation "
            "that failed. Structured rather than prose because a halt whose evidence is a "
            "sentence cannot be audited without re-running what produced it."
        ),
    )
    clears_halt_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("execution_halt.halt_id"),
        nullable=True,
        doc=(
            "On a clearance, the engagement being cleared; NULL on an engagement. UNIQUE, so a "
            "halt is cleared at most once and the log never holds two answers to 'who turned "
            "it back on'."
        ),
    )
    cleared_by: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Who cleared the halt. Non-blank when present: a halt is cleared by a person.",
    )
    clearance_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "Why it was cleared. Non-blank when present — 'the break was a stale statement' "
            "and 'the alarm was inconvenient' are both reasons, and a log that cannot tell "
            "them apart is worthless."
        ),
    )
    occurred_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc=(
            "When the condition was observed, or the clearance decided. UTC. Never an "
            "ordering key — halt_id is."
        ),
    )
    git_commit: Mapped[str] = mapped_column(
        Text, nullable=False, doc="I2: full commit SHA of the run that wrote this row."
    )
    git_dirty: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="I2: whether the working tree differed from HEAD, counting untracked files.",
    )
    data_version: Mapped[str] = mapped_column(
        Text, nullable=False, doc="I2: identifier of the data snapshot in use."
    )
    config_hash: Mapped[str] = mapped_column(
        Text, nullable=False, doc="I2: SHA-256 canonical config hash (64 lowercase hex)."
    )
    seed: Mapped[int] = mapped_column(
        BigInteger, nullable=False, doc="I2: the random seed, recorded verbatim. Dimensionless."
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Wall-clock provenance only.",
    )


class MonitoringAlert(Base):
    """One alert condition, recorded before anyone is told about it (P12.4).

    **The row is written before delivery is attempted.** If the process dies
    mid-dispatch the alert exists and the missing delivery rows say nobody was
    told; the reverse order loses the alert entirely on the same failure. That
    ordering is the whole reason this table exists rather than a log line.

    **Identity is the condition, not the occurrence.** :attr:`dedup_key` is a
    SHA-256 over the rule and the condition — the date and the specific thing
    that was wrong — and carries a ``UNIQUE`` constraint. A monitoring job that
    runs hourly through an incident re-derives the same key every hour and the
    database absorbs the repeats, so one condition produces one alert and one
    acknowledgement. Nothing that changes between two evaluations of the same
    condition (a timestamp, a counter, a UUID) is in the digest, for D-033's
    reason: a counter has to be *remembered* across a restart, and the worker
    that restarted mints a new one and pages someone twice.

    **No delivered/acknowledged flags.** Both are derived from
    ``monitoring_alert_delivery`` and ``monitoring_alert_acknowledgement``. A
    denormalised flag on an append-only table cannot be updated, and one that
    drifts from the rows it summarises is exactly the condition an audit trail
    exists to prevent (same reasoning as ``execution_order``'s absent state
    column, D-033).

    Not bitemporal, matching revisions 0005, 0007, 0009-0011 and 0014: the
    bitemporal columns describe when a fact was true in the world and when it
    became knowable to the *market* (D-011). An alert is something **we**
    observed about our own system; a ``knowledge_time`` invented for it would be
    a fabricated value in the one column whose meaning is that it is not
    fabricated (I3).

    Units and assumptions:

    - :attr:`severity` is ``'info'``, ``'warning'`` or ``'critical'``, bound by
      CHECK, matching :class:`backend.monitoring.alerts.AlertSeverity`;
    - :attr:`raised_at` is when the *condition* was observed and
      :attr:`recorded_at` is when the row was written; they differ when a
      monitoring run is replayed, and the first is the one an incident timeline
      is built from;
    - the four I2 stamp columns are the run that raised the alert, not the run
      that stored it.
    """

    __tablename__ = "monitoring_alert"
    __table_args__ = (
        # One row per condition. Enforced by the database rather than by a
        # look-then-insert in Python: two monitoring workers can both pass
        # through the window between a check and an insert, and both would page
        # the operator for one condition.
        UniqueConstraint("dedup_key", name="uq_monitoring_alert_dedup_key"),
        # CHECK names are unprefixed; the metadata naming convention expands
        # them to ck_%(table_name)s_%(constraint_name)s, so an already-prefixed
        # name would produce ck_..._ck_... and diverge from migration 0017.
        CheckConstraint("severity IN ('info', 'warning', 'critical')", name="severity_known"),
        CheckConstraint("dedup_key ~ '^[0-9a-f]{64}$'", name="dedup_key_is_digest"),
        CheckConstraint("rule_id <> ''", name="rule_id_not_empty"),
        CheckConstraint("subject <> ''", name="subject_not_empty"),
        CheckConstraint("detail <> ''", name="detail_not_empty"),
        CheckConstraint("git_commit ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'", name="git_commit_is_sha"),
        CheckConstraint("data_version <> ''", name="data_version_not_empty"),
        CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        CheckConstraint("seed >= 0", name="seed_non_negative"),
    )

    alert_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; also the raise order.",
    )
    dedup_key: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "64-character hex SHA-256 of the rule and the condition. The alert's identity: "
            "a repeat of the same condition re-derives it and is absorbed by the UNIQUE "
            "constraint. Never a counter — see the class docstring."
        ),
    )
    rule_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Which rule raised this, e.g. 'live_vs_expected'. Part of the dedup digest.",
    )
    severity: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'info', 'warning' or 'critical'. Bound by CHECK to the AlertSeverity members.",
    )
    subject: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="One line, for a channel that has room for one line.",
    )
    detail: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The full message including the numbers that produced the condition.",
    )
    payload: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        doc=(
            "The machine-readable finding — typically the halt decision's or drift finding's "
            "own to_dict(), band, disclosures and all — so the alert stands alone when the "
            "objects that produced it are gone."
        ),
    )
    raised_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc="UTC instant the CONDITION was observed. What an incident timeline is built from.",
    )
    git_commit: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: commit of the monitoring run that raised the alert (40 or 64 hex characters).",
    )
    git_dirty: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="I2: whether that run's working tree differed from its commit, untracked included.",
    )
    data_version: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: identifier of the data snapshot the monitoring run read.",
    )
    config_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: SHA-256 of the run's canonical configuration (64 hex characters).",
    )
    seed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="I2: the run's random seed, recorded verbatim. Dimensionless.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Wall-clock provenance only.",
    )


class MonitoringAlertDelivery(Base):
    """One attempt to hand one alert to one channel (P12.4).

    Delivery is a **history**, not a flag. Every attempt is a row — the failures
    especially — so "which alerts reached nobody" is a query rather than a
    guess, and an alert that was persisted and never dispatched (no rows at all)
    is distinguishable from one that was dispatched and refused (rows, all
    failed). Those are different incidents: the first means the dispatcher died,
    the second means the channel is down.

    ``UNIQUE (alert_id, channel_id, attempt)`` makes the attempt counter
    monotonic per channel and stops a retry from overwriting the record of the
    attempt it is retrying.

    A failed attempt must carry a detail (CHECK ``detail <> ''``): a failure
    that does not say why cannot be acted on, and the operator response to "SMTP
    refused the recipient" is not the response to "the credential expired".
    """

    __tablename__ = "monitoring_alert_delivery"
    __table_args__ = (
        UniqueConstraint(
            "alert_id", "channel_id", "attempt", name="uq_monitoring_alert_delivery_attempt"
        ),
        CheckConstraint("outcome IN ('delivered', 'failed')", name="outcome_known"),
        CheckConstraint("attempt >= 1", name="attempt_positive"),
        CheckConstraint("channel_id <> ''", name="channel_id_not_empty"),
        CheckConstraint("detail <> ''", name="detail_not_empty"),
    )

    delivery_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; also the attempt order.",
    )
    alert_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("monitoring_alert.alert_id", ondelete="RESTRICT"),
        nullable=False,
        doc="The alert this attempt was for.",
    )
    channel_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Identifier of the delivery channel, e.g. 'structlog', 'pagerduty'.",
    )
    attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="1-based attempt number for this (alert, channel) pair. Dimensionless count.",
    )
    outcome: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'delivered' or 'failed'. Bound by CHECK to the DeliveryOutcome members.",
    )
    detail: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "The channel's receipt on success, or why it refused on failure. Never blank: a "
            "failure that does not say why cannot be acted on."
        ),
    )
    attempted_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc="UTC instant the attempt was made.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Wall-clock provenance only.",
    )


class MonitoringAlertAcknowledgement(Base):
    """A person's signature on an alert (P12.4, §6.10).

    One row per alert, enforced by ``UNIQUE (alert_id)``. An acknowledgement
    records *who took responsibility*, so a second one is refused rather than
    overwriting the first — silently replacing that name is worse than losing
    it, because the record would still look complete.

    :attr:`note` cannot be blank. An acknowledgement with no note is a click,
    and the point of the record is that somebody looked at the alert and
    concluded something. It is also the text
    :func:`backend.monitoring.history.record_resume` shows the next operator
    before trading restarts.
    """

    __tablename__ = "monitoring_alert_acknowledgement"
    __table_args__ = (
        UniqueConstraint("alert_id", name="uq_monitoring_alert_acknowledgement_alert"),
        CheckConstraint("acknowledged_by <> ''", name="acknowledged_by_not_empty"),
        CheckConstraint("note <> ''", name="note_not_empty"),
    )

    acknowledgement_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless.",
    )
    alert_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("monitoring_alert.alert_id", ondelete="RESTRICT"),
        nullable=False,
        doc="The alert being acknowledged. UNIQUE: one signature per alert, never replaced.",
    )
    acknowledged_by: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Who signed. Free text because this platform has no operator identity model yet; "
            "never blank, because 'acknowledged by nobody' is not an acknowledgement."
        ),
    )
    note: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="What they concluded. Never blank — see the class docstring.",
    )
    acknowledged_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc="UTC instant the alert was acknowledged.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Wall-clock provenance only.",
    )


class MonitoringHaltEvent(Base):
    """One halt or resume in the trading halt history (P12.1/P12.4, §6.10).

    **There is no halt-state column anywhere in this schema.** Current state is
    derived: the earliest ``halt`` row that no ``resume`` row points at
    (:func:`backend.monitoring.history.active_halt`). A stored flag on an
    append-only table cannot be updated, and a flag that drifts from the history
    explaining it is the single worst field this schema could contain — it is
    the one an operator would trust most.

    ``UNIQUE (resolves_halt_event_id)`` makes "resume the same halt twice"
    impossible rather than merely checked, so the derived state can never be
    ambiguous. The CHECKs bind the two row shapes: a halt carries a cause and
    resolves nothing; a resume resolves exactly one halt, carries no cause, and
    names the operator who took the decision.

    :attr:`alert_dedup_key` links a halt to the alert it raised. That link is
    what makes acknowledgement load-bearing:
    :func:`backend.monitoring.history.record_resume` refuses to resume a halt
    whose alert has no acknowledgement row, so an alerting pipeline nobody reads
    cannot quietly become a formality.

    :attr:`decision` stores the halting
    :class:`~backend.monitoring.expectation.HaltDecision` payload verbatim — the
    band, the CPCV artefact identity, the disclosures and the comparison — so
    the history explains itself years later without the objects that produced
    it. Not bitemporal, for the same reason as the other three tables here.
    """

    __tablename__ = "monitoring_halt_event"
    __table_args__ = (
        # One resume per halt. The derived state depends on this being a
        # database fact rather than an application convention.
        UniqueConstraint("resolves_halt_event_id", name="uq_monitoring_halt_event_resolves"),
        CheckConstraint("kind IN ('halt', 'resume')", name="kind_known"),
        CheckConstraint("detail <> ''", name="detail_not_empty"),
        CheckConstraint("actor <> ''", name="actor_not_empty"),
        # The two row shapes, stated in SQL so a writer that skips Python is
        # still bound: a halt has a cause and resolves nothing; a resume
        # resolves exactly one halt and carries no cause.
        CheckConstraint(
            "(kind = 'halt' AND cause IS NOT NULL AND resolves_halt_event_id IS NULL) "
            "OR (kind = 'resume' AND cause IS NULL AND resolves_halt_event_id IS NOT NULL)",
            name="shape_matches_kind",
        ),
        CheckConstraint(
            "cause IS NULL OR cause IN ("
            "'below_expected_band', 'above_expected_band', 'comparison_unavailable', "
            "'stale_live_data', 'cost_basis', 'internal_error')",
            name="cause_known",
        ),
        CheckConstraint("git_commit ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'", name="git_commit_is_sha"),
        CheckConstraint("data_version <> ''", name="data_version_not_empty"),
        CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        CheckConstraint("seed >= 0", name="seed_non_negative"),
    )

    halt_event_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        doc="Surrogate row key, database-generated. Dimensionless; also the event order.",
    )
    kind: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="'halt' or 'resume'. Bound by CHECK; the two shapes differ, see the table args.",
    )
    cause: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "Why trading stopped, from backend.monitoring.expectation.HaltCause. NOT NULL on "
            "a halt row and NULL on a resume, by CHECK. A distinct value per operator "
            "response — 'comparison_unavailable' and 'below_expected_band' are not the same "
            "incident."
        ),
    )
    detail: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The decision's own words on a halt; the operator's reason on a resume.",
    )
    actor: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc=(
            "Who. 'auto:monitoring' for a monitor-raised halt, a person for a resume. Never "
            "NULL: 'who halted trading' has an answer even when the answer is a scheduled job."
        ),
    )
    occurred_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        doc="UTC instant of the decision (the halt decision's own timestamp, not the write).",
    )
    resolves_halt_event_id: Mapped[int | None] = mapped_column(
        BigInteger,
        # Named explicitly. The convention's fk template is
        # fk_<table>_<column>_<referred table>, and for a *self*-referential key
        # the table name appears twice, giving 69 characters against
        # PostgreSQL's 63-byte identifier limit. SQLAlchemy raises
        # IdentifierError while rendering the metadata, so it takes down every
        # test that touches Base.metadata, not merely this table.
        ForeignKey(
            "monitoring_halt_event.halt_event_id",
            ondelete="RESTRICT",
            name="fk_monitoring_halt_event_resolves",
        ),
        nullable=True,
        doc="On a resume, the halt it clears. UNIQUE, so a halt is resumed at most once.",
    )
    alert_dedup_key: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc=(
            "The alert this halt raised, when it raised one. Not a foreign key: an alert may "
            "be absorbed as a repeat and this column records the condition, not the row. It "
            "is what makes acknowledgement a precondition for resuming."
        ),
    )
    decision: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
        doc=(
            "The halting HaltDecision.to_dict() verbatim — band, CPCV artefact identity, "
            "disclosures, comparison — so the history explains itself without the objects "
            "that produced it. NULL on a resume."
        ),
    )
    git_commit: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: commit of the run that wrote this row (40 or 64 hex characters).",
    )
    git_dirty: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="I2: whether that run's working tree differed from its commit, untracked included.",
    )
    data_version: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: identifier of the data snapshot behind the decision.",
    )
    config_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="I2: SHA-256 of the run's canonical configuration (64 hex characters).",
    )
    seed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="I2: the run's random seed, recorded verbatim. Dimensionless.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the row was written, UTC, from the database clock. Wall-clock provenance only.",
    )
