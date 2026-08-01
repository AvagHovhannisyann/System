"""Parsing EDGAR's published artifacts, and the acceptance-time conversion (P3.2).

Everything in this module is a pure function over bytes EDGAR actually served.
Nothing here performs I/O, so every claim it makes is testable against captured
responses (``backend/tests/fixtures/edgar/``, each recording its source URL and
capture date).

The three artifacts
-------------------

**1. The quarter directory listing**
``https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{q}/index.json`` —
a JSON listing of every file EDGAR published for that quarter. The connector
enumerates it rather than constructing candidate dates, because *EDGAR answers
a request for a daily-index file that does not exist with HTTP 403* (an S3
``AccessDenied`` body), not 404 — verified live on
``.../2024/QTR1/master.20240309.idx`` (a Saturday) on 2026-08-01. A 403 is
exactly what a fair-access block also looks like, so "guess the date, treat the
error as an empty day" cannot be made safe: it would convert a throttling
response into a silent hole in the data. The listing removes the guess.

**2. The daily index** ``master.YYYYMMDD.idx`` — pipe-delimited
``CIK|Company Name|Form Type|Date Filed|File Name`` after a fixed preamble.
Three properties of the real file the parser must handle, all observed in the
2024-03-11 index (5395 data rows):

- *one accession, many CIKs.* The row is a (filing, filer) association, not a
  filing: 5299 distinct ``(accession, CIK)`` pairs cover only 3394 distinct
  accessions, and 1809 accessions are listed under more than one CIK (a joint
  Form 4 by several reporting owners; accession ``0001193805-24-000360`` in the
  2024-03-12 index is listed seven times). This is the ordinary case, not an
  oddity — which is why the parser keys on the pair and
  ``edgar_filing``'s logical key is ``(accession_number, cik)``. Collapsing to
  the accession alone would mean picking one filer's name and CIK and
  presenting it as the filing's, which is a fabricated choice.
- *duplicate rows.* 96 rows repeat a ``(accession, CIK)`` pair already listed
  (EDGAR emits one row per series/class). They are collapsed, and the number
  collapsed is reported as a measurement, not discarded silently.
- *``Date Filed`` is not the index date.* The 2024-03-11 index contains a
  CORRESP whose ``Date Filed`` is 2024-02-07 — staff correspondence is
  disseminated long after it is accepted.

**3. The filing header** ``{accession}-index-headers.html`` — one request per
filing, carrying both the submission's SGML header (acceptance datetime, form
type, period, filing date) and its document manifest. It is the only artifact
of the three that carries an acceptance timestamp: the daily index does not.

Acceptance time, and why it is the whole point
----------------------------------------------

``<ACCEPTANCE-DATETIME>`` is 14 digits, ``YYYYMMDDHHMMSS``, with **no timezone
marker**. It is US Eastern local time — ``America/New_York``, so the offset is
-05:00 or -04:00 depending on the date. That was verified two independent ways
on 2026-08-01, both recorded in
``backend/tests/fixtures/edgar/acceptance-datetime-crosscheck.json``:

1. against ``https://data.sec.gov/submissions/CIK##########.json``, whose
   ``acceptanceDateTime`` for the same accession is the same instant in UTC —
   e.g. accession ``0001104659-24-032038`` is ``20240308060045`` in the SGML
   header and ``2024-03-08T11:00:45.000Z`` in the JSON (+5h, EST), while
   ``0001104659-24-082105`` is ``20240724073816`` and
   ``2024-07-24T11:38:16.000Z`` (+4h, EDT);
2. against EDGAR's own filing-date rule, 17 CFR 232.13 — a submission
   commencing after 5:30 p.m. ET is deemed filed the next business day, except
   Forms 3/4/5, Schedules 13D/13G/14N, Form 144 and Rule 462(b) filings, which
   get the same business day until 10 p.m. ET. Accession
   ``0000950172-24-000037`` (Form D) carries acceptance ``20240308175909`` and
   filing date ``20240311``: the next business day, which follows only if
   17:59:09 is Eastern.

**data.sec.gov is not usable as the source for this.** For CIK 1750284 its
``acceptanceDateTime`` repeats the Eastern wall clock with a ``Z`` suffix
instead of converting (accession ``0001243233-24-000002``: SGML
``20241211185518``, JSON ``2024-12-11T18:55:18.000Z``, filing date
``2024-12-12`` — one business day later, which proves the value is Eastern).
The same field is a correct UTC conversion for CIKs 100726, 1000184 and
1796799. Since one field cannot be trusted to mean one thing, the connector
reads the SGML header, which is uniformly Eastern, and converts here.

Ambiguous and non-existent local times raise rather than resolve. During the
autumn fall-back hour a local time names two instants and during the spring
forward hour it names none; choosing one would be a fabricated instant, and
choosing the earlier one would be a fabricated *lookahead*. Both transitions
happen on a Sunday between 01:00 and 03:00 ET, when EDGAR neither accepts nor
disseminates filings, so this raises on data that should not exist rather than
on data that routinely does.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

from backend.ingest.errors import PermanentSourceError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "ARCHIVES_BASE",
    "DAILY_INDEX_BASE",
    "EARLIEST_ISO_NAMED_DAILY_INDEX",
    "EDGAR_TIMEZONE",
    "DailyIndexEntry",
    "DailyIndexFile",
    "DailyIndexParse",
    "FilingDocument",
    "FilingHeader",
    "QuarterListing",
    "acceptance_datetime_to_utc",
    "daily_index_file_url",
    "filing_index_headers_url",
    "parse_daily_index",
    "parse_filing_header",
    "parse_quarter_listing",
    "quarter_listing_url",
    "quarter_of",
]

ARCHIVES_BASE: Final = "https://www.sec.gov/Archives"
"""Root of EDGAR's archive tree (verified live 2026-08-01)."""

DAILY_INDEX_BASE: Final = f"{ARCHIVES_BASE}/edgar/daily-index"
"""Root of the daily index tree; ``/{year}/QTR{q}/`` beneath it."""

EDGAR_TIMEZONE: Final = ZoneInfo("America/New_York")
"""The timezone ``<ACCEPTANCE-DATETIME>`` is expressed in (module docstring).

Named zone rather than a fixed offset, deliberately: the offset is -05:00 or
-04:00 depending on the date, and a fixed offset would be wrong for half the
year — a one-hour lookahead or lag on every filing in that half.
"""

EARLIEST_ISO_NAMED_DAILY_INDEX: Final = dt.date(1998, 5, 18)
"""First date EDGAR publishes a ``master.YYYYMMDD.idx`` daily index.

Verified live on 2026-08-01 by enumerating the quarter listings: EDGAR names
its daily indices three different ways across its history — ``master.MMDDYY``
in 1994 (earliest ``master.070194.idx``), ``master.YYMMDD`` from 1995 to
mid-1998, and ``master.YYYYMMDD`` from 1998-05-18 onward, with the two later
schemes published side by side for the same days through 1998. This parser
recognizes only the ``YYYYMMDD`` form: the other two cannot be read without
guessing a century (``070194`` and ``940701`` are the same six digits in a
different order), and a wrong guess would silently file filings under the
wrong decade. Listing entries in the older schemes are **counted and reported**
by :func:`parse_quarter_listing`, never silently dropped, so the boundary is a
measured limitation rather than an invisible one.
"""

_ACCEPTANCE_DIGITS: Final = 14
"""Length of the ``YYYYMMDDHHMMSS`` acceptance value (characters)."""

_INDEX_FIELDS: Final = 5
"""Pipe-delimited fields per daily-index data row."""

_MIN_QUARTER_MONTH_SPAN: Final = 3
"""Calendar months per quarter (dimensionless)."""

_DATE_DIGITS: Final = 8
"""Length of a ``YYYYMMDD`` date field (characters).

Checked explicitly before ``strptime``, because ``%Y`` matches fewer than four
digits: ``strptime("070194", "%Y%m%d")`` succeeds and yields the year 701. That
is exactly how EDGAR's 1994 ``master.MMDDYY.idx`` names would be silently filed
under the eighth century.
"""

_HEADER_TAG = re.compile(r"^<(/?)([A-Z0-9-]+)>(.*)$")
"""One SGML header line: ``<TAG>value`` or ``</TAG>``. Values run to end of line."""

_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
"""EDGAR accession-number shape, e.g. ``0001104659-24-032038``."""


def _parse_yyyymmdd(value: str) -> dt.date:
    """Parse a strict ``YYYYMMDD`` date, rejecting shorter stamps.

    Raises:
        ValueError: if the value is not exactly eight digits naming a real
            date.
    """
    if len(value) != _DATE_DIGITS or not value.isdigit():
        msg = f"expected an 8-digit YYYYMMDD date; got {value!r}"
        raise ValueError(msg)
    return dt.datetime.strptime(value, "%Y%m%d").date()  # noqa: DTZ007 — date only, no instant


def quarter_of(day: dt.date) -> int:
    """Return the calendar quarter (1-4) containing ``day``.

    Args:
        day: any calendar date.

    Returns:
        The quarter number, 1 through 4 (dimensionless).
    """
    return (day.month - 1) // _MIN_QUARTER_MONTH_SPAN + 1


def quarter_listing_url(year: int, quarter: int) -> str:
    """Return the URL of one quarter's daily-index directory listing.

    Args:
        year: four-digit calendar year.
        quarter: quarter number, 1 through 4.

    Returns:
        The ``index.json`` URL for that quarter's daily-index directory.

    Raises:
        ValueError: if ``quarter`` is outside 1-4 — a URL built from a bad
            quarter would 403 like a fair-access block and be misread as one.
    """
    if not 1 <= quarter <= 4:
        msg = f"quarter must be 1-4; got {quarter}"
        raise ValueError(msg)
    return f"{DAILY_INDEX_BASE}/{year}/QTR{quarter}/index.json"


def daily_index_file_url(year: int, quarter: int, file_name: str) -> str:
    """Return the URL of a daily-index file **named by the quarter listing**.

    The file name is never constructed from a date: EDGAR has used three
    naming schemes over its history (see
    :data:`EARLIEST_ISO_NAMED_DAILY_INDEX`) and answers a request for a
    non-existent daily-index key with 403, which is indistinguishable from a
    fair-access block. Taking the name from the listing removes both problems.

    Args:
        year: four-digit calendar year of the quarter directory.
        quarter: quarter number, 1 through 4.
        file_name: the ``name`` field of a listing entry, e.g.
            ``master.20240311.idx``.

    Returns:
        The absolute URL of that file.
    """
    return f"{DAILY_INDEX_BASE}/{year}/QTR{quarter}/{file_name}"


def filing_index_headers_url(cik: int, accession_number: str) -> str:
    """Return the URL of a filing's ``-index-headers.html`` artifact.

    Args:
        cik: the CIK the daily index listed the filing under. EDGAR serves the
            filing directory under every filer CIK associated with it; the
            index's own CIK is therefore always a valid path.
        accession_number: dashed accession, e.g. ``0001104659-24-032038``.

    Returns:
        The absolute URL of the filing's header/manifest artifact.

    Raises:
        ValueError: if ``accession_number`` is not in EDGAR's dashed form.
    """
    if not _ACCESSION.match(accession_number):
        msg = f"not an EDGAR accession number: {accession_number!r}"
        raise ValueError(msg)
    undashed = accession_number.replace("-", "")
    return f"{ARCHIVES_BASE}/edgar/data/{cik}/{undashed}/{accession_number}-index-headers.html"


def acceptance_datetime_to_utc(raw: str) -> dt.datetime:
    """Convert an ``<ACCEPTANCE-DATETIME>`` value to a timezone-aware UTC instant.

    This is the function invariant I1 rests on for this source: its result
    becomes ``knowledge_time``, so an hour of error here is an hour of
    lookahead in every backtest built on EDGAR text.

    Args:
        raw: the 14-digit ``YYYYMMDDHHMMSS`` value exactly as EDGAR wrote it.
            Surrounding whitespace is tolerated; nothing else is. The value is
            US Eastern local time (``America/New_York``) — see the module
            docstring for how that was verified against the live service.

    Returns:
        The same instant as a timezone-aware ``datetime`` in UTC. Under EST the
        UTC value is 5 hours later, under EDT 4 hours later; a filing accepted
        in the Eastern evening therefore lands on the following UTC date.

    Raises:
        PermanentSourceError: if the value is not 14 digits, does not name a
            real calendar date and time, or names a local time that is
            ambiguous (the autumn fall-back hour, which is two instants) or
            non-existent (the spring forward hour, which is none). Retrying the
            identical request cannot fix any of these, hence *permanent*. The
            transition cases raise rather than pick an instant: picking the
            earlier one would manufacture an hour of lookahead and picking
            either would report a precision the source does not have. Both
            transitions fall on a Sunday between 01:00 and 03:00 ET, when
            EDGAR accepts nothing, so this is a guard on impossible data.
    """
    value = raw.strip()
    if len(value) != _ACCEPTANCE_DIGITS or not value.isdigit():
        msg = (
            f"EDGAR acceptance datetime must be {_ACCEPTANCE_DIGITS} digits "
            f"(YYYYMMDDHHMMSS); got {raw!r}"
        )
        raise PermanentSourceError(msg)
    try:
        naive = dt.datetime.strptime(value, "%Y%m%d%H%M%S")  # noqa: DTZ007 — zone applied below
    except ValueError as exc:
        msg = f"EDGAR acceptance datetime {raw!r} is not a real date and time: {exc}"
        raise PermanentSourceError(msg) from exc
    earlier = naive.replace(tzinfo=EDGAR_TIMEZONE, fold=0)
    later = naive.replace(tzinfo=EDGAR_TIMEZONE, fold=1)
    if earlier.utcoffset() != later.utcoffset():
        round_tripped = earlier.astimezone(dt.UTC).astimezone(EDGAR_TIMEZONE).replace(tzinfo=None)
        kind = "does not exist" if round_tripped != naive else "is ambiguous"
        msg = (
            f"EDGAR acceptance datetime {raw!r} read as US/Eastern local time {kind} "
            "(daylight-saving transition). Refusing to choose an instant: the earlier "
            "choice would manufacture an hour of lookahead and either choice would "
            "report precision the source does not carry (D-011/I1)"
        )
        raise PermanentSourceError(msg)
    return earlier.astimezone(dt.UTC)


@dataclass(frozen=True, slots=True)
class DailyIndexEntry:
    """One filing as the daily index describes it.

    Attributes:
        cik: Central Index Key the row was listed under (dimensionless
            integer). One accession is commonly listed under several CIKs — a
            joint filing's reporting owners and issuer — so an entry describes
            a (filing, filer) association rather than a filing.
        company_name: filer name exactly as the index spells it.
        form_type: EDGAR form type, e.g. ``10-K``, ``8-K``, ``4``.
        filing_date: EDGAR's assigned filing date (``Date Filed``), a calendar
            date with no time. **Not** the acceptance instant, and not
            necessarily the index date — see the module docstring.
        accession_number: dashed accession number, unique per filing.
        document_path: the archive path the index gives, e.g.
            ``edgar/data/1000184/0001104659-24-032038.txt``.
    """

    cik: int
    company_name: str
    form_type: str
    filing_date: dt.date
    accession_number: str
    document_path: str


@dataclass(frozen=True, slots=True)
class DailyIndexParse:
    """The result of parsing one daily index, with what was measured doing so.

    Attributes:
        entries: one entry per distinct ``(accession, CIK)`` pair, in the order
            the index listed them first. A filing with several filers appears
            once per filer, which is what the index states.
        rows_read: data rows in the file (count).
        duplicate_rows_collapsed: rows repeating a ``(accession, CIK)`` pair
            already seen (count).
            ``rows_read - duplicate_rows_collapsed == len(entries)``. Reported
            rather than discarded so a change in EDGAR's duplication behaviour
            is visible in the data-quality report instead of appearing as a
            coverage change.
        distinct_accessions: distinct accession numbers across ``entries``
            (count) — the number of submission headers a connector must fetch,
            which is smaller than ``len(entries)`` whenever a filing has more
            than one filer.
    """

    entries: tuple[DailyIndexEntry, ...]
    rows_read: int
    duplicate_rows_collapsed: int
    distinct_accessions: int


def parse_daily_index(text: str) -> DailyIndexParse:
    """Parse a ``master.YYYYMMDD.idx`` body into distinct filing entries.

    Args:
        text: the file body as EDGAR served it (latin-1 decoded; company names
            contain non-ASCII bytes).

    Returns:
        A :class:`DailyIndexParse`. Preamble and column-header lines are
        skipped by shape — a data row is five pipe-delimited fields whose first
        field is all digits — rather than by counting header lines, so a
        preamble that changes length does not shift the parse.

    Raises:
        PermanentSourceError: if a data row's filing date is not ``YYYYMMDD``,
            if its accession number is not in EDGAR's dashed form, or if two
            rows share an ``(accession, CIK)`` pair but disagree on form type
            or filing date. The last one is a genuine contradiction in the
            source — the same filing by the same filer described two ways — and
            silently keeping one of the values would be a fabricated choice, so
            it is refused loudly. Two rows sharing only the *accession* are not
            a contradiction: that is a filing with several filers, and both
            rows are kept.
    """
    seen: dict[tuple[str, int], DailyIndexEntry] = {}
    rows_read = 0
    duplicates = 0
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) != _INDEX_FIELDS or not fields[0].strip().isdigit():
            continue
        rows_read += 1
        entry = _index_entry(fields)
        key = (entry.accession_number, entry.cik)
        existing = seen.get(key)
        if existing is None:
            seen[key] = entry
            continue
        duplicates += 1
        if (existing.form_type, existing.filing_date) != (entry.form_type, entry.filing_date):
            msg = (
                f"daily index contradicts itself for accession {entry.accession_number} "
                f"under CIK {entry.cik}: {existing!r} then {entry!r}. Refusing to choose "
                "between two versions of the same filing's identity"
            )
            raise PermanentSourceError(msg)
    entries = tuple(seen.values())
    return DailyIndexParse(
        entries=entries,
        rows_read=rows_read,
        duplicate_rows_collapsed=duplicates,
        distinct_accessions=len({entry.accession_number for entry in entries}),
    )


def _index_entry(fields: Sequence[str]) -> DailyIndexEntry:
    """Build one :class:`DailyIndexEntry` from a split data row."""
    cik_text, company_name, form_type, filed_text, document_path = (f.strip() for f in fields)
    try:
        filing_date = _parse_yyyymmdd(filed_text)
    except ValueError as exc:
        msg = f"daily index row has an unparseable Date Filed {filed_text!r}: {exc}"
        raise PermanentSourceError(msg) from exc
    accession_number = document_path.rsplit("/", 1)[-1].removesuffix(".txt")
    if not _ACCESSION.match(accession_number):
        msg = (
            f"daily index row has no accession number in its File Name {document_path!r} "
            f"(expected .../NNNNNNNNNN-NN-NNNNNN.txt)"
        )
        raise PermanentSourceError(msg)
    return DailyIndexEntry(
        cik=int(cik_text),
        company_name=company_name,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession_number,
        document_path=document_path,
    )


@dataclass(frozen=True, slots=True)
class DailyIndexFile:
    """One ``master`` daily-index file a quarter listing publishes.

    Attributes:
        index_date: the dissemination date the file covers.
        file_name: the listing's own file name, used verbatim to build the URL.
    """

    index_date: dt.date
    file_name: str


@dataclass(frozen=True, slots=True)
class QuarterListing:
    """What one quarter directory listing publishes, and what was skipped.

    Attributes:
        files: the ``master.YYYYMMDD.idx`` files, one per date, sorted
            ascending by date. Days EDGAR published no index for are simply
            absent — the listing is the source of truth about which days exist,
            which is what lets the connector avoid guessing dates and
            misreading a 403 as an empty day.
        legacy_named_files: listing entries matching ``master.*.idx`` whose
            stamp is not ``YYYYMMDD`` (count). These are EDGAR's pre-1998-05-18
            naming schemes; they are counted rather than dropped silently so
            the coverage boundary is measured. See
            :data:`EARLIEST_ISO_NAMED_DAILY_INDEX`.
    """

    files: tuple[DailyIndexFile, ...]
    legacy_named_files: int


def parse_quarter_listing(body: str) -> QuarterListing:
    """Return the daily-index files a quarter listing actually publishes.

    Args:
        body: the quarter directory's ``index.json`` response body.

    Returns:
        A :class:`QuarterListing`. When EDGAR lists the same date under more
        than one name (it does so through 1998), the ``YYYYMMDD``-named entry
        is the one kept.

    Raises:
        PermanentSourceError: if the body is not JSON or has no
            ``directory.item`` array.
    """
    try:
        parsed: Any = json.loads(body)
    except ValueError as exc:
        msg = f"daily-index quarter listing is not JSON: {exc}"
        raise PermanentSourceError(msg) from exc
    files: dict[dt.date, DailyIndexFile] = {}
    legacy = 0
    for item in _listing_items(parsed):
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.startswith("master.") or not name.endswith(".idx"):
            continue
        stamp = name.removeprefix("master.").removesuffix(".idx")
        try:
            index_date = _parse_yyyymmdd(stamp)
        except ValueError:
            legacy += 1
            continue
        files.setdefault(index_date, DailyIndexFile(index_date=index_date, file_name=name))
    return QuarterListing(
        files=tuple(files[day] for day in sorted(files)),
        legacy_named_files=legacy,
    )


def _listing_items(parsed: object) -> Iterable[object]:
    """Return the ``directory.item`` array of a listing, or raise."""
    directory = parsed.get("directory") if isinstance(parsed, dict) else None
    items = directory.get("item") if isinstance(directory, dict) else None
    if not isinstance(items, list):
        msg = "daily-index quarter listing has no directory.item array"
        raise PermanentSourceError(msg)
    return items


@dataclass(frozen=True, slots=True)
class FilingDocument:
    """One document inside a filing, as the submission manifest lists it.

    Attributes:
        sequence: EDGAR's document sequence number within the filing
            (dimensionless). **Not contiguous** — accession
            ``0000950170-24-029225`` lists sequences 1, 2, 3, 5 — so nothing
            may infer a document count from the maximum.
        document_type: EDGAR document type, e.g. ``8-K``, ``EX-99.1``,
            ``GRAPHIC``.
        filename: file name within the filing's archive directory.
        description: the manifest's free-text description, or ``None`` when
            the manifest carries none (common on Forms 3/4/5 and Form D).
    """

    sequence: int
    document_type: str
    filename: str
    description: str | None


@dataclass(frozen=True, slots=True)
class FilingHeader:
    """The submission header and document manifest of one filing.

    Attributes:
        accession_number: dashed accession number as the header states it.
        form_type: EDGAR form type from the header's ``<TYPE>``.
        acceptance_datetime: when EDGAR accepted the submission, timezone-aware
            UTC, converted from the header's Eastern local value. This is the
            connector's ``knowledge_time``.
        filing_date: EDGAR's assigned filing date (calendar date, no time). May
            be a later business day than the acceptance date, and for the
            10 p.m. form families may be the same date as an evening
            acceptance — which is exactly why it cannot serve as a knowledge
            time.
        period_of_report: the fiscal period the filing reports on, or ``None``
            when the header declares none.
        declared_document_count: the header's ``<PUBLIC-DOCUMENT-COUNT>``
            (count), or ``None`` when absent. Kept beside ``documents`` rather
            than reconciled: a mismatch is a measurement worth reporting, not
            something to resolve by preferring one number.
        documents: the manifest, in the order listed.
    """

    accession_number: str
    form_type: str
    acceptance_datetime: dt.datetime
    filing_date: dt.date
    period_of_report: dt.date | None
    declared_document_count: int | None
    documents: tuple[FilingDocument, ...]


def parse_filing_header(body: str, *, accession_number: str) -> FilingHeader:
    """Parse a filing's ``-index-headers.html`` into header fields and a manifest.

    The artifact carries the submission's SGML header twice — once inside an
    HTML comment and once HTML-escaped inside a ``<PRE>`` block that also holds
    the document manifest. The body is unescaped once and scanned line by line;
    header scalars are taken from their first occurrence, so the two copies
    cannot disagree about which one was used.

    Args:
        body: the response body as EDGAR served it.
        accession_number: the accession the connector requested, used to verify
            the response is the filing that was asked for.

    Returns:
        A :class:`FilingHeader` with ``acceptance_datetime`` already converted
        to UTC.

    Raises:
        PermanentSourceError: if ``<ACCEPTANCE-DATETIME>``, ``<TYPE>`` or
            ``<FILING-DATE>`` is missing, if a date field is unparseable, if
            the header's accession number is not the one requested, or if a
            document block omits its sequence, type or filename. Every one of
            these means the response does not carry the contract this parser
            documents; substituting a value for any of them would put an
            invented fact into an append-only store.
    """
    scalars, documents = _scan_header(html.unescape(body))
    stated_accession = scalars.get("ACCESSION-NUMBER")
    if stated_accession is not None and stated_accession != accession_number:
        msg = (
            f"filing header for {accession_number} reports a different accession "
            f"{stated_accession!r}: refusing to attribute one filing's header to another"
        )
        raise PermanentSourceError(msg)
    acceptance_raw = _required(scalars, "ACCEPTANCE-DATETIME", accession_number)
    return FilingHeader(
        accession_number=accession_number,
        form_type=_required(scalars, "TYPE", accession_number),
        acceptance_datetime=acceptance_datetime_to_utc(acceptance_raw),
        filing_date=_header_date(
            _required(scalars, "FILING-DATE", accession_number), "FILING-DATE", accession_number
        ),
        period_of_report=(
            None
            if scalars.get("PERIOD") is None
            else _header_date(scalars["PERIOD"], "PERIOD", accession_number)
        ),
        declared_document_count=_optional_count(scalars.get("PUBLIC-DOCUMENT-COUNT")),
        documents=documents,
    )


def _scan_header(text: str) -> tuple[dict[str, str], tuple[FilingDocument, ...]]:
    """Return first-occurrence header scalars and the parsed document manifest."""
    scalars: dict[str, str] = {}
    documents: list[FilingDocument] = []
    current: dict[str, str] = {}
    in_document = False
    for line in text.splitlines():
        match = _HEADER_TAG.match(line.strip())
        if match is None:
            continue
        closing, tag, value = match.group(1) == "/", match.group(2), match.group(3).strip()
        if tag == "DOCUMENT":
            if closing:
                if in_document:
                    documents.append(_document_from(current))
                in_document = False
            else:
                in_document, current = True, {}
            continue
        if closing:
            continue
        if in_document:
            current.setdefault(tag, value)
        elif tag not in scalars:
            scalars[tag] = value
    return scalars, tuple(documents)


def _document_from(fields: Mapping[str, str]) -> FilingDocument:
    """Build one :class:`FilingDocument` from a manifest block's tags."""
    missing = [tag for tag in ("TYPE", "SEQUENCE", "FILENAME") if not fields.get(tag)]
    if missing:
        msg = (
            f"filing document manifest block is missing {', '.join(missing)}; "
            f"got tags {sorted(fields)}"
        )
        raise PermanentSourceError(msg)
    sequence = fields["SEQUENCE"]
    if not sequence.isdigit():
        msg = f"filing document sequence {sequence!r} is not a number"
        raise PermanentSourceError(msg)
    description = fields.get("DESCRIPTION")
    return FilingDocument(
        sequence=int(sequence),
        document_type=fields["TYPE"],
        filename=fields["FILENAME"],
        description=description if description else None,
    )


def _required(scalars: Mapping[str, str], tag: str, accession_number: str) -> str:
    """Return a required header scalar, or raise naming the filing that lacked it."""
    value = scalars.get(tag)
    if not value:
        msg = (
            f"filing header for {accession_number} has no <{tag}>: the response does not "
            "carry the contract backend.ingest.edgar.parse documents"
        )
        raise PermanentSourceError(msg)
    return value


def _header_date(value: str, tag: str, accession_number: str) -> dt.date:
    """Parse an SGML header ``YYYYMMDD`` date field."""
    try:
        return _parse_yyyymmdd(value)
    except ValueError as exc:
        msg = f"filing header for {accession_number} has an unparseable <{tag}> {value!r}: {exc}"
        raise PermanentSourceError(msg) from exc


def _optional_count(value: str | None) -> int | None:
    """Return a declared document count, or ``None`` when absent or non-numeric."""
    return int(value) if value is not None and value.isdigit() else None
