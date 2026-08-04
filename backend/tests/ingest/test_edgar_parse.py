"""P3.2 unit tests: acceptance-time conversion and EDGAR artifact parsing.

Every input is a captured EDGAR response (see
``backend/tests/ingest/test_edgar_fixtures.py``). The one exception is the
daylight-saving boundary suite, which calls
:func:`~backend.ingest.edgar.parse.acceptance_datetime_to_utc` with constructed
*local times* — that is exercising a pure conversion at values EDGAR cannot
produce, not inventing a source payload.

The central assertion is the cross-source one: SEC publishes the same
acceptance instant twice, once as an Eastern wall clock in each filing's SGML
header and once as a UTC string in ``data.sec.gov``'s submissions JSON. Where
those two agree, they pin the conversion exactly — no tolerance, no rounding.
Where they disagree, the disagreement is itself recorded, because that is the
finding that rules the JSON out as a knowledge-time source.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.ingest.edgar.parse import (
    EDGAR_TIMEZONE,
    acceptance_datetime_to_utc,
    filing_index_headers_url,
    parse_daily_index,
    parse_filing_header,
    parse_quarter_listing,
    quarter_listing_url,
    quarter_of,
)
from backend.ingest.errors import PermanentSourceError
from backend.tests.ingest.test_edgar_fixtures import (
    crosscheck_records,
    read_fixture,
)

# The two accessions whose data.sec.gov acceptanceDateTime carries the Eastern
# wall clock with a spurious "Z" instead of a UTC conversion. Both are under
# CIK 1750284; the fixture's _provenance block records the discriminating test
# (17 CFR 232.13 filing-date assignment) that proves which reading is right.
_SUBMISSIONS_JSON_NOT_UTC = frozenset({"0001243233-24-000002", "0000950170-24-125427"})


def _submissions_instant(value: str) -> dt.datetime:
    """Parse a data.sec.gov ``acceptanceDateTime`` string as written."""
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")


def test_conversion_reproduces_secs_own_utc_value_exactly() -> None:
    """Our ET->UTC conversion equals SEC's published UTC instant, to the second.

    Eight real filings, two independent SEC expressions of each. For the six
    whose submissions JSON is a genuine UTC conversion, our result must equal
    it exactly — that is what proves ``<ACCEPTANCE-DATETIME>`` is Eastern and
    that the conversion handles both offsets.
    """
    agreeing = 0
    for record in crosscheck_records():
        accession = str(record["accession_number"])
        converted = acceptance_datetime_to_utc(str(record["sgml_acceptance_datetime"]))
        published = _submissions_instant(str(record["submissions_acceptance_datetime"]))
        if accession in _SUBMISSIONS_JSON_NOT_UTC:
            continue
        assert converted == published, accession
        agreeing += 1
    assert agreeing == len(crosscheck_records()) - len(_SUBMISSIONS_JSON_NOT_UTC)


def test_conversion_applies_the_offset_in_force_on_the_filings_own_date() -> None:
    """The offset is -05:00 or -04:00 by date, never a fixed one.

    A fixed offset would be wrong for half of every year. The captured set
    spans both, including a February filing whose Eastern evening is the next
    UTC day.
    """
    offsets = {
        str(record["accession_number"]): (
            _submissions_instant(str(record["submissions_acceptance_datetime"]))
            - dt.datetime.strptime(str(record["sgml_acceptance_datetime"]), "%Y%m%d%H%M%S").replace(
                tzinfo=dt.UTC
            )
        )
        for record in crosscheck_records()
        if str(record["accession_number"]) not in _SUBMISSIONS_JSON_NOT_UTC
    }
    # 2024-02-22 and 2024-03-08 are EST (-05:00); 2024-03-11, 2024-03-25 and
    # 2024-07-24 are EDT (-04:00). Both appear, so a fixed-offset implementation
    # cannot pass this test.
    assert offsets["0001225208-24-002776"] == dt.timedelta(hours=5)
    assert offsets["0001104659-24-032038"] == dt.timedelta(hours=5)
    assert offsets["0001225208-24-004041"] == dt.timedelta(hours=4)
    assert offsets["0001225208-24-004467"] == dt.timedelta(hours=4)
    assert offsets["0001104659-24-082105"] == dt.timedelta(hours=4)


def test_submissions_json_is_not_a_usable_knowledge_time_source() -> None:
    """The recorded counter-examples: two filings whose JSON value is Eastern, not UTC.

    Kept as an assertion rather than a comment so the evidence behind "read the
    SGML header, not data.sec.gov" lives in the test suite. Both records also
    carry the filing date that settles the reading: an 18:55:18 acceptance with
    a next-business-day filing date can only be Eastern (17 CFR 232.13(a)(2)).
    """
    by_accession = {str(r["accession_number"]): r for r in crosscheck_records()}
    for accession in _SUBMISSIONS_JSON_NOT_UTC:
        record = by_accession[accession]
        converted = acceptance_datetime_to_utc(str(record["sgml_acceptance_datetime"]))
        published = _submissions_instant(str(record["submissions_acceptance_datetime"]))
        assert converted != published, accession
        assert published.replace(tzinfo=None) == dt.datetime.strptime(  # noqa: DTZ007 — see above
            str(record["sgml_acceptance_datetime"]), "%Y%m%d%H%M%S"
        )
    form_d = by_accession["0001243233-24-000002"]
    accepted = acceptance_datetime_to_utc(str(form_d["sgml_acceptance_datetime"]))
    assert accepted.astimezone(EDGAR_TIMEZONE).time() > dt.time(17, 30)
    assert dt.date.fromisoformat(str(form_d["submissions_filing_date"])) == dt.date(2024, 12, 12)
    assert accepted.astimezone(EDGAR_TIMEZONE).date() == dt.date(2024, 12, 11)


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        # Last second of EST in 2024 and the first second of EDT: 01:59:59 ->
        # 06:59:59Z, then the clock jumps to 03:00:00 -> 07:00:00Z. One second
        # of local time, one second of UTC time, across a one-hour jump.
        ("20240310015959", "2024-03-10T06:59:59+00:00"),
        ("20240310030000", "2024-03-10T07:00:00+00:00"),
        # Autumn: 00:59:59 EDT then 02:00:00 EST, straddling the repeated hour.
        ("20241103005959", "2024-11-03T04:59:59+00:00"),
        ("20241103020000", "2024-11-03T07:00:00+00:00"),
    ],
)
def test_daylight_saving_boundaries_convert_correctly(local: str, expected: str) -> None:
    """The conversion is right on both sides of both 2024 transitions."""
    assert acceptance_datetime_to_utc(local) == dt.datetime.fromisoformat(expected)


@pytest.mark.parametrize(
    "local",
    [
        "20240310023000",  # spring forward: this local time does not exist
        "20241103013000",  # fall back: this local time names two instants
    ],
)
def test_transition_local_times_raise_rather_than_guess(local: str) -> None:
    """An unresolvable local time raises; picking one would fabricate an instant.

    Picking the earlier of an ambiguous pair would also fabricate an hour of
    lookahead, which is the specific failure invariant I1 exists to prevent.
    """
    with pytest.raises(PermanentSourceError, match="daylight-saving"):
        acceptance_datetime_to_utc(local)


@pytest.mark.parametrize("raw", ["", "2024031120130", "202403112013080", "2024031120130x", "not"])
def test_malformed_acceptance_values_raise(raw: str) -> None:
    """A value that is not 14 digits is refused, never coerced into an instant."""
    with pytest.raises(PermanentSourceError):
        acceptance_datetime_to_utc(raw)


def test_impossible_calendar_value_raises() -> None:
    """14 digits that do not name a real moment are refused."""
    with pytest.raises(PermanentSourceError, match="not a real date"):
        acceptance_datetime_to_utc("20240231120000")


def test_acceptance_precedes_filing_date_across_a_weekend() -> None:
    """Form D 0000950172-24-000037: knowable Friday evening, dated the Monday.

    17 CFR 232.13(a)(2): a submission commencing after 5:30 p.m. Eastern is
    deemed filed the next business day. Using the filing date as knowledge time
    would hide the filing for an extra weekend.
    """
    header = parse_filing_header(
        read_fixture("0000950172-24-000037-index-headers.html"),
        accession_number="0000950172-24-000037",
    )
    assert header.acceptance_datetime == dt.datetime(2024, 3, 8, 22, 59, 9, tzinfo=dt.UTC)
    assert header.filing_date == dt.date(2024, 3, 11)
    assert header.acceptance_datetime.astimezone(EDGAR_TIMEZONE).date() == dt.date(2024, 3, 8)


def test_acceptance_follows_filing_date_into_the_next_utc_day() -> None:
    """Form 4 0001225208-24-004041: the lookahead case, in the dangerous direction.

    17 CFR 232.13(a)(4) dates Forms 3/4/5 to the same business day until
    10 p.m. Eastern, so this filing carries filing date 2024-03-11 while its
    acceptance instant is 2024-03-12T00:13:08Z. Treating the filing date as a
    knowledge time — under any convention that places it at or before
    2024-03-11T23:59Z — asserts knowledge more than 24 hours before the filing
    existed.
    """
    header = parse_filing_header(
        read_fixture("0001225208-24-004041-index-headers.html"),
        accession_number="0001225208-24-004041",
    )
    assert header.acceptance_datetime == dt.datetime(2024, 3, 12, 0, 13, 8, tzinfo=dt.UTC)
    assert header.filing_date == dt.date(2024, 3, 11)
    filing_date_midnight_utc = dt.datetime(2024, 3, 11, tzinfo=dt.UTC)
    assert header.acceptance_datetime - filing_date_midnight_utc > dt.timedelta(hours=24)


def test_filing_header_carries_the_document_manifest() -> None:
    """Sequences are read as EDGAR states them, gaps included.

    Accession 0000950170-24-029225 lists sequences 1, 2, 3, 5 — no code may
    infer a count from the maximum sequence, and none does.
    """
    header = parse_filing_header(
        read_fixture("0000950170-24-029225-index-headers.html"),
        accession_number="0000950170-24-029225",
    )
    sequences = [document.sequence for document in header.documents]
    assert sequences == sorted(sequences)
    assert sequences[:4] == [1, 2, 3, 5]
    first = header.documents[0]
    assert (first.document_type, first.filename, first.description) == (
        "8-K",
        "ufi-20240305.htm",
        "8-K",
    )
    assert header.form_type == "8-K"
    assert header.period_of_report == dt.date(2024, 3, 5)


def test_absent_manifest_description_is_none_not_empty_string() -> None:
    """Forms 3/4/5 and Form D carry no <DESCRIPTION>; absence is recorded as absence."""
    header = parse_filing_header(
        read_fixture("0001225208-24-004041-index-headers.html"),
        accession_number="0001225208-24-004041",
    )
    assert len(header.documents) == 1
    assert header.documents[0].description is None
    assert header.documents[0].document_type == "4"


def test_header_for_a_different_accession_is_refused() -> None:
    """A response for another filing is never attributed to the one requested."""
    with pytest.raises(PermanentSourceError, match="different accession"):
        parse_filing_header(
            read_fixture("0001225208-24-004041-index-headers.html"),
            accession_number="0000950170-24-029225",
        )


def test_header_without_acceptance_datetime_is_refused() -> None:
    """No acceptance value, no row: nothing is substituted for a knowledge time."""
    body = read_fixture("0001225208-24-004041-index-headers.html")
    without = "\n".join(
        line for line in body.splitlines() if "ACCEPTANCE-DATETIME" not in line.upper()
    )
    with pytest.raises(PermanentSourceError, match="ACCEPTANCE-DATETIME"):
        parse_filing_header(without, accession_number="0001225208-24-004041")


def test_daily_index_collapses_duplicate_rows_and_counts_them() -> None:
    """EDGAR repeats a row per series/class; the collapse is measured, not silent.

    The real 2024-03-11 index repeats accession 0001683863-24-001302 fourteen
    times *under one CIK*. The sample keeps all fourteen, and the parser keeps
    one — a repeated ``(accession, CIK)`` pair carries no new information.
    """
    parsed = parse_daily_index(read_fixture("master.20240311.sample.idx"))
    pairs = [(entry.accession_number, entry.cik) for entry in parsed.entries]
    assert len(pairs) == len(set(pairs))
    assert parsed.rows_read - parsed.duplicate_rows_collapsed == len(parsed.entries)
    assert parsed.duplicate_rows_collapsed == 13
    assert "0001683863-24-001302" in {accession for accession, _ in pairs}


def test_one_accession_under_several_ciks_is_kept_once_per_cik() -> None:
    """A joint filing is many (filing, filer) rows and one submission.

    Accession 0001193805-24-000360 is a Form 4 the 2024-03-12 index lists under
    seven CIKs — six Deerfield entities and the issuer. All seven are real rows
    of the source; collapsing them would mean electing one filer to stand for
    the filing. ``distinct_accessions`` is what a connector fetches headers for,
    and it is smaller than the entry count exactly here.
    """
    parsed = parse_daily_index(read_fixture("master.20240312.sample.idx"))
    joint = [e for e in parsed.entries if e.accession_number == "0001193805-24-000360"]
    assert len(joint) == 7
    assert len({entry.cik for entry in joint}) == 7
    assert {entry.form_type for entry in joint} == {"4"}
    assert {entry.filing_date for entry in joint} == {dt.date(2024, 3, 12)}
    assert 1009258 in {entry.cik for entry in joint}
    assert parsed.distinct_accessions == 3
    assert len(parsed.entries) == 9


def test_daily_index_date_filed_can_precede_the_index_date() -> None:
    """A CORRESP disseminated 2024-03-11 carries Date Filed 2024-02-07.

    The daily index date and the filing date are different facts, which is why
    ``edgar_filing`` stores both.
    """
    parsed = parse_daily_index(read_fixture("master.20240311.sample.idx"))
    by_accession = {entry.accession_number: entry for entry in parsed.entries}
    corresp = by_accession["0000950170-24-012183"]
    assert corresp.form_type == "CORRESP"
    assert corresp.filing_date == dt.date(2024, 2, 7)


def test_daily_index_row_fields_are_read_as_edgar_wrote_them() -> None:
    """CIK, company name, form type and accession come straight off the row."""
    parsed = parse_daily_index(read_fixture("master.20240311.sample.idx"))
    by_accession = {entry.accession_number: entry for entry in parsed.entries}
    unifi = by_accession["0000950170-24-029225"]
    assert (unifi.cik, unifi.company_name, unifi.form_type) == (100726, "UNIFI INC", "8-K")
    assert unifi.document_path == "edgar/data/100726/0000950170-24-029225.txt"


def test_contradictory_duplicate_rows_are_refused() -> None:
    """One accession under one CIK described two ways: refuse rather than pick one.

    This is a genuine contradiction, unlike the same accession under two
    *different* CIKs, which the test above shows is ordinary.
    """
    body = read_fixture("master.20240311.sample.idx")
    line = next(
        line for line in body.splitlines() if line.startswith("100726|") and "|8-K|" in line
    )
    with pytest.raises(PermanentSourceError, match="contradicts itself"):
        parse_daily_index(f"{body}\n{line.replace('|8-K|', '|10-K|')}")


@pytest.mark.parametrize(
    "row",
    [
        "100726|UNIFI INC|8-K|2024-03-11|edgar/data/100726/0000950170-24-029225.txt",
        "100726|UNIFI INC|8-K|20240311|edgar/data/100726/not-an-accession.txt",
    ],
)
def test_unparseable_daily_index_rows_are_refused(row: str) -> None:
    """A row whose date or accession cannot be read raises; it is never skipped."""
    with pytest.raises(PermanentSourceError):
        parse_daily_index(f"CIK|Company Name|Form Type|Date Filed|File Name\n{row}")


def test_quarter_listing_yields_only_the_dates_edgar_publishes() -> None:
    """The listing is the source of truth about which index days exist."""
    listing = parse_quarter_listing(read_fixture("daily-index-2024-QTR1.sample.json"))
    assert [entry.index_date for entry in listing.files] == [
        dt.date(2024, 3, 8),
        dt.date(2024, 3, 11),
        dt.date(2024, 3, 12),
    ]
    assert [entry.file_name for entry in listing.files] == [
        "master.20240308.idx",
        "master.20240311.idx",
        "master.20240312.idx",
    ]
    assert listing.legacy_named_files == 0


def test_legacy_named_index_files_are_counted_not_dropped() -> None:
    """EDGAR's pre-1998 naming is out of scope, and says so in a number.

    The two legacy names below are the shapes EDGAR really used (``MMDDYY`` in
    1994, ``YYMMDD`` from 1995); they are unreadable without guessing a
    century, so they are excluded from coverage and counted.
    """
    listing = parse_quarter_listing(
        '{"directory": {"item": ['
        '{"name": "master.070194.idx"}, {"name": "master.950103.idx"}, '
        '{"name": "master.20240311.idx"}]}}'
    )
    assert [entry.index_date for entry in listing.files] == [dt.date(2024, 3, 11)]
    assert listing.legacy_named_files == 2


@pytest.mark.parametrize("body", ["not json", '{"directory": {}}', "[]"])
def test_unusable_quarter_listing_is_refused(body: str) -> None:
    """An unreadable listing raises; it never becomes "no days published"."""
    with pytest.raises(PermanentSourceError):
        parse_quarter_listing(body)


def test_url_builders_match_the_urls_the_fixtures_came_from() -> None:
    """The URLs the connector constructs are the ones the captures were taken from."""
    assert (
        quarter_listing_url(2024, 1)
        == "https://www.sec.gov/Archives/edgar/daily-index/2024/QTR1/index.json"
    )
    assert quarter_of(dt.date(2024, 3, 11)) == 1
    assert quarter_of(dt.date(2024, 12, 31)) == 4
    assert filing_index_headers_url(100726, "0000950170-24-029225") == (
        "https://www.sec.gov/Archives/edgar/data/100726/000095017024029225/"
        "0000950170-24-029225-index-headers.html"
    )


def test_url_builders_reject_impossible_inputs() -> None:
    """A bad quarter or accession raises rather than producing a URL that 403s."""
    with pytest.raises(ValueError, match="quarter must be"):
        quarter_listing_url(2024, 5)
    with pytest.raises(ValueError, match="accession"):
        filing_index_headers_url(100726, "0000950170240029225")
