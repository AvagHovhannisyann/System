"""P7.2: every date writing a filing uses, stripped.

Two kinds of input here, and the difference matters.

*Captured* — the compact ``YYYYMMDD`` and ``YYYYMMDDHHMMSS`` writings, the
``Mar 11, 2024`` in a daily index preamble, the ISO timestamps in a provenance
block. These come out of ``backend/tests/fixtures/edgar/``.

*Constructed* — the prose writings (``the quarter ended March 31, 2024``,
``fiscal 2024``, ``year-over-year``) that appear in 10-K and 10-Q *bodies*. The
committed fixtures are submission headers and index files, which do not contain
prose; the connector fetches bodies at run time and none is committed. Rather
than write a document and call it a filing — which is what I3 forbids — these
exercise the rules at the writings, the way
``backend/tests/ingest/test_edgar_parse.py`` exercises the timezone conversion
at local times EDGAR cannot emit. The strings below are date expressions, not
filing text, and nothing here claims otherwise.
"""

from __future__ import annotations

import pytest

from backend.extraction.anonymize import AnonymizerConfig, anonymize
from backend.extraction.rules import MaskKind
from backend.extraction.temporal import YEAR_PRECEDING_EXEMPTIONS, year_exemption_reason
from backend.tests.extraction.edgar_text import ADAPTHEALTH_FORM4, DAILY_INDEX, read

_DATE_WRITINGS = [
    ("2024-03-11", MaskKind.DATE),
    ("2024/03/11", MaskKind.DATE),
    ("March 11, 2024", MaskKind.DATE),
    ("March 11 2024", MaskKind.DATE),
    ("11 March 2024", MaskKind.DATE),
    ("Mar. 11, 2024", MaskKind.DATE),
    ("March 11th, 2024", MaskKind.DATE),
    ("3/11/2024", MaskKind.DATE),
    ("03-11-24", MaskKind.DATE),
    ("March 2024", MaskKind.DATE),
    ("March 11", MaskKind.DATE),
    ("December 31", MaskKind.DATE),
    ("11.03.2024", MaskKind.DATE),
    ("1.3.2024", MaskKind.DATE),
    ("11th of March, 2024", MaskKind.DATE),
    ("Sept 11, 2024", MaskKind.DATE),
    ("20240311", MaskKind.DATE),
    ("20240312190139", MaskKind.DATE),
    ("2024-03-12T19:01:39Z", MaskKind.DATE),
    ("2024-03-12 19:01:39", MaskKind.DATE),
    ("Q1 2024", MaskKind.PERIOD),
    ("Q1 FY2024", MaskKind.PERIOD),
    ("Q1'24", MaskKind.PERIOD),
    ("1Q24", MaskKind.PERIOD),
    ("fiscal 2024", MaskKind.PERIOD),
    ("fiscal year 2024", MaskKind.PERIOD),
    ("FY2024", MaskKind.PERIOD),
    ("FY 24", MaskKind.PERIOD),
    ("calendar 2024", MaskKind.PERIOD),
    ("quarter ended March 31, 2024", MaskKind.PERIOD),
    ("three months ended March 31, 2024", MaskKind.PERIOD),
    ("nine months ended September 30, 2023", MaskKind.PERIOD),
    ("fiscal year ended December 31, 2023", MaskKind.PERIOD),
    ("first quarter of 2024", MaskKind.PERIOD),
    ("fourth quarter", MaskKind.PERIOD),
    ("H1 2024", MaskKind.PERIOD),
    ("1H24", MaskKind.PERIOD),
    ("H1 FY24", MaskKind.PERIOD),
    ("first half of 2024", MaskKind.PERIOD),
    ("second half", MaskKind.PERIOD),
    ("prior year", MaskKind.RELATIVE_PERIOD),
    ("last quarter", MaskKind.RELATIVE_PERIOD),
    ("next fiscal year", MaskKind.RELATIVE_PERIOD),
    ("year-over-year", MaskKind.RELATIVE_PERIOD),
    ("year to date", MaskKind.RELATIVE_PERIOD),
    ("trailing twelve months", MaskKind.RELATIVE_PERIOD),
    ("a year ago", MaskKind.RELATIVE_PERIOD),
    ("5:30 p.m. ET", MaskKind.TIME),
    ("19:01:39", MaskKind.TIME),
    ("2023", MaskKind.YEAR),
    ("September", MaskKind.MONTH),
]


@pytest.mark.parametrize(("writing", "kind"), _DATE_WRITINGS)
def test_a_date_writing_is_replaced_by_one_placeholder_of_its_kind(
    writing: str, kind: MaskKind
) -> None:
    document = anonymize(writing)
    assert document.text == f"[{kind}_1]", writing
    assert [r.kind for r in document.replacements] == [kind]


def test_a_date_split_across_a_line_break_is_still_a_date() -> None:
    """Filing text wraps; a newline inside a date does not make it two things."""
    assert anonymize("March 11,\n2024").text == "[DATE_1]"
    assert anonymize("March\n11, 2024").text == "[DATE_1]"
    assert anonymize("the quarter ended March 31,\n2024").text == "the [PERIOD_1]"


def test_a_fiscal_period_phrase_masks_whole_rather_than_leaving_the_ordinal() -> None:
    """``fiscal 2024`` leaves no ``fiscal``, ``Q1 2024`` leaves no ``Q1``.

    The stricter reading of "strip all dates", chosen because a guarantee with
    an "except the harmless parts" clause cannot be checked. The cost is that
    the duration in ``three months ended ...`` is lost with the date; see the
    module docstring of :mod:`backend.extraction.temporal`.
    """
    for phrase in ("fiscal 2024", "Q1 2024", "the first quarter of fiscal 2024"):
        masked = anonymize(phrase).text
        assert "fiscal" not in masked.casefold()
        assert "Q1" not in masked
        assert "quarter" not in masked.casefold()


def test_equal_dates_share_a_placeholder_and_different_ones_do_not() -> None:
    text = "compared with March 11, 2024 and 2024-03-11, versus March 11, 2023"
    document = anonymize(text)
    assert document.text == ("compared with [DATE_1] and [DATE_1], versus [DATE_2]")


def test_placeholder_numbers_follow_first_appearance_not_chronology() -> None:
    """Ordering reveals order of mention, never the direction of time."""
    assert anonymize("2024-06-30 then 2019-01-02").text == "[DATE_1] then [DATE_2]"


def test_eight_digit_numbers_that_are_not_dates_are_left_alone() -> None:
    """``FILM NUMBER: 24743961`` is not 2474-39-61, and the fixture proves it.

    Without the calendar check the masker would have to choose between leaving
    filing dates in and redacting film numbers. Both errors are silent.
    """
    raw = read(ADAPTHEALTH_FORM4)
    assert "24743961" in raw
    masked = anonymize(raw, []).text
    assert "24743961" in masked
    assert "20240312" not in masked


def test_captured_compact_dates_are_masked() -> None:
    raw = read(ADAPTHEALTH_FORM4)
    for captured in ("20240312190139", "20240308", "19981022", "19960318"):
        assert captured in raw
    masked = anonymize(raw, []).text
    for captured in ("20240312190139", "20240308", "19981022", "19960318"):
        assert captured not in masked


def test_a_bare_year_is_masked_but_a_share_count_and_a_statute_are_not() -> None:
    """The three stated exemptions, each of which is a hole the detector reports."""
    assert anonymize("revenue rose in 2024").text == "revenue rose in [YEAR_1]"
    assert anonymize("2000 shares were sold").text == "2000 shares were sold"
    assert anonymize("registered under the 1934 Act").text == "registered under the 1934 Act"
    assert anonymize("a charge of $2024").text == "a charge of $2024"


def test_may_survives_as_a_standalone_month_and_this_is_deliberate() -> None:
    """``May`` is also a modal verb; masking every one would cost more than it buys.

    ``May 2024`` and ``May 11, 2024`` are still masked — only the bare month is
    spared, and the leak detector reports it at ``RESIDUAL``.
    """
    assert anonymize("May").text == "May"
    assert anonymize("we may proceed").text == "we may proceed"
    assert anonymize("May 2024").text == "[DATE_1]"
    assert anonymize("May 11, 2024").text == "[DATE_1]"
    assert anonymize("March").text == "[MONTH_1]"


def test_a_lower_case_month_word_is_left_alone_and_this_is_deliberate() -> None:
    """``mar``, ``march`` and ``august`` are English words before they are months.

    The bare-month rule is the one temporal rule that is case-sensitive. Masking
    the lower-case spellings was over-masking that damaged prose in every
    document rather than removing a date from one, and coherence is load-bearing
    here: P7.9 has to tell a score that fell because contamination was removed
    from one that fell because the text stopped reading as English. A date
    written in lower case is still caught — it is the *bare* month that is
    spared.
    """
    for word in ("mar", "march", "august", "sept"):
        assert anonymize(f"they {word} on").text == f"they {word} on", word
    assert anonymize("March").text == "[MONTH_1]"
    assert anonymize("march 11, 2024").text == "[DATE_1]"


def test_a_dotted_group_is_masked_only_when_it_can_be_a_calendar_day() -> None:
    """The precision half of the dotted-date rule, at its boundaries.

    Both day/month orderings are accepted because the point is to remove the
    date rather than to interpret it; everything that is not a date under either
    reading is left, which is what keeps outline numbering out of the mask.
    """
    for masked_writing in ("11.03.2024", "3.11.2024", "31.12.1999", "29.02.2024"):
        assert anonymize(masked_writing).text == "[DATE_1]", masked_writing
    for survivor in ("2.1.3", "32.13.2024", "11.03.1800", "0.0.2024", "13.13.2024"):
        assert anonymize(survivor).text == survivor, survivor


def test_a_fourteen_digit_run_is_masked_only_when_it_is_a_real_timestamp() -> None:
    """``<ACCEPTANCE-DATETIME>`` is the shape; a long account number is not.

    The date half and the clock half are both checked, because a fourteen-digit
    run in a filing is at least as likely to be an identifier as a timestamp,
    and redacting identifiers silently is its own kind of damage.
    """
    assert anonymize("20240312190139").text == "[DATE_1]"
    for survivor in ("24743961190139", "20240312250139", "20240312196139", "20240312190161"):
        assert anonymize(survivor).text == survivor, survivor


def test_a_year_outside_the_window_is_not_a_year() -> None:
    """1899 and 2100 are page numbers or quantities far more often than dates."""
    assert anonymize("item 1899 of the list").text == "item 1899 of the list"
    assert anonymize("item 2100 of the list").text == "item 2100 of the list"
    assert year_exemption_reason("1899", 0, 4) is not None
    assert year_exemption_reason("1900", 0, 4) is None


def test_an_unparseable_date_writing_still_masks_and_keys_on_its_own_text() -> None:
    """Keying is a coherence feature; masking is the guarantee. They differ.

    ``11 of March 2024`` has no format string, so it takes its own placeholder
    rather than joining the one for the same day written unambiguously. The date
    is gone either way — only the reader's ability to see that two mentions are
    one day is lost.
    """
    document = anonymize("11 of March 2024 and 2024-03-11")
    assert document.text == "[DATE_1] and [DATE_2]"


def test_the_two_spellings_of_the_year_guard_agree() -> None:
    """The lookbehind and the exemption set must not drift apart.

    ``year_exemption_reason`` is the single statement of when a four-digit
    number is not a year, and the leak detector's severity depends on it
    agreeing with the rule the masker actually runs. If they disagreed, the
    detector would call a deliberate exemption a failure (or bless a real miss).
    """
    for char in YEAR_PRECEDING_EXEMPTIONS:
        text = f"{char}2024"
        assert anonymize(text).text == text, char
        assert year_exemption_reason(text, len(char), len(char) + 4) is not None, char
    for char in "-/ (":
        text = f"{char}2024"
        assert anonymize(text).text == f"{char}[YEAR_1]", char
        assert year_exemption_reason(text, 1, 5) is None, char


def test_a_fiscal_year_end_code_survives_and_is_recorded_as_such() -> None:
    """``FISCAL YEAR END: 1231`` is a month-day code, not a date. It survives."""
    raw = read(ADAPTHEALTH_FORM4)
    assert "<FISCAL-YEAR-END>1231" in raw
    assert "<FISCAL-YEAR-END>1231" in anonymize(raw, []).text


def test_switching_a_temporal_rule_off_leaves_that_writing_standing() -> None:
    """The config switches are real, which is what makes the leak tests meaningful."""
    raw = read(DAILY_INDEX)
    assert "20240311" in raw
    masked = anonymize(raw, [], AnonymizerConfig(mask_dates=False)).text
    assert "20240311" in masked
