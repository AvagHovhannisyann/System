"""Stripping absolute time from a filing (P7.2).

Dates are the second half of the contamination control and, unlike names, they
cannot come from metadata: a filing is full of dates that nobody declared. So
these rules are written against the writings filings actually use, and the
scanner in :mod:`backend.extraction.leak` is written independently and more
broadly so that a form these rules miss is *reported* rather than assumed
absent.

Why the whole expression, not just the number
---------------------------------------------

``Q1 2024`` masks to ``[PERIOD_1]``, not to ``Q1 [YEAR_1]``. Leaving the
quarter ordinal or the word ``fiscal`` standing would be defensible on
readability grounds — neither is lookahead on its own — but the guarantee this
module has to make is the simple one: *no absolute time survives*. A guarantee
with an "except the parts we judged harmless" clause is not checkable, and the
P7.9 probe is built on the assumption that it is.

The cost is real and is stated rather than hidden: ``three months ended March
31, 2024`` becomes one ``[PERIOD_n]``, so the *duration* is lost along with the
date. What survives is the distinction between periods — equal expressions get
equal placeholders and different ones get different placeholders — which is
what a delta-extraction task (P7.4) actually needs from the text.

Numbering is by first appearance and keyed on the expression's normalized form,
so ``2024-03-11``, ``March 11, 2024`` and ``11 March 2024`` are one
``[DATE_1]`` while ``March 11, 2023`` is ``[DATE_2]``. The ordering of the
numbers reveals the order of first mention, never the direction of time.

Bare years
----------

A standalone ``2024`` is the single cheapest way to date a document, so
four-digit numbers in 1900-2099 are masked. Three contexts are exempted because
the number is not a date there and masking it would destroy the figure: a
currency symbol or decimal point immediately before it (``$2024``, ``1.2024``),
a unit word immediately after it (``2000 shares``), and a statute reference
(``1934 Act``). :func:`year_exemption_reason` is the single statement of that
policy, shared with the leak detector so the two cannot drift into disagreeing
about whether a survivor is a deliberate exemption or a real miss. Each
exemption is a hole by construction, and the detector reports what it left
behind at ``RESIDUAL`` severity for the operator to see.

Where the strictness stops
--------------------------

Two rules here are deliberately narrower than the rest, because widening them
cost more prose than they removed time:

* the bare-month rule is **case-sensitive**, so ``mar`` and ``august`` stay as
  the English words they usually are, and ``May`` is excluded outright;
* the half-year rule requires the year, because a bare ``H1`` is a share class
  about as often as it is a period.

Units: all rules operate on characters; :data:`MIN_YEAR`/:data:`MAX_YEAR` are
calendar years.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING, Final

from backend.extraction.rules import LEFT_BOUNDARY, RIGHT_BOUNDARY, MaskKind, MaskRule

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "DATE_ALTERNATION",
    "HALF_YEAR_ALTERNATION",
    "MAX_YEAR",
    "MIN_YEAR",
    "MONTH_ALTERNATION",
    "YEAR_PRECEDING_EXEMPTIONS",
    "YEAR_UNIT_EXEMPTIONS",
    "accession_rules",
    "is_compact_date",
    "is_compact_datetime",
    "is_dotted_date",
    "temporal_rules",
    "year_exemption_reason",
]

MIN_YEAR: Final = 1900
"""Lowest four-digit number read as a year (calendar year)."""

MAX_YEAR: Final = 2099
"""Highest four-digit number read as a year (calendar year)."""

_HOURS_PER_DAY: Final = 24
_MINUTES_PER_HOUR: Final = 60
_SECONDS_PER_MINUTE: Final = 60
"""Clock bounds for validating a compact ``YYYYMMDDHHMMSS`` stamp (units: h/min/s)."""

_MONTH_NAMES: Final = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_MONTH_ABBREVIATIONS: Final = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Jun",
    "Jul",
    "Aug",
    "Sept",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_ALL_MONTH_WRITINGS: Final = (*_MONTH_NAMES, *_MONTH_ABBREVIATIONS)

MONTH_ALTERNATION: Final = (
    "(?:" + "|".join(sorted(_ALL_MONTH_WRITINGS, key=len, reverse=True)) + r")\.?"
)
"""Month name or abbreviation, longest alternative first so ``March`` beats ``Mar``."""

_SPACE: Final = r"\s"
_DAY: Final = r"\d{1,2}(?:st|nd|rd|th)?"

_ISO_DATETIME: Final = (
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_ISO_DATE: Final = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
_NUMERIC_US: Final = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
_DOTTED_NUMERIC: Final = r"\d{1,2}\.\d{1,2}\.\d{4}"
"""Dotted numeric date — ``11.03.2024``.

Separated from :data:`_NUMERIC_US` because a dot is also an outline separator
(``2.1.3``) and a decimal point, so this form is only masked when
:func:`is_dotted_date` confirms the three numbers can be a calendar day. It is
here because the captured 6-K is from a German issuer and this is how German
filings write a date; without it a fully specified date survives untouched, and
the preceding ``.`` also makes the bare-year rule skip the ``2024`` inside it,
so nothing else catches it either.
"""
_MONTH_DAY_YEAR: Final = MONTH_ALTERNATION + _SPACE + r"+" + _DAY + r",?" + _SPACE + r"*\d{4}"
_DAY_MONTH_YEAR: Final = (
    _DAY + _SPACE + r"+(?:of" + _SPACE + r"+)?" + MONTH_ALTERNATION + r",?" + _SPACE + r"*\d{4}"
)
"""``11 March 2024`` and the legal-prose ``11th of March, 2024``."""
_MONTH_YEAR: Final = MONTH_ALTERNATION + _SPACE + r"+\d{4}"
_MONTH_DAY: Final = MONTH_ALTERNATION + _SPACE + r"+" + _DAY

DATE_ALTERNATION: Final = (
    "(?:"
    + "|".join(
        (
            _ISO_DATETIME,
            _ISO_DATE,
            _MONTH_DAY_YEAR,
            _DAY_MONTH_YEAR,
            _MONTH_YEAR,
            _MONTH_DAY,
            _NUMERIC_US,
            _DOTTED_NUMERIC,
        )
    )
    + ")"
)
"""Any explicit date writing, for embedding in the period-phrase rules."""

HALF_YEAR_ALTERNATION: Final = r"(?:H[12]|[12]H)"
"""Half-year ordinal, either spelling.

Foreign private issuers report on halves — the captured SAP 6-K is one — and
``1H24`` is a fully specified period that no quarter or year rule reaches: the
``24`` is two digits, so the bare-year rule does not see it either. Without this
the whole expression survives.
"""

YEAR_UNIT_EXEMPTIONS: Final = frozenset(
    {
        "acres",
        "act",
        "barrels",
        "basis",
        "bps",
        "customers",
        "employees",
        "feet",
        "holders",
        "hours",
        "locations",
        "megawatts",
        "members",
        "miles",
        "patients",
        "restricted",
        "share",
        "shares",
        "sq",
        "square",
        "stores",
        "subscribers",
        "tons",
        "unit",
        "units",
        "votes",
    }
)
"""Words that, immediately after a four-digit number, mean it is not a year.

``2000 shares`` is a quantity and ``1934 Act`` is a statute. Every entry widens
a hole in the date guarantee, so the list is short and each member is a noun
that cannot follow a year in ordinary prose.
"""

YEAR_PRECEDING_EXEMPTIONS: Final = frozenset(".,$€£¥#0123456789")
"""Characters immediately before a four-digit number that mean it is not a year.

A preceding digit or decimal/thousands separator says the number is part of a
longer one (``1.2024`` is a ratio); a preceding currency symbol says it is money
(``$2024``). The set deliberately excludes ``-`` and ``/``: a year inside an
already-handled date is dropped by span arbitration, not by this guard, and
excluding them here would leave the ``2024`` in a path like
``daily-index/2024/QTR1/`` standing.
"""

_CURRENCY_BEFORE: Final = r"(?<![\d.,$€£¥#])"
"""Left guard for the bare-year rule, the regex spelling of
:data:`YEAR_PRECEDING_EXEMPTIONS`.

It is stated twice — once as a lookbehind so the rule is cheap, once as a set so
:func:`year_exemption_reason` can explain a decision to the leak detector, which
scans with a wider boundary and has to tell a deliberate exemption apart from a
real miss. :func:`~backend.tests.extraction.test_temporal` asserts the two
spellings agree.
"""

_PERIOD_LEAD: Final = (
    r"(?:(?:first|second|third|fourth|1st|2nd|3rd|4th|three|six|nine|twelve|thirteen|"
    r"3|6|9|12)[\s-]+)?(?:fiscal\s+)?"
)

_RELATIVE_PATTERNS: Final = (
    (
        r"(?:last|next|prior|previous|preceding|following|coming|current|this|past|same|"
        r"comparable|corresponding|year-ago|prior-year|trailing)"
        r"[\s-]+(?:fiscal\s+)?(?:year|quarter|month|period|week|day)s?"
    ),
    r"(?:year|quarter|month)[\s-]?(?:over|on)[\s-]?(?:year|quarter|month)",
    r"(?:year|quarter|month)[\s-]to[\s-]date",
    r"(?:trailing\s+|last\s+)?twelve\s+months",
    r"TTM",
    r"(?:a|one|two|three|four|five)\s+(?:years?|quarters?|months?)\s+ago",
    r"(?:year|quarter|month)\s+ago",
    r"(?:today|yesterday|tomorrow)",
)


def _valid_date(digits: str) -> bool:
    """True when eight digits parse as a calendar date inside the year window."""
    try:
        year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
        dt.date(year, month, day)
    except ValueError:
        return False
    return MIN_YEAR <= year <= MAX_YEAR


def is_compact_date(match: re.Match[str]) -> bool:
    """Accept an eight-digit run only when it is a real ``YYYYMMDD`` date.

    EDGAR headers carry both — ``<FILING-DATE>20240312`` is a date and
    ``FILM NUMBER: 24743961`` is not (month 39 does not exist). Without this
    check the masker would either leave filing dates standing or redact film
    numbers, and both errors are silent.

    Args:
        match: A match whose group 0 is exactly eight digits.

    Returns:
        True when the digits parse as a date in [1900, 2099].
    """
    return _valid_date(match.group(0))


def is_compact_datetime(match: re.Match[str]) -> bool:
    """Accept a fourteen-digit run only when it is a real ``YYYYMMDDHHMMSS`` stamp.

    This is the shape of EDGAR's ``<ACCEPTANCE-DATETIME>``, the timestamp the
    whole ingestion layer is built around — so it is exactly the value most
    likely to be present and most damaging to leave in.

    Args:
        match: A match whose group 0 is exactly fourteen digits.

    Returns:
        True when the digits parse as a date and a time of day.
    """
    raw = match.group(0)
    if not _valid_date(raw[:8]):
        return False
    hour, minute, second = int(raw[8:10]), int(raw[10:12]), int(raw[12:14])
    return hour < _HOURS_PER_DAY and minute < _MINUTES_PER_HOUR and second < _SECONDS_PER_MINUTE


_FOLLOWING_WORD: Final = re.compile(r"\s{1,3}([A-Za-z']+)")
_TAIL_RADIUS: Final = 32
"""Characters after a four-digit number inspected for a unit word (characters)."""


def year_exemption_reason(text: str, start: int, end: int) -> str | None:
    """Say why the four-digit number at ``text[start:end]`` is not read as a year.

    The masker and the leak detector both need this judgement and must not
    disagree about it: the masker uses it to decide whether to mask, and the
    detector uses it to decide whether a surviving four-digit number is a
    deliberate exemption (``RESIDUAL``) or a real miss (``LEAK``). Two
    independent spellings of "is this a year" would make the detector's severity
    meaningless, so there is one.

    This is the *only* thing the detector borrows from the masker. It borrows a
    stated policy, not a pattern: the detector still finds candidates with its
    own, wider scanner.

    Args:
        text: The text the number sits in.
        start: Character offset of the number's first digit.
        end: Character offset one past its last digit.

    Returns:
        A human-readable reason, or ``None`` when the number should be read as a
        calendar year and therefore masked.
    """
    value = int(text[start:end])
    if not MIN_YEAR <= value <= MAX_YEAR:
        return f"outside the year window [{MIN_YEAR}, {MAX_YEAR}]"
    if start > 0 and text[start - 1] in YEAR_PRECEDING_EXEMPTIONS:
        return (
            f"preceded by {text[start - 1]!r}: part of a longer number, "
            f"or a currency amount, not a year"
        )
    following = _FOLLOWING_WORD.match(text[end : end + _TAIL_RADIUS])
    if following is not None and following.group(1).casefold() in YEAR_UNIT_EXEMPTIONS:
        return f"followed by {following.group(1)!r}: a quantity or a statute, not a year"
    return None


def is_dotted_date(match: re.Match[str]) -> bool:
    """Accept ``d.m.yyyy`` only when the three numbers can be a calendar day.

    Either ordering is accepted — ``11.03.2024`` is 11 March to a German filer
    and could be 3 November to an American one — because the point is to remove
    the date, not to interpret it. Deciding which convention a document uses
    would be a guess, and a wrong guess is worse than no guess: it is the day
    and month that would be reported wrongly downstream. See
    :func:`_normalize_date_key`, which deliberately does not merge this writing
    with an unambiguous one for the same reason.

    Rejecting the rest is what keeps outline numbering (``2.1.3``) and version
    strings out of the mask.

    Args:
        match: A match whose group 0 is ``d.m.yyyy`` or ``dd.mm.yyyy``.

    Returns:
        True when some reading of the three numbers is a real date in
        [:data:`MIN_YEAR`, :data:`MAX_YEAR`].
    """
    first, second, year_text = match.group(0).split(".")
    year = int(year_text)
    if not MIN_YEAR <= year <= MAX_YEAR:
        return False
    for day, month in ((int(first), int(second)), (int(second), int(first))):
        try:
            dt.date(year, month, day)
        except ValueError:
            continue
        return True
    return False


def _year_is_a_year(match: re.Match[str]) -> bool:
    """Accept a four-digit number only when nothing exempts it from being a year."""
    return year_exemption_reason(match.string, match.start(), match.end()) is None


_DATE_FORMATS: Final = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%y",
    "%m-%d-%y",
)
"""Writings recognised well enough to key on the calendar day they name.

A writing outside this list still masks; it just keys on its own text, so two
spellings of one day may take two placeholders. That is a coherence cost, never
a masking failure. ``11 of March 2024`` is a current example: the ``of`` is not
in any format string, so it keys on its own text.
"""

_STRPTIME_ALIASES: Final = ((re.compile(r"\bSept\b", re.IGNORECASE), "Sep"),)
"""Month writings ``strptime`` will not take, rewritten to ones it will.

``%b`` wants exactly three letters, so ``Sept 11, 2024`` would otherwise fail to
parse and take a second placeholder alongside ``September 11, 2024``. ``\\b``
keeps ``September`` itself untouched.
"""


def _normalize_date_key(matched: str) -> str:
    """Collapse a date writing to the calendar day it names, when parseable.

    Makes ``2024-03-11``, ``March 11, 2024`` and ``11 March 2024`` share one
    placeholder, which is what keeps a document internally consistent after
    masking. Unparseable writings fall back to their normalized text, so two
    spellings of the same thing may still get two placeholders — a stated
    limitation, not a correctness problem.
    """
    cleaned = re.sub(r"\s+", " ", matched).strip().rstrip(".")
    plain = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", cleaned.replace(".", ""), flags=re.I)
    for pattern, replacement in _STRPTIME_ALIASES:
        plain = pattern.sub(replacement, plain)
    for candidate in (cleaned, plain):
        for fmt in _DATE_FORMATS:
            try:
                return dt.datetime.strptime(candidate, fmt).date().isoformat()  # noqa: DTZ007
            except ValueError:
                continue
    if re.fullmatch(r"\d{8}", cleaned) and _valid_date(cleaned):
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    if re.fullmatch(r"\d{14}", cleaned) and _valid_date(cleaned[:8]):
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}T{cleaned[8:10]}:{cleaned[10:12]}"
    return cleaned.casefold()


def _rule(
    name: str,
    kind: MaskKind,
    body: str,
    priority: int,
    *,
    flags: int = re.IGNORECASE,
    accept: Callable[[re.Match[str]], bool] | None = None,
) -> MaskRule:
    """Compile one bounded temporal rule."""
    return MaskRule(
        name=name,
        kind=kind,
        pattern=re.compile(LEFT_BOUNDARY + body + RIGHT_BOUNDARY, flags),
        priority=priority,
        key_fn=_normalize_date_key,
        accept=accept,
    )


def temporal_rules(
    *,
    mask_dates: bool = True,
    mask_periods: bool = True,
    mask_bare_years: bool = True,
    mask_bare_months: bool = True,
    mask_relative_periods: bool = True,
    mask_times: bool = True,
) -> tuple[MaskRule, ...]:
    """Build the temporal rule set.

    Rules are returned in decreasing specificity. Arbitration is by span, not by
    this order (see :mod:`backend.extraction.rules`); the order only breaks
    exact ties.

    Args:
        mask_dates: Explicit calendar dates, including EDGAR's compact
            ``YYYYMMDD`` and ``YYYYMMDDHHMMSS`` forms.
        mask_periods: Fiscal and quarterly expressions, and ``... ended <date>``
            phrases.
        mask_bare_years: Standalone four-digit years, subject to
            :data:`YEAR_UNIT_EXEMPTIONS`.
        mask_bare_months: A month name with no day or year beside it. ``May`` is
            excluded because it is also a modal verb.
        mask_relative_periods: ``the prior quarter``, ``year-over-year`` and
            similar. These carry no absolute time; masking them is the stricter
            reading of "strip all dates" and costs some fluency.
        mask_times: Clock times, with or without a zone.

    Returns:
        Compiled rules ready for :func:`~backend.extraction.rules.apply_rules`.
    """
    rules: list[MaskRule] = []

    if mask_periods:
        rules.append(
            _rule(
                "period.ended_phrase",
                MaskKind.PERIOD,
                _PERIOD_LEAD
                + r"(?:month|quarter|year|period|week)s?\s+(?:then\s+)?"
                + r"end(?:ed|ing)(?:\s+(?:on|as\s+of))?\s+"
                + DATE_ALTERNATION,
                10,
            )
        )
        rules.extend(
            (
                _rule(
                    "period.quarter_year",
                    MaskKind.PERIOD,
                    r"Q[1-4](?:\s*(?:FY|CY)?\s*['\u2019]?\s*\d{2,4})?",
                    20,
                ),
                _rule(
                    "period.year_quarter",
                    MaskKind.PERIOD,
                    r"[1-4]Q\s*(?:FY|CY)?\s*['\u2019]?\s*\d{2,4}",
                    21,
                ),
                _rule(
                    "period.ordinal_quarter",
                    MaskKind.PERIOD,
                    r"(?:first|second|third|fourth)\s+(?:fiscal\s+)?quarter"
                    r"(?:\s+of(?:\s+fiscal)?\s+(?:\d{4}|\d{2}))?",
                    22,
                ),
                # The year is required here, unlike in ``period.quarter_year``.
                # A bare ``Q1`` is unambiguous; a bare ``H1`` is not (share
                # classes and exhibit numbers are written that way), and masking
                # it would redact text that carries no time at all. ``first
                # half`` without a year is covered by ``period.ordinal_half``.
                _rule(
                    "period.half_year",
                    MaskKind.PERIOD,
                    HALF_YEAR_ALTERNATION + r"\s*(?:FY|CY)?\s*['\u2019]?\s*\d{2,4}",
                    25,
                ),
                _rule(
                    "period.ordinal_half",
                    MaskKind.PERIOD,
                    r"(?:first|second)\s+(?:fiscal\s+)?half"
                    r"(?:\s+of(?:\s+fiscal)?\s+(?:\d{4}|\d{2}))?",
                    26,
                ),
                _rule(
                    "period.fiscal_year",
                    MaskKind.PERIOD,
                    r"(?:fiscal|financial|calendar)\s+(?:year\s+)?"
                    r"(?:ended\s+|ending\s+)?(?:\d{4}|\d{2})",
                    23,
                ),
                _rule(
                    "period.fy_abbreviation",
                    MaskKind.PERIOD,
                    r"(?:FY|CY)\s*['\u2019]?\s*\d{2,4}",
                    24,
                ),
            )
        )

    if mask_dates:
        rules.extend(
            (
                _rule("date.iso_datetime", MaskKind.DATE, _ISO_DATETIME, 30),
                _rule("date.iso", MaskKind.DATE, _ISO_DATE, 31),
                _rule("date.month_day_year", MaskKind.DATE, _MONTH_DAY_YEAR, 32),
                _rule("date.day_month_year", MaskKind.DATE, _DAY_MONTH_YEAR, 33),
                _rule("date.month_year", MaskKind.DATE, _MONTH_YEAR, 34),
                _rule("date.month_day", MaskKind.DATE, _MONTH_DAY, 35),
                _rule("date.numeric_us", MaskKind.DATE, _NUMERIC_US, 36),
                _rule(
                    "date.dotted_numeric",
                    MaskKind.DATE,
                    _DOTTED_NUMERIC,
                    39,
                    accept=is_dotted_date,
                ),
                _rule(
                    "date.compact_datetime",
                    MaskKind.DATE,
                    r"\d{14}",
                    37,
                    accept=is_compact_datetime,
                ),
                _rule("date.compact", MaskKind.DATE, r"\d{8}", 38, accept=is_compact_date),
            )
        )

    if mask_relative_periods:
        rules.extend(
            _rule(f"relative.{index}", MaskKind.RELATIVE_PERIOD, body, 40 + index)
            for index, body in enumerate(_RELATIVE_PATTERNS)
        )

    if mask_times:
        rules.append(
            _rule(
                "time.clock",
                MaskKind.TIME,
                r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?"
                r"\s*(?:ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|UTC|GMT)?",
                50,
            )
        )

    if mask_bare_years:
        rules.append(
            MaskRule(
                name="year.bare",
                kind=MaskKind.YEAR,
                pattern=re.compile(_CURRENCY_BEFORE + r"(?:19|20)\d{2}" + RIGHT_BOUNDARY),
                priority=60,
                key_fn=_normalize_date_key,
                accept=_year_is_a_year,
            )
        )

    if mask_bare_months:
        months = tuple(m for m in _ALL_MONTH_WRITINGS if m != "May")
        rules.append(
            # Case-sensitive, unlike every other temporal rule here. A month
            # name in a filing is capitalised; the lower-case spellings are
            # ordinary English — `mar` and `march` are verbs, `august` is an
            # adjective, `sept` is a spelling of a moat. Masking those was
            # over-masking that destroyed prose in every document rather than
            # removing a date from one, and coherence is load-bearing here: an
            # extraction score that falls because the text stopped reading as
            # English is indistinguishable, in P7.9, from one that falls
            # because contamination was removed. `May` stays excluded outright
            # because it is a modal verb even capitalised, at the start of a
            # sentence. The full-date rules stay case-insensitive: `march 11,
            # 2024` carries a date whatever its casing.
            _rule(
                "month.bare",
                MaskKind.MONTH,
                "(?:" + "|".join(sorted(months, key=len, reverse=True)) + r")\.?",
                70,
                flags=0,
            )
        )

    return tuple(rules)


def accession_rules(*, mask_accession_numbers: bool = True) -> Sequence[MaskRule]:
    """Rule for EDGAR accession numbers, which name both the filer and the year.

    ``0001193805-24-000360`` identifies the filing agent, the year and the
    sequence. It is self-describing enough to match without metadata, and it is
    a direct route to re-identification, so it is masked by default.

    Args:
        mask_accession_numbers: Set False to leave accession numbers in place.

    Returns:
        Zero or one rule.
    """
    if not mask_accession_numbers:
        return ()
    return (
        MaskRule(
            name="edgar.accession",
            kind=MaskKind.ACCESSION,
            pattern=re.compile(LEFT_BOUNDARY + r"\d{10}-\d{2}-\d{6}" + RIGHT_BOUNDARY),
            priority=5,
        ),
    )
