"""The leak detector: checking the anonymization instead of trusting it (P7.2).

The masker is a pile of regular expressions. Regular expressions miss things,
and a miss is silent — the document still reads fine, the extraction still
returns a number, and the contamination the P7.9 probe eventually measures gets
attributed to the model rather than to the masking. This module exists so the
guarantee is *checked* on every document instead of assumed.

It is deliberately **not** built from the masker's rules. Re-running them could
only ever confirm that they do what they do; a rule that fails to match still
fails to match on the second pass. So the scanners here are written
independently and more broadly:

* names are searched for over a whitespace-collapsed copy of the anonymized
  text, which catches a name the masker missed because a line break fell inside
  it;
* the temporal scanner accepts anything date-shaped — any month name, any
  four-digit number in the year window, any slashed or dashed numeric group,
  any eight- or fourteen-digit run that parses as a date — and it applies none
  of the masker's *boundary* guards, so it can see a year the masker's rule
  never even looked at.

One thing is shared rather than duplicated:
:func:`~backend.extraction.temporal.year_exemption_reason`, the statement of
when a four-digit number is not a year. It is used here only to *classify* a
finding the detector made on its own, never to decide whether to look. Writing a
second, independent opinion on that question would not make the detector
stronger; it would only make the two disagree, and a detector that calls the
masker's documented behaviour a failure is a detector whose failures get
ignored.

Being broader means it fires on things the masker left on purpose. That is the
intended behaviour, and it is why findings carry a severity:

``LEAK``
    A declared entity, or an unambiguous date, survived. This must be zero.
``RESIDUAL``
    Something date-shaped or name-fragment-shaped survived that the masker's
    stated limits predict — a bare year the unit exemption spared, a single
    generic token of a multi-word name. Not a pass/fail, but the number belongs
    in the operator's quality view (§6.5) because it is the surface through
    which contamination could still arrive.
``INFO``
    A structural observation, e.g. the source document already contained
    placeholder-shaped text, which makes re-identification ambiguous.

Units: ``start``/``end`` are character offsets into the *anonymized* text, and
counts are occurrences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from backend.extraction.entities import Entity, EntityKind
from backend.extraction.rules import LEFT_BOUNDARY, PLACEHOLDER_SHAPE, RIGHT_BOUNDARY
from backend.extraction.surface import COMMON_WORD_SURNAMES, GENERIC_NAME_TOKENS, core_tokens
from backend.extraction.temporal import (
    HALF_YEAR_ALTERNATION,
    MONTH_ALTERNATION,
    is_compact_date,
    is_compact_datetime,
    is_dotted_date,
    year_exemption_reason,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "LeakFinding",
    "LeakKind",
    "LeakReport",
    "Severity",
    "detect_leaks",
]

_CONTEXT_RADIUS: Final = 40
"""Characters of surrounding text carried on a finding, for the operator view."""


class Severity(StrEnum):
    """How much a finding matters."""

    LEAK = "LEAK"
    RESIDUAL = "RESIDUAL"
    INFO = "INFO"


class LeakKind(StrEnum):
    """What kind of thing survived."""

    COMPANY_NAME = "COMPANY_NAME"
    NAME_FRAGMENT = "NAME_FRAGMENT"
    PERSON_NAME = "PERSON_NAME"
    TICKER = "TICKER"
    IDENTIFIER = "IDENTIFIER"
    DATE = "DATE"
    YEAR = "YEAR"
    MONTH = "MONTH"
    ACCESSION = "ACCESSION"
    PLACEHOLDER_COLLISION = "PLACEHOLDER_COLLISION"


@dataclass(frozen=True, slots=True)
class LeakFinding:
    """One thing the detector found in the anonymized text.

    Attributes:
        kind: What sort of residue this is.
        severity: ``LEAK`` findings must be zero; the rest are for review.
        matched: The exact text found.
        start: Character offset in the anonymized text.
        end: Character offset one past the match.
        context: Surrounding characters, for the document inspector.
        detail: Why this fired, in words.
    """

    kind: LeakKind
    severity: Severity
    matched: str
    start: int
    end: int
    context: str
    detail: str


@dataclass(frozen=True, slots=True)
class LeakReport:
    """Everything the detector found, plus what it expected to find.

    Attributes:
        findings: All findings, in scan order.
        occurrences_in_original: Per declared entity name, how many surface
            occurrences the same scanners found in the *original* text. A zero
            here means the anonymized document proves nothing about that
            entity — the name was never in the document to begin with — and a
            test asserting "no leaks" would be vacuous for it.
    """

    findings: tuple[LeakFinding, ...]
    occurrences_in_original: dict[str, int]

    @property
    def leaks(self) -> tuple[LeakFinding, ...]:
        """Findings at ``LEAK`` severity — the ones that fail the guarantee."""
        return tuple(f for f in self.findings if f.severity is Severity.LEAK)

    @property
    def residuals(self) -> tuple[LeakFinding, ...]:
        """Findings at ``RESIDUAL`` severity — the stated, measured limits."""
        return tuple(f for f in self.findings if f.severity is Severity.RESIDUAL)

    @property
    def clean(self) -> bool:
        """True when nothing leaked. Residual and info findings do not fail it."""
        return not self.leaks

    def summary(self) -> str:
        """One-line human summary, safe to log: counts only, never the values."""
        return f"leaks={len(self.leaks)} residuals={len(self.residuals)} total={len(self.findings)}"


_WHITESPACE = re.compile(r"\s+")


def _collapsed(text: str) -> tuple[str, list[int]]:
    r"""Collapse whitespace runs to single spaces, keeping offsets into the source.

    A name broken across a line break (``AdaptHealth\n  Corp.``) is one name.
    Searching the collapsed copy finds it; the offset table maps the hit back to
    where it actually is, so a finding still points at real text.
    """
    out: list[str] = []
    offsets: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                out.append(" ")
                offsets.append(index)
            previous_space = True
            continue
        out.append(char)
        offsets.append(index)
        previous_space = False
    offsets.append(len(text))
    return "".join(out), offsets


def _context(text: str, start: int, end: int) -> str:
    """Return the match with up to :data:`_CONTEXT_RADIUS` characters either side."""
    left = max(0, start - _CONTEXT_RADIUS)
    right = min(len(text), end + _CONTEXT_RADIUS)
    return _WHITESPACE.sub(" ", text[left:right]).strip()


def _finding(
    kind: LeakKind,
    severity: Severity,
    text: str,
    start: int,
    end: int,
    detail: str,
) -> LeakFinding:
    """Build a finding with its context filled in."""
    return LeakFinding(
        kind=kind,
        severity=severity,
        matched=text[start:end],
        start=start,
        end=end,
        context=_context(text, start, end),
        detail=detail,
    )


def _literal_pattern(value: str) -> re.Pattern[str]:
    """Case-insensitive, whitespace-flexible, word-bounded literal search."""
    parts = [re.escape(p) for p in re.split(r"\s+", value.strip()) if p]
    body = r"\s+".join(parts)
    return re.compile(LEFT_BOUNDARY + body + RIGHT_BOUNDARY, re.IGNORECASE)


def _search(text: str, offsets: list[int], pattern: re.Pattern[str]) -> Iterator[tuple[int, int]]:
    """Yield source offsets for every match of ``pattern`` in the collapsed copy."""
    for match in pattern.finditer(text):
        yield offsets[match.start()], offsets[match.end()]


def _entity_probes(entity: Entity) -> list[tuple[str, LeakKind, Severity, str]]:
    """Surface forms this detector looks for, with the severity each carries.

    Whole declared writings are ``LEAK``: they were given to the masker and it
    was supposed to remove them. A single distinctive token of a multi-word name
    is ``RESIDUAL``: masking it is the masker's optional, precision-losing
    behaviour, so its presence is a limit to review rather than a failure.
    """
    kind = {
        EntityKind.COMPANY: LeakKind.COMPANY_NAME,
        EntityKind.PERSON: LeakKind.PERSON_NAME,
        EntityKind.TICKER: LeakKind.TICKER,
        EntityKind.IDENTIFIER: LeakKind.IDENTIFIER,
    }[entity.kind]
    probes: list[tuple[str, LeakKind, Severity, str]] = [
        (writing, kind, Severity.LEAK, "declared writing survived anonymization")
        for writing in entity.writings
    ]
    if entity.kind is EntityKind.PERSON and entity.surname:
        # A surname that is also an ordinary English word is one the masker
        # states it will not mask on its own (COMMON_WORD_SURNAMES: masking
        # every "May" would destroy more than it protects). Reporting that as a
        # LEAK would make the pass/fail gate unachievable for such a person and
        # would drown the real failures, so it is reported as the stated limit
        # it is — visible, counted, and not a failure.
        common = entity.surname.casefold() in COMMON_WORD_SURNAMES
        probes.append(
            (
                entity.surname,
                LeakKind.PERSON_NAME,
                Severity.RESIDUAL if common else Severity.LEAK,
                (
                    "surname is also a common English word, which the masker's "
                    "stated exemption spares"
                    if common
                    else "surname survived anonymization"
                ),
            )
        )
    if entity.kind is EntityKind.COMPANY:
        for writing in entity.writings:
            tokens = core_tokens(writing)
            if len(tokens) < 2:
                continue
            for token in tokens:
                if len(token) < 4 or token.casefold() in GENERIC_NAME_TOKENS:
                    continue
                probes.append(
                    (
                        token,
                        LeakKind.NAME_FRAGMENT,
                        Severity.RESIDUAL,
                        "a token of a multi-word declared name survived",
                    )
                )
    if entity.kind is EntityKind.IDENTIFIER:
        for writing in entity.writings:
            if writing.isdigit():
                probes.append(
                    (
                        writing.lstrip("0") or writing,
                        LeakKind.IDENTIFIER,
                        Severity.LEAK,
                        "unpadded form of a declared identifier survived",
                    )
                )
    return probes


_TEMPORAL_PROBES: Final[tuple[tuple[str, str, LeakKind, Severity, str], ...]] = (
    (
        "date.iso",
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        LeakKind.DATE,
        Severity.LEAK,
        "ISO-style date survived",
    ),
    (
        "date.numeric",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        LeakKind.DATE,
        Severity.LEAK,
        "numeric date survived",
    ),
    (
        "date.month_name",
        MONTH_ALTERNATION + r"[\s]+\d{1,2}(?:st|nd|rd|th)?(?:,?[\s]*\d{4})?",
        LeakKind.DATE,
        Severity.LEAK,
        "month-name date survived",
    ),
    (
        "date.month_year",
        MONTH_ALTERNATION + r"[\s]+\d{4}",
        LeakKind.DATE,
        Severity.LEAK,
        "month and year survived",
    ),
    (
        "period.quarter",
        r"(?:Q[1-4]|[1-4]Q)[\s]*['\u2019]?[\s]*(?:FY|CY)?[\s]*\d{2,4}",
        LeakKind.DATE,
        Severity.LEAK,
        "quarter-and-year expression survived",
    ),
    (
        "period.fiscal",
        r"(?:fiscal|FY|CY)[\s]*(?:year[\s]+)?['\u2019]?[\s]*\d{2,4}",
        LeakKind.DATE,
        Severity.LEAK,
        "fiscal-period expression survived",
    ),
    (
        "period.half_year",
        HALF_YEAR_ALTERNATION + r"[\s]*['\u2019]?[\s]*(?:FY|CY)?[\s]*\d{2,4}",
        LeakKind.DATE,
        Severity.LEAK,
        "half-year expression survived",
    ),
    (
        "date.accession",
        r"\d{10}-\d{2}-\d{6}",
        LeakKind.ACCESSION,
        Severity.LEAK,
        "EDGAR accession number survived",
    ),
)


def _scan_temporal(text: str, offsets: list[int], collapsed: str) -> list[LeakFinding]:
    """Run the independent, deliberately broad temporal scanners."""
    findings: list[LeakFinding] = []
    for _name, body, kind, severity, detail in _TEMPORAL_PROBES:
        pattern = re.compile(LEFT_BOUNDARY + body + RIGHT_BOUNDARY, re.IGNORECASE)
        findings.extend(
            _finding(kind, severity, text, start, end, detail)
            for start, end in _search(collapsed, offsets, pattern)
        )

    compact = re.compile(LEFT_BOUNDARY + r"\d{14}" + RIGHT_BOUNDARY)
    for match in compact.finditer(text):
        if is_compact_datetime(match):
            findings.append(
                _finding(
                    LeakKind.DATE,
                    Severity.LEAK,
                    text,
                    match.start(),
                    match.end(),
                    "compact YYYYMMDDHHMMSS timestamp survived",
                )
            )
    dotted = re.compile(LEFT_BOUNDARY + r"\d{1,2}\.\d{1,2}\.\d{4}" + RIGHT_BOUNDARY)
    for match in dotted.finditer(text):
        if is_dotted_date(match):
            findings.append(
                _finding(
                    LeakKind.DATE,
                    Severity.LEAK,
                    text,
                    match.start(),
                    match.end(),
                    "dotted numeric date survived",
                )
            )
    eight = re.compile(LEFT_BOUNDARY + r"\d{8}" + RIGHT_BOUNDARY)
    for match in eight.finditer(text):
        if is_compact_date(match):
            findings.append(
                _finding(
                    LeakKind.DATE,
                    Severity.LEAK,
                    text,
                    match.start(),
                    match.end(),
                    "compact YYYYMMDD date survived",
                )
            )

    # The scanner's own boundary is wider than the masker's: it matches a
    # four-digit number after "$" or ".", which the masker's rule never sees.
    # That is on purpose — the detector must be able to *find* what the masker
    # skipped — but a hit there is a stated exemption, not a miss, so the
    # severity comes from the one shared statement of the policy.
    #
    # The alternation spans exactly [MIN_YEAR, MAX_YEAR], so the one reason
    # `year_exemption_reason` can give that is *not* an exemption — being
    # outside the window — cannot arise here, and every reason it does give is
    # a deliberate one.
    year = re.compile(LEFT_BOUNDARY + r"(?:19|20)\d{2}" + RIGHT_BOUNDARY)
    for match in year.finditer(text):
        reason = year_exemption_reason(text, match.start(), match.end())
        findings.append(
            _finding(
                LeakKind.YEAR,
                Severity.RESIDUAL if reason is not None else Severity.LEAK,
                text,
                match.start(),
                match.end(),
                (
                    f"four-digit year left by a stated exemption: {reason}"
                    if reason is not None
                    else "bare four-digit year survived"
                ),
            )
        )

    # Case-sensitive, matching the masker's bare-month rule. A lower-case
    # "march" or "august" is an ordinary English word that the masker
    # deliberately leaves (see the note on `month.bare`), and reporting every
    # one of them would bury the capitalised months that actually survived —
    # which are the ones worth an operator's attention.
    month = re.compile(LEFT_BOUNDARY + MONTH_ALTERNATION + RIGHT_BOUNDARY)
    findings.extend(
        _finding(
            LeakKind.MONTH,
            Severity.RESIDUAL,
            text,
            match.start(),
            match.end(),
            "standalone month name survived",
        )
        for match in month.finditer(text)
    )
    return findings


def detect_leaks(
    *,
    original: str,
    anonymized: str,
    entities: Sequence[Entity] = (),
    scan_dates: bool = True,
) -> LeakReport:
    """Report every declared entity or date-shaped string still in ``anonymized``.

    The check that matters is ``report.leaks == ()``. Everything else is
    measurement: residual findings are the stated limits of the masker, counted
    so they can be watched rather than assumed constant.

    Args:
        original: The document before masking. Used only to count how many
            occurrences of each entity were there to remove, so a vacuous
            "nothing leaked because nothing was ever there" result is visible.
        anonymized: The document after masking — the text that would be sent.
        entities: The same declarations the masker was given.
        scan_dates: Run the temporal scanners. Off is for callers that
            deliberately kept dates (there is no such caller in the extraction
            pipeline; it exists so date-free unit tests stay readable).

    Returns:
        A :class:`LeakReport`.
    """
    collapsed, offsets = _collapsed(anonymized)
    original_collapsed, original_offsets = _collapsed(original)

    findings: list[LeakFinding] = []
    counted: dict[str, int] = {}

    for entity in entities:
        for value, kind, severity, detail in _entity_probes(entity):
            pattern = _literal_pattern(value)
            counted[value] = counted.get(value, 0) + sum(
                1 for _ in _search(original_collapsed, original_offsets, pattern)
            )
            findings.extend(
                _finding(kind, severity, anonymized, start, end, detail)
                for start, end in _search(collapsed, offsets, pattern)
            )

    if scan_dates:
        findings.extend(_scan_temporal(anonymized, offsets, collapsed))

    findings.extend(
        _finding(
            LeakKind.PLACEHOLDER_COLLISION,
            Severity.INFO,
            original,
            match.start(),
            match.end(),
            "source document already contained placeholder-shaped text",
        )
        for match in PLACEHOLDER_SHAPE.finditer(original)
    )

    findings.sort(key=lambda f: (f.start, f.end, f.kind))
    return LeakReport(findings=tuple(findings), occurrences_in_original=counted)
