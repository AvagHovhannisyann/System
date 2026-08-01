"""The anonymization pass itself: document in, safe payload plus mapping out (P7.2).

This is the control that lets a historical filing be read by a model without
the model answering from what it remembers happened next. If the model can
name the company and date the document, a "prediction" extracted from it may be
recall rather than reading, and the resulting feature is a lookahead — the same
class of failure as I1, arriving through the prompt instead of through a query.

So the pass removes the two things that make recall possible: **who** (declared
company names in every writing the filing uses, tickers, named individuals,
registry identifiers) and **when** (every date, period and clock time). What it
leaves is a document that still reads as itself, because coherence is not a
nicety here: an extraction score that drops because the text became unreadable
is indistinguishable, downstream, from one that drops because contamination was
removed, and P7.9 has to tell those apart.

What this does not do, stated plainly
-------------------------------------

Masking is metadata-driven and token-based, and both of those choices have
edges. These are the known ones. Each is a residual channel the P7.9 probe will
measure through, which is the point of writing them down rather than claiming
completeness.

*Who:*

* **Nicknames and undeclared referents.** A company the filing calls only by a
  brand name nobody passed in survives, and so does an officer referred to only
  by a first name. Nothing here infers an entity it was not told about.
* **Names inside identifiers and markup.** ``e619363_4-ahcorp.xml`` is a real
  document filename from a captured AdaptHealth filing; ``ahcorp`` sits inside a
  token, and token-boundary matching does not reach it. A name split by markup
  rather than whitespace (``Adapt<b>Health</b>``) is the same problem: the
  flexible-whitespace patterns match across a line break, not across a tag.
  Substring matching would reach both and would redact fragments of ordinary
  words throughout the document, so this is left and reported.
* **Spaced variants of closed compounds.** ``Adapt Health`` for
  ``AdaptHealth`` is not matched; seeing inside a compound needs word
  boundaries that are not in the declared name.
* **Undeclared registry numbers.** IRS numbers, file and film numbers, phone
  numbers and street addresses are not detected; pass them as
  :func:`~backend.extraction.entities.identifier` entities if they matter.
* **Lower-case tickers.** The ticker rule is case-sensitive because many
  symbols are ordinary words (``ALL``, ``KEY``, ``ON``); ``ahco`` in a URL slug
  survives.
* **Surnames that are ordinary words.** ``May``, ``March`` and ``Will`` are
  exempt from the surname-alone rule
  (:data:`~backend.extraction.surface.COMMON_WORD_SURNAMES`), so a person
  referred to only as "Ms. May" survives. Full-name writings still mask.

*When:*

* **Unusual date writings.** The rules cover the writings observed in EDGAR
  text and the common prose forms; a filing inventing another one is a miss.
  This is why :mod:`backend.extraction.leak` scans with independently written,
  broader patterns rather than re-running these rules.
* **Lower-case bare months.** ``march`` and ``august`` are left, because they
  are English words far more often than they are months in a filing. A
  lower-case *date* (``march 11, 2024``) is still masked.
* **Fiscal-year-end codes.** ``FISCAL YEAR END: 1231`` is a month-day code, not
  a date, and survives.
* **Bare years in three contexts.** A four-digit number attached to a currency
  symbol or decimal point, or followed by a unit word or ``Act``, is read as
  money, a quantity or a statute rather than a year. See
  :func:`~backend.extraction.temporal.year_exemption_reason`; the leak detector
  reports each one at ``RESIDUAL``.
* **Durations.** ``three months ended March 31, 2024`` masks whole, so the
  length of the period is lost with the date it names.

*Both:*

* **Over-masking is the chosen direction of error.** A distinctive leading
  token is masked everywhere it appears, and a declared surname is masked even
  where it reads as a common noun. That costs readability, is visible in
  :attr:`~backend.extraction.entities.AnonymizedDocument.replacements`, and is
  switchable per-run. Under-masking is not visible anywhere, which is why it is
  not the direction chosen.

Units: character offsets throughout; no timestamps are produced, only removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.extraction.entities import AnonymizedDocument, EntityKind, EntityMapping
from backend.extraction.rules import PLACEHOLDER_SHAPE, apply_rules
from backend.extraction.surface import entity_rules
from backend.extraction.temporal import accession_rules, temporal_rules

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.extraction.entities import Entity
    from backend.extraction.rules import MaskRule

__all__ = ["AnonymizerConfig", "anonymize", "build_rules"]


@dataclass(frozen=True, slots=True)
class AnonymizerConfig:
    """Which masking rules are active.

    Defaults are the strict setting — everything on. Each switch that turns
    something off widens a re-identification channel, so the defaults are what
    the extraction pipeline should run with and the switches exist for
    diagnosis and for the leak-detector tests, which need a deliberately
    under-masked document to prove the detector fires.

    Attributes:
        mask_leading_token: Mask a company's distinctive first token alone.
        mask_surname_alone: Mask a person's surname alone.
        mask_dates: Mask explicit calendar dates.
        mask_periods: Mask fiscal and quarterly period expressions.
        mask_bare_years: Mask standalone four-digit years.
        mask_bare_months: Mask standalone month names.
        mask_relative_periods: Mask relative period expressions.
        mask_times: Mask clock times.
        mask_accession_numbers: Mask EDGAR accession numbers.
        mask_tickers: Mask declared tickers.
        mask_identifiers: Mask declared registry identifiers.
    """

    mask_leading_token: bool = True
    mask_surname_alone: bool = True
    mask_dates: bool = True
    mask_periods: bool = True
    mask_bare_years: bool = True
    mask_bare_months: bool = True
    mask_relative_periods: bool = True
    mask_times: bool = True
    mask_accession_numbers: bool = True
    mask_tickers: bool = True
    mask_identifiers: bool = True


def build_rules(
    entities: Sequence[Entity], config: AnonymizerConfig | None = None
) -> tuple[MaskRule, ...]:
    """Assemble the full rule set for ``entities`` under ``config``.

    Exposed separately from :func:`anonymize` so the operator's document
    inspector (§6.5) can show which rules were in force for a stored
    extraction, and so tests can assert on the rule set without running it.

    Args:
        entities: Declared entities, in a stable order.
        config: Rule switches. ``None`` means the strict defaults.

    Returns:
        Entity rules followed by temporal and structural rules. Order is
        significant only for exact ties (see :mod:`backend.extraction.rules`).
    """
    settings = config if config is not None else AnonymizerConfig()
    selected = [
        entity
        for entity in entities
        if not (entity.kind is EntityKind.TICKER and not settings.mask_tickers)
        and not (entity.kind is EntityKind.IDENTIFIER and not settings.mask_identifiers)
    ]
    rules: list[MaskRule] = list(
        entity_rules(
            selected,
            mask_leading_token=settings.mask_leading_token,
            mask_surname_alone=settings.mask_surname_alone,
        )
    )
    rules.extend(accession_rules(mask_accession_numbers=settings.mask_accession_numbers))
    rules.extend(
        temporal_rules(
            mask_dates=settings.mask_dates,
            mask_periods=settings.mask_periods,
            mask_bare_years=settings.mask_bare_years,
            mask_bare_months=settings.mask_bare_months,
            mask_relative_periods=settings.mask_relative_periods,
            mask_times=settings.mask_times,
        )
    )
    return tuple(rules)


def anonymize(
    text: str,
    entities: Sequence[Entity] = (),
    config: AnonymizerConfig | None = None,
) -> AnonymizedDocument:
    """Mask ``text`` and return the payload alongside the mapping that reverses it.

    Deterministic: the same text, entities and config always produce the same
    output, byte for byte. The P7.9 contamination probe compares two scorings of
    one document, and a masker whose output drifted between runs would make that
    comparison measure itself.

    Args:
        text: The document. Not modified.
        entities: What is known to identify it — filer and issuer names with
            their former names, reporting owners, tickers, CIKs. Declaration
            order fixes the placeholder numbers, so keep it stable across runs.
        config: Rule switches. ``None`` means the strict defaults.

    Returns:
        An :class:`~backend.extraction.entities.AnonymizedDocument`. Send
        :attr:`~backend.extraction.entities.AnonymizedDocument.text` and
        nothing else.
    """
    pre_existing = tuple(dict.fromkeys(PLACEHOLDER_SHAPE.findall(text)))
    application = apply_rules(text, build_rules(entities, config))
    return AnonymizedDocument(
        text=application.text,
        mapping=EntityMapping(entries=application.allocated),
        replacements=application.replacements,
        pre_existing_placeholders=pre_existing,
    )
