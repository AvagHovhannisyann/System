"""Accounting-language shift: changes in how the same item is described (P7.4).

The fourth delta §5-P7 names. The construct is the *description*, holding the
item fixed: the same revenue policy, the same set of critical estimates, the
same adjusted measures — described differently. That framing is what makes it a
delta rather than a scoring of accounting quality, which would be a level, would
be dominated by industry, and would be a claim this platform has no business
making about an issuer.

Three separately signed shifts, and why they are not one score
---------------------------------------------------------------

:class:`~backend.extraction.tasks.library.AccountingLanguageShift` reports
recognition-favourable language, stated reliance on management estimates, and
emphasis on non-GAAP measures as three fields rather than one composite. A
composite would need weights, and there is no basis for any set of weights until
somebody measures which component predicts anything (P7.8, then Phase 8). Three
fields are also what lets a reviewer look at the document inspector (§6.5) and
see *which* of them moved.

``policy_change_disclosed`` is a boolean and deliberately not folded into the
shifts: an explicitly disclosed change in policy or presentation is a different
kind of fact from a drift in wording, and averaging the two would hide the one
the issuer chose to announce.

Why the pair is two periodic reports of the same form
------------------------------------------------------

Accounting policy language is restated every reporting period, so
:data:`~backend.extraction.tasks.delta.anchor.PERIODIC_REPORT` admits 10-K,
10-Q, 20-F and 40-F, with the baseline matched on **exact** form. Matching on
exact form is doing real work here beyond the amendment barrier: a 10-K's
critical-accounting-estimates section is audited and far longer than a 10-Q's,
so a cross-form pair would report a drop in estimate reliance every year at the
first quarter and a rise every year at the annual report — a seasonal artefact
that would look like a signal and would be stable enough to survive a naive
validation.

The prompt, the schema and the sign conventions live in
:mod:`backend.extraction.tasks.library`: positive always means "more of that,
now" — more aggressive, more estimate-reliant, more non-GAAP emphasis.
"""

from __future__ import annotations

from typing import Final

from backend.extraction.tasks.delta.anchor import PERIODIC_REPORT
from backend.extraction.tasks.delta.spec import DeltaTaskSpec
from backend.extraction.tasks.library import ACCOUNTING_LANGUAGE_SHIFT

__all__ = ["ACCOUNTING_LANGUAGE_SHIFT_SPEC"]

ACCOUNTING_LANGUAGE_SHIFT_SPEC: Final = DeltaTaskSpec(
    task=ACCOUNTING_LANGUAGE_SHIFT,
    document_class=PERIODIC_REPORT,
    construct=(
        "Change in how accounting policy, critical estimates and adjusted measures are "
        "described, between one periodic report and the previous report of the same form "
        "knowable when it was filed."
    ),
)
"""Accounting-language shift, over consecutive periodic reports of the same form."""
