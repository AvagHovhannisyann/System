"""Risk-factor language change: how Item 1A *moved* between filings (P7.4).

The first delta §5-P7 names. The construct is the **writing**, not the roster:
two annual reports can list an identical set of risks while one hedges every
sentence and the other states them flatly, and that difference is the signal.
Which risks were added or dropped is a different question and belongs to
:mod:`backend.extraction.tasks.delta.risk_factor_set`; keeping them apart is
what stops one score from mixing "they now describe the supply-chain risk in
concrete terms" with "they added a supply-chain risk".

Why the pair is two **annual** reports, matched on exact form
-------------------------------------------------------------

Item 1A in a 10-Q is an *update* to the annual set, not a restatement of it, so
a 10-K compared against a 10-Q would report most of the annual set as having
vanished. :data:`~backend.extraction.tasks.delta.anchor.ANNUAL_REPORT` therefore
admits 10-K, 20-F and 40-F, and the baseline must carry the *same* form type as
the current document — which also excludes ``10-K/A``, because an amendment
restates the document it amends and cannot be the thing that document is
compared against.

Why the baseline is a query and not "the last one"
--------------------------------------------------

"The previous 10-K" is not a property of a filer; it is the answer to a question
asked at an instant. Answered today, an issuer that restated its 2021 annual
report has one previous 10-K; answered on the day the 2022 report was accepted,
it has another. The second is the only one a model could have read. The anchor
is therefore the current filing's own knowledge time and the query runs on an
``as_of()`` session pinned to it
(:mod:`backend.extraction.tasks.delta.resolve`).

The prompt, the schema and the sign conventions live in
:mod:`backend.extraction.tasks.library`
(:class:`~backend.extraction.tasks.library.RiskFactorLanguageDelta`): three
signed shifts in ``[-1, 1]`` — severity, hedging, specificity — where positive
always means "more of that, now", zero means no detectable change, and the
prompt says explicitly that zero is a normal answer.
"""

from __future__ import annotations

from typing import Final

from backend.extraction.tasks.delta.anchor import ANNUAL_REPORT
from backend.extraction.tasks.delta.spec import DeltaTaskSpec
from backend.extraction.tasks.library import RISK_FACTOR_LANGUAGE_DELTA

__all__ = ["RISK_FACTOR_LANGUAGE_DELTA_SPEC"]

RISK_FACTOR_LANGUAGE_DELTA_SPEC: Final = DeltaTaskSpec(
    task=RISK_FACTOR_LANGUAGE_DELTA,
    document_class=ANNUAL_REPORT,
    construct=(
        "Change in how risk factors are written — severity, hedging, specificity — between "
        "one annual report and the previous one knowable when it was filed."
    ),
)
"""Risk-factor language change, over consecutive annual reports of the same form."""
