"""Q&A evasiveness: non-answers in earnings-call Q&A (P7.4) — source blocked on B1.

The third delta §5-P7 names. The construct is a **change** rather than a level
for a reason worth stating: how directly a management team answers analysts is
largely a fixed trait of that team and its counsel. A level would rank
management styles, which is stable, cross-sectionally strong and useless as a
predictor because it does not move. What moves is the quarter in which a team
that normally answers plainly starts deferring — and that is what
:class:`~backend.extraction.tasks.library.QaEvasivenessShift` measures, together
with the count of questions that received no substantive answer in each half, so
a shift score can be checked against something countable.

This task cannot run, and that is recorded rather than worked around
---------------------------------------------------------------------

Its baseline lives in
:attr:`~backend.extraction.tasks.delta.anchor.BaselineSource.EARNINGS_CALL_TRANSCRIPT`,
and **there is no transcript store**: P3.6 is blocked on B1, so nothing has been
ingested and no table exists. Three things follow, each deliberate:

* :data:`~backend.extraction.tasks.delta.anchor.EARNINGS_CALL` declares an empty
  form-type set, because a transcript is not an EDGAR submission and giving it a
  form type would be inventing a fact about a source nobody has seen;
* :func:`~backend.extraction.tasks.delta.resolve.prior_filing_statement` raises
  :class:`~backend.extraction.tasks.delta.anchor.BaselineSourceUnavailableError`
  for this class rather than reading ``edgar_filing`` as an approximation. An
  8-K exhibit is not a transcript, and a task that quietly read one would be
  answering a different question under this task's name;
* the refusal is an **exception, not a**
  :class:`~backend.extraction.tasks.delta.outcome.NoComparisonPossible`. An
  absent prior transcript would be an observation about an issuer; an unbuilt
  connector is a blocker, and a blocker recorded as a data value is a blocker
  that stops being visible (§9.4, §9.8).

The prompt, the schema and the sign conventions are nonetheless real and
complete — they are what P7.8's golden set will be labelled against, and they do
not depend on the connector. What is missing is the input, and it is missing
loudly.
"""

from __future__ import annotations

from typing import Final

from backend.extraction.tasks.delta.anchor import EARNINGS_CALL
from backend.extraction.tasks.delta.spec import DeltaTaskSpec
from backend.extraction.tasks.library import QA_EVASIVENESS_SHIFT

__all__ = ["QA_EVASIVENESS_SHIFT_SPEC"]

QA_EVASIVENESS_SHIFT_SPEC: Final = DeltaTaskSpec(
    task=QA_EVASIVENESS_SHIFT,
    document_class=EARNINGS_CALL,
    construct=(
        "Change in how directly analyst questions are answered between one earnings call "
        "and the previous one knowable when it was held. Input unavailable: the transcript "
        "connector is P3.6, blocked on B1."
    ),
)
"""Q&A evasiveness change, over consecutive earnings calls. **Unresolvable today.**"""
