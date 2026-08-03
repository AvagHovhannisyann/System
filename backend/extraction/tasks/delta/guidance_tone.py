"""Guidance tone versus magnitude: the gap between what is said and the number (P7.4).

The second delta §5-P7 names, and the one whose construct is a *gap* rather than
a level or a simple change. Tone on its own measures the drafting firm and the
industry; the number on its own is already in the fundamentals. What neither
contains is the discrepancy — management raising guidance by a rounding error
while sounding triumphant, or cutting it while sounding calm — and that
discrepancy is what
:class:`~backend.extraction.tasks.library.GuidanceToneVsMagnitude` asks for
directly, as ``tone_magnitude_gap``, alongside the two components so a reviewer
can see where the gap came from.

Why the model is asked for the gap rather than handed a subtraction
--------------------------------------------------------------------

Because "more upbeat than the size of the change supports" is a judgement about
one text, and computing it here as ``tone_shift - magnitude_size`` would be an
arithmetic identity dressed up as a measurement: the two fields are on different
scales, only one is signed, and nothing establishes that a unit of tone trades
against a unit of magnitude. Deriving a number from an assumed exchange rate
between two dimensionless self-reports would be a fabricated quantity (I3). The
components are kept so the derivation can be *checked*, never so it can be
performed silently.

Why the pair is two periodic reports of the same form
------------------------------------------------------

Forward-looking language lives in MD&A and is restated every reporting period,
so :data:`~backend.extraction.tasks.delta.anchor.PERIODIC_REPORT` admits 10-K,
10-Q, 20-F and 40-F. Matching is on **exact** form, so a 10-Q's baseline is the
previous 10-Q and never the annual report: the two differ in scope and in audit
status, and a delta across them would measure that difference rather than the
issuer's outlook.

Earnings calls would be the richer source and are deliberately not used here:
that store does not exist (P3.6, blocked on B1), and
:mod:`backend.extraction.tasks.delta.qa_evasiveness` is where that dependency is
declared rather than worked around.

The prompt, the schema and the sign conventions live in
:mod:`backend.extraction.tasks.library`. ``magnitude_direction`` is a closed set
— raised, reiterated, lowered, withdrawn, absent — and ``absent`` is a real
answer: an issuer that gives no guidance is not an issuer whose guidance was
flat.
"""

from __future__ import annotations

from typing import Final

from backend.extraction.tasks.delta.anchor import PERIODIC_REPORT
from backend.extraction.tasks.delta.spec import DeltaTaskSpec
from backend.extraction.tasks.library import GUIDANCE_TONE_VS_MAGNITUDE

__all__ = ["GUIDANCE_TONE_VS_MAGNITUDE_SPEC"]

GUIDANCE_TONE_VS_MAGNITUDE_SPEC: Final = DeltaTaskSpec(
    task=GUIDANCE_TONE_VS_MAGNITUDE,
    document_class=PERIODIC_REPORT,
    construct=(
        "Change in the tone of forward-looking language measured against the direction and "
        "size of the guidance change itself, between one periodic report and the previous "
        "report of the same form knowable when it was filed."
    ),
)
"""Guidance tone relative to guidance magnitude, over consecutive periodic reports."""
