"""Added and removed risk factors: the set, not the wording (P7.4).

The fifth delta §5-P7 names, and the one whose output is not a score. A newly
disclosed risk factor is a statement the issuer chose to make and had a reason
to make; a dropped one is a claim that something stopped mattering. The two
lists are the observation, and
:class:`~backend.extraction.tasks.library.RiskFactorSetDelta` returns them as
lists plus a count of those present in both, so a consumer can see the size of
the change against the size of the set rather than against nothing.

Why the counts are not turned into a rate here
-----------------------------------------------

``len(added) / retained_count`` is one line away and is not written, because a
churn rate is a modelling choice and this package's job ends at the observation.
The feature library is where a rate would be defined, under the 30-feature cap
(§5-P5), with its own availability lag. Computing it here would put an
unregistered derived feature in the extraction store where nothing counts it.

Why "match by subject, not by wording" is in the prompt
--------------------------------------------------------

Issuers re-title and re-order risk factors constantly. A literal set difference
over headings would report almost every filing as having replaced its entire
risk section, and the resulting series would be noise with a strong calendar
component. The instruction is therefore explicit that a re-titled risk about the
same subject is *retained*, not simultaneously added and removed — which is also
why the whole pair must reach the model in one call
(``WHOLE_DOCUMENT``): matching by subject requires seeing both lists at once,
and the same-placeholder-in-both-halves property that
:func:`~backend.extraction.tasks.base.paired_document` provides is what makes
"the risk about ``COMPANY_1``'s supplier" identifiable across them.

Why the pair is two **annual** reports, matched on exact form
--------------------------------------------------------------

Only an annual report carries the complete risk-factor set; Item 1A in a 10-Q
lists *material changes* to it. Comparing a 10-K against a 10-Q would report the
whole annual set as removed and then as added again three months later — the
largest possible artefact, produced by the most obvious mistake. Exact-form
matching also keeps a ``10-K/A`` out of the baseline position: an amendment
restates the set it amends, so a delta against it would measure the amendment.
"""

from __future__ import annotations

from typing import Final

from backend.extraction.tasks.delta.anchor import ANNUAL_REPORT
from backend.extraction.tasks.delta.spec import DeltaTaskSpec
from backend.extraction.tasks.library import RISK_FACTOR_SET_DELTA

__all__ = ["RISK_FACTOR_SET_DELTA_SPEC"]

RISK_FACTOR_SET_DELTA_SPEC: Final = DeltaTaskSpec(
    task=RISK_FACTOR_SET_DELTA,
    document_class=ANNUAL_REPORT,
    construct=(
        "Risk factors added and removed between one annual report and the previous one "
        "knowable when it was filed, matched by subject rather than by title."
    ),
)
"""Added/removed risk factors, over consecutive annual reports of the same form."""
