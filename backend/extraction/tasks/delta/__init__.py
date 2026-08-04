"""The five delta-oriented extraction tasks §5-P7 names (P7.4).

§5-P7 is explicit: *"Extract deltas, not states."* Every task here compares
**two documents from different points in time** and reports a change:

======================================  ====================================
task                                    the change it measures
======================================  ====================================
``risk_factor_language_delta``           how Item 1A is *written*, filing to filing
``guidance_tone_vs_magnitude``           tone against the size of the guidance change
``qa_evasiveness_shift``                 non-answers in analyst Q&A, call to call
``accounting_language_shift``            how the same accounting item is described
``risk_factor_set_delta``                which risk factors were added and dropped
======================================  ====================================

The prompt, the output schema and the sign conventions for each live in
:mod:`backend.extraction.tasks.library`, which is where extraction tasks
register. This package adds the half a delta needs and a single-document task
does not: **which two documents, chosen when.**

What comparing across time forces
----------------------------------

*The baseline is a query, and a query has an instant.* "The previous 10-K" is
not a property of a filer — it is the answer to a question, and the answer
changes with when it is asked. Asked today, an issuer that restated an earlier
year has one previous annual report; asked on the day the later report was
accepted, it has a different one, and only the second is a document anybody
could have read. So the anchor is the current document's own ``knowledge_time``,
the baseline query runs on an :func:`backend.db.as_of` session pinned to it, and
:meth:`~backend.extraction.tasks.delta.runner.DeltaExtractor.extract` exposes no
parameter that could carry a different instant
(:mod:`backend.extraction.tasks.delta.anchor`,
:mod:`backend.extraction.tasks.delta.resolve`).

*A restatement published later must clear three separate barriers to become a
baseline, and clears none of them.* It is not visible at the anchor (the as-of
bound); it is not the same form type (``10-K/A`` is not ``10-K``, and no
document class admits an amendment); and it was not accepted strictly before the
document it would be compared against. Each is checked independently, and each
**raises** rather than filtering, because a rule that can never fire under a
correct read path is exactly the rule whose firing means the read path is wrong.

*A delta has two knowledge times, and the feature's is the later one.* Nobody
could compute the comparison before the later document existed.
:class:`~backend.extraction.tasks.delta.outcome.DeltaStamp` carries both, orders
them at construction, and derives ``feature_knowledge_time`` as the maximum.

*A missing prior document is not a delta of zero.* "No change" and "no
comparison possible" are different facts, so they are different types:
:class:`~backend.extraction.tasks.delta.outcome.DeltaMeasured` carries an output
whose zeros mean *measured and unchanged*, while
:class:`~backend.extraction.tasks.delta.outcome.NoComparisonPossible` carries no
number at all — nothing to average, rank or plot. Same shape as D-030's
payload-free retraction and D-031's ``UnmeasurableFeature``, and the same
failure avoided: a zero that entered a cross-sectional score and described a
first-time registrant as an issuer whose language did not move.

*An unbuilt source is neither of those.* Q&A evasiveness reads earnings-call
transcripts, and no transcript store exists (P3.6, blocked on B1). That raises
:class:`~backend.extraction.tasks.delta.anchor.BaselineSourceUnavailableError`
rather than becoming a recorded "no comparison possible": an absent baseline is
an observation about an issuer, an unbuilt connector is a blocker, and a blocker
stored as a data value stops being visible (§9.4).

**Nothing here has ever run against a model.** B4 leaves the platform with no
provider key and no configured spend cap, so the model call is made through
:class:`~backend.extraction.tasks.client.ModelClient` exactly as P7.3 built it —
whose default implementation raises, and whose governed wrapper (P7.7) refuses
without caps. These tasks are *definitions*: prompt, schema, pairing rule,
temporal anchor. What the tests exercise is everything on either side of that
seam, driven through declared doubles.

Module layout, and what is deliberately not re-exported below:

- :mod:`~backend.extraction.tasks.delta.anchor` — the temporal contract, pure.
- :mod:`~backend.extraction.tasks.delta.outcome` — the three outcomes, pure.
- :mod:`~backend.extraction.tasks.delta.spec` — a task bound to a document
  class, plus the registry.
- the five task modules — one per delta, each a single
  :class:`~backend.extraction.tasks.delta.spec.DeltaTaskSpec` with the reasoning
  for its pairing rule.
- :mod:`~backend.extraction.tasks.delta.resolve` and
  :mod:`~backend.extraction.tasks.delta.runner` — **not re-exported**, following
  the precedent :mod:`backend.extraction.tasks` set for
  :mod:`backend.extraction.tasks.store`: importing them pulls in
  :mod:`backend.db`, and keeping that one import away is what lets everything
  above stay usable, and testable, in a process with no database. A caller that
  wants to run a delta names the module.
"""

from __future__ import annotations

from backend.extraction.tasks.delta.accounting_language import ACCOUNTING_LANGUAGE_SHIFT_SPEC
from backend.extraction.tasks.delta.anchor import (
    ANNUAL_REPORT,
    EARNINGS_CALL,
    PERIODIC_REPORT,
    BaselineIdentityError,
    BaselineSource,
    BaselineSourceUnavailableError,
    BaselineTemporalIntegrityError,
    CurrentDocument,
    DeltaAnchorError,
    DocumentClass,
    DocumentClassMismatchError,
    FilingCandidate,
    KnowledgeAnchor,
    PriorDocumentRef,
    select_baseline,
)
from backend.extraction.tasks.delta.guidance_tone import GUIDANCE_TONE_VS_MAGNITUDE_SPEC
from backend.extraction.tasks.delta.outcome import (
    DeltaMeasured,
    DeltaOutcome,
    DeltaRejected,
    DeltaStamp,
    NoComparisonPossible,
    NoComparisonReason,
)
from backend.extraction.tasks.delta.qa_evasiveness import QA_EVASIVENESS_SHIFT_SPEC
from backend.extraction.tasks.delta.risk_factor_language import RISK_FACTOR_LANGUAGE_DELTA_SPEC
from backend.extraction.tasks.delta.risk_factor_set import RISK_FACTOR_SET_DELTA_SPEC
from backend.extraction.tasks.delta.spec import (
    DeltaTaskNotRegisteredError,
    DeltaTaskRegistry,
    DeltaTaskSpec,
)

__all__ = [
    "ACCOUNTING_LANGUAGE_SHIFT_SPEC",
    "ANNUAL_REPORT",
    "EARNINGS_CALL",
    "GUIDANCE_TONE_VS_MAGNITUDE_SPEC",
    "PERIODIC_REPORT",
    "QA_EVASIVENESS_SHIFT_SPEC",
    "RISK_FACTOR_LANGUAGE_DELTA_SPEC",
    "RISK_FACTOR_SET_DELTA_SPEC",
    "BaselineIdentityError",
    "BaselineSource",
    "BaselineSourceUnavailableError",
    "BaselineTemporalIntegrityError",
    "CurrentDocument",
    "DeltaAnchorError",
    "DeltaMeasured",
    "DeltaOutcome",
    "DeltaRejected",
    "DeltaStamp",
    "DeltaTaskNotRegisteredError",
    "DeltaTaskRegistry",
    "DeltaTaskSpec",
    "DocumentClass",
    "DocumentClassMismatchError",
    "FilingCandidate",
    "KnowledgeAnchor",
    "NoComparisonPossible",
    "NoComparisonReason",
    "PriorDocumentRef",
    "delta_tasks",
    "select_baseline",
]


def delta_tasks() -> DeltaTaskRegistry:
    """Return a registry holding the five delta task specs §5-P7 names.

    A function rather than a module constant so each caller gets its own
    registry and no code path can mutate a shared one into a state another
    caller depends on — the same reasoning as
    :func:`backend.extraction.tasks.library.builtin_tasks`, whose task names
    these specs cover exactly.

    Returns:
        A :class:`~backend.extraction.tasks.delta.spec.DeltaTaskRegistry` in the
        order the directive lists the examples.
    """
    return DeltaTaskRegistry(
        (
            RISK_FACTOR_LANGUAGE_DELTA_SPEC,
            GUIDANCE_TONE_VS_MAGNITUDE_SPEC,
            QA_EVASIVENESS_SHIFT_SPEC,
            ACCOUNTING_LANGUAGE_SHIFT_SPEC,
            RISK_FACTOR_SET_DELTA_SPEC,
        )
    )
