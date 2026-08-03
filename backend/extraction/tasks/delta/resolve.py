"""The baseline query, answered at the anchor and nowhere else (P7.4, I1).

One function does one thing: given the document a delta is being extracted for,
find the earlier document it should be compared against, **as that earlier
document was knowable at the current one's knowledge time**.

Where the temporal safety comes from
------------------------------------

**The bound is the session's, not this module's.** :func:`resolve_prior_filing`
opens :func:`backend.db.as_of` at :attr:`KnowledgeAnchor.instant` and issues its
query on that session; rows whose ``knowledge_time`` is later simply do not
exist as far as the query is concerned (D-011). So the statement this module
builds carries **no** ``knowledge_time`` predicate of its own, and a test
asserts the absence rather than leaving it to be noticed. Re-stating the bound
here would either duplicate it — harmless, but misleading about where
enforcement lives — or contradict it, and a second weaker filter quietly wins.
The same reasoning is written out at length in
:mod:`backend.features.factors._prices`.

**The instant is not a parameter.** There is no ``as_of``, ``session``,
``instant`` or ``now`` argument anywhere in this module's public surface: the
anchor is derived from the current document
(:meth:`~backend.extraction.tasks.delta.anchor.KnowledgeAnchor.of`) and passed
straight to :func:`~backend.db.as_of`. A caller cannot answer the query at a
different instant without editing this file, which is the difference between a
guarantee and a convention. Asserted structurally, from the AST and from
``inspect.signature``, in ``backend/tests/extraction/delta/test_resolve.py`` —
the same discipline D-033 applied to the absence of a ``venue`` parameter.

**What arrives is still checked.** The rows are handed to
:func:`~backend.extraction.tasks.delta.anchor.select_baseline`, which re-derives
every eligibility condition — including the knowledge bound the as-of layer has
already applied. If a row arrives that should have been invisible, that is a
defect in the read path and it raises. A restatement filed later therefore has
to defeat three independent barriers to become a baseline: it is not visible at
the anchor, it is not the same form type, and it was not accepted before the
document it would be compared against.

Why more than one candidate is fetched
--------------------------------------

``LIMIT 1`` would make a tie invisible. Two distinct accessions of the same form
by the same filer at the identical acceptance instant is a store anomaly, and
:func:`~backend.extraction.tasks.delta.anchor.select_baseline` refuses to pick
one by result order — but it can only refuse what it can see, so the query
returns a few rows and lets the pure function decide.

Units: instants are timezone-aware UTC; ``cik`` is a dimensionless EDGAR Central
Index Key; :data:`CANDIDATE_LIMIT` is a row count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import sqlalchemy as sa

from backend.db import as_of
from backend.db.models import EdgarFiling
from backend.extraction.tasks.delta.anchor import (
    BaselineSource,
    BaselineSourceUnavailableError,
    FilingCandidate,
    KnowledgeAnchor,
    select_baseline,
)

if TYPE_CHECKING:
    from backend.extraction.tasks.delta.anchor import (
        CurrentDocument,
        DocumentClass,
        PriorDocumentRef,
    )

__all__ = ["CANDIDATE_LIMIT", "prior_filing_statement", "resolve_prior_filing"]

CANDIDATE_LIMIT: Final = 4
"""How many candidate rows the baseline query returns (count).

More than one so a tie on acceptance instant is visible to
:func:`~backend.extraction.tasks.delta.anchor.select_baseline` rather than
silently resolved by the planner. Small because only the newest matters: the
extra rows exist to be *counted*, not to be searched.
"""


def prior_filing_statement(
    current: CurrentDocument, document_class: DocumentClass
) -> sa.Select[Any]:
    """Build the baseline query for ``current``, without executing it.

    Separated from :func:`resolve_prior_filing` so the statement can be compiled
    and inspected in a test with no database — in particular so the *absence* of
    a ``knowledge_time`` predicate is asserted rather than assumed.

    The predicates, and what each one keeps out:

    * ``cik ==`` — another filer's history;
    * ``form_type ==`` — a 10-Q standing in for a 10-K, and, by the same
      comparison, an amendment (``10-K/A`` is not ``10-K``);
    * ``valid_from <`` — the current document itself, and anything accepted
      after it. ``valid_from`` is the EDGAR acceptance instant (D-011,
      :class:`~backend.db.models.EdgarFiling`), so this is the event-time axis;
      the knowledge-time axis is the session's and does not appear here.

    Args:
        current: the document the delta is being extracted for.
        document_class: which family of documents this task compares. Used only
            to refuse a class whose baselines do not live in ``edgar_filing``.

    Returns:
        An ORM ``Select`` over :class:`~backend.db.models.EdgarFiling`, newest
        acceptance first, limited to :data:`CANDIDATE_LIMIT` rows.

    Raises:
        BaselineSourceUnavailableError: ``document_class`` reads a store that
            does not exist.
    """
    if document_class.source is not BaselineSource.EDGAR_FILING:
        msg = (
            f"document class {document_class.name!r} takes its baseline from "
            f"{document_class.source.value}, for which no store exists in this platform. "
            "Earnings-call transcripts are P3.6, blocked on B1: the connector has not been "
            "built and nothing here will read a table that would have to be invented to "
            "exist (I3, §9.4). This is a blocker, not a missing baseline — it must not be "
            "recorded as a data condition"
        )
        raise BaselineSourceUnavailableError(msg)
    return (
        sa.select(EdgarFiling)
        .where(EdgarFiling.cik == current.cik)
        .where(EdgarFiling.form_type == current.form_type)
        .where(EdgarFiling.valid_from < current.acceptance)
        .order_by(EdgarFiling.valid_from.desc())
        .limit(CANDIDATE_LIMIT)
    )


async def resolve_prior_filing(
    current: CurrentDocument, *, document_class: DocumentClass
) -> PriorDocumentRef | None:
    """Return the baseline for ``current``, as knowable at ``current``'s own anchor.

    The one read path for a delta's other half. It opens the store at
    :meth:`KnowledgeAnchor.of(current) <backend.extraction.tasks.delta.anchor.KnowledgeAnchor.of>`
    and takes no argument that could point it at another instant.

    Args:
        current: the document the delta is being extracted for. Its
            ``knowledge_time`` **is** the anchor.
        document_class: which family of documents this task compares.

    Returns:
        The chosen :class:`~backend.extraction.tasks.delta.anchor.PriorDocumentRef`,
        or ``None`` when nothing eligible was knowable at the anchor. ``None`` is
        "no comparison possible", never a delta of zero
        (:mod:`backend.extraction.tasks.delta.outcome`).

    Raises:
        BaselineSourceUnavailableError: the class's baseline store does not
            exist (earnings-call transcripts, P3.6/B1).
        backend.extraction.tasks.delta.anchor.DocumentClassMismatchError: the
            current document's form is not read by this class.
        backend.extraction.tasks.delta.anchor.BaselineTemporalIntegrityError: a
            returned row was not knowable at the anchor — a read-path defect.
        backend.extraction.tasks.delta.anchor.BaselineIdentityError: a returned
            row is not a comparable earlier document, or two tie on acceptance.
    """
    anchor = KnowledgeAnchor.of(current)
    statement = prior_filing_statement(current, document_class)
    async with as_of(anchor.instant) as session:
        rows = (await session.scalars(statement)).all()
    candidates = tuple(
        FilingCandidate(
            accession_number=row.accession_number,
            cik=row.cik,
            company_name=row.company_name,
            form_type=row.form_type,
            acceptance=row.valid_from,
            knowledge_time=row.knowledge_time,
        )
        for row in rows
    )
    return select_baseline(
        candidates, current=current, anchor=anchor, document_class=document_class
    )
