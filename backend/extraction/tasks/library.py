"""The delta-oriented extraction tasks §5-P7 names (P7.3, first cut of P7.4).

Five tasks, one per example the directive gives: risk-factor language change,
guidance tone against guidance magnitude, evasiveness in analyst Q&A,
accounting-language shift, and the added/removed risk-factor set. Each is a real
task — a real output schema, a real prompt, real validation — not a placeholder.
What none of them can do yet is run against a provider, because B4 is
unresolved; that seam is :mod:`backend.extraction.tasks.client`, and nothing
here substitutes for it.

Every one of them reads a **pair** of documents, composed by
:func:`~backend.extraction.tasks.base.paired_document`, and reports a *change*.
That is the point of §5-P7's insistence on deltas: a tone level mostly measures
the industry and the drafting firm, and a predictor trained on levels learns the
roster rather than anything about the future. A change measured within one call
also differences away the drafting style the two filings share, which is the
largest nuisance term available to remove for free.

Sign conventions, stated once and repeated per field
-----------------------------------------------------

Every shift field runs ``[-1, 1]``, and **positive always means "more of the
thing the field is named after" in the current document than in the prior one**:
more severe, more hedged, more evasive, more aggressive. Zero means no
detectable change — which is the modal answer on consecutive filings and the
prompts say so, because a model that feels obliged to find a change will find
one.

Anonymization and what the prompts must not ask for
----------------------------------------------------

The payload reaches these prompts already masked: company names, tickers, people
and **every date** are placeholders like ``COMPANY_1`` and ``DATE_3``
(:mod:`backend.extraction.anonymize`). Every prompt here therefore states that
the text is redacted, that placeholders are meaningful and stable within the
document, and that the model must not guess the issuer, the period or the
outcome. That instruction is not politeness: the contamination probe (P7.9)
measures whether the model is recalling what happened rather than reading, and a
prompt that invited identification would be measuring the invitation.

Prompts also forbid using knowledge from outside the text and require quoted
evidence, so that a reviewer looking at the document inspector (§6.5) can check
the number against the passage it came from.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field

from backend.extraction.tasks.base import ChunkingPolicy, ExtractionTask, TaskRegistry
from backend.extraction.tasks.schema import ExtractionOutput, schema_text

__all__ = [
    "AccountingLanguageShift",
    "GuidanceToneVsMagnitude",
    "QaEvasivenessShift",
    "RiskFactorLanguageDelta",
    "RiskFactorSetDelta",
    "builtin_tasks",
]

Shift = Annotated[float, Field(ge=-1.0, le=1.0)]
"""A signed change, dimensionless, in ``[-1, 1]``. Positive means "more, now"."""

Level = Annotated[float, Field(ge=0.0, le=1.0)]
"""An unsigned level, dimensionless, in ``[0, 1]``."""

Evidence = Annotated[tuple[str, ...], Field(max_length=8)]
"""Short verbatim quotations from the payload, at most eight.

Capped because the cap is also a cost control: an uncapped evidence list is an
invitation to return the document. Quotations are from the *anonymized* text, so
they contain placeholders — that is correct and they must not be un-masked
before storage.
"""


class RiskFactorLanguageDelta(ExtractionOutput):
    """Change in how risk factors are *written* between two filings.

    About language, not about which risks are listed —
    :class:`RiskFactorSetDelta` covers the set. Two filings can list identical
    risks while one hedges every sentence and the other states them flatly, and
    that difference is the signal here.
    """

    severity_shift: Shift = Field(
        description=(
            "Change in the severity with which risks are described, [-1, 1]. "
            "Positive means the current document describes them as more severe."
        )
    )
    hedging_shift: Shift = Field(
        description=(
            "Change in hedged, qualified or conditional phrasing, [-1, 1]. "
            "Positive means the current document hedges more."
        )
    )
    specificity_shift: Shift = Field(
        description=(
            "Change in concreteness — named mechanisms, quantities, consequences — [-1, 1]. "
            "Positive means the current document is more specific."
        )
    )
    evidence: Evidence = Field(
        description="Short verbatim quotations from the text supporting the scores."
    )
    confidence: Level = Field(
        description=(
            "The model's own confidence in this reading, [0, 1]. A self-report, not a "
            "calibrated probability."
        )
    )


class GuidanceToneVsMagnitude(ExtractionOutput):
    """Guidance tone measured *against* the size of the guidance change.

    The construct §5-P7 asks for is the gap: management raising guidance by a
    rounding error while sounding triumphant, or cutting it while sounding calm,
    is the observation. Tone alone is a house-style measurement.
    """

    tone_shift: Shift = Field(
        description=(
            "Change in the confidence and positivity of the language around guidance, "
            "[-1, 1]. Positive means the current document sounds more confident."
        )
    )
    magnitude_direction: str = Field(
        description=(
            "Direction of the guidance change itself. Exactly one of: raised, reiterated, "
            "lowered, withdrawn, absent."
        ),
        pattern="^(raised|reiterated|lowered|withdrawn|absent)$",
    )
    magnitude_size: Level = Field(
        description=(
            "Size of the guidance change as described in the text, [0, 1], where 0 is no "
            "change and 1 is a change the text itself calls substantial. 0 when guidance "
            "is absent or withdrawn."
        )
    )
    tone_magnitude_gap: Shift = Field(
        description=(
            "Tone relative to what the magnitude justifies, [-1, 1]. Positive means the "
            "language is more upbeat than the size of the change supports; negative means "
            "more subdued."
        )
    )
    evidence: Evidence = Field(
        description="Short verbatim quotations from the text supporting the scores."
    )
    confidence: Level = Field(
        description="The model's own confidence, [0, 1]. A self-report, not a probability."
    )


class QaEvasivenessShift(ExtractionOutput):
    """Change in how directly analyst questions are answered.

    Measured as a change between two calls rather than as a level, because how
    directly a management team answers is largely a fixed trait of that team and
    a level would rank teams rather than detect news.
    """

    evasiveness_shift: Shift = Field(
        description=(
            "Change in evasiveness — redirection, non-answers, deferral — [-1, 1]. "
            "Positive means the current document is more evasive."
        )
    )
    specificity_shift: Shift = Field(
        description=(
            "Change in the concreteness of answers, [-1, 1]. Positive means answers in the "
            "current document contain more specifics."
        )
    )
    declined_questions_current: int = Field(
        ge=0,
        description=(
            "Questions in the current document that received no substantive answer (count). "
            "0 when the current document contains no question-and-answer section."
        ),
    )
    declined_questions_prior: int = Field(
        ge=0,
        description="The same count for the prior document.",
    )
    evidence: Evidence = Field(
        description="Short verbatim quotations from the text supporting the scores."
    )
    confidence: Level = Field(
        description="The model's own confidence, [0, 1]. A self-report, not a probability."
    )


class AccountingLanguageShift(ExtractionOutput):
    """Change in the language around accounting policy, estimates and measures."""

    aggressiveness_shift: Shift = Field(
        description=(
            "Change towards recognition-favourable language — earlier revenue, longer "
            "amortisation, capitalisation over expensing — [-1, 1]. Positive means the "
            "current document reads as more aggressive."
        )
    )
    estimate_reliance_shift: Shift = Field(
        description=(
            "Change in stated reliance on management estimates, judgements and assumptions, "
            "[-1, 1]. Positive means more reliance in the current document."
        )
    )
    non_gaap_emphasis_shift: Shift = Field(
        description=(
            "Change in emphasis on adjusted or non-GAAP measures relative to reported ones, "
            "[-1, 1]. Positive means more emphasis in the current document."
        )
    )
    policy_change_disclosed: bool = Field(
        description=(
            "Whether the current document explicitly discloses a change in accounting "
            "policy, estimate or presentation. True only when the text says so."
        )
    )
    evidence: Evidence = Field(
        description="Short verbatim quotations from the text supporting the scores."
    )
    confidence: Level = Field(
        description="The model's own confidence, [0, 1]. A self-report, not a probability."
    )


class RiskFactorSetDelta(ExtractionOutput):
    """Which risk factors were added and which were dropped between two filings.

    The set, not the wording. A newly added risk factor is a disclosure the
    issuer chose to make and had a reason to make; a dropped one is a claim that
    something stopped mattering.
    """

    added: Annotated[tuple[str, ...], Field(max_length=40)] = Field(
        description=(
            "Short titles of risk factors present in the current document and absent from "
            "the prior one, in the order they appear."
        )
    )
    removed: Annotated[tuple[str, ...], Field(max_length=40)] = Field(
        description=(
            "Short titles of risk factors present in the prior document and absent from "
            "the current one, in the order they appeared."
        )
    )
    retained_count: int = Field(
        ge=0,
        description="Risk factors present in both documents (count).",
    )
    confidence: Level = Field(
        description="The model's own confidence, [0, 1]. A self-report, not a probability."
    )


_SHARED_RULES: Final = """\
The text below is redacted. Company names, people, tickers and every date have \
been replaced by placeholders such as COMPANY_1, PERSON_2 and DATE_3. A given \
placeholder refers to the same entity everywhere it appears in this text.

Rules, all of them binding:
- Do not guess, state or hint at which company, person, period or year this is. \
If you believe you recognise it, ignore that belief entirely and read only what \
is written.
- Use only the text below. Do not use anything you know about any company or \
about what happened after this document was written.
- Leave placeholders exactly as they are in any text you quote.
- The two sections are the same kind of document from two consecutive periods. \
Compare the CURRENT section against the PRIOR section, in that direction.
- No change is a normal and common answer. Report 0 when you find no difference. \
Do not manufacture a difference to have something to report.
- Reply with a single JSON object and nothing else: no code fence, no commentary \
before or after, no trailing text.
"""
"""Instructions shared by every task's system prompt.

Written once because it is the anti-contamination contract, and a contract
restated in five places drifts in five directions. It is embedded into each
task's ``system`` text, so it is inside every prompt's content address — editing
it re-addresses every prompt and invalidates every cache entry, which is exactly
what changing the rules of the task should do.
"""


def _system(purpose: str, model: type[ExtractionOutput]) -> str:
    """Assemble a task's system instruction from its purpose and its schema.

    The JSON schema is generated from the output model rather than written out
    by hand, so the instruction and the validator cannot disagree: adding a
    field changes both, in the same commit, and changes the prompt's content
    address with them.

    Args:
        purpose: One paragraph saying what to extract.
        model: The output model the response must satisfy.

    Returns:
        The system instruction.
    """
    return (
        f"{purpose}\n\n{_SHARED_RULES}\n"
        f"The JSON object must validate against this schema exactly, with no "
        f"additional properties:\n{schema_text(model)}\n"
    )


_TEMPLATE: Final = "$document\n"
"""Every task's user message: the anonymized payload, nothing else.

All instruction lives in the system text. Keeping the user message to the
payload alone means the only thing that varies per call is the document, which
is what the cache key assumes.
"""


RISK_FACTOR_LANGUAGE_DELTA: Final = ExtractionTask(
    name="risk_factor_language_delta",
    output_model=RiskFactorLanguageDelta,
    system=_system(
        "You are comparing the risk-factor language of two consecutive filings by the "
        "same issuer. Report how the *writing* changed: severity, hedging and "
        "specificity. Ignore which risks are listed — another task handles the set.",
        RiskFactorLanguageDelta,
    ),
    template=_TEMPLATE,
    chunking_policy=ChunkingPolicy.WHOLE_DOCUMENT,
    description="Change in risk-factor language between consecutive filings.",
)

GUIDANCE_TONE_VS_MAGNITUDE: Final = ExtractionTask(
    name="guidance_tone_vs_magnitude",
    output_model=GuidanceToneVsMagnitude,
    system=_system(
        "You are comparing how two consecutive documents from the same issuer talk about "
        "forward guidance. Report the change in tone, the direction and size of the "
        "guidance change itself, and — the point of the task — the gap between them: "
        "whether the language is more or less upbeat than the size of the change "
        "supports.",
        GuidanceToneVsMagnitude,
    ),
    template=_TEMPLATE,
    chunking_policy=ChunkingPolicy.WHOLE_DOCUMENT,
    description="Guidance tone relative to guidance magnitude.",
)

QA_EVASIVENESS_SHIFT: Final = ExtractionTask(
    name="qa_evasiveness_shift",
    output_model=QaEvasivenessShift,
    system=_system(
        "You are comparing the analyst question-and-answer sections of two consecutive "
        "earnings calls by the same issuer. Report how directly questions were answered "
        "in the current section compared with the prior one, and count the questions that "
        "received no substantive answer in each.",
        QaEvasivenessShift,
    ),
    template=_TEMPLATE,
    chunking_policy=ChunkingPolicy.WHOLE_DOCUMENT,
    max_tokens=1536,
    description="Change in evasiveness in analyst Q&A between consecutive calls.",
)

ACCOUNTING_LANGUAGE_SHIFT: Final = ExtractionTask(
    name="accounting_language_shift",
    output_model=AccountingLanguageShift,
    system=_system(
        "You are comparing the accounting language of two consecutive filings by the same "
        "issuer: policy descriptions, critical estimates, and the treatment of adjusted "
        "measures. Report how it shifted.",
        AccountingLanguageShift,
    ),
    template=_TEMPLATE,
    chunking_policy=ChunkingPolicy.WHOLE_DOCUMENT,
    description="Shift in accounting-policy and estimate language between filings.",
)

RISK_FACTOR_SET_DELTA: Final = ExtractionTask(
    name="risk_factor_set_delta",
    output_model=RiskFactorSetDelta,
    system=_system(
        "You are comparing the sets of risk factors disclosed in two consecutive filings "
        "by the same issuer. List the short titles of risk factors that were added and "
        "those that were dropped, and count those present in both. Match risks by subject, "
        "not by wording: a re-titled risk about the same subject is retained, not both "
        "added and removed.",
        RiskFactorSetDelta,
    ),
    template=_TEMPLATE,
    chunking_policy=ChunkingPolicy.WHOLE_DOCUMENT,
    max_tokens=2048,
    description="Risk factors added and removed between consecutive filings.",
)


def builtin_tasks() -> TaskRegistry:
    """Return a registry holding the five delta tasks §5-P7 names.

    A function rather than a module constant so each caller gets its own
    registry and no code path can mutate a shared one into a state another
    caller depends on.

    Returns:
        A :class:`~backend.extraction.tasks.base.TaskRegistry` in the order the
        directive lists the examples.
    """
    return TaskRegistry(
        (
            RISK_FACTOR_LANGUAGE_DELTA,
            GUIDANCE_TONE_VS_MAGNITUDE,
            QA_EVASIVENESS_SHIFT,
            ACCOUNTING_LANGUAGE_SHIFT,
            RISK_FACTOR_SET_DELTA,
        )
    )
