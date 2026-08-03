"""P7.4: "no change" and "no comparison possible" are different values.

The failure these tests exist to prevent is the cheapest one available: a
missing baseline rendered as ``0.0``, which reads as *measured and unchanged*
everywhere downstream, enters a cross-sectional z-score, and describes a
first-time registrant as an issuer whose language did not move. Nothing about
the number looks wrong afterwards, which is why the distinction is enforced by
the type system rather than by a convention about sentinel values (D-030,
D-031).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import typing

import pytest

from backend.extraction.tasks.delta.anchor import KnowledgeAnchor
from backend.extraction.tasks.delta.outcome import (
    DeltaMeasured,
    DeltaOutcome,
    DeltaRejected,
    DeltaStamp,
    NoComparisonPossible,
    NoComparisonReason,
)
from backend.extraction.tasks.library import RiskFactorLanguageDelta
from backend.tests.extraction.delta.doubles import MODEL, utc

PRIOR_KNOWN = utc(2023, 2, 15, 21, 4)
CURRENT_KNOWN = utc(2024, 2, 20, 22, 11)
GAP = CURRENT_KNOWN - PRIOR_KNOWN


def _stamp(
    *,
    prior_known: dt.datetime = PRIOR_KNOWN,
    current_known: dt.datetime = CURRENT_KNOWN,
    anchor: dt.datetime | None = None,
    gap: dt.timedelta = GAP,
    prior_id: str = "0001725255-23-000004",
    current_id: str = "0001725255-24-000010",
) -> DeltaStamp:
    """Build a stamp for the ordering tests."""
    return DeltaStamp(
        task="risk_factor_language_delta",
        prompt_version_hash="c0ffee" * 5,
        model=MODEL,
        payload_digest="deadbeef" * 4,
        anchor=anchor if anchor is not None else current_known,
        prior_document_id=prior_id,
        current_document_id=current_id,
        prior_knowledge_time=prior_known,
        current_knowledge_time=current_known,
        gap=gap,
    )


def _unchanged() -> RiskFactorLanguageDelta:
    """An output saying, in schema, that nothing moved. A measurement."""
    return RiskFactorLanguageDelta(
        severity_shift=0.0,
        hedging_shift=0.0,
        specificity_shift=0.0,
        evidence=(),
        confidence=0.8,
    )


# ---------------------------------------------------------------------------
# The feature's knowledge time is the later of the two
# ---------------------------------------------------------------------------


def test_the_features_knowledge_time_is_the_later_of_the_two() -> None:
    """Nobody could compute the comparison before the later document existed."""
    stamp = _stamp()
    assert stamp.feature_knowledge_time == CURRENT_KNOWN
    assert stamp.feature_knowledge_time != stamp.prior_knowledge_time
    assert stamp.feature_knowledge_time == max(
        stamp.prior_knowledge_time, stamp.current_knowledge_time
    )


def test_a_pair_composed_backwards_is_refused() -> None:
    """Every signed score would carry the opposite sign."""
    with pytest.raises(ValueError, match="wrong way round"):
        _stamp(prior_known=CURRENT_KNOWN, current_known=PRIOR_KNOWN, anchor=PRIOR_KNOWN)


def test_the_stamp_refuses_an_anchor_that_is_not_the_current_knowledge_time() -> None:
    """The baseline query is answered at that instant and at no other."""
    with pytest.raises(ValueError, match="must be answered at the current document"):
        _stamp(anchor=CURRENT_KNOWN + dt.timedelta(days=90))


def test_the_stamp_refuses_a_pair_of_one_document() -> None:
    """Zero by construction is not a measurement."""
    with pytest.raises(ValueError, match="both halves of the pair"):
        _stamp(prior_id="0001725255-24-000010")


def test_the_stamp_refuses_a_non_positive_gap() -> None:
    """A baseline is accepted strictly before the document it is compared against."""
    with pytest.raises(ValueError, match="gap between"):
        _stamp(gap=dt.timedelta(0))


@pytest.mark.parametrize(
    "field_name",
    ["task", "prompt_version_hash", "model", "payload_digest"],
)
def test_the_stamp_requires_every_i2_component(field_name: str) -> None:
    """A result that cannot say how it was produced cannot exist (I2, §5-P7)."""
    values: dict[str, object] = {
        "task": "risk_factor_language_delta",
        "prompt_version_hash": "c0ffee" * 5,
        "model": MODEL,
        "payload_digest": "deadbeef" * 4,
        "anchor": CURRENT_KNOWN,
        "prior_document_id": "0001725255-23-000004",
        "current_document_id": "0001725255-24-000010",
        "prior_knowledge_time": PRIOR_KNOWN,
        "current_knowledge_time": CURRENT_KNOWN,
        "gap": GAP,
    }
    values[field_name] = "  "
    with pytest.raises(ValueError, match="must be non-empty"):
        DeltaStamp(**values)  # type: ignore[arg-type]


def test_the_stamp_names_the_prompt_version_that_produced_the_answer() -> None:
    """A delta's comparison instructions live in the prompt, so the address matters."""
    assert _stamp().prompt_version_hash == "c0ffee" * 5


# ---------------------------------------------------------------------------
# A measured zero is a measurement
# ---------------------------------------------------------------------------


def test_an_all_zero_output_is_a_measurement_and_says_no_change() -> None:
    """Zero is the modal honest answer on consecutive filings."""
    measured = DeltaMeasured(
        stamp=_stamp(), output=_unchanged(), raw_response="{}", cache_hit=False
    )
    assert isinstance(measured, DeltaMeasured)
    assert isinstance(measured.output, RiskFactorLanguageDelta)
    assert measured.output.severity_shift == 0.0
    assert measured.output.hedging_shift == 0.0


# ---------------------------------------------------------------------------
# "No comparison possible" carries no number at all
# ---------------------------------------------------------------------------


def _no_comparison(
    reason: NoComparisonReason = NoComparisonReason.NO_PRIOR_DOCUMENT,
) -> NoComparisonPossible:
    """Build a no-comparison record."""
    return NoComparisonPossible(
        task="risk_factor_language_delta",
        reason=reason,
        anchor=KnowledgeAnchor(instant=CURRENT_KNOWN, anchored_to="0001725255-24-000010"),
        current_document_id="0001725255-24-000010",
        detail="no annual report by this filer was knowable at the anchor",
    )


def test_no_comparison_possible_has_no_numeric_field() -> None:
    """Nothing on it to average, rank or plot — D-031's shape, applied here.

    Structural rather than behavioural on purpose: a test that merely asserted
    ``result != 0.0`` would keep passing the moment somebody added a
    ``value: float = 0.0`` field for the convenience of a chart.
    """
    record = _no_comparison()
    declared = {field.name: str(field.type) for field in dataclasses.fields(NoComparisonPossible)}
    assert not [name for name, hint in declared.items() if hint in {"int", "float", "complex"}]
    numeric = [
        name
        for name in dir(record)
        if not name.startswith("_") and isinstance(getattr(record, name), int | float | complex)
    ]
    assert numeric == []


def test_no_comparison_possible_cannot_be_coerced_to_a_number() -> None:
    """No ``__float__``, no ``__int__``, no ``__index__`` — not even by accident."""
    record = _no_comparison()
    for protocol in ("__float__", "__int__", "__index__", "__complex__"):
        assert not hasattr(record, protocol)
    with pytest.raises(TypeError):
        float(record)  # type: ignore[arg-type]


def test_the_two_no_comparison_reasons_are_distinct_facts() -> None:
    """One is about the issuer; the other is about this platform."""
    assert set(NoComparisonReason) == {
        NoComparisonReason.NO_PRIOR_DOCUMENT,
        NoComparisonReason.PRIOR_TEXT_UNAVAILABLE,
    }
    absent = _no_comparison(NoComparisonReason.NO_PRIOR_DOCUMENT)
    unfetched = _no_comparison(NoComparisonReason.PRIOR_TEXT_UNAVAILABLE)
    assert absent != unfetched
    assert absent.reason is not unfetched.reason


def test_a_no_comparison_record_carries_the_instant_it_was_answered_at() -> None:
    """An absence only means something with the instant it was measured at."""
    record = _no_comparison()
    assert record.anchor.instant == CURRENT_KNOWN
    assert record.anchor.anchored_to == record.current_document_id


def test_a_no_comparison_record_must_explain_itself() -> None:
    """A bare reason code is not enough for the operator's document inspector."""
    with pytest.raises(ValueError, match="detail must be non-empty"):
        NoComparisonPossible(
            task="risk_factor_language_delta",
            reason=NoComparisonReason.NO_PRIOR_DOCUMENT,
            anchor=KnowledgeAnchor(instant=CURRENT_KNOWN, anchored_to="x"),
            current_document_id="0001725255-24-000010",
            detail="",
        )


# ---------------------------------------------------------------------------
# The three outcomes, and that they stay three
# ---------------------------------------------------------------------------


def test_the_outcome_union_has_exactly_three_members() -> None:
    """A consumer pattern-matching on it has to name the no-comparison case."""
    assert set(typing.get_args(DeltaOutcome)) == {
        DeltaMeasured,
        DeltaRejected,
        NoComparisonPossible,
    }


def test_a_rejection_is_not_a_measurement_and_has_no_output() -> None:
    """A malformed response is data about the prompt, never a number."""
    rejected = DeltaRejected(
        stamp=_stamp(),
        raw_response='{"severity_shift": "0.4"}',
        validation_errors=("severity_shift: Input should be a valid number",),
        cache_hit=False,
    )
    assert not hasattr(rejected, "output")
    assert rejected.raw_response


def test_a_rejection_without_a_stated_reason_is_refused() -> None:
    """Otherwise it cannot be told apart from a result that was dropped."""
    with pytest.raises(ValueError, match="at least one validation error"):
        DeltaRejected(stamp=_stamp(), raw_response="{}", validation_errors=(), cache_hit=False)
