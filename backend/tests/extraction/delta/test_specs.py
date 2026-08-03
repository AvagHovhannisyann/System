"""P7.4: the five delta tasks, and that they are the five §5-P7 names.

Each spec binds one registered extraction task to the pair of documents it
compares. These tests check the binding — that every task §5-P7 names has one,
that none has two, that each demands the whole pair in one call, and that each
reads a family of documents whose baselines can actually be the thing the task
claims to compare against.
"""

from __future__ import annotations

import pytest

from backend.extraction.tasks.base import ChunkingPolicy, ExtractionTask
from backend.extraction.tasks.delta import delta_tasks
from backend.extraction.tasks.delta.accounting_language import ACCOUNTING_LANGUAGE_SHIFT_SPEC
from backend.extraction.tasks.delta.anchor import (
    ANNUAL_REPORT,
    EARNINGS_CALL,
    PERIODIC_REPORT,
    BaselineSource,
)
from backend.extraction.tasks.delta.guidance_tone import GUIDANCE_TONE_VS_MAGNITUDE_SPEC
from backend.extraction.tasks.delta.qa_evasiveness import QA_EVASIVENESS_SHIFT_SPEC
from backend.extraction.tasks.delta.risk_factor_language import RISK_FACTOR_LANGUAGE_DELTA_SPEC
from backend.extraction.tasks.delta.risk_factor_set import RISK_FACTOR_SET_DELTA_SPEC
from backend.extraction.tasks.delta.spec import (
    DeltaTaskNotRegisteredError,
    DeltaTaskRegistry,
    DeltaTaskSpec,
)
from backend.extraction.tasks.library import builtin_tasks
from backend.extraction.tasks.schema import ExtractionOutput

_EXPECTED = (
    "risk_factor_language_delta",
    "guidance_tone_vs_magnitude",
    "qa_evasiveness_shift",
    "accounting_language_shift",
    "risk_factor_set_delta",
)
"""The five examples §5-P7 gives, in the order the directive lists them."""


def test_every_registered_extraction_task_has_exactly_one_delta_spec() -> None:
    """No task without a pairing rule, and no pairing rule without a task.

    A task with no spec would have a prompt and a schema and no statement of
    which two documents it compares — which is the half a delta cannot do
    without. A spec with no task would name a prompt history that does not
    exist.
    """
    registry = delta_tasks()
    assert registry.names == _EXPECTED
    assert set(registry.names) == set(builtin_tasks().names)


def test_the_registry_is_a_fresh_object_each_time() -> None:
    """So no caller can mutate a shared one into a state another depends on."""
    first, second = delta_tasks(), delta_tasks()
    assert first is not second
    assert first.names == second.names


def test_an_unregistered_name_raises_rather_than_returning_none() -> None:
    """A run must not proceed against a silently absent task."""
    with pytest.raises(DeltaTaskNotRegisteredError, match="no delta task named"):
        delta_tasks().get("tone_level")


def test_two_specs_under_one_name_are_refused() -> None:
    """They would share a prompt history while asking different questions."""
    with pytest.raises(ValueError, match="duplicate delta task name"):
        DeltaTaskRegistry((RISK_FACTOR_LANGUAGE_DELTA_SPEC, RISK_FACTOR_LANGUAGE_DELTA_SPEC))


def test_every_spec_demands_the_whole_pair_in_one_call() -> None:
    """Half a comparison is a different question whose answer would look real."""
    for spec in delta_tasks():
        assert spec.task.chunking_policy is ChunkingPolicy.WHOLE_DOCUMENT


def test_a_spec_that_would_tolerate_a_split_cannot_be_constructed() -> None:
    """The guard, not just the current values."""

    class _Score(ExtractionOutput):
        """A minimal output, for exercising the guard rather than a task."""

        shift: float

    splittable = ExtractionTask(
        name="splittable",
        output_model=_Score,
        system="Compare CURRENT against PRIOR.",
        template="$document\n",
        chunking_policy=ChunkingPolicy.PER_CHUNK,
    )
    with pytest.raises(ValueError, match="chunking; a paired document must reach"):
        DeltaTaskSpec(task=splittable, document_class=ANNUAL_REPORT, construct="a change")


def test_every_spec_names_the_change_it_measures() -> None:
    """Phrased as a delta. A construct that reads as a level is the wrong shape."""
    for spec in delta_tasks():
        assert spec.construct.strip()
        assert "hange" in spec.construct or "added and removed" in spec.construct


def test_a_spec_with_no_construct_is_refused() -> None:
    """The operator's task list (§6.5) needs to say what the number means."""
    with pytest.raises(ValueError, match="must name the change it measures"):
        DeltaTaskSpec(
            task=RISK_FACTOR_LANGUAGE_DELTA_SPEC.task,
            document_class=ANNUAL_REPORT,
            construct="  ",
        )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (RISK_FACTOR_LANGUAGE_DELTA_SPEC, ANNUAL_REPORT),
        (RISK_FACTOR_SET_DELTA_SPEC, ANNUAL_REPORT),
        (GUIDANCE_TONE_VS_MAGNITUDE_SPEC, PERIODIC_REPORT),
        (ACCOUNTING_LANGUAGE_SHIFT_SPEC, PERIODIC_REPORT),
        (QA_EVASIVENESS_SHIFT_SPEC, EARNINGS_CALL),
    ],
)
def test_each_task_reads_the_family_of_documents_its_construct_needs(
    spec: DeltaTaskSpec, expected: object
) -> None:
    """Risk factors are annual; MD&A language is periodic; Q&A is a call.

    The two risk-factor tasks are pinned to annual reports because only an
    annual report carries the complete set — Item 1A in a 10-Q lists material
    *changes* to it, so a cross-form pair would report the whole set as removed
    and then re-added a quarter later.
    """
    assert spec.document_class is expected


def test_the_quarterly_form_is_readable_only_by_the_periodic_class() -> None:
    """The concrete consequence of the previous test, stated as form membership."""
    assert not ANNUAL_REPORT.admits("10-Q")
    assert PERIODIC_REPORT.admits("10-Q")
    assert PERIODIC_REPORT.admits("10-K")


def test_the_only_unresolvable_task_is_the_transcript_one() -> None:
    """Four of five read ``edgar_filing``; the fifth declares its blocker (P3.6/B1)."""
    unresolvable = [
        spec.name
        for spec in delta_tasks()
        if spec.document_class.source is not BaselineSource.EDGAR_FILING
    ]
    assert unresolvable == ["qa_evasiveness_shift"]


def test_the_blocked_task_says_so_in_its_construct() -> None:
    """So an operator reading the task list (§6.5) sees the dependency, not a gap."""
    assert "B1" in QA_EVASIVENESS_SHIFT_SPEC.construct


def test_a_specs_name_is_its_tasks_name() -> None:
    """One identity, so a prompt history and a pairing rule cannot drift apart."""
    for spec in delta_tasks():
        assert spec.name == spec.task.name


def test_every_task_prompt_takes_only_the_anonymized_payload() -> None:
    """Only ``$document``, on every one of the five.

    Re-asserted here because a delta prompt is the one most tempted to ask for a
    period label: "compare the 2024 filing against the 2023 one" would undo the
    date masking in the instruction rather than in the text.
    """
    for spec in delta_tasks():
        assert spec.task.prompt.variables == frozenset({"document"})


def test_every_task_prompt_forbids_identifying_the_issuer_or_the_period() -> None:
    """The anti-contamination contract, present in all five system instructions."""
    for spec in delta_tasks():
        system = spec.task.system
        assert "Do not guess, state or hint at which company" in system
        assert "No change is a normal and common answer" in system
