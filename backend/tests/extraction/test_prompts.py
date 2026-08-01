"""P7.10 prompt tests: content addressing, history, diff, golden scores, rollback.

§6.5 asks for four things on the operator's page — full version history, a
side-by-side diff, a golden-set score attached to each version, and one-click
rollback. All four follow from one decision, so that decision is what most of
these tests are about: **a prompt version's identity is a digest of its own
content.**

The consequence worth naming is that *rollback is not an operation*. There is no
``rollback`` function to test, and its absence is the design: going back is
:meth:`~backend.extraction.prompts.store.InMemoryPromptStore.activate` with a
hash already in the history — the same call as going forward. What the tests
below check is that this really is byte-identical restoration rather than a
best-effort one: the returned version is the same object content that was
measured, and the golden-set score attached to that hash still describes it.

The durable store (:mod:`backend.extraction.prompts.postgres`) implements the
same protocol against real tables; its round-trips need a database and are
exercised by the migrated integration suite. The semantics are pinned here,
where they are cheap to pin.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.extraction.prompts import (
    AUDIT_FIELD,
    AUDIT_SCOPE,
    DiffOp,
    GoldenSetScore,
    InMemoryPromptStore,
    NoActivePromptError,
    PromptRecord,
    PromptRenderError,
    PromptStore,
    PromptVersion,
    PromptVersionNotFoundError,
    content_digest,
    diff_versions,
    golden_verdict,
    unified_lines,
)
from backend.extraction.prompts.postgres import PostgresPromptStore

_ACTOR = "operator"
_SET_ID = "golden-set-v1"


def _version(
    *,
    name: str = "risk_factor_language_delta",
    system: str = "Compare the CURRENT section against the PRIOR section.",
    template: str = "$document\n",
    schema_digest: str = "0123456789abcdef",
) -> PromptVersion:
    """Build a prompt version."""
    return PromptVersion(name=name, system=system, template=template, schema_digest=schema_digest)


def _score(version: PromptVersion, agreement: float = 0.83) -> GoldenSetScore:
    """Build a golden-set score bound to a version's address."""
    return GoldenSetScore(
        name=version.name,
        version_hash=version.version_hash,
        golden_set_id=_SET_ID,
        agreement=agreement,
        document_count=300,
        scored_at=dt.datetime(2026, 8, 1, 9, 0, tzinfo=dt.UTC),
        actor=_ACTOR,
    )


# --------------------------------------------------------------------------
# Content addressing
# --------------------------------------------------------------------------


def test_identical_content_is_one_version_everywhere() -> None:
    """Two writings of the same prompt are the same version, with no reconciliation."""
    assert _version().version_hash == _version().version_hash
    assert len(_version().version_hash) == 32


@pytest.mark.parametrize(
    "changed",
    [
        {"name": "other_task"},
        {"system": "Compare the CURRENT section against the PRIOR section. Be brief."},
        {"template": "Read this:\n$document\n"},
        {"schema_digest": "fedcba9876543210"},
    ],
)
def test_changing_any_addressed_field_produces_a_different_version(
    changed: dict[str, str],
) -> None:
    """All four fields change what the model is asked, so all four are in the address."""
    assert _version(**changed).version_hash != _version().version_hash


def test_the_address_is_derived_on_every_read_and_never_stored() -> None:
    """A stored hash is a hash that can go stale; the cache's guarantee rests on this."""
    assert "version_hash" not in PromptVersion.__dataclass_fields__
    assert isinstance(PromptVersion.version_hash, property)


def test_the_digest_is_length_prefixed_so_field_boundaries_cannot_blur() -> None:
    """There is no delimiter that cannot occur inside a prompt."""
    assert content_digest("ab", "c") != content_digest("a", "bc")


def test_who_saved_a_version_and_when_are_not_part_of_its_address() -> None:
    """Otherwise one prompt text would have two addresses depending on who typed it.

    Those facts describe the *act of saving* a version, not the version, so they
    live on the record rather than in the digest.
    """
    assert set(PromptVersion.__dataclass_fields__) == {
        "name",
        "system",
        "template",
        "schema_digest",
    }
    record_only = {"actor", "notes", "correlation_id", "recorded_at", "sequence"}
    assert record_only <= set(PromptRecord.__dataclass_fields__)
    assert not record_only & set(PromptVersion.__dataclass_fields__)


# --------------------------------------------------------------------------
# Version construction and rendering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", " padded", "padded "])
def test_a_padded_or_empty_prompt_name_is_refused(name: str) -> None:
    """Trimming would make ``tone`` and `` tone`` one key at write time and two in every log."""
    with pytest.raises(ValueError, match="prompt name must be non-empty and unpadded"):
        _version(name=name)


def test_an_empty_template_is_refused() -> None:
    """There is nothing to send."""
    with pytest.raises(ValueError, match="empty template"):
        _version(template="")


def test_a_missing_schema_digest_is_refused() -> None:
    """The response schema is part of the address; leaving it out breaks invalidation."""
    with pytest.raises(ValueError, match="empty schema_digest"):
        _version(schema_digest="")


def test_a_template_with_a_bare_dollar_is_refused_at_construction() -> None:
    """Extraction prompts contain JSON, so template syntax errors must fail early and loudly."""
    with pytest.raises(ValueError, match="invalid template"):
        _version(template="cost is $ per token: $document")


def test_rendering_reports_both_directions_of_variable_mismatch() -> None:
    """A silently-ignored variable is how a prompt describes a document it was never given."""
    version = _version()
    assert version.render({"document": "payload"}) == "payload\n"
    with pytest.raises(PromptRenderError, match="missing=\\['document'\\]"):
        version.render({})
    with pytest.raises(PromptRenderError, match="unexpected=\\['ticker'\\]"):
        version.render({"document": "payload", "ticker": "ACME"})


# --------------------------------------------------------------------------
# Diff: per-section alignment for the two-column view
# --------------------------------------------------------------------------


def test_the_diff_covers_three_sections_in_a_fixed_order() -> None:
    """system, template and schema fail differently; merging them hides the third."""
    diff = diff_versions(_version(), _version(system="Something else entirely."))
    assert [section.section for section in diff.sections] == ["system", "template", "schema"]


def test_a_schema_only_change_is_visible_even_though_the_prompt_text_is_identical() -> None:
    """The case an operator is least likely to be expecting, and the reason to split sections."""
    diff = diff_versions(_version(), _version(schema_digest="fedcba9876543210"))
    changed = {section.section for section in diff.sections if section.changed}
    assert changed == {"schema"}
    assert diff.changed


def test_the_alignment_covers_both_sides_completely_including_unchanged_runs() -> None:
    """A side-by-side view needs the equal runs to keep its two columns in step."""
    old = _version(system="line one\nline two\nline three")
    new = _version(system="line one\nline two changed\nline three")
    section = next(s for s in diff_versions(old, new).sections if s.section == "system")

    assert [segment.op for segment in section.segments] == [
        DiffOp.EQUAL,
        DiffOp.REPLACE,
        DiffOp.EQUAL,
    ]
    assert "".join("".join(segment.old_lines) for segment in section.segments) == (
        "line oneline twoline three"
    )
    assert section.added_lines == 1
    assert section.removed_lines == 1


def test_an_unchanged_pair_diffs_to_nothing_and_agrees_with_the_address() -> None:
    """``changed`` and hash inequality can never disagree; the address covers the sections."""
    diff = diff_versions(_version(), _version())
    assert not diff.changed
    assert diff.old_version_hash == diff.new_version_hash


def test_diffing_two_different_prompts_is_refused() -> None:
    """A cross-prompt diff renders as a total rewrite, which hides the mistake."""
    with pytest.raises(ValueError, match="cannot diff versions of different prompts"):
        diff_versions(_version(), _version(name="other_task"))


def test_the_unified_rendering_skips_unchanged_sections_and_labels_the_rest() -> None:
    """A convenience for logs and commit messages; nothing parses it back."""
    old, new = _version(), _version(system="Compare them the other way round.")
    lines = list(unified_lines(diff_versions(old, new)))
    assert any(line.startswith("---") and ":system@" in line for line in lines)
    assert not any(":template@" in line for line in lines)


# --------------------------------------------------------------------------
# History, activation and rollback
# --------------------------------------------------------------------------


async def test_saving_the_same_text_twice_is_one_version() -> None:
    """A version *is* its content, so re-saving identical bytes changed nothing."""
    store = InMemoryPromptStore()
    first = await store.save(_version(), actor=_ACTOR, notes="first")
    second = await store.save(_version(), actor="someone else", notes="again")

    assert second is first
    assert second.actor == _ACTOR, "the first save's circumstances are the ones kept"
    assert len(await store.history(_version().name)) == 1


async def test_history_is_newest_first_and_keeps_every_version() -> None:
    """§6.5's full version history: nothing is superseded out of existence."""
    store = InMemoryPromptStore()
    v1 = _version()
    v2 = _version(system="Second wording.")
    v3 = _version(system="Third wording.")
    for version in (v1, v2, v3):
        await store.save(version, actor=_ACTOR)

    history = await store.history(v1.name)
    assert [record.version_hash for record in history] == [
        v3.version_hash,
        v2.version_hash,
        v1.version_hash,
    ]
    assert [record.sequence for record in history] == [3, 2, 1]


async def test_an_unsaved_prompt_has_an_empty_history_rather_than_an_error() -> None:
    """Absence of history is not an error for a history read."""
    assert await InMemoryPromptStore().history("never_saved") == ()


async def test_saving_a_version_does_not_put_it_in_force() -> None:
    """Otherwise saving a draft would silently deploy it."""
    store = InMemoryPromptStore()
    await store.save(_version(), actor=_ACTOR)
    with pytest.raises(NoActivePromptError, match="has no active version"):
        await store.active(_version().name)


async def test_activating_an_unsaved_hash_is_refused() -> None:
    """A task pointed at an unsaved hash would fail at its next call, far from the mistake."""
    store = InMemoryPromptStore()
    with pytest.raises(PromptVersionNotFoundError, match="no saved version with hash"):
        await store.activate(_version().name, "0" * 32, actor=_ACTOR)


async def test_rollback_is_selecting_an_earlier_hash_and_restores_it_byte_for_byte() -> None:
    """§6.5's one-click rollback, which is the same call as moving forward.

    Nothing is restored because nothing was destroyed, so the version returned
    to is byte-identical to the one that was measured — which is what lets the
    golden-set score attached to that hash keep describing the text in force.
    """
    store = InMemoryPromptStore()
    original = _version()
    edited = _version(system="A rewording that turned out worse.")
    for version in (original, edited):
        await store.save(version, actor=_ACTOR)
    await store.attach_golden_score(_score(original, agreement=0.83))

    await store.activate(original.name, original.version_hash, actor=_ACTOR)
    await store.activate(edited.name, edited.version_hash, actor=_ACTOR)
    rollback = await store.activate(original.name, original.version_hash, actor=_ACTOR)

    assert rollback.is_rollback is True
    assert rollback.previous_version_hash == edited.version_hash
    in_force = await store.active(original.name)
    assert in_force.version == original
    assert in_force.version.system == original.system
    scores = await store.golden_scores(original.name, original.version_hash)
    assert [score.agreement for score in scores] == [0.83]


async def test_moving_forward_is_not_recorded_as_a_rollback() -> None:
    """``is_rollback`` is derived from the history, so it cannot disagree with the record."""
    store = InMemoryPromptStore()
    original, edited = _version(), _version(system="A new wording.")
    for version in (original, edited):
        await store.save(version, actor=_ACTOR)

    first = await store.activate(original.name, original.version_hash, actor=_ACTOR)
    second = await store.activate(edited.name, edited.version_hash, actor=_ACTOR)

    assert first.is_rollback is False
    assert first.previous_version_hash is None
    assert second.is_rollback is False
    assert second.previous_version_hash == original.version_hash


async def test_the_activation_history_is_newest_first() -> None:
    """Which version was in force when is reconstructed by walking this."""
    store = InMemoryPromptStore()
    original, edited = _version(), _version(system="A new wording.")
    for version in (original, edited):
        await store.save(version, actor=_ACTOR)
    await store.activate(original.name, original.version_hash, actor=_ACTOR)
    await store.activate(edited.name, edited.version_hash, actor=_ACTOR)

    activations = await store.activations(original.name)
    assert [activation.sequence for activation in activations] == [2, 1]
    assert activations[0].version_hash == edited.version_hash


@pytest.mark.parametrize("actor", ["", "   "])
async def test_an_unattributed_save_or_activation_is_refused(actor: str) -> None:
    """A version with no recorded author is not history, and §6.11 wants who as well as what."""
    store = InMemoryPromptStore()
    with pytest.raises(ValueError, match="actor is required"):
        await store.save(_version(), actor=actor)
    await store.save(_version(), actor=_ACTOR)
    with pytest.raises(ValueError, match="actor is required"):
        await store.activate(_version().name, _version().version_hash, actor=actor)


def test_activation_is_recorded_under_the_audit_scope_the_audit_log_names() -> None:
    """§6.11: config changes are events. The pointer is a config value like any other."""
    assert AUDIT_SCOPE == "prompt"
    assert AUDIT_FIELD == "active_version_hash"


# --------------------------------------------------------------------------
# Golden-set scores, and the threshold that is deliberately absent (D-014)
# --------------------------------------------------------------------------


async def test_a_score_must_name_a_version_that_can_be_looked_up() -> None:
    """A score for text nobody can retrieve is not evidence."""
    store = InMemoryPromptStore()
    with pytest.raises(PromptVersionNotFoundError):
        await store.attach_golden_score(_score(_version()))


async def test_scores_are_returned_newest_first_and_filterable_by_version() -> None:
    """A per-version score trend (§6.5) is the query this serves."""
    store = InMemoryPromptStore()
    original, edited = _version(), _version(system="A new wording.")
    for version in (original, edited):
        await store.save(version, actor=_ACTOR)
    await store.attach_golden_score(_score(original, agreement=0.80))
    await store.attach_golden_score(_score(edited, agreement=0.84))
    await store.attach_golden_score(_score(original, agreement=0.81))

    assert [s.agreement for s in await store.golden_scores(original.name)] == [0.81, 0.84, 0.80]
    assert [
        s.agreement for s in await store.golden_scores(original.name, original.version_hash)
    ] == [0.81, 0.80]


@pytest.mark.parametrize("agreement", [-0.01, 1.01, 85.0])
def test_an_agreement_outside_the_unit_interval_is_refused(agreement: float) -> None:
    """§8: agreement is a fraction. ``85`` is what passing a percentage looks like."""
    with pytest.raises(ValueError, match="agreement must be a fraction"):
        GoldenSetScore(
            name="p",
            version_hash="h",
            golden_set_id=_SET_ID,
            agreement=agreement,
            document_count=300,
            scored_at=dt.datetime(2026, 8, 1, tzinfo=dt.UTC),
            actor=_ACTOR,
        )


def test_a_score_over_no_documents_is_refused() -> None:
    """Agreement over zero documents is not a measurement."""
    with pytest.raises(ValueError, match="document_count must be"):
        GoldenSetScore(
            name="p",
            version_hash="h",
            golden_set_id=_SET_ID,
            agreement=0.9,
            document_count=0,
            scored_at=dt.datetime(2026, 8, 1, tzinfo=dt.UTC),
            actor=_ACTOR,
        )


def test_judging_a_score_requires_a_stated_derivation_for_the_threshold() -> None:
    """D-014: the directive's 85% is the operator's prior, and the noise floor is unmeasured.

    A default threshold would be a fabricated number where a reader would take
    it as authoritative (I3), so there is none — and a bar with no stated
    derivation is refused outright.
    """
    score = _score(_version(), agreement=0.87)
    with pytest.raises(ValueError, match="threshold_basis is required"):
        golden_verdict(score, threshold=0.85, threshold_basis="   ")


def test_a_percentage_threshold_is_refused() -> None:
    """The same unit trap on the other side of the comparison."""
    with pytest.raises(ValueError, match="threshold must be a fraction"):
        golden_verdict(_score(_version()), threshold=85, threshold_basis="the directive says 85%")


def test_a_verdict_carries_the_basis_alongside_the_result() -> None:
    """The provisional-ness must be visible next to any figure derived from the bar."""
    basis = "provisional: the intra-rater noise floor is unmeasured (B3), so this is a placeholder"
    verdict = golden_verdict(
        _score(_version(), agreement=0.87), threshold=0.85, threshold_basis=basis
    )

    assert verdict.meets_threshold is True
    assert verdict.threshold_basis == basis
    assert verdict.score.agreement == 0.87


def test_a_score_below_the_bar_does_not_meet_it() -> None:
    """The boundary is inclusive; the failing case must actually fail."""
    basis = "provisional, unmeasured floor (B3)"
    assert not golden_verdict(
        _score(_version(), agreement=0.84), threshold=0.85, threshold_basis=basis
    ).meets_threshold
    assert golden_verdict(
        _score(_version(), agreement=0.85), threshold=0.85, threshold_basis=basis
    ).meets_threshold


# --------------------------------------------------------------------------
# Both stores answer the same protocol
# --------------------------------------------------------------------------


def test_both_stores_satisfy_the_prompt_store_protocol() -> None:
    """A caller must not be able to depend on which implementation it holds.

    The durable store's round-trips need a database and run under the migrated
    integration suite; this asserts the surfaces cannot drift apart, which is
    what would let a caller work against one and fail against the other.
    """
    assert isinstance(InMemoryPromptStore(), PromptStore)
    assert isinstance(PostgresPromptStore(), PromptStore)
    protocol_methods = {name for name in dir(PromptStore) if not name.startswith("_")}
    for implementation in (InMemoryPromptStore, PostgresPromptStore):
        missing = protocol_methods - set(dir(implementation))
        assert not missing, f"{implementation.__name__} is missing {sorted(missing)}"
