"""Prompt history, activation and golden-set scores (P7.10, §6.5).

This module holds the *record* around a prompt version: who saved it, when, why,
which version a task is currently pointed at, and what the golden set said about
each one. The version itself is content-addressed and immutable
(:mod:`backend.extraction.prompts.versioning`), so everything here is append-only
by construction — there is no text to overwrite.

Activation, and why rollback is not an operation
------------------------------------------------

"Which version is in force" is a *pointer*, and changing it is a configuration
change, so it is recorded as an event
(:mod:`backend.db.audit`, scope ``"prompt"``, field ``"active_version_hash"``)
exactly like every other configuration change in the system: §6.11 says config
changes are events, not mutations.

There is therefore no ``rollback`` function, and its absence is the design.
§6.5's "one-click rollback" is :meth:`PromptStore.activate` called with a hash
that is already in the history — the same call the operator makes when moving
*forward*. Nothing is restored, because nothing was destroyed; the earlier
version was never edited, so returning to it is byte-identical to the version
that was measured, and the golden-set score attached to that hash still
describes the text now in force. A rollback implemented as "restore the old
text into the current row" could not make that claim.

Golden-set scores, and the threshold that is not here
------------------------------------------------------

A score is attached to a ``(prompt name, version hash)`` pair, so it describes
text that cannot change underneath it. §5-P7 requires that any prompt change
re-runs the golden set and the score is recorded — that is
:meth:`PromptStore.attach_golden_score`.

What this module deliberately does **not** contain is a pass/fail verdict or a
default threshold. Per D-014, the directive's "≥ 85% agreement" is the
operator's *prior*, not a measurement: a model-agreement gate above the human
labeller's own intra-rater agreement cannot be met by any model, and a gate near
it is measuring label noise. The noise floor has not been measured yet (B3), so
any constant written here would be a fabricated threshold in the one place a
reader would take as authoritative (I3). :func:`golden_verdict` therefore takes
the threshold **and the statement of where it came from** as required arguments,
and refuses to judge without both.

Units
-----

``agreement`` is a **fraction in [0, 1]**, never a percentage — §8 exists
because unit confusion in this domain is silent, and "85" versus "0.85" is
exactly that class of bug. ``document_count`` is a count of documents.
Timestamps are timezone-aware UTC.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.extraction.prompts.versioning import PromptVersion

__all__ = [
    "AUDIT_FIELD",
    "AUDIT_SCOPE",
    "Activation",
    "GoldenSetScore",
    "GoldenVerdict",
    "InMemoryPromptStore",
    "NoActivePromptError",
    "PromptRecord",
    "PromptStore",
    "PromptVersionNotFoundError",
    "golden_verdict",
]

AUDIT_SCOPE = "prompt"
"""``config_change_event.scope`` under which prompt activations are recorded.

The scope :mod:`backend.db.audit` names for Phase 7 prompts, so the audit
browser (CC.4) groups them under one heading alongside ``extraction_task``.
"""

AUDIT_FIELD = "active_version_hash"
"""``config_change_event.field`` naming the pointer, per prompt.

The target is the prompt name, so one prompt's activation history is
``config_history(scope="prompt", target=name, field="active_version_hash")``.
"""


class PromptVersionNotFoundError(LookupError):
    """The requested ``(prompt name, version hash)`` pair has never been saved.

    Raised rather than returning ``None`` so that activating a hash that does
    not exist fails loudly. Pointing a task at an unsaved hash would leave it
    with no resolvable prompt at the next call, and the failure would surface
    far from the mistake.
    """


class NoActivePromptError(LookupError):
    """No version of this prompt has ever been activated.

    Distinct from "the prompt has no versions": a saved version is not in force
    until something points at it, and defaulting to "the newest one" would mean
    saving a draft silently deployed it.
    """


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One saved prompt version, with the circumstances of its saving.

    Attributes:
        version: The immutable, content-addressed version.
        actor: Who saved it, as asserted by the caller — the platform has no
            authentication yet (§1.1), so this is not a verified identity.
        notes: Free text explaining the change, or ``None``.
        correlation_id: Request id (D-003) it was saved under, or ``None``
            outside a request.
        recorded_at: When it was first saved, UTC.
        sequence: Dimensionless, strictly increasing per prompt in save order.
            The ordering key — a clock can tie, and under concurrency can run
            backwards relative to the sequence of events (see
            :mod:`backend.db.audit`), so ``recorded_at`` answers "when" and this
            answers "after what".
    """

    version: PromptVersion
    actor: str
    notes: str | None
    correlation_id: str | None
    recorded_at: dt.datetime
    sequence: int

    @property
    def version_hash(self) -> str:
        """The version's content address (32 lowercase hex characters)."""
        return self.version.version_hash


@dataclass(frozen=True, slots=True)
class Activation:
    """One act of pointing a prompt at a version.

    Attributes:
        name: The prompt.
        version_hash: The version put in force.
        previous_version_hash: The version that was in force before, or ``None``
            when this is the prompt's first activation.
        actor: Who did it (a caller assertion, not an authenticated identity).
        correlation_id: Request id (D-003), or ``None``.
        recorded_at: When, UTC.
        sequence: Dimensionless, strictly increasing per prompt in activation
            order. The ordering key, for the reason given on
            :attr:`PromptRecord.sequence`.
        is_rollback: True when ``version_hash`` was already in force at some
            earlier point in this prompt's activation history — i.e. the
            operator went *back*. Derived from the history, never asserted by
            the caller, so it cannot disagree with the record.
    """

    name: str
    version_hash: str
    previous_version_hash: str | None
    actor: str
    correlation_id: str | None
    recorded_at: dt.datetime
    sequence: int
    is_rollback: bool


@dataclass(frozen=True, slots=True)
class GoldenSetScore:
    """A golden-set result attached to one prompt version.

    Attributes:
        name: The prompt scored.
        version_hash: The exact version scored. Immutable, so this score can
            never come to describe different text.
        golden_set_id: Identifier of the labelled set used, including its own
            version. Two scores are only comparable when this matches; a score
            without it would silently mix a 300-document set with a 500-document
            one.
        agreement: Agreement with the human labels, a **fraction in [0, 1]**.
        document_count: Documents scored (count). Recorded because agreement
            over 12 documents and over 400 are not the same number.
        scored_at: When the run finished, UTC.
        actor: Who ran it (a caller assertion, not an authenticated identity).
        notes: Free text, or ``None``.
    """

    name: str
    version_hash: str
    golden_set_id: str
    agreement: float
    document_count: int
    scored_at: dt.datetime
    actor: str
    notes: str | None = None

    def __post_init__(self) -> None:
        """Reject a score that could not be a measurement.

        Raises:
            ValueError: if ``agreement`` is outside [0, 1] — which is what a
                caller passing a percentage looks like — or if
                ``document_count`` is not positive, or if ``name``,
                ``version_hash`` or ``golden_set_id`` is empty.
        """
        if not 0.0 <= self.agreement <= 1.0:
            msg = (
                f"agreement must be a fraction in [0, 1]; got {self.agreement!r}. "
                "A value above 1 is a percentage — the units are fractions (§8)"
            )
            raise ValueError(msg)
        if self.document_count < 1:
            msg = f"document_count must be >= 1; got {self.document_count}"
            raise ValueError(msg)
        for label, value in (
            ("name", self.name),
            ("version_hash", self.version_hash),
            ("golden_set_id", self.golden_set_id),
        ):
            if not value:
                msg = f"{label} must be non-empty"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GoldenVerdict:
    """Whether a golden-set score clears an explicitly-justified threshold.

    Attributes:
        score: The score judged.
        threshold: The bar it was judged against, a fraction in [0, 1].
        threshold_basis: Where that bar came from, in words. Required — see
            :func:`golden_verdict`.
        meets_threshold: ``score.agreement >= threshold``.
    """

    score: GoldenSetScore
    threshold: float
    threshold_basis: str
    meets_threshold: bool


def golden_verdict(
    score: GoldenSetScore, *, threshold: float, threshold_basis: str
) -> GoldenVerdict:
    """Judge a golden-set score against a threshold the caller must justify.

    There is no default threshold, and that is the point (D-014). Gate G7's
    literal "≥ 85%" is the operator's prior; the real bar is set *relative to
    the labeller's measured intra-rater agreement*, which has not been measured
    yet (B3). A default here would be a fabricated number in the one place a
    reader would take as authoritative (I3), and a gate above the measurement
    instrument's own reproducibility cannot be met by any model.

    So the caller supplies both the number and the sentence explaining where the
    number came from, and the sentence is stored with the verdict. When the
    floor is measured, that sentence cites the measurement; until then it can
    only honestly say the bar is provisional, which is exactly the fact that
    should be visible next to any figure derived from it.

    Args:
        score: The golden-set result.
        threshold: The agreement bar, a **fraction in [0, 1]** — not a
            percentage.
        threshold_basis: Non-empty statement of the threshold's derivation,
            e.g. "measured intra-rater agreement 0.81 (TESTING_LEDGER row 14),
            gate set at floor - 0.03".

    Returns:
        A :class:`GoldenVerdict`.

    Raises:
        ValueError: if ``threshold`` is outside [0, 1], or if
            ``threshold_basis`` is empty or blank.
    """
    if not 0.0 <= threshold <= 1.0:
        msg = (
            f"threshold must be a fraction in [0, 1]; got {threshold!r}. "
            "A value above 1 is a percentage — the units are fractions (§8)"
        )
        raise ValueError(msg)
    if not threshold_basis.strip():
        msg = (
            "threshold_basis is required and must be non-blank: a golden-set threshold "
            "with no stated derivation is an invented number (D-014, I3). State the "
            "measured human noise floor it was derived from, or say plainly that it is "
            "provisional and unmeasured"
        )
        raise ValueError(msg)
    return GoldenVerdict(
        score=score,
        threshold=threshold,
        threshold_basis=threshold_basis,
        meets_threshold=score.agreement >= threshold,
    )


@runtime_checkable
class PromptStore(Protocol):
    """Where prompt versions, activations and golden-set scores are kept.

    Async throughout, because the durable implementation
    (:mod:`backend.extraction.prompts.postgres`) is; the in-memory
    implementation matches the same surface so a caller cannot depend on which
    one it holds.

    Every method is append-only. Nothing in this protocol can change a saved
    version, retract an activation, or amend a score.
    """

    async def save(
        self,
        version: PromptVersion,
        *,
        actor: str,
        notes: str | None = None,
        correlation_id: str | None = None,
    ) -> PromptRecord:
        """Record a prompt version, or return the existing record for its hash."""
        ...

    async def get(self, name: str, version_hash: str) -> PromptRecord:
        """Return one saved version, or raise :class:`PromptVersionNotFoundError`."""
        ...

    async def history(self, name: str) -> tuple[PromptRecord, ...]:
        """Return every saved version of one prompt, newest first."""
        ...

    async def activate(
        self,
        name: str,
        version_hash: str,
        *,
        actor: str,
        correlation_id: str | None = None,
    ) -> Activation:
        """Point a prompt at a saved version. Rollback is this, with an older hash."""
        ...

    async def active(self, name: str) -> PromptRecord:
        """Return the version in force, or raise :class:`NoActivePromptError`."""
        ...

    async def activations(self, name: str) -> tuple[Activation, ...]:
        """Return every activation of one prompt, newest first."""
        ...

    async def attach_golden_score(self, score: GoldenSetScore) -> GoldenSetScore:
        """Attach a golden-set result to a saved version."""
        ...

    async def golden_scores(
        self, name: str, version_hash: str | None = None
    ) -> tuple[GoldenSetScore, ...]:
        """Return golden-set results, newest first, optionally for one version."""
        ...


@dataclass(slots=True)
class _PromptState:
    """Per-prompt state of the in-memory store."""

    records: dict[str, PromptRecord] = field(default_factory=dict)
    activations: list[Activation] = field(default_factory=list)
    scores: list[GoldenSetScore] = field(default_factory=list)


class InMemoryPromptStore:
    """A :class:`PromptStore` held in process memory.

    The reference implementation of the protocol's semantics, and what the unit
    tests exercise: it has the same append-only behaviour as the durable store
    without needing a database, so the rules (a saved version is never edited, a
    rollback is an activation, a score is bound to a hash) are tested where they
    are cheap to test.

    It is **not** a cache of the durable store and does not read from one.
    Anything saved here lives until the process ends. Suitable for tests, for a
    dry run, and for the golden-set harness's local scratch use; not for
    anything an operator will expect to still be there tomorrow.

    Not thread-safe, and not safe across event loops. The async methods never
    await, so there is no interleaving *within* this object under a single
    loop; two loops sharing one instance is unsupported.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._prompts: dict[str, _PromptState] = {}
        self._clock_counter = 0

    def _state(self, name: str) -> _PromptState:
        """Return the per-prompt state, creating it if this prompt is new."""
        return self._prompts.setdefault(name, _PromptState())

    def _now(self) -> dt.datetime:
        """Return a UTC timestamp for a new record.

        Uses the wall clock, like the durable store's database default. The
        value is descriptive only: every ordering in this module keys on
        ``sequence``, never on the clock, for the reason given in
        :mod:`backend.db.audit`.
        """
        return dt.datetime.now(tz=dt.UTC)

    async def save(
        self,
        version: PromptVersion,
        *,
        actor: str,
        notes: str | None = None,
        correlation_id: str | None = None,
    ) -> PromptRecord:
        """Record a version, or return the record already held for its hash.

        Saving is idempotent on the content address: re-saving identical text
        is not a new version, because a version *is* its content. The first
        save's actor, notes and timestamp are the ones kept — a later save of
        the same bytes changed nothing, so recording it as a change would be
        recording an event that did not happen.

        Args:
            version: The prompt version.
            actor: Who saved it (a caller assertion, not authenticated).
            notes: Free text explaining the change, or ``None``.
            correlation_id: Request id (D-003), or ``None``.

        Returns:
            The :class:`PromptRecord` for this hash.

        Raises:
            ValueError: if ``actor`` is empty or blank.
        """
        if not actor.strip():
            msg = "actor is required: a prompt version with no recorded author is not history"
            raise ValueError(msg)
        state = self._state(version.name)
        existing = state.records.get(version.version_hash)
        if existing is not None:
            return existing
        record = PromptRecord(
            version=version,
            actor=actor,
            notes=notes,
            correlation_id=correlation_id,
            recorded_at=self._now(),
            sequence=len(state.records) + 1,
        )
        state.records[version.version_hash] = record
        return record

    async def get(self, name: str, version_hash: str) -> PromptRecord:
        """Return one saved version.

        Args:
            name: The prompt.
            version_hash: Its content address.

        Returns:
            The record.

        Raises:
            PromptVersionNotFoundError: if the pair was never saved.
        """
        record = self._prompts.get(name, _PromptState()).records.get(version_hash)
        if record is None:
            msg = f"prompt {name!r} has no saved version with hash {version_hash!r}"
            raise PromptVersionNotFoundError(msg)
        return record

    async def history(self, name: str) -> tuple[PromptRecord, ...]:
        """Return every saved version of one prompt, newest first.

        Args:
            name: The prompt.

        Returns:
            Records ordered by :attr:`PromptRecord.sequence` descending; empty
            when the prompt has never been saved.
        """
        records = self._prompts.get(name, _PromptState()).records.values()
        return tuple(sorted(records, key=lambda r: r.sequence, reverse=True))

    async def activate(
        self,
        name: str,
        version_hash: str,
        *,
        actor: str,
        correlation_id: str | None = None,
    ) -> Activation:
        """Put a saved version in force.

        This is also how a rollback happens: pass a hash that is already in the
        history. Nothing is restored and nothing is rewritten — the pointer
        moves, and :attr:`Activation.is_rollback` records that it moved
        backwards, derived from the activation history rather than asserted.

        Args:
            name: The prompt.
            version_hash: The version to put in force. Must already be saved.
            actor: Who is doing it (a caller assertion, not authenticated).
            correlation_id: Request id (D-003), or ``None``.

        Returns:
            The recorded :class:`Activation`.

        Raises:
            PromptVersionNotFoundError: if the version was never saved.
            ValueError: if ``actor`` is empty or blank.
        """
        if not actor.strip():
            msg = "actor is required: an activation with no recorded author is not an audit trail"
            raise ValueError(msg)
        await self.get(name, version_hash)
        state = self._state(name)
        previous = state.activations[-1].version_hash if state.activations else None
        activation = Activation(
            name=name,
            version_hash=version_hash,
            previous_version_hash=previous,
            actor=actor,
            correlation_id=correlation_id,
            recorded_at=self._now(),
            sequence=len(state.activations) + 1,
            is_rollback=any(a.version_hash == version_hash for a in state.activations),
        )
        state.activations.append(activation)
        return activation

    async def active(self, name: str) -> PromptRecord:
        """Return the version currently in force.

        Args:
            name: The prompt.

        Returns:
            The record the most recent activation points at.

        Raises:
            NoActivePromptError: if the prompt has never been activated. A saved
                version is not in force until something points at it.
        """
        state = self._prompts.get(name, _PromptState())
        if not state.activations:
            msg = (
                f"prompt {name!r} has no active version; saving a version does not put it "
                "in force — activate a hash explicitly"
            )
            raise NoActivePromptError(msg)
        return await self.get(name, state.activations[-1].version_hash)

    async def activations(self, name: str) -> tuple[Activation, ...]:
        """Return every activation of one prompt, newest first.

        Args:
            name: The prompt.

        Returns:
            Activations ordered by :attr:`Activation.sequence` descending;
            empty when the prompt has never been activated.
        """
        return tuple(reversed(self._prompts.get(name, _PromptState()).activations))

    async def attach_golden_score(self, score: GoldenSetScore) -> GoldenSetScore:
        """Attach a golden-set result to a saved version.

        Args:
            score: The result. Its ``version_hash`` must already be saved —
                a score for text nobody can look up is not evidence.

        Returns:
            ``score`` unchanged.

        Raises:
            PromptVersionNotFoundError: if the scored version was never saved.
        """
        await self.get(score.name, score.version_hash)
        self._state(score.name).scores.append(score)
        return score

    async def golden_scores(
        self, name: str, version_hash: str | None = None
    ) -> tuple[GoldenSetScore, ...]:
        """Return golden-set results for a prompt, newest first.

        Args:
            name: The prompt.
            version_hash: Restrict to one version, or ``None`` for all.

        Returns:
            Scores in reverse attachment order; empty when there are none.
        """
        scores: Sequence[GoldenSetScore] = self._prompts.get(name, _PromptState()).scores
        if version_hash is not None:
            scores = [s for s in scores if s.version_hash == version_hash]
        return tuple(reversed(scores))
