"""The durable prompt store: versions, activations and golden scores in PostgreSQL (P7.10).

The same surface as :class:`~backend.extraction.prompts.store.PromptStore`, kept
where an operator will still find it tomorrow. The semantics are the in-memory
store's semantics — this module adds durability, an audit event per activation,
and the database-level guarantee that none of it can be edited afterwards.

Three tables, all append-only by trigger (migration 0010):

``extraction_prompt_version``
    One row per ``(name, version_hash)``. Saving is **idempotent on the content
    address**: re-saving identical text returns the existing row rather than
    writing a second one, because a version *is* its content and a later save of
    the same bytes changed nothing. Recording it as a change would be recording
    an event that did not happen.

``extraction_prompt_activation``
    One row per act of pointing a prompt at a version. This is where §6.5's
    "one-click rollback" lives, and it is not an operation: rolling back is
    :meth:`PostgresPromptStore.activate` with a hash already in the history —
    the same call as moving forward. Nothing is restored because nothing was
    destroyed, so the version returned to is byte-identical to the one that was
    measured and the golden-set score attached to that hash still describes the
    text now in force.

``extraction_golden_score``
    One row per golden-set run against one version. Bound to a hash, so a score
    can never come to describe different text.

Ordering under concurrency
--------------------------

Both ``sequence`` columns are read-max-then-append, which is a race. Writes for
one prompt are serialized by a transaction-scoped PostgreSQL advisory lock
(:func:`_prompt_lock_key`), the same construction
:mod:`backend.extraction.providers.assignments` uses, so two concurrent saves
produce sequences *n+1* and *n+2* rather than two rows claiming *n+1*. Different
prompts never contend.

Nothing in this module orders by the clock. ``recorded_at`` answers "when";
``sequence`` answers "after what". The reasoning is
:mod:`backend.db.audit`'s and applies unchanged.

State first, audit second
-------------------------

An activation is two writes in two transactions: the activation row, then the
``config_change_event``. :mod:`backend.db.audit` opens and commits its own
transaction and accepts no caller session, so the ordering is chosen rather than
assumed — if the audit write fails the exception propagates unswallowed, so the
operator learns the log may be missing an entry instead of the call reporting
success. The reverse order would let the log assert an activation that never
took effect. The same known gap
:mod:`backend.extraction.providers.registry` records, with the same fix: a
session-accepting recorder in :mod:`backend.db.audit`.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa

from backend.core.logging import get_logger
from backend.db import ingest_writer_session
from backend.db.audit import record_config_changes
from backend.db.models import (
    ExtractionGoldenScore,
    ExtractionPromptActivation,
    ExtractionPromptVersion,
)
from backend.extraction.prompts.store import (
    AUDIT_FIELD,
    AUDIT_SCOPE,
    Activation,
    GoldenSetScore,
    NoActivePromptError,
    PromptRecord,
    PromptVersionNotFoundError,
)
from backend.extraction.prompts.versioning import PromptVersion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["PostgresPromptStore"]

_logger = get_logger(__name__)

_LOCK_KEY_BYTES: Final = 8
"""Digest width of the advisory-lock key: 8 bytes == one PostgreSQL ``bigint``."""

_LOCK_NAMESPACE: Final = "extraction_prompt"
"""Prefix mixed into the advisory-lock digest so this module's locks cannot collide
with another module's locks over the same name."""


def _prompt_lock_key(name: str) -> int:
    """Return the advisory-lock key serializing writes to one prompt.

    Args:
        name: The prompt's name.

    Returns:
        A signed 64-bit integer for ``pg_advisory_xact_lock(bigint)``, derived
        from a BLAKE2b digest of the module namespace and the name joined by a
        NUL byte — which cannot occur inside either, so no two distinct keys
        collide by concatenation. Hashed in Python rather than with
        PostgreSQL's ``hashtext`` so the mapping is deterministic, documented
        and testable.
    """
    material = "\x00".join((_LOCK_NAMESPACE, name)).encode()
    digest = hashlib.blake2b(material, digest_size=_LOCK_KEY_BYTES).digest()
    return int.from_bytes(digest, "big", signed=True)


def _require_actor(actor: str, what: str) -> str:
    """Return ``actor``, or refuse an unattributed change.

    Args:
        actor: Who is making the change, as asserted by the caller. There is no
            authentication yet (§1.1), so this is not a verified identity.
        what: Noun naming the act, for the error message.

    Returns:
        The actor, unchanged.

    Raises:
        ValueError: if ``actor`` is empty or blank.
    """
    if not actor.strip():
        msg = f"actor is required: a {what} with no recorded author is not history"
        raise ValueError(msg)
    return actor


def _record(row: ExtractionPromptVersion) -> PromptRecord:
    """Convert a stored version row into the read model."""
    return PromptRecord(
        version=PromptVersion(
            name=row.name,
            system=row.system,
            template=row.template,
            schema_digest=row.schema_digest,
        ),
        actor=row.actor,
        notes=row.notes,
        correlation_id=row.correlation_id,
        recorded_at=row.recorded_at,
        sequence=row.sequence,
    )


def _activation(row: ExtractionPromptActivation) -> Activation:
    """Convert a stored activation row into the read model."""
    return Activation(
        name=row.name,
        version_hash=row.version_hash,
        previous_version_hash=row.previous_version_hash,
        actor=row.actor,
        correlation_id=row.correlation_id,
        recorded_at=row.recorded_at,
        sequence=row.sequence,
        is_rollback=row.is_rollback,
    )


def _score(row: ExtractionGoldenScore) -> GoldenSetScore:
    """Convert a stored golden-score row into the read model.

    ``agreement`` is ``NUMERIC`` in the database and comes back as a
    :class:`~decimal.Decimal`; it is widened to ``float`` because that is what
    every consumer needs and because at five decimal places the conversion is
    exact enough that no consumer can observe a difference. The database keeps
    the decimal.
    """
    return GoldenSetScore(
        name=row.name,
        version_hash=row.version_hash,
        golden_set_id=row.golden_set_id,
        agreement=float(row.agreement),
        document_count=row.document_count,
        scored_at=row.scored_at,
        actor=row.actor,
        notes=row.notes,
    )


class PostgresPromptStore:
    """A :class:`~backend.extraction.prompts.store.PromptStore` in PostgreSQL.

    Stateless: every method opens its own writer session. There is nothing to
    construct and nothing to close, so a caller can hold one instance for the
    process lifetime or build one per request without either being wrong.
    """

    async def _version_row(
        self, session: AsyncSession, name: str, version_hash: str
    ) -> ExtractionPromptVersion | None:
        """Return one saved version row on an open session, or ``None``."""
        statement = sa.select(ExtractionPromptVersion).where(
            ExtractionPromptVersion.name == name,
            ExtractionPromptVersion.version_hash == version_hash,
        )
        return (await session.scalars(statement)).one_or_none()

    async def save(
        self,
        version: PromptVersion,
        *,
        actor: str,
        notes: str | None = None,
        correlation_id: str | None = None,
    ) -> PromptRecord:
        """Record a prompt version, or return the record already held for its hash.

        Idempotent on the content address — see the module docstring.

        Args:
            version: The prompt version to save.
            actor: Who saved it (a caller assertion, not authenticated).
            notes: Free text explaining the change, or ``None``.
            correlation_id: Request id (D-003), or ``None`` to leave it unset.

        Returns:
            The :class:`~backend.extraction.prompts.store.PromptRecord` for this
            hash — the existing one when the text had already been saved.

        Raises:
            ValueError: if ``actor`` is empty or blank.
        """
        _require_actor(actor, "prompt version")
        version_hash = version.version_hash
        lock_key = sa.literal(_prompt_lock_key(version.name), sa.BigInteger)
        async with ingest_writer_session() as session:
            await session.execute(sa.select(sa.func.pg_advisory_xact_lock(lock_key)))
            existing = await self._version_row(session, version.name, version_hash)
            if existing is not None:
                return _record(existing)
            next_sequence = (
                await session.scalars(
                    sa.select(sa.func.max(ExtractionPromptVersion.sequence)).where(
                        ExtractionPromptVersion.name == version.name
                    )
                )
            ).one() or 0
            row = ExtractionPromptVersion(
                name=version.name,
                version_hash=version_hash,
                system=version.system,
                template=version.template,
                schema_digest=version.schema_digest,
                actor=actor,
                notes=notes,
                correlation_id=correlation_id,
                sequence=next_sequence + 1,
            )
            session.add(row)
            await session.flush()
            await session.refresh(row)
            record = _record(row)
            await session.commit()
        _logger.info(
            "prompt_version_saved",
            prompt=version.name,
            version_hash=version_hash,
            sequence=record.sequence,
            actor=actor,
        )
        return record

    async def get(self, name: str, version_hash: str) -> PromptRecord:
        """Return one saved version.

        Args:
            name: The prompt.
            version_hash: Its content address.

        Returns:
            The record.

        Raises:
            PromptVersionNotFoundError: the pair was never saved.
        """
        async with ingest_writer_session() as session:
            row = await self._version_row(session, name, version_hash)
        if row is None:
            msg = f"prompt {name!r} has no saved version with hash {version_hash!r}"
            raise PromptVersionNotFoundError(msg)
        return _record(row)

    async def history(self, name: str) -> tuple[PromptRecord, ...]:
        """Return every saved version of one prompt, newest first.

        Args:
            name: The prompt.

        Returns:
            Records ordered by ``sequence`` descending; empty when the prompt
            has never been saved. Absence of history is not an error for a
            history read.
        """
        statement = (
            sa.select(ExtractionPromptVersion)
            .where(ExtractionPromptVersion.name == name)
            .order_by(ExtractionPromptVersion.sequence.desc())
        )
        async with ingest_writer_session() as session:
            rows = (await session.scalars(statement)).all()
        return tuple(_record(row) for row in rows)

    async def activate(
        self,
        name: str,
        version_hash: str,
        *,
        actor: str,
        correlation_id: str | None = None,
    ) -> Activation:
        """Put a saved version in force, and record the change as an event.

        Rollback is this call with an older hash (§6.5). ``is_rollback`` is
        **derived** from the activation history inside the same transaction that
        writes the row, never asserted by the caller, so it cannot disagree with
        the record.

        Args:
            name: The prompt.
            version_hash: The version to put in force. Must already be saved.
            actor: Who is doing it (a caller assertion, not authenticated).
            correlation_id: Request id (D-003), or ``None`` to resolve it from
                the request in flight when the audit event is written.

        Returns:
            The recorded :class:`~backend.extraction.prompts.store.Activation`.

        Raises:
            PromptVersionNotFoundError: the version was never saved. Pointing a
                task at an unsaved hash would leave it with no resolvable prompt
                at the next call, and the failure would surface far from the
                mistake.
            ValueError: if ``actor`` is empty or blank.
        """
        _require_actor(actor, "prompt activation")
        lock_key = sa.literal(_prompt_lock_key(name), sa.BigInteger)
        async with ingest_writer_session() as session:
            await session.execute(sa.select(sa.func.pg_advisory_xact_lock(lock_key)))
            if await self._version_row(session, name, version_hash) is None:
                msg = f"prompt {name!r} has no saved version with hash {version_hash!r}"
                raise PromptVersionNotFoundError(msg)
            prior = (
                await session.scalars(
                    sa.select(ExtractionPromptActivation)
                    .where(ExtractionPromptActivation.name == name)
                    .order_by(ExtractionPromptActivation.sequence.desc())
                )
            ).all()
            row = ExtractionPromptActivation(
                name=name,
                version_hash=version_hash,
                previous_version_hash=prior[0].version_hash if prior else None,
                actor=actor,
                correlation_id=correlation_id,
                sequence=len(prior) + 1,
                is_rollback=any(item.version_hash == version_hash for item in prior),
            )
            session.add(row)
            await session.flush()
            await session.refresh(row)
            activation = _activation(row)
            await session.commit()
        await record_config_changes(
            AUDIT_SCOPE,
            name,
            {AUDIT_FIELD: version_hash},
            actor=actor,
            correlation_id=correlation_id,
        )
        _logger.info(
            "prompt_version_activated",
            prompt=name,
            version_hash=version_hash,
            previous_version_hash=activation.previous_version_hash,
            is_rollback=activation.is_rollback,
            actor=actor,
        )
        return activation

    async def active(self, name: str) -> PromptRecord:
        """Return the version currently in force.

        Args:
            name: The prompt.

        Returns:
            The record the most recent activation points at.

        Raises:
            NoActivePromptError: the prompt has never been activated. Saving a
                version does not put it in force — defaulting to "the newest
                one" would mean saving a draft silently deployed it.
        """
        statement = (
            sa.select(ExtractionPromptActivation)
            .where(ExtractionPromptActivation.name == name)
            .order_by(ExtractionPromptActivation.sequence.desc())
            .limit(1)
        )
        async with ingest_writer_session() as session:
            row = (await session.scalars(statement)).one_or_none()
            if row is None:
                msg = (
                    f"prompt {name!r} has no active version; saving a version does not put it "
                    "in force — activate a hash explicitly"
                )
                raise NoActivePromptError(msg)
            version_row = await self._version_row(session, name, row.version_hash)
        if version_row is None:  # pragma: no cover - the FK makes this unreachable
            msg = (
                f"prompt {name!r} is active on hash {row.version_hash!r}, which has no saved "
                "version. The foreign key should make this impossible; the schema is damaged"
            )
            raise PromptVersionNotFoundError(msg)
        return _record(version_row)

    async def activations(self, name: str) -> tuple[Activation, ...]:
        """Return every activation of one prompt, newest first.

        Args:
            name: The prompt.

        Returns:
            Activations ordered by ``sequence`` descending; empty when the
            prompt has never been activated.
        """
        statement = (
            sa.select(ExtractionPromptActivation)
            .where(ExtractionPromptActivation.name == name)
            .order_by(ExtractionPromptActivation.sequence.desc())
        )
        async with ingest_writer_session() as session:
            rows = (await session.scalars(statement)).all()
        return tuple(_activation(row) for row in rows)

    async def attach_golden_score(self, score: GoldenSetScore) -> GoldenSetScore:
        """Attach a golden-set result to a saved version.

        Args:
            score: The result. Its ``version_hash`` must already be saved — a
                score for text nobody can look up is not evidence.

        Returns:
            ``score`` unchanged.

        Raises:
            PromptVersionNotFoundError: the scored version was never saved.
        """
        async with ingest_writer_session() as session:
            if await self._version_row(session, score.name, score.version_hash) is None:
                msg = (
                    f"prompt {score.name!r} has no saved version with hash "
                    f"{score.version_hash!r}; a golden-set score must name text that can be "
                    "looked up"
                )
                raise PromptVersionNotFoundError(msg)
            session.add(
                ExtractionGoldenScore(
                    name=score.name,
                    version_hash=score.version_hash,
                    golden_set_id=score.golden_set_id,
                    # Bound as Decimal rather than float: the column is
                    # NUMERIC(6, 5), and handing the driver a float there is
                    # what turns 0.87 into 0.86999999999999999 on a quality
                    # trend chart (§6.5).
                    agreement=Decimal(str(score.agreement)).quantize(Decimal("0.00001")),
                    document_count=score.document_count,
                    scored_at=score.scored_at,
                    actor=score.actor,
                    notes=score.notes,
                )
            )
            await session.commit()
        _logger.info(
            "prompt_golden_score_attached",
            prompt=score.name,
            version_hash=score.version_hash,
            golden_set_id=score.golden_set_id,
            agreement=score.agreement,
            document_count=score.document_count,
        )
        return score

    async def golden_scores(
        self, name: str, version_hash: str | None = None
    ) -> tuple[GoldenSetScore, ...]:
        """Return golden-set results for a prompt, newest first.

        Args:
            name: The prompt.
            version_hash: Restrict to one version, or ``None`` for all.

        Returns:
            Scores ordered by ``score_id`` descending — insertion order, not the
            clock; empty when there are none.
        """
        statement = sa.select(ExtractionGoldenScore).where(ExtractionGoldenScore.name == name)
        if version_hash is not None:
            statement = statement.where(ExtractionGoldenScore.version_hash == version_hash)
        statement = statement.order_by(ExtractionGoldenScore.score_id.desc())
        async with ingest_writer_session() as session:
            rows = (await session.scalars(statement)).all()
        return tuple(_score(row) for row in rows)
