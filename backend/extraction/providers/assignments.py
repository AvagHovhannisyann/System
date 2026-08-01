"""Per-task model assignment, versioned rather than mutated (P7.1, §6.5).

§6.5 states the rule this module implements: each extraction task is
independently assigned a provider, model, temperature, max tokens and timeout,
and **changing an assignment creates a new configuration version rather than
mutating the current one**. So the stored unit is a version, not a task:

* ``assign_task_model`` appends version *n+1*; it never touches version *n*;
* the assignment in force is the greatest version for that task;
* every superseded version stays readable exactly as written, enforced by
  migration 0009's ``BEFORE UPDATE OR DELETE`` trigger rather than by
  convention — no session and no role edits a past version.

Why versions and not just the audit log
---------------------------------------

Every change here is *also* recorded in ``config_change_event``
(:mod:`backend.db.audit`) with actor and correlation id, so the two records
overlap. They answer different questions and both are needed. The audit log
answers "what did the operator change", across every subsystem, in one place.
This table answers "what configuration did extraction run *N* use" — and an
extraction result that can only name its own configuration by replaying a
cross-subsystem event log is a reproducibility hazard (I2), because the answer
then depends on correctly reconstructing state rather than on reading a row.

Temperature 0
-------------

:data:`DEFAULT_TEMPERATURE` is ``0.0`` and every entry point defaults to it,
because §5-P7 requires temperature 0 everywhere for extraction. Non-zero values
are *storable* — §6.5 makes temperature a per-task setting, and a registry that
silently refused to record what the operator configured would be lying to them
— but nothing in this module ever supplies a non-zero default, and a test
pins that.

Ordering under concurrency
--------------------------

Assigning a version is read-max-then-append, which is a race. Writes for one
task are serialized by a transaction-scoped PostgreSQL advisory lock
(:func:`_task_lock_key`), so two concurrent assignments to the same task
produce versions *n+1* and *n+2* rather than two rows claiming *n+1* (which the
unique constraint would reject anyway — the lock turns a lost write into a
queued one). Different tasks never contend.

State first, audit second — the same ordering and the same known limitation as
:mod:`backend.extraction.providers.registry`; its module docstring carries the
reasoning and the pointer to what would close the gap.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa

from backend.core.logging import get_logger
from backend.db import ingest_writer_session
from backend.db.audit import record_config_changes
from backend.db.models import ExtractionModelAssignment
from backend.extraction.providers.catalog import Provider

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "AUDIT_SCOPE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "MAX_TEMPERATURE",
    "AssignmentNotConfiguredError",
    "ModelAssignment",
    "assign_task_model",
    "assignment_versions",
    "current_assignment",
    "list_current_assignments",
]

_logger = get_logger(__name__)

AUDIT_SCOPE: Final = "extraction_task"
"""``config_change_event.scope`` under which assignment changes are recorded.

The same scope :mod:`backend.db.audit` names as the Phase 7 example, so the
audit browser (CC.4) groups extraction configuration under one heading.
"""

DEFAULT_TEMPERATURE: Final = 0.0
"""Sampling temperature every entry point defaults to. **Zero, per §5-P7.**"""

MAX_TEMPERATURE: Final = 2.0
"""Largest temperature accepted (dimensionless).

The union of the providers' accepted ranges rather than any one provider's
ceiling: a value a specific provider rejects is that provider's error to state
at call time, and hard-coding one vendor's limit here would silently misreport
another's.
"""

DEFAULT_MAX_TOKENS: Final = 1024
"""Response cap in tokens when the caller does not choose one.

A bound, not a recommendation. Extraction returns schema-validated structured
values (P7.3), not prose, so a low cap is the right shape of default; a task
that needs more says so explicitly.
"""

DEFAULT_TIMEOUT_S: Final = 60.0
"""Per-request wall-clock timeout in seconds when the caller does not choose one."""

_LOCK_KEY_BYTES: Final = 8
"""Digest width of the advisory-lock key: 8 bytes == one PostgreSQL ``bigint``."""

_LOCK_NAMESPACE: Final = "extraction_model_assignment"
"""Prefix mixed into the advisory-lock digest so this module's locks cannot collide
with another module's locks over the same task name."""


class AssignmentNotConfiguredError(LookupError):
    """The requested extraction task has no assignment at all.

    Distinct from "the task's current assignment is unusable": that is a
    provider-credential or provider-side failure, raised elsewhere. Raised
    rather than returning ``None`` so a caller about to run an extraction
    cannot proceed on a falsy value it forgot to check.
    """


@dataclass(frozen=True, slots=True)
class ModelAssignment:
    """One version of one task's model assignment, as recorded.

    Attributes:
        task: the extraction task this configures.
        version: dimensionless configuration version; 1 for the task's first
            assignment, +1 per change. The greatest version is in force.
        provider: provider the task calls.
        model: provider-side model identifier, verbatim.
        temperature: dimensionless sampling temperature, 0 to 2.
        max_tokens: response cap in tokens (count).
        timeout_s: per-request wall-clock timeout in seconds.
        actor: who made this assignment; a caller assertion, not an
            authenticated identity.
        correlation_id: request id (D-003) it was made under, or ``None``.
        recorded_at: when it was recorded, UTC, from the database clock.
    """

    task: str
    version: int
    provider: Provider
    model: str
    temperature: float
    max_tokens: int
    timeout_s: float
    actor: str
    correlation_id: str | None
    recorded_at: dt.datetime


def _assignment(row: ExtractionModelAssignment) -> ModelAssignment:
    """Convert a stored row into the read model.

    ``temperature`` and ``timeout_s`` are ``NUMERIC`` in the database and come
    back as :class:`~decimal.Decimal`; they are widened to ``float`` here
    because that is what every consumer needs — a request payload, a JSON audit
    value — and because at three decimal places the conversion is exact enough
    that no consumer can observe a difference. The database keeps the decimal.
    """
    return ModelAssignment(
        task=row.task,
        version=row.version,
        provider=Provider(row.provider),
        model=row.model,
        temperature=float(row.temperature),
        max_tokens=row.max_tokens,
        timeout_s=float(row.timeout_s),
        actor=row.actor,
        correlation_id=row.correlation_id,
        recorded_at=row.recorded_at,
    )


def _task_lock_key(task: str) -> int:
    """Return the PostgreSQL advisory-lock key serializing writes to one task.

    Args:
        task: the extraction task name.

    Returns:
        A signed 64-bit integer for ``pg_advisory_xact_lock(bigint)``, derived
        from a BLAKE2b digest of the module namespace and the task joined by a
        NUL byte (which cannot occur inside either, so no two distinct keys
        collide by concatenation). Hashed in Python rather than with
        PostgreSQL's ``hashtext`` so the mapping is deterministic, documented
        and testable instead of resting on an internal server function — the
        same construction :mod:`backend.db.audit` uses for configuration keys.
    """
    material = "\x00".join((_LOCK_NAMESPACE, task)).encode()
    digest = hashlib.blake2b(material, digest_size=_LOCK_KEY_BYTES).digest()
    return int.from_bytes(digest, "big", signed=True)


def _require_task(task: str) -> str:
    """Return the task name, or refuse it.

    Whitespace is rejected rather than trimmed, matching
    :mod:`backend.db.audit`: silently trimming would make ``"tone"`` and
    ``" tone"`` the same key at write time and different keys in every log
    line, dashboard and hand-written query about them.

    Raises:
        ValueError: the name is empty or carries leading/trailing whitespace.
    """
    if not task:
        msg = "task must be a non-empty string"
        raise ValueError(msg)
    if task != task.strip():
        msg = (
            f"task must not have leading or trailing whitespace; got {task!r}. Whitespace "
            "is rejected rather than trimmed so two spellings of one task cannot both exist"
        )
        raise ValueError(msg)
    return task


def _require_settings(temperature: float, max_tokens: int, timeout_s: float) -> None:
    """Validate the numeric settings before they reach the database.

    The database CHECK constraints assert the same bounds; these exist so the
    caller gets a message naming the offending setting rather than an
    ``IntegrityError`` quoting a constraint name.

    Raises:
        ValueError: temperature is outside 0 to :data:`MAX_TEMPERATURE`,
            ``max_tokens`` is below 1, or ``timeout_s`` is not positive.
    """
    if not 0.0 <= temperature <= MAX_TEMPERATURE:
        msg = f"temperature must be between 0 and {MAX_TEMPERATURE}; got {temperature!r}"
        raise ValueError(msg)
    if max_tokens < 1:
        msg = f"max_tokens must be at least 1 token; got {max_tokens!r}"
        raise ValueError(msg)
    if timeout_s <= 0:
        msg = f"timeout_s must be a positive number of seconds; got {timeout_s!r}"
        raise ValueError(msg)


async def _greatest_version(session: AsyncSession, task: str) -> int:
    """Return the greatest recorded version for *task*, or 0 when it has none."""
    statement = sa.select(sa.func.max(ExtractionModelAssignment.version)).where(
        ExtractionModelAssignment.task == task
    )
    return (await session.scalars(statement)).one() or 0


async def assign_task_model(
    task: str,
    provider: Provider,
    model: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    actor: str,
    correlation_id: str | None = None,
) -> ModelAssignment:
    """Record a **new version** of *task*'s model assignment and return it.

    This is the only way an assignment changes. Nothing updates a stored
    version — the previous one remains readable at its own version number
    (§6.5), and the database refuses an UPDATE or DELETE against it. Recording
    the same settings again is still a new version: the operator pressed the
    button, and a configuration history that disagrees with the operator's
    memory of having done so is worse than a redundant row.

    The provider is **not** required to have a credential configured. An
    assignment is a statement of intent, and forcing key configuration first
    would invert the natural setup order; the refusal belongs at call time,
    where :mod:`backend.extraction.providers.probe` and the extraction client
    raise rather than invent a result (I3).

    Args:
        task: extraction task to configure. Non-empty, no surrounding
            whitespace.
        provider: provider the task should call.
        model: provider-side model identifier, stored verbatim and never
            interpreted here. Non-empty.
        temperature: dimensionless sampling temperature, 0 to :data:`MAX_TEMPERATURE`.
            Defaults to :data:`DEFAULT_TEMPERATURE`, which is 0 (§5-P7).
        max_tokens: response cap in tokens (count, >= 1).
        timeout_s: per-request wall-clock timeout in seconds (> 0).
        actor: who is making the change. Recorded verbatim in both this table
            and the audit log; a caller assertion, not an authenticated
            identity.
        correlation_id: request id to record, or ``None`` to resolve it from
            the request in flight.

    Returns:
        The newly recorded version.

    Raises:
        ValueError: the task or model is empty or whitespace-padded, or a
            numeric setting is out of range.
    """
    checked_task = _require_task(task)
    if not model or model != model.strip():
        msg = f"model must be a non-empty string without surrounding whitespace; got {model!r}"
        raise ValueError(msg)
    _require_settings(temperature, max_tokens, timeout_s)
    lock_key = sa.literal(_task_lock_key(checked_task), sa.BigInteger)
    async with ingest_writer_session() as session:
        await session.execute(sa.select(sa.func.pg_advisory_xact_lock(lock_key)))
        row = ExtractionModelAssignment(
            task=checked_task,
            version=await _greatest_version(session, checked_task) + 1,
            provider=provider.value,
            model=model,
            # Bound as Decimal rather than float: the column is NUMERIC, and
            # handing the driver a float there is what turns 0.7 into
            # 0.6999999999999999 in somebody's configuration page.
            temperature=Decimal(str(temperature)),
            max_tokens=max_tokens,
            timeout_s=Decimal(str(timeout_s)),
            actor=actor,
            correlation_id=correlation_id,
        )
        session.add(row)
        await session.flush()
        await session.refresh(row)
        assignment = _assignment(row)
        await session.commit()
    await record_config_changes(
        AUDIT_SCOPE,
        checked_task,
        {
            "version": assignment.version,
            "provider": assignment.provider.value,
            "model": assignment.model,
            "temperature": assignment.temperature,
            "max_tokens": assignment.max_tokens,
            "timeout_s": assignment.timeout_s,
        },
        actor=actor,
        correlation_id=correlation_id,
    )
    _logger.info(
        "extraction_model_assigned",
        task=checked_task,
        version=assignment.version,
        provider=assignment.provider.value,
        model=assignment.model,
        temperature=assignment.temperature,
        actor=actor,
    )
    return assignment


async def current_assignment(task: str) -> ModelAssignment:
    """Return the assignment version currently in force for *task*.

    Args:
        task: extraction task to read.

    Returns:
        The greatest recorded version for the task.

    Raises:
        AssignmentNotConfiguredError: the task has never been assigned.
        ValueError: the task name is empty or whitespace-padded.
    """
    checked_task = _require_task(task)
    statement = (
        sa.select(ExtractionModelAssignment)
        .where(ExtractionModelAssignment.task == checked_task)
        .order_by(ExtractionModelAssignment.version.desc())
        .limit(1)
    )
    async with ingest_writer_session() as session:
        row = (await session.scalars(statement)).one_or_none()
    if row is None:
        msg = (
            f"extraction task {checked_task!r} has no model assignment; assign a provider "
            "and model before running it"
        )
        raise AssignmentNotConfiguredError(msg)
    return _assignment(row)


async def assignment_versions(task: str) -> tuple[ModelAssignment, ...]:
    """Return every recorded version of *task*'s assignment, newest first.

    The superseded versions are the point: §6.5 requires the old configuration
    to remain readable after a change, and this is where a reader reads it.

    Args:
        task: extraction task to read.

    Returns:
        Every version, greatest first. Empty when the task has never been
        assigned — absence of configuration is not an error for a history read,
        only for :func:`current_assignment`.

    Raises:
        ValueError: the task name is empty or whitespace-padded.
    """
    checked_task = _require_task(task)
    statement = (
        sa.select(ExtractionModelAssignment)
        .where(ExtractionModelAssignment.task == checked_task)
        .order_by(ExtractionModelAssignment.version.desc())
    )
    async with ingest_writer_session() as session:
        rows = (await session.scalars(statement)).all()
    return tuple(_assignment(row) for row in rows)


async def list_current_assignments() -> tuple[ModelAssignment, ...]:
    """Return the version in force for every assigned task, ordered by task name.

    The shape the Agents & Extraction page wants: one round trip for the whole
    table rather than one per task.
    """
    statement = (
        sa.select(ExtractionModelAssignment)
        .distinct(ExtractionModelAssignment.task)
        .order_by(ExtractionModelAssignment.task, ExtractionModelAssignment.version.desc())
    )
    async with ingest_writer_session() as session:
        rows = (await session.scalars(statement)).all()
    return tuple(_assignment(row) for row in rows)
