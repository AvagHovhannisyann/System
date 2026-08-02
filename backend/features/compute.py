"""The compute path: where an availability lag becomes an enforced as-of (P5.1).

Declaring a lag is documentation. This module is what makes it a control.

Invariant I1 says no query may return a fact whose ``knowledge_time`` is later
than the query's ``as_of``. :func:`backend.db.as_of` enforces that for a given
instant. The question this module answers is *which instant a feature is
entitled to*, and the answer is fixed by the feature's own declaration::

    as_of = midnight_utc(compute_date) - spec.availability_lag

The enforcement is structural rather than advisory, in three layers:

**1. The instant is derived, never supplied.** :func:`compute_feature` computes
it from the declaration. There is exactly one ``as_of(...)`` call site in this
package and it is passed the derived value; no argument reaches the session
that could widen it.

**2. A caller may only move the instant backwards.** ``requested_as_of`` exists
because a historical replay legitimately wants to reconstruct a feature as it
stood at some *earlier* instant. Anything fresher than the declaration permits
raises :class:`~backend.features.errors.AvailabilityLagViolationError` before a
session is opened, so the refusal costs no I/O and leaves nothing half-done.

**3. The computation is handed a pinned session and nothing else.** It receives
an ``AsyncSession`` already scoped by :func:`backend.db.as_of` and a frozen
request. It is given no engine, no session factory and no way to construct
either — :mod:`backend.db.engine` is banned from import outside ``backend/db``
by the project's ruff configuration (D-011 layer 3). Below that, the Core-level
guard in :mod:`backend.db._guard` is **default-deny**: a statement that is not
the as-of rewrite raises at the cursor boundary rather than executing
unversioned. So a computation cannot see fresher data by writing cleverer SQL;
it would have to acquire a differently-pinned session, and the API never gives
it the means.

What this module does *not* do is guess a feature's value. The registry's
extension point is an async callable; the baseline factors that implement those
callables are P5.3's task and are not stubbed here. A feature with no
computation is a feature that cannot be registered, which is the honest state
of affairs before P5.3 lands (I3).

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

**Values are ``float64``, one per requested security, in the declaration's
units,** aligned positionally with ``request.security_ids``. Alignment is
positional rather than by key because every downstream consumer — the P5.2
transform pipeline, the LightGBM design matrix — is array-shaped, and a
misaligned array is the one defect that produces plausible numbers attached to
the wrong companies.

**NaN means "not available"** and propagates. Per the frozen contract, a NaN is
never filled with a plausible number; that would be I3 fabrication. A
computation that has no value for a security returns NaN for it.

**All instants are timezone-aware UTC**, matching the store's
``knowledge_time`` column and :func:`backend.db.as_of`, which rejects naive or
non-UTC timestamps outright.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from backend.db import as_of
from backend.features.errors import (
    AvailabilityLagViolationError,
    FeatureComputeError,
    MalformedFeatureVectorError,
)
from backend.features.registry import default_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.registry import FeatureRegistry
    from backend.features.spec import FeatureSpec

__all__ = [
    "FeatureComputation",
    "FeatureComputeRequest",
    "FeatureVector",
    "FloatArray",
    "compute_feature",
    "resolve_as_of",
]

type FloatArray = npt.NDArray[np.float64]
"""One-dimensional ``float64`` array. Units are stated by the feature's spec."""


@dataclass(frozen=True, slots=True)
class FeatureComputeRequest:
    """Everything a feature computation is told about the job it is doing.

    Frozen, and deliberately small: a computation is given the securities to
    value, the date it is valuing them for, and the instant its session is
    pinned at — the last of these for logging and for arithmetic that needs a
    reference time, **not** as something it can act on. Re-pinning is not
    possible from here; see the module docstring.

    Attributes:
        feature: name of the feature being computed.
        compute_date: the calendar date the value is for (the rebalance date).
        as_of: the instant the session is pinned at (tz-aware UTC), equal to
            ``midnight_utc(compute_date) - availability_lag`` unless the caller
            asked for an older instant. Facts with ``knowledge_time`` at or
            before it are visible; nothing later is.
        security_ids: securities to value, in the order the returned values
            must be in. May be empty (an empty universe on a date is a fact,
            not an error); never contains duplicates.
    """

    feature: str
    compute_date: dt.date
    as_of: dt.datetime
    security_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """One feature's values for one date, with the provenance to interpret them.

    Attributes:
        feature: name of the feature.
        units: the declaration's units, carried alongside the numbers so a
            consumer never has to look them up to know what it is holding
            (directive §8).
        compute_date: the calendar date the values are for.
        as_of: the instant the values were read at (tz-aware UTC). Recording it
            is what lets a downstream artifact prove which knowledge set
            produced it (I2).
        security_ids: securities, in the order of :attr:`values`.
        values: one ``float64`` per security, in :attr:`units`. NaN means "not
            available" and is never a filler for a value that could not be
            computed.
    """

    feature: str
    units: str
    compute_date: dt.date
    as_of: dt.datetime
    security_ids: tuple[int, ...]
    values: FloatArray


class FeatureComputation(Protocol):
    """The extension point: an async callable that values one feature.

    Implemented by P5.3's baseline factors and P7's LLM features, registered
    with :func:`backend.features.registry.feature`. Nothing in this package
    implements one — a plausible-looking stub would be exactly the fabrication
    invariant I3 forbids.
    """

    async def __call__(
        self, session: AsyncSession, request: FeatureComputeRequest, /
    ) -> FloatArray:
        """Compute the feature for the requested securities.

        Args:
            session: an ``AsyncSession`` **already pinned** by
                :func:`backend.db.as_of` to ``request.as_of``. Read through it
                and do not open another; it is the only session that reflects
                what this feature is allowed to know.
            request: the securities, the compute date, and the pinned instant.

        Returns:
            One ``float64`` per entry of ``request.security_ids``, in that
            order, in the units the declaration states. NaN for securities the
            value is unavailable for.
        """
        ...


def resolve_as_of(
    spec: FeatureSpec,
    compute_date: dt.date,
    *,
    requested_as_of: dt.datetime | None = None,
) -> dt.datetime:
    """Return the as-of instant a feature may be computed at, or raise.

    The single point where the declared availability lag turns into a number.
    Pure: no I/O, no session, no clock reading — so the rule it implements can
    be tested exhaustively without a database.

    Args:
        spec: the feature's declaration; its ``availability_lag`` is the
            authority.
        compute_date: the calendar date being computed for (a ``date``, not a
            ``datetime``).
        requested_as_of: optional caller override, tz-aware UTC. May be
            **older** than the permitted instant — a historical replay
            reconstructing what was knowable earlier is I1-safe and legitimate.
            May not be fresher.

    Returns:
        The permitted instant (tz-aware UTC) when ``requested_as_of`` is
        ``None``, otherwise ``requested_as_of`` itself.

    Raises:
        AvailabilityLagViolationError: if ``requested_as_of`` is later than
            ``midnight_utc(compute_date) - spec.availability_lag``.
        FeatureComputeError: if ``compute_date`` is a ``datetime``, or
            ``requested_as_of`` is naive or not UTC. Naive timestamps are not
            merely refused for tidiness: comparing one against the permitted
            instant raises ``TypeError`` in Python, so a naive value could
            never be safely bounds-checked at all.
    """
    permitted = spec.knowledge_cutoff(compute_date)
    if requested_as_of is None:
        return permitted
    if requested_as_of.tzinfo is None or requested_as_of.utcoffset() is None:
        msg = (
            f"requested_as_of for feature {spec.name!r} is naive ({requested_as_of!r}); "
            f"it must be timezone-aware UTC. A naive instant cannot be compared "
            f"against the {permitted.isoformat()} the declaration permits, so it "
            f"cannot be bounds-checked."
        )
        raise FeatureComputeError(msg)
    if requested_as_of.utcoffset() != dt.timedelta(0):
        msg = (
            f"requested_as_of for feature {spec.name!r} has UTC offset "
            f"{requested_as_of.utcoffset()}; UTC (offset 0) is required, matching "
            f"the store's knowledge_time column and backend.db.as_of()."
        )
        raise FeatureComputeError(msg)
    if requested_as_of > permitted:
        raise AvailabilityLagViolationError(
            feature=spec.name,
            compute_date=compute_date,
            availability_lag=spec.availability_lag,
            permitted_as_of=permitted,
            requested_as_of=requested_as_of,
        )
    return requested_as_of


async def compute_feature(
    name: str,
    *,
    compute_date: dt.date,
    security_ids: Sequence[int],
    requested_as_of: dt.datetime | None = None,
    registry: FeatureRegistry | None = None,
) -> FeatureVector:
    """Compute one registered feature for one date, pinned to its declared lag.

    The only sanctioned way to run a feature computation. The order of
    operations is the enforcement: the as-of instant is resolved from the
    declaration (raising before any I/O if the caller asked for something too
    fresh), a session is opened pinned to exactly that instant, and only then
    does the registered computation exist in a position to read anything.

    Args:
        name: registered feature name.
        compute_date: calendar date to compute for — the rebalance date. Must
            be a ``date``; a ``datetime`` is refused rather than truncated.
        security_ids: securities to value, in the order the results are wanted.
            Duplicates are refused: a repeated identifier would be silently
            double-weighted by every consumer downstream.
        requested_as_of: optional override, tz-aware UTC, which may only move
            the instant **backwards** (see :func:`resolve_as_of`).
        registry: registry to look the feature up in; defaults to
            :func:`backend.features.registry.default_registry`.

    Returns:
        A :class:`FeatureVector` carrying the values, their units, and the
        instant they were read at.

    Raises:
        UnknownFeatureError: if ``name`` is not registered (before any I/O).
        AvailabilityLagViolationError: if ``requested_as_of`` is fresher than
            the declaration permits (before any I/O).
        FeatureComputeError: if ``compute_date`` is a ``datetime``,
            ``requested_as_of`` is naive or non-UTC, or ``security_ids``
            contains duplicates.
        MalformedFeatureVectorError: if the registered computation returns
            something that is not one ``float64`` per requested security.
        AsOfTimestampError: from :func:`backend.db.as_of` if the resolved
            instant is in the future — which happens when a feature is computed
            for a date whose lag window has not yet elapsed. That is a real
            condition, not a bug in this layer, and it is left to the as-of
            layer to name.
    """
    source = default_registry() if registry is None else registry
    spec = source.spec(name)
    # Reaching into the registry's private accessor is the point: the callable
    # is not public precisely so that this function is the only thing that can
    # invoke it, and only after pinning the session (see the registry docstring).
    computation = source._computation(name)
    securities = tuple(security_ids)
    if len(set(securities)) != len(securities):
        duplicates = sorted({sid for sid in securities if securities.count(sid) > 1})
        msg = (
            f"security_ids for feature {name!r} contains duplicate identifier(s) "
            f"{duplicates}. Each security must appear once: a repeat would be "
            f"double-counted by every cross-sectional statistic downstream."
        )
        raise FeatureComputeError(msg)

    as_of_ts = resolve_as_of(spec, compute_date, requested_as_of=requested_as_of)
    request = FeatureComputeRequest(
        feature=spec.name,
        compute_date=compute_date,
        as_of=as_of_ts,
        security_ids=securities,
    )
    async with as_of(as_of_ts) as session:
        raw = await computation(session, request)
    return FeatureVector(
        feature=spec.name,
        units=spec.units,
        compute_date=compute_date,
        as_of=as_of_ts,
        security_ids=securities,
        values=_validated_values(raw, spec=spec, expected=len(securities)),
    )


def _validated_values(raw: object, *, spec: FeatureSpec, expected: int) -> FloatArray:
    """Coerce and check a computation's output, or raise.

    A computation is trusted for its arithmetic and not for its shape: a
    length mismatch means the values no longer describe the securities they are
    paired with, and every number downstream would be attached to the wrong
    company while looking entirely healthy.

    Args:
        raw: whatever the computation returned.
        spec: the feature's declaration, for the error message.
        expected: number of securities requested (count).

    Returns:
        A fresh one-dimensional ``float64`` copy of the values. Copied so a
        computation that hands back its own working array cannot mutate a
        vector already returned to a caller.

    Raises:
        MalformedFeatureVectorError: if the values are not numeric, not
            one-dimensional, or not of length ``expected``.
    """
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MalformedFeatureVectorError(
            feature=spec.name, detail=f"values are not numeric ({exc})"
        ) from exc
    if values.ndim != 1:
        raise MalformedFeatureVectorError(
            feature=spec.name,
            detail=f"expected a one-dimensional array, got shape {values.shape}",
        )
    if int(values.shape[0]) != expected:
        raise MalformedFeatureVectorError(
            feature=spec.name,
            detail=(
                f"expected {expected} value(s) for {expected} requested "
                f"security/securities, got {int(values.shape[0])}"
            ),
        )
    return np.array(values, dtype=np.float64, copy=True)
