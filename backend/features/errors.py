"""Failure taxonomy for the feature library (directive §5 Phase 5).

Two of this package's rules are *budget* rules rather than correctness rules,
and budgets are only real when breaking them is an exception rather than a
warning:

**The 30-feature cap.** Directive §5 Phase 5 fixes the feature count at 30
including the Phase 7 LLM features, because feature proliferation is the main
route to overfitting a cross-sectional ranker: every additional column is
another chance for the model to fit noise, and the count is the one thing a
reviewer can check at a glance. A 31st feature therefore raises
:class:`FeatureCapExceededError`. Swapping one out (``unregister`` then
``register``) is the sanctioned path and the directive requires the swap to be
logged in ``DECISIONS.md``.

**The availability lag.** A feature declares how long after an event its value
is actually knowable — a quarterly fundamental is not knowable on the quarter
end, it is knowable when the filing is accepted. Invariant I1 says no
computation may see a fact whose ``knowledge_time`` is later than the instant
it is entitled to. At the feature layer that entitlement is
``midnight_utc(compute_date) - availability_lag``, and asking for anything
fresher raises :class:`AvailabilityLagViolationError` *before* a session is
opened, so the refusal costs no I/O and cannot be half-completed.

Every error carries the numbers that produced it, so the message alone is
enough to diagnose the cause without re-running under a debugger.
"""

from __future__ import annotations

import datetime as dt

__all__ = [
    "AvailabilityLagViolationError",
    "DuplicateFeatureError",
    "FeatureCapExceededError",
    "FeatureComputeError",
    "FeatureError",
    "FeatureSpecError",
    "MalformedFeatureVectorError",
    "UnknownFeatureError",
]


class FeatureError(Exception):
    """Base class for every failure raised by :mod:`backend.features`."""


class FeatureSpecError(FeatureError):
    """Raised when a feature declaration cannot describe a well-posed feature.

    Examples: a name that is not ``snake_case`` (the name is a stable
    identifier used in configs, model artifacts and the dashboard, so it is
    pinned to one spelling), an empty ``units`` string, a negative availability
    lag, or an empty ``source_tables`` set — a feature that reads nothing has
    nothing to be point-in-time about.
    """


class DuplicateFeatureError(FeatureError):
    """Raised when a name already present in the registry is registered again.

    Silent replacement is the failure mode this prevents. Two modules
    registering ``momentum_12_1`` with different definitions would leave the
    winner decided by import order, which is exactly the kind of
    order-dependent result that cannot be reproduced from a config hash (I2).

    Attributes:
        name: the name that was already taken.
    """

    def __init__(self, name: str) -> None:
        """Build the error from the offending feature name.

        Args:
            name: the feature name that is already registered.
        """
        self.name = name
        super().__init__(
            f"feature {name!r} is already registered. Registration is not "
            f"replacement: two declarations under one name would make the "
            f"surviving definition depend on import order. Call unregister("
            f"{name!r}) first if the swap is intended, and log it in DECISIONS.md."
        )


class UnknownFeatureError(FeatureError):
    """Raised when a name is looked up and is not registered.

    Attributes:
        name: the name that was requested.
        known: the registered names at the time of the lookup, sorted.
    """

    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        """Build the error from the missing name and the registry's contents.

        Args:
            name: the feature name that was requested.
            known: registered names, sorted, for the message.
        """
        self.name = name
        self.known = known
        rendered = ", ".join(known) if known else "<registry is empty>"
        super().__init__(f"no feature named {name!r} is registered. Registered: {rendered}")


class FeatureCapExceededError(FeatureError):
    """Raised when registering a feature would exceed the hard cap.

    Directive §5 Phase 5: *"Hard cap: 30 features total, including LLM features
    from Phase 7. Adding a 31st requires removing one and logging the swap in
    ``DECISIONS.md``."* The cap is a property of the class, not a constructor
    argument, so there is no ordinary way to raise it.

    Attributes:
        name: the feature whose registration was refused.
        cap: the maximum number of features (count).
        registered: names registered at the time of the refusal, sorted.
    """

    def __init__(self, name: str, *, cap: int, registered: tuple[str, ...]) -> None:
        """Build the error from the refused name and the full registry.

        Args:
            name: the feature whose registration was refused.
            cap: the maximum number of features (count).
            registered: registered names, sorted, for the message.
        """
        self.name = name
        self.cap = cap
        self.registered = registered
        super().__init__(
            f"refusing to register {name!r}: the registry already holds {len(registered)} "
            f"of a maximum {cap} features. The cap exists because feature "
            f"proliferation is the main route to overfitting a cross-sectional "
            f"ranker, and it is not configurable. Remove a feature with "
            f"unregister(...) and log the swap in DECISIONS.md. Registered: "
            f"{', '.join(registered)}"
        )


class FeatureComputeError(FeatureError):
    """Base class for failures on the compute path.

    Covers arguments that cannot describe a well-posed computation: a
    ``compute_date`` that is a ``datetime`` rather than a ``date``, a requested
    as-of that is naive or not UTC, duplicate security identifiers.
    """


class AvailabilityLagViolationError(FeatureComputeError):
    """Raised when a caller asks to compute a feature from data it may not see.

    The feature's declaration is the authority on what is knowable: the compute
    path pins its session at ``midnight_utc(compute_date) - availability_lag``
    and a caller may only move that instant *backwards* (reading older data is
    always I1-safe and is what a historical replay does). Moving it forwards
    would produce a feature value that could not have existed on the compute
    date, a model that learns from it, and a backtest that looks excellent and
    is worthless.

    Attributes:
        feature: name of the feature whose declaration was violated.
        compute_date: the date the feature was being computed for.
        availability_lag: the declared lag.
        permitted_as_of: the freshest instant the declaration allows
            (tz-aware UTC).
        requested_as_of: the instant the caller asked for (tz-aware UTC).
    """

    def __init__(
        self,
        *,
        feature: str,
        compute_date: dt.date,
        availability_lag: dt.timedelta,
        permitted_as_of: dt.datetime,
        requested_as_of: dt.datetime,
    ) -> None:
        """Build the error from the declaration and the request that broke it.

        Args:
            feature: name of the feature.
            compute_date: date the feature was being computed for.
            availability_lag: the declared availability lag.
            permitted_as_of: freshest instant the declaration allows.
            requested_as_of: instant the caller asked for.
        """
        self.feature = feature
        self.compute_date = compute_date
        self.availability_lag = availability_lag
        self.permitted_as_of = permitted_as_of
        self.requested_as_of = requested_as_of
        excess = requested_as_of - permitted_as_of
        super().__init__(
            f"feature {feature!r} declares an availability lag of {availability_lag} "
            f"and may therefore read the store as of "
            f"{permitted_as_of.isoformat()} at the freshest when computing for "
            f"{compute_date.isoformat()}; the caller requested "
            f"{requested_as_of.isoformat()}, which is {excess} too fresh. "
            f"A requested as-of may only move backwards. Refusing before any "
            f"session is opened (I1)."
        )


class MalformedFeatureVectorError(FeatureComputeError):
    """Raised when a registered computation returns something unusable.

    A computation must return one ``float64`` value per requested security, in
    the requested order, in the units its declaration states. A wrong length is
    the dangerous case: it means the values are misaligned with the securities
    they describe, and a silently truncated or broadcast array would attribute
    one company's fundamentals to another for the rest of the pipeline.

    Attributes:
        feature: name of the offending feature.
        detail: what was wrong, in words.
    """

    def __init__(self, *, feature: str, detail: str) -> None:
        """Build the error from the feature name and the defect.

        Args:
            feature: name of the offending feature.
            detail: what was wrong with the returned vector.
        """
        self.feature = feature
        self.detail = detail
        super().__init__(
            f"feature {feature!r} returned a malformed vector: {detail}. A "
            f"computation must return a one-dimensional float64 array with one "
            f"value per requested security, in the requested order; use NaN for "
            f"'not available' rather than a filler value (I3)."
        )
