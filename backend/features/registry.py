"""The feature registry and its hard 30-feature cap (P5.1).

Directive §5 Phase 5: *"Hard cap: 30 features total, including LLM features
from Phase 7. Adding a 31st requires removing one and logging the swap in
``DECISIONS.md``."*

The cap is not a style preference, and it is not enforced here because 30 is a
magic number that happens to work. It is enforced because feature count is the
single lever that most reliably converts a mediocre cross-sectional ranker into
an overfitted one, and because it is the only complexity measure a reviewer can
check in one second. A model with 30 candidate columns and a purged CV can be
believed; the same model with 200 columns cannot, whatever its information
coefficient says.

So the cap is enforced *structurally*, in three ways that together make
exceeding it something you have to set out to do rather than something you
drift into:

1. :data:`~backend.features.spec.MAX_FEATURES` is compared against inside
   :meth:`FeatureRegistry.register`, which is the **only** mutation point that
   adds a name — there is no ``update``, no ``__setitem__``, no bulk loader,
   and the backing dictionaries are private;
2. the limit is a property of the class, not a constructor argument. There is
   no ``FeatureRegistry(max_features=...)`` to reach for when the exception
   fires, so raising the cap requires editing the module and appears in a diff;
3. removal is explicit. :meth:`FeatureRegistry.unregister` is the sanctioned
   path to a 31st feature — swap one out, and (per the directive) log the swap
   in ``DECISIONS.md``. The count never exceeds 30 at any point in that
   sequence.

--------------------------------------------------------------------------
Determinism
--------------------------------------------------------------------------

Registration order must not change what the registry *contains*, because
imports happen in whatever order Python finds convenient and a config hash
computed from the catalog has to be reproducible (I2). Two guarantees:

- **duplicate names raise** (:class:`~backend.features.errors.DuplicateFeatureError`)
  rather than replacing, so the surviving definition of a name can never depend
  on import order;
- **enumeration is sorted by name**, never by insertion, so
  :meth:`FeatureRegistry.names` and :meth:`FeatureRegistry.specs` return the
  same sequence for the same set of registrations however they were made.

What is legitimately order-dependent is *which* registration is refused once
the registry is full — the 31st fails, whichever it happens to be. That is the
cap doing its job, and the invariant that matters (never more than 30) holds
after every operation regardless.

--------------------------------------------------------------------------
The computation is deliberately not public
--------------------------------------------------------------------------

A registered feature is a declaration plus a callable, but the callable is
reachable only through the module-private
:meth:`FeatureRegistry._computation`. Handing it out would let a caller invoke
a feature computation against any session it liked, which is exactly the
lookahead this package exists to prevent: the availability lag is enforced by
the fact that :func:`backend.features.compute.compute_feature` pins the session
before the computation ever sees one. The public surface therefore exposes the
declarations (for the catalog, §6.4) and no way to run one unpinned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from backend.features.errors import (
    DuplicateFeatureError,
    FeatureCapExceededError,
    UnknownFeatureError,
)
from backend.features.spec import MAX_FEATURES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from backend.features.compute import FeatureComputation
    from backend.features.spec import FeatureSpec

__all__ = [
    "FeatureRegistry",
    "default_registry",
    "feature",
]


class FeatureRegistry:
    """A collection of at most :data:`~backend.features.spec.MAX_FEATURES` features.

    Not a singleton by nature — tests and analyses build their own — but the
    process-wide catalog the platform actually runs is
    :func:`default_registry`, and it is that instance the cap is about.

    The registry holds declarations and their computations; it does not compute
    anything and never touches the database. Running a feature goes through
    :func:`backend.features.compute.compute_feature`, which is what turns a
    declared availability lag into a pinned as-of session.
    """

    __slots__ = ("_computations", "_specs")

    def __init__(self) -> None:
        """Create an empty registry.

        Takes no arguments **on purpose**: the 30-feature cap is a property of
        the class and there is no per-instance override, so the exception
        cannot be worked around by constructing a roomier registry.
        """
        self._specs: dict[str, FeatureSpec] = {}
        self._computations: dict[str, FeatureComputation] = {}

    @property
    def max_features(self) -> int:
        """The hard cap on registered features (count). Read-only, class-wide."""
        return MAX_FEATURES

    def register(self, spec: FeatureSpec, computation: FeatureComputation) -> None:
        """Add a feature to the registry.

        The only operation that grows the registry, and therefore the only
        place the cap has to be checked.

        Args:
            spec: the declaration. Already validated by its own constructor;
                by the time it arrives here it is well-formed.
            computation: an async callable
                ``(session, request) -> NDArray[float64]`` returning one value
                per requested security, in the requested order, in
                ``spec.units``. It is called only by
                :func:`backend.features.compute.compute_feature`, with a
                session already pinned to the instant the declaration permits.

        Raises:
            DuplicateFeatureError: if the name is already registered.
                Registration is never replacement — see the module docstring on
                determinism.
            FeatureCapExceededError: if the registry already holds
                :data:`~backend.features.spec.MAX_FEATURES` features. Remove one
                with :meth:`unregister` and log the swap in ``DECISIONS.md``.
        """
        if spec.name in self._specs:
            raise DuplicateFeatureError(spec.name)
        if len(self._specs) >= MAX_FEATURES:
            raise FeatureCapExceededError(spec.name, cap=MAX_FEATURES, registered=self.names())
        self._specs[spec.name] = spec
        self._computations[spec.name] = computation

    def unregister(self, name: str) -> FeatureSpec:
        """Remove a feature, returning its declaration.

        The sanctioned half of a swap: the directive allows a 31st feature only
        in exchange for an existing one, with the trade recorded in
        ``DECISIONS.md``. Removing first and adding second keeps the count at or
        below the cap at every instant.

        Args:
            name: the registered feature name.

        Returns:
            The removed declaration, so the caller can quote it in the
            decision log.

        Raises:
            UnknownFeatureError: if no such feature is registered.
        """
        if name not in self._specs:
            raise UnknownFeatureError(name, self.names())
        del self._computations[name]
        return self._specs.pop(name)

    def spec(self, name: str) -> FeatureSpec:
        """Return one feature's declaration.

        Args:
            name: the registered feature name.

        Returns:
            The declaration, including its units and availability lag.

        Raises:
            UnknownFeatureError: if no such feature is registered.
        """
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownFeatureError(name, self.names()) from None

    def names(self) -> tuple[str, ...]:
        """Return every registered name, sorted.

        Returns:
            Names in lexicographic order — never insertion order, so the result
            depends on the set of registrations and not on how they were
            sequenced.
        """
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[FeatureSpec, ...]:
        """Return every declaration, sorted by name.

        Returns:
            The catalog, in the order :meth:`names` reports. This is what the
            Features page (§6.4) renders and what a config hash is taken over.
        """
        return tuple(self._specs[name] for name in self.names())

    def remaining_capacity(self) -> int:
        """Return how many more features may be registered (count).

        Returns:
            ``MAX_FEATURES - len(self)``, never negative. Useful to the
            dashboard, which shows the budget rather than waiting for the
            exception.
        """
        return MAX_FEATURES - len(self._specs)

    def _computation(self, name: str) -> FeatureComputation:
        """Return the callable registered for ``name`` (module-private).

        Deliberately not public: see the module docstring. The only sanctioned
        caller is :func:`backend.features.compute.compute_feature`, which pins
        the session to the instant the declaration permits before invoking it.

        Args:
            name: the registered feature name.

        Returns:
            The registered computation.

        Raises:
            UnknownFeatureError: if no such feature is registered.
        """
        try:
            return self._computations[name]
        except KeyError:
            raise UnknownFeatureError(name, self.names()) from None

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is a registered feature name."""
        return name in self._specs

    def __len__(self) -> int:
        """Return the number of registered features (count)."""
        return len(self._specs)

    def __iter__(self) -> Iterator[FeatureSpec]:
        """Iterate declarations in sorted-name order."""
        return iter(self.specs())

    def __repr__(self) -> str:
        """Return a summary naming the occupancy against the cap."""
        return f"FeatureRegistry({len(self._specs)}/{MAX_FEATURES} features)"


_DEFAULT_REGISTRY: Final = FeatureRegistry()
"""The process-wide catalog. Populated by the factor modules on import (P5.3)."""


def default_registry() -> FeatureRegistry:
    """Return the process-wide feature registry.

    This is the instance the 30-feature cap is about: the set of features the
    platform will actually offer a model. Phase 5's factor modules and Phase
    7's LLM features register into it at import time, and they compete for the
    same 30 slots.

    Returns:
        The singleton :class:`FeatureRegistry`.
    """
    return _DEFAULT_REGISTRY


def feature(
    spec: FeatureSpec, *, registry: FeatureRegistry | None = None
) -> Callable[[FeatureComputation], FeatureComputation]:
    """Decorate an async computation to register it under ``spec``.

    The intended extension point for P5.3's baseline factors and P7's LLM
    features::

        @feature(FeatureSpec(name="momentum_12_1", ...))
        async def momentum_12_1(session, request):
            ...

    The function is returned unchanged, so it remains an ordinary callable for
    unit tests that want to exercise the arithmetic directly against a session
    they have pinned themselves.

    Args:
        spec: the declaration to register the function under.
        registry: registry to register into; defaults to
            :func:`default_registry`.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        DuplicateFeatureError: if the name is already registered.
        FeatureCapExceededError: if the registry is full.
    """
    target = default_registry() if registry is None else registry

    def decorate(computation: FeatureComputation) -> FeatureComputation:
        target.register(spec, computation)
        return computation

    return decorate
