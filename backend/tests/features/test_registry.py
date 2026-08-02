"""P5.1: the thirty-feature cap, and that no ordering of registrations beats it.

Directive §5 Phase 5 fixes the feature count at thirty, LLM features from Phase
7 included. The cap is not a style rule: feature count is the lever that most
reliably turns a believable cross-sectional ranker into an overfitted one, and
it is the only complexity measure a reviewer can check at a glance.

A cap is only worth the line of code that states it if it cannot be reached
around, so this file attacks it from four sides.

**The oracle property.** A model set is kept alongside the registry and random
sequences of ``register`` / ``unregister`` / duplicate-``register`` are replayed
against both. After *every* operation the registry must agree with the model on
its contents and its ordering, and ``len(registry) <= MAX_FEATURES`` must hold —
not merely at the end, when a failed registration might have been undone, but at
every intermediate state. The model also pins *which* error fires, so a registry
that raised ``FeatureCapExceededError`` for a duplicate name (or the reverse)
fails here even though its contents would look right.

**Order independence.** Registration order must not change what the registry
contains, because imports happen in whatever order Python finds convenient and
a config hash over the catalog has to be reproducible (I2). The same set of
declarations, registered in any permutation, must yield identical ``names()``
and ``specs()``.

**Saturation from every direction.** Thirty-one distinct features offered in a
random order always leave exactly thirty registered, and the one refused is
whichever happened to arrive thirty-first — the cap doing its job, not an
accident of the sequence.

**The surface.** The cap is checked in ``register``, so the claim depends on
``register`` being the only thing that grows the registry. That is asserted
structurally: the constructor accepts no override, ``max_features`` is
read-only, and the public method set is enumerated and pinned, so a future
``update``/``__setitem__``/bulk loader added without a cap check fails this file
rather than shipping.

The decorator is exercised against the same claims, because it is the entry
point P5.3 and P7 will actually use — a cap enforced in ``register`` but not
reached by ``@feature`` would be a cap in name only.
"""

from __future__ import annotations

import datetime as dt
import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.features.errors import (
    DuplicateFeatureError,
    FeatureCapExceededError,
    UnknownFeatureError,
)
from backend.features.registry import FeatureRegistry, default_registry, feature
from backend.features.spec import MAX_FEATURES, FeatureSpec

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def unused_computation(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """A computation that is registered but never run.

    Every test in this file is about the catalog, not about values. It returns
    NaN rather than a number so that a test which accidentally *did* run it
    could not mistake the result for a computed feature (I3).
    """
    del session
    return np.full(len(request.security_ids), np.nan, dtype=np.float64)


def make_spec(name: str, *, lag_days: int = 45) -> FeatureSpec:
    """Build a well-formed declaration named ``name``."""
    return FeatureSpec(
        name=name,
        definition=f"Declaration {name}, registered to exercise the catalog.",
        units="dimensionless ratio",
        availability_lag=dt.timedelta(days=lag_days),
        source_tables=frozenset({"price_bar"}),
    )


def numbered(index: int) -> str:
    """Return a legal snake_case feature name for an integer."""
    return f"probe_feature_{index}"


def filled_registry(count: int = MAX_FEATURES) -> FeatureRegistry:
    """Return a registry holding ``count`` distinct declarations."""
    registry = FeatureRegistry()
    for index in range(count):
        registry.register(make_spec(numbered(index)), unused_computation)
    return registry


_indices = st.integers(min_value=0, max_value=MAX_FEATURES + 9)
_actions = st.lists(
    st.tuples(st.sampled_from(["register", "unregister"]), _indices),
    min_size=1,
    max_size=150,
)
"""Random operation sequences over a name space slightly larger than the cap.

Ten spare names is deliberate: enough that the cap is reachable and that
duplicates and unknown-name removals occur often, few enough that saturation
happens early in most sequences rather than at the very end.
"""


# ---------------------------------------------------------------------------
# The catalog behaves like a catalog
# ---------------------------------------------------------------------------


def test_an_empty_registry_reports_full_capacity() -> None:
    registry = FeatureRegistry()
    assert len(registry) == 0
    assert registry.names() == ()
    assert registry.specs() == ()
    assert registry.remaining_capacity() == MAX_FEATURES
    assert registry.max_features == MAX_FEATURES


def test_a_registered_feature_is_retrievable_by_name() -> None:
    registry = FeatureRegistry()
    spec = make_spec("book_to_price")
    registry.register(spec, unused_computation)
    assert registry.spec("book_to_price") is spec
    assert "book_to_price" in registry
    assert len(registry) == 1
    assert registry.remaining_capacity() == MAX_FEATURES - 1


def test_enumeration_is_sorted_by_name_not_by_insertion() -> None:
    """Import order must not reach the catalog, or a config hash over it drifts (I2)."""
    registry = FeatureRegistry()
    for name in ["size", "accruals", "momentum_12_1", "book_to_price"]:
        registry.register(make_spec(name), unused_computation)
    assert registry.names() == ("accruals", "book_to_price", "momentum_12_1", "size")
    assert [spec.name for spec in registry.specs()] == list(registry.names())
    assert [spec.name for spec in registry] == list(registry.names())


def test_looking_up_an_unregistered_name_names_what_is_registered() -> None:
    registry = filled_registry(3)
    with pytest.raises(UnknownFeatureError) as excinfo:
        registry.spec("no_such_feature")
    assert excinfo.value.name == "no_such_feature"
    assert excinfo.value.known == registry.names()
    assert "probe_feature_0" in str(excinfo.value)


def test_an_empty_registry_says_so_when_a_lookup_fails() -> None:
    with pytest.raises(UnknownFeatureError, match="registry is empty"):
        FeatureRegistry().spec("momentum_12_1")


def test_the_repr_states_occupancy_against_the_cap() -> None:
    assert repr(filled_registry(4)) == f"FeatureRegistry(4/{MAX_FEATURES} features)"


# ---------------------------------------------------------------------------
# Duplicates raise rather than replace
# ---------------------------------------------------------------------------


def test_registering_a_taken_name_raises_and_leaves_the_original_in_place() -> None:
    """Silent replacement would make the surviving definition depend on import order."""
    registry = FeatureRegistry()
    original = make_spec("momentum_12_1", lag_days=1)
    registry.register(original, unused_computation)

    replacement = make_spec("momentum_12_1", lag_days=400)
    with pytest.raises(DuplicateFeatureError) as excinfo:
        registry.register(replacement, unused_computation)

    assert excinfo.value.name == "momentum_12_1"
    assert "unregister" in str(excinfo.value)
    assert "DECISIONS.md" in str(excinfo.value)
    assert registry.spec("momentum_12_1") is original
    assert registry.spec("momentum_12_1").availability_lag == dt.timedelta(days=1)
    assert len(registry) == 1


def test_a_duplicate_is_refused_even_when_the_declarations_are_identical() -> None:
    registry = FeatureRegistry()
    registry.register(make_spec("size"), unused_computation)
    with pytest.raises(DuplicateFeatureError):
        registry.register(make_spec("size"), unused_computation)


def test_a_duplicate_at_full_capacity_is_reported_as_a_duplicate() -> None:
    """The name is the reason the registration failed; the cap is not the story here."""
    registry = filled_registry()
    with pytest.raises(DuplicateFeatureError):
        registry.register(make_spec(numbered(0)), unused_computation)
    assert len(registry) == MAX_FEATURES


# ---------------------------------------------------------------------------
# The cap
# ---------------------------------------------------------------------------


def test_the_thirty_first_feature_is_refused() -> None:
    registry = filled_registry()
    assert len(registry) == MAX_FEATURES
    assert registry.remaining_capacity() == 0

    with pytest.raises(FeatureCapExceededError) as excinfo:
        registry.register(make_spec("one_too_many"), unused_computation)

    error = excinfo.value
    assert error.name == "one_too_many"
    assert error.cap == MAX_FEATURES
    assert error.registered == registry.names()
    assert len(error.registered) == MAX_FEATURES


def test_the_cap_refusal_carries_enough_context_to_act_on() -> None:
    """The message must say what was refused, how full the registry is, and the way out."""
    with pytest.raises(FeatureCapExceededError) as excinfo:
        filled_registry().register(make_spec("one_too_many"), unused_computation)
    message = str(excinfo.value)
    assert "one_too_many" in message
    assert f"{MAX_FEATURES}" in message
    assert "unregister" in message
    assert "DECISIONS.md" in message
    assert numbered(0) in message


def test_a_refused_registration_changes_nothing() -> None:
    """No half-registration: the name must not appear, and the count must not move."""
    registry = filled_registry()
    before = registry.names()
    with pytest.raises(FeatureCapExceededError):
        registry.register(make_spec("one_too_many"), unused_computation)
    assert registry.names() == before
    assert "one_too_many" not in registry
    assert len(registry) == MAX_FEATURES
    with pytest.raises(UnknownFeatureError):
        registry.spec("one_too_many")


def test_repeated_attempts_at_the_thirty_first_keep_failing() -> None:
    """The refusal is not a one-shot latch that a retry loop could wear down."""
    registry = filled_registry()
    for attempt in range(5):
        with pytest.raises(FeatureCapExceededError):
            registry.register(make_spec(f"one_too_many_{attempt}"), unused_computation)
    assert len(registry) == MAX_FEATURES


def test_the_sanctioned_swap_is_unregister_then_register() -> None:
    """The count never exceeds thirty at any point in the exchange."""
    registry = filled_registry()
    removed = registry.unregister(numbered(0))
    assert removed.name == numbered(0)
    assert len(registry) == MAX_FEATURES - 1
    assert registry.remaining_capacity() == 1

    registry.register(make_spec("its_replacement"), unused_computation)
    assert len(registry) == MAX_FEATURES
    assert "its_replacement" in registry
    assert numbered(0) not in registry


def test_unregistering_a_name_that_is_not_there_raises() -> None:
    registry = filled_registry(2)
    with pytest.raises(UnknownFeatureError) as excinfo:
        registry.unregister("never_registered")
    assert excinfo.value.known == registry.names()
    assert len(registry) == 2


def test_a_removed_feature_takes_its_computation_with_it() -> None:
    """A stale callable left behind would be reachable under a name later reused."""
    registry = FeatureRegistry()
    registry.register(make_spec("size"), unused_computation)
    registry.unregister("size")
    with pytest.raises(UnknownFeatureError):
        registry._computation("size")  # the accessor under test is module-private


def test_remaining_capacity_tracks_the_occupancy() -> None:
    registry = FeatureRegistry()
    for count in range(MAX_FEATURES):
        assert registry.remaining_capacity() == MAX_FEATURES - count
        registry.register(make_spec(numbered(count)), unused_computation)
    assert registry.remaining_capacity() == 0


# ---------------------------------------------------------------------------
# The cap has no override
# ---------------------------------------------------------------------------


def test_the_constructor_accepts_no_capacity_override() -> None:
    """There is no roomier registry to reach for when the exception fires."""
    with pytest.raises(TypeError):
        FeatureRegistry(max_features=40)  # type: ignore[call-arg]
    assert inspect.signature(FeatureRegistry.__init__).parameters.keys() == {"self"}


def test_max_features_is_read_only() -> None:
    registry = FeatureRegistry()
    with pytest.raises(AttributeError):
        registry.max_features = 40  # type: ignore[misc]
    assert registry.max_features == MAX_FEATURES


def test_the_registry_has_no_per_instance_state_beyond_its_two_dictionaries() -> None:
    """``__slots__`` means a cap override cannot even be monkey-patched onto an instance."""
    assert FeatureRegistry.__slots__ == ("_computations", "_specs")
    with pytest.raises(AttributeError):
        FeatureRegistry().max_features_override = 40  # type: ignore[attr-defined]


def test_register_is_the_only_public_method_that_grows_the_registry() -> None:
    """Pinned so a future bulk loader or ``__setitem__`` cannot skip the cap check unnoticed.

    If this fails, the new method is not necessarily wrong — but it has to be
    read against the cap and then added here deliberately.
    """
    public = {
        name
        for name in dir(FeatureRegistry)
        if not name.startswith("_") and callable(getattr(FeatureRegistry, name, None))
    }
    assert public == {"register", "unregister", "spec", "names", "specs", "remaining_capacity"}
    assert "__setitem__" not in vars(FeatureRegistry)
    assert "update" not in vars(FeatureRegistry)


# ---------------------------------------------------------------------------
# The decorator is held to the same rules
# ---------------------------------------------------------------------------


def test_the_decorator_registers_and_returns_the_function_unchanged() -> None:
    """Applied by hand so the undecorated function is still in scope to compare against.

    A factor module keeps calling its own function after decoration, and P5.3's
    unit tests exercise the arithmetic directly against a session they pinned
    themselves; a decorator that returned a wrapper would break both silently.
    """
    registry = FeatureRegistry()
    spec = make_spec("momentum_12_1")

    async def momentum_12_1(
        session: AsyncSession, request: FeatureComputeRequest
    ) -> FloatArray:  # pragma: no cover — registered, never run here
        return await unused_computation(session, request)

    returned = feature(spec, registry=registry)(momentum_12_1)

    assert returned is momentum_12_1
    assert registry.spec("momentum_12_1") is spec
    assert registry._computation("momentum_12_1") is momentum_12_1


def test_the_decorator_hits_the_same_cap() -> None:
    """The entry point P5.3 and P7 actually use must not be a way around the count."""
    registry = filled_registry()
    with pytest.raises(FeatureCapExceededError):

        @feature(make_spec("one_too_many"), registry=registry)
        async def one_too_many(
            session: AsyncSession, request: FeatureComputeRequest
        ) -> FloatArray:  # pragma: no cover — registration raises before this is callable
            return await unused_computation(session, request)

    assert len(registry) == MAX_FEATURES


def test_the_decorator_refuses_duplicates_too() -> None:
    registry = FeatureRegistry()
    registry.register(make_spec("size"), unused_computation)
    with pytest.raises(DuplicateFeatureError):

        @feature(make_spec("size"), registry=registry)
        async def size(
            session: AsyncSession, request: FeatureComputeRequest
        ) -> FloatArray:  # pragma: no cover — registration raises first
            return await unused_computation(session, request)


def test_the_decorator_defaults_to_the_process_wide_registry() -> None:
    """The cap is about *that* instance: P5.3 factors and P7 LLM features share its thirty slots."""
    registry = default_registry()
    assert default_registry() is registry
    name = "probe_default_registry_feature"
    assert name not in registry

    try:

        @feature(make_spec(name))
        async def probe_default_registry_feature(
            session: AsyncSession, request: FeatureComputeRequest
        ) -> FloatArray:  # pragma: no cover — registered, never run here
            return await unused_computation(session, request)

        assert name in registry
        assert registry.spec(name).units == "dimensionless ratio"
    finally:
        registry.unregister(name)
    assert name not in registry


# ---------------------------------------------------------------------------
# Properties: no ordering of operations beats the cap
# ---------------------------------------------------------------------------


@given(actions=_actions)
@hypothesis_settings(max_examples=400, deadline=None)
def test_the_registry_matches_an_oracle_after_every_operation(
    actions: list[tuple[str, int]],
) -> None:
    """Contents, ordering, occupancy and *which* error fires, checked at every step.

    The model is a plain ``set`` of names plus the two rules the registry
    claims: a taken name is a duplicate, and a full registry refuses. Any
    divergence — a registration that should have been refused and was not, a
    cap error raised where a duplicate error belongs, contents that drifted
    from insertion order — fails here.
    """
    registry = FeatureRegistry()
    model: set[str] = set()

    for action, index in actions:
        name = numbered(index)
        if action == "register":
            if name in model:
                with pytest.raises(DuplicateFeatureError):
                    registry.register(make_spec(name), unused_computation)
            elif len(model) >= MAX_FEATURES:
                with pytest.raises(FeatureCapExceededError):
                    registry.register(make_spec(name), unused_computation)
            else:
                registry.register(make_spec(name), unused_computation)
                model.add(name)
        elif name in model:
            registry.unregister(name)
            model.remove(name)
        else:
            with pytest.raises(UnknownFeatureError):
                registry.unregister(name)

        assert len(registry) <= MAX_FEATURES
        assert len(registry) == len(model)
        assert registry.names() == tuple(sorted(model))
        assert registry.remaining_capacity() == MAX_FEATURES - len(model)


@given(
    names=st.lists(
        _indices, min_size=MAX_FEATURES + 1, max_size=MAX_FEATURES + 10, unique=True
    ).flatmap(st.permutations)
)
@hypothesis_settings(max_examples=300, deadline=None)
def test_more_than_thirty_distinct_features_always_leave_exactly_thirty(
    names: list[int],
) -> None:
    """Whatever the order, exactly thirty land and every later one is refused.

    The refusals are also checked to have happened *only* at saturation: a
    registry that refused early for some unrelated reason would leave fewer
    than thirty registered and fail the count.
    """
    registry = FeatureRegistry()
    accepted: list[str] = []
    refused: list[tuple[str, str, int]] = []

    for index in names:
        name = numbered(index)
        try:
            registry.register(make_spec(name), unused_computation)
        except FeatureCapExceededError as error:
            refused.append((name, error.name, len(registry)))
        else:
            accepted.append(name)
            assert len(registry) == len(accepted)

    # Every refusal happened at saturation and named the feature it turned away.
    assert all(occupancy == MAX_FEATURES for _, _, occupancy in refused)
    assert [name for name, _, _ in refused] == [named for _, named, _ in refused]
    assert len(registry) == MAX_FEATURES
    assert accepted == [numbered(index) for index in names[:MAX_FEATURES]]
    assert [name for name, _, _ in refused] == [numbered(i) for i in names[MAX_FEATURES:]]
    assert registry.names() == tuple(sorted(accepted))


@given(
    indices=st.lists(_indices, min_size=0, max_size=MAX_FEATURES, unique=True).flatmap(
        lambda chosen: st.tuples(st.just(chosen), st.permutations(chosen))
    )
)
@hypothesis_settings(max_examples=300, deadline=None)
def test_registration_order_does_not_change_the_catalog(
    indices: tuple[list[int], list[int]],
) -> None:
    """Same set of declarations, any two orders, identical catalog (I2)."""
    first_order, second_order = indices
    first = FeatureRegistry()
    for index in first_order:
        first.register(make_spec(numbered(index)), unused_computation)
    second = FeatureRegistry()
    for index in second_order:
        second.register(make_spec(numbered(index)), unused_computation)

    assert first.names() == second.names()
    assert first.specs() == second.specs()
    assert len(first) == len(second)


@given(actions=_actions)
@hypothesis_settings(max_examples=300, deadline=None)
def test_a_swap_is_the_only_way_a_new_name_enters_a_full_registry(
    actions: list[tuple[str, int]],
) -> None:
    """Whenever occupancy is at the cap, the next successful registration is preceded by a removal.

    Stated as the transition the directive actually permits: registrations may
    only take occupancy from ``k < 30`` to ``k + 1``, never from ``30`` to
    ``31``, whatever sequence led there.
    """
    registry = FeatureRegistry()
    for action, index in actions:
        name = numbered(index)
        before = len(registry)
        try:
            if action == "register":
                registry.register(make_spec(name), unused_computation)
                assert before < MAX_FEATURES
                assert len(registry) == before + 1
            else:
                registry.unregister(name)
                assert len(registry) == before - 1
        except (DuplicateFeatureError, FeatureCapExceededError, UnknownFeatureError):
            assert len(registry) == before
        assert len(registry) <= MAX_FEATURES
