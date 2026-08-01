"""P7.1 integration: the provider registry against real PostgreSQL.

What is proved here and nowhere else:

- an API key round-trips **encrypted**: the bytes actually sitting in the
  column are not the plaintext, and only the KEK turns them back;
- the masked display never contains the full key, and no response body from the
  mounted router does either;
- changing a task's assignment writes a **new version** and leaves the old one
  readable, with the database refusing to update or delete a past version;
- every credential and assignment change appears in ``config_change_event``
  with actor and correlation id — including deletion, which is the change
  easiest to forget to record;
- rotation preserves the *history* (previous and new masked renderings, version
  counter either side) while the superseded ciphertext stops existing;
- the probe refuses when no key is configured, against a real empty table.

Migration 0009 and the ORM models are written independently — this repository
hand-writes its migrations rather than autogenerating them — so the schema
agreement is asserted rather than assumed.

**The probe's live path is still unexercised.** B4 leaves no provider key
anywhere in this repository or its CI, so the only probe behaviour tested here
is its refusal. Nothing below stubs a provider success and calls it coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import sqlalchemy as sa
import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Table
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.api.app import create_app
from backend.api.routes.providers import router as providers_router
from backend.api.security.settings import ApiSecuritySettings
from backend.core.config import Settings, get_settings
from backend.core.crypto import decrypt_secret, generate_key_encryption_key
from backend.db import create_admin_engine
from backend.db.audit import config_history
from backend.db.models import ExtractionModelAssignment, LlmProviderCredential
from backend.extraction.providers import (
    Provider,
    ProviderKeyAlreadyConfiguredError,
    ProviderKeyNotConfiguredError,
    add_provider_key,
    assign_task_model,
    assignment_versions,
    configured_providers,
    current_assignment,
    delete_provider_key,
    get_provider_key,
    list_current_assignments,
    list_provider_keys,
    load_provider_secret,
    probe_provider,
    rotate_provider_key,
)
from backend.extraction.providers.assignments import AUDIT_SCOPE as ASSIGNMENT_SCOPE
from backend.extraction.providers.registry import (
    AUDIT_SCOPE as PROVIDER_SCOPE,
)
from backend.extraction.providers.registry import (
    FIELD_KEY_VERSION,
    FIELD_MASKED_KEY,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

_ACTOR = "operator"
_TASK = "risk_factor_delta"
_KEY_ONE = "sk-ant-api03-" + "a1b2c3d4" * 8 + "4f2a"
_KEY_TWO = "sk-ant-api03-" + "e5f6a7b8" * 8 + "9c0d"


@pytest.fixture(autouse=True)
def _kek(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give the whole module a real, throwaway KEK.

    Generated per run rather than hard-coded, so a fixed key cannot drift into
    a deployment through a copy-paste, and so the encryption under test is the
    real Fernet path rather than a fixture-shaped special case.
    """
    monkeypatch.setenv("SECRETS_KEK", generate_key_encryption_key())
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
async def _clean_registry() -> AsyncIterator[None]:
    """Truncate the registry and audit tables after every test in this module.

    The shared integration fixture truncates the fact tables and the ingestion
    log; these three arrive in migrations 0007 and 0009 and hold no foreign key
    for a CASCADE to follow. TRUNCATE is deliberately left unblocked by the
    append-only trigger (0009, following 0003/0004/0007) precisely so the
    sanctioned reset path exists. An ordinary admin engine suffices: the Core
    guard only refuses SQL naming a *bitemporal* fact table, and none of these
    is one.
    """
    yield
    engine = create_admin_engine()
    try:
        async with engine.begin() as connection:
            await connection.execute(
                sa.text(
                    "TRUNCATE TABLE llm_provider_credential, extraction_model_assignment, "
                    "config_change_event RESTART IDENTITY"
                )
            )
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
def _no_ambient_request() -> Iterator[None]:
    """Start every test with no bound request id, so correlation defaults are explicit."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


async def _stored_ciphertext(provider: Provider) -> str:
    """Read the raw ciphertext column with a Core select — the actual stored bytes."""
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                sa.text(
                    "SELECT ciphertext FROM llm_provider_credential WHERE provider = :provider"
                ),
                {"provider": provider.value},
            )
            return cast("str", result.scalar_one())
    finally:
        await engine.dispose()


async def _raw_execute(statement: str) -> None:
    """Run one textual statement on an admin connection."""
    engine = create_admin_engine()
    try:
        async with engine.begin() as connection:
            await connection.execute(sa.text(statement))
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------
# Encryption at rest (§7, I5)
# --------------------------------------------------------------------------


async def test_a_stored_key_is_ciphertext_in_the_column_not_the_plaintext() -> None:
    """Asserted against the bytes actually in the database, not against a return value.

    Reading the column directly matters: a registry that returned a masked view
    while storing the plaintext would pass every test written against its
    public functions, and would still be the exact failure I5 forbids.
    """
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)

    stored = await _stored_ciphertext(Provider.ANTHROPIC)
    assert stored != _KEY_ONE
    assert _KEY_ONE not in stored
    # Not merely different — different because it is a Fernet token that the
    # environment KEK, and only the environment KEK, turns back into the key.
    assert stored.startswith("gAAAAA")
    assert decrypt_secret(stored) == _KEY_ONE
    assert await load_provider_secret(Provider.ANTHROPIC) == _KEY_ONE


async def test_encrypting_the_same_key_twice_produces_different_ciphertext() -> None:
    """Fernet carries its own IV, so a column value never fingerprints the key it holds."""
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)
    first = await _stored_ciphertext(Provider.ANTHROPIC)
    await rotate_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)
    second = await _stored_ciphertext(Provider.ANTHROPIC)

    assert first != second
    assert decrypt_secret(first) == decrypt_secret(second) == _KEY_ONE


async def test_a_ciphertext_written_under_another_kek_does_not_silently_decrypt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database dump alone decrypts nothing: the KEK comes from the environment (§7)."""
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)

    monkeypatch.setenv("SECRETS_KEK", generate_key_encryption_key())
    get_settings.cache_clear()
    with pytest.raises(Exception, match="failed authentication"):
        await load_provider_secret(Provider.ANTHROPIC)


async def test_the_masked_display_never_contains_the_full_key() -> None:
    """The only rendering a reader gets is strictly shorter than what it stands for (§6.5)."""
    view = await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)

    assert _KEY_ONE not in view.masked_key
    assert view.masked_key.endswith(_KEY_ONE[-4:])
    assert len(view.masked_key) < len(_KEY_ONE)
    stored = await get_provider_key(Provider.ANTHROPIC)
    assert stored.masked_key == view.masked_key
    assert _KEY_ONE not in repr(stored)


# --------------------------------------------------------------------------
# Add, rotate, delete — and their audit record
# --------------------------------------------------------------------------


async def test_adding_a_key_twice_is_refused_so_a_rotation_is_recorded_as_one() -> None:
    """Add and rotate mean different things to an operator and must stay distinguishable."""
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)

    with pytest.raises(ProviderKeyAlreadyConfiguredError):
        await add_provider_key(Provider.ANTHROPIC, _KEY_TWO, actor=_ACTOR)
    assert await load_provider_secret(Provider.ANTHROPIC) == _KEY_ONE


async def test_rotating_replaces_the_key_bumps_the_version_and_destroys_the_old_ciphertext() -> (
    None
):
    """Rotation is not an append: the superseded token stops existing on purpose.

    Keeping it would leave every retired key decryptable for as long as the KEK
    lives — turning a KEK compromise from "every key in use" into "every key
    ever used" — while rotation is normally performed because the old key
    should stop existing. The history that *is* kept is asserted in the next
    test.
    """
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)
    old_ciphertext = await _stored_ciphertext(Provider.ANTHROPIC)

    view = await rotate_provider_key(Provider.ANTHROPIC, _KEY_TWO, actor="second-operator")

    assert view.key_version == 2
    assert view.rotated_at >= view.created_at
    new_ciphertext = await _stored_ciphertext(Provider.ANTHROPIC)
    assert new_ciphertext != old_ciphertext
    assert decrypt_secret(new_ciphertext) == _KEY_TWO
    assert await load_provider_secret(Provider.ANTHROPIC) == _KEY_TWO

    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                sa.text("SELECT ciphertext FROM llm_provider_credential")
            )
            stored = [row[0] for row in rows]
    finally:
        await engine.dispose()
    assert old_ciphertext not in stored


async def test_rotation_preserves_the_history_as_masked_values_in_the_audit_log() -> None:
    """The key before a rotation is answerable as ``sk-...4f2a`` — all §6.5 permits."""
    first = await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)
    second = await rotate_provider_key(Provider.ANTHROPIC, _KEY_TWO, actor="second-operator")

    events = await config_history(
        scope=PROVIDER_SCOPE, target=Provider.ANTHROPIC.value, field=FIELD_MASKED_KEY
    )
    assert [event.new_value for event in events] == [second.masked_key, first.masked_key]
    rotation = events[0]
    assert rotation.previous_value == first.masked_key
    assert rotation.is_initial is False
    assert rotation.actor == "second-operator"
    assert events[1].is_initial is True
    assert events[1].actor == _ACTOR

    versions = await config_history(
        scope=PROVIDER_SCOPE, target=Provider.ANTHROPIC.value, field=FIELD_KEY_VERSION
    )
    assert [event.new_value for event in versions] == [2, 1]

    # The record is safe to keep forever precisely because nothing in it is
    # recoverable: no event carries a key or a ciphertext.
    for event in (*events, *versions):
        assert _KEY_ONE not in str(event.new_value) + str(event.previous_value)
        assert _KEY_TWO not in str(event.new_value) + str(event.previous_value)


async def test_deleting_a_key_removes_it_and_is_recorded_as_a_change() -> None:
    """A deletion is the change easiest to leave unlogged; the log must show it happened."""
    view = await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)

    await delete_provider_key(Provider.ANTHROPIC, actor="third-operator")

    assert await configured_providers() == frozenset()
    with pytest.raises(ProviderKeyNotConfiguredError):
        await load_provider_secret(Provider.ANTHROPIC)

    events = await config_history(
        scope=PROVIDER_SCOPE, target=Provider.ANTHROPIC.value, field=FIELD_MASKED_KEY
    )
    deletion = events[0]
    assert deletion.new_value is None
    assert deletion.previous_value == view.masked_key
    assert deletion.actor == "third-operator"
    versions = await config_history(
        scope=PROVIDER_SCOPE, target=Provider.ANTHROPIC.value, field=FIELD_KEY_VERSION
    )
    assert versions[0].new_value is None


async def test_deleting_a_key_that_was_never_configured_is_refused() -> None:
    """A no-op deletion would write a misleading event; refuse instead."""
    with pytest.raises(ProviderKeyNotConfiguredError):
        await delete_provider_key(Provider.OPENAI, actor=_ACTOR)
    assert await config_history(scope=PROVIDER_SCOPE) == ()


async def test_credential_changes_carry_the_request_correlation_id() -> None:
    """§6.11 wants who and when; D-003's request id is how a change joins its request."""
    structlog.contextvars.bind_contextvars(request_id="req-provider-1")
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)

    events = await config_history(scope=PROVIDER_SCOPE, correlation_id="req-provider-1")
    assert {event.field for event in events} == {FIELD_MASKED_KEY, FIELD_KEY_VERSION}


async def test_a_provider_outside_the_vocabulary_cannot_be_inserted_at_all() -> None:
    """The CHECK is real enforcement, not a comment: raw SQL is refused too.

    Matched on the constraint name rather than on "some error happened": an
    unqualified ``pytest.raises`` here would also pass against a database where
    the table did not exist at all, which is the shape of false green a
    migration mistake produces.
    """
    with pytest.raises(
        (IntegrityError, DBAPIError), match="ck_llm_provider_credential_provider_known"
    ):
        await _raw_execute(
            "INSERT INTO llm_provider_credential "
            "(provider, ciphertext, masked_display, key_version) "
            "VALUES ('gemini', 'x', 'y', 1)"
        )
    with pytest.raises(
        (IntegrityError, DBAPIError), match="ck_extraction_model_assignment_provider_known"
    ):
        await _raw_execute(
            "INSERT INTO extraction_model_assignment "
            "(task, version, provider, model, temperature, max_tokens, timeout_s, actor) "
            "VALUES ('t', 1, 'gemini', 'm', 0, 1, 1, 'operator')"
        )


async def test_listing_reports_only_configured_providers_and_only_their_masks() -> None:
    """The list read path returns the same maskable-only view as the single read."""
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)
    await add_provider_key(Provider.OPENAI, _KEY_TWO, actor=_ACTOR)

    views = await list_provider_keys()
    assert [view.provider for view in views] == [Provider.ANTHROPIC, Provider.OPENAI]
    rendered = repr(views)
    assert _KEY_ONE not in rendered
    assert _KEY_TWO not in rendered


# --------------------------------------------------------------------------
# Per-task assignment: a change is a version (§6.5)
# --------------------------------------------------------------------------


async def test_changing_an_assignment_creates_a_version_and_leaves_the_old_one_readable() -> None:
    """§6.5's rule, asserted directly: the previous configuration is still there afterwards."""
    first = await assign_task_model(
        _TASK, Provider.ANTHROPIC, "cheap-model-a", max_tokens=512, actor=_ACTOR
    )
    second = await assign_task_model(
        _TASK,
        Provider.OPENAI,
        "cheap-model-b",
        temperature=0.2,
        max_tokens=2048,
        timeout_s=30.0,
        actor="second-operator",
    )

    assert (first.version, second.version) == (1, 2)
    assert (await current_assignment(_TASK)).version == 2

    history = await assignment_versions(_TASK)
    assert [item.version for item in history] == [2, 1]
    superseded = history[1]
    assert superseded.provider is Provider.ANTHROPIC
    assert superseded.model == "cheap-model-a"
    assert superseded.max_tokens == 512
    assert superseded.actor == _ACTOR
    assert second.temperature == pytest.approx(0.2)
    assert second.timeout_s == pytest.approx(30.0)


async def test_a_new_assignment_defaults_temperature_to_zero() -> None:
    """§5-P7 all the way through to the stored row, not just to the function signature."""
    assignment = await assign_task_model(_TASK, Provider.ANTHROPIC, "cheap-model", actor=_ACTOR)
    assert assignment.temperature == 0.0

    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            stored = await connection.execute(
                sa.text("SELECT temperature FROM extraction_model_assignment WHERE task = :task"),
                {"task": _TASK},
            )
            assert float(stored.scalar_one()) == 0.0
    finally:
        await engine.dispose()


async def test_a_superseded_assignment_version_cannot_be_edited_or_removed() -> None:
    """The database refuses it, so "the old one remains readable" is enforced, not promised."""
    await assign_task_model(_TASK, Provider.ANTHROPIC, "cheap-model-a", actor=_ACTOR)
    await assign_task_model(_TASK, Provider.OPENAI, "cheap-model-b", actor=_ACTOR)

    with pytest.raises((IntegrityError, DBAPIError), match="append-only"):
        await _raw_execute(
            "UPDATE extraction_model_assignment SET model = 'rewritten' WHERE version = 1"
        )
    with pytest.raises((IntegrityError, DBAPIError), match="append-only"):
        await _raw_execute("DELETE FROM extraction_model_assignment WHERE version = 1")

    history = await assignment_versions(_TASK)
    assert [item.model for item in history] == ["cheap-model-b", "cheap-model-a"]


async def test_an_assignment_change_writes_an_audit_event_with_actor_and_correlation_id() -> None:
    """Provider and assignment changes are configuration changes; the log must show them."""
    structlog.contextvars.bind_contextvars(request_id="req-assign-1")
    await assign_task_model(
        _TASK, Provider.ANTHROPIC, "cheap-model-a", temperature=0.0, actor=_ACTOR
    )
    await assign_task_model(
        _TASK, Provider.OPENAI, "cheap-model-b", temperature=0.5, actor="second-operator"
    )

    events = await config_history(scope=ASSIGNMENT_SCOPE, target=_TASK, field="model")
    assert [event.new_value for event in events] == ["cheap-model-b", "cheap-model-a"]
    assert events[0].previous_value == "cheap-model-a"
    assert events[0].actor == "second-operator"
    assert events[0].correlation_id == "req-assign-1"

    versions = await config_history(scope=ASSIGNMENT_SCOPE, target=_TASK, field="version")
    assert [event.new_value for event in versions] == [2, 1]
    temperatures = await config_history(scope=ASSIGNMENT_SCOPE, target=_TASK, field="temperature")
    assert [event.new_value for event in temperatures] == [0.5, 0.0]


async def test_reassigning_identical_settings_still_records_a_version() -> None:
    """The operator pressed the button; a history that disagrees with their memory is worse."""
    await assign_task_model(_TASK, Provider.ANTHROPIC, "cheap-model", actor=_ACTOR)
    await assign_task_model(_TASK, Provider.ANTHROPIC, "cheap-model", actor=_ACTOR)

    assert [item.version for item in await assignment_versions(_TASK)] == [2, 1]


async def test_tasks_are_versioned_independently_of_one_another() -> None:
    """A task's version counter is its own; one task's change does not renumber another."""
    await assign_task_model(_TASK, Provider.ANTHROPIC, "a", actor=_ACTOR)
    await assign_task_model(_TASK, Provider.ANTHROPIC, "b", actor=_ACTOR)
    await assign_task_model("guidance_tone", Provider.OPENAI, "c", actor=_ACTOR)

    current = {item.task: item.version for item in await list_current_assignments()}
    assert current == {_TASK: 2, "guidance_tone": 1}


async def test_a_task_can_be_assigned_to_a_provider_with_no_key_configured() -> None:
    """Configuration order is the operator's to choose; the refusal belongs at call time."""
    assignment = await assign_task_model(_TASK, Provider.OPENAI, "cheap-model", actor=_ACTOR)

    assert assignment.provider is Provider.OPENAI
    assert await configured_providers() == frozenset()
    with pytest.raises(ProviderKeyNotConfiguredError):
        await probe_provider(Provider.OPENAI)


async def test_deleting_a_credential_leaves_assignments_naming_it_untouched() -> None:
    """Cascading would silently rewrite the operator's pipeline configuration."""
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)
    await assign_task_model(_TASK, Provider.ANTHROPIC, "cheap-model", actor=_ACTOR)

    await delete_provider_key(Provider.ANTHROPIC, actor=_ACTOR)

    assignment = await current_assignment(_TASK)
    assert assignment.provider is Provider.ANTHROPIC
    assert assignment.version == 1


# --------------------------------------------------------------------------
# The probe, against a real empty registry
# --------------------------------------------------------------------------


async def test_the_probe_refuses_when_no_key_is_configured() -> None:
    """No key means a clear refusal, never a hopeful "ok" — a probe reports what it measured."""
    with pytest.raises(ProviderKeyNotConfiguredError, match="no API key is configured"):
        await probe_provider(Provider.ANTHROPIC)


# --------------------------------------------------------------------------
# Schema agreement: hand-written migration versus hand-written ORM
# --------------------------------------------------------------------------


def _table(model: type) -> Table:
    """Return the mapped Table of a declarative model (typed for mypy)."""
    return cast("Table", cast("Any", model).__table__)


@pytest.mark.parametrize(
    "model", [LlmProviderCredential, ExtractionModelAssignment], ids=lambda m: m.__tablename__
)
async def test_the_migration_and_the_orm_agree_on_the_columns(model: type) -> None:
    """Written independently, so agreement is asserted rather than assumed."""
    table = _table(model)
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                sa.text(
                    "SELECT column_name, is_nullable FROM information_schema.columns "
                    "WHERE table_name = :name"
                ),
                {"name": table.name},
            )
            actual = {name: nullable == "YES" for name, nullable in rows}
    finally:
        await engine.dispose()
    assert actual == {column.name: column.nullable for column in table.columns}


async def test_neither_registry_table_is_in_the_bitemporal_registry() -> None:
    """An operator credential has no market knowability; a knowledge_time here is invented."""
    from backend.db.bitemporal import bitemporal_tables

    names = {table.name for table in bitemporal_tables()}
    assert "llm_provider_credential" not in names
    assert "extraction_model_assignment" not in names


# --------------------------------------------------------------------------
# The HTTP surface, end to end, against a real stored key
# --------------------------------------------------------------------------


def _app() -> FastAPI:
    """Build the real application with the provider router mounted, rate limiting off."""
    app = create_app(
        settings=Settings(environment="test"),
        security=ApiSecuritySettings(rate_limit_enabled=False),
    )
    app.include_router(providers_router, prefix="/api")
    return app


async def test_no_readable_endpoint_returns_the_stored_key_in_any_form() -> None:
    """The behavioural half of "no reveal path": a real key, every readable route, no leak.

    The structural half — that no such endpoint *exists* — is asserted in
    ``backend/tests/extraction/test_providers_api.py``. This half catches a leak
    through a field nobody thought to name suspiciously.
    """
    await add_provider_key(Provider.ANTHROPIC, _KEY_ONE, actor=_ACTOR)
    await assign_task_model(_TASK, Provider.ANTHROPIC, "cheap-model", actor=_ACTOR)
    ciphertext = await _stored_ciphertext(Provider.ANTHROPIC)

    paths = [
        "/api/providers",
        "/api/extraction-tasks",
        f"/api/extraction-tasks/{_TASK}/assignment",
        f"/api/extraction-tasks/{_TASK}/assignment/versions",
    ]
    app = _app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        for path in paths:
            response = await client.get(path)
            assert response.status_code == 200, path
            body = response.text
            assert _KEY_ONE not in body, path
            assert ciphertext not in body, path

        listing = (await client.get("/api/providers")).json()["providers"]
    by_provider = {entry["provider"]: entry for entry in listing}
    assert by_provider["anthropic"]["configured"] is True
    assert by_provider["anthropic"]["masked_key"].endswith(_KEY_ONE[-4:])
    assert by_provider["openai"]["configured"] is False
    assert by_provider["openai"]["masked_key"] is None
