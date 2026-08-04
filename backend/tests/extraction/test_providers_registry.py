"""P7.1 unit tests: vocabulary, validation, defaults, and the probe (no database).

What is proved here rather than in the integration suite: everything that is a
property of the *code* — the provider vocabulary agreeing across its spellings
(the enum, the ORM, and the migration that last widened it, with the earlier
revisions pinned to the set they shipped with), the refusals that happen before
any I/O, temperature defaulting to
0, and the exact wire shape of a probe request. Anything that needs a real
PostgreSQL round trip (ciphertext at rest, versioning, audit events, the
append-only trigger) lives in ``backend/tests/integration/test_providers_db.py``.

**The probe's live path is not exercised anywhere.** B4 is unresolved, so there
is no provider key in this repository or its CI. The tests below drive the
outcome mapping through a stub transport and assert the request construction
directly; none of them claims that a real provider was contacted, and a stubbed
200 is treated as evidence about *this module's* branching only (I3).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

import backend.db.models  # noqa: F401 — populates the metadata the drift scan reads
from backend.core.crypto import mask_secret
from backend.db.base import Base
from backend.db.models import PROVIDER_NAMES_SQL
from backend.extraction.providers import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    MAX_TEMPERATURE,
    ProbeOutcome,
    ProbeResult,
    Provider,
    ProviderKeyView,
    provider_from_name,
)
from backend.extraction.providers import assignments as assignments_module
from backend.extraction.providers import probe as probe_module
from backend.extraction.providers import registry as registry_module
from backend.extraction.providers.probe import (
    ANTHROPIC_VERSION_HEADER_VALUE,
    PROBE_ENDPOINTS,
    build_probe_request,
    probe_provider,
)
from backend.extraction.providers.registry import ProviderKeyNotConfiguredError

if TYPE_CHECKING:
    from collections.abc import Callable

_ANTHROPIC_KEY = "sk-ant-api03-" + "a1b2c3d4" * 8 + "4f2a"
_OPENAI_KEY = "sk-proj-" + "z9y8x7w6" * 8 + "1e3d"


def _sql_vocabulary(sql: str) -> set[str]:
    """Return the provider names quoted inside a SQL ``IN`` list fragment."""
    return set(re.findall(r"'([^']+)'", sql))


# --------------------------------------------------------------------------
# Vocabulary: one current set of providers, plus the history that produced it
# --------------------------------------------------------------------------


def test_orm_check_vocabulary_matches_the_provider_enum() -> None:
    """The database CHECK and the Python enum name exactly the same providers.

    ``backend.db`` must not import ``backend.extraction``, so the vocabulary is
    duplicated rather than shared. This is the test that makes the duplication
    safe: drift fails here instead of reaching a database, where it would show
    up as an IntegrityError on an operator's first save.
    """
    assert _sql_vocabulary(PROVIDER_NAMES_SQL) == {member.value for member in Provider}


def test_the_latest_widening_declares_the_same_provider_vocabulary_as_the_orm() -> None:
    """The head vocabulary revision and the ORM name exactly the same providers.

    This is the assertion that used to sit on 0009, and it moved because the
    vocabulary grew. A migration records what the schema was at its point in the
    chain; when a provider is added, earlier revisions keep the set they shipped
    with and a new revision widens it. Pointing this test at 0009 would have
    forced an edit to an applied migration — making a database migrated last week
    and one created today disagree about what 0009 did, with nothing in the chain
    saying so.

    Consequently *this* test must be repointed at the newest widening whenever a
    provider is added. That is deliberate: it is one line, and it is the line that
    makes the addition visible in the diff.
    """
    migration = importlib.import_module(
        "backend.db.migrations.versions.0018_provider_vocabulary_groq"
    )
    assert _sql_vocabulary(migration._PROVIDER_NAMES_SQL) == _sql_vocabulary(PROVIDER_NAMES_SQL)


def test_the_earlier_revisions_keep_the_vocabulary_they_shipped_with() -> None:
    """0009 and 0013 are pinned to history, so nobody 'fixes' them into agreement.

    Frozen literals rather than a comparison against the enum: the whole point is
    that these must *not* track the current set. If a future provider is added by
    editing these revisions instead of adding a widening, this fails.
    """
    registry = importlib.import_module("backend.db.migrations.versions.0009_provider_registry")
    ledger = importlib.import_module("backend.db.migrations.versions.0013_llm_spend_ledger")
    shipped = {"anthropic", "openai"}
    assert _sql_vocabulary(registry._PROVIDER_NAMES_SQL) == shipped
    assert _sql_vocabulary(ledger._PROVIDER_NAMES_SQL) == shipped


def test_the_widening_covers_every_table_that_constrains_a_provider() -> None:
    """A provider-bearing table left out of the widening keeps the narrow CHECK.

    The failure that motivates this is asymmetric and quiet: credentials would
    save, assignments would save, and the *spend ledger* would reject the first
    governed call with an IntegrityError — at call time, in a path that already
    has a refusal taxonomy, where it would read like a governor decision rather
    than a missed migration.
    """
    migration = importlib.import_module(
        "backend.db.migrations.versions.0018_provider_vocabulary_groq"
    )
    constrained = {
        name
        for name, table in Base.metadata.tables.items()
        if any(constraint.name == f"ck_{name}_provider_known" for constraint in table.constraints)
    }
    assert constrained, "no mapped table declares provider_known; this test has gone vacuous"
    assert set(migration._PROVIDER_TABLES) == constrained


def test_the_widening_names_constraints_the_way_the_database_does() -> None:
    """The ALTER must use the expanded name, not the bare one 0009 was written with.

    0009 and 0013 declare the CHECK as a bare ``provider_known`` and let the
    metadata naming convention expand it while the table is being created. An
    ``ALTER TABLE ... DROP CONSTRAINT`` gets no such expansion, so a widening
    written in 0009's style fails at run time with "constraint does not exist" —
    on a migration, against a real database, which is the worst place to find it
    and the one place this repository's test suite cannot reach without Docker.

    Asserted against the ORM metadata so the two cannot drift apart silently.
    """
    migration = importlib.import_module(
        "backend.db.migrations.versions.0018_provider_vocabulary_groq"
    )
    for table_name in migration._PROVIDER_TABLES:
        table = Base.metadata.tables[table_name]
        declared = {
            constraint.name for constraint in table.constraints if isinstance(constraint.name, str)
        }
        assert migration._constraint_name(table_name) in declared, (
            f"{table_name}: the widening would ALTER a constraint name the schema "
            f"does not use; metadata declares {sorted(declared)}"
        )


def test_migration_0009_follows_0008_and_is_append_only_where_it_claims_to_be() -> None:
    """The revision chain is linear and the assignment table carries its trigger."""
    migration = importlib.import_module("backend.db.migrations.versions.0009_provider_registry")
    assert migration.revision == "0009"
    assert migration.down_revision == "0008"
    source = Path(str(migration.__file__)).read_text()
    assert "BEFORE UPDATE OR DELETE ON extraction_model_assignment" in source
    # The credential table must NOT get one: rotation overwrites the superseded
    # ciphertext on purpose, so an append-only trigger there would defeat the
    # security property the design rests on.
    assert "BEFORE UPDATE OR DELETE ON llm_provider_credential" not in source


def test_every_provider_has_a_probe_endpoint() -> None:
    """A provider with no endpoint could be stored but never tested; refuse that state."""
    assert set(PROBE_ENDPOINTS) == set(Provider)


@pytest.mark.parametrize("name", ["Anthropic", " anthropic", "anthropic ", "gemini", ""])
def test_provider_from_name_refuses_anything_but_an_exact_match(name: str) -> None:
    """Case folding and trimming are refused, so one provider cannot get two spellings."""
    with pytest.raises(ValueError, match="unknown provider"):
        provider_from_name(name)


def test_provider_from_name_accepts_the_exact_spelling() -> None:
    """The happy path exists and returns the enum member, not a string."""
    assert provider_from_name("anthropic") is Provider.ANTHROPIC


# --------------------------------------------------------------------------
# Nothing that leaves this package can carry a key
# --------------------------------------------------------------------------


def test_provider_key_view_has_no_field_that_could_hold_a_key() -> None:
    """The only credential view has no key field, so no serialization of it can leak one."""
    fields = set(ProviderKeyView.__dataclass_fields__)
    assert fields == {"provider", "masked_key", "key_version", "created_at", "rotated_at"}
    assert not {"api_key", "ciphertext", "secret", "plaintext"} & fields


def test_probe_result_has_no_field_that_could_hold_a_key() -> None:
    """Same property for the probe report, which is built while the key is in scope."""
    fields = set(ProbeResult.__dataclass_fields__)
    assert not {"api_key", "ciphertext", "headers", "secret", "request"} & fields


@pytest.mark.parametrize("key", [_ANTHROPIC_KEY, _OPENAI_KEY])
def test_masked_rendering_is_strictly_shorter_and_reveals_only_a_tail(key: str) -> None:
    """The stored display can never be inflated back into the key it stands for."""
    masked = mask_secret(key)
    assert key not in masked
    assert len(masked) < len(key)
    assert masked.endswith(key[-4:])
    # Everything between the published vendor prefix and the last four
    # characters is absent, not obscured: there is nothing to brute-force.
    assert key[10:-4] not in masked


def test_registry_module_exposes_no_reveal_helper() -> None:
    """There is exactly one decryption entry point and no unmasking counterpart."""
    exported = set(registry_module.__all__)
    assert "load_provider_secret" in exported
    assert not {name for name in exported if re.search(r"reveal|unmask|plaintext", name)}


# --------------------------------------------------------------------------
# Refusals that happen before any I/O
# --------------------------------------------------------------------------


@pytest.mark.parametrize("api_key", ["", "   ", "sk-1234"])
async def test_adding_an_implausible_key_is_refused_without_echoing_it(api_key: str) -> None:
    """Validation precedes encryption and the database, and the message never quotes the input."""
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011 — message asserted below
        await registry_module.add_provider_key(Provider.ANTHROPIC, api_key, actor="operator")
    if api_key.strip():
        assert api_key not in str(excinfo.value)


@pytest.mark.parametrize("task", ["", " tone", "tone "])
async def test_assigning_a_whitespace_padded_task_is_refused(task: str) -> None:
    """Whitespace is rejected rather than trimmed, so one task cannot get two spellings."""
    with pytest.raises(ValueError, match="task must"):
        await assignments_module.assign_task_model(
            task, Provider.ANTHROPIC, "some-model", actor="operator"
        )


@pytest.mark.parametrize(
    ("temperature", "max_tokens", "timeout_s"),
    [
        (-0.1, 1024, 60.0),
        (MAX_TEMPERATURE + 0.1, 1024, 60.0),
        (0.0, 0, 60.0),
        (0.0, 1024, 0.0),
        (0.0, 1024, -1.0),
    ],
)
async def test_out_of_range_settings_are_refused_before_the_database(
    temperature: float, max_tokens: int, timeout_s: float
) -> None:
    """The caller gets a message naming the setting, not an IntegrityError quoting a CHECK."""
    with pytest.raises(ValueError, match=r"temperature|max_tokens|timeout_s"):
        await assignments_module.assign_task_model(
            "tone",
            Provider.ANTHROPIC,
            "some-model",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            actor="operator",
        )


# --------------------------------------------------------------------------
# Temperature 0 (§5-P7)
# --------------------------------------------------------------------------


def test_default_temperature_constant_is_zero() -> None:
    """§5-P7: temperature 0 everywhere. The constant is the single source of that default."""
    assert DEFAULT_TEMPERATURE == 0.0


def test_assign_task_model_defaults_temperature_to_zero() -> None:
    """The callable's own default is 0 — not merely the constant it happens to reference."""
    signature = inspect.signature(assignments_module.assign_task_model)
    assert signature.parameters["temperature"].default == 0.0
    assert signature.parameters["max_tokens"].default == DEFAULT_MAX_TOKENS
    assert signature.parameters["timeout_s"].default == DEFAULT_TIMEOUT_S


def test_no_module_in_the_package_supplies_a_non_zero_temperature_default() -> None:
    """A second default hiding in another module would silently defeat §5-P7.

    Scans the package's source for any assignment or keyword default named
    ``temperature`` and asserts every literal one is zero. A future author who
    adds ``temperature=0.7`` anywhere in this package fails here.
    """
    package_root = Path(str(registry_module.__file__)).parent
    offenders: list[str] = []
    for path in sorted(package_root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg == "temperature":
                value = node.value
                if isinstance(value, ast.Constant) and value.value != 0:
                    offenders.append(f"{path.name}:{node.value.lineno}")
            if isinstance(node, ast.AnnAssign | ast.Assign):
                targets = [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                if "DEFAULT_TEMPERATURE" in names:
                    assert isinstance(node.value, ast.Constant)
                    assert node.value.value == 0.0
    assert not offenders, f"non-zero temperature literals found: {offenders}"


# --------------------------------------------------------------------------
# Probe: request construction (pure, so it is testable without B4)
# --------------------------------------------------------------------------


def test_anthropic_probe_request_is_the_documented_authenticated_listing_call() -> None:
    """GET /v1/models with x-api-key and the required dated version header."""
    request = build_probe_request(Provider.ANTHROPIC, _ANTHROPIC_KEY)
    assert request.method == "GET"
    assert request.url == "https://api.anthropic.com/v1/models"
    assert request.headers["x-api-key"] == _ANTHROPIC_KEY
    assert request.headers["anthropic-version"] == ANTHROPIC_VERSION_HEADER_VALUE


def test_openai_probe_request_is_the_documented_authenticated_listing_call() -> None:
    """GET /v1/models with a bearer credential."""
    request = build_probe_request(Provider.OPENAI, _OPENAI_KEY)
    assert request.method == "GET"
    assert request.url == "https://api.openai.com/v1/models"
    assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}"


@pytest.mark.parametrize(
    ("provider", "key"), [(Provider.ANTHROPIC, _ANTHROPIC_KEY), (Provider.OPENAI, _OPENAI_KEY)]
)
def test_the_key_travels_in_a_header_and_never_in_the_url(provider: Provider, key: str) -> None:
    """A URL reaches proxy logs and error messages; a credential in one is a leak by design."""
    request = build_probe_request(provider, key)
    assert key not in request.url
    assert sum(key in value for value in request.headers.values()) == 1


@pytest.mark.parametrize("provider", list(Provider))
def test_probe_request_headers_cannot_be_mutated_by_a_caller(provider: Provider) -> None:
    """The header mapping is read-only, so no caller can redirect a credential."""
    request = build_probe_request(provider, _ANTHROPIC_KEY)
    with pytest.raises(TypeError):
        request.headers["x-api-key"] = "other"  # type: ignore[index]


def test_probe_endpoints_mapping_is_read_only() -> None:
    """Module state cannot be rewritten to point a probe — and a key — at another host."""
    with pytest.raises(TypeError):
        PROBE_ENDPOINTS[Provider.OPENAI] = "https://example.invalid"  # type: ignore[index]


# --------------------------------------------------------------------------
# Probe: refusal without a key, and the outcome mapping under a stub transport
# --------------------------------------------------------------------------


def _stub_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Wrap a handler as an httpx transport standing in for a provider."""
    return httpx.MockTransport(handler)


async def test_probe_refuses_without_a_key_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key means a refusal, not a result — and not a request either.

    A probe that reported *any* outcome for an unconfigured provider would be
    reporting a measurement it did not make (I3). The transport records whether
    it was ever entered, so this asserts the refusal happens before the network,
    not merely that the returned value looked like a failure.
    """
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        sent.append(request)
        return httpx.Response(200, json={})

    async def refuse(_provider: Provider) -> str:
        raise ProviderKeyNotConfiguredError("no API key is configured for provider 'anthropic'")

    monkeypatch.setattr(probe_module, "load_provider_secret", refuse)
    with pytest.raises(ProviderKeyNotConfiguredError, match="no API key is configured"):
        await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert sent == []


@pytest.fixture
def _stub_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the probe's credential lookup succeed without a database."""

    async def load(_provider: Provider) -> str:
        return _ANTHROPIC_KEY

    monkeypatch.setattr(probe_module, "load_provider_secret", load)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (200, ProbeOutcome.OK),
        (204, ProbeOutcome.OK),
        (401, ProbeOutcome.AUTHENTICATION_FAILED),
        (403, ProbeOutcome.AUTHENTICATION_FAILED),
        (429, ProbeOutcome.RATE_LIMITED),
        (400, ProbeOutcome.PROVIDER_ERROR),
        (500, ProbeOutcome.PROVIDER_ERROR),
    ],
)
@pytest.mark.usefixtures("_stub_key")
async def test_http_status_maps_to_the_outcome_the_operator_needs(
    status_code: int, expected: ProbeOutcome
) -> None:
    """Each status says something different about the key; the mapping must not blur them.

    Exercises this module's branching against a stub. It is **not** evidence
    that a real provider returns these statuses — B4 leaves that unverified.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="provider says something")

    result = await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert result.outcome is expected
    assert result.http_status == status_code
    assert result.latency_ms >= 0.0
    assert result.provider is Provider.ANTHROPIC


@pytest.mark.usefixtures("_stub_key")
async def test_a_timeout_is_reported_as_a_timeout_not_as_a_bad_key() -> None:
    """A provider that never answered has not rejected the credential; say so."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    result = await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert result.outcome is ProbeOutcome.TIMED_OUT
    assert result.http_status is None
    assert "neither accepted nor rejected" in result.detail


@pytest.mark.usefixtures("_stub_key")
async def test_an_unreachable_host_is_reported_as_unreachable() -> None:
    """Same distinction for a transport failure: local trouble is not a key verdict."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    result = await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert result.outcome is ProbeOutcome.UNREACHABLE
    assert result.http_status is None


@pytest.mark.usefixtures("_stub_key")
async def test_a_provider_that_echoes_the_key_cannot_launder_it_into_the_result() -> None:
    """The detail is scrubbed of the exact credential before it is ever stored or shown."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid api key: {_ANTHROPIC_KEY}")

    result = await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert _ANTHROPIC_KEY not in result.detail
    assert _ANTHROPIC_KEY not in repr(result)


@pytest.mark.usefixtures("_stub_key")
async def test_a_body_containing_some_other_credential_is_withheld_entirely() -> None:
    """Removing the key we sent is not enough; credential *shapes* are refused too."""
    other = "sk-ant-api03-" + "9f8e7d6c" * 8 + "0000"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"upstream rejected {other}")

    result = await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert other not in result.detail
    assert "withheld" in result.detail


@pytest.mark.usefixtures("_stub_key")
async def test_the_probe_sends_exactly_one_request() -> None:
    """No retries: a failed probe is a report, and retrying would blur the latency figure."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, text="boom")

    await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert len(calls) == 1


@pytest.mark.usefixtures("_stub_key")
async def test_the_probe_never_logs_the_credential(capsys: pytest.CaptureFixture[str]) -> None:
    """The one place the key exists is the outbound header; it must not reach stdout."""
    from backend.core.config import Settings
    from backend.core.logging import configure_logging

    configure_logging(Settings(log_level="INFO"), force=True)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="{}")

    await probe_provider(Provider.ANTHROPIC, transport=_stub_transport(handler))
    assert _ANTHROPIC_KEY not in capsys.readouterr().out
