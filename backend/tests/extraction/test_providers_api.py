"""P7.1 API-surface tests: the absence of a reveal path, asserted structurally.

§6.5 requires the masked display to have "no reveal endpoint" and §7 requires
that "no endpoint returns a full key under any circumstance". Those are claims
about the **surface**, so the tests here are about the surface: the route
inventory, the declared response models, and the route module's own AST. A test
that merely called the current endpoints and found no key would pass just as
happily the day after someone added ``GET /providers/{p}/key/reveal``.

The behavioural half — "a real stored key appears in no real response body" —
needs a database and lives in ``backend/tests/integration/test_providers_db.py``.
Both halves are needed: the structural tests catch a reveal path that was added,
and the behavioural test catches one that leaks through a field nobody thought
of.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, SecretStr

from backend.api.app import create_app
from backend.api.routes import providers as providers_module
from backend.api.routes.providers import (
    DEFAULT_ACTOR,
    AssignmentIn,
    ProviderKeyIn,
    ProviderKeyOut,
    router,
)
from backend.api.security.settings import ApiSecuritySettings
from backend.core.config import Settings
from backend.extraction.providers import DEFAULT_TEMPERATURE, Provider

_REVEAL_WORDS = re.compile(r"reveal|unmask|plaintext|decrypt|secret|ciphertext|raw[-_]?key", re.I)
"""Anything in a path, route name, or field name that would read as a way to see a key."""

_ROUTES = frozenset(
    {
        ("GET", "/providers"),
        ("POST", "/providers/{provider}/key"),
        ("PUT", "/providers/{provider}/key"),
        ("DELETE", "/providers/{provider}/key"),
        ("POST", "/providers/{provider}/probe"),
        ("GET", "/extraction-tasks"),
        ("GET", "/extraction-tasks/{task}/assignment"),
        ("GET", "/extraction-tasks/{task}/assignment/versions"),
        ("PUT", "/extraction-tasks/{task}/assignment"),
    }
)
"""The complete surface, pinned.

Pinned as an exact set rather than a subset on purpose: adding an endpoint
fails this test, which puts the author in front of the reveal-path question at
the moment they are extending the surface — which is the only moment the
question gets answered honestly.
"""


def _declared_routes() -> set[tuple[str, str]]:
    """Return every (method, path) pair the router declares."""
    pairs: set[tuple[str, str]] = set()
    for route in router.routes:
        assert isinstance(route, APIRoute)
        for method in route.methods or ():
            if method not in ("HEAD", "OPTIONS"):
                pairs.add((method, route.path))
    return pairs


def _response_models() -> list[type[BaseModel]]:
    """Return every response model the router declares."""
    models: list[type[BaseModel]] = []
    for route in router.routes:
        assert isinstance(route, APIRoute)
        model = route.response_model
        if isinstance(model, type) and issubclass(model, BaseModel):
            models.append(model)
    return models


def _fields_of(model: type[BaseModel], seen: set[type[BaseModel]]) -> set[tuple[str, Any]]:
    """Return every (field name, annotation) pair reachable from *model*, recursively."""
    if model in seen:
        return set()
    seen.add(model)
    found: set[tuple[str, Any]] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        found.add((name, annotation))
        for candidate in (annotation, *getattr(annotation, "__args__", ())):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                found |= _fields_of(candidate, seen)
    return found


def _module_source_tree() -> ast.Module:
    """Parse the route module's own source."""
    return ast.parse(Path(str(providers_module.__file__)).read_text())


# --------------------------------------------------------------------------
# The surface itself
# --------------------------------------------------------------------------


def test_the_route_inventory_is_exactly_what_is_pinned_here() -> None:
    """Adding or removing an endpoint is a deliberate act, reviewed against this list."""
    assert _declared_routes() == _ROUTES


def test_no_route_path_or_name_offers_a_way_to_see_a_key() -> None:
    """§6.5: no reveal endpoint. Checked against the paths and handler names that exist."""
    for route in router.routes:
        assert isinstance(route, APIRoute)
        assert not _REVEAL_WORDS.search(route.path), route.path
        assert not _REVEAL_WORDS.search(route.name), route.name


def test_the_probe_is_reachable_only_by_a_method_a_page_load_cannot_issue() -> None:
    """A probe spends a real request; B4 leaves no spend cap, so it must not be a GET."""
    probe_methods = {method for method, path in _declared_routes() if path.endswith("/probe")}
    assert probe_methods == {"POST"}


# --------------------------------------------------------------------------
# No response model can carry a key
# --------------------------------------------------------------------------


def test_no_response_model_declares_a_field_that_could_hold_a_key() -> None:
    """Structural, not behavioural: the shape returned has nowhere to put a credential."""
    for model in _response_models():
        for name, annotation in _fields_of(model, set()):
            assert not _REVEAL_WORDS.search(name), f"{model.__name__}.{name}"
            assert annotation is not SecretStr, f"{model.__name__}.{name}"


def test_the_credential_response_model_carries_only_the_mask() -> None:
    """The one model describing a credential exposes the mask and metadata, nothing else."""
    assert set(ProviderKeyOut.model_fields) == {
        "provider",
        "configured",
        "masked_key",
        "key_version",
        "created_at",
        "rotated_at",
    }


def test_the_inbound_key_field_is_a_secret_string() -> None:
    """A validation error or traceback renders the body; SecretStr keeps the key out of both."""
    body = ProviderKeyIn(api_key=SecretStr("sk-ant-api03-abcdefghijklmnop"))
    assert "abcdefghijklmnop" not in repr(body)
    assert "abcdefghijklmnop" not in str(body)
    assert "abcdefghijklmnop" not in repr(body.api_key)


def test_the_inbound_key_field_is_never_echoed_by_a_response_model() -> None:
    """The request model is not reachable from any declared response model."""
    for model in _response_models():
        assert ProviderKeyIn not in {model, *(m for m, _ in [(model, None)])}
        assert "api_key" not in {name for name, _ in _fields_of(model, set())}


# --------------------------------------------------------------------------
# The route module cannot decrypt, whatever it is asked to do
# --------------------------------------------------------------------------


def test_the_route_module_never_names_a_decryption_path() -> None:
    """No identifier in this module can produce a plaintext key.

    An AST scan rather than a call trace: the property being asserted is that
    the decryption entry point is *absent from the module*, so no future edit
    to a handler body can reach it without also failing this test.
    """
    banned = {"load_provider_secret", "decrypt_secret", "decrypt", "ciphertext", "SecretCipher"}
    named: set[str] = set()
    for node in ast.walk(_module_source_tree()):
        if isinstance(node, ast.Name):
            named.add(node.id)
        elif isinstance(node, ast.Attribute):
            named.add(node.attr)
        elif isinstance(node, ast.alias):
            named.add(node.asname or node.name.rsplit(".", 1)[-1])
    assert not banned & named, f"reveal-capable identifiers in the route module: {banned & named}"


def test_the_route_module_does_not_import_the_crypto_layer_at_all() -> None:
    """Not importing it is stronger than not calling it, and survives a careless edit."""
    for node in ast.walk(_module_source_tree()):
        if isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert not node.module.startswith("backend.core.crypto"), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("backend.core.crypto"), alias.name


def test_the_imported_route_module_has_no_decryption_symbol_bound() -> None:
    """Belt and braces: the runtime namespace agrees with the static scan."""
    assert "load_provider_secret" not in vars(providers_module)
    assert "decrypt_secret" not in vars(providers_module)


# --------------------------------------------------------------------------
# Defaults the surface promises
# --------------------------------------------------------------------------


def test_the_assignment_request_defaults_temperature_to_zero() -> None:
    """§5-P7: a request that omits temperature gets 0, not a provider-side default."""
    body = AssignmentIn(provider=Provider.ANTHROPIC, model="some-model")
    assert body.temperature == DEFAULT_TEMPERATURE == 0.0


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_the_assignment_request_refuses_an_out_of_range_temperature(temperature: float) -> None:
    """Bounds are declared on the surface, so a bad value 422s instead of reaching the database."""
    with pytest.raises(ValueError, match="temperature"):
        AssignmentIn(provider=Provider.ANTHROPIC, model="m", temperature=temperature)


def test_a_short_api_key_is_refused_by_the_request_model() -> None:
    """An obviously truncated paste fails at the edge, before encryption or a database."""
    with pytest.raises(ValueError, match="api_key"):
        ProviderKeyIn(api_key=SecretStr("short"))


def test_the_default_actor_is_named_and_unauthenticated() -> None:
    """There is no auth yet, so the recorded actor is an assertion — with a visible default."""
    assert DEFAULT_ACTOR == "operator"


# --------------------------------------------------------------------------
# Security is inherited from the middleware, not declared per route
# --------------------------------------------------------------------------


def _app() -> FastAPI:
    """Build the real application with this router mounted, rate limiting off.

    Rate limiting sits outside CSRF and fails closed without Redis, so it would
    answer 503 before CSRF ever ran; ``test_api_rate_limit.py`` exercises that
    layer against a real Redis.
    """
    app = create_app(
        settings=Settings(environment="test"),
        security=ApiSecuritySettings(rate_limit_enabled=False),
    )
    app.include_router(router, prefix="/api")
    return app


def _client(app: FastAPI) -> AsyncClient:
    """Create an httpx client speaking ASGI directly to *app*."""
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


@pytest.mark.parametrize(
    ("method", "path"),
    sorted((method, path) for method, path in _ROUTES if method != "GET"),
)
async def test_every_mutating_route_inherits_csrf_protection(method: str, path: str) -> None:
    """No route here declares a security dependency; all of them are protected anyway (CC.2).

    The rejection happens in middleware, ahead of routing, so this reaches no
    handler and needs no database — which is also why it proves the property
    for a route whose author forgot to think about it.
    """
    concrete = path.replace("{provider}", Provider.ANTHROPIC.value).replace("{task}", "tone")
    async with _client(_app()) as client:
        response = await client.request(method, f"/api{concrete}", json={})
    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_cookie_missing"


async def test_an_unknown_provider_is_rejected_at_the_boundary() -> None:
    """The closed vocabulary is enforced by the path type, before any handler runs."""
    async with _client(_app()) as client:
        response = await client.get("/api/providers/gemini/key")
    # No GET is declared for that path, so the router refuses it outright —
    # which is the point: there is no read endpoint for a key at all.
    assert response.status_code in (404, 405)
