"""HTTP surface of the provider registry and per-task model assignment (P7.1, §6.5).

**This module contains no reveal path, and that is a property of the surface
rather than of its current call sites.** It never imports
:func:`backend.extraction.providers.registry.load_provider_secret`, never
imports anything from :mod:`backend.core.crypto`, and never reads a
``ciphertext`` attribute; every credential response model is built from
:class:`~backend.extraction.providers.registry.ProviderKeyView`, which has no
field that can hold a key. ``backend/tests/extraction/test_providers_api.py``
asserts all of that against the module's AST and against the declared response
models, so an endpoint that returned a key would fail the suite before it could
be deployed (§6.5, §7, I5).

Security posture is inherited, not declared
-------------------------------------------

Every mutating route here is protected by CSRF and rate limiting without saying
so, because :mod:`backend.api.security` installs both as default-on middleware
in front of routing (CC.2). The routes below are written exactly as a forgetful
author would write them, and the tests assert that a POST without a CSRF token
is still rejected — which is the property worth having.

The probe is a POST for the same reason. §6.5's "Test connection" spends a real
request against a real provider, and B4 leaves the platform with no spend cap
(P7.7), so it must be something an operator *does*, not something a page load
can cause. A GET would be reachable by a link prefetch, a crawler, or a browser
speculating on a hover.

Actor
-----

There is no authentication (§1.1: single operator, local deployment), so the
actor recorded against every change is read from the ``X-Operator`` header and
falls back to :data:`DEFAULT_ACTOR`. It is a **caller assertion, not a verified
identity** — the same caveat :mod:`backend.db.audit` records for the column it
lands in — and nothing downstream may present it as authenticated until there
is an authentication system to authenticate it.

Mounting
--------

P7.1 does not own ``backend/api/app.py``, so this router is **not yet mounted**.
Wiring it is one line there — ``app.include_router(providers_router,
prefix="/api")`` — and until it lands these endpoints are reachable only by a
caller that includes the router itself, which is how the tests exercise them.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Final

from fastapi import APIRouter, Header, HTTPException, Path, status
from pydantic import BaseModel, Field, SecretStr

from backend.extraction.providers.assignments import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    MAX_TEMPERATURE,
    AssignmentNotConfiguredError,
    ModelAssignment,
    assign_task_model,
    assignment_versions,
    current_assignment,
    list_current_assignments,
)
from backend.extraction.providers.catalog import Provider
from backend.extraction.providers.probe import ProbeOutcome, ProbeResult, probe_provider
from backend.extraction.providers.registry import (
    ProviderKeyAlreadyConfiguredError,
    ProviderKeyNotConfiguredError,
    ProviderKeyView,
    add_provider_key,
    delete_provider_key,
    list_provider_keys,
    rotate_provider_key,
)

__all__ = ["DEFAULT_ACTOR", "OPERATOR_HEADER", "router"]

router = APIRouter(tags=["providers"])

OPERATOR_HEADER: Final = "X-Operator"
"""Header naming who is making the change. Unauthenticated — see the module docstring."""

DEFAULT_ACTOR: Final = "operator"
"""Actor recorded when the request carries no ``X-Operator`` header."""

_MIN_KEY_CHARS: Final = 8
"""Mirror of the registry's floor, so an obviously-truncated paste 422s at the edge."""

ActorHeader = Annotated[str | None, Header(alias=OPERATOR_HEADER)]
ProviderPath = Annotated[Provider, Path(description="Provider name, e.g. 'anthropic'.")]
TaskPath = Annotated[str, Path(min_length=1, description="Extraction task name.")]


def _actor(header_value: str | None) -> str:
    """Return the actor to record for this request.

    Args:
        header_value: raw ``X-Operator`` header, or ``None``.

    Returns:
        The trimmed header value when it carries one, else :data:`DEFAULT_ACTOR`.
        Trimming here (rather than rejecting, as the audit log does for its own
        arguments) keeps a stray space in a hand-typed header from turning into
        a 500 on an otherwise valid configuration change.
    """
    if header_value is None or not header_value.strip():
        return DEFAULT_ACTOR
    return header_value.strip()


class ProviderKeyIn(BaseModel):
    """Request body carrying an operator-supplied API key.

    ``api_key`` is a :class:`~pydantic.SecretStr`, so the model's ``repr`` —
    which is what a validation error, a traceback frame, or a debug log would
    render — shows ``**********`` rather than the credential.
    """

    api_key: SecretStr = Field(
        min_length=_MIN_KEY_CHARS,
        description="The provider API key. Stored encrypted; never returned by any endpoint.",
    )


class ProviderKeyOut(BaseModel):
    """A provider's registry entry as the operator may see it.

    Every field here is safe to render. There is deliberately no field for the
    key or its ciphertext, so no response built from this model can carry one
    regardless of how the handler is written (§6.5: display permanently masked,
    no reveal endpoint).
    """

    provider: Provider
    configured: bool = Field(description="Whether a credential is stored for this provider.")
    masked_key: str | None = Field(
        default=None,
        description="Permanent masked rendering, e.g. 'sk-...4f2a'. Null when unconfigured.",
    )
    key_version: int | None = Field(
        default=None, description="Rotation counter: 1 when first added, +1 per rotation."
    )
    created_at: dt.datetime | None = None
    rotated_at: dt.datetime | None = None


class ProviderListOut(BaseModel):
    """Every known provider, configured or not.

    Unconfigured providers are included on purpose: the operator's question on
    this page is as often "what have I not set up yet" as "what have I".
    """

    providers: list[ProviderKeyOut]


class AssignmentIn(BaseModel):
    """Request body assigning a model to an extraction task.

    ``temperature`` defaults to :data:`DEFAULT_TEMPERATURE`, which is **0** —
    §5-P7 requires temperature 0 everywhere for extraction, so a request that
    omits it gets 0 rather than a provider-side default nobody chose.
    """

    provider: Provider
    model: str = Field(min_length=1, description="Provider-side model id, stored verbatim.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=MAX_TEMPERATURE,
        description="Dimensionless sampling temperature. Defaults to 0 (§5-P7).",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS, ge=1, description="Response cap in tokens (count)."
    )
    timeout_s: float = Field(
        default=DEFAULT_TIMEOUT_S, gt=0.0, description="Per-request timeout in seconds."
    )


class AssignmentOut(BaseModel):
    """One recorded version of a task's assignment."""

    task: str
    version: int = Field(description="Configuration version; the greatest is in force (§6.5).")
    provider: Provider
    model: str
    temperature: float
    max_tokens: int
    timeout_s: float
    actor: str = Field(description="Who assigned it, as asserted — not an authenticated identity.")
    correlation_id: str | None
    recorded_at: dt.datetime


class AssignmentHistoryOut(BaseModel):
    """Every recorded version of one task's assignment, newest first."""

    task: str
    versions: list[AssignmentOut]


class AssignmentListOut(BaseModel):
    """The version in force for every assigned task."""

    assignments: list[AssignmentOut]


class ProbeOut(BaseModel):
    """Result of one explicitly-invoked connection probe (§6.5)."""

    provider: Provider
    outcome: ProbeOutcome
    latency_ms: float = Field(description="Measured round-trip duration in milliseconds.")
    http_status: int | None = Field(description="Provider HTTP status, or null if none arrived.")
    detail: str


def _key_out(provider: Provider, view: ProviderKeyView | None) -> ProviderKeyOut:
    """Render one provider's registry entry, configured or not."""
    if view is None:
        return ProviderKeyOut(provider=provider, configured=False)
    return ProviderKeyOut(
        provider=provider,
        configured=True,
        masked_key=view.masked_key,
        key_version=view.key_version,
        created_at=view.created_at,
        rotated_at=view.rotated_at,
    )


def _assignment_out(assignment: ModelAssignment) -> AssignmentOut:
    """Render one assignment version."""
    return AssignmentOut(
        task=assignment.task,
        version=assignment.version,
        provider=assignment.provider,
        model=assignment.model,
        temperature=assignment.temperature,
        max_tokens=assignment.max_tokens,
        timeout_s=assignment.timeout_s,
        actor=assignment.actor,
        correlation_id=assignment.correlation_id,
        recorded_at=assignment.recorded_at,
    )


def _probe_out(result: ProbeResult) -> ProbeOut:
    """Render a probe result."""
    return ProbeOut(
        provider=result.provider,
        outcome=result.outcome,
        latency_ms=result.latency_ms,
        http_status=result.http_status,
        detail=result.detail,
    )


@router.get("/providers", response_model=ProviderListOut)
async def get_providers() -> ProviderListOut:
    """List every known provider with its masked credential state.

    Never returns a key. The masked rendering is the only view of a credential
    this API has (§6.5).
    """
    stored = {view.provider: view for view in await list_provider_keys()}
    return ProviderListOut(
        providers=[_key_out(provider, stored.get(provider)) for provider in Provider]
    )


@router.post(
    "/providers/{provider}/key",
    response_model=ProviderKeyOut,
    status_code=status.HTTP_201_CREATED,
)
async def post_provider_key(
    provider: ProviderPath,
    body: ProviderKeyIn,
    x_operator: ActorHeader = None,
) -> ProviderKeyOut:
    """Add a provider's first API key.

    Distinct from rotation on purpose: adding and replacing mean different
    things to an operator and are recorded as different events. Replacing an
    existing key is ``PUT``.

    Returns 201 with the masked entry; 409 when the provider already has a key.
    """
    try:
        view = await add_provider_key(
            provider, body.api_key.get_secret_value(), actor=_actor(x_operator)
        )
    except ProviderKeyAlreadyConfiguredError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _key_out(provider, view)


@router.put("/providers/{provider}/key", response_model=ProviderKeyOut)
async def put_provider_key(
    provider: ProviderPath,
    body: ProviderKeyIn,
    x_operator: ActorHeader = None,
) -> ProviderKeyOut:
    """Rotate a provider's API key, replacing the stored one.

    The superseded ciphertext is overwritten rather than archived, so a retired
    key stops being decryptable; the record of the rotation — actor,
    correlation id, and the masked renderings either side — is kept in the
    append-only audit log.

    Returns 200 with the new masked entry; 404 when there is nothing to rotate.
    """
    try:
        view = await rotate_provider_key(
            provider, body.api_key.get_secret_value(), actor=_actor(x_operator)
        )
    except ProviderKeyNotConfiguredError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _key_out(provider, view)


@router.delete("/providers/{provider}/key", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider_key_route(
    provider: ProviderPath,
    x_operator: ActorHeader = None,
) -> None:
    """Delete a provider's stored API key.

    Task assignments naming the provider are left untouched — an assignment is
    a statement of intent, and cascading a credential deletion into it would
    silently rewrite the operator's pipeline. Calls for that provider refuse
    until a key is configured again.

    Returns 204; 404 when there is nothing to delete.
    """
    try:
        await delete_provider_key(provider, actor=_actor(x_operator))
    except ProviderKeyNotConfiguredError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/providers/{provider}/probe", response_model=ProbeOut)
async def post_provider_probe(provider: ProviderPath) -> ProbeOut:
    """Issue one minimal live call to the provider and report status and latency.

    **Spends a real request.** POST rather than GET so a page load, prefetch or
    crawler cannot trigger it: B4 is unresolved and the cost governor (P7.7)
    does not exist, so the probe must be an action an operator takes.

    Returns 200 with the measured result; **409 when no key is configured** —
    the probe refuses rather than reporting a result it did not measure (I3).
    """
    try:
        result = await probe_provider(provider)
    except ProviderKeyNotConfiguredError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _probe_out(result)


@router.get("/extraction-tasks", response_model=AssignmentListOut)
async def get_extraction_tasks() -> AssignmentListOut:
    """List the assignment version currently in force for every assigned task."""
    return AssignmentListOut(
        assignments=[_assignment_out(item) for item in await list_current_assignments()]
    )


@router.get("/extraction-tasks/{task}/assignment", response_model=AssignmentOut)
async def get_task_assignment(task: TaskPath) -> AssignmentOut:
    """Return the assignment version currently in force for one task.

    Returns 404 when the task has never been assigned — which is different from
    a task whose assignment is unusable, and is reported differently on purpose.
    """
    try:
        return _assignment_out(await current_assignment(task))
    except AssignmentNotConfiguredError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/extraction-tasks/{task}/assignment/versions", response_model=AssignmentHistoryOut)
async def get_task_assignment_versions(task: TaskPath) -> AssignmentHistoryOut:
    """Return every recorded version of one task's assignment, newest first.

    The superseded versions are the point of §6.5's versioning rule: this is
    where the operator reads the configuration a past extraction run used.
    """
    try:
        versions = await assignment_versions(task)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return AssignmentHistoryOut(task=task, versions=[_assignment_out(item) for item in versions])


@router.put(
    "/extraction-tasks/{task}/assignment",
    response_model=AssignmentOut,
    status_code=status.HTTP_201_CREATED,
)
async def put_task_assignment(
    task: TaskPath,
    body: AssignmentIn,
    x_operator: ActorHeader = None,
) -> AssignmentOut:
    """Assign a model to an extraction task, creating a **new version**.

    201 rather than 200 because the call creates a resource — version *n+1* —
    instead of overwriting version *n*, which is exactly the distinction §6.5
    draws. The previous version stays readable at its own version number.
    """
    try:
        assignment = await assign_task_model(
            task,
            body.provider,
            body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            timeout_s=body.timeout_s,
            actor=_actor(x_operator),
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _assignment_out(assignment)
