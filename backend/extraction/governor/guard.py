"""The enforcement point: a model client that cannot be called past its cap (P7.7).

:class:`GovernedModelClient` implements
:class:`~backend.extraction.tasks.client.ModelClient` and wraps another one. It
is the whole of P7.7's "enforced before the call, not after", and the reason it
is a wrapper rather than a check the pipeline is asked to remember is
structural:

**there is no path to the inner client that does not go through the
authorization.** The inner client is a private attribute of this object. A
caller holding a governed client cannot reach past it, cannot call it "just this
once" without the check, and cannot forget to check — there is no method that
would let them. Compare the alternative, a ``governor.check()`` the pipeline
calls before ``client.complete()``: identical behaviour when written correctly,
and one refactor away from a call site that no longer checks. The property this
module needs is not "the check happens", it is "the call cannot happen without
it", and only one of those two shapes provides it.

The consequence for wiring is that nothing else changes.
:class:`~backend.extraction.tasks.pipeline.ExtractionPipeline` already takes a
``client``; handing it a governed one governs every call it makes, per chunk,
per document, including the ensemble's parallel calls (P7.5) — with no edit to
the pipeline.

What "before" buys, concretely
-------------------------------

On a refusal, the inner client is **never touched**. That is asserted the only
way it can honestly be asserted — by injecting a client that fails the test if
it is called at all — because a test that only checked the raised exception
would pass just as happily if the request had been sent and the response thrown
away. See ``backend/tests/extraction/governor/test_guard.py``.

Failure accounting, and the one thing this client cannot know
--------------------------------------------------------------

When the inner call raises, this client **settles the reservation at its upper
bound** rather than releasing it. It does not know whether the request left the
host. A connection refused before the first byte cost nothing; a timeout after
the model generated 900 tokens cost nearly the bound; both arrive here as the
same :class:`~backend.extraction.tasks.client.ModelCallError`. Charging the
bound over-counts, which for a control is the safe direction — an over-counted
cap refuses calls that would have fit, an under-counted one lets real money out.

:meth:`~backend.extraction.governor.governor.CostGovernor.abandon` exists for
callers that *can* prove nothing was sent. This client never calls it, and that
asymmetry is deliberate rather than an omission.

Secrets (I5)
------------

Nothing here reads, holds, or logs a credential. The governed client sees a
:class:`~backend.extraction.tasks.client.ModelRequest` — model identifier,
system instruction, anonymized prompt — and hands it to the inner client, which
is the object that knows how to authenticate. No API key exists in this module's
scope, so none can reach its logs or its exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.core.logging import get_logger
from backend.extraction.tasks.client import ModelCallError

if TYPE_CHECKING:
    from backend.extraction.governor.estimate import Money
    from backend.extraction.governor.governor import Authorization, CostGovernor
    from backend.extraction.governor.ledger import SpendRecord
    from backend.extraction.tasks.client import ModelClient, ModelRequest, ModelResponse

__all__ = ["GovernedCompletion", "GovernedModelClient"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GovernedCompletion:
    """One governed call: what was asked, what answered, and what it cost.

    Attributes:
        response: the provider's response, verbatim.
        requested_model: the qualified model the caller asked for.
        served_model: the qualified model that actually answered. Differs from
            *requested_model* exactly when the cap's policy degraded the call —
            **this is the field an artefact must carry for the result to be
            reproducible (I2)**.
        record: the settled ledger row, the durable audit trail of this call —
            including both models, both token bounds, and the reconciliation.
    """

    response: ModelResponse
    requested_model: str
    served_model: str
    record: SpendRecord

    @property
    def degraded(self) -> bool:
        """Whether a cheaper model was substituted for the requested one."""
        return self.served_model != self.requested_model

    @property
    def estimated_cost(self) -> Money:
        """The upper bound that was reserved before the call, in its own currency."""
        return self.record.estimated_cost

    @property
    def actual_cost(self) -> Money | None:
        """What the call really cost, or ``None`` when the provider reported no tokens."""
        return self.record.actual_cost


class GovernedModelClient:
    """Wraps a model client so no call can be made past its provider's cap.

    Satisfies :class:`~backend.extraction.tasks.client.ModelClient`, so it drops
    straight into :class:`~backend.extraction.tasks.pipeline.ExtractionPipeline`
    in place of the client it governs.
    """

    def __init__(self, *, inner: ModelClient, governor: CostGovernor) -> None:
        """Build the governed client.

        Args:
            inner: the client that actually reaches a provider. Private to this
                object: no accessor exposes it, because an accessor would be a
                supported way to bypass the cap.
            governor: what authorizes, reconciles and records.
        """
        self._inner = inner
        self._governor = governor

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Authorize, call, reconcile — and return only the provider's response.

        The :class:`~backend.extraction.tasks.client.ModelClient` surface, so an
        existing caller is governed without knowing it. A caller that needs to
        see which model actually answered uses :meth:`complete_governed`; note
        that :attr:`~backend.extraction.tasks.client.ModelResponse.model` also
        carries what the provider reported serving.

        Args:
            request: the call to make, with a qualified model identifier.

        Returns:
            The provider's response, verbatim.

        Raises:
            backend.extraction.governor.errors.SpendCapExceededError: the call
                would breach a cap. **The inner client was not touched.**
            backend.extraction.governor.errors.GovernorConfigurationError: no
                cap, no price, or an unusable degrade configuration. Also no
                call.
            backend.extraction.tasks.client.ModelCallError: the provider call
                failed. The reservation is settled at its bound first, then the
                error propagates unchanged — a retry loop built around this
                exception behaves exactly as it did before governance, except
                that its retries are themselves capped.
        """
        return (await self.complete_governed(request)).response

    async def complete_governed(self, request: ModelRequest) -> GovernedCompletion:
        """Make one governed call and return the full record of it.

        Args:
            request: the call to make, with a qualified model identifier.

        Returns:
            The :class:`GovernedCompletion`, naming the model that actually
            answered and the ledger row that recorded it.

        Raises:
            backend.extraction.governor.errors.GovernorError: any refusal — cap
                breached, price unknown, cap unconfigured, degradation
                unavailable. In every case **no request was made**.
            backend.extraction.tasks.client.ModelCallError: the provider call
                failed, after the reservation was settled at its bound.
        """
        authorization: Authorization = await self._governor.authorize(request)
        try:
            response = await self._inner.complete(authorization.request)
        except ModelCallError:
            # Settled, not released: this client cannot know whether the request
            # left the host, and a control assumes the worst (module docstring).
            await self._governor.settle_failed(authorization)
            raise
        record = await self._governor.settle(authorization, response)
        if authorization.degraded:
            _logger.warning(
                "llm_call_served_by_substituted_model",
                requested_model=authorization.requested_model,
                served_model=authorization.served_model,
                provider=authorization.cap.provider.value,
                estimated_cost=str(record.estimated_cost),
                actual_cost=None if record.actual_cost is None else str(record.actual_cost),
                reservation_id=record.reservation_id,
            )
        return GovernedCompletion(
            response=response,
            requested_model=authorization.requested_model,
            served_model=authorization.served_model,
            record=record,
        )
