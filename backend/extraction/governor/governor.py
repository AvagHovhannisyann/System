"""The decision: authorize a call, or refuse it, before it is made (P7.7, §6.5).

:class:`CostGovernor` is the thing that says yes or no. It reads the provider's
cap (:mod:`backend.extraction.governor.caps`), bounds what the call would cost
(:mod:`backend.extraction.governor.estimate`), and asks the ledger to admit that
bound atomically (:mod:`backend.extraction.governor.ledger`). Every path out of
:meth:`CostGovernor.authorize` is either a reservation in hand or an exception;
there is no path that returns "probably fine".

The ordering is the point. **Nothing here is reachable after a call has been
made**, because the object that makes calls
(:class:`~backend.extraction.governor.guard.GovernedModelClient`) cannot reach
its inner client without going through :meth:`authorize` first. A governor that
recorded spend afterwards would be an accounting system: correct, auditable, and
useless — by the time it noticed, the money was gone.

Halt versus degrade, per provider
----------------------------------

§6.5 makes hard-stop behaviour "configurable between halt and
degrade-to-cheaper-model", and :class:`~backend.extraction.governor.caps.CapPolicy`
carries that choice **on the provider's cap**, not globally. A global policy
would be meaningless: degradation names a specific cheaper model on a specific
provider, and there is no such statement to make across providers.

Under :attr:`~backend.extraction.governor.caps.CapPolicy.HALT` a breach raises
and the extraction stops. Under
:attr:`~backend.extraction.governor.caps.CapPolicy.DEGRADE` the governor prices
the configured substitute, requires it to be **strictly cheaper for this call**,
and re-attempts the reservation with it. Three refusals remain, and they matter:

* the substitute has no configured price →
  :class:`~backend.extraction.governor.errors.ModelPriceUnknownError`. A cap
  enforced against a made-up price enforces nothing (I3);
* the substitute is not cheaper for this call →
  :class:`~backend.extraction.governor.errors.DegradationUnavailableError`. A
  "fallback" that costs the same is a second attempt at the same spend;
* the substitute does not fit either →
  :class:`~backend.extraction.governor.errors.SpendCapExceededError`. **Halt is
  the floor of degrade**, not an alternative to it. Degrading is a way to spend
  less, and when there is nothing left it has nothing to offer.

Note that cheapness is compared on **whole-call estimates**, not headline rates.
Which model is cheaper for a given call depends on the prompt-to-response ratio,
and a model with a lower output rate can still cost more on a long prompt.

A substituted model must reach the artefact (I2)
-------------------------------------------------

Silently answering with a cheaper model changes the experiment. A result
attributed to a model that never read the document is not reproducible from
"git commit + data version + config hash + seed", because the config does not
say what happened.

So every authorization carries both models, and **every ledger row carries both
models** (:attr:`~backend.extraction.governor.ledger.SpendRecord.served_model`).
The ledger is append-only and durable, so "which model actually answered this
call" is a recorded fact joinable by correlation id, not an inference. The
governed client additionally returns the served model to its caller on
:class:`~backend.extraction.governor.guard.GovernedCompletion`.

**A known gap, stated rather than hidden.** The P7.3 pipeline
(:mod:`backend.extraction.tasks.pipeline`) records the model *it was asked for*
on :class:`~backend.extraction.tasks.pipeline.ChunkExtraction` and addresses its
cache by that same string, both computed before the client is called. A pipeline
wired to a degrading governor would therefore file the substitute's answer under
the primary's name. That file is not P7.7's to change; the consequence is
recorded here, pinned by a characterization test in
``backend/tests/extraction/governor/test_pipeline_interaction.py``, and reported
as follow-up work for whoever owns the pipeline. Until it is closed, the ledger
row — not the extraction row — is the record of which model answered.

Units: all monetary amounts are
:class:`~backend.extraction.governor.estimate.Money` and carry their currency
(I4). All instants are timezone-aware UTC.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from backend.core.logging import get_logger
from backend.extraction.governor.caps import CapPolicy
from backend.extraction.governor.errors import (
    CurrencyMismatchError,
    DegradationUnavailableError,
    SpendCapExceededError,
)
from backend.extraction.governor.estimate import (
    REQUEST_FRAMING_TOKENS,
    Money,
    actual_call_cost,
    estimate_call_cost,
    provider_of,
)
from backend.extraction.governor.ledger import CallOutcome
from backend.extraction.providers.catalog import provider_from_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from backend.extraction.governor.caps import CapBook, ProviderCap, SpendWindow
    from backend.extraction.governor.estimate import CallEstimate, PriceBook, TokenPrice
    from backend.extraction.governor.ledger import Reservation, SpendLedger, SpendRecord
    from backend.extraction.providers.catalog import Provider
    from backend.extraction.tasks.client import ModelRequest, ModelResponse

__all__ = ["Authorization", "CostGovernor"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Authorization:
    """Permission to make one specific call, with the headroom already consumed.

    Attributes:
        reservation: the ledger reservation backing this permission.
        request: the outbound request **as it must be sent** — identical to the
            caller's request except that ``model`` names the served model. A
            caller that sends something else has sent an unauthorized call, so
            the governed client sends this object and never the original.
        price: the rates of the served model, kept so settlement reconciles
            against what actually answered rather than against what was asked
            for. Degrading and then settling at the primary's price would book a
            cheap call at an expensive rate.
        cap: the provider cap in force at authorization time.

    Frozen, because an authorization that could be edited between the check and
    the call is not an authorization.
    """

    reservation: Reservation
    request: ModelRequest
    price: TokenPrice
    cap: ProviderCap

    @property
    def requested_model(self) -> str:
        """The qualified model the caller asked for."""
        return self.reservation.requested_model

    @property
    def served_model(self) -> str:
        """The qualified model that will actually be called."""
        return self.reservation.served_model

    @property
    def degraded(self) -> bool:
        """Whether a cheaper model was substituted (§6.5, I2)."""
        return self.reservation.degraded

    @property
    def estimate(self) -> CallEstimate:
        """The upper bound reserved for this call."""
        return self.reservation.estimate


class CostGovernor:
    """Authorizes provider calls against configured per-provider caps.

    Holds its collaborators rather than constructing them, so what a run is
    permitted to spend, at what prices, recorded where, are all the caller's
    decisions:

    * ``caps`` — the configured limits and ceiling behaviour. There is no
      default, and an empty book cannot be constructed (B4);
    * ``prices`` — the rates each model is charged at. Missing prices refuse
      (I3);
    * ``ledger`` — where headroom is atomically admitted and durably recorded.

    Nothing here touches a credential. The governor decides *whether* a call may
    be made; :mod:`backend.extraction.providers.registry` decides how it is
    authenticated, and the two never meet — no API key reaches this module, its
    logs, or its exceptions (I5).
    """

    def __init__(
        self,
        *,
        caps: CapBook,
        prices: PriceBook,
        ledger: SpendLedger,
        framing_tokens: int = REQUEST_FRAMING_TOKENS,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        """Build a governor.

        Args:
            caps: the configured caps. Constructing an empty
                :class:`~backend.extraction.governor.caps.CapBook` already
                raised, so a governor cannot exist without at least one cap.
            prices: the model price book. May be empty — every call then refuses
                with :class:`~backend.extraction.governor.errors.ModelPriceUnknownError`,
                which is the correct behaviour under B4.
            ledger: where reservations are admitted and recorded.
            framing_tokens: envelope allowance added to every input-token bound
                (:data:`~backend.extraction.governor.estimate.REQUEST_FRAMING_TOKENS`).
            clock: returns the current instant, timezone-aware. Injected so the
                window-boundary behaviour is testable without waiting for
                midnight UTC; defaults to the real clock.
        """
        self._caps = caps
        self._prices = prices
        self._ledger = ledger
        self._framing_tokens = framing_tokens
        self._clock = clock if clock is not None else _utc_now

    @property
    def caps(self) -> CapBook:
        """The configured caps, for the operator's gauges (§6.5)."""
        return self._caps

    @property
    def ledger(self) -> SpendLedger:
        """The ledger this governor books against."""
        return self._ledger

    async def authorize(
        self, request: ModelRequest, *, correlation_id: str | None = None
    ) -> Authorization:
        """Reserve headroom for *request*, or refuse it — no call is made here.

        The sequence, in order, with every step able to refuse:

        1. resolve the provider from the qualified model identifier;
        2. read that provider's cap — absent is a refusal, not a default (B4);
        3. read the model's price — absent is a refusal, not a guess (I3);
        4. bound the call's cost *above*
           (:func:`~backend.extraction.governor.estimate.estimate_call_cost`);
        5. ask the ledger to admit that bound atomically. On a breach, halt or
           degrade per the cap's policy.

        Args:
            request: the call the caller wants to make. Its ``model`` must be
                qualified (``"provider:model"``).
            correlation_id: request id (D-003) to record, or ``None`` to resolve
                it from the request in flight.

        Returns:
            An :class:`Authorization` whose ``request`` is what must actually be
            sent.

        Raises:
            ValueError: ``request.model`` carries no provider qualifier.
            backend.extraction.governor.errors.CapNotConfiguredError: the
                provider has no configured cap.
            backend.extraction.governor.errors.ModelPriceUnknownError: the
                model — or, on the degrade path, the substitute — has no
                configured price.
            backend.extraction.governor.errors.CurrencyMismatchError: the
                price and the cap are denominated differently.
            backend.extraction.governor.errors.DegradationUnavailableError: the
                cap says degrade and the substitute is not usable or not
                cheaper.
            backend.extraction.governor.errors.SpendCapExceededError: the call
                would breach a cap and either the policy is halt or the
                substitute does not fit either.
        """
        provider = provider_from_name(provider_of(request.model))
        cap = self._caps.cap_for(provider)
        price = self._prices.price_for(request.model)
        estimate = self._estimate(request, price)
        _require_comparable(estimate.cost, cap)
        at = self._now()
        resolved_correlation_id = (
            correlation_id if correlation_id is not None else _current_correlation_id()
        )
        try:
            reservation = await self._reserve(
                provider=provider,
                cap=cap,
                requested_model=request.model,
                served_model=request.model,
                estimate=estimate,
                correlation_id=resolved_correlation_id,
                at=at,
            )
        except SpendCapExceededError as breach:
            if cap.policy is CapPolicy.HALT:
                _logger.warning(
                    "llm_spend_halted",
                    provider=provider.value,
                    requested_model=request.model,
                    window=breach.window.value,
                    window_key=breach.window_key,
                    limit=str(breach.limit),
                    committed=str(breach.committed),
                    estimated_cost=str(breach.estimate),
                )
                raise
            return await self._degrade(
                request=request,
                cap=cap,
                provider=provider,
                primary_estimate=estimate,
                breach=breach,
                correlation_id=resolved_correlation_id,
                at=at,
            )
        return Authorization(reservation=reservation, request=request, price=price, cap=cap)

    async def settle(self, authorization: Authorization, response: ModelResponse) -> SpendRecord:
        """Reconcile a completed call against what it actually cost.

        The cost is computed from the token counts **the provider reported**,
        at the price of the model that actually served the call. When the
        provider reported no counts the reservation settles at its upper bound
        — never at a re-estimate, which would be a fabricated measurement (I3).

        Args:
            authorization: the permission the call was made under.
            response: what came back.

        Returns:
            The ``settled`` ledger row.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: this
                authorization was already settled or released.
        """
        actual = actual_call_cost(response, authorization.price)
        return await self._ledger.settle(
            authorization.reservation,
            actual=actual,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            outcome=CallOutcome.SUCCEEDED,
        )

    async def settle_failed(self, authorization: Authorization) -> SpendRecord:
        """Settle a call that raised, **at its upper bound**.

        A failed provider call may have cost nothing or nearly everything, and
        nothing observable from here distinguishes the two. The bound is charged
        because over-counting refuses calls that would have fit, while
        under-counting lets real money out — and a control chooses the first.

        Args:
            authorization: the permission the failed call was made under.

        Returns:
            The ``settled`` ledger row, with ``reconciled=False`` and outcome
            :attr:`~backend.extraction.governor.ledger.CallOutcome.FAILED`.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: already
                settled or released.
        """
        return await self._ledger.settle(
            authorization.reservation,
            actual=None,
            input_tokens=None,
            output_tokens=None,
            outcome=CallOutcome.FAILED,
        )

    async def abandon(self, authorization: Authorization) -> SpendRecord:
        """Give headroom back for a call the caller can **prove** was never sent.

        Only for that case. A caller that merely *believes* nothing was sent
        should settle instead — see
        :meth:`settle_failed`. The governed client never calls this, because it
        cannot prove it; a scheduler that reserved for an ensemble and then
        found a cache hit can.

        Args:
            authorization: the permission being given up.

        Returns:
            The ``released`` ledger row.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: already
                settled or released.
        """
        return await self._ledger.release(authorization.reservation)

    async def spend_to_date(
        self, provider: Provider, window: SpendWindow, *, at: dt.datetime | None = None
    ) -> Money:
        """Return committed spend in one window — the operator's gauge (§6.5).

        Args:
            provider: whose budget to report.
            window: daily or monthly.
            at: the instant whose window to report, timezone-aware. Defaults to
                now.

        Returns:
            Settled spend plus outstanding reservations, in the cap's currency.
            Reservations are included because money that is promised is money
            that is not available, and a gauge that showed only settled spend
            would read low during exactly the burst an operator is watching for.

        Raises:
            backend.extraction.governor.errors.CapNotConfiguredError: the
                provider has no cap, so there is no currency to report in.
            ValueError: *at* is naive.
        """
        cap = self._caps.cap_for(provider)
        moment = at if at is not None else self._now()
        return await self._ledger.committed(
            provider=provider,
            window=window,
            window_key=window.key(moment),
            currency=cap.currency,
        )

    async def headroom(
        self, provider: Provider, window: SpendWindow, *, at: dt.datetime | None = None
    ) -> Money:
        """Return what is left in one window, floored at zero.

        Args:
            provider: whose budget to report.
            window: daily or monthly.
            at: the instant whose window to report. Defaults to now.

        Returns:
            ``limit - committed``, or zero when committed spend has reached or
            passed the limit. Floored because a negative headroom is not a debt
            anyone can pay down, and rendering one on a gauge invites reading it
            as one.

        Raises:
            backend.extraction.governor.errors.CapNotConfiguredError: the
                provider has no cap.
        """
        cap = self._caps.cap_for(provider)
        committed = await self.spend_to_date(provider, window, at=at)
        limit = cap.limits[window]
        return limit - committed if limit >= committed else limit.zero_like()

    def _estimate(
        self, request: ModelRequest, price: TokenPrice, *, model: str | None = None
    ) -> CallEstimate:
        """Bound *request*'s cost above at *price*, using this governor's allowance."""
        return estimate_call_cost(request, price, framing_tokens=self._framing_tokens, model=model)

    async def _reserve(
        self,
        *,
        provider: Provider,
        cap: ProviderCap,
        requested_model: str,
        served_model: str,
        estimate: CallEstimate,
        correlation_id: str | None,
        at: dt.datetime,
    ) -> Reservation:
        """Ask the ledger to admit *estimate* atomically against both windows."""
        return await self._ledger.reserve(
            provider=provider,
            requested_model=requested_model,
            served_model=served_model,
            estimate=estimate,
            windows=cap.windows(at),
            limits=cap.limits,
            policy=cap.policy,
            correlation_id=correlation_id,
            occurred_at=at,
        )

    async def _degrade(
        self,
        *,
        request: ModelRequest,
        cap: ProviderCap,
        provider: Provider,
        primary_estimate: CallEstimate,
        breach: SpendCapExceededError,
        correlation_id: str | None,
        at: dt.datetime,
    ) -> Authorization:
        """Substitute the configured cheaper model, or refuse.

        Raises:
            backend.extraction.governor.errors.DegradationUnavailableError: the
                substitute is the model that just breached, or is not strictly
                cheaper for *this* call.
            backend.extraction.governor.errors.ModelPriceUnknownError: the
                substitute has no configured price.
            backend.extraction.governor.errors.SpendCapExceededError: the
                substitute does not fit either — halt is the floor of degrade.
        """
        fallback = cap.require_degrade_target()
        if fallback == request.model:
            msg = (
                f"provider {provider.value!r} is configured to degrade to {fallback!r}, which "
                "is the model that just breached the cap. A fallback to itself is not a "
                "degradation; no call was made"
            )
            raise DegradationUnavailableError(msg)
        fallback_price = self._prices.price_for(fallback)
        fallback_estimate = self._estimate(request, fallback_price, model=fallback)
        if fallback_estimate.cost >= primary_estimate.cost:
            msg = (
                f"refusing to degrade {request.model!r} to {fallback!r}: this call would cost "
                f"at most {fallback_estimate.cost} on the fallback against "
                f"{primary_estimate.cost} on the primary, so the substitution does not spend "
                "less. A fallback that costs the same or more is a second attempt at the same "
                "spend under a different name; no call was made"
            )
            raise DegradationUnavailableError(msg)
        reservation = await self._reserve(
            provider=provider,
            cap=cap,
            requested_model=request.model,
            served_model=fallback,
            estimate=fallback_estimate,
            correlation_id=correlation_id,
            at=at,
        )
        _logger.warning(
            "llm_spend_degraded",
            provider=provider.value,
            requested_model=request.model,
            served_model=fallback,
            window=breach.window.value,
            window_key=breach.window_key,
            limit=str(breach.limit),
            primary_estimate=str(primary_estimate.cost),
            fallback_estimate=str(fallback_estimate.cost),
            reservation_id=reservation.reservation_id,
        )
        return Authorization(
            reservation=reservation,
            request=dataclasses.replace(request, model=fallback),
            price=fallback_price,
            cap=cap,
        )

    def _now(self) -> dt.datetime:
        """Return the current instant, refusing a naive clock.

        Raises:
            ValueError: the injected clock returned a naive datetime, which
                would place spend in a window that depends on the host's
                timezone.
        """
        moment = self._clock()
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            msg = (
                f"the governor's clock returned naive timestamp {moment!r}; spend windows are "
                "UTC calendar periods and a naive instant names no period"
            )
            raise ValueError(msg)
        return moment


def _utc_now() -> dt.datetime:
    """Return the current instant in UTC."""
    return dt.datetime.now(tz=dt.UTC)


def _require_comparable(cost: Money, cap: ProviderCap) -> None:
    """Refuse a price whose currency the cap cannot be compared against.

    Args:
        cost: the estimated cost of a call.
        cap: the provider cap it would be checked against.

    Raises:
        backend.extraction.governor.errors.CurrencyMismatchError: the two are
            denominated differently. Checked here, before the ledger, so the
            message names the misconfigured pair rather than surfacing as an
            arithmetic failure deep inside a reservation.
    """
    if cost.currency != cap.currency:
        msg = (
            f"provider {cap.provider.value!r} is capped in {cap.currency} but this call is "
            f"priced at {cost}: a cap enforced against another currency is a cap enforced "
            "against an unstated conversion, and this system holds no exchange rate (I4)"
        )
        raise CurrencyMismatchError(msg)


def _current_correlation_id() -> str | None:
    """Return the request id bound to this context, or ``None`` outside a request.

    Reads the ``request_id`` bound by
    :class:`backend.api.middleware.CorrelationIdMiddleware` through
    ``structlog.contextvars`` — the same value every log line for the request
    carries (D-003), which is what makes a spend row joinable to the request
    that caused it. Returns ``None`` rather than inventing an id when nothing is
    bound (a Celery worker, a shell, a backfill script).

    Duplicated from :mod:`backend.db.audit` rather than imported: that module's
    resolver is private to it, and reaching into another module's private helper
    to avoid four lines is a worse dependency than the four lines.
    """
    bound = structlog.contextvars.get_contextvars().get("request_id")
    if isinstance(bound, str) and bound:
        return bound
    return None
