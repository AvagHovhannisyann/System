"""What the cost governor refuses, and why each refusal is its own type (P7.7).

Every failure here means *no provider call was made*. That is the point of the
module: a governor that reports a problem after the request left is an
accounting system, and the money is already gone. So each of these is raised on
the path to the call, before it, and the caller can rely on that.

Why these do **not** inherit from :class:`~backend.extraction.tasks.client.ModelCallError`
------------------------------------------------------------------------------------------

``ModelCallError`` means *the provider call failed* — transport, auth, rate
limit, timeout — and it is the class a retry loop is built around. A cap
refusal is not a provider failure: it is this system declining to spend, and it
will decline identically on every retry until either the window rolls over or a
human raises the cap. Making it a ``ModelCallError`` would hand it to the retry
loop, which would then hammer a cap check for as long as its backoff allowed
and log the result as a provider problem.

The consequence is stated rather than hidden: a caller that catches only
``ModelCallError`` around :meth:`~backend.extraction.tasks.client.ModelClient.complete`
will **not** catch a cap refusal, and the exception will propagate out of the
extraction pipeline. That is the intended behaviour — a backfill that hits its
cap should stop, loudly, not continue quietly on the documents that happened to
fit.

Configuration refusals versus runtime refusals
----------------------------------------------

:class:`GovernorConfigurationError` and its subclasses mean *the governor was
asked to operate on something nobody configured* — a provider with no cap, a
model with no price, a degrade policy with no substitute. Under B4 these are
the normal state of this repository, and they are errors rather than defaults
on purpose: a default cap is a number nobody chose, and a default price is a
fabricated one (I3).

:class:`SpendCapExceededError` is the runtime refusal — everything *was*
configured, and the call would have breached the configured limit.

Units: every monetary field on these exceptions is a
:class:`~backend.extraction.governor.estimate.Money`, which carries its own
currency. Nothing here reports a bare number (I4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.extraction.governor.caps import SpendWindow
    from backend.extraction.governor.estimate import Money
    from backend.extraction.providers.catalog import Provider

__all__ = [
    "CapNotConfiguredError",
    "CapsNotConfiguredError",
    "CurrencyMismatchError",
    "DegradationUnavailableError",
    "GovernorConfigurationError",
    "GovernorError",
    "LedgerIntegrityError",
    "ModelPriceUnknownError",
    "SpendCapExceededError",
]


class GovernorError(RuntimeError):
    """Base class for every governor refusal. Never raised directly.

    Deliberately a :class:`RuntimeError` and deliberately *not* a
    :class:`~backend.extraction.tasks.client.ModelCallError` — see the module
    docstring for what that costs and why it is worth it.
    """


class GovernorConfigurationError(GovernorError):
    """The governor was asked to operate on something nobody configured.

    Distinct from :class:`SpendCapExceededError`, which means the configuration
    exists and the call would breach it. The two need telling apart in a
    dashboard: one is "an operator has work to do", the other is "the control
    worked".
    """


class CapsNotConfiguredError(GovernorConfigurationError):
    """No spend cap is configured at all, so nothing may be spent (B4).

    Raised when a :class:`~backend.extraction.governor.caps.CapBook` is built
    empty. There is no default cap because a default cap is a number nobody
    chose, and B4 states the requirement directly: *the cost governor refuses to
    run without configured caps.*
    """


class CapNotConfiguredError(GovernorConfigurationError):
    """This provider has no cap, so calls to it are refused.

    A provider that is missing from the cap book is not "uncapped", it is
    "ungoverned", and the governor treats the two as the same thing: it refuses.
    Falling back to another provider's cap would spend one budget against
    another; falling back to no cap would spend without one.
    """


class ModelPriceUnknownError(GovernorConfigurationError):
    """No price is configured for this model, so its cost cannot be bounded (I3).

    The governor cannot enforce a cap it cannot measure against. Guessing a
    price would produce a cap that enforces an invented number — which is worse
    than no cap, because it looks like one. The catalog
    (:mod:`backend.extraction.providers.catalog`) deliberately ships no prices,
    so under B4 this is the expected refusal until an operator configures a
    :class:`~backend.extraction.governor.estimate.PriceBook`.
    """


class CurrencyMismatchError(GovernorConfigurationError):
    """Two monetary amounts in different currencies were combined or compared.

    Raised rather than converted. This system holds no exchange rate and has no
    business inventing one; a cap in one currency enforced against a price in
    another is a cap enforced against an unstated conversion (I4).
    """


class DegradationUnavailableError(GovernorError):
    """The cap says degrade, but there is no usable cheaper model to degrade to.

    Raised when the configured fallback is the model that already breached, or
    when its configured price is not strictly cheaper than the primary's for
    this call. A "fallback" that costs the same or more is not a degradation,
    it is a second attempt at the same spend under a different name.

    The call is **not** made. Degrading is a policy for spending less, and when
    it cannot spend less the policy has nothing to offer but a halt.
    """


class LedgerIntegrityError(GovernorError):
    """The spend ledger was asked to record something inconsistent with itself.

    Settling a reservation twice, settling one that was released, or settling
    one the ledger never issued. Each would corrupt the running total that every
    later cap check reads, so each is refused rather than absorbed.
    """


class SpendCapExceededError(GovernorError):
    """The call would breach a configured cap, so it was not made.

    The one runtime refusal. Carries the numbers that produced it so the API and
    the operator's gauges (§6.5) can render the decision without re-deriving it,
    and so the message itself states what was compared with what.

    Attributes:
        provider: whose cap bound.
        window: which window bound — daily or monthly.
        window_key: the window's label, ``YYYY-MM-DD`` for a day and
            ``YYYY-MM`` for a month, both in **UTC** (see
            :mod:`backend.extraction.governor.caps` for why UTC and what that
            does not mean).
        limit: the configured cap for that window, in its own currency.
        committed: spend already settled or reserved in that window, same
            currency.
        estimate: the **upper bound** on what this call would have cost, same
            currency. An upper bound, not an expectation — see
            :func:`~backend.extraction.governor.estimate.estimate_call_cost`.
        requested_model: the qualified model identifier the caller asked for.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        window: SpendWindow,
        window_key: str,
        limit: Money,
        committed: Money,
        estimate: Money,
        requested_model: str,
    ) -> None:
        """Build the refusal, message included.

        Args:
            provider: whose cap bound.
            window: which window bound.
            window_key: the window's UTC label.
            limit: the configured cap for that window.
            committed: spend already settled or reserved in that window.
            estimate: upper-bound cost of the refused call.
            requested_model: qualified model identifier that was asked for.
        """
        self.provider = provider
        self.window = window
        self.window_key = window_key
        self.limit = limit
        self.committed = committed
        self.estimate = estimate
        self.requested_model = requested_model
        headroom = limit - committed if limit >= committed else limit.zero_like()
        super().__init__(
            f"refusing the call to {requested_model!r}: it would cost at most {estimate} and "
            f"the {provider.value} {window.value} cap for {window_key} (UTC) is {limit}, of "
            f"which {committed} is already settled or reserved — headroom {headroom}. "
            "No request was made."
        )
