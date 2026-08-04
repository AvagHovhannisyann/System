"""The LLM cost governor: per-provider caps enforced **before** each call (P7.7, §6.5).

§5-P7 asks for "a hard daily spend cap per provider, enforced before the call,
not after", and §6.5 widens it to daily *and* monthly caps with the ceiling
behaviour configurable between halting and degrading to a cheaper model. That
one word — *before* — is the whole design. A component that totals spend after
the fact is an accounting system: correct, auditable, and useless as a control,
because by the time it notices, the money is gone.

So enforcement is not a function anyone is asked to remember to call. It is a
:class:`~backend.extraction.governor.guard.GovernedModelClient` wrapped around
the real model client, which means **there is no reachable path to the provider
that does not pass the cap check**. A caller cannot forget it; there is no
method that would let them.

The pieces
----------

:mod:`~backend.extraction.governor.caps`
    What the limits are, whose they are, and what happens at the ceiling. No
    default cap exists, and an empty :class:`~backend.extraction.governor.caps.CapBook`
    raises — B4 requires it and the reason is that a default cap is a number
    nobody chose.

:mod:`~backend.extraction.governor.estimate`
    Money, prices, and the **upper bound** a cap is checked against. The
    estimate precedes the response, so it bounds rather than predicts: output
    tokens at ``max_tokens``, input tokens at the UTF-8 byte length. Prices come
    from the provider catalog, which ships none, so an unpriced model refuses
    rather than being guessed at (I3).

:mod:`~backend.extraction.governor.ledger`
    Where headroom is **atomically** admitted. There is no separate "check", so
    the check-then-act race cannot be written: reserving is the check.

:mod:`~backend.extraction.governor.postgres`
    The durable, cross-process ledger — an advisory lock per provider around
    read-compare-insert, and the append-only audit trail (migration 0013).

:mod:`~backend.extraction.governor.governor`
    The decision: authorize, degrade, or refuse; then reconcile the reserved
    bound against what the call actually cost.

:mod:`~backend.extraction.governor.guard`
    The enforcement point on the call path.

:mod:`~backend.extraction.governor.errors`
    Every refusal, and why each is its own type.

Nothing in this package is configured (B4)
-------------------------------------------

There is no cap and no price in this repository. That is deliberate: B4 leaves
the provider keys, the negotiated rates and the approved spend with a human, and
inventing any of them here would produce a governor that looks like it works.
Constructing one therefore requires the operator to supply both a
:class:`~backend.extraction.governor.caps.CapBook` and a
:class:`~backend.extraction.governor.estimate.PriceBook`, and every path through
the unconfigured state raises rather than defaults.

Nothing here touches a credential (I5). The governor decides *whether* a call
may be made; :mod:`backend.extraction.providers.registry` decides how it is
authenticated. No API key is in scope in any module of this package, so none can
reach a log line or an exception message.
"""

from backend.extraction.governor.caps import (
    CapBook,
    CapPolicy,
    ProviderCap,
    SpendWindow,
)
from backend.extraction.governor.errors import (
    CapNotConfiguredError,
    CapsNotConfiguredError,
    CurrencyMismatchError,
    DegradationUnavailableError,
    GovernorConfigurationError,
    GovernorError,
    LedgerIntegrityError,
    ModelPriceUnknownError,
    RequestExceedsTierCeilingError,
    SpendCapExceededError,
    ThroughputInfeasibleError,
    ThroughputLimitsNotConfiguredError,
    ThroughputLimitUnknownError,
)
from backend.extraction.governor.estimate import (
    CATALOG_PRICES,
    REQUEST_FRAMING_TOKENS,
    CallEstimate,
    Money,
    PriceBook,
    TokenPrice,
    actual_call_cost,
    bound_input_tokens,
    estimate_call_cost,
)
from backend.extraction.governor.governor import Authorization, CostGovernor
from backend.extraction.governor.guard import GovernedCompletion, GovernedModelClient
from backend.extraction.governor.ledger import (
    CallOutcome,
    InMemorySpendLedger,
    Reservation,
    SpendEvent,
    SpendLedger,
    SpendRecord,
)
from backend.extraction.governor.postgres import PostgresSpendLedger
from backend.extraction.governor.throughput import (
    CATALOG_THROUGHPUT_LIMITS,
    BatchPlan,
    ModelThroughputLimit,
    ThroughputBook,
    ThroughputWindow,
    daily_call_capacity,
    plan_batch,
    require_request_fits,
)

__all__ = [
    "CATALOG_PRICES",
    "CATALOG_THROUGHPUT_LIMITS",
    "REQUEST_FRAMING_TOKENS",
    "Authorization",
    "BatchPlan",
    "CallEstimate",
    "CallOutcome",
    "CapBook",
    "CapNotConfiguredError",
    "CapPolicy",
    "CapsNotConfiguredError",
    "CostGovernor",
    "CurrencyMismatchError",
    "DegradationUnavailableError",
    "GovernedCompletion",
    "GovernedModelClient",
    "GovernorConfigurationError",
    "GovernorError",
    "InMemorySpendLedger",
    "LedgerIntegrityError",
    "ModelPriceUnknownError",
    "ModelThroughputLimit",
    "Money",
    "PostgresSpendLedger",
    "PriceBook",
    "ProviderCap",
    "RequestExceedsTierCeilingError",
    "Reservation",
    "SpendCapExceededError",
    "SpendEvent",
    "SpendLedger",
    "SpendRecord",
    "SpendWindow",
    "ThroughputBook",
    "ThroughputInfeasibleError",
    "ThroughputLimitUnknownError",
    "ThroughputLimitsNotConfiguredError",
    "ThroughputWindow",
    "TokenPrice",
    "actual_call_cost",
    "bound_input_tokens",
    "daily_call_capacity",
    "estimate_call_cost",
    "plan_batch",
    "require_request_fits",
]
