"""Order management for a paper-trading platform (Phase 11, P11.2).

This package is the order lifecycle that sits **beneath** any adapter: the state
machine, the content-derived idempotency keys, and the append-only persistence
that records every transition. P11.1 — the IBKR paper adapter — is a separate
task, blocked on B2, and nothing here depends on it or anticipates a seam for it
to plug into.

Paper-only, and why it is structural rather than absent
-------------------------------------------------------

Directive §1.1 and §9.5 put live trading permanently out of scope: "No code path
may place a real order. This is not configurable." Six independent facts make
that true here, and none of them is a switch that happens to be off:

1. **No transport exists.** This package opens no socket, constructs no client,
   reads no setting or environment variable, and imports nothing that could.
   ``record_order`` and ``append_transition`` take a database session and a value
   object. Nothing here can reach anything.
2. **No routing seam exists.** There is no adapter ``Protocol``, no abstract
   base class with a ``submit`` method, no registry, no factory, no
   dependency-injection point, and no ``Callable`` parameter that a venue client
   could be passed as. A live endpoint cannot be dropped in by configuration
   because there is nothing to configure it into — it would take new code and a
   new migration.
3. **"Live" has no representation.**
   :class:`~backend.execution.orders.ExecutionVenue` has exactly one member. A
   second one is a code change, not a config value.
4. **No venue argument exists anywhere.** No constructor or function in this
   package accepts one. The column takes a server default under a
   ``CHECK (venue = 'paper')``, so the value cannot be supplied by any writer,
   including raw SQL.
5. **A live fill cannot be recorded.**
   :class:`~backend.execution.orders.FillSource` has no live member and the
   column's CHECK admits only the two non-live literals — so a simulated fill is
   never mistakable for a broker fill (I3), and neither is mistakable for a real
   execution.
6. **It is checked on both sides of the database.** The write side is bound by
   the CHECK; the read side refuses a non-paper row anyway
   (:class:`~backend.execution.errors.NotPaperOrderError`) rather than trusting
   the write side.

``backend/tests/execution/test_paper_only.py`` proves points 1, 2, 3 and 5 by
tokenizing this package's own source, so the claim degrades into a test failure
rather than into prose if anyone weakens it.

Cost provenance
---------------

Every persisted fill carries ``fill_cost_basis = 'lower_bound'``
(:data:`~backend.execution.orders.PAPER_FILL_COST_BASIS`), constrained by the
schema. Paper and simulated fills are optimistic — they fill at the touch and
model no queue position — so slippage measured from them bounds the true cost
from below and is never an estimate of it (D-013). The label is on the row, so it
reaches every query, export and blotter rather than living in a document nobody
reads at query time.
"""

from backend.execution.errors import (
    ConcurrentTransitionError,
    DuplicateOrderError,
    ExecutionError,
    FillAccountingError,
    IdempotencyCollisionError,
    IllegalTransitionError,
    NotPaperOrderError,
    OrderNotFoundError,
    OrderValidationError,
    TerminalOrderError,
    TransitionChainError,
)
from backend.execution.idempotency import (
    IDEMPOTENCY_SCHEMA,
    idempotency_key,
    idempotency_preimage,
)
from backend.execution.lifecycle import (
    ILLEGAL_TRANSITIONS,
    INITIAL_STATE,
    TERMINAL_STATES,
    TRANSITIONS,
    OrderEvent,
    OrderState,
    Transition,
    TransitionRefusal,
    apply_event,
    fill_event,
    replay,
)
from backend.execution.orders import (
    PAPER_FILL_COST_BASIS,
    ExecutionVenue,
    FillReport,
    FillSource,
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
)
from backend.execution.store import (
    LoadedOrder,
    RecordedOrder,
    append_transition,
    load_order,
    load_order_by_key,
    record_order,
)

__all__ = [
    "IDEMPOTENCY_SCHEMA",
    "ILLEGAL_TRANSITIONS",
    "INITIAL_STATE",
    "PAPER_FILL_COST_BASIS",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "ConcurrentTransitionError",
    "DuplicateOrderError",
    "ExecutionError",
    "ExecutionVenue",
    "FillAccountingError",
    "FillReport",
    "FillSource",
    "IdempotencyCollisionError",
    "IllegalTransitionError",
    "LoadedOrder",
    "NotPaperOrderError",
    "OrderEvent",
    "OrderIntent",
    "OrderNotFoundError",
    "OrderState",
    "OrderType",
    "OrderValidationError",
    "RecordedOrder",
    "Side",
    "TerminalOrderError",
    "TimeInForce",
    "Transition",
    "TransitionChainError",
    "TransitionRefusal",
    "append_transition",
    "apply_event",
    "fill_event",
    "idempotency_key",
    "idempotency_preimage",
    "load_order",
    "load_order_by_key",
    "record_order",
    "replay",
]
