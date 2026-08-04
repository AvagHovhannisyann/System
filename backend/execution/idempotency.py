"""Content-addressed idempotency keys, so a retried submission cannot double-send.

The failure this prevents
-------------------------

A submission is retried after a timeout, a worker is restarted mid-cycle, a
broker event stream is replayed from an earlier offset. Each of those sends the
same order twice unless something refuses the second copy. The duplicate is not
a logged warning: it is a doubled position, and in the paper case a corrupted
experiment whose realised slippage no longer corresponds to the intended trade.

The key derives from content, never from a counter or a clock
-------------------------------------------------------------

:func:`idempotency_key` is a SHA-256 digest of the canonical JSON rendering of
the order's own fields — the instrument, the side, the quantity, the pricing
instruction, the slice coordinates, the venue, the rebalance date, and the four
components of the I2 reproducibility stamp. Nothing else goes in.

That exclusion is the whole design. A counter or a UUID would have to be
*remembered* across the retry, so the retry that lost its memory (the process
restart, the failed-over worker) would mint a new one and send the order again —
the key would be an identifier rather than an identity. A timestamp is worse: it
guarantees the two submissions differ. Deriving from content means the retry
recomputes the identical key from the identical instruction, with nothing to
remember and nothing to coordinate.

The stamp is part of the content, which gives a property worth stating: two runs
of the *same plan* — same commit, same config hash, same data version, same
seed, same rebalance date — produce the same keys, so re-running a rebalance is
absorbed rather than doubling the book. Change any input and the config hash or
data version changes with it, and the resulting orders are correctly new.

Uniqueness is enforced at the database, not here
------------------------------------------------

This module computes keys. It does not decide whether one is already taken —
that is a ``UNIQUE`` constraint on ``execution_order.idempotency_key``
(migration 0014). A Python-side check would be a read followed by a write with a
window between them, and two workers racing through that window both find the
key free and both insert. The database has no such window: the loser of the race
gets an ``IntegrityError``, whatever else is running and whatever process it is
in. See :func:`backend.execution.store.record_order` for what is done with it.

Schema versioning
-----------------

:data:`IDEMPOTENCY_SCHEMA` is the first field of the preimage. If the recipe ever
changes — a field added, a rendering altered — the version must change with it,
so keys computed under the old recipe and the new one cannot collide. The
preimage itself is stored beside the digest on every order row, which is what
makes a collision *detectable* rather than assumed: the digest is verifiable
against the text that produced it.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Final

from backend.execution.orders import ExecutionVenue, price_text
from backend.tracking.stamp import canonical_config_json

if TYPE_CHECKING:
    from backend.execution.orders import OrderIntent

__all__ = [
    "IDEMPOTENCY_SCHEMA",
    "idempotency_key",
    "idempotency_preimage",
    "key_of_preimage",
]

IDEMPOTENCY_SCHEMA: Final = "execution.order.v1"
"""Version of the preimage recipe, and the first field of every preimage.

Bump this whenever a field is added to, removed from, or rendered differently in
:func:`idempotency_preimage`. Two orders that differ only in the recipe that
hashed them must not be able to land on one key — see
:class:`~backend.execution.errors.IdempotencyCollisionError`, whose message says
exactly this to whoever trips it.
"""


def idempotency_preimage(intent: OrderIntent) -> str:
    """Render an order's identity as canonical JSON — the text that gets hashed.

    Canonicalisation is delegated to
    :func:`backend.tracking.stamp.canonical_config_json`, the same function that
    produces the config hash the stamp carries: keys sorted at every level,
    minimal separators, non-finite floats and non-string keys refused. Sorting is
    what makes the rendering an identity rather than a serialisation — the key
    must not depend on the order the fields were assembled in.

    Every value is rendered as a string, an integer or ``null``. No float
    appears: a binary float cannot represent a decimal price exactly, and a price
    that round-trips differently on two machines would give one order two keys.
    The limit price is rendered at the fixed schema scale, so ``Decimal("1.5")``
    and ``Decimal("1.50")`` — the same price — produce the same text.

    Deliberately **not** in the preimage: any timestamp, any counter, any
    database key, any process or host identifier, any retry attempt number. See
    the module docstring for why each of those would defeat the point.

    Args:
        intent: the order to render. Already validated by its own constructor.

    Returns:
        Canonical JSON text, UTF-8, no trailing newline.
    """
    stamp = intent.stamp
    content: dict[str, object] = {
        "schema": IDEMPOTENCY_SCHEMA,
        # Constant, and read from the enum rather than from a parameter: there is
        # no venue argument anywhere in this package (backend.execution.orders).
        "venue": ExecutionVenue.PAPER.value,
        "rebalance_date": intent.rebalance_date.isoformat(),
        "security_id": intent.security_id,
        "side": intent.side.value,
        "quantity_shares": intent.quantity_shares,
        "order_type": intent.order_type.value,
        "time_in_force": intent.time_in_force.value,
        "limit_price_usd": (
            None if intent.limit_price_usd is None else price_text(intent.limit_price_usd)
        ),
        "slice_index": intent.slice_index,
        "slice_count": intent.slice_count,
        # The I2 stamp, flattened. git_dirty is included because a dirty tree is
        # a code state no commit identifies, so two plans produced from two
        # different dirty trees are not the same computation and must not share
        # an order identity.
        "git_commit": stamp.git_commit,
        "git_dirty": stamp.git_dirty,
        "data_version": stamp.data_version,
        "config_hash": stamp.config_hash,
        "seed": stamp.seed,
    }
    return canonical_config_json(content)


def key_of_preimage(preimage: str) -> str:
    """Return the SHA-256 hex digest of a preimage.

    Split out from :func:`idempotency_key` so a stored preimage can be
    re-hashed and checked against the stored key without reconstructing the
    order — the verification that makes
    :class:`~backend.execution.errors.IdempotencyCollisionError` a detection
    rather than a guess.

    Args:
        preimage: canonical JSON text from :func:`idempotency_preimage`.

    Returns:
        Lowercase 64-character hex digest.
    """
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def idempotency_key(intent: OrderIntent) -> str:
    """Return the content-derived idempotency key for an order.

    Units: 64 lowercase hex characters. Deterministic across processes, hosts
    and restarts, because every input is a field of the order itself.

    Args:
        intent: the order to key.

    Returns:
        Lowercase 64-character SHA-256 hex digest of
        :func:`idempotency_preimage`.
    """
    return key_of_preimage(idempotency_preimage(intent))
