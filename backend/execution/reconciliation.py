"""Position and cash reconciliation for one execution cycle (P11.3).

What this module compares, and what it deliberately does not do
---------------------------------------------------------------

:func:`reconcile` is a **pure function of two snapshots it is given**. It fetches
nothing. There is no adapter here, no session, no clock read, and no parameter
through which a destination could be supplied — the same structural claim
:mod:`backend.execution.orders` makes, for the same reason (directive §1.1,
§9.5). P11.1, the paper adapter that would *produce* a statement snapshot, is
blocked on B2 and does not exist; designing this module against a snapshot it is
handed rather than one it retrieves is what lets reconciliation be built,
tested and reasoned about without it.

I3 rides on :class:`SnapshotOrigin`. A snapshot says where it came from, and the
only values are ``internal_ledger`` (our own books), ``paper_broker`` (a paper
account statement) and ``simulated`` (a constructed fixture). There is no member
for a real account, and every fixture in the test suite carries ``simulated`` —
so a fabricated statement is a *different value on the row* rather than a
distinction a reader has to reconstruct.

The tolerance, its units, and why this magnitude
------------------------------------------------

**Cash: 1 US cent, absolute** (:data:`CASH_TOLERANCE_USD`). Exact equality on
cash is wrong and "close enough" is worse, so the number has to be derived from
something rather than picked.

The *only* legitimate source of divergence between two correctly-maintained USD
cash balances is representation. A statement reports cash quantised to the minor
unit — two decimal places. Our own balance is carried at
:data:`~backend.execution.orders.PRICE_SCALE` (six) decimal places, because it is
built from quantities (exact integers) times prices (exact decimals at scale
six). Quantising a scale-six value to scale two moves it by at most **half a
cent**. One cent is two quantisation ticks: it strictly bounds one rounding step
with a factor-of-two margin, and it bounds nothing else.

That second half is the point. The tolerance must be *smaller than the smallest
real break*, or it stops being a representation allowance and becomes a
blindfold. The cheapest thing that can actually go wrong is larger than a cent by
orders of magnitude: the smallest commission on any IBKR tier is tens of cents,
one share of any name a price screen admits is dollars, and a dropped fill is
tens or hundreds of dollars. Nothing real fits under a cent.

**Positions: zero tolerance, and no parameter to widen it.** Share counts are
integers and are compared with ``==``. There is no arithmetic that legitimately
produces an off-by-one share, so a tolerance on quantities could only ever hide a
break. The asymmetry with cash is deliberate and is the reason the two are not
handled by one "tolerance" knob.

**A ceiling on the tolerance itself** (:data:`MAX_CASH_TOLERANCE_USD`, 5 cents).
The tolerance is an argument so a caller can tighten it, and a bound exists so
nobody can loosen it past the point where the derivation above still holds. Five
cents is ten quantisation ticks; above that a tolerance is absorbing an event
rather than a rounding step, and :func:`reconcile` refuses rather than obliges.

**Contemporaneity: 300 seconds** (:data:`MAX_SNAPSHOT_SKEW_SECONDS`). The mirror
of a too-wide tolerance is a comparison that manufactures breaks. Two snapshots
of the same book taken far enough apart differ because of activity between them,
and a "mismatch" that means "a fill landed in the gap" trains an operator to
ignore the alarm. Snapshots more than five minutes apart are refused outright —
they are not evidence about one book, and refusing produces no reconciliation for
the cycle, which the kill switch treats as an unknown condition and halts on
(:mod:`backend.execution.killswitch`). Failing to a halt is the correct direction.

Determinism and re-runnability
------------------------------

A break that cannot be re-examined afterwards cannot be investigated, so this
module is built so the verdict is reproducible from stored bytes alone:

- :func:`reconcile` reads no clock, no environment and no database, and its
  output ordering is fixed (:func:`_finding_sort_key`) rather than inherited from
  mapping iteration order.
- Every snapshot renders to canonical JSON (:meth:`PositionSnapshot.as_json`) and
  digests to SHA-256 over exactly that text, so the stored payload *is* the
  preimage of the stored digest and neither can drift from the other.
- :attr:`ReconciliationResult.result_digest` covers the cycle, the tolerance, both
  snapshot digests and every finding — and deliberately **excludes the I2 stamp**.
  Re-running a stored break at a later commit must reproduce the same verdict; if
  the stamp were in the digest, every re-run would differ by construction and the
  property would be untestable. The stamp is stored beside the digest instead
  (I2), so which code produced the original verdict is still recorded.
- :func:`rerun` recomputes a stored reconciliation from its stored snapshots and
  raises :class:`~backend.execution.errors.ReconciliationReplayError` if the
  digest moves. Silent divergence between the stored verdict and a re-derived one
  is the failure that makes an audit trail worthless.

Units
-----

Positions are **signed whole shares** (positive long, negative short). Cash is
**US dollars**, carried as :class:`decimal.Decimal` at
:data:`~backend.execution.orders.PRICE_SCALE` decimal places and never as
``float``. Timestamps are timezone-aware instants.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa

from backend.db.models import ExecutionReconciliation as ExecutionReconciliationRow
from backend.execution.errors import (
    ReconciliationMismatchError,
    ReconciliationReplayError,
    SnapshotValidationError,
)
from backend.execution.orders import PRICE_SCALE
from backend.tracking.stamp import ReproducibilityStamp, canonical_config_json

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "CASH_TOLERANCE_USD",
    "MAX_CASH_TOLERANCE_USD",
    "MAX_SNAPSHOT_SKEW_SECONDS",
    "RESULT_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "Finding",
    "MismatchKind",
    "PositionSnapshot",
    "ReconciliationResult",
    "Severity",
    "SnapshotOrigin",
    "StoredReconciliation",
    "load_reconciliation",
    "reconcile",
    "record_reconciliation",
    "require_matched",
    "rerun",
    "snapshot_of_json",
]

SNAPSHOT_SCHEMA: Final = "execution.reconciliation.snapshot.v1"
"""Version of the snapshot rendering, and the first field of every snapshot payload.

Bump it whenever :meth:`PositionSnapshot.as_json` changes shape. Two snapshots
rendered under two recipes must not be able to land on one digest, or a stored
break would compare against a preimage that no longer means what it meant.
"""

RESULT_SCHEMA: Final = "execution.reconciliation.result.v1"
"""Version of the result rendering, and the first field of every result payload."""

CASH_TOLERANCE_USD: Final = Decimal("0.01")
"""Allowed absolute cash divergence, in **US dollars** (one cent).

Derived, not chosen: a statement renders cash at two decimal places while our
balance is carried at six, so one quantisation step moves the value by at most
half a cent, and one cent bounds that with a factor of two to spare. It bounds
nothing else — the cheapest real break (a commission, a single share, a dropped
fill) is larger by orders of magnitude. See the module docstring.

A difference **equal** to the tolerance is within it; only a strictly larger one
is a break.
"""

MAX_CASH_TOLERANCE_USD: Final = Decimal("0.05")
"""Largest tolerance :func:`reconcile` will accept, in **US dollars** (five cents).

Ten quantisation ticks. A caller may tighten the tolerance freely — zero is
allowed and is the strictest possible setting — but cannot widen it past the
point where the representation argument that justifies it still holds. Above five
cents a tolerance is absorbing an event, and there is no magnitude of event this
system is willing to call rounding.
"""

MAX_SNAPSHOT_SKEW_SECONDS: Final = 300
"""Largest gap between the two snapshots' observation instants, in **seconds**.

Two observations of one book taken far apart differ because of activity between
them. A "mismatch" that means "a fill landed in the gap" is a false positive, and
false positives are how a real alarm gets ignored. Snapshots further apart than
this are refused: no reconciliation is produced for the cycle, which the kill
switch reads as an unknown condition and halts on.
"""

_MONEY_QUANTUM: Final = Decimal(1).scaleb(-PRICE_SCALE)
"""One unit at the schema's price scale — the smallest representable cash step."""


class SnapshotOrigin(StrEnum):
    """Where a snapshot came from. No member denotes a real-money account.

    I3 at the type level, in the same shape as
    :class:`~backend.execution.orders.FillSource`:

    - ``INTERNAL_LEDGER`` — our own books, derived from the order and transition
      log. The only value :func:`reconcile` accepts on the internal side.
    - ``PAPER_BROKER`` — a statement from a paper account (P11.1, blocked on B2).
      No capital moves and nothing here can produce one today.
    - ``SIMULATED`` — a constructed snapshot. Every fixture in the test suite
      carries this value, which is what makes a fabricated statement a distinct
      value on the row rather than something a reader has to infer.

    There is deliberately no fourth member, and migration 0016 restates the
    admissible values as CHECK constraints so a writer bypassing Python is bound
    too.
    """

    INTERNAL_LEDGER = "internal_ledger"
    PAPER_BROKER = "paper_broker"
    SIMULATED = "simulated"


REPORTABLE_ORIGINS: Final[frozenset[SnapshotOrigin]] = frozenset(
    {SnapshotOrigin.PAPER_BROKER, SnapshotOrigin.SIMULATED}
)
"""Origins admissible on the *reported* side of a reconciliation.

Our own ledger cannot stand in for the statement it is being checked against: a
reconciliation of a snapshot with itself always passes and proves nothing, which
is the most comfortable way for this control to become decorative.
"""


class MismatchKind(StrEnum):
    """The kinds of disagreement a reconciliation can find.

    The taxonomy exists because the *operational response* differs, and a single
    "mismatch" verdict would erase exactly the distinctions an operator needs:

    - ``POSITION_UNKNOWN_TO_US`` — the statement holds shares our books do not.
      **The most serious case.** There is capital at the venue that our risk
      model, our optimizer, our exposure report and our drawdown monitor do not
      know exists. It is not sized, not hedged, not scheduled for exit, and every
      aggregate we publish is wrong by an amount we cannot compute. The
      correction is a decision (adopt it or flatten it), never an edit.
    - ``POSITION_UNKNOWN_TO_STATEMENT`` — our books hold shares the statement does
      not. Serious, and serious *differently*: no unmanaged capital exists, but
      our books overstate the account, so the next order computed from them is
      wrong in a specific and dangerous way — it can instruct a sale of shares
      that are not there.
    - ``POSITION_QUANTITY_DISAGREES`` — both sides name the instrument and the
      counts differ. A partial fill booked once and reported twice, a missed
      cancel, a corporate action applied on one side only.
    - ``POSITION_FLAT_BUT_ONE_SIDE_SILENT`` — both sides agree there is no
      exposure, but one said so explicitly and the other simply did not mention
      the instrument. **Not a break**, and recorded anyway, because a missing
      position and a zero position are different facts. An explicit zero is a
      statement ("I looked; there is nothing"); an absence is silence ("I did not
      say"). Coercing absence to zero would make a truncated statement — a feed
      that dropped a leg — indistinguishable from a confirmed flat book, and that
      is the failure mode a reconciliation exists to catch.
    - ``CASH_DISAGREES`` — the cash balances differ by more than the tolerance.
    """

    POSITION_UNKNOWN_TO_US = "position_unknown_to_us"
    POSITION_UNKNOWN_TO_STATEMENT = "position_unknown_to_statement"
    POSITION_QUANTITY_DISAGREES = "position_quantity_disagrees"
    POSITION_FLAT_BUT_ONE_SIDE_SILENT = "position_flat_but_one_side_silent"
    CASH_DISAGREES = "cash_disagrees"


class Severity(StrEnum):
    """Whether a finding halts the cycle.

    ``BREAK`` halts. ``OBSERVATION`` does not, and exists so a fact worth
    recording is not forced to choose between "raise an alarm" and "be discarded"
    — the choice that quietly deletes evidence.
    """

    BREAK = "break"
    OBSERVATION = "observation"


BREAKING_KINDS: Final[frozenset[MismatchKind]] = frozenset(
    {
        MismatchKind.POSITION_UNKNOWN_TO_US,
        MismatchKind.POSITION_UNKNOWN_TO_STATEMENT,
        MismatchKind.POSITION_QUANTITY_DISAGREES,
        MismatchKind.CASH_DISAGREES,
    }
)
"""Kinds whose severity is :attr:`Severity.BREAK`.

Declared as data and checked against every member of :class:`MismatchKind` at
import, so a kind added later cannot acquire a silent default severity — the same
device :mod:`backend.execution.lifecycle` uses for unclassified transitions.
"""


def _severity_of(kind: MismatchKind) -> Severity:
    """Return the severity of a mismatch kind.

    Args:
        kind: the mismatch kind to classify.

    Returns:
        :attr:`Severity.BREAK` for a kind in :data:`BREAKING_KINDS`, otherwise
        :attr:`Severity.OBSERVATION`.
    """
    return Severity.BREAK if kind in BREAKING_KINDS else Severity.OBSERVATION


def _verify_severity_table() -> None:
    """Check that every mismatch kind is classified, and that breaks exist.

    Raises:
        SnapshotValidationError: if :data:`BREAKING_KINDS` names something that is
            not a mismatch kind, or if it is empty. Raised at import: a
            reconciliation whose every finding is an observation is a control that
            cannot fail, which is worse than no control because it is trusted.
    """
    unknown = BREAKING_KINDS - set(MismatchKind)
    if unknown:
        msg = f"BREAKING_KINDS names non-members: {sorted(item for item in unknown)}"
        raise SnapshotValidationError(msg)
    if not BREAKING_KINDS:
        msg = (
            "no mismatch kind is classified as a break, so no reconciliation could ever "
            "halt anything; a control that cannot fire is worse than no control"
        )
        raise SnapshotValidationError(msg)


_verify_severity_table()


def _require(condition: bool, message: str) -> None:
    """Raise :class:`SnapshotValidationError` when ``condition`` is false.

    Args:
        condition: the requirement that must hold.
        message: what the caller got wrong, quoted verbatim in the error.

    Raises:
        SnapshotValidationError: when ``condition`` is false.
    """
    if not condition:
        raise SnapshotValidationError(message)


def _require_whole(name: str, value: object) -> None:
    """Refuse a share count that is not a plain ``int``.

    ``bool`` is refused explicitly: ``True`` is an ``int`` and would silently
    reconcile as one share. The parameter is typed ``object`` because this guard
    also runs over values read back out of JSONB, where the annotation is a claim
    rather than a guarantee.

    Args:
        name: field name, for the message.
        value: the value to check.

    Raises:
        SnapshotValidationError: if ``value`` is a ``bool`` or not an ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be a whole number of shares (bool is refused), got {value!r}"
        raise SnapshotValidationError(msg)


def _as_int(value: object) -> int:
    """Narrow a value already checked by :func:`_require_whole` to ``int``.

    Args:
        value: the checked value.

    Returns:
        The value as an ``int``.

    Raises:
        SnapshotValidationError: if the value is not an ``int`` after all — which
            can only happen if a caller skipped the check, and is worth failing on
            rather than coercing.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"expected a whole number, got {value!r}"
        raise SnapshotValidationError(msg)
    return value


def _text_field(payload: Mapping[str, object], field: str) -> str:
    """Read one required string field out of a stored payload.

    Args:
        payload: the stored payload.
        field: the field name.

    Returns:
        The field's value.

    Raises:
        SnapshotValidationError: if the field is absent or is not a string. A
            payload that cannot be read is reported rather than partially
            interpreted: a half-read snapshot reconciles against a book that never
            existed.
    """
    _require(field in payload, f"snapshot payload is missing {field!r}")
    value = payload[field]
    _require(
        isinstance(value, str),
        f"snapshot payload's {field} must be a string, got {type(value).__name__}",
    )
    return value if isinstance(value, str) else ""


def _require_money(name: str, amount: Decimal) -> None:
    """Refuse a cash amount that is not exactly representable at the schema scale.

    Args:
        name: field name, for the message.
        amount: the amount to check, in **US dollars**.

    Raises:
        SnapshotValidationError: if the amount is not a :class:`~decimal.Decimal`,
            is non-finite, or carries more than
            :data:`~backend.execution.orders.PRICE_SCALE` decimal places.
            Non-finite is refused here rather than compared later because a
            ``Decimal("NaN")`` makes every comparison either false or an
            exception — a reconciliation against a NaN balance would report a
            clean book, which is a fail-open no control may have.
    """
    checked: object = amount
    if not isinstance(checked, Decimal):
        msg = f"{name} must be a Decimal in US dollars, got {type(checked).__name__}"
        raise SnapshotValidationError(msg)
    _require(
        amount.is_finite(),
        f"{name}={amount!r} is not finite; a non-finite balance compares false against every "
        f"tolerance and would report a clean book",
    )
    exponent = amount.as_tuple().exponent
    _require(
        isinstance(exponent, int) and exponent >= -PRICE_SCALE,
        f"{name}={amount!r} carries more than {PRICE_SCALE} decimal places, which the "
        f"Numeric(18, {PRICE_SCALE}) column cannot store; the persisted value would then "
        f"disagree with the digest computed from the submitted one",
    )


def _money_text(amount: Decimal) -> str:
    """Render a US dollar amount at the fixed schema scale, for hashing and storage.

    ``Decimal("1.5")`` and ``Decimal("1.50")`` are the same amount and must
    produce the same text, or one balance would digest to two values and a stored
    reconciliation could not be re-run.

    Args:
        amount: a finite amount in **US dollars** with at most
            :data:`~backend.execution.orders.PRICE_SCALE` decimal places, already
            checked by :func:`_require_money`.

    Returns:
        Plain decimal notation at exactly
        :data:`~backend.execution.orders.PRICE_SCALE` places, e.g. ``"-12.500000"``.
    """
    return f"{amount.quantize(_MONEY_QUANTUM):f}"


def _require_aware(name: str, moment: dt.datetime) -> None:
    """Refuse a naive timestamp.

    Args:
        name: field name, for the message.
        moment: the timestamp to check.

    Raises:
        SnapshotValidationError: if ``moment`` carries no timezone. Two snapshots
            are compared for contemporaneity, and a naive instant is a number
            whose meaning depends on the machine that wrote it.
    """
    checked: object = moment
    if not isinstance(checked, dt.datetime):
        msg = f"{name} must be a datetime, got {type(checked).__name__}"
        raise SnapshotValidationError(msg)
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        msg = (
            f"{name}={moment!r} is timezone-naive; two snapshots are compared for "
            f"contemporaneity and an instant whose meaning depends on the writer's locale "
            f"cannot be compared with one that does not"
        )
        raise SnapshotValidationError(msg)


def _digest(payload: Mapping[str, object]) -> str:
    """Return the SHA-256 hex digest of a payload's canonical JSON.

    Canonicalisation is :func:`backend.tracking.stamp.canonical_config_json` — the
    same function that produces the I2 config hash: keys sorted at every level,
    minimal separators, non-finite floats and non-string keys refused.

    Args:
        payload: a JSON-serialisable mapping.

    Returns:
        Lowercase 64-character hex digest.
    """
    return hashlib.sha256(canonical_config_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """One observation of a book: what is held, how much cash, when, and by whom.

    Immutable and canonicalised on construction — ``positions`` is replaced by a
    read-only mapping in ascending ``security_id`` order, so two snapshots built
    from differently-ordered inputs render to identical JSON and identical
    digests. A digest that depended on insertion order would be a serialisation
    rather than an identity.

    Units: ``positions`` values are **signed whole shares** (positive long,
    negative short); ``cash_usd`` is **US dollars**; ``observed_at`` is a
    timezone-aware instant.

    Attributes:
        origin: where this observation came from. No value denotes a real-money
            account (:class:`SnapshotOrigin`).
        observed_at: when the book looked like this, timezone-aware.
        cash_usd: the cash balance in US dollars.
        positions: ``security_id`` to signed whole shares. A ``security_id`` may
            map to ``0``, and that is not the same fact as being absent — see
            :attr:`MismatchKind.POSITION_FLAT_BUT_ONE_SIDE_SILENT`.
    """

    origin: SnapshotOrigin
    observed_at: dt.datetime
    cash_usd: Decimal
    positions: Mapping[int, int]

    def __post_init__(self) -> None:
        """Validate every field and canonicalise ``positions`` into sorted order.

        Raises:
            SnapshotValidationError: on an unknown origin, a naive or non-datetime
                ``observed_at``, a cash balance that is not a finite decimal at the
                schema scale, a non-positive or non-integral ``security_id``, or a
                share count that is not a plain ``int``.
        """
        origin: object = self.origin
        _require(
            isinstance(origin, SnapshotOrigin),
            f"origin must be a SnapshotOrigin, got {type(origin).__name__}",
        )
        _require_aware("observed_at", self.observed_at)
        _require_money("cash_usd", self.cash_usd)
        supplied: object = self.positions
        _require(
            hasattr(supplied, "items"),
            f"positions must be a mapping of security_id to shares, got {type(supplied).__name__}",
        )
        canonical: dict[int, int] = {}
        for security_id, shares in sorted(self.positions.items()):
            _require_whole("security_id", security_id)
            _require(
                security_id > 0,
                f"security_id must be positive, got {security_id}",
            )
            _require_whole(f"positions[{security_id}]", shares)
            canonical[security_id] = shares
        object.__setattr__(self, "positions", MappingProxyType(canonical))

    def as_json(self) -> dict[str, object]:
        """Render this snapshot as the canonical payload its digest is taken over.

        The instant is normalised to UTC before rendering, so two equal instants
        written in different zones produce one payload and one digest.

        Returns:
            A JSON-serialisable mapping. Positions render as a list of
            ``[security_id, shares]`` pairs in ascending ``security_id`` order
            rather than as an object, because JSON object keys would have to be
            strings and would then sort lexicographically (``"10" < "2"``).
        """
        return {
            "schema": SNAPSHOT_SCHEMA,
            "origin": self.origin.value,
            "observed_at": self.observed_at.astimezone(dt.UTC).isoformat(),
            "cash_usd": _money_text(self.cash_usd),
            "positions": [[security_id, shares] for security_id, shares in self.positions.items()],
        }

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of :meth:`as_json`, 64 lowercase hex characters."""
        return _digest(self.as_json())


def snapshot_of_json(payload: Mapping[str, object]) -> PositionSnapshot:
    """Rebuild a snapshot from a stored payload, for re-running a stored break.

    The inverse of :meth:`PositionSnapshot.as_json`, and the reason a stored
    reconciliation is investigable at all: the row holds the payload, this rebuilds
    the value object, and :func:`reconcile` re-derives the verdict from it.

    Args:
        payload: the stored JSON payload, exactly as
            :meth:`PositionSnapshot.as_json` produced it.

    Returns:
        The reconstructed snapshot.

    Raises:
        SnapshotValidationError: if the payload names a different schema version,
            is missing a field, or holds a value the snapshot's own validation
            refuses. A payload that cannot be rebuilt is reported rather than
            partially interpreted: a half-read snapshot reconciles against a book
            that never existed.
    """
    _require(
        payload.get("schema") == SNAPSHOT_SCHEMA,
        f"snapshot payload declares schema {payload.get('schema')!r}, not {SNAPSHOT_SCHEMA!r}; "
        f"a payload rendered under another recipe does not mean what this one means",
    )
    origin_value = _text_field(payload, "origin")
    _require(
        origin_value in {member.value for member in SnapshotOrigin},
        f"snapshot payload declares origin {origin_value!r}, which is not a known origin",
    )
    observed_text = _text_field(payload, "observed_at")
    cash_text = _text_field(payload, "cash_usd")
    rows: object = payload.get("positions")
    _require(
        isinstance(rows, list),
        f"snapshot payload's positions must be a list of pairs, got {type(rows).__name__}",
    )
    positions: dict[int, int] = {}
    entries: list[object] = list(rows) if isinstance(rows, list) else []
    for entry in entries:
        _require(
            isinstance(entry, list) and len(entry) == 2,
            f"snapshot payload's positions entry {entry!r} is not a [security_id, shares] pair",
        )
        pair: list[object] = list(entry) if isinstance(entry, list) else []
        security_id, shares = pair[0], pair[1]
        _require_whole("security_id", security_id)
        _require_whole("shares", shares)
        positions[_as_int(security_id)] = _as_int(shares)
    try:
        observed_at = dt.datetime.fromisoformat(observed_text)
        cash_usd = Decimal(cash_text)
    except (ValueError, ArithmeticError) as exc:
        msg = f"snapshot payload could not be parsed: {exc}"
        raise SnapshotValidationError(msg) from exc
    return PositionSnapshot(
        origin=SnapshotOrigin(origin_value),
        observed_at=observed_at,
        cash_usd=cash_usd,
        positions=positions,
    )


@dataclass(frozen=True, slots=True)
class Finding:
    """One disagreement, or one recorded non-disagreement, between two snapshots.

    Both sides' raw values are carried, including the distinction between "said
    zero" (``0``) and "did not mention it" (``None``). That distinction is the
    whole reason this record is not a pair of integers: collapsing absence to zero
    would erase a truncated statement into a confirmed flat book.

    Units: ``internal_shares`` and ``reported_shares`` are **signed whole shares**;
    every ``*_usd`` field is **US dollars**.

    Attributes:
        kind: the classification (:class:`MismatchKind`).
        severity: whether this halts the cycle (:class:`Severity`).
        security_id: the instrument, or ``None`` for a cash finding.
        internal_shares: what our books said — ``None`` when they did not name the
            instrument at all.
        reported_shares: what the statement said — ``None`` when it did not name
            the instrument at all.
        internal_cash_usd: our cash balance, on a cash finding only.
        reported_cash_usd: the statement's cash balance, on a cash finding only.
        difference_usd: ``internal - reported`` on a cash finding only. Signed, so
            the direction of the error is on the record.
        tolerance_usd: the tolerance in force when this finding was made, copied
            onto the record so a tolerance changed tomorrow cannot rewrite
            today's verdict.
        detail: prose naming what disagrees and why it matters.
    """

    kind: MismatchKind
    severity: Severity
    security_id: int | None
    internal_shares: int | None
    reported_shares: int | None
    internal_cash_usd: Decimal | None
    reported_cash_usd: Decimal | None
    difference_usd: Decimal | None
    tolerance_usd: Decimal | None
    detail: str

    @property
    def is_break(self) -> bool:
        """Whether this finding halts the cycle."""
        return self.severity is Severity.BREAK

    def as_json(self) -> dict[str, object]:
        """Render this finding as the canonical payload the result digest covers.

        Returns:
            A JSON-serialisable mapping. Decimals render as fixed-scale strings,
            never as floats: a binary float has no exact decimal rendering and a
            digest taken over one would not survive a round trip.
        """
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "security_id": self.security_id,
            "internal_shares": self.internal_shares,
            "reported_shares": self.reported_shares,
            "internal_cash_usd": (
                None if self.internal_cash_usd is None else _money_text(self.internal_cash_usd)
            ),
            "reported_cash_usd": (
                None if self.reported_cash_usd is None else _money_text(self.reported_cash_usd)
            ),
            "difference_usd": (
                None if self.difference_usd is None else _money_text(self.difference_usd)
            ),
            "tolerance_usd": (
                None if self.tolerance_usd is None else _money_text(self.tolerance_usd)
            ),
            "detail": self.detail,
        }


def _finding_sort_key(finding: Finding) -> tuple[int, int, str]:
    """Return the total order findings are emitted in.

    Cash first (it describes the whole account), then positions in ascending
    ``security_id``, then by kind for the pathological case of two findings on one
    instrument. Fixed rather than inherited from mapping iteration, because the
    result digest is taken over the ordered list and a digest that depends on
    iteration order is not reproducible.

    Args:
        finding: the finding to key.

    Returns:
        A sort key.
    """
    return (
        0 if finding.security_id is None else 1,
        -1 if finding.security_id is None else finding.security_id,
        finding.kind.value,
    )


def _position_finding(
    *, security_id: int, internal_shares: int | None, reported_shares: int | None
) -> Finding | None:
    """Classify one instrument's two reported quantities.

    The classification is a total function of the pair, and the four position
    kinds are exactly its cases. ``None`` means "the side did not name this
    instrument"; ``0`` means "the side named it and said flat". They are kept
    apart all the way to the record.

    Args:
        security_id: the instrument.
        internal_shares: our books' count, or ``None`` if absent from them.
        reported_shares: the statement's count, or ``None`` if absent from it.

    Returns:
        A :class:`Finding`, or ``None`` when the two sides agree and both named
        the instrument — the only case with nothing to record.
    """
    ours = 0 if internal_shares is None else internal_shares
    theirs = 0 if reported_shares is None else reported_shares
    silent = internal_shares is None or reported_shares is None
    if ours == theirs:
        if not silent:
            return None
        return Finding(
            kind=MismatchKind.POSITION_FLAT_BUT_ONE_SIDE_SILENT,
            severity=_severity_of(MismatchKind.POSITION_FLAT_BUT_ONE_SIDE_SILENT),
            security_id=security_id,
            internal_shares=internal_shares,
            reported_shares=reported_shares,
            internal_cash_usd=None,
            reported_cash_usd=None,
            difference_usd=None,
            tolerance_usd=None,
            detail=(
                f"security_id={security_id}: both sides are flat, but one named the "
                f"instrument and the other did not. Not a break — and recorded, because an "
                f"explicit zero is a statement and an absence is silence, and a statement "
                f"that dropped a leg looks exactly like a confirmed flat book once the two "
                f"are collapsed together"
            ),
        )
    if ours == 0:
        kind = MismatchKind.POSITION_UNKNOWN_TO_US
        detail = (
            f"security_id={security_id}: the statement holds {theirs} shares that our books "
            f"do not (our side "
            f"{'did not name the instrument' if internal_shares is None else 'says flat'}). "
            f"This is the most serious break: capital is at the venue that the risk model, "
            f"the optimizer and the drawdown monitor do not know exists, so it is unsized, "
            f"unhedged and unscheduled, and every exposure number we publish is wrong by an "
            f"amount we cannot compute"
        )
    elif theirs == 0:
        kind = MismatchKind.POSITION_UNKNOWN_TO_STATEMENT
        detail = (
            f"security_id={security_id}: our books hold {ours} shares the statement does not "
            f"(the statement "
            f"{'did not name the instrument' if reported_shares is None else 'says flat'}). "
            f"No unmanaged capital exists, but our books overstate the account, so the next "
            f"order computed from them can instruct a sale of shares that are not there"
        )
    else:
        kind = MismatchKind.POSITION_QUANTITY_DISAGREES
        detail = (
            f"security_id={security_id}: our books hold {ours} shares and the statement "
            f"holds {theirs}, a difference of {ours - theirs}"
            f"{' with the sign reversed' if (ours > 0) != (theirs > 0) else ''}"
        )
    return Finding(
        kind=kind,
        severity=_severity_of(kind),
        security_id=security_id,
        internal_shares=internal_shares,
        reported_shares=reported_shares,
        internal_cash_usd=None,
        reported_cash_usd=None,
        difference_usd=None,
        tolerance_usd=None,
        detail=detail,
    )


def _cash_finding(
    *, internal_cash_usd: Decimal, reported_cash_usd: Decimal, tolerance_usd: Decimal
) -> Finding | None:
    """Compare two cash balances against the tolerance.

    Args:
        internal_cash_usd: our balance, in **US dollars**.
        reported_cash_usd: the statement's balance, in **US dollars**.
        tolerance_usd: the allowed absolute divergence, in **US dollars**.

    Returns:
        A :class:`Finding` when the absolute difference **exceeds** the tolerance,
        otherwise ``None``. A difference exactly equal to the tolerance is within
        it; both sides of that boundary are pinned by tests.
    """
    difference = internal_cash_usd - reported_cash_usd
    if abs(difference) <= tolerance_usd:
        return None
    return Finding(
        kind=MismatchKind.CASH_DISAGREES,
        severity=_severity_of(MismatchKind.CASH_DISAGREES),
        security_id=None,
        internal_shares=None,
        reported_shares=None,
        internal_cash_usd=internal_cash_usd,
        reported_cash_usd=reported_cash_usd,
        difference_usd=difference,
        tolerance_usd=tolerance_usd,
        detail=(
            f"cash: our books say {_money_text(internal_cash_usd)} USD and the statement says "
            f"{_money_text(reported_cash_usd)} USD, a difference of {_money_text(difference)} "
            f"USD against a tolerance of {_money_text(tolerance_usd)} USD. The tolerance bounds "
            f"one quantisation step between a two-decimal statement and a six-decimal ledger "
            f"and nothing else, so a difference above it is an event, not a rounding"
        ),
    )


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """The verdict of one cycle's reconciliation, and everything needed to re-derive it.

    Attributes:
        cycle_id: the execution cycle this reconciliation belongs to. A result
            from another cycle is not evidence about this one, and the kill switch
            checks that (:mod:`backend.execution.killswitch`).
        internal: our own snapshot.
        reported: the statement snapshot.
        cash_tolerance_usd: the tolerance in force, in **US dollars**, carried on
            the result so the verdict states its own threshold.
        findings: every finding, in :func:`_finding_sort_key` order.
        stamp: the I2 reproducibility stamp of the run that produced the verdict.
    """

    cycle_id: str
    internal: PositionSnapshot
    reported: PositionSnapshot
    cash_tolerance_usd: Decimal
    findings: tuple[Finding, ...]
    stamp: ReproducibilityStamp

    @property
    def breaks(self) -> tuple[Finding, ...]:
        """The findings whose severity is :attr:`Severity.BREAK`, in result order."""
        return tuple(finding for finding in self.findings if finding.is_break)

    @property
    def matched(self) -> bool:
        """Whether the two snapshots reconcile — no breaks, observations allowed."""
        return not self.breaks

    def as_json(self) -> dict[str, object]:
        """Render the payload :attr:`result_digest` is taken over.

        The I2 stamp is deliberately absent. A stored break must re-derive to the
        same verdict when re-run at a later commit; including the stamp would make
        every re-run differ by construction and the property untestable. The stamp
        is persisted in its own columns instead.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "schema": RESULT_SCHEMA,
            "cycle_id": self.cycle_id,
            "cash_tolerance_usd": _money_text(self.cash_tolerance_usd),
            "internal_digest": self.internal.digest,
            "reported_digest": self.reported.digest,
            "findings": [finding.as_json() for finding in self.findings],
        }

    @property
    def result_digest(self) -> str:
        """SHA-256 hex digest of :meth:`as_json`, 64 lowercase hex characters."""
        return _digest(self.as_json())


def reconcile(
    *,
    cycle_id: str,
    internal: PositionSnapshot,
    reported: PositionSnapshot,
    stamp: ReproducibilityStamp,
    cash_tolerance_usd: Decimal = CASH_TOLERANCE_USD,
) -> ReconciliationResult:
    """Compare our book against a statement and return the cycle's verdict.

    Pure: reads no clock, no environment and no database, and takes no parameter
    through which anything could be fetched. The classification is a total
    function over the union of the two snapshots' instruments, so an instrument
    named by either side is always accounted for.

    Args:
        cycle_id: the execution cycle this reconciliation belongs to, non-blank.
        internal: our own snapshot; its origin must be
            :attr:`SnapshotOrigin.INTERNAL_LEDGER`.
        reported: the statement snapshot; its origin must be in
            :data:`REPORTABLE_ORIGINS`.
        stamp: the I2 stamp of the run producing this verdict.
        cash_tolerance_usd: allowed absolute cash divergence in **US dollars**.
            Defaults to :data:`CASH_TOLERANCE_USD`; may be tightened to zero and
            may not exceed :data:`MAX_CASH_TOLERANCE_USD`.

    Returns:
        A :class:`ReconciliationResult` whose findings are in a fixed total order.

    Raises:
        SnapshotValidationError: on a blank ``cycle_id``, an origin on the wrong
            side, a tolerance that is negative, non-finite or above the ceiling,
            or two snapshots more than :data:`MAX_SNAPSHOT_SKEW_SECONDS` apart.
            Each of these produces *no verdict* rather than a wrong one, and the
            kill switch treats a cycle with no verdict as an unknown condition and
            halts — which is the direction a safety control must fail in.
    """
    _require(cycle_id.strip() != "", "cycle_id is blank; a verdict must name the cycle it judges")
    _require(
        internal.origin is SnapshotOrigin.INTERNAL_LEDGER,
        f"the internal snapshot's origin is {internal.origin.value!r}, not "
        f"{SnapshotOrigin.INTERNAL_LEDGER.value!r}",
    )
    _require(
        reported.origin in REPORTABLE_ORIGINS,
        f"the reported snapshot's origin is {reported.origin.value!r}; a reconciliation of our "
        f"ledger against itself always passes and proves nothing",
    )
    _require_money("cash_tolerance_usd", cash_tolerance_usd)
    _require(
        cash_tolerance_usd >= 0,
        f"cash_tolerance_usd={cash_tolerance_usd} is negative; a negative tolerance makes "
        f"every balance a break, which is an alarm nobody can act on",
    )
    _require(
        cash_tolerance_usd <= MAX_CASH_TOLERANCE_USD,
        f"cash_tolerance_usd={cash_tolerance_usd} exceeds the ceiling "
        f"{MAX_CASH_TOLERANCE_USD}: above it a tolerance absorbs an event rather than a "
        f"quantisation step, and no magnitude of event is rounding",
    )
    skew = abs((internal.observed_at - reported.observed_at).total_seconds())
    _require(
        skew <= MAX_SNAPSHOT_SKEW_SECONDS,
        f"the two snapshots were observed {skew:.3f} s apart, beyond the "
        f"{MAX_SNAPSHOT_SKEW_SECONDS} s limit; they describe two different books and any "
        f"difference between them is explained by the activity in the gap",
    )
    findings: list[Finding] = []
    for security_id in sorted(set(internal.positions) | set(reported.positions)):
        finding = _position_finding(
            security_id=security_id,
            internal_shares=internal.positions.get(security_id),
            reported_shares=reported.positions.get(security_id),
        )
        if finding is not None:
            findings.append(finding)
    # Cash is appended last and sorted to the front, rather than appended first.
    # Discovery order and result order are then deliberately *different*, which is
    # what makes :func:`_finding_sort_key` the single place the canonical order is
    # decided instead of a no-op that happens to agree with the loop above. A
    # mutation removing the sort changes the output, and a test catches it.
    cash = _cash_finding(
        internal_cash_usd=internal.cash_usd,
        reported_cash_usd=reported.cash_usd,
        tolerance_usd=cash_tolerance_usd,
    )
    if cash is not None:
        findings.append(cash)
    return ReconciliationResult(
        cycle_id=cycle_id,
        internal=internal,
        reported=reported,
        cash_tolerance_usd=cash_tolerance_usd,
        findings=tuple(sorted(findings, key=_finding_sort_key)),
        stamp=stamp,
    )


def require_matched(result: ReconciliationResult) -> None:
    """Raise unless the reconciliation is clean.

    The explicit form of "a mismatch halts": a caller that wants the cycle to stop
    at the reconciliation step calls this instead of inspecting a boolean, so the
    stop cannot be forgotten by omission. The kill switch does not use it — it
    records the mismatch as a halt reason instead
    (:mod:`backend.execution.killswitch`) — because a halt has to be *persisted*,
    not merely raised.

    Args:
        result: the verdict to check.

    Raises:
        ReconciliationMismatchError: if ``result`` has any break.
    """
    if result.matched:
        return
    raise ReconciliationMismatchError(
        cycle_id=result.cycle_id,
        break_count=len(result.breaks),
        result_digest=result.result_digest,
        detail="; ".join(finding.detail for finding in result.breaks),
    )


@dataclass(frozen=True, slots=True)
class StoredReconciliation:
    """One persisted reconciliation, read back for investigation or re-running.

    Attributes:
        reconciliation_id: database key.
        cycle_id: the cycle the verdict judged.
        internal_payload: the stored internal snapshot payload.
        reported_payload: the stored statement snapshot payload.
        cash_tolerance_usd: the tolerance that was in force, in **US dollars**.
        result_digest: the digest of the verdict as it was recorded.
        matched: whether the verdict was clean.
        break_count: how many findings were breaks.
    """

    reconciliation_id: int
    cycle_id: str
    internal_payload: Mapping[str, object]
    reported_payload: Mapping[str, object]
    cash_tolerance_usd: Decimal
    result_digest: str
    matched: bool
    break_count: int


async def record_reconciliation(session: AsyncSession, result: ReconciliationResult) -> int:
    """Persist one verdict, with both snapshots, so it can be re-run later.

    Both snapshot payloads are stored in full beside their digests. That is what
    makes a break investigable after the fact: the row is not a summary of a
    comparison, it is the comparison's inputs plus its output, and
    :func:`rerun` re-derives the second from the first.

    Does **not** commit: the caller owns the transaction, matching
    :mod:`backend.execution.store`.

    Args:
        session: any writable ``AsyncSession``.
        result: the verdict to persist.

    Returns:
        The new ``reconciliation_id``.
    """
    payload = result.as_json()
    statement = (
        sa.insert(ExecutionReconciliationRow)
        .values(
            cycle_id=result.cycle_id,
            internal_origin=result.internal.origin.value,
            reported_origin=result.reported.origin.value,
            internal_snapshot=result.internal.as_json(),
            internal_digest=result.internal.digest,
            reported_snapshot=result.reported.as_json(),
            reported_digest=result.reported.digest,
            internal_observed_at=result.internal.observed_at,
            reported_observed_at=result.reported.observed_at,
            cash_tolerance_usd=result.cash_tolerance_usd,
            findings=payload["findings"],
            finding_count=len(result.findings),
            break_count=len(result.breaks),
            matched=result.matched,
            result_digest=result.result_digest,
            git_commit=result.stamp.git_commit,
            git_dirty=result.stamp.git_dirty,
            data_version=result.stamp.data_version,
            config_hash=result.stamp.config_hash,
            seed=result.stamp.seed,
        )
        .returning(ExecutionReconciliationRow.reconciliation_id)
    )
    inserted = await session.execute(statement)
    return int(inserted.scalar_one())


async def load_reconciliation(
    session: AsyncSession, reconciliation_id: int
) -> StoredReconciliation:
    """Read one persisted verdict back, with the snapshots it was taken over.

    Args:
        session: any readable ``AsyncSession``.
        reconciliation_id: the row's database key.

    Returns:
        A :class:`StoredReconciliation`.

    Raises:
        ReconciliationReplayError: if no row has that id. A miss is an error
            rather than ``None`` because every caller here is investigating a
            break it believes was recorded.
    """
    row = (
        await session.execute(
            sa.select(
                ExecutionReconciliationRow.reconciliation_id,
                ExecutionReconciliationRow.cycle_id,
                ExecutionReconciliationRow.internal_snapshot,
                ExecutionReconciliationRow.reported_snapshot,
                ExecutionReconciliationRow.cash_tolerance_usd,
                ExecutionReconciliationRow.result_digest,
                ExecutionReconciliationRow.matched,
                ExecutionReconciliationRow.break_count,
            ).where(ExecutionReconciliationRow.reconciliation_id == reconciliation_id)
        )
    ).first()
    if row is None:
        msg = (
            f"no reconciliation has reconciliation_id={reconciliation_id}; a break that cannot "
            f"be read back cannot be investigated"
        )
        raise ReconciliationReplayError(msg)
    internal_payload: Mapping[str, object] = row[2]
    reported_payload: Mapping[str, object] = row[3]
    return StoredReconciliation(
        reconciliation_id=int(row[0]),
        cycle_id=str(row[1]),
        internal_payload=internal_payload,
        reported_payload=reported_payload,
        cash_tolerance_usd=Decimal(str(row[4])),
        result_digest=str(row[5]),
        matched=bool(row[6]),
        break_count=int(row[7]),
    )


def rerun(stored: StoredReconciliation, *, stamp: ReproducibilityStamp) -> ReconciliationResult:
    """Re-derive a stored verdict from its stored snapshots, and prove it unchanged.

    This is the function that makes "deterministic and re-runnable" a claim rather
    than an aspiration. The stamp passed here is normally a *different* one from
    the stamp the original verdict carried — a later commit, a later data version
    — and the digest must be identical anyway, because the digest covers the
    inputs and the findings and not the stamp.

    Args:
        stored: the row read by :func:`load_reconciliation`.
        stamp: the I2 stamp of the run doing the re-derivation.

    Returns:
        The re-derived :class:`ReconciliationResult`.

    Raises:
        ReconciliationReplayError: if the re-derived digest differs from the
            stored one. Raised rather than reported, because a stored verdict that
            no longer re-derives means either the stored payload or the comparison
            has changed, and neither may be presented as the original finding.
        SnapshotValidationError: if a stored payload cannot be rebuilt.
    """
    internal = snapshot_of_json(stored.internal_payload)
    reported = snapshot_of_json(stored.reported_payload)
    result = reconcile(
        cycle_id=stored.cycle_id,
        internal=internal,
        reported=reported,
        stamp=stamp,
        cash_tolerance_usd=stored.cash_tolerance_usd,
    )
    if result.result_digest != stored.result_digest:
        msg = (
            f"re-running reconciliation_id={stored.reconciliation_id} produced digest "
            f"{result.result_digest}, but the stored verdict is {stored.result_digest}. Either "
            f"the stored payload or the comparison changed; a re-derived verdict that differs "
            f"from the recorded one cannot be presented as the original finding"
        )
        raise ReconciliationReplayError(msg)
    return result
