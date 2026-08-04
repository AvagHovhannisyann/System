"""The kill switch: fail closed, halt within the cycle that observed the cause (P11.5).

Directive §5 Phase 11 names four triggers — drawdown breach, stale data,
reconciliation mismatch, manual operator action — and one timing guarantee: the
switch "halts within one cycle when triggered".

Fail closed, and what that actually requires
---------------------------------------------

A kill switch that can fail to fire is worse than no kill switch, because it is
trusted. Everything downstream — position sizing, the drawdown limit, the
operator's willingness to leave the system running unattended — is written on the
assumption that this control works. So the default answer here is *halt*, and a
non-halt is something the evaluation has to earn by seeing four well-formed
measurements that are all inside their limits.

Concretely, every one of these halts, under
:attr:`~backend.execution.halt.HaltTrigger.UNKNOWN_CONDITION`:

- **A measurement that is absent.** ``None`` equity, ``None`` data age, ``None``
  reconciliation. A cycle that did not reconcile has not proved its book; a cycle
  that could not measure its drawdown does not know whether it breached one.
- **A limit that is not configured.** "No limit set" is not "no limit". An
  unconfigured drawdown limit or maximum data age means nobody decided, and
  nobody deciding is not permission.
- **A number that is not a number.** ``Decimal("NaN")`` compares false against
  every threshold, so ``if drawdown > limit`` silently *passes* on a NaN. That is
  the textbook shape of a fail-open safety check, and it is refused explicitly
  rather than left to the comparison.
- **A number that is impossible.** A negative data age means a clock moved
  backwards; the data is not fresh, its age is unmeasurable. A peak equity of
  zero or less makes the drawdown fraction undefined. A drawdown limit outside
  ``(0, 1]`` is a misconfiguration, and a misconfigured limit is not a licence.
- **A probe that raised.** Each of the four evaluations runs inside its own
  ``try``, and an exception becomes a halt reason rather than propagating. An
  exception escaping the kill switch would be indistinguishable, from the
  caller's perspective, from the kill switch never having run.
- **Evidence from the wrong cycle.** A reconciliation whose ``cycle_id`` is not
  this cycle's is not evidence about this cycle. Reusing yesterday's clean
  verdict is the most comfortable way for a reconciliation gate to become
  decorative.

The one deliberate exception is a *manual* trigger's absence. Silence from an
operator is a legitimate absence, not a missing measurement — so
``manual is None`` means "nobody pulled the switch" and halts nothing. The
converse asymmetry matters more: :class:`ManualHaltRequest` **never raises**. A
blank name or a blank reason is normalised, because refusing to construct an
operator's halt request over a formatting problem would be a fail-open path
through the one trigger a human reaches for in an emergency.

"Within one cycle", made testable
---------------------------------

:func:`run_cycle` is the cycle's safety pass, and the guarantee is stated as an
equality rather than a duration: for every trigger, the halt row written by the
cycle that observed the condition carries **that cycle's own** ``cycle_id``. Zero
further cycles elapse between the condition being observable and trading being
stopped, and the claim is checkable after the fact from the halt log alone, not
only at the moment it happened.

The second half of the guarantee is that the halt actually gates something:
:attr:`CycleOutcome.release_permitted` is false, and
:func:`~backend.execution.halt.assert_not_halted` — which every release path
calls — raises from that same cycle onward, in this process and in any process
that starts later.

Units
-----

Equity and its peak are **US dollars**. The drawdown limit is a **fraction of the
peak** in ``(0, 1]`` — not a percentage and not basis points; the field name says
``fraction`` because this is the bug class §8 calls out as silent. Data age and
its maximum are **seconds**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from backend.execution.halt import (
    HaltReason,
    HaltTrigger,
    assert_not_halted,
    engage_halt,
    open_halts,
)
from backend.execution.reconciliation import ReconciliationResult

if TYPE_CHECKING:
    import datetime as dt

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.tracking.stamp import ReproducibilityStamp

__all__ = [
    "ARITHMETIC_PRECISION",
    "CycleObservation",
    "CycleOutcome",
    "DataFreshnessObservation",
    "DrawdownObservation",
    "KillSwitchDecision",
    "ManualHaltRequest",
    "evaluate",
    "guard_before_release",
    "run_cycle",
]

ARITHMETIC_PRECISION: Final = 50
"""Decimal precision the drawdown arithmetic runs at, in significant digits.

Pinned inside a local context rather than inherited from the process-wide decimal
context, which any other code can change. A safety threshold whose verdict
depends on a global someone else set is not a threshold.

Fifty digits is far more than the inputs carry (equity is at most eighteen
significant digits at six decimal places), so the comparison below is exact.
"""

_DRAWDOWN_LIMIT_CEILING: Final = Decimal(1)
"""Largest meaningful drawdown limit: a fraction of 1 is a total loss of the peak.

A limit above it can never be breached, which is a limit that does nothing while
reading like protection — refused as a misconfiguration.
"""


@dataclass(frozen=True, slots=True)
class DrawdownObservation:
    """Equity, its running peak, and the fraction of the peak we may give back.

    Every field is optional, and every ``None`` halts. That is not defensive
    padding: this object is built from measurements that can genuinely be
    unavailable — a valuation that has not run, a peak that has not been
    initialised, a limit nobody configured — and the alternative to modelling the
    absence is a caller inventing a zero.

    Units: ``equity_usd`` and ``peak_equity_usd`` are **US dollars**;
    ``limit_fraction`` is a **fraction of the peak** in ``(0, 1]``.

    Attributes:
        equity_usd: current account equity, or ``None`` if not measured.
        peak_equity_usd: the running peak equity the drawdown is measured from,
            or ``None`` if not measured.
        limit_fraction: the largest fraction of the peak that may be given back
            before trading stops, or ``None`` if not configured.
    """

    equity_usd: Decimal | None
    peak_equity_usd: Decimal | None
    limit_fraction: Decimal | None


@dataclass(frozen=True, slots=True)
class DataFreshnessObservation:
    """How old the data driving this cycle is, and how old it may be.

    Units: both fields are **seconds**.

    Attributes:
        age_seconds: age of the freshest datum the cycle depends on, or ``None``
            if not measured.
        max_age_seconds: the largest age tolerated, or ``None`` if not configured.
    """

    age_seconds: Decimal | None
    max_age_seconds: Decimal | None


@dataclass(frozen=True, slots=True)
class ManualHaltRequest:
    """An operator pulling the switch. Constructing one never fails.

    The validation asymmetry is the design. Everywhere else in this package a
    malformed value is refused; here a blank name or a blank reason is
    *normalised*, because the only thing worse than an unattributed manual halt is
    an exception where a manual halt should have been. A human reaching for this
    in an emergency must not be able to be turned away by a format check.

    Attributes:
        requested_by: who asked for the halt. Normalised to ``"unattributed"``
            when blank, so the halt still fires and the gap is visible.
        reason: why. Normalised to ``"no reason given"`` when blank.
    """

    requested_by: str
    reason: str

    def __post_init__(self) -> None:
        """Normalise blank fields. Never raises, deliberately."""
        if not self.requested_by.strip():
            object.__setattr__(self, "requested_by", "unattributed")
        if not self.reason.strip():
            object.__setattr__(self, "reason", "no reason given")


@dataclass(frozen=True, slots=True)
class CycleObservation:
    """Everything one cycle knows about whether it is safe to trade.

    Assembled by the caller and handed here; nothing in this module measures
    anything, for the same reason nothing in this package opens a connection.

    Attributes:
        cycle_id: identifies this cycle. Non-blank — a halt has to name the cycle
            it was engaged in, and that name is the whole "within one cycle"
            proof.
        observed_at: when the cycle observed all of this, timezone-aware.
        drawdown: the drawdown measurement, or ``None`` if the cycle could not
            take one — which halts.
        freshness: the data-age measurement, or ``None`` — which halts.
        reconciliation: this cycle's reconciliation verdict, or ``None`` — which
            halts, because a cycle that did not reconcile has not proved its book.
        manual: an operator's halt request, or ``None``. The only ``None`` that
            does not halt: silence from an operator is a legitimate absence.
        stamp: the I2 stamp of the run, recorded on every halt row this
            observation produces.
    """

    cycle_id: str
    observed_at: dt.datetime
    drawdown: DrawdownObservation | None
    freshness: DataFreshnessObservation | None
    reconciliation: ReconciliationResult | None
    manual: ManualHaltRequest | None
    stamp: ReproducibilityStamp


@dataclass(frozen=True, slots=True)
class KillSwitchDecision:
    """What one evaluation concluded, and why.

    Attributes:
        cycle_id: the cycle evaluated.
        reasons: every condition that fired, in a fixed order (drawdown, data,
            reconciliation, manual, then any evaluation failure). All of them, not
            the first: an operator investigating a halt needs to know whether one
            thing went wrong or four.
    """

    cycle_id: str
    reasons: tuple[HaltReason, ...] = field(default_factory=tuple)

    @property
    def should_halt(self) -> bool:
        """Whether this cycle must halt. True whenever anything at all fired."""
        return bool(self.reasons)


def _unknown(detail: str, evidence: dict[str, object]) -> HaltReason:
    """Build the fail-closed halt reason.

    Args:
        detail: what could not be established.
        evidence: the measurements as they were seen.

    Returns:
        A :class:`~backend.execution.halt.HaltReason` under
        :attr:`~backend.execution.halt.HaltTrigger.UNKNOWN_CONDITION`.
    """
    return HaltReason(
        trigger=HaltTrigger.UNKNOWN_CONDITION,
        detail=detail,
        evidence=MappingProxyType(evidence),
    )


def _money_evidence(amount: Decimal | None) -> str | None:
    """Render a decimal for the evidence payload without losing exactness.

    Args:
        amount: the value, or ``None``.

    Returns:
        The value as a string, or ``None``. A string rather than a float because
        the evidence is stored as JSONB and a binary float has no exact decimal
        rendering — the number an operator reads must be the number the decision
        used.
    """
    return None if amount is None else str(amount)


def _measured(amount: Decimal | None) -> Decimal | None:
    """Return the value when it is a usable measurement, otherwise ``None``.

    ``None`` is not a measurement and a non-finite decimal is not a number.
    Filtered here, before any comparison, because ``Decimal("NaN") > limit`` is
    either ``False`` or an exception — and the ``False`` branch is a silent
    fail-open in the exact shape a safety check must never have.

    Returning the value rather than a boolean is deliberate: it collapses "check
    it" and "use it" into one step, so there is no path where a caller tests one
    field and then reads another.

    Args:
        amount: the candidate measurement.

    Returns:
        The value when it is a finite :class:`~decimal.Decimal`, else ``None``.
    """
    if isinstance(amount, Decimal) and amount.is_finite():
        return amount
    return None


def _evaluate_drawdown(observation: CycleObservation) -> HaltReason | None:
    """Decide whether the drawdown limit is breached, or cannot be established.

    The verdict is taken on an exact multiplication — ``peak - equity >
    limit * peak`` — rather than on a division, so no rounding step sits between
    the measurement and the decision. The drawdown *fraction* reported in the
    evidence is divided out afterwards purely for the operator to read, and
    cannot change the verdict.

    A drawdown exactly equal to the limit is **not** a breach: the limit is the
    largest tolerated drawdown, so breaching means exceeding it. Both sides of
    that boundary are pinned by tests.

    Args:
        observation: the cycle's observation.

    Returns:
        A halt reason, or ``None`` when the drawdown is measured and within limit.
    """
    drawdown = observation.drawdown
    if drawdown is None:
        return _unknown(
            "no drawdown measurement was taken this cycle, so whether the limit is breached "
            "is unknown; an unmeasured drawdown is not a drawdown of zero",
            {"drawdown": None},
        )
    evidence: dict[str, object] = {
        "equity_usd": _money_evidence(drawdown.equity_usd),
        "peak_equity_usd": _money_evidence(drawdown.peak_equity_usd),
        "limit_fraction": _money_evidence(drawdown.limit_fraction),
    }
    equity = _measured(drawdown.equity_usd)
    peak = _measured(drawdown.peak_equity_usd)
    limit = _measured(drawdown.limit_fraction)
    if equity is None or peak is None:
        return _unknown(
            "equity or peak equity is missing or not a finite number, so the drawdown cannot "
            "be computed; a non-finite value compares false against every threshold and would "
            "pass a naive check",
            evidence,
        )
    if limit is None:
        return _unknown(
            "no drawdown limit is configured, or it is not a finite number. 'No limit set' is "
            "not 'no limit': nobody decided, and nobody deciding is not permission to trade",
            evidence,
        )
    if peak <= 0:
        return _unknown(
            f"peak equity is {peak} USD, so the drawdown fraction is undefined; a peak of zero "
            f"or less means the peak was never established and the limit measures nothing",
            evidence,
        )
    if not (0 < limit <= _DRAWDOWN_LIMIT_CEILING):
        return _unknown(
            f"the drawdown limit is {limit}, outside (0, {_DRAWDOWN_LIMIT_CEILING}]. A limit of "
            f"zero or less halts unconditionally and one above a total loss can never be "
            f"breached; both are misconfigurations, and a misconfigured limit is not a licence",
            evidence,
        )
    with localcontext() as context:
        context.prec = ARITHMETIC_PRECISION
        shortfall = peak - equity
        allowed = limit * peak
        breached = shortfall > allowed
        fraction = shortfall / peak
    evidence["drawdown_fraction"] = str(fraction)
    evidence["allowed_shortfall_usd"] = str(allowed)
    evidence["shortfall_usd"] = str(shortfall)
    if not breached:
        return None
    return HaltReason(
        trigger=HaltTrigger.DRAWDOWN_BREACH,
        detail=(
            f"drawdown {fraction} of peak equity {peak} USD exceeds the limit {limit}: equity "
            f"is {equity} USD, a shortfall of {shortfall} USD against an allowance of "
            f"{allowed} USD"
        ),
        evidence=MappingProxyType(evidence),
    )


def _evaluate_freshness(observation: CycleObservation) -> HaltReason | None:
    """Decide whether the cycle's data is too old, or its age cannot be established.

    Args:
        observation: the cycle's observation.

    Returns:
        A halt reason, or ``None`` when the age is measured and within the limit.
    """
    freshness = observation.freshness
    if freshness is None:
        return _unknown(
            "no data-age measurement was taken this cycle, so whether the data is stale is "
            "unknown; unmeasured is not fresh",
            {"freshness": None},
        )
    evidence: dict[str, object] = {
        "age_seconds": _money_evidence(freshness.age_seconds),
        "max_age_seconds": _money_evidence(freshness.max_age_seconds),
    }
    age = _measured(freshness.age_seconds)
    limit = _measured(freshness.max_age_seconds)
    if age is None:
        return _unknown(
            "the data age is missing or not a finite number, so staleness cannot be decided",
            evidence,
        )
    if limit is None:
        return _unknown(
            "no maximum data age is configured, or it is not a finite number; an unconfigured "
            "staleness limit is not an unlimited one",
            evidence,
        )
    if age < 0:
        return _unknown(
            f"the data age is {age} s, which is negative: a clock moved backwards, so the age "
            f"is unmeasurable rather than small. A naive comparison would read this as the "
            f"freshest data the system has ever seen",
            evidence,
        )
    if limit <= 0:
        return _unknown(
            f"the maximum data age is {limit} s, which is not positive; no datum can ever "
            f"satisfy it, so the limit is a misconfiguration rather than a threshold",
            evidence,
        )
    if age <= limit:
        return None
    return HaltReason(
        trigger=HaltTrigger.STALE_DATA,
        detail=(
            f"the data driving this cycle is {age} s old, past the {limit} s limit; a decision "
            f"taken on stale prices is a decision about a book that no longer exists"
        ),
        evidence=MappingProxyType(evidence),
    )


def _evaluate_reconciliation(observation: CycleObservation) -> HaltReason | None:
    """Decide whether this cycle's reconciliation permits trading.

    Three ways to fail, and only one of them is a mismatch: no verdict at all, a
    verdict about a different cycle, or a verdict with breaks. The middle case is
    the one that would otherwise rot quietly — a stale clean verdict reused
    forever is a reconciliation gate that never closes.

    Args:
        observation: the cycle's observation.

    Returns:
        A halt reason, or ``None`` when this cycle's own verdict is clean.
    """
    # Typed `object` on purpose: the annotation on the field is a claim about
    # what the caller intended, and this probe is the one place in the system
    # that must not trust a claim. A wrong type here has to become a halt, not a
    # TypeError three frames away.
    candidate: object = observation.reconciliation
    if candidate is None:
        return _unknown(
            "no reconciliation was produced for this cycle, so the book is unproved. A cycle "
            "that did not reconcile has not established that the positions it is about to "
            "trade against are the positions that exist",
            {"reconciliation": None},
        )
    if not isinstance(candidate, ReconciliationResult):
        return _unknown(
            f"the cycle's reconciliation is a {type(candidate).__name__}, not a verdict this "
            f"module can read",
            {"reconciliation": type(candidate).__name__},
        )
    result = candidate
    if result.cycle_id != observation.cycle_id:
        return _unknown(
            f"the reconciliation supplied belongs to cycle {result.cycle_id!r}, not to "
            f"{observation.cycle_id!r}. A verdict about another cycle is not evidence about "
            f"this one, and reusing a clean one is how a reconciliation gate stops closing",
            {
                "reconciliation_cycle_id": result.cycle_id,
                "observation_cycle_id": observation.cycle_id,
                "result_digest": result.result_digest,
            },
        )
    if result.matched:
        return None
    return HaltReason(
        trigger=HaltTrigger.RECONCILIATION_MISMATCH,
        detail=(
            f"reconciliation for cycle {result.cycle_id} found {len(result.breaks)} break(s): "
            + "; ".join(finding.detail for finding in result.breaks)
        ),
        evidence=MappingProxyType(
            {
                "result_digest": result.result_digest,
                "internal_digest": result.internal.digest,
                "reported_digest": result.reported.digest,
                "break_count": len(result.breaks),
                "kinds": [finding.kind.value for finding in result.breaks],
                "security_ids": [finding.security_id for finding in result.breaks],
            }
        ),
    )


def _evaluate_manual(observation: CycleObservation) -> HaltReason | None:
    """Decide whether an operator has pulled the switch.

    The only probe whose ``None`` is benign: an operator who said nothing has not
    failed to measure something.

    Args:
        observation: the cycle's observation.

    Returns:
        A halt reason when a request is present, otherwise ``None``.
    """
    manual = observation.manual
    if manual is None:
        return None
    return HaltReason(
        trigger=HaltTrigger.MANUAL,
        detail=f"halt requested by {manual.requested_by}: {manual.reason}",
        evidence=MappingProxyType({"requested_by": manual.requested_by, "reason": manual.reason}),
    )


_PROBE_NAMES: Final[tuple[str, ...]] = (
    "drawdown",
    "data_freshness",
    "reconciliation",
    "manual",
)
"""Names of the four probes, in evaluation order, for the failure evidence."""


def evaluate(observation: CycleObservation) -> KillSwitchDecision:
    """Evaluate the four triggers and return every reason to halt.

    Pure and total: reads no clock, touches no database, and **never raises**. An
    exception escaping here would be indistinguishable, to the caller, from the
    kill switch never having run — so each probe is wrapped, and a probe that
    raises produces an
    :attr:`~backend.execution.halt.HaltTrigger.UNKNOWN_CONDITION` reason carrying
    the exception. Halting because the check broke is the only safe reading of a
    check that broke.

    Every firing reason is returned, not the first: one halt caused by four
    conditions and one caused by a single condition are different situations and
    an operator has to be able to tell them apart.

    Args:
        observation: everything the cycle knows.

    Returns:
        A :class:`KillSwitchDecision`. ``should_halt`` is true whenever any reason
        fired, which includes every fail-closed path.
    """
    reasons: list[HaltReason] = []
    probes = (
        _evaluate_drawdown,
        _evaluate_freshness,
        _evaluate_reconciliation,
        _evaluate_manual,
    )
    for name, probe in zip(_PROBE_NAMES, probes, strict=True):
        # Deliberately broad, and deliberately not re-raised. A safety check that
        # can propagate an exception is a safety check the caller can lose.
        try:
            reason = probe(observation)
        except Exception as exc:
            reasons.append(
                _unknown(
                    f"the {name} check raised {type(exc).__name__}: {exc}. The condition it was "
                    f"meant to rule out is therefore unknown, and unknown halts",
                    {"probe": name, "error": type(exc).__name__, "message": str(exc)},
                )
            )
            continue
        if reason is not None:
            reasons.append(reason)
    cycle_id = _cycle_id_of(observation)
    return KillSwitchDecision(cycle_id=cycle_id, reasons=tuple(reasons))


def _cycle_id_of(observation: CycleObservation) -> str:
    """Return the observation's cycle id, substituting a marker for a blank one.

    A blank cycle id is a real defect — the halt row's CHECK refuses one — but
    raising here would mean a malformed observation could not be halted on. The
    marker keeps the halt writable and makes the defect visible in the log.

    Args:
        observation: the cycle's observation.

    Returns:
        The cycle id, or a marker naming the problem.
    """
    candidate: object = observation.cycle_id
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    return "unnamed-cycle"


@dataclass(frozen=True, slots=True)
class CycleOutcome:
    """What one cycle's safety pass concluded and wrote.

    Attributes:
        cycle_id: the cycle.
        decision: the evaluation, including every reason that fired.
        engaged_halt_ids: the ``halt_id`` of every engagement written by this
            cycle, in the order the reasons fired.
        pre_existing_halt_ids: halts that were already open when the cycle began.
            A cycle that starts halted stays halted whatever it measures.
        release_permitted: whether orders may be released. False whenever this
            cycle halted **or** was already halted, and the value every caller
            checks in addition to — never instead of —
            :func:`~backend.execution.halt.assert_not_halted`.
    """

    cycle_id: str
    decision: KillSwitchDecision
    engaged_halt_ids: tuple[int, ...]
    pre_existing_halt_ids: tuple[int, ...]
    release_permitted: bool

    @property
    def halted(self) -> bool:
        """Whether trading is stopped after this cycle, for any reason."""
        return not self.release_permitted


async def run_cycle(session: AsyncSession, observation: CycleObservation) -> CycleOutcome:
    """Run one cycle's safety pass: evaluate, engage, and report whether to trade.

    The order is the guarantee. The evaluation happens first, every reason it
    produces is written as a halt engagement **before** this function returns, and
    the returned outcome forbids release. There is no window in which a condition
    has been observed and trading is still permitted, and there is no cycle
    boundary between the two — the halt row carries this cycle's own ``cycle_id``,
    which is what makes "within one cycle" checkable from the log afterwards.

    Halts already open are read first and reported separately: a cycle that begins
    halted stays halted regardless of what it measures, because clearing is an
    explicit act (:func:`~backend.execution.halt.clear_halt`) and never a
    side effect of a clean measurement.

    Does **not** commit — the caller owns the transaction, matching
    :mod:`backend.execution.store`. The caller must commit for the halt to survive
    a restart.

    Args:
        session: any writable ``AsyncSession``.
        observation: everything this cycle knows.

    Returns:
        A :class:`CycleOutcome`.

    Raises:
        HaltStateUnavailableError: if the halt log cannot be read. Propagated
            rather than swallowed: a cycle that cannot determine its own halt
            state must not proceed, and this is the one failure that must stop the
            caller rather than be summarised into an outcome it might ignore.
    """
    existing = await open_halts(session)
    decision = evaluate(observation)
    engaged: list[int] = []
    for reason in decision.reasons:
        engaged.append(
            await engage_halt(
                session,
                cycle_id=decision.cycle_id,
                reason=reason,
                occurred_at=observation.observed_at,
                stamp=observation.stamp,
            )
        )
    return CycleOutcome(
        cycle_id=decision.cycle_id,
        decision=decision,
        engaged_halt_ids=tuple(engaged),
        pre_existing_halt_ids=tuple(halt.halt_id for halt in existing),
        release_permitted=not existing and not decision.should_halt,
    )


async def guard_before_release(session: AsyncSession) -> None:
    """The check a release path calls immediately before releasing anything.

    A thin alias for :func:`~backend.execution.halt.assert_not_halted`, kept here
    so the release-side contract is stated in the module the release path already
    imports, and so a test can assert that the guard and the cycle read the *same*
    log rather than two caches that could disagree.

    Args:
        session: any readable ``AsyncSession``.

    Raises:
        SystemHaltedError: if any halt is open.
        HaltStateUnavailableError: if the halt log cannot be read.
    """
    await assert_not_halted(session)
