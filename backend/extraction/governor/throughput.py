"""Throughput limits: what a tier lets you send, in tokens rather than money (P7.7, D-047).

Why this exists alongside the spend governor
--------------------------------------------

:mod:`backend.extraction.governor.governor` bounds a call in **money** and
refuses when a configured cap would be breached. That control is correct and it
is blind on a free tier, for a reason worth stating plainly: **the price is
genuinely zero, so a dollar cap against it can never trip.** Configure fifty
dollars a month against a zero price and the governor reports healthy spend
right up to the moment the provider starts answering 429, two-thirds of the way
through a run, with a partial extraction set already written.

That is the same failure D-032 named — *a cap enforced against a made-up price
enforces nothing* — in its degenerate form. The price is not made up here. It is
correct, and correct is not the same as **binding**.

The binding limit on a free tier is tokens-per-day, published per model, scoped
to the account. This module models that, and nothing else: it is arithmetic over
configured limits, and it holds no state, contacts no provider, and decides no
spend.

Three refusals, and they are not the same refusal
-------------------------------------------------

1. :class:`~backend.extraction.governor.errors.ThroughputLimitUnknownError` — the
   model has no configured limit. An operator has work to do.
2. :class:`~backend.extraction.governor.errors.RequestExceedsTierCeilingError` —
   this request can never be sent, at any time. **A per-minute token allowance is
   also a per-request ceiling**: one request larger than the whole minute's
   allowance cannot fit inside a minute. Providers answer 413, not 429, and
   backoff never helps. The ceiling sits far below the advertised context window
   (roughly 6k-12k against 131,072 on the tier that motivated this), so a caller
   who sizes chunks against the context window builds a workload where *every*
   request fails.
3. :class:`~backend.extraction.governor.errors.ThroughputInfeasibleError` — each
   request would fit, but the batch cannot finish inside its horizon. Raised
   before the first call, so the choice — smaller universe, shorter history,
   narrower ensemble, or accept the wall-clock — is made while it is still a
   choice.

What is deliberately not here
-----------------------------

**No numbers.** :data:`CATALOG_THROUGHPUT_LIMITS` ships empty, exactly as
:data:`~backend.extraction.governor.estimate.CATALOG_PRICES` does, and for the
same reason: a published rate limit is a fact about a vendor's current pricing
page, not about this system. Written into a module, it would be read as fact
long after it stopped being true, and a limit that is stale in the permissive
direction produces the 429 this module exists to prevent. The figures measured
on 2026-08-04 are recorded in ``DECISIONS.md`` D-047, where they carry a date.

**No consumption ledger.** Planning assumes a full daily allowance. That is an
honest assumption for a batch planned before it starts and a wrong one for a
batch resumed mid-day; :func:`plan_batch` states it in its own docstring rather
than burying it, and a caller who has already spent part of the day should pass
the remaining allowance rather than the published one.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from backend.extraction.governor.errors import (
    RequestExceedsTierCeilingError,
    ThroughputInfeasibleError,
    ThroughputLimitsNotConfiguredError,
    ThroughputLimitUnknownError,
)
from backend.extraction.governor.estimate import provider_of

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "CATALOG_THROUGHPUT_LIMITS",
    "DAY_WINDOW_FORMAT",
    "MINUTE_WINDOW_FORMAT",
    "BatchPlan",
    "ModelThroughputLimit",
    "ThroughputBook",
    "ThroughputWindow",
    "daily_call_capacity",
    "plan_batch",
    "require_request_fits",
]

MINUTE_WINDOW_FORMAT: Final = "%Y-%m-%dT%H:%M"
"""UTC key format for a per-minute window."""

DAY_WINDOW_FORMAT: Final = "%Y-%m-%d"
"""UTC key format for a per-day window. Same shape the spend ledger's daily window uses."""


class ThroughputWindow(StrEnum):
    """A period a throughput limit is measured over.

    Both windows always constrain, and they constrain differently: the minute
    window bounds a burst *and* the size of any single request, while the day
    window bounds the run. A workload can satisfy either alone and still be
    impossible — a batch that fits the day's tokens but exceeds the minute's
    ceiling on every request never completes a single call.
    """

    MINUTE = "minute"
    DAY = "day"

    def key(self, at: dt.datetime) -> str:
        """Return the window key *at* falls in, in UTC.

        Args:
            at: the instant to classify. Must be timezone-aware; it is converted
                to UTC before formatting, so a caller passing a local time gets
                the correct UTC window rather than a silently shifted one.

        Returns:
            ``"YYYY-MM-DDTHH:MM"`` for :attr:`MINUTE`, ``"YYYY-MM-DD"`` for
            :attr:`DAY`.

        Raises:
            ValueError: *at* is naive. Providers reset these windows on **their**
                UTC clock; a naive instant names no window, and two machines
                would disagree about which day's allowance a call spent.
        """
        if at.tzinfo is None or at.tzinfo.utcoffset(at) is None:
            msg = (
                f"cannot place naive timestamp {at!r} in a {self.value} window; throughput "
                "windows are UTC periods and a naive instant names no period"
            )
            raise ValueError(msg)
        moment = at.astimezone(dt.UTC)
        fmt = MINUTE_WINDOW_FORMAT if self is ThroughputWindow.MINUTE else DAY_WINDOW_FORMAT
        return moment.strftime(fmt)


@dataclass(frozen=True, slots=True)
class ModelThroughputLimit:
    """What one model's tier permits, per minute and per day.

    Every field is optional because providers genuinely leave limits off: the
    free tier this was written against publishes no tokens-per-day for some
    models, and the main alternative publishes none at all — which is precisely
    what makes that alternative better for token-heavy documents (D-047).
    ``None`` therefore means **not capped**, and it is a different statement from
    zero, which means capped at nothing.

    Attributes:
        model: qualified ``"provider:model"`` identifier. Limits are per model,
            not per provider, because that is how they are published — two
            models on one account do not share a token budget, and treating them
            as though they did would either strand capacity or overrun it.
        requests_per_minute: request-rate ceiling, or ``None`` if uncapped.
        requests_per_day: daily request allowance, or ``None`` if uncapped.
        tokens_per_minute: token-rate ceiling, or ``None`` if uncapped. Doubles
            as the **largest single request** this tier can accept.
        tokens_per_day: daily token allowance, or ``None`` if uncapped. Usually
            the binding constraint, and usually not the one that gets quoted.
    """

    model: str
    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None

    def __post_init__(self) -> None:
        """Refuse a limit that is unqualified, negative, empty, or self-contradictory."""
        provider_of(self.model)  # raises ValueError if the model is not "provider:model"

        fields = {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_day": self.requests_per_day,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_day": self.tokens_per_day,
        }
        for name, value in fields.items():
            if value is not None and value < 0:
                msg = (
                    f"{self.model}: {name} must not be negative; got {value}. Zero is legal "
                    "and means the tier permits nothing; None means it is not capped."
                )
                raise ValueError(msg)

        if all(value is None for value in fields.values()):
            msg = (
                f"{self.model}: a limit with every field None states nothing. An entry that "
                "constrains nothing is an unconfigured model wearing a configured model's "
                "clothes — omit it from the book instead, so the refusal names it."
            )
            raise ValueError(msg)

        # A minute's allowance cannot exceed a day's: a day contains the minute.
        # Catches the commonest transcription slip, which is reading a row of a
        # vendor's limits table into the wrong pair of columns.
        for per_minute, per_day, unit in (
            (self.requests_per_minute, self.requests_per_day, "requests"),
            (self.tokens_per_minute, self.tokens_per_day, "tokens"),
        ):
            if per_minute is not None and per_day is not None and per_minute > per_day:
                msg = (
                    f"{self.model}: {unit}_per_minute ({per_minute}) exceeds {unit}_per_day "
                    f"({per_day}); a day contains the minute, so this pair cannot both hold. "
                    "Most likely the per-minute and per-day columns were read the wrong way round."
                )
                raise ValueError(msg)

    @property
    def largest_admissible_request(self) -> int | None:
        """Tokens in the biggest single request this tier can accept, or ``None``.

        This is :attr:`tokens_per_minute` under its operational name. It is worth
        a name of its own because the two readings lead to different code: as a
        *rate* it suggests waiting, and as a *ceiling* it says no amount of
        waiting will help. The provider's 413 is the second reading.
        """
        return self.tokens_per_minute


CATALOG_THROUGHPUT_LIMITS: Final[Mapping[str, ModelThroughputLimit]] = MappingProxyType({})
"""No throughput limits ship with this platform (B4, I3).

Empty for the same reason :data:`~backend.extraction.governor.estimate.CATALOG_PRICES`
is empty. A vendor's published limits are a fact about their pricing page on the
day it was read; committed to a module they would be read as fact long after they
stopped being true, and a stale limit that is too *permissive* produces exactly
the 429 this module exists to prevent. The figures observed on 2026-08-04 live in
``DECISIONS.md`` D-047, dated.
"""


@dataclass(frozen=True, slots=True)
class ThroughputBook:
    """The configured throughput limits, keyed by qualified model.

    Attributes:
        limits: mapping of ``"provider:model"`` to its limit.
    """

    limits: Mapping[str, ModelThroughputLimit]

    def __post_init__(self) -> None:
        """Refuse an empty book, or one whose keys disagree with its entries."""
        if not self.limits:
            raise ThroughputLimitsNotConfiguredError(
                "no throughput limits are configured, so no call can be admitted. A free "
                "tier's price is genuinely zero, so the spend cap cannot bind; the limit "
                "that can be breached is tokens-per-day and it has to be stated (D-047)."
            )
        for key, limit in self.limits.items():
            if key != limit.model:
                msg = (
                    f"throughput limit for {limit.model!r} is filed under {key!r}; a limit "
                    "looked up by the wrong key governs the wrong model's budget"
                )
                raise ThroughputLimitsNotConfiguredError(msg)

    @classmethod
    def of(cls, *limits: ModelThroughputLimit) -> ThroughputBook:
        """Build a book from *limits*, refusing a model named twice.

        Raises:
            ThroughputLimitsNotConfiguredError: two limits name one model. Which
                one won would depend on argument order, and a budget that depends
                on argument order is not a budget.
        """
        book: dict[str, ModelThroughputLimit] = {}
        for limit in limits:
            if limit.model in book:
                msg = (
                    f"model {limit.model!r} is configured twice; which limit applies would "
                    "depend on argument order"
                )
                raise ThroughputLimitsNotConfiguredError(msg)
            book[limit.model] = limit
        return cls(limits=MappingProxyType(book))

    def limit_for(self, model: str) -> ModelThroughputLimit:
        """Return *model*'s limit, or refuse.

        Raises:
            ThroughputLimitUnknownError: *model* is absent. A model missing from
                the book is not "unlimited", it is "ungoverned", and those are
                treated as the same thing — as they are for prices and caps.
        """
        limit = self.limits.get(model)
        if limit is None:
            known = ", ".join(sorted(self.limits)) or "(none)"
            msg = (
                f"no throughput limit configured for {model!r}; configured models are "
                f"{known}. An unconfigured model is refused rather than assumed unlimited."
            )
            raise ThroughputLimitUnknownError(msg)
        return limit


def require_request_fits(limit: ModelThroughputLimit, tokens: int) -> None:
    """Refuse a request that no minute of this tier could ever contain.

    Args:
        limit: the model's configured limit.
        tokens: upper bound on the request's total tokens — prompt plus the
            response allowance. Use
            :func:`~backend.extraction.governor.estimate.bound_input_tokens` plus
            the request's ``max_tokens``, never a ratio-based estimate: an
            estimate that is right on average is wrong on exactly the
            token-dense inputs, which are the ones that breach.

    Raises:
        ValueError: *tokens* is negative.
        RequestExceedsTierCeilingError: the request exceeds
            :attr:`~ModelThroughputLimit.largest_admissible_request`. This is
            terminal for the request as written — the fix is smaller chunks, not
            a retry.
    """
    if tokens < 0:
        msg = f"tokens must not be negative; got {tokens}"
        raise ValueError(msg)
    ceiling = limit.largest_admissible_request
    if ceiling is not None and tokens > ceiling:
        msg = (
            f"{limit.model}: a request bounded at {tokens} tokens cannot be sent on a tier "
            f"whose per-minute allowance is {ceiling}. A single request larger than the "
            "whole minute's allowance never fits in a minute, so this is a ceiling and not "
            "a rate: the provider answers 413 and backoff does not help. Reduce the chunk "
            "size. Note the ceiling is a property of the tier, not of the model's context "
            "window, and is usually far smaller."
        )
        raise RequestExceedsTierCeilingError(msg)


def daily_call_capacity(limit: ModelThroughputLimit, tokens_per_call: int) -> int | None:
    """Return how many calls of *tokens_per_call* fit in one day, or ``None`` if unbounded.

    The answer is the tighter of the two daily constraints — the request
    allowance and the token allowance — because both must hold.

    Args:
        limit: the model's configured limit.
        tokens_per_call: upper bound on one call's total tokens. Must be
            positive; a zero-token call would divide the token allowance by zero
            and the resulting "infinite capacity" would be arithmetic, not truth.

    Returns:
        The number of calls admissible in one UTC day, or ``None`` when neither
        daily limit is configured.

    Raises:
        ValueError: *tokens_per_call* is not positive.
        RequestExceedsTierCeilingError: a single call cannot be sent at all, in
            which case a per-day figure would be meaningless.
    """
    if tokens_per_call <= 0:
        msg = (
            f"tokens_per_call must be positive; got {tokens_per_call}. A call costing no "
            "tokens would imply unbounded daily capacity, which is arithmetic rather than a "
            "fact about the tier."
        )
        raise ValueError(msg)
    require_request_fits(limit, tokens_per_call)

    bounds: list[int] = []
    if limit.requests_per_day is not None:
        bounds.append(limit.requests_per_day)
    if limit.tokens_per_day is not None:
        bounds.append(limit.tokens_per_day // tokens_per_call)
    if not bounds:
        return None
    return min(bounds)


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """What a batch would cost in wall-clock, and which limit decides that.

    Attributes:
        calls: number of calls the batch requires.
        tokens_per_call: the per-call bound the plan was computed from.
        calls_per_day: total admissible calls per UTC day across every model the
            plan may use, or ``None`` if no daily limit applies.
        days_required: whole days the batch needs, or ``None`` if unbounded
            capacity makes the question moot. Whole days because the allowance
            resets on a calendar boundary, not on a rolling window: finishing a
            day's quota at noon buys nothing until midnight UTC.
        binding_models: the models whose limits were summed, in order.
    """

    calls: int
    tokens_per_call: int
    calls_per_day: int | None
    days_required: int | None
    binding_models: tuple[str, ...]

    @property
    def feasible_today(self) -> bool:
        """Whether the whole batch fits inside a single day's allowance."""
        return self.days_required is not None and self.days_required <= 1


def plan_batch(
    book: ThroughputBook,
    models: Iterable[str],
    *,
    calls: int,
    tokens_per_call: int,
    horizon_days: int | None = None,
) -> BatchPlan:
    """Plan *calls* across *models*, and refuse a batch that cannot finish in time.

    Capacity is **summed** across models because a free tier's allowances are
    per model: sharding one workload over several models is the legitimate way to
    raise daily throughput, and a planner that could not express it would
    understate what the tier permits by the number of models available.

    **The assumption, stated rather than buried:** every model is treated as
    having its full daily allowance untouched. That is right for a batch planned
    before it starts and wrong for one resumed mid-day. A caller resuming should
    pass limits carrying the *remaining* allowance, not the published one —
    there is no consumption ledger behind this function and it does not pretend
    otherwise.

    Args:
        book: configured limits.
        models: qualified models the batch may use. Order is preserved in
            :attr:`BatchPlan.binding_models` for reporting only.
        calls: how many calls the batch needs. Must be positive.
        tokens_per_call: upper bound on one call's total tokens.
        horizon_days: refuse if the batch would need more days than this. ``None``
            plans without refusing, which is the right shape for a report and the
            wrong one for a gate.

    Returns:
        The plan.

    Raises:
        ValueError: *calls* is not positive, *models* is empty, or *horizon_days*
            is not positive.
        ThroughputLimitUnknownError: a named model has no configured limit.
        RequestExceedsTierCeilingError: a call cannot be sent on one of the
            named models. Refused rather than skipped: silently dropping a model
            would quietly re-plan the batch onto the others and report a
            wall-clock the operator never agreed to.
        ThroughputInfeasibleError: the batch needs more than *horizon_days*, or
            no admissible capacity exists at all.
    """
    named = tuple(models)
    if not named:
        msg = "plan_batch needs at least one model; a batch with nowhere to run has no plan"
        raise ValueError(msg)
    if calls <= 0:
        msg = f"calls must be positive; got {calls}"
        raise ValueError(msg)
    if horizon_days is not None and horizon_days <= 0:
        msg = f"horizon_days must be positive when given; got {horizon_days}"
        raise ValueError(msg)

    capacities = [daily_call_capacity(book.limit_for(model), tokens_per_call) for model in named]

    if any(capacity is None for capacity in capacities):
        # At least one model has no daily ceiling, so the batch is not
        # day-bounded at all. Reported as unbounded rather than as a very large
        # number, because a number here would invite comparison against a horizon
        # it does not actually respect.
        return BatchPlan(
            calls=calls,
            tokens_per_call=tokens_per_call,
            calls_per_day=None,
            days_required=None,
            binding_models=named,
        )

    calls_per_day = sum(capacity for capacity in capacities if capacity is not None)
    if calls_per_day <= 0:
        msg = (
            f"the configured limits admit no calls of {tokens_per_call} tokens on any of "
            f"{', '.join(named)}, so the batch of {calls} cannot start, let alone finish"
        )
        raise ThroughputInfeasibleError(msg)

    days_required = math.ceil(calls / calls_per_day)
    if horizon_days is not None and days_required > horizon_days:
        msg = (
            f"{calls} calls of {tokens_per_call} tokens need {days_required} days at "
            f"{calls_per_day} calls/day across {', '.join(named)}, which exceeds the "
            f"{horizon_days}-day horizon. Refused before the first call rather than "
            "discovered at the provider two-thirds of the way through, where it would leave "
            "a partial extraction set. Shrink the universe, shorten the history, narrow the "
            "ensemble, or raise the horizon deliberately."
        )
        raise ThroughputInfeasibleError(msg)

    return BatchPlan(
        calls=calls,
        tokens_per_call=tokens_per_call,
        calls_per_day=calls_per_day,
        days_required=days_required,
        binding_models=named,
    )
