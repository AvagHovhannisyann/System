"""Per-provider daily and monthly spend caps, and what happens at the ceiling (P7.7, §6.5).

§6.5 asks for "daily and monthly caps per provider ... hard-stop behavior
configurable between halt and degrade-to-cheaper-model". This module is the
configuration half of that sentence: what the limits are, whose they are, and
which of the two behaviours applies when one binds. The enforcing half is
:mod:`backend.extraction.governor.ledger` (the atomic check-and-reserve) and
:mod:`backend.extraction.governor.governor` (the decision).

No default cap, ever (B4)
-------------------------

:class:`CapBook` refuses to be built empty and refuses to answer for a provider
it was not given. There is no fallback limit, no environment default, no
"unlimited" sentinel, and no shared pool a provider can borrow from. B4 states
the requirement plainly — *the cost governor refuses to run without configured
caps* — and the reason is worth restating where the code is: **a default cap is
a number nobody chose.** It would be enforced with exactly the same machinery,
produce exactly the same green dashboard, and mean nothing. A missing cap is not
"uncapped", it is "ungoverned", and the two are the same refusal here.

Per provider, not global
------------------------

Both the limits and the halt-versus-degrade policy hang off one provider. A
global cap would let a cheap provider's traffic exhaust an expensive provider's
budget, which is not a budget; and a global *policy* would be worse, because
degradation is a statement about a specific pair of models on a specific
provider — "when the good one is too expensive, this cheaper one on the same
provider will do". There is no such statement to make across providers: a
cross-provider substitution changes which credential is used, which cap is
spent, and which vendor's terms the document goes to. So
:attr:`ProviderCap.degrade_to` must name a model on the **same provider**, and
the constructor enforces it.

Windows are UTC, and what that does not claim
----------------------------------------------

A daily window is a UTC calendar day (``YYYY-MM-DD``); a monthly window is a UTC
calendar month (``YYYY-MM``). UTC because it is the only clock anything in this
platform is stored in, and because a window boundary that depends on a
configured timezone is a boundary two processes can disagree about.

What this **does not** mean: it is not the provider's billing day. A vendor's
invoice may roll at a different hour in a different zone, so this cap is the
platform's own control on its own spend, not a reconciliation of anyone's bill.
Reading it as the latter would be reading a control as an accounting record —
the exact confusion the whole module exists to avoid. Stated here because the
number will end up on an operator's gauge next to a vendor's dashboard, and
someone will compare them.

Units: every limit is a :class:`~backend.extraction.governor.estimate.Money` and
therefore carries its own currency. A cap's daily and monthly limits must be in
the same currency, and the model prices checked against them must be too — there
are no exchange rates in this system (I4).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from backend.extraction.governor.errors import (
    CapNotConfiguredError,
    CapsNotConfiguredError,
    CurrencyMismatchError,
    GovernorConfigurationError,
)
from backend.extraction.governor.estimate import Money, provider_of
from backend.extraction.providers.catalog import Provider

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DAILY_WINDOW_FORMAT",
    "MONTHLY_WINDOW_FORMAT",
    "CapBook",
    "CapPolicy",
    "ProviderCap",
    "SpendWindow",
]

DAILY_WINDOW_FORMAT: Final = "%Y-%m-%d"
"""``strftime`` format of a daily window key: ``2026-08-02``, UTC."""

MONTHLY_WINDOW_FORMAT: Final = "%Y-%m"
"""``strftime`` format of a monthly window key: ``2026-08``, UTC."""


class SpendWindow(StrEnum):
    """A period a cap is measured over.

    ``StrEnum`` so the member is usable directly as the text stored in
    ``llm_spend_ledger`` and as a path parameter on the operator's gauge
    endpoints, with an unrecognized window rejected at the HTTP boundary rather
    than inside a handler.

    Both windows are always checked. They are not alternatives: a daily cap
    bounds a burst, a monthly cap bounds a backfill, and B4 sizes the two
    differently on purpose (a generous monthly steady-state figure alongside a
    pilot backfill that must not run away in an afternoon).
    """

    DAILY = "daily"
    MONTHLY = "monthly"

    def key(self, at: dt.datetime) -> str:
        """Return the window key *at* falls in, in UTC.

        Args:
            at: the instant to classify. Must be timezone-aware; it is converted
                to UTC before the key is formatted, so a caller passing a local
                time gets the correct UTC window rather than a silently shifted
                one.

        Returns:
            ``"YYYY-MM-DD"`` for :attr:`DAILY`, ``"YYYY-MM"`` for
            :attr:`MONTHLY`.

        Raises:
            ValueError: *at* is naive. A naive timestamp has no window: it would
                be interpreted as UTC on one machine and as local time on
                another, and the two would disagree about which day's budget a
                call spent.
        """
        if at.tzinfo is None or at.tzinfo.utcoffset(at) is None:
            msg = (
                f"cannot place naive timestamp {at!r} in a {self.value} window; spend windows "
                "are UTC calendar periods and a naive instant names no period"
            )
            raise ValueError(msg)
        moment = at.astimezone(dt.UTC)
        fmt = DAILY_WINDOW_FORMAT if self is SpendWindow.DAILY else MONTHLY_WINDOW_FORMAT
        return moment.strftime(fmt)


class CapPolicy(StrEnum):
    """What to do when a call would breach this provider's cap (§6.5).

    Attributes:
        HALT: refuse the call. The extraction stops, loudly.
        DEGRADE: retry the *authorization* against a cheaper model on the same
            provider, and make the call with that model instead — recorded as a
            substitution.
    """

    HALT = "halt"
    DEGRADE = "degrade"


@dataclass(frozen=True, slots=True)
class ProviderCap:
    """One provider's limits and ceiling behaviour.

    Attributes:
        provider: whose budget this is.
        daily_limit: the most that may be spent in one UTC day, inclusive — a
            call is admitted when committed spend plus the call's **upper
            bound** is less than or equal to this.
        monthly_limit: the same for one UTC calendar month.
        policy: :attr:`CapPolicy.HALT` or :attr:`CapPolicy.DEGRADE`.
        degrade_to: the cheaper model to substitute, as a qualified identifier
            (``"provider:model"``). Required when *policy* is
            :attr:`CapPolicy.DEGRADE` and forbidden otherwise — a fallback
            configured under a halt policy is a fallback that will never be
            used, which reads to an operator as protection that does not exist.

    Zero is a legal limit and means "spend nothing on this provider": a
    deliberate, temporary stop that keeps the configuration intact. Negative is
    not, since it would refuse every call while claiming a budget.
    """

    provider: Provider
    daily_limit: Money
    monthly_limit: Money
    policy: CapPolicy
    degrade_to: str | None = None

    def __post_init__(self) -> None:
        """Reject a cap that could not be enforced or could not be acted on.

        Raises:
            ValueError: a limit is negative, or *degrade_to* is empty,
                whitespace-padded, or present/absent contrary to *policy*.
            backend.extraction.governor.errors.CurrencyMismatchError: the daily
                and monthly limits are in different currencies, which would make
                "how much is left" a question with two incomparable answers.
            backend.extraction.governor.errors.GovernorConfigurationError:
                *degrade_to* names a model on a different provider. Substituting
                across providers spends a different budget under a different
                credential — see the module docstring.
        """
        if self.daily_limit.currency != self.monthly_limit.currency:
            msg = (
                f"provider {self.provider.value!r} has a daily cap of {self.daily_limit} and a "
                f"monthly cap of {self.monthly_limit}: one budget cannot be denominated in two "
                "currencies, and this system holds no exchange rate"
            )
            raise CurrencyMismatchError(msg)
        for label, limit in (
            ("daily_limit", self.daily_limit),
            ("monthly_limit", self.monthly_limit),
        ):
            if limit.amount < 0:
                msg = (
                    f"{label} for provider {self.provider.value!r} must not be negative; got "
                    f"{limit}. Zero is the way to say 'spend nothing'"
                )
                raise ValueError(msg)
        if self.policy is CapPolicy.DEGRADE:
            if self.degrade_to is None:
                msg = (
                    f"provider {self.provider.value!r} is configured to degrade but names no "
                    "model to degrade to. Degrading with no substitute is a halt wearing a "
                    "different label"
                )
                raise ValueError(msg)
            if not self.degrade_to or self.degrade_to != self.degrade_to.strip():
                msg = (
                    f"degrade_to must be a non-empty qualified model identifier without "
                    f"surrounding whitespace; got {self.degrade_to!r}"
                )
                raise ValueError(msg)
            fallback_provider = provider_of(self.degrade_to)
            if fallback_provider != self.provider.value:
                msg = (
                    f"provider {self.provider.value!r} cannot degrade to {self.degrade_to!r}, "
                    f"which belongs to {fallback_provider!r}. A cross-provider substitution "
                    "spends a different cap under a different credential and sends the document "
                    "to a different vendor — that is a configuration change, not a degradation"
                )
                raise GovernorConfigurationError(msg)
        elif self.degrade_to is not None:
            msg = (
                f"provider {self.provider.value!r} has policy {self.policy.value!r} but names "
                f"degrade_to={self.degrade_to!r}. A fallback that can never be reached reads as "
                "protection that does not exist"
            )
            raise ValueError(msg)

    @property
    def currency(self) -> str:
        """ISO-4217 code both limits are denominated in."""
        return self.daily_limit.currency

    @property
    def limits(self) -> Mapping[SpendWindow, Money]:
        """Every window this cap constrains, mapped to its limit.

        Both windows, always — see :class:`SpendWindow` for why they are not
        alternatives.
        """
        return MappingProxyType(
            {
                SpendWindow.DAILY: self.daily_limit,
                SpendWindow.MONTHLY: self.monthly_limit,
            }
        )

    def windows(self, at: dt.datetime) -> Mapping[SpendWindow, str]:
        """Return the window keys *at* falls in.

        Args:
            at: timezone-aware instant, converted to UTC.

        Returns:
            ``{DAILY: "YYYY-MM-DD", MONTHLY: "YYYY-MM"}``.

        Raises:
            ValueError: *at* is naive.
        """
        return MappingProxyType({window: window.key(at) for window in SpendWindow})

    def require_degrade_target(self) -> str:
        """Return the model to substitute, for a cap that degrades.

        Returns:
            The qualified fallback identifier.

        Raises:
            backend.extraction.governor.errors.GovernorConfigurationError: this
                cap does not degrade. Callers reach this only by asking a
                halting cap for a fallback, which is a programming error rather
                than an operator one.
        """
        if self.policy is not CapPolicy.DEGRADE or self.degrade_to is None:
            msg = (
                f"provider {self.provider.value!r} has policy {self.policy.value!r} and no "
                "degrade target; there is nothing to substitute"
            )
            raise GovernorConfigurationError(msg)
        return self.degrade_to


@dataclass(frozen=True, slots=True)
class CapBook:
    """The configured caps, one per provider. Empty is refused (B4).

    Attributes:
        caps: provider to its :class:`ProviderCap`.

    Built through :meth:`of` or directly; either way an empty book raises, and a
    provider that is absent raises when asked for. Both refusals are the same
    statement: this governor will not authorize spending against a limit nobody
    set.
    """

    caps: Mapping[Provider, ProviderCap]

    def __post_init__(self) -> None:
        """Reject an empty book, or one whose keys disagree with its values.

        Raises:
            backend.extraction.governor.errors.CapsNotConfiguredError: the book
                is empty. Under B4 this is the current state of the repository
                and it is an error rather than a permissive default (§9.8: never
                proceed on a guess).
            backend.extraction.governor.errors.GovernorConfigurationError: a cap
                is filed under a provider other than its own, which would
                enforce one provider's limit against another's calls.
        """
        if not self.caps:
            msg = (
                "no spend cap is configured, so no provider call may be authorized (B4). The "
                "cost governor refuses to run without configured caps: a default cap is a "
                "number nobody chose, and it would look exactly like a cap that works"
            )
            raise CapsNotConfiguredError(msg)
        for provider, cap in self.caps.items():
            if cap.provider is not provider:
                msg = (
                    f"cap for {cap.provider.value!r} is filed under {provider.value!r}; that "
                    "would enforce one provider's limit against another's calls"
                )
                raise GovernorConfigurationError(msg)

    @classmethod
    def of(cls, *caps: ProviderCap) -> CapBook:
        """Build a book from caps, keyed by their own providers.

        Args:
            *caps: one cap per provider. At least one is required.

        Returns:
            The book.

        Raises:
            backend.extraction.governor.errors.CapsNotConfiguredError: no caps
                were given.
            backend.extraction.governor.errors.GovernorConfigurationError: two
                caps name the same provider. Silently keeping the last would
                make the effective budget depend on argument order.
        """
        indexed: dict[Provider, ProviderCap] = {}
        for cap in caps:
            if cap.provider in indexed:
                msg = (
                    f"two caps were given for provider {cap.provider.value!r}; keeping either "
                    "one silently would make the budget depend on argument order"
                )
                raise GovernorConfigurationError(msg)
            indexed[cap.provider] = cap
        return cls(MappingProxyType(indexed))

    def cap_for(self, provider: Provider) -> ProviderCap:
        """Return *provider*'s cap, or refuse.

        Args:
            provider: whose budget is about to be spent.

        Returns:
            The configured cap.

        Raises:
            backend.extraction.governor.errors.CapNotConfiguredError: this
                provider has no cap. Not "uncapped" — ungoverned, which is the
                same refusal. Falling back to another provider's cap would spend
                one budget against another; falling back to no cap would spend
                without one.
        """
        cap = self.caps.get(provider)
        if cap is None:
            configured = ", ".join(sorted(member.value for member in self.caps)) or "(none)"
            msg = (
                f"no spend cap is configured for provider {provider.value!r}, so calls to it "
                f"are refused (B4). Configured providers: {configured}"
            )
            raise CapNotConfiguredError(msg)
        return cap

    @property
    def providers(self) -> frozenset[Provider]:
        """The providers this book governs."""
        return frozenset(self.caps)
