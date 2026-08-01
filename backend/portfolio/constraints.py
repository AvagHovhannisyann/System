"""Portfolio constraint set and the relaxation ladder above it (P9.2).

This module holds the *policy* half of the optimizer: what a portfolio is
allowed to look like, and — when no portfolio can look like that — the
documented order in which those requirements are loosened.

**Why a ladder exists at all.** The constraint set the directive specifies
(§5 Phase 9) is a conjunction of five requirements over a universe the
optimizer does not control. Universes shrink: a borrow-availability filter
(Phase 4) can strip a sector down to two names, a delisting can leave a sector
empty, and a 2% position cap over forty names cannot add up to a fully invested
book no matter how the weights are arranged. When that happens the solver
returns ``infeasible`` and the caller is left holding nothing. The three things
an optimizer can do at that point are: return garbage, return the previous
weights, or say what it relaxed. Only the third is compatible with a system
whose backtests are meant to be believed, so the relaxations are enumerated,
ordered, and reported.

**The order is a risk judgement, not an implementation detail.** It is fixed in
:data:`RELAXATION_LADDER` and reasoned about there. Two constraints are
**never** relaxed and are not in the ladder at all:

- the **position cap**, because it is the constraint that bounds single-name
  loss. Every other constraint here shapes the portfolio's exposures; this one
  bounds what a single wrong name can do, and an optimizer that widens it to
  make its own life easier has inverted the purpose of a risk limit.
- the **gross-exposure limit**, because raising it is leverage, and leverage is
  a non-goal under directive §1.1. It is a hard ceiling
  (:data:`MAX_GROSS_EXPOSURE`) rather than a default.

**Units.** Every weight in this module is a fraction of portfolio capital:
``0.02`` is two percent of the book, never ``2`` and never ``2 bps``. Betas are
dimensionless. Directive §8 calls unit confusion the most common silent bug
class in this domain; the field names carry the unit and the docstrings repeat
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "DEFAULT_CONSTRAINTS",
    "DEFAULT_RELAXATION_LADDER",
    "MAX_GROSS_EXPOSURE",
    "RELAXATION_LADDER",
    "PortfolioConstraints",
    "RelaxableConstraint",
    "Relaxation",
    "RelaxationLadder",
]


def _is_finite(value: float) -> bool:
    """Return whether a value is a finite real number.

    Args:
        value: the number to test.

    Returns:
        ``True`` if the value is neither NaN nor an infinity.
    """
    return math.isfinite(value)


MAX_GROSS_EXPOSURE: Final = 1.0
"""Hard ceiling on gross exposure, as a fraction of capital (dimensionless).

``sum(|w_i|) <= 1`` is the no-leverage rule. Directive §1.1 lists leverage and
margin as non-goals, so this is a ceiling on what may be *requested*, not a
default that a caller can raise: :class:`PortfolioConstraints` refuses a gross
exposure above it, and no rung of the relaxation ladder moves it.

Note the arithmetic this produces at the shipped defaults: ``sum(w) == 1``
together with ``sum(|w|) <= 1`` forces ``w >= 0`` elementwise. The long-only
character of the default book is therefore a *consequence* of the
no-leverage rule rather than a separate constraint — a long-short book is
obtained by asking for a net exposure below its gross limit, not by removing a
flag.
"""

DEFAULT_POSITION_CAP: Final = 0.02
"""Maximum absolute weight in any single name, as a fraction of capital.

2% per directive §5 Phase 9. Never relaxed by the ladder — see the module
docstring.
"""

DEFAULT_SECTOR_CAP: Final = 0.20
"""Maximum gross weight in any single sector, as a fraction of capital.

20% per directive §5 Phase 9. "Gross" means ``sum(|w_i|)`` over the sector, so
that a long and a short inside one sector cannot net each other into apparent
compliance while carrying two positions' worth of sector risk. For a long-only
book gross and net coincide.
"""

DEFAULT_NET_EXPOSURE: Final = 1.0
"""Target for ``sum(w)``, as a fraction of capital — the full-investment constraint.

1.0 means the book is fully invested: every dollar of capital is in a position,
none is idle. This is the one constraint in the set that pins the *scale* of
the portfolio; without it the risk penalty in the objective would choose the
scale, and "fully invested" would become "as invested as the risk aversion
happened to feel like".

A caller who wants a dollar-neutral long-short book sets this to ``0.0``, sets
the sector targets to zero, and sets the beta target to zero. Read the note on
:meth:`PortfolioConstraints.full_investment_is_binding` first: at zero net
exposure this constraint no longer pins the scale, because ``sum(w) == 0`` is
satisfied by the empty portfolio, and gross deployment becomes a property of
the objective rather than of the constraint set. That is a real limitation of
convex optimization, not an oversight — ``sum(|w|) == g`` is not a convex
constraint, and no formulation in this module can make it one.
"""


class RelaxableConstraint(StrEnum):
    """The constraints the ladder is permitted to loosen, as stable identifiers.

    A :class:`StrEnum` so that the identifier survives serialization into a run
    artifact or an API response unchanged: an operator reading
    ``"sector_neutrality"`` in a stored backtest record six months later needs
    it to mean the same thing it meant when it was written.

    Members are ordered here for readability only. The order that matters —
    the order they are tried in — is :data:`RELAXATION_LADDER`.
    """

    SECTOR_NEUTRALITY = "sector_neutrality"
    """The per-sector net-weight target becomes a band around the target."""

    BETA_NEUTRALITY = "beta_neutrality"
    """The portfolio-beta target becomes a band around the target."""

    SECTOR_CAP = "sector_cap"
    """The maximum gross weight per sector is widened toward a stated ceiling."""

    FULL_INVESTMENT = "full_investment"
    """The net-exposure target is reduced to the most the constraints admit."""


RELAXATION_LADDER: Final = (
    RelaxableConstraint.SECTOR_NEUTRALITY,
    RelaxableConstraint.BETA_NEUTRALITY,
    RelaxableConstraint.SECTOR_CAP,
    RelaxableConstraint.FULL_INVESTMENT,
)
"""The order constraints are relaxed in. Fixed policy; a test pins it.

Relaxations are **cumulative**: rung *k* applies the first *k* entries. The
alternative — searching subsets to find the minimal relaxation — is
combinatorial in the number of relaxable constraints and would make the
optimizer's behaviour depend on a search order nobody documented.

**What cumulative relaxation costs, stated plainly.** The rung that buys
feasibility also carries every rung beneath it, and the objective will spend
that slack whether or not it needed it: a book rescued by widening the sector
cap will generally still come back sitting on the full sector-neutrality band,
because tilting into it improved expected return. So the *set* of relaxations
reported is the minimal **prefix** of this ladder, not the minimal set — and
:attr:`Relaxation.was_consumed` says which limits the returned portfolio
actually breaks, which is the fact an operator acts on, but it cannot say which
of them was necessary. Nothing here is silent; the concession is that a
relaxation report can overstate what the universe demanded, in the direction of
naming more relaxations rather than fewer.

**The ordering principle** is: relax first whatever admits the least unintended
risk per unit of feasibility bought, and never relax a constraint that bounds
loss before one that merely shapes exposure.

1. :attr:`RelaxableConstraint.SECTOR_NEUTRALITY` — an equality on eleven
   numbers, and the constraint most sensitive to universe composition: one
   sector left with two names after a borrow filter can make it unsatisfiable
   on its own. The exposure it admits is a residual tilt bounded by the band,
   spread across sectors, and it is *reported* — so it is the cheapest honest
   concession available.
2. :attr:`RelaxableConstraint.BETA_NEUTRALITY` — the same shape of concession,
   ranked second because market beta is the single largest common factor in
   equity returns. A residual beta of 0.02 contributes more variance than a
   sector tilt of the same nominal size, so it is given up only after the
   sector tilt has failed to buy feasibility.
3. :attr:`RelaxableConstraint.SECTOR_CAP` — a concentration limit. Widening it
   admits clustered idiosyncratic risk rather than a small factor residual,
   which is a larger and less diversifiable concession than either band above,
   so it comes after both. It is widened only to a stated ceiling
   (:attr:`RelaxationLadder.sector_cap_ceiling`), never removed.
4. :attr:`RelaxableConstraint.FULL_INVESTMENT` — deploy less capital. This
   *removes* risk rather than adding it, which is why it might look like it
   belongs first. It is last because it is the only rung that abandons part of
   the mandate instead of approximating it: the first three still run the
   strategy, slightly off-target, while this one declines to run part of it at
   all. It is also the only rung that can rescue a universe with too few names
   to fill the book, which is why it must exist as the final release valve
   before refusal.
"""


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """The constraint set a portfolio must satisfy. All weights are fractions of capital.

    Immutable: a relaxation produces a new instance via :meth:`relaxed_with`
    rather than mutating this one, so the requested set and the set actually
    used are both available to the caller afterwards and can be compared.

    Attributes:
        position_cap: maximum ``|w_i|`` for every name (fraction of capital).
            Never relaxed.
        sector_cap: maximum ``sum(|w_i|)`` within a sector (fraction of
            capital).
        net_exposure: target for ``sum(w)`` (fraction of capital) — the
            full-investment constraint. Enforced as an equality.
        gross_exposure: maximum ``sum(|w_i|)`` (fraction of capital). Capped at
            :data:`MAX_GROSS_EXPOSURE` by directive §1.1 and never relaxed.
        sector_neutrality_band: half-width of the permitted deviation of each
            sector's net weight from its target (fraction of capital). ``0.0``
            makes sector neutrality an exact equality, which is the requested
            default.
        beta_neutrality_band: half-width of the permitted deviation of the
            portfolio beta from its target (dimensionless, in beta units).
            ``0.0`` makes beta neutrality an exact equality.
    """

    position_cap: float = DEFAULT_POSITION_CAP
    sector_cap: float = DEFAULT_SECTOR_CAP
    net_exposure: float = DEFAULT_NET_EXPOSURE
    gross_exposure: float = MAX_GROSS_EXPOSURE
    sector_neutrality_band: float = 0.0
    beta_neutrality_band: float = 0.0

    def __post_init__(self) -> None:
        """Validate the constraint set for internal coherence.

        Raises:
            OptimizerInputError: if any value is non-finite; if a cap is
                non-positive; if the position cap exceeds the sector cap (a
                per-name limit looser than the sector limit containing it is
                incoherent); if a band is negative; if the gross exposure
                exceeds :data:`MAX_GROSS_EXPOSURE` (directive §1.1 forbids
                leverage); or if the net-exposure target exceeds the gross
                limit, which no portfolio can satisfy because
                ``|sum(w)| <= sum(|w|)``.
        """
        # Imported here rather than at module scope: the error module imports
        # nothing from this one, but keeping the dependency one-directional and
        # local documents that constraints are data and errors are behaviour.
        from backend.portfolio.optimizer_errors import OptimizerInputError

        numeric = {
            "position_cap": self.position_cap,
            "sector_cap": self.sector_cap,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "sector_neutrality_band": self.sector_neutrality_band,
            "beta_neutrality_band": self.beta_neutrality_band,
        }
        for name, value in numeric.items():
            if not _is_finite(value):
                msg = f"{name} must be a finite number; got {value!r}"
                raise OptimizerInputError(msg)

        if self.position_cap <= 0.0:
            msg = (
                f"position_cap must be > 0; got {self.position_cap!r}. A cap of zero "
                f"admits only the empty portfolio."
            )
            raise OptimizerInputError(msg)
        if self.sector_cap <= 0.0:
            msg = f"sector_cap must be > 0; got {self.sector_cap!r}"
            raise OptimizerInputError(msg)
        if self.position_cap > self.sector_cap:
            msg = (
                f"position_cap {self.position_cap!r} exceeds sector_cap "
                f"{self.sector_cap!r}: a per-name limit looser than the sector limit "
                f"that contains it cannot bind and is almost certainly a units error "
                f"(both are fractions of capital, so 2% is 0.02, not 2)."
            )
            raise OptimizerInputError(msg)
        if self.gross_exposure <= 0.0:
            msg = f"gross_exposure must be > 0; got {self.gross_exposure!r}"
            raise OptimizerInputError(msg)
        if self.gross_exposure > MAX_GROSS_EXPOSURE:
            msg = (
                f"gross_exposure {self.gross_exposure!r} exceeds the hard ceiling "
                f"{MAX_GROSS_EXPOSURE!r}. Gross exposure above 1.0 is leverage, which "
                f"directive §1.1 lists as a non-goal; this is not a default that can "
                f"be raised."
            )
            raise OptimizerInputError(msg)
        if self.net_exposure < 0.0:
            msg = (
                f"net_exposure must be >= 0; got {self.net_exposure!r}. A negative "
                f"net exposure is a net short book, which is not a configuration this "
                f"system builds."
            )
            raise OptimizerInputError(msg)
        if self.net_exposure > self.gross_exposure:
            msg = (
                f"net_exposure {self.net_exposure!r} exceeds gross_exposure "
                f"{self.gross_exposure!r}; no portfolio satisfies both, since "
                f"|sum(w)| <= sum(|w|) always."
            )
            raise OptimizerInputError(msg)
        if self.sector_neutrality_band < 0.0 or self.beta_neutrality_band < 0.0:
            msg = (
                f"neutrality bands must be >= 0; got sector "
                f"{self.sector_neutrality_band!r} and beta "
                f"{self.beta_neutrality_band!r}"
            )
            raise OptimizerInputError(msg)

    @property
    def full_investment_is_binding(self) -> bool:
        """Whether the net-exposure equality actually pins the portfolio's scale.

        ``True`` when :attr:`net_exposure` is positive. Then ``sum(w) == net``
        forces capital into positions and "fully invested" is enforced exactly.

        ``False`` at ``net_exposure == 0.0`` — the dollar-neutral long-short
        configuration — where the equality is satisfied by the empty portfolio
        and the amount actually deployed is decided by the objective's risk
        aversion against the gross limit, not by this constraint set. The
        honest expression of full investment there would be
        ``sum(|w|) == gross``, which is **not convex** (it is the boundary of a
        convex set), so no reformulation available to cvxpy enforces it. The
        optimizer therefore reports the realized gross exposure on every result
        and this flag says whether the constraint or the objective decided it.

        Returns:
            ``True`` if the full-investment constraint pins the scale.
        """
        return self.net_exposure > 0.0

    def relaxed_with(
        self,
        *,
        sector_cap: float | None = None,
        net_exposure: float | None = None,
        sector_neutrality_band: float | None = None,
        beta_neutrality_band: float | None = None,
    ) -> PortfolioConstraints:
        """Return a copy with the named limits replaced, re-validated.

        Only the four relaxable quantities can be changed. The position cap and
        the gross-exposure limit are absent by construction, which is how the
        "never relaxed" rule in the module docstring is enforced in code rather
        than in prose.

        Args:
            sector_cap: new maximum gross sector weight (fraction of capital).
            net_exposure: new target for ``sum(w)`` (fraction of capital).
            sector_neutrality_band: new per-sector band half-width (fraction of
                capital).
            beta_neutrality_band: new portfolio-beta band half-width
                (dimensionless).

        Returns:
            A new, validated :class:`PortfolioConstraints`.

        Raises:
            OptimizerInputError: if the resulting set is incoherent.
        """
        return PortfolioConstraints(
            position_cap=self.position_cap,
            sector_cap=self.sector_cap if sector_cap is None else sector_cap,
            net_exposure=self.net_exposure if net_exposure is None else net_exposure,
            gross_exposure=self.gross_exposure,
            sector_neutrality_band=(
                self.sector_neutrality_band
                if sector_neutrality_band is None
                else sector_neutrality_band
            ),
            beta_neutrality_band=(
                self.beta_neutrality_band if beta_neutrality_band is None else beta_neutrality_band
            ),
        )


DEFAULT_CONSTRAINTS: Final = PortfolioConstraints()
"""The constraint set of directive §5 Phase 9: 2% names, 20% sectors, fully invested.

Sector and beta neutrality are exact equalities here (both bands zero); the
targets themselves are supplied per call, because they depend on the universe
rather than on policy.
"""


@dataclass(frozen=True, slots=True)
class RelaxationLadder:
    """How far each rung of :data:`RELAXATION_LADDER` is permitted to loosen.

    The *order* of relaxation is policy fixed in :data:`RELAXATION_LADDER`. The
    *amounts* are here, because they are calibration rather than principle: a
    fund willing to carry a 100 bp sector tilt and one willing to carry 25 bp
    are running the same construction with a different appetite, and both
    should be expressible without editing the ladder's logic.

    Attributes:
        sector_neutrality_band: band half-width granted at the sector rung, as
            a fraction of capital. ``0.005`` permits each sector's net weight to
            sit 50 bp of capital away from its target.
        beta_neutrality_band: band half-width granted at the beta rung, in beta
            units (dimensionless). ``0.02`` permits a residual portfolio beta of
            two hundredths.
        sector_cap_ceiling: the widest the sector cap may become, as a fraction
            of capital. ``0.30`` against a 20% request: half again as
            concentrated, and no further.
        minimum_net_exposure: the least capital the book may be reduced to at
            the full-investment rung, as a fraction of capital. Below this the
            optimizer refuses rather than returning a mostly-cash portfolio: a
            book that can deploy only a third of its capital is a universe
            problem for a human to look at, and its backtested returns would be
            a statement about cash more than about the model.
    """

    sector_neutrality_band: float = 0.005
    beta_neutrality_band: float = 0.02
    sector_cap_ceiling: float = 0.30
    minimum_net_exposure: float = 0.50

    def __post_init__(self) -> None:
        """Validate the ladder's amounts.

        Raises:
            OptimizerInputError: if any amount is non-finite or negative, or if
                ``minimum_net_exposure`` exceeds :data:`MAX_GROSS_EXPOSURE`.
        """
        from backend.portfolio.optimizer_errors import OptimizerInputError

        numeric = {
            "sector_neutrality_band": self.sector_neutrality_band,
            "beta_neutrality_band": self.beta_neutrality_band,
            "sector_cap_ceiling": self.sector_cap_ceiling,
            "minimum_net_exposure": self.minimum_net_exposure,
        }
        for name, value in numeric.items():
            if not _is_finite(value):
                msg = f"{name} must be a finite number; got {value!r}"
                raise OptimizerInputError(msg)
            if value < 0.0:
                msg = f"{name} must be >= 0; got {value!r}"
                raise OptimizerInputError(msg)
        if self.minimum_net_exposure > MAX_GROSS_EXPOSURE:
            msg = (
                f"minimum_net_exposure {self.minimum_net_exposure!r} exceeds the "
                f"no-leverage ceiling {MAX_GROSS_EXPOSURE!r}"
            )
            raise OptimizerInputError(msg)


DEFAULT_RELAXATION_LADDER: Final = RelaxationLadder()
"""The shipped relaxation amounts. See :class:`RelaxationLadder` for each."""


_CONSUMPTION_TOLERANCE: Final = 1e-9
"""Slack allowed when deciding whether a relaxation was actually used.

Dimensionless in the sense that it applies in whatever unit the relaxation is
measured in; at 1e-9 of capital, or 1e-9 of a beta, it is well below anything
a solver residual or a portfolio decision operates at.
"""


@dataclass(frozen=True, slots=True)
class Relaxation:
    """A record of one constraint having been loosened, and by how much.

    This is the object that makes "I gave you a different portfolio from the
    one you asked for" expressible. A result carrying an empty tuple of these
    is the requested portfolio; a result carrying any of them is not, and the
    caller can tell which is which without inspecting the weights.

    Attributes:
        constraint: which constraint was loosened.
        original: the value requested (units in :attr:`units`).
        relaxed: the value actually used (same units).
        units: the unit both values are in, spelled out for the operator —
            directive §8 requires financial quantities to state their units, and
            a relaxation report that says "0.005" without saying "fraction of
            capital" is exactly the ambiguity that rule exists to prevent.
        rationale: why this rung sits where it does in the ladder.
        realized: where the returned solution actually landed on this axis, in
            the same units, filled in after the solve. ``None`` before the
            solve completes. Read it against :attr:`original` (see
            :attr:`was_consumed`) to learn whether the portfolio returned
            actually breaks the limit that was requested, or merely had
            permission to.
    """

    constraint: RelaxableConstraint
    original: float
    relaxed: float
    units: str
    rationale: str
    realized: float | None = None

    @property
    def was_consumed(self) -> bool | None:
        """Whether the returned portfolio actually lies outside the requested limit.

        Compares :attr:`realized` against :attr:`original` with a small
        absolute tolerance, in the direction the relaxation loosens: bands and
        caps loosen upward, so consumption means the realized value exceeded
        the original limit; the full-investment rung loosens downward, so
        consumption means the realized net exposure fell short of it.

        **This is not a claim that the relaxation was necessary**, and the
        distinction is worth being exact about. The objective spends whatever
        slack it is given — a sector band granted at rung 1 will generally come
        back consumed to its last basis point even when the feasibility was
        actually bought two rungs higher, simply because tilting into it
        improved expected return. What this property reports is the fact an
        operator has to act on: the portfolio in hand does or does not obey the
        limit that was asked for. Which rung bought feasibility is
        :attr:`~backend.portfolio.optimizer.OptimizationResult.rung` and the
        attempt log beneath it, not this.

        Returns:
            ``True`` if the solution lies outside the *original* limit,
            ``False`` if it stayed inside it despite the permission, ``None``
            if the solve has not been recorded.
        """
        if self.realized is None:
            return None
        if self.constraint is RelaxableConstraint.FULL_INVESTMENT:
            return bool(self.realized < self.original - _CONSUMPTION_TOLERANCE)
        return bool(self.realized > self.original + _CONSUMPTION_TOLERANCE)

    def describe(self) -> str:
        """Return a one-line operator-readable description of the relaxation.

        Returns:
            A sentence naming the constraint, both values, and the units.
        """
        consumed = self.was_consumed
        suffix = ""
        if consumed is not None:
            state = "consumed" if consumed else "granted but unused"
            suffix = f"; realized {self.realized!r} ({state})"
        return (
            f"{self.constraint.value}: {self.original!r} -> {self.relaxed!r} [{self.units}]{suffix}"
        )
