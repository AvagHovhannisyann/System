"""Portfolio constraint set and relaxation ladder: policy, coherence, units (P9.2).

What is pinned here is *policy*, not arithmetic. The optimizer's own tests check
that a solution satisfies its constraints; these check that the constraints mean
what the directive says they mean, and that the ladder above them cannot quietly
change:

- the shipped set is directive §5 Phase 9's — 2% names, 20% sectors, fully
  invested;
- the **position cap and the gross-exposure limit are unrelaxable by
  construction**, not by convention: :meth:`PortfolioConstraints.relaxed_with`
  has no parameter for either, so no rung can reach them even by mistake. A
  signature test enforces that, because prose in a module docstring is not a
  constraint on future code;
- the **ladder order is fixed**, and every relaxable constraint appears in it
  exactly once — a member missing from the ladder would be a relaxation the
  optimizer could name but never try;
- an incoherent constraint set is **refused up front**, with the two numbers
  that disagree named, rather than being handed to a solver that can only say
  ``infeasible``;
- :attr:`Relaxation.was_consumed` reports what it actually measures. Cumulative
  relaxation grants slack the objective will happily spend, so "consumed" means
  *the portfolio lies outside the requested limit*, which is the fact an
  operator acts on — not "this relaxation was necessary", which it cannot know.
"""

from __future__ import annotations

import inspect

import pytest

from backend.portfolio.constraints import (
    _CONSUMPTION_TOLERANCE,
    DEFAULT_CONSTRAINTS,
    DEFAULT_RELAXATION_LADDER,
    MAX_GROSS_EXPOSURE,
    RELAXATION_LADDER,
    PortfolioConstraints,
    RelaxableConstraint,
    Relaxation,
    RelaxationLadder,
)
from backend.portfolio.optimizer_errors import OptimizerInputError

# --------------------------------------------------------------------------
# The shipped set is the directive's set
# --------------------------------------------------------------------------


def test_default_constraints_are_the_directive_phase_9_set() -> None:
    assert DEFAULT_CONSTRAINTS.position_cap == 0.02
    assert DEFAULT_CONSTRAINTS.sector_cap == 0.20
    assert DEFAULT_CONSTRAINTS.net_exposure == 1.0
    assert DEFAULT_CONSTRAINTS.gross_exposure == MAX_GROSS_EXPOSURE
    # Neutrality is an exact equality as requested; the bands are what the
    # ladder opens, and they start closed.
    assert DEFAULT_CONSTRAINTS.sector_neutrality_band == 0.0
    assert DEFAULT_CONSTRAINTS.beta_neutrality_band == 0.0


def test_full_investment_binds_only_at_a_positive_net_exposure() -> None:
    assert DEFAULT_CONSTRAINTS.full_investment_is_binding is True
    dollar_neutral = PortfolioConstraints(net_exposure=0.0)
    # sum(w) == 0 is satisfied by the empty portfolio, so at zero net exposure
    # the constraint set no longer decides how much capital is deployed.
    assert dollar_neutral.full_investment_is_binding is False


# --------------------------------------------------------------------------
# Incoherent sets are refused, with the numbers named
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"position_cap": 0.0}, "position_cap must be > 0"),
        ({"position_cap": -0.02}, "position_cap must be > 0"),
        ({"sector_cap": 0.0}, "sector_cap must be > 0"),
        ({"position_cap": 0.3, "sector_cap": 0.2}, "exceeds sector_cap"),
        ({"gross_exposure": 0.0}, "gross_exposure must be > 0"),
        ({"gross_exposure": 1.5}, "leverage"),
        ({"net_exposure": -0.1}, "net_exposure must be >= 0"),
        ({"net_exposure": 0.9, "gross_exposure": 0.5}, "exceeds gross_exposure"),
        ({"sector_neutrality_band": -1e-9}, "bands must be >= 0"),
        ({"beta_neutrality_band": -1e-9}, "bands must be >= 0"),
        ({"position_cap": float("nan")}, "finite"),
        ({"sector_cap": float("inf")}, "finite"),
    ],
)
def test_incoherent_constraint_sets_are_refused(
    kwargs: dict[str, float], expected_fragment: str
) -> None:
    with pytest.raises(OptimizerInputError, match=expected_fragment):
        PortfolioConstraints(**kwargs)


def test_leverage_ceiling_is_a_ceiling_not_a_default() -> None:
    """§1.1 lists leverage as a non-goal, so gross exposure has no way up."""
    assert MAX_GROSS_EXPOSURE == 1.0
    with pytest.raises(OptimizerInputError, match="non-goal"):
        PortfolioConstraints(gross_exposure=MAX_GROSS_EXPOSURE + 1e-9)


def test_position_cap_above_sector_cap_reads_as_a_units_error() -> None:
    """2% is 0.02, not 2 — the error says so, because that is the actual mistake."""
    with pytest.raises(OptimizerInputError, match="units error"):
        PortfolioConstraints(position_cap=2.0, sector_cap=0.2)


# --------------------------------------------------------------------------
# What the ladder may and may not touch
# --------------------------------------------------------------------------


def test_relaxed_with_cannot_reach_the_position_cap_or_the_gross_limit() -> None:
    """The "never relaxed" rule is enforced by the signature, not by prose.

    A future rung that wanted to widen the position cap would have to add a
    parameter here, which is a visible edit to a documented policy rather than
    a plausible-looking keyword argument.
    """
    parameters = set(inspect.signature(PortfolioConstraints.relaxed_with).parameters)
    assert "position_cap" not in parameters
    assert "gross_exposure" not in parameters
    assert parameters == {
        "self",
        "sector_cap",
        "net_exposure",
        "sector_neutrality_band",
        "beta_neutrality_band",
    }


def test_relaxed_with_preserves_the_unrelaxable_limits() -> None:
    relaxed = DEFAULT_CONSTRAINTS.relaxed_with(sector_cap=0.30, net_exposure=0.6)
    assert relaxed.position_cap == DEFAULT_CONSTRAINTS.position_cap
    assert relaxed.gross_exposure == DEFAULT_CONSTRAINTS.gross_exposure
    assert relaxed.sector_cap == 0.30
    assert relaxed.net_exposure == 0.6
    # The original is untouched: the requested set and the used set are both
    # available afterwards, which is what makes a relaxation reportable.
    assert DEFAULT_CONSTRAINTS.sector_cap == 0.20
    assert DEFAULT_CONSTRAINTS.net_exposure == 1.0


def test_relaxed_with_revalidates_the_result() -> None:
    with pytest.raises(OptimizerInputError, match="exceeds gross_exposure"):
        DEFAULT_CONSTRAINTS.relaxed_with(net_exposure=1.5)


def test_ladder_order_is_pinned() -> None:
    """The order is a risk judgement (see RELAXATION_LADDER), so it is pinned.

    Reordering these is a policy change: it decides whether the optimizer gives
    up a beta neutrality or a concentration limit first.
    """
    assert RELAXATION_LADDER == (
        RelaxableConstraint.SECTOR_NEUTRALITY,
        RelaxableConstraint.BETA_NEUTRALITY,
        RelaxableConstraint.SECTOR_CAP,
        RelaxableConstraint.FULL_INVESTMENT,
    )


def test_every_relaxable_constraint_appears_in_the_ladder_exactly_once() -> None:
    """A member absent from the ladder is a relaxation that can be named but never tried."""
    assert len(RELAXATION_LADDER) == len(set(RELAXATION_LADDER))
    assert set(RELAXATION_LADDER) == set(RelaxableConstraint)


def test_relaxable_constraint_identifiers_are_stable_strings() -> None:
    """They are written into stored artifacts, so the wire values are pinned."""
    assert [item.value for item in RELAXATION_LADDER] == [
        "sector_neutrality",
        "beta_neutrality",
        "sector_cap",
        "full_investment",
    ]
    assert str(RelaxableConstraint.SECTOR_CAP) == "sector_cap"


# --------------------------------------------------------------------------
# The ladder's amounts
# --------------------------------------------------------------------------


def test_default_ladder_amounts_are_the_shipped_policy() -> None:
    assert DEFAULT_RELAXATION_LADDER.sector_neutrality_band == 0.005
    assert DEFAULT_RELAXATION_LADDER.beta_neutrality_band == 0.02
    assert DEFAULT_RELAXATION_LADDER.sector_cap_ceiling == 0.30
    assert DEFAULT_RELAXATION_LADDER.minimum_net_exposure == 0.50


def test_sector_cap_ceiling_is_a_widening_not_a_removal() -> None:
    """Rung 3 widens the cap toward a stated ceiling; it never lifts it."""
    assert DEFAULT_RELAXATION_LADDER.sector_cap_ceiling < 1.0
    assert DEFAULT_RELAXATION_LADDER.sector_cap_ceiling > DEFAULT_CONSTRAINTS.sector_cap


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sector_neutrality_band": -1e-9},
        {"beta_neutrality_band": float("nan")},
        {"sector_cap_ceiling": float("inf")},
        {"minimum_net_exposure": -0.1},
        {"minimum_net_exposure": MAX_GROSS_EXPOSURE + 1e-9},
    ],
)
def test_incoherent_ladder_amounts_are_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(OptimizerInputError):
        RelaxationLadder(**kwargs)


# --------------------------------------------------------------------------
# What a Relaxation record claims
# --------------------------------------------------------------------------


def _relaxation(
    constraint: RelaxableConstraint, *, original: float, relaxed: float, realized: float | None
) -> Relaxation:
    return Relaxation(
        constraint=constraint,
        original=original,
        relaxed=relaxed,
        units="fraction of capital",
        rationale="test",
        realized=realized,
    )


def test_was_consumed_is_unknown_before_the_solve() -> None:
    record = _relaxation(RelaxableConstraint.SECTOR_CAP, original=0.2, relaxed=0.3, realized=None)
    assert record.was_consumed is None
    assert "realized" not in record.describe()


def test_upward_relaxations_are_consumed_when_the_original_limit_is_exceeded() -> None:
    inside = _relaxation(RelaxableConstraint.SECTOR_CAP, original=0.2, relaxed=0.3, realized=0.19)
    outside = _relaxation(RelaxableConstraint.SECTOR_CAP, original=0.2, relaxed=0.3, realized=0.25)
    assert inside.was_consumed is False
    assert outside.was_consumed is True
    assert "granted but unused" in inside.describe()
    assert "consumed" in outside.describe()


def test_full_investment_is_the_one_relaxation_that_loosens_downward() -> None:
    """Every other rung raises a limit; this one lowers the capital deployed.

    Getting the direction wrong would report a book that deployed *less* than
    it was asked to as having stayed inside its limit.
    """
    reduced = _relaxation(
        RelaxableConstraint.FULL_INVESTMENT, original=1.0, relaxed=0.6, realized=0.6
    )
    unreduced = _relaxation(
        RelaxableConstraint.FULL_INVESTMENT, original=1.0, relaxed=0.6, realized=1.0
    )
    assert reduced.was_consumed is True
    assert unreduced.was_consumed is False


def test_consumption_ignores_movement_below_the_tolerance() -> None:
    """Solver dust is not a policy concession."""
    dust = _relaxation(
        RelaxableConstraint.SECTOR_CAP,
        original=0.2,
        relaxed=0.3,
        realized=0.2 + _CONSUMPTION_TOLERANCE / 2.0,
    )
    assert dust.was_consumed is False


def test_describe_states_the_units() -> None:
    """§8: a relaxation report that says "0.005" without saying of what is the bug."""
    record = _relaxation(
        RelaxableConstraint.SECTOR_NEUTRALITY, original=0.0, relaxed=0.005, realized=0.004
    )
    described = record.describe()
    assert "sector_neutrality" in described
    assert "fraction of capital" in described
    assert "0.005" in described
