"""P7.7 unit tests for the throughput governor (D-047).

What is proved here: that a free tier's real constraint can be *stated*, that a
request too large for the tier is refused as terminal rather than retryable, and
that a batch which cannot finish is refused before the first call rather than at
the provider two-thirds of the way through.

**No number in this file is a claim about any vendor.** The limits below are
scaffolding chosen to make the arithmetic legible; the observed figures live in
``DECISIONS.md`` D-047 with a date on them, and the shipped catalog is empty
(I3).
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.extraction.governor.errors import (
    GovernorConfigurationError,
    GovernorError,
    RequestExceedsTierCeilingError,
    ThroughputInfeasibleError,
    ThroughputLimitsNotConfiguredError,
    ThroughputLimitUnknownError,
)
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

MODEL = "groq:test-small"
OTHER_MODEL = "groq:test-large"
AT = dt.datetime(2026, 8, 4, 12, 30, tzinfo=dt.UTC)


def limit(
    model: str = MODEL,
    *,
    rpm: int | None = 30,
    rpd: int | None = 1_000,
    tpm: int | None = 8_000,
    tpd: int | None = 200_000,
) -> ModelThroughputLimit:
    """A limit with legible round numbers. Scaffolding, not a vendor fact."""
    return ModelThroughputLimit(
        model=model,
        requests_per_minute=rpm,
        requests_per_day=rpd,
        tokens_per_minute=tpm,
        tokens_per_day=tpd,
    )


# --------------------------------------------------------------------------
# The catalog ships nothing, and the book refuses to be empty
# --------------------------------------------------------------------------


def test_the_catalog_ships_no_throughput_limits() -> None:
    """A vendor's limits are a fact about their pricing page, not about this system.

    The exact counterpart of ``test_the_catalog_ships_no_prices``. A limit
    committed to a module is read as fact long after it stops being true, and a
    stale limit that is too permissive produces precisely the 429 this module
    exists to prevent.
    """
    assert dict(CATALOG_THROUGHPUT_LIMITS) == {}


def test_an_empty_book_refuses_rather_than_permitting_everything() -> None:
    """No configuration must mean no calls, not unlimited calls."""
    with pytest.raises(ThroughputLimitsNotConfiguredError, match="no throughput limits"):
        ThroughputBook(limits={})


def test_a_model_absent_from_the_book_is_refused_not_assumed_unlimited() -> None:
    """Ungoverned and unlimited are treated as the same thing, as they are for prices."""
    book = ThroughputBook.of(limit())
    with pytest.raises(ThroughputLimitUnknownError, match="no throughput limit configured"):
        book.limit_for("groq:not-configured")


def test_a_limit_filed_under_the_wrong_key_is_refused() -> None:
    """A limit looked up by the wrong key governs the wrong model's budget."""
    with pytest.raises(ThroughputLimitsNotConfiguredError, match="filed under"):
        ThroughputBook(limits={"groq:wrong-key": limit()})


def test_one_model_configured_twice_is_refused() -> None:
    """A budget that depends on argument order is not a budget."""
    with pytest.raises(ThroughputLimitsNotConfiguredError, match="configured twice"):
        ThroughputBook.of(limit(), limit(tpd=999_999))


# --------------------------------------------------------------------------
# Validating a limit
# --------------------------------------------------------------------------


def test_an_unqualified_model_is_refused() -> None:
    """The provider is read from the qualifier, never guessed."""
    with pytest.raises(ValueError, match="not qualified"):
        limit(model="test-small")


def test_a_negative_limit_is_refused_but_zero_is_legal() -> None:
    """Zero means the tier permits nothing; negative means someone mistyped.

    All four zeroed together, because "permits nothing" has to be stated
    consistently — see
    ``test_a_zero_day_allowance_still_cannot_carry_a_nonzero_minute_allowance``.
    """
    with pytest.raises(ValueError, match="must not be negative"):
        limit(tpd=-1)
    forbidden = limit(rpm=0, rpd=0, tpm=0, tpd=0)
    assert forbidden.tokens_per_day == 0
    assert forbidden.largest_admissible_request == 0


def test_a_zero_day_allowance_still_cannot_carry_a_nonzero_minute_allowance() -> None:
    """Permitting nothing must be stated consistently or it is not a statement.

    ``tokens_per_minute=8000`` beside ``tokens_per_day=0`` reads as "no tokens
    today, but eight thousand in any given minute". The containment check catches
    it, and it caught it here first: an earlier draft of these tests encoded
    "disabled" by zeroing only the daily field and the module refused the
    result — correctly.
    """
    with pytest.raises(ValueError, match="tokens_per_minute"):
        limit(tpm=8_000, tpd=0)


def test_a_limit_that_constrains_nothing_is_refused() -> None:
    """An entry with every field None is an unconfigured model in a configured model's clothes.

    It would pass ``limit_for`` and then permit everything, which is the one
    outcome this module exists to make impossible. Omitting it from the book is
    the honest encoding, because then the refusal names it.
    """
    with pytest.raises(ValueError, match="states nothing"):
        ModelThroughputLimit(model=MODEL)


@pytest.mark.parametrize(
    ("kwargs", "unit"),
    [({"rpm": 100, "rpd": 10}, "requests"), ({"tpm": 100, "tpd": 10}, "tokens")],
)
def test_a_per_minute_allowance_exceeding_the_per_day_allowance_is_refused(
    kwargs: dict[str, int], unit: str
) -> None:
    """A day contains the minute, so the pair cannot both hold.

    Catches the commonest transcription slip: reading a row of a vendor's limits
    table into the wrong pair of columns.
    """
    with pytest.raises(ValueError, match=f"{unit}_per_minute"):
        limit(**kwargs)  # type: ignore[arg-type]


def test_none_and_zero_are_different_statements() -> None:
    """``None`` is "not capped"; ``0`` is "capped at nothing". Conflating them inverts the meaning.

    Stated over the *request* limits so the two cases differ in one field only:
    a zeroed token limit would additionally trip the per-request ceiling, which
    is a different refusal and would obscure the comparison being made here.
    """
    uncapped = limit(rpd=None, tpd=None)
    forbidden = limit(rpm=0, rpd=0, tpm=None, tpd=None)
    assert daily_call_capacity(uncapped, 100) is None
    assert daily_call_capacity(forbidden, 100) == 0


# --------------------------------------------------------------------------
# The per-request ceiling — a 413, not a 429
# --------------------------------------------------------------------------


def test_a_request_larger_than_the_minute_allowance_is_refused_as_terminal() -> None:
    """One request bigger than a whole minute's allowance can never fit in a minute."""
    with pytest.raises(RequestExceedsTierCeilingError, match="cannot be sent"):
        require_request_fits(limit(tpm=8_000), 8_001)


def test_a_request_exactly_at_the_ceiling_is_admitted() -> None:
    """The ceiling is inclusive: a request that exactly fills the minute still fits in it."""
    require_request_fits(limit(tpm=8_000), 8_000)


def test_the_ceiling_refusal_is_not_a_configuration_error() -> None:
    """The taxonomy has to separate "operator has work to do" from "this cannot be sent".

    A caller that catches ``GovernorConfigurationError`` to prompt for setup must
    not swallow a ceiling breach, whose fix is smaller chunks and not
    configuration.
    """
    with pytest.raises(RequestExceedsTierCeilingError) as caught:
        require_request_fits(limit(tpm=10), 11)
    assert isinstance(caught.value, GovernorError)
    assert not isinstance(caught.value, GovernorConfigurationError)


def test_no_ceiling_applies_when_tokens_per_minute_is_uncapped() -> None:
    """An uncapped minute admits any single request, however large."""
    require_request_fits(limit(tpm=None, tpd=None), 10**9)


# --------------------------------------------------------------------------
# Daily capacity
# --------------------------------------------------------------------------


def test_capacity_is_the_tighter_of_the_two_daily_limits() -> None:
    """Both constraints must hold, so the answer is the smaller."""
    # 200_000 tokens/day / 1_000 tokens per call = 200 calls, below the 1_000 request allowance.
    assert daily_call_capacity(limit(), 1_000) == 200
    # 200_000 / 100 = 2_000 calls, now above the 1_000 request allowance, which binds instead.
    assert daily_call_capacity(limit(), 100) == 1_000


def test_tokens_per_day_binds_more_often_than_requests_per_day() -> None:
    """The headline "requests per day" figure is usually not the limit that stops you.

    This is the misreading D-047 records: a tier advertising a large daily
    request count can still admit far fewer calls once each carries a realistic
    token load.
    """
    generous_requests = limit(rpd=14_400, tpd=500_000)
    assert daily_call_capacity(generous_requests, 2_300) == 217  # not 14_400


def test_capacity_is_unbounded_when_neither_daily_limit_is_configured() -> None:
    """Reported as ``None`` rather than as a large number nobody chose."""
    assert daily_call_capacity(limit(rpd=None, tpd=None), 1_000) is None


def test_a_zero_token_call_is_refused_rather_than_implying_infinite_capacity() -> None:
    """Dividing the allowance by zero would produce arithmetic, not a fact about the tier."""
    with pytest.raises(ValueError, match="must be positive"):
        daily_call_capacity(limit(), 0)


# --------------------------------------------------------------------------
# Planning a batch
# --------------------------------------------------------------------------


def test_a_batch_that_fits_the_day_is_feasible_today() -> None:
    book = ThroughputBook.of(limit())
    plan = plan_batch(book, [MODEL], calls=150, tokens_per_call=1_000)
    assert plan.calls_per_day == 200
    assert plan.days_required == 1
    assert plan.feasible_today


def test_days_are_whole_because_the_allowance_resets_on_a_calendar_boundary() -> None:
    """Finishing a day's quota at noon buys nothing until midnight UTC.

    A fractional answer would suggest a rolling window and understate the
    wall-clock an operator actually waits.
    """
    book = ThroughputBook.of(limit())
    plan = plan_batch(book, [MODEL], calls=201, tokens_per_call=1_000)
    assert plan.days_required == 2
    assert not plan.feasible_today


def test_capacity_sums_across_models_because_free_tier_allowances_are_per_model() -> None:
    """Sharding a workload across models is the legitimate way to raise daily throughput."""
    book = ThroughputBook.of(limit(), limit(OTHER_MODEL, tpd=400_000))
    plan = plan_batch(book, [MODEL, OTHER_MODEL], calls=600, tokens_per_call=1_000)
    assert plan.calls_per_day == 200 + 400
    assert plan.days_required == 1
    assert plan.binding_models == (MODEL, OTHER_MODEL)


def test_a_batch_beyond_the_horizon_is_refused_before_the_first_call() -> None:
    """The whole point: discovered by arithmetic, not by a 429 mid-run.

    The message must carry the numbers, because the operator's next move is to
    choose which of them to change.
    """
    book = ThroughputBook.of(limit())
    with pytest.raises(ThroughputInfeasibleError) as caught:
        plan_batch(book, [MODEL], calls=200_000, tokens_per_call=1_000, horizon_days=7)
    message = str(caught.value)
    assert "1000 days" in message
    assert "200 calls/day" in message
    assert "7-day horizon" in message


def test_a_horizon_is_optional_so_the_same_arithmetic_can_report_without_gating() -> None:
    """A report wants the number; a gate wants the refusal. One function, both shapes."""
    book = ThroughputBook.of(limit())
    plan = plan_batch(book, [MODEL], calls=200_000, tokens_per_call=1_000)
    assert plan.days_required == 1_000


def test_limits_admitting_nothing_refuse_rather_than_reporting_infinite_days() -> None:
    """Zero capacity has no day count; dividing by it would be an exception in disguise."""
    book = ThroughputBook.of(limit(rpm=0, rpd=0, tpm=None, tpd=None))
    with pytest.raises(ThroughputInfeasibleError, match="cannot start"):
        plan_batch(book, [MODEL], calls=1, tokens_per_call=1_000)


def test_an_uncapped_model_makes_the_batch_unbounded_rather_than_very_large() -> None:
    """A number here would invite comparison against a horizon it does not respect."""
    book = ThroughputBook.of(limit(), limit(OTHER_MODEL, rpd=None, tpd=None))
    plan = plan_batch(book, [MODEL, OTHER_MODEL], calls=10**6, tokens_per_call=1_000)
    assert plan.days_required is None
    assert plan.calls_per_day is None
    assert not plan.feasible_today


def test_a_model_whose_ceiling_the_call_exceeds_is_refused_not_silently_dropped() -> None:
    """Skipping it would silently re-plan the batch onto the other models.

    The operator would then be handed a wall-clock they never agreed to, derived
    from a model set they did not choose.
    """
    book = ThroughputBook.of(limit(), limit(OTHER_MODEL, tpm=500, tpd=400_000))
    with pytest.raises(RequestExceedsTierCeilingError):
        plan_batch(book, [MODEL, OTHER_MODEL], calls=10, tokens_per_call=1_000)


def test_planning_with_no_models_is_refused() -> None:
    book = ThroughputBook.of(limit())
    with pytest.raises(ValueError, match="at least one model"):
        plan_batch(book, [], calls=1, tokens_per_call=1_000)


@pytest.mark.parametrize(("calls", "horizon"), [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_nonsensical_batch_parameters_are_refused(calls: int, horizon: int) -> None:
    book = ThroughputBook.of(limit())
    with pytest.raises(ValueError, match="must be positive"):
        plan_batch(book, [MODEL], calls=calls, tokens_per_call=1_000, horizon_days=horizon)


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


def test_window_keys_are_utc_calendar_periods() -> None:
    assert ThroughputWindow.MINUTE.key(AT) == "2026-08-04T12:30"
    assert ThroughputWindow.DAY.key(AT) == "2026-08-04"


def test_a_local_time_is_converted_not_reinterpreted() -> None:
    """Providers reset these windows on their own UTC clock, not the caller's."""
    local = dt.datetime(2026, 8, 4, 20, 30, tzinfo=dt.timezone(dt.timedelta(hours=-8)))
    assert ThroughputWindow.DAY.key(local) == "2026-08-05"
    assert ThroughputWindow.MINUTE.key(local) == "2026-08-05T04:30"


def test_a_naive_timestamp_names_no_window() -> None:
    """Two machines would disagree about which day's allowance a call spent."""
    with pytest.raises(ValueError, match="naive"):
        ThroughputWindow.DAY.key(dt.datetime(2026, 8, 4, 12, 30))  # noqa: DTZ001


# --------------------------------------------------------------------------
# Non-vacuity
# --------------------------------------------------------------------------


def test_the_plan_reports_the_inputs_it_was_derived_from() -> None:
    """A plan that dropped its inputs could not be checked against a later run."""
    book = ThroughputBook.of(limit())
    plan = plan_batch(book, [MODEL], calls=42, tokens_per_call=1_500)
    assert plan == BatchPlan(
        calls=42,
        tokens_per_call=1_500,
        calls_per_day=133,
        days_required=1,
        binding_models=(MODEL,),
    )
