"""P5.1: the declaration schema, and the arithmetic the lag control rests on.

Three things are pinned here, in rising order of how much damage their absence
would do.

**Units are mandatory and non-empty.** Directive §8 names unit confusion as the
most common silent bug class in this domain. A ``float64`` column carries no
unit, so the declaration is the only place the answer can live; a spec that is
allowed to omit it produces a catalog that *looks* complete and tells a reader
nothing.

**A malformed declaration cannot be constructed.** Every refusal below is
asserted to be a :class:`FeatureSpecError` — a member of this package's
taxonomy — and not an incidental ``TypeError`` or ``AttributeError`` from a
string method. The distinction matters because declarations do not only arrive
from type-checked call sites: the module says so itself, guarding a bare number
passed as ``availability_lag`` and a bare string passed as ``source_tables``.
Anything that reaches ``__post_init__`` from a config file or a fixture must
fail the same way, or the "every check is at construction" promise is only true
for callers mypy already covered.

**``knowledge_cutoff`` is exactly ``midnight_utc(compute_date) - lag``.** This
is the number the whole I1 control at the feature layer reduces to; the compute
path adds no arithmetic of its own. It is checked against a hand-computed
example, against an independent oracle over random dates and lags, and for
monotonicity: a longer declared lag can never produce a fresher instant. That
last property is what makes "when the true lag is uncertain, round up" safe
advice rather than a hope.

The properties over :func:`hypothesis.strategies.dates` deliberately run to the
edges of the ``date`` domain. That is not decoration — it is where the
subtraction stops being representable, and a bare ``OverflowError`` escaping
from a spec's own method is a failure this file is entitled to catch.
"""

from __future__ import annotations

import datetime as dt

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.features.errors import FeatureComputeError, FeatureSpecError
from backend.features.spec import (
    MAX_AVAILABILITY_LAG,
    MAX_FEATURES,
    FeatureSpec,
    compute_instant,
)

# ---------------------------------------------------------------------------
# Helpers and strategies
# ---------------------------------------------------------------------------


def make_spec(
    name: str = "probe_feature",
    *,
    definition: str = "A declaration used to exercise the framework.",
    units: str = "dimensionless ratio",
    availability_lag: object = dt.timedelta(days=45),
    source_tables: object = frozenset({"price_bar"}),
) -> FeatureSpec:
    """Build a well-formed declaration, overriding one field at a time.

    The odd ``object`` annotations on the last two parameters are the point:
    tests here pass values a type checker would reject, because config files
    and fixtures do too.
    """
    return FeatureSpec(
        name=name,
        definition=definition,
        units=units,
        availability_lag=availability_lag,  # type: ignore[arg-type]
        source_tables=source_tables,  # type: ignore[arg-type]
    )


_lags = st.timedeltas(min_value=dt.timedelta(0), max_value=MAX_AVAILABILITY_LAG)
"""Every declarable lag, from zero to the ten-year guard, inclusive."""


def _span_to_datetime_min(compute_date: dt.date) -> dt.timedelta:
    """Return the distance from ``datetime.min`` to midnight opening ``compute_date``.

    An oracle independent of the implementation: it counts days with
    :meth:`datetime.date.toordinal` instead of doing ``datetime`` arithmetic, so
    it can say whether ``midnight(compute_date) - lag`` is representable without
    performing the subtraction under test.
    """
    return dt.timedelta(days=compute_date.toordinal() - dt.date.min.toordinal())


# ---------------------------------------------------------------------------
# A well-formed declaration
# ---------------------------------------------------------------------------


def test_a_declaration_keeps_every_field_verbatim() -> None:
    spec = FeatureSpec(
        name="book_to_price",
        definition="Common equity over market capitalization.",
        units="dimensionless ratio",
        availability_lag=dt.timedelta(days=45),
        source_tables=frozenset({"fundamentals", "price_bar"}),
    )
    assert spec.name == "book_to_price"
    assert spec.definition == "Common equity over market capitalization."
    assert spec.units == "dimensionless ratio"
    assert spec.availability_lag == dt.timedelta(days=45)
    assert spec.source_tables == frozenset({"fundamentals", "price_bar"})


def test_a_declaration_is_frozen_after_construction() -> None:
    """Nothing may edit a spec once registered: a config hash over it (I2) is a lie otherwise."""
    spec = make_spec()
    with pytest.raises(AttributeError):
        spec.availability_lag = dt.timedelta(0)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        spec.units = "percent"  # type: ignore[misc]


def test_a_declaration_is_hashable_and_compares_by_value() -> None:
    """Specs go into config hashes and sets, so equal declarations must collide."""
    assert make_spec() == make_spec()
    assert hash(make_spec()) == hash(make_spec())
    assert len({make_spec(), make_spec(), make_spec("other_feature")}) == 2


def test_a_mutable_source_table_set_is_frozen_into_the_declaration() -> None:
    """The caller's set is copied: mutating it afterwards must not edit the spec."""
    tables = {"price_bar", "fundamentals"}
    spec = make_spec(source_tables=tables)
    tables.add("sneaked_in_later")
    assert isinstance(spec.source_tables, frozenset)
    assert spec.source_tables == frozenset({"price_bar", "fundamentals"})


# ---------------------------------------------------------------------------
# Units: required, non-empty, and refused in the package's own taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("units", ["", " ", "\t", "\n  \n"])
def test_empty_units_are_refused(units: str) -> None:
    with pytest.raises(FeatureSpecError, match="empty units"):
        make_spec(units=units)


def test_the_units_refusal_names_the_feature_and_the_directive_reason() -> None:
    with pytest.raises(FeatureSpecError) as excinfo:
        make_spec("gross_profitability", units="")
    message = str(excinfo.value)
    assert "gross_profitability" in message
    assert "§8" in message


@pytest.mark.parametrize("units", [None, 0, 1.0, b"bps", ["bps"]])
def test_units_that_are_not_text_are_refused_as_a_spec_error(units: object) -> None:
    """A non-string unit must fail in this package's taxonomy, not as AttributeError.

    Declarations arrive from config files as well as from type-checked code;
    the module guards a bare number in ``availability_lag`` for exactly that
    reason, and a unit is no different.
    """
    with pytest.raises(FeatureSpecError, match="units"):
        make_spec(units=units)  # type: ignore[arg-type]


def test_a_unit_string_is_taken_verbatim_including_surrounding_text() -> None:
    """Units are free text: the check is non-emptiness, not a vocabulary."""
    assert make_spec(units="basis points of notional").units == "basis points of notional"
    assert make_spec(units="USD").units == "USD"


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("definition", ["", "   ", "\n"])
def test_an_empty_definition_is_refused(definition: str) -> None:
    with pytest.raises(FeatureSpecError, match="empty definition"):
        make_spec(definition=definition)


@pytest.mark.parametrize("definition", [None, 3, b"prose"])
def test_a_definition_that_is_not_text_is_refused_as_a_spec_error(definition: object) -> None:
    with pytest.raises(FeatureSpecError, match="definition"):
        make_spec(definition=definition)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["momentum_12_1", "roic", "book_to_price", "x1", "amihud_illiquidity_21d", "a"],
)
def test_snake_case_names_are_accepted(name: str) -> None:
    assert make_spec(name).name == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Momentum",
        "MOMENTUM",
        "momentum-12-1",
        "momentum 12 1",
        "_momentum",
        "momentum_",
        "momentum__12",
        "12_momentum",
        "momentum.12",
        "momentum\n",
    ],
)
def test_names_outside_snake_case_are_refused(name: str) -> None:
    with pytest.raises(FeatureSpecError, match="snake_case"):
        make_spec(name)


@pytest.mark.parametrize("name", [None, 12, b"momentum"])
def test_a_name_that_is_not_text_is_refused_as_a_spec_error(name: object) -> None:
    with pytest.raises(FeatureSpecError, match="name"):
        make_spec(name)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Source tables
# ---------------------------------------------------------------------------


def test_a_declaration_with_no_source_tables_is_refused() -> None:
    """A feature that reads nothing has no lag to enforce and nothing to be point-in-time about."""
    with pytest.raises(FeatureSpecError, match="no source_tables"):
        make_spec(source_tables=frozenset())
    with pytest.raises(FeatureSpecError, match="no source_tables"):
        make_spec(source_tables=set())
    with pytest.raises(FeatureSpecError, match="no source_tables"):
        make_spec(source_tables=[])


def test_a_bare_string_of_source_tables_is_refused_rather_than_split_into_letters() -> None:
    """``"price_bar"`` would silently become eight one-character table names."""
    with pytest.raises(FeatureSpecError, match="bare string"):
        make_spec(source_tables="price_bar")


@pytest.mark.parametrize(
    "table",
    ["PriceBar", "PRICE_BAR", "public.price_bar", '"price_bar"', "price bar", "1price", ""],
)
def test_source_tables_must_be_lowercase_unquoted_identifiers(table: str) -> None:
    with pytest.raises(FeatureSpecError, match="lowercase unquoted"):
        make_spec(source_tables=frozenset({table}))


def test_a_source_table_that_is_not_text_is_refused_as_a_spec_error() -> None:
    with pytest.raises(FeatureSpecError, match="source table"):
        make_spec(source_tables=frozenset({"price_bar", 7}))


def test_source_tables_accept_any_iterable_of_names() -> None:
    assert make_spec(source_tables=["price_bar", "fundamentals", "price_bar"]).source_tables == (
        frozenset({"price_bar", "fundamentals"})
    )


# ---------------------------------------------------------------------------
# Availability lag
# ---------------------------------------------------------------------------


def test_a_zero_lag_is_legal() -> None:
    """Prices already in the store at the previous close are knowable immediately."""
    assert make_spec(availability_lag=dt.timedelta(0)).availability_lag == dt.timedelta(0)


def test_a_negative_lag_is_refused_as_lookahead() -> None:
    with pytest.raises(FeatureSpecError, match="negative availability_lag"):
        make_spec(availability_lag=dt.timedelta(microseconds=-1))


@pytest.mark.parametrize("lag", [45, 45.0, "45 days", None, dt.date(2026, 1, 1)])
def test_a_lag_that_is_not_a_timedelta_is_refused(lag: object) -> None:
    """A bare number has no unit — days? seconds? — which is the §8 bug class exactly."""
    with pytest.raises(FeatureSpecError, match="not a timedelta"):
        make_spec(availability_lag=lag)


def test_the_maximum_lag_is_declarable_and_one_microsecond_more_is_not() -> None:
    assert make_spec(availability_lag=MAX_AVAILABILITY_LAG).availability_lag == MAX_AVAILABILITY_LAG
    with pytest.raises(FeatureSpecError, match="above the"):
        make_spec(availability_lag=MAX_AVAILABILITY_LAG + dt.timedelta(microseconds=1))


# ---------------------------------------------------------------------------
# compute_instant: the calendar-date convention
# ---------------------------------------------------------------------------


def test_a_compute_date_denotes_midnight_utc_opening_it() -> None:
    assert compute_instant(dt.date(2026, 3, 1)) == dt.datetime(2026, 3, 1, tzinfo=dt.UTC)


def test_a_datetime_compute_date_is_refused_rather_than_truncated() -> None:
    """Silently dropping 16:00 would hand back an instant the caller never asked for."""
    with pytest.raises(FeatureComputeError, match="must be a date, not a datetime"):
        compute_instant(dt.datetime(2026, 3, 1, 16, 0, tzinfo=dt.UTC))


@given(compute_date=st.dates())
def test_the_instant_is_always_tz_aware_utc_and_at_midnight(compute_date: dt.date) -> None:
    instant = compute_instant(compute_date)
    assert instant.utcoffset() == dt.timedelta(0)
    assert (instant.hour, instant.minute, instant.second, instant.microsecond) == (0, 0, 0, 0)
    assert instant.date() == compute_date


# ---------------------------------------------------------------------------
# knowledge_cutoff: the number the whole lag control reduces to
# ---------------------------------------------------------------------------


def test_the_worked_example_from_the_declaration_docstring() -> None:
    """45 days before 2026-03-01 is 2026-01-15 — checked by hand, not by re-running the code."""
    spec = make_spec("book_to_price", availability_lag=dt.timedelta(days=45))
    assert spec.knowledge_cutoff(dt.date(2026, 3, 1)) == dt.datetime(2026, 1, 15, tzinfo=dt.UTC)


def test_a_zero_lag_feature_sees_up_to_midnight_opening_the_compute_date() -> None:
    """And therefore nothing published during the compute date itself."""
    spec = make_spec(availability_lag=dt.timedelta(0))
    assert spec.knowledge_cutoff(dt.date(2026, 3, 1)) == dt.datetime(2026, 3, 1, tzinfo=dt.UTC)


def test_the_cutoff_is_refused_for_a_datetime_compute_date() -> None:
    with pytest.raises(FeatureComputeError, match="must be a date, not a datetime"):
        make_spec().knowledge_cutoff(dt.datetime(2026, 3, 1, 16, tzinfo=dt.UTC))


@given(compute_date=st.dates(), lag=_lags)
@hypothesis_settings(max_examples=500)
def test_the_cutoff_is_exactly_the_instant_minus_the_lag(
    compute_date: dt.date, lag: dt.timedelta
) -> None:
    """The one arithmetic identity the compute path is entitled to assume.

    Where the subtraction leaves the representable ``datetime`` range the method
    must still fail inside this package's taxonomy — an ``OverflowError``
    escaping from a declaration's own method carries no feature name, no date
    and no lag.
    """
    spec = make_spec(availability_lag=lag)
    representable = lag <= _span_to_datetime_min(compute_date)
    if not representable:
        with pytest.raises(FeatureComputeError):
            spec.knowledge_cutoff(compute_date)
        return
    cutoff = spec.knowledge_cutoff(compute_date)
    assert cutoff == compute_instant(compute_date) - lag
    assert cutoff.utcoffset() == dt.timedelta(0)


@given(compute_date=st.dates(min_value=dt.date(1900, 1, 1)), lag=_lags)
@hypothesis_settings(max_examples=500)
def test_the_cutoff_is_never_later_than_the_compute_date_itself(
    compute_date: dt.date, lag: dt.timedelta
) -> None:
    """No declaration, whatever its lag, may read past the instant the date opens."""
    cutoff = make_spec(availability_lag=lag).knowledge_cutoff(compute_date)
    assert cutoff <= compute_instant(compute_date)


@given(
    compute_date=st.dates(min_value=dt.date(1900, 1, 1)),
    short=_lags,
    extra=st.timedeltas(min_value=dt.timedelta(0), max_value=dt.timedelta(days=365)),
)
@hypothesis_settings(max_examples=500)
def test_a_longer_lag_never_yields_a_fresher_cutoff(
    compute_date: dt.date, short: dt.timedelta, extra: dt.timedelta
) -> None:
    """Monotonicity is what makes "when in doubt, round the lag up" safe advice.

    Over-declaring must only ever cost staleness. If a longer lag could produce
    a fresher instant, rounding up would silently introduce the lookahead it is
    meant to avoid.
    """
    longer = min(short + extra, MAX_AVAILABILITY_LAG)
    shorter_cutoff = make_spec(availability_lag=short).knowledge_cutoff(compute_date)
    longer_cutoff = make_spec(availability_lag=longer).knowledge_cutoff(compute_date)
    assert longer_cutoff <= shorter_cutoff


@given(
    earlier=st.dates(min_value=dt.date(1900, 1, 1), max_value=dt.date(9000, 1, 1)),
    gap=st.integers(min_value=0, max_value=4000),
)
@hypothesis_settings(max_examples=300)
def test_a_later_compute_date_never_yields_an_older_cutoff(earlier: dt.date, gap: int) -> None:
    """The cutoff tracks the compute date one-for-one, so a replay is ordered in time."""
    spec = make_spec(availability_lag=dt.timedelta(days=45))
    later = earlier + dt.timedelta(days=gap)
    assert spec.knowledge_cutoff(later) - spec.knowledge_cutoff(earlier) == dt.timedelta(days=gap)


# ---------------------------------------------------------------------------
# The cap constant itself
# ---------------------------------------------------------------------------


def test_the_cap_constant_is_the_directive_number() -> None:
    """Directive §5 Phase 5: thirty features total, LLM features from Phase 7 included."""
    assert MAX_FEATURES == 30
