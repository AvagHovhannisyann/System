"""P7.4: the baseline query is answered at the anchor, and at no other instant.

Three independent kinds of evidence, because one alone would be weak:

* **the statement**, compiled and read — it filters on filer, form and strictly
  earlier acceptance, and it carries **no** ``knowledge_time`` predicate, because
  that bound belongs to the session and a second weaker filter would quietly
  win;
* **the structure** — no module in the package takes an as-of, session, instant
  or clock parameter, asserted from the AST and again through
  ``inspect.signature``, so a later edit cannot add a "convenience" override
  without failing a test (the discipline D-033 used for the absent ``venue``
  parameter);
* **the behaviour** — a recording store that reports which instant it was opened
  at, and a deliberately broken one that ignores the anchor, so the difference
  the anchor makes is visible rather than argued.

The store double stands in for the database and applies D-011's read semantics
only (``backend/tests/extraction/delta/doubles.py`` says exactly what it does
and does not model). The as-of layer itself is exercised against a real
PostgreSQL in ``backend/tests/integration/``.
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
import pathlib
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from backend.extraction.tasks.delta import anchor as anchor_module
from backend.extraction.tasks.delta import outcome as outcome_module
from backend.extraction.tasks.delta import resolve as resolve_module
from backend.extraction.tasks.delta import runner as runner_module
from backend.extraction.tasks.delta import spec as spec_module
from backend.extraction.tasks.delta.anchor import (
    ANNUAL_REPORT,
    EARNINGS_CALL,
    PERIODIC_REPORT,
    BaselineIdentityError,
    BaselineSourceUnavailableError,
    BaselineTemporalIntegrityError,
    CurrentDocument,
)
from backend.extraction.tasks.delta.resolve import (
    CANDIDATE_LIMIT,
    prior_filing_statement,
    resolve_prior_filing,
)
from backend.tests.extraction.delta.doubles import (
    ExplodingStore,
    FilingRow,
    IgnoresAnchorStore,
    RecordingStore,
    utc,
)

# Constructed instants: a 2023 annual report, the 2024 one that follows it, and
# a header correction to the 2023 row published four months after the 2024
# report was filed. That last row is the restatement the anchor must exclude.
PRIOR_ACCEPTED = utc(2023, 2, 15, 21, 4)
CURRENT_ACCEPTED = utc(2024, 2, 20, 22, 11)
RESTATED_KNOWN = utc(2024, 6, 1, 13, 30)

CIK = 1725255
PRIOR_ACCESSION = "0001725255-23-000004"
CURRENT_ACCESSION = "0001725255-24-000010"


def _current(*, form_type: str = "10-K") -> CurrentDocument:
    """The 2024 annual report: the document a delta is extracted for."""
    return CurrentDocument(
        accession_number=CURRENT_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type=form_type,
        acceptance=CURRENT_ACCEPTED,
        knowledge_time=CURRENT_ACCEPTED,
        text="Item 1A. Risk Factors. Reimbursement rates may decline.",
    )


def _original_prior() -> FilingRow:
    """The 2023 annual report as it was knowable when it was filed."""
    return FilingRow(
        accession_number=PRIOR_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        valid_from=PRIOR_ACCEPTED,
        knowledge_time=PRIOR_ACCEPTED,
    )


def _restated_prior() -> FilingRow:
    """The same fact, re-versioned later — the restatement.

    Same logical key and same event time as :func:`_original_prior`, a later
    ``knowledge_time``, and a different stated filer name so the two are
    distinguishable in an assertion. This is the shape D-011 exists to express
    and the shape a delta must not silently adopt.
    """
    return FilingRow(
        accession_number=PRIOR_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corporation (as restated)",
        form_type="10-K",
        valid_from=PRIOR_ACCEPTED,
        knowledge_time=RESTATED_KNOWN,
    )


def _install(monkeypatch: pytest.MonkeyPatch, store: object) -> None:
    """Point the resolver's ``as_of`` at a store double."""
    monkeypatch.setattr(resolve_module, "as_of", store.as_of)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1. The statement
# ---------------------------------------------------------------------------


_PG_DIALECT = postgresql.dialect()  # type: ignore[no-untyped-call]


def _compiled() -> str:
    """Return the baseline statement compiled against the PostgreSQL dialect."""
    return str(prior_filing_statement(_current(), ANNUAL_REPORT).compile(dialect=_PG_DIALECT))


def test_the_statement_filters_on_filer_form_and_strictly_earlier_acceptance() -> None:
    """The three predicates that keep a wrong document out of the baseline slot."""
    sql = _compiled()
    assert "edgar_filing.cik = " in sql
    assert "edgar_filing.form_type = " in sql
    assert "edgar_filing.valid_from < " in sql
    assert "ORDER BY edgar_filing.valid_from DESC" in sql


def test_the_statement_carries_no_knowledge_time_predicate_of_its_own() -> None:
    """The bound is the session's (D-011). A second, weaker filter would win.

    Asserted as an absence because that is what the design claims: this module
    issues no ``knowledge_time`` comparison, so the only thing that can bound
    the read is the instant :func:`backend.db.as_of` was opened at — which is
    what makes the anchor load-bearing rather than decorative.

    Scoped to the ``WHERE`` clause: ``select(EdgarFiling)`` selects every column
    including ``knowledge_time``, and selecting it is not filtering on it.
    """
    sql = _compiled()
    where = sql[sql.index("WHERE") :]
    assert "knowledge_time" not in where


def test_the_statement_returns_more_than_one_row_so_a_tie_is_visible() -> None:
    """``LIMIT 1`` would let the planner break a tie nobody chose to break."""
    assert CANDIDATE_LIMIT > 1
    assert "LIMIT" in _compiled()


def test_the_statement_is_refused_for_a_class_with_no_store() -> None:
    """Earnings-call transcripts: P3.6, blocked on B1. A blocker, not a value."""
    with pytest.raises(BaselineSourceUnavailableError, match="blocked on B1"):
        prior_filing_statement(_current(), EARNINGS_CALL)


# ---------------------------------------------------------------------------
# 2. The structure
# ---------------------------------------------------------------------------

_FORBIDDEN_PARAMETERS = frozenset(
    {
        "as_of",
        "asof",
        "as_of_ts",
        "anchor_instant",
        "instant",
        "now",
        "clock",
        "session",
        "knowledge_time",
        "timestamp",
    }
)
"""Parameter names that would let a caller answer the baseline query elsewhere.

``anchor`` itself is permitted where it is *derived* — :func:`select_baseline`
takes one and cross-checks it against the document's own — but nothing may take
a bare instant or a session.
"""

_PACKAGE_MODULES = (anchor_module, outcome_module, resolve_module, runner_module, spec_module)


def _error_class_bodies(tree: ast.Module) -> set[int]:
    """Return the ids of AST nodes inside an exception class definition.

    Exception constructors are exempt from the parameter ban: an error that
    carries the offending ``knowledge_time`` so a message can quote it is
    reporting a violation, not performing one. Nothing else is exempt.
    """
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
        if any(name.endswith("Error") for name in bases):
            exempt.update(id(child) for child in ast.walk(node))
    return exempt


def test_no_callable_in_the_package_takes_an_as_of_or_a_session() -> None:
    """A structural claim, checked structurally (D-033's discipline)."""
    offenders: list[str] = []
    for module in _PACKAGE_MODULES:
        source = pathlib.Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        exempt = _error_class_bodies(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if id(node) in exempt:
                continue
            arguments = node.args
            names = {
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
            }
            offenders.extend(
                f"{module.__name__}.{node.name}({bad}=...)"
                for bad in sorted(names & _FORBIDDEN_PARAMETERS)
            )
    assert offenders == []


def test_the_resolver_signature_has_exactly_two_parameters() -> None:
    """Verified on the runtime object as well as in the source."""
    parameters = inspect.signature(resolve_prior_filing).parameters
    assert list(parameters) == ["current", "document_class"]


def test_the_resolver_obtains_its_session_only_from_as_of() -> None:
    """No writer session, no admin engine, no hand-built session.

    ``ingest_writer_session`` would read unversioned and raise; a raw engine
    would bypass the ORM hook entirely. Neither is imported here, and the test
    says so rather than trusting that nobody will add one.
    """
    source = pathlib.Path(str(resolve_module.__file__)).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    assert "as_of" in imported
    assert not imported & {"ingest_writer_session", "create_admin_engine", "Session"}


def test_the_only_session_opened_is_opened_with_the_anchor_expression() -> None:
    """``as_of(anchor.instant)`` — read out of the AST, not out of a comment."""
    source = pathlib.Path(str(resolve_module.__file__)).read_text(encoding="utf-8")
    opened: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "as_of"
        ):
            opened.append(ast.unparse(node))
    assert opened == ["as_of(anchor.instant)"]


# ---------------------------------------------------------------------------
# 3. The behaviour
# ---------------------------------------------------------------------------


async def test_the_store_is_opened_at_the_current_documents_knowledge_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not at the wall clock, and not at the document's acceptance instant."""
    current = _current()
    store = RecordingStore([_original_prior()])
    _install(monkeypatch, store)

    chosen = await resolve_prior_filing(current, document_class=ANNUAL_REPORT)

    assert store.opened_at == [current.knowledge_time]
    assert store.opened_at[0] < dt.datetime.now(dt.UTC)
    assert chosen is not None
    assert chosen.chosen_at.instant == current.knowledge_time


async def test_a_restatement_published_later_does_not_become_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline property. Both versions are in the store; only one is knowable.

    The 2023 annual report was re-versioned four months *after* the 2024 report
    was filed. Answered at the 2024 report's own knowledge time, the baseline is
    the original — the document a reader could actually have had. The restated
    version is not chosen, and its stated filer name is nowhere in the result.
    """
    current = _current()
    store = RecordingStore([_original_prior(), _restated_prior()])
    _install(monkeypatch, store)

    chosen = await resolve_prior_filing(current, document_class=ANNUAL_REPORT)

    assert chosen is not None
    assert chosen.accession_number == PRIOR_ACCESSION
    assert chosen.company_name == "AdaptHealth Corp."
    assert chosen.knowledge_time == PRIOR_ACCEPTED
    assert chosen.knowledge_time <= current.knowledge_time
    # Non-vacuity: the restatement really is in the store, and really is the
    # version a query answered today would return.
    today = store.visible_at(dt.datetime.now(dt.UTC))
    assert [row.company_name for row in today] == ["AdaptHealth Corporation (as restated)"]


async def test_an_amendment_filed_between_acceptance_and_knowability_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one shape where the event-time barrier is the *only* thing that fires.

    The current 10-K was accepted on one day and re-versioned by a header
    correction a fortnight later, so its anchor is the later instant. A second
    filing accepted in between is therefore genuinely knowable at the anchor —
    the knowledge-time barrier does not touch it — and it is still not a
    baseline, because a document filed after the one it would be compared
    against cannot be what that document changed from.
    """
    corrected = CurrentDocument(
        accession_number=CURRENT_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        acceptance=CURRENT_ACCEPTED,
        knowledge_time=utc(2024, 3, 5, 12, 0),
        text="Item 1A. Risk Factors. Reimbursement rates may decline.",
    )
    intervening = FilingRow(
        accession_number="0001725255-24-000055",
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        valid_from=utc(2024, 2, 25, 18, 0),
        knowledge_time=utc(2024, 2, 25, 18, 0),
    )
    assert intervening.knowledge_time < corrected.knowledge_time
    assert intervening.valid_from > corrected.acceptance
    _install(monkeypatch, RecordingStore([intervening]))

    with pytest.raises(BaselineIdentityError, match="at or after the current document"):
        await resolve_prior_filing(corrected, document_class=ANNUAL_REPORT)


async def test_reading_the_store_at_the_wall_clock_is_caught_rather_than_believed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect, made into an object: a store opened at ``now``, not the anchor.

    This is what the "break the temporal anchor" mutation does to shipped code.
    The second barrier — the re-verification in ``select_baseline`` — refuses the
    restated row instead of adopting it, so the run stops rather than producing
    a delta against a document nobody could read.
    """
    current = _current()
    store = IgnoresAnchorStore([_original_prior(), _restated_prior()])
    _install(monkeypatch, store)

    with pytest.raises(BaselineTemporalIntegrityError) as caught:
        await resolve_prior_filing(current, document_class=ANNUAL_REPORT)

    assert caught.value.knowledge_time == RESTATED_KNOWN
    assert caught.value.anchor == current.knowledge_time


async def test_a_row_the_as_of_layer_should_have_hidden_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken read path stops the run; it does not get filtered out quietly."""
    current = _current()
    store = RecordingStore([_restated_prior()], honour_knowledge_bound=False)
    _install(monkeypatch, store)

    with pytest.raises(BaselineTemporalIntegrityError):
        await resolve_prior_filing(current, document_class=ANNUAL_REPORT)


async def test_a_correction_knowable_before_the_anchor_is_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction, and it must work: latest knowledge *at the anchor* wins.

    A correction published before the current report was filed is what a reader
    would have seen, so it is the right baseline. Excluding every re-versioned
    row would be as wrong as including the late ones.
    """
    current = _current()
    early_correction = FilingRow(
        accession_number=PRIOR_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp. (corrected)",
        form_type="10-K",
        valid_from=PRIOR_ACCEPTED,
        knowledge_time=utc(2023, 3, 1, 9, 0),
    )
    store = RecordingStore([_original_prior(), early_correction, _restated_prior()])
    _install(monkeypatch, store)

    chosen = await resolve_prior_filing(current, document_class=ANNUAL_REPORT)

    assert chosen is not None
    assert chosen.company_name == "AdaptHealth Corp. (corrected)"


async def test_no_eligible_filing_yields_none_and_never_a_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first-time registrant: nothing was knowable, and that is not zero."""
    current = _current()
    store = RecordingStore([])
    _install(monkeypatch, store)

    assert await resolve_prior_filing(current, document_class=ANNUAL_REPORT) is None
    assert store.opened_at == [current.knowledge_time]


async def test_a_transcript_baseline_refuses_before_reading_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No transcript store exists (P3.6/B1), and no table is read as a substitute."""
    _install(monkeypatch, ExplodingStore())
    with pytest.raises(BaselineSourceUnavailableError, match="not a missing baseline"):
        await resolve_prior_filing(_current(), document_class=EARNINGS_CALL)


async def test_a_row_from_another_filer_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The query filters it out; if one arrives anyway, the two disagree."""
    current = _current()
    foreign = FilingRow(
        accession_number="0000320193-23-000106",
        cik=320193,
        company_name="Apple Inc.",
        form_type="10-K",
        valid_from=PRIOR_ACCEPTED,
        knowledge_time=PRIOR_ACCEPTED,
    )
    _install(monkeypatch, RecordingStore([foreign]))

    with pytest.raises(BaselineIdentityError, match="filed under CIK"):
        await resolve_prior_filing(current, document_class=ANNUAL_REPORT)


async def test_a_quarterly_report_resolves_against_the_previous_quarterly_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact-form matching, exercised on the wider periodic-report class."""
    current = _current(form_type="10-Q")
    annual = FilingRow(
        accession_number=PRIOR_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        valid_from=PRIOR_ACCEPTED,
        knowledge_time=PRIOR_ACCEPTED,
    )
    quarterly = FilingRow(
        accession_number="0001725255-23-000031",
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-Q",
        valid_from=utc(2023, 11, 7, 21, 0),
        knowledge_time=utc(2023, 11, 7, 21, 0),
    )
    # The real query never returns the annual report for a 10-Q — the form
    # predicate excludes it — so a store that does is a disagreement, and it
    # stops the run rather than being filtered here.
    _install(monkeypatch, RecordingStore([annual, quarterly]))
    with pytest.raises(BaselineIdentityError, match="baselines match by exact form"):
        await resolve_prior_filing(current, document_class=PERIODIC_REPORT)

    _install(monkeypatch, RecordingStore([quarterly]))
    chosen = await resolve_prior_filing(current, document_class=PERIODIC_REPORT)
    assert chosen is not None
    assert chosen.form_type == "10-Q"


def test_the_statement_is_an_orm_select_over_the_filing_entity() -> None:
    """So the as-of hook rewrites it; a Core select or textual SQL would be refused."""
    statement: Any = prior_filing_statement(_current(), ANNUAL_REPORT)
    entities = [description["entity"] for description in statement.column_descriptions]
    assert [entity.__name__ for entity in entities] == ["EdgarFiling"]
