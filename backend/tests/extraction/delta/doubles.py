"""Declared doubles for the P7.4 delta tests, and what each one stands in for.

Three doubles, and the boundary each replaces is stated so no test can be read
as evidence about something it never touched:

:class:`RecordingStore`
    **Stands in for the database, not for the as-of layer.** It records the
    instant it was opened at and serves the rows a test declared, applying
    exactly one rule of its own: D-011's read semantics — a row is visible when
    ``knowledge_time <= as_of``, and among visible versions of one fact the
    latest ``knowledge_time`` wins. That rule is two lines and it is written out
    below so a reader can check it against
    :mod:`backend.db.asof`. Everything else — which filer, which form, which
    acceptance window — is the test's construction, because those predicates are
    asserted separately against the compiled statement in ``test_resolve.py``.

    The as-of layer itself is exercised against a real PostgreSQL in
    ``backend/tests/integration/test_asof_layer.py`` and
    ``test_bitemporal_property.py``. Nothing here re-proves it; what these tests
    prove is **which instant the resolver opens it at** and **what the resolver
    does with what comes back**.

:class:`RecordingClient`
    A model client that records requests and replays canned text. The canned
    text is the test's input, never a provider's output. B4 leaves no provider
    key in this repository, so no code under test has ever spoken to a model,
    and no assertion here is about how one behaves.

:class:`StubTextSource`
    Supplies a baseline document's body from a mapping the test declares.
    ``edgar_filing`` stores a manifest and a URL, never a body, so this is the
    seam :class:`~backend.extraction.tasks.delta.runner.PriorTextSource` exists
    to be — not a stand-in for a connector that exists.

Every document body used through these doubles is a captured EDGAR response
from ``backend/tests/fixtures/edgar/`` (see ``edgar_text.py``). Instants are
constructed, which is what a test of temporal ordering has to do; they are
labelled at their use sites.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from backend.extraction.tasks.client import ModelRequest, ModelResponse

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from backend.extraction.tasks.delta.anchor import PriorDocumentRef
    from backend.extraction.tasks.delta.runner import PriorDocumentText

MODEL: Final = "anthropic:a-cost-tier-model"
"""A qualified model identifier for these tests. Names no real deployment (B4)."""


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    """Return a timezone-aware UTC instant.

    Args:
        year: calendar year.
        month: calendar month.
        day: calendar day.
        hour: hour of day.
        minute: minute of hour.

    Returns:
        The instant.
    """
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC)


@dataclass(frozen=True, slots=True)
class FilingRow:
    """One ``edgar_filing`` row as the store would hold it.

    Attribute names match :class:`backend.db.models.EdgarFiling` exactly —
    including ``valid_from`` for the acceptance instant — because
    :func:`~backend.extraction.tasks.delta.resolve.resolve_prior_filing` reads
    them off whatever the session returns, and a double with different names
    would let a rename in the model pass unnoticed.
    """

    accession_number: str
    cik: int
    company_name: str
    form_type: str
    valid_from: dt.datetime
    knowledge_time: dt.datetime


class _Scalars:
    """What ``session.scalars(...)`` returns: something with ``.all()``."""

    def __init__(self, rows: Sequence[FilingRow]) -> None:
        """Hold the rows.

        Args:
            rows: the visible rows, in the order the query would return them.
        """
        self._rows = tuple(rows)

    def all(self) -> tuple[FilingRow, ...]:
        """Return every row."""
        return self._rows


class _Session:
    """A session that answers one query with a fixed, pre-filtered row set."""

    def __init__(self, rows: Sequence[FilingRow]) -> None:
        """Hold the rows this session can see.

        Args:
            rows: rows already reduced by the visibility rule.
        """
        self._rows = tuple(rows)
        self.statements: list[object] = []

    async def scalars(self, statement: object) -> _Scalars:
        """Record the statement and return the visible rows.

        Args:
            statement: the ORM select. Recorded rather than executed — its shape
                is asserted against the compiled SQL in ``test_resolve.py``,
                which is a stronger check than re-implementing it here.

        Returns:
            The rows.
        """
        self.statements.append(statement)
        return _Scalars(self._rows)


class RecordingStore:
    """Stands in for the database. Records the as-of it was opened at.

    Applies D-011's read semantics and nothing else (module docstring).
    """

    def __init__(self, rows: Sequence[FilingRow], *, honour_knowledge_bound: bool = True) -> None:
        """Build the store.

        Args:
            rows: every version of every row the store holds, in any order.
            honour_knowledge_bound: when ``False`` the store returns every row
                regardless of its knowledge time. That is a **broken** as-of
                layer, used deliberately to prove that
                :func:`~backend.extraction.tasks.delta.anchor.select_baseline`
                refuses a row the layer should have hidden instead of adopting
                it. It is never the default.
        """
        self._rows = tuple(rows)
        self._honour = honour_knowledge_bound
        self.opened_at: list[dt.datetime] = []
        self.sessions: list[_Session] = []

    def visible_at(self, instant: dt.datetime) -> tuple[FilingRow, ...]:
        """Return the rows a D-011 as-of read at ``instant`` would yield.

        Two rules, exactly as :mod:`backend.db.asof` documents them:

        1. a version exists only when ``knowledge_time <= instant`` (boundary
           inclusive);
        2. among versions of one fact — one ``(accession, cik, valid_from)`` —
           the greatest ``knowledge_time`` wins.

        Args:
            instant: the as-of timestamp.

        Returns:
            The winning versions, newest acceptance first, which is the order
            the real query's ``ORDER BY valid_from DESC`` produces.
        """
        candidates = [
            row for row in self._rows if not self._honour or row.knowledge_time <= instant
        ]
        winners: dict[tuple[str, int, dt.datetime], FilingRow] = {}
        for row in candidates:
            key = (row.accession_number, row.cik, row.valid_from)
            held = winners.get(key)
            if held is None or row.knowledge_time > held.knowledge_time:
                winners[key] = row
        return tuple(sorted(winners.values(), key=lambda row: row.valid_from, reverse=True))

    def as_of(self, instant: dt.datetime) -> _StoreSession:
        """Open the store at ``instant``, recording that it was asked for.

        Args:
            instant: the as-of timestamp the caller pinned to.

        Returns:
            An async context manager yielding a session over the visible rows.
        """
        self.opened_at.append(instant)
        session = _Session(self.visible_at(instant))
        self.sessions.append(session)
        return _StoreSession(session)


class _StoreSession:
    """Async context manager wrapper, matching ``as_of``'s call shape."""

    def __init__(self, session: _Session) -> None:
        """Hold the session to yield.

        Args:
            session: the session handed to the caller.
        """
        self._session = session

    async def __aenter__(self) -> _Session:
        """Return the session."""
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        """Close nothing; the double holds no resources."""
        return


class IgnoresAnchorStore(RecordingStore):
    """A store that answers every query at the wall clock instead of the anchor.

    Not a fixture for convenience: it is the *defect* — reading the store as it
    is today rather than as it was when the current document was filed — made
    into an object, so a test can show what that defect does without editing
    shipped code.
    """

    def as_of(self, instant: dt.datetime) -> _StoreSession:
        """Discard ``instant`` and open the store at ``now``.

        Args:
            instant: the anchor the caller asked for. Deliberately ignored.

        Returns:
            An async context manager over the rows visible right now.
        """
        del instant
        return super().as_of(dt.datetime.now(dt.UTC))


class ExplodingStore:
    """A store that fails the test if it is ever opened.

    Used where the resolver must refuse before reading anything. A test that
    only checked the raised exception would pass just as happily if the store
    had been opened and the result discarded.
    """

    def as_of(self, instant: dt.datetime) -> _StoreSession:
        """Fail loudly.

        Args:
            instant: the instant the caller asked for, quoted in the failure.

        Raises:
            AssertionError: always.
        """
        msg = f"the resolver opened the store at {instant.isoformat()}, which it must not have"
        raise AssertionError(msg)


class RecordingClient:
    """A model client that records every request and replays canned responses.

    The canned text is supplied by the test that constructs it, so it is the
    test's input rather than any provider's output. Its purpose is to make the
    request *observable*, which is what the anonymization assertions need.
    """

    def __init__(self, responses: Sequence[str] | str = "{}") -> None:
        """Build the double.

        Args:
            responses: one response per call, in order, or a single string
                replayed for every call.
        """
        self.requests: list[ModelRequest] = []
        self._responses = [responses] if isinstance(responses, str) else list(responses)

    @property
    def calls(self) -> int:
        """How many requests this client was handed (count)."""
        return len(self.requests)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the request and return the next canned response.

        Args:
            request: the request the pipeline built.

        Returns:
            The canned response.
        """
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        return ModelResponse(
            text=self._responses[index],
            model=request.model,
            input_tokens=13,
            output_tokens=5,
            latency_ms=2.5,
        )


class StubTextSource:
    """Serves baseline bodies from a mapping the test declares.

    Returns ``None`` for an accession it does not hold, which is exactly what a
    real source must do when it cannot supply a body: the run then records
    ``PRIOR_TEXT_UNAVAILABLE`` rather than comparing against nothing.
    """

    def __init__(self, bodies: Mapping[str, PriorDocumentText]) -> None:
        """Build the source.

        Args:
            bodies: accession number to the document text and any extra entity
                declarations that go with it.
        """
        self._bodies = dict(bodies)
        self.asked: list[str] = []

    async def load(self, ref: PriorDocumentRef) -> PriorDocumentText | None:
        """Return the body for ``ref``, or ``None``.

        Args:
            ref: the baseline chosen at the anchor.

        Returns:
            The body, or ``None`` when this source does not hold it.
        """
        self.asked.append(ref.accession_number)
        return self._bodies.get(ref.accession_number)


class EmptyTextSource:
    """A source that holds nothing at all, for the unavailable-text outcome."""

    def __init__(self) -> None:
        """Record nothing; hold nothing."""
        self.asked: list[str] = []

    async def load(self, ref: PriorDocumentRef) -> PriorDocumentText | None:
        """Return ``None`` for every reference.

        Args:
            ref: the baseline chosen at the anchor.

        Returns:
            ``None``, always.
        """
        self.asked.append(ref.accession_number)
        return None


class UnaskedTextSource:
    """A source that fails the test if it is ever asked for a body."""

    async def load(self, ref: PriorDocumentRef) -> PriorDocumentText | None:
        """Fail loudly.

        Args:
            ref: the reference that should never have been resolved.

        Raises:
            AssertionError: always.
        """
        msg = f"a baseline body was requested for {ref.accession_number!r}, which must not happen"
        raise AssertionError(msg)
