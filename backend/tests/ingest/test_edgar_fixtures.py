"""Captured EDGAR responses and the transport that serves them (P3.2 test support).

Every byte the EDGAR tests parse came off ``www.sec.gov`` and is committed under
``backend/tests/fixtures/edgar/``. Each file carries its own provenance block —
source URL, capture date, and why that particular filing was chosen — in a form
the corresponding parser ignores (an HTML comment, ``#`` preamble lines, a
JSON ``_provenance`` key). Nothing here was written to look like EDGAR output;
if a case is not in the fixtures it is not asserted.

The provenance blocks are the only edit made to the captured bytes, and
``test_provenance_block_is_inert`` proves they change no parse result.

This module holds no assertions of its own beyond that one; it exists so the
unit tests and the integration tests load the same captured responses through
the same transport rather than two drifting copies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx

from backend.ingest.edgar.parse import parse_daily_index, parse_filing_header

if TYPE_CHECKING:
    from collections.abc import Iterable

FIXTURE_DIR: Final = Path(__file__).resolve().parents[1] / "fixtures" / "edgar"
"""Directory holding the captured EDGAR responses."""

PROVENANCE_END: Final = "PROVENANCE-END"
"""Marker ending the provenance block prepended to captured HTML fixtures."""

QUARTER_LISTING_PATH: Final = "/Archives/edgar/daily-index/2024/QTR1/index.json"
"""Path of the one quarter listing the fixtures cover."""

# Accessions whose -index-headers.html is committed, with the daily index they
# were disseminated in. Kept explicit so a test that references a filing not in
# the fixtures fails loudly rather than silently hitting a 404 branch.
CAPTURED_FILINGS: Final = {
    "0000950170-24-012183": "CORRESP",
    "0000950170-24-029225": "8-K",
    "0000950170-24-030003": "8-K",
    "0000950172-24-000037": "D",
    "0001104659-24-082105": "6-K",
    "0001214659-24-004416": "8-K",
    "0001225208-24-002776": "4",
    "0001225208-24-004041": "4",
    "0001193805-24-000360": "4",
    "0001225208-24-004467": "4/A",
    "0001243233-24-000002": "D",
}


class NonBlockingClock:
    """A virtual clock that advances a full second per sleep.

    Used by the EDGAR tests that assert *what* the connector fetches and
    builds, not how it paces: the rate limiter never blocks, so those tests
    cost no real time. Pacing itself is asserted separately, against wall time.

    Advancing generously rather than by exactly the requested delay is
    deliberate, and works around a defect in the shared token bucket rather
    than hiding one in this connector: with a virtual clock advanced by exactly
    the delay it asked for, floating-point accumulation leaves the refill one
    ulp short of a whole token (``0.4 - 0.30000000000000004`` is
    ``0.09999999999999998``, and times a rate of 10 that is
    ``0.9999999999999998``), after which the next computed delay rounds to zero
    and :meth:`~backend.ingest.ratelimit.TokenBucket.acquire` spins forever. It
    is reported against P3.1 rather than patched here.

    Attributes:
        seconds: current monotonic reading, in seconds.
        sleeps: every delay the limiter asked for, in order, in seconds.
    """

    def __init__(self) -> None:
        """Start the clock at zero with no recorded sleeps."""
        self.seconds = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        """Return the current reading, in seconds."""
        return self.seconds

    async def sleep(self, delay: float) -> None:
        """Record ``delay`` and advance the clock by at least a second."""
        self.sleeps.append(delay)
        self.seconds += max(delay, 1.0)


def read_fixture(name: str) -> str:
    """Return a captured response body, decoded exactly as the client decodes it.

    Args:
        name: file name under ``backend/tests/fixtures/edgar/``.

    Returns:
        The file's bytes decoded as latin-1 — the same decoding
        :meth:`backend.ingest.edgar.client.EdgarClient.get_text` applies, so a
        parser test sees precisely what a live run would.
    """
    return (FIXTURE_DIR / name).read_bytes().decode("latin-1")


def strip_provenance(body: str) -> str:
    """Return a captured HTML body with its prepended provenance block removed."""
    marker = body.find(PROVENANCE_END)
    if marker < 0:
        return body
    return body[body.index("\n", body.index("-->", marker)) + 1 :]


def crosscheck_records() -> list[dict[str, object]]:
    """Return the captured SGML-vs-``data.sec.gov`` acceptance-time evidence."""
    document = json.loads(read_fixture("acceptance-datetime-crosscheck.json"))
    records: list[dict[str, object]] = document["records"]
    return records


def edgar_transport(
    *,
    available_index_dates: Iterable[str] = ("20240308", "20240311", "20240312"),
    request_log: list[str] | None = None,
) -> httpx.MockTransport:
    """Return a transport serving the captured EDGAR responses.

    Args:
        available_index_dates: ``YYYYMMDD`` stamps whose daily-index fixture
            should be served. Restricting this simulates a day whose index the
            connector must not request.
        request_log: list appended with every requested URL, in order, so a
            test can assert *which* requests were made — the only way to prove
            a checkpoint actually skipped work rather than silently re-reading.

    Returns:
        An :class:`httpx.MockTransport`. Anything not in the fixture set
        answers **403**, which is what EDGAR itself answers for a daily-index
        key that does not exist (verified live on a Saturday's ``master.idx``),
        so a test that wanders off the captured set fails the way production
        would rather than in some friendlier way.
    """
    dates = set(available_index_dates)

    def handler(request: httpx.Request) -> httpx.Response:
        if request_log is not None:
            request_log.append(str(request.url))
        path = request.url.path
        if path == QUARTER_LISTING_PATH:
            return _fixture_response("daily-index-2024-QTR1.sample.json")
        if path.startswith("/Archives/edgar/daily-index/2024/QTR1/master."):
            stamp = path.rsplit("master.", 1)[1].removesuffix(".idx")
            if stamp in dates:
                return _fixture_response(f"master.{stamp}.sample.idx")
            return _access_denied(request)
        if path.endswith("-index-headers.html"):
            accession = path.rsplit("/", 1)[1].removesuffix("-index-headers.html")
            if accession in CAPTURED_FILINGS:
                return _fixture_response(f"{accession}-index-headers.html")
            return httpx.Response(404, text=f"no fixture for {accession}", request=request)
        return _access_denied(request)

    return httpx.MockTransport(handler)


def _fixture_response(name: str) -> httpx.Response:
    """Return a 200 carrying a captured fixture's exact bytes."""
    return httpx.Response(200, content=(FIXTURE_DIR / name).read_bytes())


def _access_denied(request: httpx.Request) -> httpx.Response:
    """Return EDGAR's real answer for an archive key that does not exist."""
    return httpx.Response(
        403,
        content=(
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b"<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"
        ),
        request=request,
    )


def test_provenance_block_is_inert() -> None:
    """The provenance headers added at capture change no parse result.

    The fixtures are the captured bytes plus a provenance block, so every
    assertion made against them rests on that block being invisible to the
    parsers. This proves it for both file shapes rather than assuming it.
    """
    for accession in CAPTURED_FILINGS:
        body = read_fixture(f"{accession}-index-headers.html")
        with_block = parse_filing_header(body, accession_number=accession)
        without_block = parse_filing_header(strip_provenance(body), accession_number=accession)
        assert with_block == without_block

    for stamp in ("20240308", "20240311", "20240312"):
        body = read_fixture(f"master.{stamp}.sample.idx")
        stripped = "\n".join(line for line in body.splitlines() if not line.startswith("#"))
        assert parse_daily_index(body) == parse_daily_index(stripped)
