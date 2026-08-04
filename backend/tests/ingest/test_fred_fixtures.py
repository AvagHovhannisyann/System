"""FRED test payloads and the transport that serves them (P3.8 test support).

Provenance, stated first because it is the honest part
------------------------------------------------------

The EDGAR fixtures are captured bytes: every one came off ``www.sec.gov``.
**These are not, and cannot be.** Every FRED endpoint requires an API key, and
P3.8 was built with none provisioned — verified against the live service on
2026-08-01, where
``GET https://api.stlouisfed.org/fred/series/observations?series_id=GDPC1&file_type=json``
answers ``HTTP 400`` with ``{"error_code":400,"error_message":"Bad Request.
Variable api_key is not set. ..."}``.

So the fixtures under ``backend/tests/fixtures/fred/`` are of two kinds, and
each file says which it is in its own ``_provenance`` block:

- ``observations-gnpca-docs-example.json`` — **captured**: the complete JSON
  example response published on FRED's own ``fred/series/observations``
  documentation page. Genuinely FRED's output, so it pins this parser against
  the envelope FRED documents rather than one this repository imagined.
- everything else — **constructed**: written to that same documented schema to
  exercise revision, missing-value and paging logic, which the single-vintage
  documentation example cannot reach. Their values (1.1, 2.2, 3.3, 6.1 ...)
  were chosen to be obviously not economic measurements, their series ids say
  what they are (``REVISEDTWICE``, ``WITHMISSING``, ``PAGEDSERIES``), and their
  units strings read "Constructed Test Units (not an economic measure)".

This distinction is enforced, not merely documented:
:func:`test_every_fixture_declares_its_provenance` fails if any fixture omits
the block or claims an origin outside the two above, so a later file cannot
quietly present constructed numbers as captured ones (invariant I3).

What this buys and what it does not
-----------------------------------

Constructed payloads prove the connector's *logic* — that a revision becomes a
later-knowledge row, that ``"."`` never becomes ``0``, that paging follows
``count``/``offset``. They cannot prove FRED's real responses look like this.
That gap closes the day a key exists, by re-capturing these files from the live
API; until then it is a stated limitation rather than an implied guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

FIXTURE_DIR: Final = Path(__file__).resolve().parents[1] / "fixtures" / "fred"
"""Directory holding the FRED test payloads."""

PROVENANCE_KEY: Final = "_provenance"
"""Key carrying each fixture's provenance block. Ignored by every parser."""

VALID_ORIGINS: Final = frozenset({"captured", "constructed"})
"""The only two provenance origins a fixture may declare."""

OBSERVATION_FIXTURES: Final = {
    "REVISEDTWICE": ("observations-revised-twice.json",),
    "WITHMISSING": ("observations-missing-value.json",),
    "PAGEDSERIES": ("observations-paged-page1.json", "observations-paged-page2.json"),
    "GNPCA": ("observations-gnpca-docs-example.json",),
}
"""Series id -> its observation pages, in offset order."""

SERIES_FIXTURES: Final = {
    "REVISEDTWICE": "series-revised-twice.json",
    "WITHMISSING": "series-missing-value.json",
    "PAGEDSERIES": "series-paged.json",
}
"""Series id -> its ``fred/series`` metadata payload."""

TEST_API_KEY: Final = "abcdefghijklmnopqrstuvwxyz123456"
"""The key FRED's documentation uses in every example, explicitly *"for
demonstration purposes only"*. Used here so no real credential appears in the
test suite, and so a leak assertion has a concrete string to search for."""


class NonBlockingClock:
    """A virtual clock that advances a full second per sleep.

    Used by the tests that assert *what* the connector fetches and builds, not
    how it paces: the rate limiter never blocks, so those tests cost no real
    time. Advancing generously rather than by exactly the requested delay works
    around the same shared token-bucket rounding defect
    ``backend/tests/ingest/test_edgar_fixtures.py`` documents, rather than
    hiding one in this connector.

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


def read_fixture(name: str) -> dict[str, Any]:
    """Return one fixture payload, parsed.

    Args:
        name: file name under ``backend/tests/fixtures/fred/``.

    Returns:
        The decoded JSON object, provenance block included — parsers ignore
        unknown keys, and ``test_provenance_block_is_inert`` proves it.
    """
    payload: dict[str, Any] = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return payload


def fred_transport(
    *,
    request_log: list[httpx.URL] | None = None,
    unavailable: bool = False,
    status: int | None = None,
) -> httpx.MockTransport:
    """Return a transport serving the FRED test payloads.

    Args:
        request_log: list appended with every requested URL, in order, so a
            test can assert *which* requests were made — the only way to prove
            a checkpoint actually skipped work rather than silently re-reading,
            and the way the API key is proven absent from what gets logged.
        unavailable: when true, every request raises
            :class:`httpx.ConnectError`, simulating an unreachable source.
        status: when set, every request answers this HTTP status with FRED's
            documented JSON error envelope.

    Returns:
        An :class:`httpx.MockTransport`. A request for a series with no fixture
        answers **404**, which is one of the statuses FRED documents, so a test
        that wanders off the fixture set fails the way production would.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request_log is not None:
            request_log.append(request.url)
        if unavailable:
            msg = "connection refused (simulated unreachable source)"
            raise httpx.ConnectError(msg, request=request)
        if status is not None:
            return _error_response(status, request)
        series_id = request.url.params.get("series_id", "")
        if request.url.path.endswith("/series/observations"):
            pages = OBSERVATION_FIXTURES.get(series_id)
            if pages is None:
                return _error_response(404, request)
            offset = int(request.url.params.get("offset", "0"))
            page = _page_for_offset(pages, offset)
            if page is None:
                return _error_response(404, request)
            return httpx.Response(200, json=read_fixture(page), request=request)
        if request.url.path.endswith("/series"):
            name = SERIES_FIXTURES.get(series_id)
            if name is None:
                return _error_response(404, request)
            return httpx.Response(200, json=read_fixture(name), request=request)
        return _error_response(404, request)

    return httpx.MockTransport(handler)


def _page_for_offset(pages: Iterable[str], offset: int) -> str | None:
    """Return the fixture page whose declared ``offset`` matches, or ``None``.

    Matching on the payload's own declared offset rather than on position keeps
    the transport honest: it serves what the connector asked for, so a
    connector that computes the wrong offset gets a 404 instead of the next
    page in sequence.
    """
    for name in pages:
        if read_fixture(name).get("offset") == offset:
            return name
    return None


def _error_response(status: int, request: httpx.Request) -> httpx.Response:
    """Return FRED's documented JSON error envelope with ``status``.

    Shape from ``https://fred.stlouisfed.org/docs/api/fred/errors.html``:
    ``{"error_code": 400, "error_message": "..."}``.
    """
    return httpx.Response(
        status,
        json={"error_code": status, "error_message": f"Simulated FRED error {status}"},
        request=request,
    )


# --- provenance assertions ---------------------------------------------------


def _fixture_files() -> list[Path]:
    """Return every fixture file in the FRED fixture directory."""
    return sorted(FIXTURE_DIR.glob("*.json"))


def test_fixture_directory_is_not_empty() -> None:
    """Guard the guards: a glob that matches nothing must not pass vacuously."""
    assert _fixture_files(), f"no fixtures found under {FIXTURE_DIR}"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda path: path.name)
def test_every_fixture_declares_its_provenance(path: Path) -> None:
    """Every payload states whether it was captured or constructed, and why.

    This is the enforcement behind the module docstring. Without it, a later
    contributor could drop an invented payload beside the captured one and
    nothing would notice — which is precisely how synthetic data starts being
    treated as real (invariant I3).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get(PROVENANCE_KEY)
    assert isinstance(provenance, dict), f"{path.name} has no {PROVENANCE_KEY} block"
    assert provenance.get("origin") in VALID_ORIGINS, (
        f"{path.name} declares origin {provenance.get('origin')!r}, "
        f"which is not one of {sorted(VALID_ORIGINS)}"
    )
    if provenance["origin"] == "captured":
        # For captured output the source URL *is* the schema reference: it is
        # where the bytes came from, which is a stronger claim than citing a
        # spec they were written to match.
        assert provenance.get("source_url"), f"{path.name} claims capture with no source URL"
    else:
        assert provenance.get("schema_reference"), (
            f"{path.name} is constructed and must cite the documentation its shape follows"
        )
        assert "NOT CAPTURED" in provenance.get("warning", ""), (
            f"{path.name} is constructed but its warning does not say so unmistakably"
        )


def test_constructed_fixtures_do_not_impersonate_real_series() -> None:
    """No constructed payload may carry a real FRED series id.

    A constructed history filed under ``GDPC1`` would be indistinguishable
    from real GDP vintages to anyone reading the directory, and the labelling
    in the file would not travel with a copied number. The captured fixture is
    the only one permitted to name a real series.
    """
    for path in _fixture_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload[PROVENANCE_KEY]["origin"] == "captured":
            continue
        ids = {entry["id"] for entry in payload.get("seriess", [])}
        ids |= set(OBSERVATION_FIXTURES) & {
            series for series, pages in OBSERVATION_FIXTURES.items() if path.name in pages
        }
        for series_id in ids:
            assert series_id in {"REVISEDTWICE", "WITHMISSING", "PAGEDSERIES"}, (
                f"{path.name} uses series id {series_id!r}; constructed fixtures must use "
                "an obviously-constructed identifier"
            )


def test_provenance_block_is_inert() -> None:
    """The provenance block changes no parse result.

    Every assertion made against these fixtures rests on that block being
    invisible to the parsers. Proven for both payload shapes rather than
    assumed, exactly as the EDGAR fixtures prove it for their preamble.
    """
    from backend.ingest.fred.parse import parse_observations, parse_series_versions

    for series_id, pages in OBSERVATION_FIXTURES.items():
        for name in pages:
            payload = read_fixture(name)
            stripped = {key: value for key, value in payload.items() if key != PROVENANCE_KEY}
            assert parse_observations(payload, series_id=series_id) == parse_observations(
                stripped, series_id=series_id
            )

    for series_id, name in SERIES_FIXTURES.items():
        payload = read_fixture(name)
        stripped = {key: value for key, value in payload.items() if key != PROVENANCE_KEY}
        assert parse_series_versions(payload, series_id=series_id) == parse_series_versions(
            stripped, series_id=series_id
        )
