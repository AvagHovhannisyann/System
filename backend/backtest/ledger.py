"""Read-only reader for ``TESTING_LEDGER.md`` — the trial count for P10.3.

The Deflated Sharpe Ratio needs ``N``, the number of configurations that were
evaluated before the reported one was selected. Directive §9.7 makes the ledger
append-only precisely so that ``N`` cannot shrink: "Deleting failed experiments
is how backtest overfitting becomes invisible."

This module **only reads**. It has no write path, and it must never grow one:
the ledger is a state file, appended to by the training entrypoint (P8.4), and a
consumer that could edit it would defeat the guarantee it exists to provide.

Deliberately conservative parsing
---------------------------------

The reader extracts the trial count and the raw cells of every row. It does
**not** try to mine Sharpe ratios out of the free-text ``metric(s)``/``result``
columns. Those columns have no committed grammar, and guessing at one would be
inventing an interface (directive §9.4) whose failure mode is a silently
understated ``V`` — the variance term that makes the whole deflation work.
Callers that need trial Sharpe ratios must get them from a structured source.

The parser fails closed. A row whose cell count disagrees with the header, or a
file with no recognisable table, raises rather than returning a smaller number:
an under-counted ledger is exactly the failure that makes a DSR a lie, and
silence is the one response that must not be possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "LedgerFormatError",
    "LedgerRow",
    "TrialLedger",
    "read_testing_ledger",
]


class LedgerFormatError(ValueError):
    """Raised when ``TESTING_LEDGER.md`` cannot be parsed unambiguously.

    Fatal on purpose. The alternative to raising is returning a trial count that
    might be too small, and a too-small trial count inflates every Deflated
    Sharpe Ratio computed from it.
    """


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One evaluated configuration recorded in the ledger.

    Attributes:
        line_number: 1-based line number in the source file, so a malformed row
            can be located by a human without searching.
        cells: the row's cells, stripped, in header order.
    """

    line_number: int
    cells: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrialLedger:
    """The parsed contents of ``TESTING_LEDGER.md``.

    Attributes:
        header: the table's column names, in order.
        rows: every data row, in file order. One row is one evaluated
            configuration — one *trial*.
    """

    header: tuple[str, ...]
    rows: tuple[LedgerRow, ...]

    @property
    def trial_count(self) -> int:
        """Return ``N``, the number of trials, for the Deflated Sharpe Ratio.

        This is a plain row count. It is an **upper bound** on the number of
        *independent* trials — a ledger full of near-identical perturbations
        represents fewer effective trials than its row count suggests, and using
        the raw count therefore over-deflates. Over-deflation is the safe
        direction, so it is what this returns; see
        :mod:`backend.backtest.dsr` for when that matters.
        """
        return len(self.rows)


def _split_markdown_row(line: str) -> tuple[str, ...]:
    """Split one pipe-delimited markdown table row into stripped cells."""
    stripped = line.strip()
    body = stripped.removeprefix("|").removesuffix("|")
    return tuple(cell.strip() for cell in body.split("|"))


def _is_separator(cells: tuple[str, ...]) -> bool:
    """Return whether the cells form a markdown header separator (``---``)."""
    return all(cell != "" and set(cell) <= set("-: ") for cell in cells)


def read_testing_ledger(path: Path) -> TrialLedger:
    """Parse ``TESTING_LEDGER.md`` and return its rows.

    The file's shape is fixed by the ledger's own preamble: a single markdown
    table whose header row is followed by a ``|---|`` separator and then one row
    per evaluated configuration. Everything outside the table (the preamble, the
    rules, a trailing italic note when the table is empty) is ignored.

    Args:
        path: filesystem path to ``TESTING_LEDGER.md``.

    Returns:
        A :class:`TrialLedger`. An empty table yields zero rows, which is an
        honest answer meaning "nothing has been evaluated yet" — not a licence
        to treat the trial count as 1.

    Raises:
        FileNotFoundError: if ``path`` does not exist. A missing ledger is never
            an empty ledger.
        LedgerFormatError: if no table header followed by a separator is found,
            or if any data row's cell count disagrees with the header's.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    header: tuple[str, ...] | None = None
    rows: list[LedgerRow] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("|"):
            cells = _split_markdown_row(line)
            separator_follows = index + 1 < len(lines) and lines[index + 1].strip().startswith("|")
            if (
                header is None
                and separator_follows
                and _is_separator(_split_markdown_row(lines[index + 1]))
            ):
                header = cells
                index += 2
                continue
            if header is not None:
                if _is_separator(cells):
                    index += 1
                    continue
                if len(cells) != len(header):
                    msg = (
                        f"{path}: row on line {index + 1} has {len(cells)} cells but the "
                        f"header declares {len(header)}; the trial count cannot be trusted "
                        "until the row is corrected"
                    )
                    raise LedgerFormatError(msg)
                rows.append(LedgerRow(line_number=index + 1, cells=cells))
        index += 1

    if header is None:
        msg = f"{path}: no markdown table header followed by a separator row was found"
        raise LedgerFormatError(msg)
    return TrialLedger(header=header, rows=tuple(rows))
