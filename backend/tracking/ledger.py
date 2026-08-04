"""Append-only writer for ``TESTING_LEDGER.md`` (CC.3, directive §9.7).

The ledger is the trial count *N* for the Deflated Sharpe Ratio.
:mod:`backend.backtest.ledger` reads it; this module is the only sanctioned way
to add to it, and it can do nothing else. There is no update path, no delete
path, and no rewrite path — §9.7: *"Never delete from TESTING_LEDGER.md.
Deleting failed experiments is how backtest overfitting becomes invisible."*

How append-only is enforced, not just intended
----------------------------------------------

The file is opened in append mode (``"a"``) and one line is written. Every byte
already in the file stays where it is; nothing is read into memory and written
back, so there is no code path here that *could* alter an existing row even
through a bug. If the file does not end in a newline, a newline is written
first — still purely additive, and it prevents the new row from being glued onto
a truncated last line.

The row's contents come from a :class:`~backend.tracking.stamp.ReproducibilityStamp`,
so a row cannot be written without all four I2 components. The dirty-tree marker
travels in the ``git commit`` cell (``<sha>-dirty``), where a human reading the
table sees it next to the result rather than in a flag elsewhere.

Validation is done before anything is written: cells that would corrupt the
markdown table are refused, because a corrupted row makes the reader refuse the
whole file, and a ledger that cannot be read cannot supply *N*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.backtest.ledger import read_testing_ledger
from backend.tracking.stamp import ReproducibilityStamp

__all__ = [
    "LEDGER_COLUMNS",
    "LedgerAppendError",
    "TrialRecord",
    "append_trial",
]

LEDGER_COLUMNS: tuple[str, ...] = (
    "#",
    "timestamp (UTC)",
    "phase/task",
    "experiment id",
    "config_hash",
    "data_version",
    "git commit",
    "seed",
    "CV scheme",
    "metric(s)",
    "result",
    "notes",
)
"""The committed ledger schema, in order. Checked against the file's own header."""

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_REQUIRED_CELLS = frozenset(
    {"phase/task", "experiment id", "CV scheme", "metric(s)", "result"},
)
_FORBIDDEN_IN_CELL = ("|", "\n", "\r")


class LedgerAppendError(ValueError):
    """Raised when a trial row cannot be appended safely.

    Every case is fatal rather than best-effort: an unappended row is a visible
    failure the caller must handle, whereas a mangled row silently breaks the
    reader for every future Deflated Sharpe calculation.
    """


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One evaluated configuration, ready to be appended to the ledger.

    Every training run, hyperparameter set, feature-set variant, label-horizon
    choice, or strategy configuration that is *evaluated* gets one of these —
    including the failures. That is what makes *N* honest.

    Attributes:
        phase_task: task id from ``PLAN.md``, e.g. ``"P8.4"``.
        experiment_id: identifier of the recorded run. The MLflow run id from
            :class:`~backend.tracking.mlflow_run.TrackedRun` is the intended
            value: it links the row to the run holding the params and artifacts.
        stamp: the four I2 components for the run.
        cv_scheme: cross-validation scheme, e.g. ``"purged 5-fold, 21d embargo"``.
        metrics: the metric names and values, with units. Free text — the reader
            deliberately does not parse this column.
        result: the outcome, including failures ("IC 0.004, t=0.3, rejected").
        notes: anything a later reader needs, e.g. why the variant was tried.
    """

    phase_task: str
    experiment_id: str
    stamp: ReproducibilityStamp
    cv_scheme: str
    metrics: str
    result: str
    notes: str = ""


def _validate_cell(column: str, value: str) -> str:
    """Validate one cell and return it stripped.

    Args:
        column: column name, for the error message.
        value: the cell's text.

    Returns:
        The stripped text.

    Raises:
        LedgerAppendError: if the cell contains a pipe or newline (which would
            change the row's cell count and make the reader refuse the file), or
            if a required column is blank.
    """
    supplied: object = value
    if not isinstance(supplied, str):
        msg = f"{column} must be a string, got {type(supplied).__name__}"
        raise LedgerAppendError(msg)
    for character in _FORBIDDEN_IN_CELL:
        if character in value:
            msg = (
                f"{column}={value!r} contains {character!r}; it would break the "
                "markdown row and make the whole ledger unreadable"
            )
            raise LedgerAppendError(msg)
    stripped = value.strip()
    if stripped == "" and column in _REQUIRED_CELLS:
        msg = f"{column} is empty; a trial row without it cannot be interpreted later"
        raise LedgerAppendError(msg)
    return stripped


def _format_timestamp(moment: datetime) -> str:
    """Format a timezone-aware instant as ``YYYY-MM-DDTHH:MM:SSZ`` in UTC.

    Args:
        moment: the instant. Must be timezone-aware.

    Returns:
        The formatted UTC timestamp.

    Raises:
        LedgerAppendError: if ``moment`` is naive. A naive timestamp in a shared
            ledger cannot be ordered against rows written in another timezone.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        msg = f"timestamp {moment!r} is timezone-naive; the ledger records UTC instants"
        raise LedgerAppendError(msg)
    return moment.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)


def append_trial(
    ledger_path: Path,
    record: TrialRecord,
    *,
    timestamp: datetime | None = None,
) -> int:
    """Append one trial row to ``ledger_path`` and return its trial number.

    Purely additive: the existing bytes of the file are never read back and
    rewritten, so prior rows are byte-identical afterwards.

    The trial number written in the ``#`` column is ``existing rows + 1``,
    counted with :func:`backend.backtest.ledger.read_testing_ledger` — the same
    reader the Deflated Sharpe Ratio uses, so the number in the column and the
    number feeding the DSR cannot drift apart.

    Concurrency: the write is a single ``O_APPEND`` write of one short line, so
    rows from simultaneous writers cannot interleave mid-line. The ``#`` value,
    however, is computed from a read that happened just before, so two truly
    simultaneous appends can repeat a number. That is cosmetic — *N* is the row
    count, not the largest ``#`` — but take a lock around this call if a strict
    sequence is ever needed.

    Args:
        ledger_path: path to ``TESTING_LEDGER.md``.
        record: the trial to record.
        timestamp: the instant to record. Defaults to now (UTC). Must be
            timezone-aware.

    Returns:
        The trial number of the appended row (1-based).

    Raises:
        FileNotFoundError: if the ledger does not exist. It is a committed state
            file; creating one here would start a fresh trial count at zero and
            silently discard every experiment already recorded.
        LedgerFormatError: from the reader, if the existing file cannot be
            parsed. An unparseable ledger is not appended to — the new row would
            be as unreadable as the rest.
        LedgerAppendError: if the file's header is not the committed schema, or
            if any cell is invalid.
    """
    existing = read_testing_ledger(ledger_path)
    if tuple(existing.header) != LEDGER_COLUMNS:
        msg = (
            f"{ledger_path} header is {existing.header}, expected {LEDGER_COLUMNS}; "
            "the schema changed and this writer would produce rows that do not match it"
        )
        raise LedgerAppendError(msg)

    moment = datetime.now(UTC) if timestamp is None else timestamp
    number = existing.trial_count + 1
    values = {
        "#": str(number),
        "timestamp (UTC)": _format_timestamp(moment),
        "phase/task": record.phase_task,
        "experiment id": record.experiment_id,
        "config_hash": record.stamp.config_hash,
        "data_version": record.stamp.data_version,
        "git commit": record.stamp.git_reference,
        "seed": str(record.stamp.seed),
        "CV scheme": record.cv_scheme,
        "metric(s)": record.metrics,
        "result": record.result,
        "notes": record.notes,
    }
    cells = tuple(_validate_cell(column, values[column]) for column in LEDGER_COLUMNS)
    line = "| " + " | ".join(cells) + " |\n"
    existing_text = ledger_path.read_text(encoding="utf-8")
    prefix = "" if existing_text.endswith("\n") or existing_text == "" else "\n"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + line)

    # Read back with the reader the DSR uses. This catches the one failure the
    # pre-write validation cannot: a row that lands but is not *counted* — e.g.
    # if it parsed as a markdown separator and were skipped.
    updated = read_testing_ledger(ledger_path)
    if updated.trial_count != number or updated.rows[-1].cells != cells:
        msg = (
            f"{ledger_path}: the appended row did not read back as expected "
            f"(trial count {updated.trial_count}, expected {number}). The row was "
            "written — inspect the file; nothing here removes it."
        )
        raise LedgerAppendError(msg)
    return number
