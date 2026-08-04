"""Binding a task to the pair of documents it compares (P7.4).

:mod:`backend.extraction.tasks.library` says *what to ask* — the prompt, the
output schema, the sign conventions. It says nothing about *what to ask it of*,
and for a delta task that second half carries all of the temporal risk: which
document is the baseline, when the question about it is answered, and what
happens when there is no answer.

A :class:`DeltaTaskSpec` is that binding and nothing more. Like an
:class:`~backend.extraction.tasks.base.ExtractionTask` it is **data, not a
subclass**: there is no method for a task to override, so no task can arrange to
be compared against a document chosen some other way.

Every spec declares ``WHOLE_DOCUMENT`` chunking, and the registry refuses one
that does not. Half of a two-document comparison is not a weaker comparison, it
is a different and unstated question, and its answer would sit in the store
looking exactly like a real one (I3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.extraction.tasks.base import ChunkingPolicy

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from backend.extraction.tasks.base import ExtractionTask
    from backend.extraction.tasks.delta.anchor import DocumentClass

__all__ = ["DeltaTaskNotRegisteredError", "DeltaTaskRegistry", "DeltaTaskSpec"]


@dataclass(frozen=True, slots=True)
class DeltaTaskSpec:
    """One delta task: what to ask, and which two documents to ask it of.

    Attributes:
        task: the extraction task — prompt, output schema, chunking policy
            (:mod:`backend.extraction.tasks.library`).
        document_class: the family of documents this task compares, which fixes
            both the eligible forms and the store the baseline comes from
            (:mod:`backend.extraction.tasks.delta.anchor`).
        construct: one line naming the *change* being measured, for the
            operator's task list (§6.5). Deliberately phrased as a delta: if it
            can be written as a level, the task is the wrong shape for §5-P7.
    """

    task: ExtractionTask
    document_class: DocumentClass
    construct: str

    def __post_init__(self) -> None:
        """Reject a spec that could produce an answer from half a comparison.

        Raises:
            ValueError: if the task does not declare ``WHOLE_DOCUMENT`` chunking,
                or if ``construct`` is blank.
        """
        if self.task.chunking_policy is not ChunkingPolicy.WHOLE_DOCUMENT:
            msg = (
                f"delta task {self.task.name!r} declares "
                f"{self.task.chunking_policy.value} chunking; a paired document must reach "
                "the model in one call or the answer is a comparison of one half against "
                "nothing (I3)"
            )
            raise ValueError(msg)
        if not self.construct.strip():
            msg = f"delta task {self.task.name!r} must name the change it measures"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """The task's name — this spec's identity as well."""
        return self.task.name


class DeltaTaskNotRegisteredError(LookupError):
    """A delta task was requested by a name the registry does not hold.

    Raised rather than returning ``None`` so a run cannot proceed against a
    silently absent task and store results under a name nothing defines.
    """


class DeltaTaskRegistry:
    """A named collection of delta task specs.

    A plain object built from an explicit list, for the reason
    :class:`~backend.extraction.tasks.base.TaskRegistry` gives: import-time
    registration makes the set of tasks depend on which modules happened to be
    imported, and that is implicit configuration a run cannot record (I2).
    """

    def __init__(self, specs: Iterable[DeltaTaskSpec] = ()) -> None:
        """Build a registry.

        Args:
            specs: the specs to hold.

        Raises:
            ValueError: if two specs share a task name. Two specs under one name
                would share a prompt history and a golden-set score while
                comparing different documents.
        """
        self._specs: dict[str, DeltaTaskSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                msg = f"duplicate delta task name {spec.name!r}"
                raise ValueError(msg)
            self._specs[spec.name] = spec

    def __len__(self) -> int:
        """Number of registered specs (count)."""
        return len(self._specs)

    def __iter__(self) -> Iterator[DeltaTaskSpec]:
        """Iterate the specs in registration order."""
        return iter(self._specs.values())

    def __contains__(self, name: object) -> bool:
        """Whether a task name is registered."""
        return name in self._specs

    @property
    def names(self) -> tuple[str, ...]:
        """Registered task names, in registration order."""
        return tuple(self._specs)

    def get(self, name: str) -> DeltaTaskSpec:
        """Return one spec by name.

        Args:
            name: the task name.

        Returns:
            The spec.

        Raises:
            DeltaTaskNotRegisteredError: if no spec has that name.
        """
        try:
            return self._specs[name]
        except KeyError as exc:
            msg = f"no delta task named {name!r}; registered: {sorted(self._specs)}"
            raise DeltaTaskNotRegisteredError(msg) from exc
