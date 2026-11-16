"""EUNIS legend: the classes of the habitat map and their hierarchy.

Each class sits at one of three levels. Its ``path`` lists the identifiers of
its ancestors and itself, for example ``("MA2", "MA21", "MA211")``. A class
whose path is a prefix of another class's path is a parent of it. Classes
without children are the finest-level classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import polars as pl

MAX_LEVEL = 3


@dataclass(frozen=True)
class EunisClass:
    """One class of the EUNIS legend and its position in the hierarchy."""

    code: int
    path: tuple[str, ...]
    name: str | None

    @property
    def identifier(self) -> str:
        """Return the class identifier, for example ``MA211`` or ``URBAN``."""
        return self.path[-1]

    @property
    def depth(self) -> int:
        """Return the level of the class, from 1 (broadest) to 3."""
        return len(self.path)


class Legend:
    """EUNIS classes indexed by map code and by identifier."""

    def __init__(self, classes: Iterable[EunisClass]) -> None:
        """Index the classes and find the finest-level ones."""
        self._by_code = {c.code: c for c in classes}
        ordered = list(self._by_code.values())
        self._by_identifier = _index_by_identifier(ordered)
        self._finest = _finest_identifiers(ordered)

    def __iter__(self) -> Iterator[EunisClass]:
        """Yield every class, ordered by map code."""
        return iter(sorted(self._by_code.values(), key=lambda c: c.code))

    def finest_identifiers(self) -> tuple[str, ...]:
        """Return the identifiers of the finest-level classes, sorted."""
        return self._finest

    def finest_classes(self) -> list[EunisClass]:
        """Return the finest-level classes, ordered by map code."""
        finest = set(self._finest)
        return [c for c in self if c.identifier in finest]

    def identifier_for_code(self, code: int) -> str:
        """Return the identifier of a map code."""
        return self._by_code[code].identifier

    def identifiers_for_codes(self, codes: Iterable[int]) -> frozenset[str]:
        """Return the identifiers of several map codes."""
        return frozenset(self.identifier_for_code(code) for code in codes)

    def identifier_at(self, identifier: str, level: int) -> str:
        """Return the ancestor of a class at a level, or the class if it is broader.

        A class at depth 2 seen at level 1 gives its level-1 ancestor. A class
        at depth 1 seen at level 3 is returned unchanged, because it has no
        finer level.
        """
        found = self._by_identifier[identifier]
        if found.depth >= level:
            return found.path[level - 1]
        return found.identifier

    def roll_up(self, identifiers: Iterable[str], level: int) -> frozenset[str]:
        """Return the set of classes at a level covering several classes."""
        return frozenset(self.identifier_at(i, level) for i in identifiers)


def _index_by_identifier(classes: list[EunisClass]) -> dict[str, EunisClass]:
    index = {c.identifier: c for c in classes}
    if len(index) != len(classes):
        msg = "EUNIS identifiers must be unique"
        raise ValueError(msg)
    return index


def _finest_identifiers(classes: list[EunisClass]) -> tuple[str, ...]:
    parents = {c.path[:k] for c in classes for k in range(1, c.depth)}
    leaves = [c.identifier for c in classes if c.path not in parents]
    return tuple(sorted(leaves))


def load_legend(table: pl.DataFrame) -> Legend:
    """Build a legend from the detailed EUNIS legend table.

    Expects the columns ``Code``, ``EUNIS1``, ``EUNIS2``, ``EUNIS3`` and the
    matching ``EUNIS<level>_name`` columns. Empty cells are missing values.
    """
    classes = [_class_from_row(row) for row in table.iter_rows(named=True)]
    return Legend(classes)


def _class_from_row(row: dict[str, object]) -> EunisClass:
    levels = [_text(row[f"EUNIS{level}"]) for level in range(1, MAX_LEVEL + 1)]
    path = tuple(value for value in levels if value is not None)
    depth = len(path)
    name = _text(row[f"EUNIS{depth}_name"])
    return EunisClass(code=int(str(row["Code"])), path=path, name=name)


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
