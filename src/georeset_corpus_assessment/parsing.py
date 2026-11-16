"""Turn a model answer into a list of valid finest-level identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

FINAL_PATTERN = re.compile(
    r"^\s*\W*FINAL\W*\s*:\s*(?P<body>.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_STRIP_CHARS = " \t*`'\"[]()."
_NONE_WORDS = frozenset({"", "none"})


@dataclass(frozen=True)
class Prediction:
    """Parsed answer: valid identifiers, rejected tokens and any parse error."""

    identifiers: tuple[str, ...]
    invalid: tuple[str, ...]
    error: str | None


def parse_prediction(text: str, valid: frozenset[str]) -> Prediction:
    """Read the last ``FINAL:`` line and keep identifiers present in ``valid``.

    Duplicates are removed and the answer order is kept. Tokens outside
    ``valid`` are returned as ``invalid``. A missing ``FINAL:`` line is an
    error. A truncated answer is checked by the caller.
    """
    matches = FINAL_PATTERN.findall(text)
    if not matches:
        return Prediction(identifiers=(), invalid=(), error="no_final_line")
    body = matches[-1].strip(_STRIP_CHARS)
    if body.lower() in _NONE_WORDS:
        return Prediction(identifiers=(), invalid=(), error=None)
    tokens = _split_tokens(body)
    return Prediction(
        identifiers=_unique(_known(tokens, valid)),
        invalid=_unique(_unknown(tokens, valid)),
        error=None,
    )


def _split_tokens(body: str) -> list[str]:
    return [token.strip(_STRIP_CHARS) for token in body.split(",")]


def _known(tokens: Iterable[str], valid: frozenset[str]) -> list[str]:
    return [token for token in tokens if token in valid]


def _unknown(tokens: Iterable[str], valid: frozenset[str]) -> list[str]:
    return [token for token in tokens if token and token not in valid]


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))
