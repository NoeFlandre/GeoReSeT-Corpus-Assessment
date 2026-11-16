"""Fail when any function's CRAP score reaches the threshold.

CRAP = complexity**2 * (1 - coverage)**3 + complexity, per function.
Run `coverage json` first; this script reads the resulting coverage.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Function

CRAP_THRESHOLD = 6.0


@dataclass(frozen=True)
class FunctionScore:
    """Complexity and line coverage of one function, with its CRAP score."""

    path: str
    name: str
    line: int
    complexity: int
    coverage: float

    @property
    def crap(self) -> float:
        """Return the CRAP score of the function."""
        return self.complexity**2 * (1 - self.coverage) ** 3 + self.complexity


def _coverage(block: Function, statements: set[int], executed: set[int]) -> float:
    lines = {n for n in statements if block.lineno <= n <= block.endline}
    if not lines:
        return 1.0
    return len(lines & executed) / len(lines)


def score_file(
    path: str,
    source: str,
    executed: set[int],
    missing: set[int],
) -> list[FunctionScore]:
    """Score every top-level or nested function in one source file."""
    statements = executed | missing
    return [
        FunctionScore(
            path,
            block.name,
            block.lineno,
            block.complexity,
            _coverage(block, statements, executed),
        )
        for block in cc_visit(source)
        if isinstance(block, Function)
    ]


def collect_scores(report: dict) -> list[FunctionScore]:
    """Score every function in every file listed in a coverage.json report."""
    scores: list[FunctionScore] = []
    for path, data in report["files"].items():
        source = Path(path).read_text(encoding="utf-8")
        scores.extend(
            score_file(
                path,
                source,
                set(data["executed_lines"]),
                set(data["missing_lines"]),
            ),
        )
    return scores


def describe(score: FunctionScore) -> str:
    """Return a one-line description of a function score."""
    return (
        f"{score.path}:{score.line} {score.name} CRAP={score.crap:.2f} "
        f"(complexity {score.complexity}, coverage {score.coverage:.0%})"
    )


def main(report_path: Path) -> int:
    """Print every function at or above the threshold and return an exit code."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scores = collect_scores(report)
    failures = [score for score in scores if score.crap >= CRAP_THRESHOLD]
    for score in failures:
        print(describe(score))
    if failures:
        print(f"FAILED: {len(failures)} function(s) with CRAP >= {CRAP_THRESHOLD:g}")
        return 1
    print(f"OK: {len(scores)} function(s), all CRAP < {CRAP_THRESHOLD:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
