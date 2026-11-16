import json
from pathlib import Path

import pytest

from check_crap import FunctionScore, main, score_file

BRANCHY = (
    "def f(x):\n"
    "    if x:\n"
    "        return 1\n"
    "    if x > 1:\n"
    "        return 2\n"
    "    return 3\n"
)


def write_report(tmp_path: Path, executed: list[int], missing: list[int]) -> Path:
    source_path = tmp_path / "m.py"
    source_path.write_text(BRANCHY, encoding="utf-8")
    entry = {"executed_lines": executed, "missing_lines": missing}
    report = {"files": {str(source_path): entry}}
    report_path = tmp_path / "coverage.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_crap_is_complexity_when_fully_covered() -> None:
    score = FunctionScore("m.py", "f", 1, complexity=5, coverage=1.0)

    assert score.crap == pytest.approx(5.0)


def test_crap_grows_with_complexity_when_uncovered() -> None:
    score = FunctionScore("m.py", "f", 1, complexity=5, coverage=0.0)

    assert score.crap == pytest.approx(30.0)


def test_score_file_reports_per_function_line_coverage() -> None:
    source = "def f(x):\n    if x:\n        return 1\n    return 2\n"

    scores = score_file("m.py", source, executed={1, 2, 4}, missing={3})

    assert len(scores) == 1
    assert scores[0].name == "f"
    assert scores[0].complexity == 2
    assert scores[0].coverage == pytest.approx(3 / 4)


def test_score_file_ignores_classes_and_counts_methods() -> None:
    source = "class C:\n    def m(self):\n        return 1\n"

    scores = score_file("m.py", source, executed={1, 2, 3}, missing=set())

    assert [score.name for score in scores] == ["m"]


def test_function_without_measured_statements_counts_as_covered() -> None:
    scores = score_file("m.py", "def f():\n    pass\n", executed=set(), missing=set())

    assert scores[0].coverage == pytest.approx(1.0)


def test_main_fails_when_a_function_reaches_the_threshold(tmp_path: Path) -> None:
    report_path = write_report(tmp_path, executed=[], missing=[1, 2, 3, 4, 5, 6])

    assert main(report_path) == 1


def test_main_passes_when_every_function_is_below_threshold(tmp_path: Path) -> None:
    report_path = write_report(tmp_path, executed=[1, 2, 3, 4, 5, 6], missing=[])

    assert main(report_path) == 0
