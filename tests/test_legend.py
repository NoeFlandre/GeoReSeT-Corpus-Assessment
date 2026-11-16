import pytest

from georeset_corpus_assessment.legend import EunisClass, Legend


def test_finest_classes_are_the_leaves(legend: Legend) -> None:
    assert legend.finest_identifiers() == ("MA211", "MA22", "URBAN")


def test_finest_classes_are_listed_by_code(legend: Legend) -> None:
    assert [c.identifier for c in legend.finest_classes()] == ["URBAN", "MA211", "MA22"]


def test_iteration_is_ordered_by_code(legend: Legend) -> None:
    assert [c.code for c in legend] == [0, 1000, 1100, 1101, 1200]


def test_identifier_for_code(legend: Legend) -> None:
    assert legend.identifier_for_code(1101) == "MA211"


def test_identifiers_for_codes(legend: Legend) -> None:
    assert legend.identifiers_for_codes([1101, 1000]) == frozenset({"MA211", "MA2"})


def test_identifier_at_returns_ancestor_at_each_level(legend: Legend) -> None:
    assert [legend.identifier_at("MA211", level) for level in (1, 2, 3)] == [
        "MA2",
        "MA21",
        "MA211",
    ]


def test_identifier_at_keeps_broader_classes_unchanged(legend: Legend) -> None:
    assert legend.identifier_at("MA22", 3) == "MA22"
    assert legend.identifier_at("MA2", 2) == "MA2"


def test_roll_up_merges_classes_at_a_level(legend: Legend) -> None:
    assert legend.roll_up(["MA211", "MA22"], 1) == frozenset({"MA2"})
    assert legend.roll_up(["MA211", "MA22"], 2) == frozenset({"MA21", "MA22"})


def test_duplicate_identifiers_are_rejected() -> None:
    classes = [EunisClass(1, ("A",), None), EunisClass(2, ("A",), None)]
    with pytest.raises(ValueError, match="unique"):
        Legend(classes)
