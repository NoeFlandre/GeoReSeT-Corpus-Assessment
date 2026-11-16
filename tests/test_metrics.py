import pytest

from georeset_corpus_assessment.legend import Legend
from georeset_corpus_assessment.metrics import f1, score, score_levels


def test_f1_is_zero_when_nothing_is_present() -> None:
    assert f1(0, 0, 0) == 0.0


def test_f1_matches_the_standard_formula() -> None:
    assert f1(2, 1, 1) == pytest.approx(4 / 6)


def test_score_is_perfect_when_predictions_match() -> None:
    gold = [frozenset({"a"}), frozenset({"a", "b"})]
    result = score(gold, gold)
    assert result.micro_f1 == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.labels == 2


def test_micro_and_macro_differ_with_unbalanced_labels() -> None:
    gold = [frozenset({"common"}), frozenset({"common"}), frozenset({"rare"})]
    predicted = [frozenset({"common"}), frozenset({"common"}), frozenset()]
    result = score(gold, predicted)
    assert result.micro_f1 == pytest.approx(4 / 5)
    assert result.macro_f1 == pytest.approx((1.0 + 0.0) / 2)


def test_score_of_empty_input_is_zero() -> None:
    result = score([], [])
    assert result.micro_f1 == 0.0
    assert result.macro_f1 == 0.0
    assert result.labels == 0


def test_score_levels_rolls_up_before_scoring(legend: Legend) -> None:
    gold = [frozenset({"MA211"})]
    predicted = [frozenset({"MA22"})]
    scores = score_levels(legend, gold, predicted)
    assert scores[1].micro_f1 == pytest.approx(1.0)
    assert scores[3].micro_f1 == pytest.approx(0.0)
