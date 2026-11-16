"""Multilabel F1 scores, computed at each EUNIS level."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from georeset_corpus_assessment.legend import MAX_LEVEL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from georeset_corpus_assessment.legend import Legend


@dataclass(frozen=True)
class F1Scores:
    """Micro and macro F1 over one set of predictions."""

    micro_f1: float
    macro_f1: float
    labels: int


def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    """Return the F1 score, or 0.0 when no label is present on either side."""
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return 2 * true_positive / denominator


def score(
    gold: Sequence[frozenset[str]],
    predicted: Sequence[frozenset[str]],
) -> F1Scores:
    """Compute micro and macro F1 over paired gold and predicted label sets."""
    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()
    for truth, guess in zip(gold, predicted, strict=True):
        true_positive.update(truth & guess)
        false_positive.update(guess - truth)
        false_negative.update(truth - guess)
    labels = set(true_positive) | set(false_positive) | set(false_negative)
    micro = f1(
        sum(true_positive.values()),
        sum(false_positive.values()),
        sum(false_negative.values()),
    )
    macro = (
        sum(
            f1(true_positive[label], false_positive[label], false_negative[label])
            for label in labels
        )
        / len(labels)
        if labels
        else 0.0
    )
    return F1Scores(micro_f1=micro, macro_f1=macro, labels=len(labels))


def score_levels(
    legend: Legend,
    gold: Sequence[frozenset[str]],
    predicted: Sequence[frozenset[str]],
) -> dict[int, F1Scores]:
    """Score the same predictions at every EUNIS level, rolling labels up."""
    return {
        level: score(
            [legend.roll_up(truth, level) for truth in gold],
            [legend.roll_up(guess, level) for guess in predicted],
        )
        for level in range(1, MAX_LEVEL + 1)
    }
