from collections.abc import Sequence

import polars as pl

from georeset_corpus_assessment.pipeline import Generation, RunConfig

CONFIG = RunConfig(
    model_id="model",
    model_revision="rev",
    dataset_revision="data",
    seed=1,
    gpu="A100",
)


class FakeEngine:
    """Answer every prompt with one fixed text and count the calls."""

    def __init__(self, text: str = "FINAL: MA211") -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompts: Sequence[list[dict[str, str]]]) -> list[Generation]:
        self.calls += 1
        return [Generation(self.text, "stop") for _ in prompts]


def make_order(count: int) -> pl.DataFrame:
    """Return a small order table with the columns the pipeline expects."""
    return pl.DataFrame(
        {
            "id": [f"s{index}" for index in range(count)],
            "text": [f"sentence {index}" for index in range(count)],
            "region": ["r"] * count,
            "polygon_id": [f"way/{index}" for index in range(count)],
            "cell": ["c"] * count,
            "gold_codes": [[1101]] * count,
            "round": list(range(count)),
            "position": list(range(count)),
        },
        schema_overrides={"gold_codes": pl.List(pl.Int64)},
    )
