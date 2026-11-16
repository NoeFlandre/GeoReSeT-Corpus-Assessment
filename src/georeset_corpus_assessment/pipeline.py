"""Run the model over the round-robin order, in resumable Parquet shards.

Each batch of sentences is written as one shard. A rerun skips every id that
already appears in a shard, so an interrupted job resumes where it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import polars as pl

from georeset_corpus_assessment.metrics import score_levels
from georeset_corpus_assessment.parsing import Prediction, parse_prediction
from georeset_corpus_assessment.prompt import PROMPT_VERSION, build_messages

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from georeset_corpus_assessment.legend import Legend

SHARD_PATTERN = "part-*.parquet"
TRUNCATED = "length"


@dataclass(frozen=True)
class Generation:
    """Raw text returned by the engine for one prompt."""

    text: str
    finish_reason: str


class Engine(Protocol):
    """Anything that answers a batch of chat prompts."""

    def generate(self, prompts: Sequence[list[dict[str, str]]]) -> list[Generation]:
        """Return one generation per prompt, in the same order."""
        ...


@dataclass(frozen=True)
class RunConfig:
    """Everything that must be recorded with each result for reproducibility."""

    model_id: str
    model_revision: str
    dataset_revision: str
    seed: int
    gpu: str


def done_ids(output_dir: Path) -> set[str]:
    """Return the sentence ids already written to shards in ``output_dir``."""
    shards = sorted(output_dir.glob(SHARD_PATTERN))
    if not shards:
        return set()
    return set(pl.read_parquet(shards, columns=["id"])["id"].to_list())


def next_batch(
    order: pl.DataFrame,
    done: set[str],
    batch_size: int,
    limit: int,
) -> pl.DataFrame:
    """Return the next pending rows among the first ``limit`` of the order."""
    pending = order.head(limit).filter(~pl.col("id").is_in(done))
    return pending.head(batch_size)


def interpret(generation: Generation, valid: frozenset[str]) -> Prediction:
    """Turn a generation into a prediction, treating truncation as an error."""
    if generation.finish_reason == TRUNCATED:
        return Prediction(identifiers=(), invalid=(), error="truncated")
    return parse_prediction(generation.text, valid)


def build_rows(
    batch: pl.DataFrame,
    generations: Sequence[Generation],
    legend: Legend,
    config: RunConfig,
    created_at: str,
) -> pl.DataFrame:
    """Return one result row per sentence of the batch."""
    valid = frozenset(legend.finest_identifiers())
    rows = []
    for sentence, generation in zip(
        batch.iter_rows(named=True), generations, strict=True
    ):
        prediction = interpret(generation, valid)
        gold_codes = list(sentence["gold_codes"])
        rows.append(
            {
                "id": sentence["id"],
                "text": sentence["text"],
                "region": sentence["region"],
                "polygon_id": sentence["polygon_id"],
                "cell": sentence["cell"],
                "round": int(sentence["round"]),
                "position": int(sentence["position"]),
                "gold_codes": gold_codes,
                "gold_identifiers": sorted(legend.identifiers_for_codes(gold_codes)),
                "raw_output": generation.text,
                "finish_reason": generation.finish_reason,
                "predicted": list(prediction.identifiers),
                "invalid": list(prediction.invalid),
                "parse_error": prediction.error or "",
                "prompt_version": PROMPT_VERSION,
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "dataset_revision": config.dataset_revision,
                "seed": config.seed,
                "gpu": config.gpu,
                "created_at": created_at,
            },
        )
    return pl.DataFrame(rows, schema=RESULT_SCHEMA)


def write_shard(rows: pl.DataFrame, output_dir: Path) -> Path:
    """Write one shard atomically, numbered after the existing ones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(output_dir.glob(SHARD_PATTERN)))
    final = output_dir / f"part-{index:05d}.parquet"
    temporary = final.with_suffix(".parquet.tmp")
    rows.write_parquet(temporary)
    temporary.replace(final)
    return final


def run(  # noqa: PLR0913
    order: pl.DataFrame,
    legend: Legend,
    engine: Engine,
    output_dir: Path,
    config: RunConfig,
    *,
    limit: int,
    batch_size: int,
    created_at: str,
) -> int:
    """Generate answers for the pending sentences and return how many were written."""
    written = 0
    while True:
        batch = next_batch(order, done_ids(output_dir), batch_size, limit)
        if batch.height == 0:
            return written
        prompts = [build_messages(text, legend) for text in batch["text"].to_list()]
        generations = engine.generate(prompts)
        rows = build_rows(batch, generations, legend, config, created_at)
        write_shard(rows, output_dir)
        written += rows.height


def score_results(output_dir: Path, legend: Legend) -> pl.DataFrame:
    """Score every shard at each EUNIS level and return one row per level."""
    results = pl.read_parquet(sorted(output_dir.glob(SHARD_PATTERN)))
    gold = [
        frozenset(identifiers) for identifiers in results["gold_identifiers"].to_list()
    ]
    predicted = [
        frozenset(identifiers) for identifiers in results["predicted"].to_list()
    ]
    scores = score_levels(legend, gold, predicted)
    return pl.DataFrame(
        [
            {
                "level": level,
                "micro_f1": scored.micro_f1,
                "macro_f1": scored.macro_f1,
                "labels": scored.labels,
                "sentences": results.height,
                "parse_errors": int((results["parse_error"] != "").sum()),
            }
            for level, scored in scores.items()
        ],
    )


RESULT_SCHEMA = {
    "id": pl.String,
    "text": pl.String,
    "region": pl.String,
    "polygon_id": pl.String,
    "cell": pl.String,
    "round": pl.Int64,
    "position": pl.Int64,
    "gold_codes": pl.List(pl.Int64),
    "gold_identifiers": pl.List(pl.String),
    "raw_output": pl.String,
    "finish_reason": pl.String,
    "predicted": pl.List(pl.String),
    "invalid": pl.List(pl.String),
    "parse_error": pl.String,
    "prompt_version": pl.String,
    "model_id": pl.String,
    "model_revision": pl.String,
    "dataset_revision": pl.String,
    "seed": pl.Int64,
    "gpu": pl.String,
    "created_at": pl.String,
}
