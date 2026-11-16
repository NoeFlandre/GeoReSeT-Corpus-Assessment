from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest

from georeset_corpus_assessment.legend import Legend
from georeset_corpus_assessment.pipeline import (
    Generation,
    RunConfig,
    build_rows,
    done_ids,
    interpret,
    next_batch,
    run,
    score_results,
    shard_subset,
    write_shard,
)

CONFIG = RunConfig(
    model_id="model",
    model_revision="rev",
    dataset_revision="data",
    seed=1,
    gpu="T4",
)


class FakeEngine:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompts: Sequence[list[dict[str, str]]]) -> list[Generation]:
        self.calls += 1
        return [Generation(text=self.text, finish_reason="stop") for _ in prompts]


def make_order(count: int) -> pl.DataFrame:
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


def test_done_ids_is_empty_without_shards(tmp_path: Path) -> None:
    assert done_ids(tmp_path) == set()


def test_next_batch_respects_limit_batch_size_and_done_ids() -> None:
    order = make_order(5)
    assert next_batch(order, {"s0"}, batch_size=5, limit=3)["id"].to_list() == [
        "s1",
        "s2",
    ]
    assert next_batch(order, {"s0"}, batch_size=1, limit=3)["id"].to_list() == ["s1"]


def test_interpret_treats_truncation_as_an_error(legend: Legend) -> None:
    valid = frozenset(legend.finest_identifiers())
    truncated = interpret(Generation("FINAL: MA211", "length"), valid)
    assert truncated.identifiers == ()
    assert truncated.error == "truncated"


def test_interpret_parses_a_complete_answer(legend: Legend) -> None:
    valid = frozenset(legend.finest_identifiers())
    answer = interpret(Generation("FINAL: MA211", "stop"), valid)
    assert answer.identifiers == ("MA211",)


def test_build_rows_records_gold_prediction_and_run_metadata(legend: Legend) -> None:
    rows = build_rows(
        make_order(1),
        [Generation("FINAL: MA211", "stop")],
        legend,
        CONFIG,
        created_at="2026-01-01T00:00:00+00:00",
    )
    row = rows.row(0, named=True)
    assert row["gold_codes"] == [1101]
    assert row["gold_identifiers"] == ["MA211"]
    assert row["predicted"] == ["MA211"]
    assert row["parse_error"] == ""
    assert row["gpu"] == "T4"
    assert row["model_revision"] == "rev"


def test_build_rows_stores_parse_errors(legend: Legend) -> None:
    rows = build_rows(
        make_order(1),
        [Generation("no answer", "stop")],
        legend,
        CONFIG,
        created_at="t",
    )
    assert rows["parse_error"].to_list() == ["no_final_line"]


def test_write_shard_numbers_files_and_leaves_no_temporary(
    tmp_path: Path,
    legend: Legend,
) -> None:
    rows = build_rows(
        make_order(1),
        [Generation("FINAL: none", "stop")],
        legend,
        CONFIG,
        created_at="t",
    )
    first = write_shard(rows, tmp_path)
    second = write_shard(rows, tmp_path)
    assert first.name == "part-00000.parquet"
    assert second.name == "part-00001.parquet"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "part-00000.parquet",
        "part-00001.parquet",
    ]


def test_run_resumes_and_respects_the_limit(tmp_path: Path, legend: Legend) -> None:
    engine = FakeEngine("FINAL: MA211")
    order = make_order(5)
    written = run(
        order,
        legend,
        engine,
        tmp_path,
        CONFIG,
        limit=4,
        batch_size=3,
        created_at="t",
    )
    assert written == 4
    assert engine.calls == 2
    assert done_ids(tmp_path) == {"s0", "s1", "s2", "s3"}

    resumed = run(
        order,
        legend,
        engine,
        tmp_path,
        CONFIG,
        limit=5,
        batch_size=3,
        created_at="t",
    )
    assert resumed == 1
    assert done_ids(tmp_path) == {"s0", "s1", "s2", "s3", "s4"}

    finished = run(
        order,
        legend,
        engine,
        tmp_path,
        CONFIG,
        limit=5,
        batch_size=3,
        created_at="t",
    )
    assert finished == 0
    assert engine.calls == 3


def test_score_results_reports_each_level(tmp_path: Path, legend: Legend) -> None:
    run(
        make_order(3),
        legend,
        FakeEngine("FINAL: MA211"),
        tmp_path,
        CONFIG,
        limit=3,
        batch_size=3,
        created_at="t",
    )
    metrics = score_results(tmp_path, legend)
    assert metrics["level"].to_list() == [1, 2, 3]
    assert metrics["sentences"].to_list() == [3, 3, 3]
    assert metrics.filter(pl.col("level") == 3)["micro_f1"].to_list() == [1.0]


def test_shard_subsets_cover_the_limit_without_overlap() -> None:
    order = make_order(10)
    seen = []
    for shard in range(3):
        subset = shard_subset(order, limit=7, shard=shard, shards=3)
        seen.extend(subset["id"].to_list())
    assert sorted(seen) == [f"s{index}" for index in range(7)]


def test_shard_subset_rejects_an_out_of_range_shard() -> None:
    with pytest.raises(ValueError, match="shard must satisfy"):
        shard_subset(make_order(3), limit=3, shard=3, shards=3)
