from pathlib import Path

import polars as pl

from georeset_corpus_assessment.legend import EunisClass, Legend
from georeset_corpus_assessment.prepare import (
    build_pool,
    legend_table_from_csv,
    write_pool,
)
from georeset_corpus_assessment.prompt import (
    INSTRUCTIONS,
    PROMPT_VERSION,
    build_messages,
    candidate_block,
)


def test_build_pool_skips_polygons_without_eunis_label() -> None:
    sentences = pl.DataFrame(
        {
            "id": ["a", "b"],
            "text": ["t", "u"],
            "translation": ["original", "original"],
            "landuse": [True, True],
            "polygon_ids": [["way/1"], ["way/2"]],
            "region": ["r", "r"],
            "lat": [45.0, 45.0],
            "lon": [5.0, 5.0],
        },
    )
    labels = pl.DataFrame(
        {
            "polygon_id": ["way/1", "way/2"],
            "eunis_code": [[1101], None],
            "eunis_shares": [[1.0], None],
        },
        schema_overrides={"eunis_code": pl.List(pl.Int64)},
    )
    pool = build_pool(sentences, labels)
    assert pool["id"].to_list() == ["a"]
    assert pool["gold_codes"].to_list() == [[1101]]
    assert "cell" in pool.columns


def test_legend_table_from_csv_reads_hierarchy_columns(tmp_path: Path) -> None:
    csv = tmp_path / "legend.csv"
    csv.write_text(
        "Code,EUNIS1,EUNIS2,EUNIS3,EUNIS1_name,EUNIS2_name,EUNIS3_name\n"
        "1101,MA2,MA21,MA211,Littoral,Arctic,Saltmarsh\n",
        encoding="utf-8",
    )
    table = legend_table_from_csv(csv)
    assert table["EUNIS3"].to_list() == ["MA211"]
    assert table["Code"].to_list() == [1101]


def test_write_pool_creates_directory_and_file(tmp_path: Path) -> None:
    target = write_pool(pl.DataFrame({"id": ["a"]}), tmp_path / "work")
    assert target == tmp_path / "work" / "pool.parquet"
    assert pl.read_parquet(target)["id"].to_list() == ["a"]


def test_candidate_block_lists_finest_classes_with_names(legend: Legend) -> None:
    block = candidate_block(legend)
    assert block.splitlines() == [
        "- URBAN: Urban",
        "- MA211: Saltmarsh",
        "- MA22: Atlantic",
    ]


def test_candidate_block_falls_back_to_identifier_without_name() -> None:
    nameless = Legend([EunisClass(code=1, path=("X1",), name=None)])
    assert candidate_block(nameless) == "- X1: X1"


def test_build_messages_puts_instructions_and_sentence_in_order(legend: Legend) -> None:
    messages = build_messages("Salt marsh at the estuary.", legend)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == INSTRUCTIONS
    assert "Salt marsh at the estuary." in messages[1]["content"]
    assert "- MA211: Saltmarsh" in messages[1]["content"]
    assert PROMPT_VERSION == "zero-shot-v1"
