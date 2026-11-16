"""Turn the published corpus into the evaluation pool and the EUNIS legend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from georeset_corpus_assessment.pool import add_cells, attach_gold, select_sentences

if TYPE_CHECKING:
    from pathlib import Path

SENTENCE_COLUMNS = [
    "id",
    "text",
    "translation",
    "landuse",
    "polygon_ids",
    "region",
    "lat",
    "lon",
]
LABEL_COLUMNS = ["polygon_id", "eunis_code", "eunis_shares"]


def build_pool(sentences: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Return the evaluation pool: selected sentences with gold codes and cells."""
    labelled = labels.filter(pl.col("eunis_code").is_not_null())
    return add_cells(attach_gold(select_sentences(sentences), labelled))


def legend_table_from_csv(path: Path) -> pl.DataFrame:
    """Read the detailed EUNIS legend CSV, keeping the hierarchy columns."""
    return pl.read_csv(path, infer_schema_length=None)


def write_pool(pool: pl.DataFrame, output_dir: Path) -> Path:
    """Write the evaluation pool as Parquet and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "pool.parquet"
    pool.write_parquet(target)
    return target
