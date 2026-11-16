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
    return add_cells(attach_gold(select_sentences(sentences), compact_labels(labels)))


def select_part(files: list[str], part: int, parts: int) -> list[str]:
    """Return the files of one part, dealt round-robin over the sorted list."""
    if not 0 <= part < parts:
        msg = "part must satisfy 0 <= part < parts"
        raise ValueError(msg)
    ordered = sorted(files)
    return [name for index, name in enumerate(ordered) if index % parts == part]


def compact_labels(labels: pl.DataFrame) -> pl.DataFrame:
    """Keep the labelled polygons and the columns needed to build the pool."""
    return labels.filter(pl.col("eunis_code").is_not_null()).select(LABEL_COLUMNS)


def merge_pool_parts(parts: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate the pool parts built from disjoint sets of source files."""
    return pl.concat(parts, how="vertical")


def legend_table_from_csv(path: Path) -> pl.DataFrame:
    """Read the detailed EUNIS legend CSV, keeping the hierarchy columns."""
    return pl.read_csv(path, infer_schema_length=None)


def write_pool(pool: pl.DataFrame, output_dir: Path) -> Path:
    """Write the evaluation pool as Parquet and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "pool.parquet"
    pool.write_parquet(target)
    return target
