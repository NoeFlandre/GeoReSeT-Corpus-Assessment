"""Build the evaluation pool from sentences and EUNIS polygon labels."""

from __future__ import annotations

import h3
import polars as pl

GOLD_MIN_SHARE = 0.1
H3_RESOLUTION = 5


def select_sentences(sentences: pl.DataFrame) -> pl.DataFrame:
    """Keep original English sentences flagged as land use, linked to one polygon."""
    return sentences.filter(
        (pl.col("translation") == "original")
        & pl.col("landuse")
        & (pl.col("polygon_ids").list.len() == 1),
    ).select(
        "id",
        "text",
        "region",
        "lat",
        "lon",
        pl.col("polygon_ids").list.first().alias("polygon_id"),
    )


def attach_gold(pool: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Join EUNIS labels and keep codes whose share reaches the minimum.

    ``labels`` needs ``polygon_id``, ``eunis_code`` and ``eunis_shares``.
    Polygons whose shares all fall below the minimum are dropped, because
    they have no gold label to score against.
    """
    joined = pool.join(
        labels.select("polygon_id", "eunis_code", "eunis_shares"),
        on="polygon_id",
        how="inner",
    )
    gold = [
        [
            code
            for code, share in zip(codes, shares, strict=True)
            if share >= GOLD_MIN_SHARE
        ]
        for codes, shares in zip(
            joined["eunis_code"].to_list(),
            joined["eunis_shares"].to_list(),
            strict=True,
        )
    ]
    return (
        joined.with_columns(pl.Series("gold_codes", gold, dtype=pl.List(pl.Int64)))
        .filter(pl.col("gold_codes").list.len() > 0)
        .drop("eunis_code", "eunis_shares")
    )


def add_cells(pool: pl.DataFrame) -> pl.DataFrame:
    """Add the H3 cell of each sentence from its coordinates."""
    return pool.with_columns(
        pl.struct("lat", "lon")
        .map_elements(
            lambda point: h3.latlng_to_cell(point["lat"], point["lon"], H3_RESOLUTION),
            return_dtype=pl.String,
        )
        .alias("cell"),
    )
