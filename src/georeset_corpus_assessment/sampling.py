"""Round-robin ordering of sentences across H3 cells.

Each round takes one sentence from every cell that still has sentences left.
Any prefix of the resulting order therefore holds cells with counts that
differ by at most one, so partial results stay balanced across the area.
The order is fully determined by the seed and the input rows.
"""

from __future__ import annotations

import hashlib

import polars as pl

_KEY_BYTES = 8


def stable_key(seed: int, value: str) -> int:
    """Return a reproducible 64-bit key for a value under a seed."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:_KEY_BYTES], "big")


def round_robin(pool: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Order the pool round-robin across cells.

    ``pool`` needs ``id`` and ``cell`` columns. The result adds ``round``
    (0-based, the rank of the row within its cell), ``position`` (index in the
    final order) and the helper keys.
    """
    keyed = pool.with_columns(
        pl.col("id")
        .map_elements(
            lambda value: stable_key(seed, value),
            return_dtype=pl.UInt64,
        )
        .alias("row_key"),
        pl.col("cell")
        .map_elements(
            lambda value: stable_key(seed, value),
            return_dtype=pl.UInt64,
        )
        .alias("cell_key"),
    )
    return (
        keyed.with_columns(
            (pl.col("row_key").rank("ordinal").over("cell") - 1)
            .cast(pl.Int64)
            .alias("round"),
        )
        .sort(["round", "cell_key", "cell", "row_key"])
        .with_row_index("position")
        .drop("row_key", "cell_key")
    )
