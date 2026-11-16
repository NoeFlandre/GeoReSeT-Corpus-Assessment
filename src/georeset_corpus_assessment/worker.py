"""Shared-queue worker: claim shards one at a time until none are left or time runs out.

Shards live in ``<out>/shard-XX``. A shard is finished when ``done`` exists in its
folder. A worker claims a shard by creating ``<out>/claims/shard-XX``, which is
atomic on a shared filesystem. A worker that dies leaves its claim behind, and
``reclaim`` removes claims of unfinished shards once no worker is alive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from georeset_corpus_assessment.pipeline import run, shard_subset

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import polars as pl

    from georeset_corpus_assessment.legend import Legend
    from georeset_corpus_assessment.pipeline import Engine, RunConfig

DONE_MARKER = "done"
CLAIMS_DIR = "claims"


def shard_dir(out: Path, shard: int) -> Path:
    """Return the folder that holds one shard's results."""
    return out / f"shard-{shard:02d}"


def is_done(out: Path, shard: int) -> bool:
    """Return True when a shard has been finished."""
    return (shard_dir(out, shard) / DONE_MARKER).exists()


def try_claim(out: Path, shard: int) -> bool:
    """Claim a shard atomically; return False when someone already has it."""
    claims = out / CLAIMS_DIR
    claims.mkdir(parents=True, exist_ok=True)
    try:
        (claims / f"shard-{shard:02d}").mkdir()
    except FileExistsError:
        return False
    return True


def next_shard(out: Path, shards: int) -> int | None:
    """Claim the lowest unfinished, unclaimed shard, or return None."""
    for shard in range(shards):
        if not is_done(out, shard) and try_claim(out, shard):
            return shard
    return None


def reclaim(out: Path, shards: int) -> list[int]:
    """Free claims of unfinished shards; call only when no worker is alive."""
    freed = []
    for shard in range(shards):
        claim = out / CLAIMS_DIR / f"shard-{shard:02d}"
        if claim.exists() and not is_done(out, shard):
            claim.rmdir()
            freed.append(shard)
    return freed


def work(  # noqa: PLR0913
    order: pl.DataFrame,
    legend: Legend,
    engine: Engine,
    out: Path,
    config: RunConfig,
    *,
    limit: int,
    shards: int,
    budget_seconds: float,
    estimate_seconds: float,
    clock: Callable[[], float],
    created_at: str,
) -> list[int]:
    """Process shards until none are left or the budget cannot fit another one."""
    deadline = clock() + budget_seconds
    finished: list[int] = []
    while clock() + estimate_seconds <= deadline:
        shard = next_shard(out, shards)
        if shard is None:
            break
        subset = shard_subset(order, limit, shard, shards)
        folder = shard_dir(out, shard)
        run(
            subset,
            legend,
            engine,
            folder,
            config,
            limit=subset.height,
            batch_size=max(1, subset.height),
            created_at=created_at,
        )
        (folder / DONE_MARKER).touch()
        finished.append(shard)
    return finished
