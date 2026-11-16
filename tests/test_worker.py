from pathlib import Path

from georeset_corpus_assessment.legend import Legend
from georeset_corpus_assessment.worker import (
    DONE_MARKER,
    is_done,
    next_shard,
    reclaim,
    shard_dir,
    work,
)
from support import CONFIG, FakeEngine, make_order


def test_a_single_worker_finishes_every_shard_once(
    tmp_path: Path, legend: Legend
) -> None:
    order = make_order(6)
    finished = work(
        order,
        legend,
        FakeEngine(),
        tmp_path,
        CONFIG,
        limit=6,
        shards=3,
        budget_seconds=1e9,
        estimate_seconds=1,
        clock=lambda: 0.0,
        created_at="t",
    )
    assert finished == [0, 1, 2]
    assert all(is_done(tmp_path, shard) for shard in range(3))


def test_finished_shards_are_skipped_and_claimed_ones_are_not_taken(
    tmp_path: Path,
) -> None:
    (shard_dir(tmp_path, 0)).mkdir(parents=True)
    (shard_dir(tmp_path, 0) / DONE_MARKER).touch()
    assert next_shard(tmp_path, 3) == 1
    assert next_shard(tmp_path, 3) == 2
    assert next_shard(tmp_path, 3) is None


def test_reclaim_frees_claims_of_unfinished_shards_only(tmp_path: Path) -> None:
    assert next_shard(tmp_path, 2) == 0
    (shard_dir(tmp_path, 0) / DONE_MARKER).parent.mkdir(parents=True, exist_ok=True)
    (shard_dir(tmp_path, 0) / DONE_MARKER).touch()
    assert next_shard(tmp_path, 2) == 1
    assert reclaim(tmp_path, 2) == [1]
    assert next_shard(tmp_path, 2) == 1


def test_work_stops_before_a_shard_that_cannot_fit_the_budget(
    tmp_path: Path,
    legend: Legend,
) -> None:
    finished = work(
        make_order(4),
        legend,
        FakeEngine(),
        tmp_path,
        CONFIG,
        limit=4,
        shards=2,
        budget_seconds=10,
        estimate_seconds=20,
        clock=lambda: 0.0,
        created_at="t",
    )
    assert finished == []
