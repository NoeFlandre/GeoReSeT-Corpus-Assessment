"""Command-line entry point. Each command runs on Grid'5000, not on a laptop."""

from __future__ import annotations

import argparse
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder

from georeset_corpus_assessment import settings
from georeset_corpus_assessment.engine import TransformersEngine
from georeset_corpus_assessment.legend import load_legend
from georeset_corpus_assessment.pipeline import (
    RunConfig,
    run,
    score_results,
    shard_subset,
)
from georeset_corpus_assessment.prepare import (
    LABEL_COLUMNS,
    SENTENCE_COLUMNS,
    build_pool,
    compact_labels,
    legend_table_from_csv,
    merge_pool_parts,
    select_part,
    write_pool,
)
from georeset_corpus_assessment.sampling import round_robin
from georeset_corpus_assessment.worker import reclaim, work


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Parse the command line and dispatch to one command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


def _build_parser() -> argparse.ArgumentParser:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="georeset")
    commands = parser.add_subparsers(required=True)

    labels_cmd = commands.add_parser("prepare-labels", help="Compact EUNIS labels")
    labels_cmd.add_argument("--work", type=Path, required=True)
    labels_cmd.set_defaults(handler=_prepare_labels)

    part_cmd = commands.add_parser("prepare-part", help="Build one pool part")
    part_cmd.add_argument("--work", type=Path, required=True)
    part_cmd.add_argument("--part", type=int, required=True)
    part_cmd.add_argument("--parts", type=int, required=True)
    part_cmd.set_defaults(handler=_prepare_part)

    merge_cmd = commands.add_parser("prepare-merge", help="Merge parts and legend")
    merge_cmd.add_argument("--work", type=Path, required=True)
    merge_cmd.add_argument("--parts", type=int, required=True)
    merge_cmd.add_argument("--wait-minutes", type=int, default=45)
    merge_cmd.set_defaults(handler=_prepare_merge)

    run_cmd = commands.add_parser("run", help="Label sentences with the model")
    run_cmd.add_argument("--work", type=Path, required=True)
    run_cmd.add_argument("--out", type=Path, required=True)
    run_cmd.add_argument("--limit", type=int, required=True)
    run_cmd.add_argument("--batch-size", type=int, default=500)
    run_cmd.add_argument("--gpu", default="unknown")
    run_cmd.add_argument("--shard", type=int, default=0)
    run_cmd.add_argument("--shards", type=int, default=1)
    run_cmd.set_defaults(handler=_run)

    score_cmd = commands.add_parser("score", help="Score a results folder per level")
    score_cmd.add_argument("--work", type=Path, required=True)
    score_cmd.add_argument("--out", type=Path, required=True)
    score_cmd.set_defaults(handler=_score)

    work_cmd = commands.add_parser("work", help="Claim and run shards from the queue")
    work_cmd.add_argument("--work", type=Path, required=True)
    work_cmd.add_argument("--out", type=Path, required=True)
    work_cmd.add_argument("--limit", type=int, required=True)
    work_cmd.add_argument("--shards", type=int, required=True)
    work_cmd.add_argument("--budget-minutes", type=float, required=True)
    work_cmd.add_argument("--estimate-minutes", type=float, default=8.0)
    work_cmd.add_argument("--gpu", default="unknown")
    work_cmd.set_defaults(handler=_work)

    reclaim_cmd = commands.add_parser(
        "reclaim", help="Free claims of unfinished shards"
    )
    reclaim_cmd.add_argument("--out", type=Path, required=True)
    reclaim_cmd.add_argument("--shards", type=int, required=True)
    reclaim_cmd.set_defaults(handler=_reclaim)

    publish_cmd = commands.add_parser("publish", help="Upload Parquet files to HF")
    publish_cmd.add_argument("--out", type=Path, required=True)
    publish_cmd.set_defaults(handler=_publish)
    return parser


def _source_files(prefix: str) -> list[str]:  # pragma: no cover
    files = HfApi().list_repo_files(
        settings.SOURCE_DATASET,
        repo_type="dataset",
        revision=settings.SOURCE_DATASET_REVISION,
    )
    return [
        name for name in files if name.startswith(prefix) and name.endswith(".parquet")
    ]


def _download(names: list[str]) -> list[Path]:  # pragma: no cover
    return [
        Path(
            hf_hub_download(
                settings.SOURCE_DATASET,
                filename=name,
                repo_type="dataset",
                revision=settings.SOURCE_DATASET_REVISION,
            ),
        )
        for name in names
    ]


def _prepare_labels(args: argparse.Namespace) -> None:  # pragma: no cover
    paths = _download(_source_files("polygons_labels_shares/"))
    labels = pl.read_parquet(paths, columns=LABEL_COLUMNS)
    args.work.mkdir(parents=True, exist_ok=True)
    compact_labels(labels).write_parquet(args.work / "labels.parquet")


def _prepare_part(args: argparse.Namespace) -> None:  # pragma: no cover
    names = select_part(_source_files("data/"), args.part, args.parts)
    sentences = pl.read_parquet(_download(names), columns=SENTENCE_COLUMNS)
    labels = pl.read_parquet(args.work / "labels.parquet")
    parts_dir = args.work / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    final = parts_dir / f"pool-{args.part:03d}.parquet"
    temporary = final.with_suffix(".parquet.tmp")
    build_pool(sentences, labels).write_parquet(temporary)
    temporary.replace(final)


def _wait_for_parts(
    expected: list[Path], wait_minutes: int
) -> None:  # pragma: no cover
    deadline = time.monotonic() + wait_minutes * 60
    while not all(path.exists() for path in expected):
        if time.monotonic() > deadline:
            msg = "pool parts are still missing; check the part jobs"
            raise SystemExit(msg)
        time.sleep(30)


def _prepare_merge(args: argparse.Namespace) -> None:  # pragma: no cover
    parts_dir = args.work / "parts"
    expected = [parts_dir / f"pool-{index:03d}.parquet" for index in range(args.parts)]
    _wait_for_parts(expected, args.wait_minutes)
    pool = merge_pool_parts([pl.read_parquet(path) for path in expected])
    write_pool(pool, args.work)
    legend_csv = args.work / "eunis_legend_detailed.csv"
    urllib.request.urlretrieve(settings.LEGEND_URL, legend_csv)  # noqa: S310
    legend_table_from_csv(legend_csv).write_parquet(args.work / "legend.parquet")


def _run(args: argparse.Namespace) -> None:  # pragma: no cover
    pool = pl.read_parquet(args.work / "pool.parquet")
    legend = load_legend(pl.read_parquet(args.work / "legend.parquet"))
    order = round_robin(pool, settings.SEED)
    subset = shard_subset(order, args.limit, args.shard, args.shards)
    out_dir = args.out / f"shard-{args.shard:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    subset.write_parquet(out_dir / "manifest.parquet")
    config = RunConfig(
        model_id=settings.MODEL_ID,
        model_revision=settings.MODEL_REVISION,
        dataset_revision=settings.SOURCE_DATASET_REVISION,
        seed=settings.SEED,
        gpu=args.gpu,
    )
    engine = TransformersEngine(
        settings.MODEL_ID,
        settings.MODEL_REVISION,
        settings.MAX_NEW_TOKENS,
    )
    written = run(
        subset,
        legend,
        engine,
        out_dir,
        config,
        limit=subset.height,
        batch_size=args.batch_size,
        created_at=datetime.now(tz=UTC).isoformat(),
    )
    print(f"wrote {written} rows to {out_dir}")  # noqa: T201


def _work(args: argparse.Namespace) -> None:  # pragma: no cover
    pool = pl.read_parquet(args.work / "pool.parquet")
    legend = load_legend(pl.read_parquet(args.work / "legend.parquet"))
    order = round_robin(pool, settings.SEED)
    config = RunConfig(
        model_id=settings.MODEL_ID,
        model_revision=settings.MODEL_REVISION,
        dataset_revision=settings.SOURCE_DATASET_REVISION,
        seed=settings.SEED,
        gpu=args.gpu,
    )
    engine = TransformersEngine(
        settings.MODEL_ID,
        settings.MODEL_REVISION,
        settings.MAX_NEW_TOKENS,
    )
    finished = work(
        order,
        legend,
        engine,
        args.out,
        config,
        limit=args.limit,
        shards=args.shards,
        budget_seconds=args.budget_minutes * 60,
        estimate_seconds=args.estimate_minutes * 60,
        clock=time.monotonic,
        created_at=datetime.now(tz=UTC).isoformat(),
    )
    print(f"finished shards: {finished}")  # noqa: T201


def _reclaim(args: argparse.Namespace) -> None:  # pragma: no cover
    freed = reclaim(args.out, args.shards)
    print(f"freed claims: {freed}")  # noqa: T201


def _score(args: argparse.Namespace) -> None:  # pragma: no cover
    legend = load_legend(pl.read_parquet(args.work / "legend.parquet"))
    metrics = score_results(args.out, legend)
    metrics.write_parquet(args.out / "metrics.parquet")
    print(metrics)  # noqa: T201


def _publish(args: argparse.Namespace) -> None:  # pragma: no cover
    create_repo(settings.RESULTS_DATASET, repo_type="dataset", exist_ok=True)
    upload_folder(
        folder_path=args.out,
        path_in_repo=args.out.name,
        repo_id=settings.RESULTS_DATASET,
        repo_type="dataset",
        allow_patterns=["*.parquet", "*/*.parquet"],
    )
